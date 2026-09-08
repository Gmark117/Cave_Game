"""Mission orchestration and runtime resource setup.

Constructing `MissionControl` prepares mission state only. Calling `run()`
initializes the mission window, agents, pathfinding resources, and worker
threads before entering the main loop.
"""

import random as rand
import threading
from pathlib import Path
from typing import List, Tuple, Any, Optional

import numpy as np
import pygame

from asset_config.helpers import wall_hit
from agents.factory import AgentFactory
from ui.control_center.facade import ControlCenter
from mapping.rover_targets import RoverTargetService
from mapping.exploration_sectors import (
    ExplorationSectorCoordinator,
    SectorCheckInResult,
)
from mapping.terrain_fusion import TerrainFusionService
from mapping.terrain_knowledge import TerrainKnowledge
from mapping.terrain_sharing import TerrainSharingService
from mapping.wall_mapping import WallMappingSnapshot, wall_mapping_snapshot
from mission.debug_info import MissionDebugInfo
from mission.frame_timing import FrameProfiler
from mission.objectives import build_mission_objective
from mission.pause_control import PauseCoordinator, SimulationClock
from mission.runtime_trace import RuntimeTraceLogger
from contracts import (
    MissionDebugDependencies,
    MissionRendererDependencies,
    RoverTargetDependencies,
    SlamViewDependencies,
    TerrainFusionDependencies,
    TerrainSharingDependencies,
)
from navigation.pathfinding import PathfindingService
from navigation.astar_pathfinder import PathResult
from mission.presentation_adapter import PresentationAdapter
from rendering.slam_renderer import SlamRenderer
from rendering.mission_renderer import MissionRenderer
from rendering.sector_renderer import SectorRenderer
from rendering.slam_view import SlamViewService
from mission.lifecycle import MissionControlLifecycleMixin


class MissionControl(MissionControlLifecycleMixin):
    """Orchestrates the simulation mission.

    Construction is side-effect-light and does not start threads, processes,
    shared memory, or the mission loop. Runtime resources are created by
    `run()` through `_initialize_runtime()`.
    """
    def __init__(self, game: Any) -> None:
        """Prepare mission state without starting the mission.

        Args:
            game: The `Game` instance owning this mission (typed as `Any`
                  to avoid circular imports).
        """
        # Seed every mission from the menu settings so generated caves and
        # agent color/order choices remain reproducible for a given seed.
        rand.seed(game.sim_settings.mission_config.seed)

        self.game         = game
        self.settings     = game.sim_settings 
        self.objective    = (
            getattr(game, "mission_objective", None)
            or build_mission_objective(
                self.settings.mission_config.objective
            )
        )
        self.cartographer = game.cartographer
        self.map_matrix   = self.cartographer.bin_map # Get the binary map representation
        self.map_h, self.map_w = np.asarray(self.map_matrix).shape
        # Map generation may be bypassed or mocked in tests, so normalize the
        # optional roughness layer to the cave matrix shape before sensors use it.
        terrain_roughness_src = np.array(
            getattr(self.cartographer, 'terrain_roughness', np.zeros_like(self.map_matrix)),
            dtype=np.float32
        )
        if terrain_roughness_src.shape != np.asarray(self.map_matrix).shape:
            terrain_roughness_src = np.zeros(np.asarray(self.map_matrix).shape, dtype=np.float32)
        self.terrain_roughness = terrain_roughness_src
        # Mission aggregate for telemetry and combined UI rendering only.
        # Active agent decisions must use their own local knowledge.
        self.terrain_knowledge = TerrainKnowledge(self.map_matrix)
        self.rover_assignment_lock = threading.Lock()
        self.rover_assignments = {}
        self.completed_rover_targets = set()
        # Rover motion stays disabled until its local-knowledge policy is defined.
        self.rover_motion_enabled = False

        # Runtime resources are initialized explicitly by run().
        # Pathfinding owns external resources (shared memory and a process pool)
        # but does not allocate them until ``run`` calls ``start``.
        self.pathfinding = PathfindingService(
            self.map_matrix,
            self.settings.mission_config.num_drones,
        )
        self.mission_event = threading.Event()
        self.exploration_completion_event = threading.Event()
        self.simulation_clock = SimulationClock()
        self.pause_coordinator = PauseCoordinator(self.mission_event)
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.is_paused = False
        self.clock: Optional[pygame.time.Clock] = None
        self.drones = []
        self.rovers = []
        self.num_drones = self.settings.mission_config.num_drones
        self.num_rovers = 0
        self.control_center: Optional[ControlCenter] = None
        self._runtime_initialized = False
        self._running = False
        self._has_run = False
        self.restart_requested = False
        self.exit_requested = False
        self.frame_profiler = FrameProfiler()
        self.runtime_trace = RuntimeTraceLogger(
            Path(__file__).resolve().parents[1],
            self.settings.trace,
        )
        self.runtime_trace.record(
            "mission_constructed",
            seed=self.settings.mission_config.seed,
            map_dim=self.settings.mission_config.map_dim,
            drones=self.settings.mission_config.num_drones,
            map_width=self.map_w,
            map_height=self.map_h,
            exploration_policy=self.settings.exploration.policy,
            exploration_completion=(
                "team_wall_tolerance_or_rover_sector_frontier_exhaustion"
            ),
            exploration_progress="exposed_wall_slam_coverage",
            wall_completion_tolerance_pixels=(
                self.settings.mission_config
                .wall_completion_tolerance_pixels
            ),
            terrain_role="rover_checkpoint_and_navigation",
            frontier_policy=(
                "rover_epoch_sectors_with_in_sector_frontier_guidance"
            ),
            sector_cell_size=self.settings.frontier.global_cell_size,
            sector_check_in="single_arrival_and_departure_exchange",
            sector_check_in_path="astar_with_breadcrumb_fallback",
            sector_assignment_delivery="rover_signalled_no_polling",
            sector_outcome_memory="component_local_immediate_zero_gain",
            sector_workload_estimate=(
                "scan_approach_dispersion_and_terrain"
            ),
            frontier_unknown_basin_rescue="interior_basins_only",
            rover_periodic_sharing=False,
            rover_drone_pair_sharing=False,
            frontier_stride=self.settings.frontier.stride,
            frontier_minimum_cluster_cells=(
                self.settings.frontier.minimum_cluster_cells
            ),
            frontier_minimum_unknown_support_cells=(
                self.settings.frontier.minimum_unknown_support_cells
            ),
            frontier_distance_band=self.settings.frontier.distance_band,
            frontier_wall_continuation_weight=(
                self.settings.frontier.wall_continuation_weight
            ),
            frontier_cluster_size_weight=(
                self.settings.frontier.cluster_size_weight
            ),
            frontier_cluster_proximity_weight=(
                self.settings.frontier.cluster_proximity_weight
            ),
            frontier_global_cell_size=(
                self.settings.frontier.global_cell_size
            ),
            frontier_global_refresh_interval=(
                self.settings.frontier.global_refresh_interval
            ),
            frontier_global_ownership_weight=(
                self.settings.frontier.global_ownership_weight
            ),
            frontier_maximum_path_circuity=(
                self.settings.frontier.maximum_path_circuity
            ),
            stagnation_distance=(
                self.settings.exploration.stagnation_distance
            ),
            coverage_memory_cell_size=(
                self.settings.exploration.coverage_memory_cell_size
            ),
            coverage_memory_decay_seconds=(
                self.settings.exploration.coverage_memory_decay_seconds
            ),
            coverage_visit_weight=(
                self.settings.exploration.coverage_visit_weight
            ),
            coverage_edge_weight=(
                self.settings.exploration.coverage_edge_weight
            ),
            stagnation_min_sensor_cells_per_px=(
                self.settings.exploration.stagnation_min_sensor_cells_per_px
            ),
            wall_direction_bias=(
                self.settings.exploration.wall_direction_bias
            ),
            unexplored_direction_bias=(
                self.settings.exploration.unexplored_direction_bias
            ),
            separation_direction_bias=(
                self.settings.exploration.separation_direction_bias
            ),
        )
        
        self.delay = 1/15 # Set a delay for frame updates


        self.completed = False # Track whether the mission is completed

        # Initialize presentation adapter for UI state and map rendering
        self.presentation = PresentationAdapter(self.map_w, self.map_h)
        self.slam_renderer = SlamRenderer(
            self.map_w,
            self.map_h,
            frontier_confidence_threshold=(
                self.settings.frontier.confidence_threshold
            ),
        )
        self.sector_renderer = SectorRenderer(self.map_w, self.map_h)
        self.last_explored_update = 0.0
        self.wall_mapping_progress = WallMappingSnapshot(0, 0, 0.0, False)
        self.explored_update_interval = 0.5
        # Dependency bundles keep services decoupled from the large
        # MissionControl object while still giving them the callbacks they need.
        self.terrain_fusion_dependencies = TerrainFusionDependencies(
            terrain_knowledge=self.terrain_knowledge,
            presentation=self.presentation,
        )
        self.terrain_fusion = TerrainFusionService(
            self.terrain_fusion_dependencies
        )
        self.terrain_sharing = TerrainSharingService(
            TerrainSharingDependencies(
                sharing=self.settings.sharing,
                cave_map=np.asarray(self.map_matrix),
                map_width=self.map_w,
                map_height=self.map_h,
                terrain_knowledge=self.terrain_knowledge,
                get_drones=lambda: self.drones,
                get_rovers=lambda: self.rovers,
                presentation=self.presentation,
                simulation_time=self.simulation_time,
                runtime_trace=self.runtime_trace,
                periodic_rover_sharing_enabled=False,
            )
        )
        self.rover_targets = RoverTargetService(
            RoverTargetDependencies(
                cave_map=np.asarray(self.map_matrix),
                terrain_knowledge=self.terrain_knowledge,
                assignment_lock=self.rover_assignment_lock,
                assignments=self.rover_assignments,
                completed_targets=self.completed_rover_targets,
                norm_width=self.game.width,
                norm_height=self.game.height,
            )
        )
        self.slam_view = SlamViewService(
            SlamViewDependencies(
                rendering=self.settings.rendering,
                terrain_knowledge=self.terrain_knowledge,
                presentation=self.presentation,
                slam_renderer=self.slam_renderer,
                get_drones=lambda: self.drones,
                get_window=lambda: self.game.window,
            )
        )
        self.debug_info = MissionDebugInfo(
            MissionDebugDependencies(
                get_drones=lambda: self.drones,
                presentation=self.presentation,
                dirty_map_count=self.slam_view.dirty_map_count,
                simulation_time=self.simulation_time,
                frame_profiler=self.frame_profiler,
                runtime_trace=self.runtime_trace,
            )
        )
        self.renderer = MissionRenderer(
            MissionRendererDependencies(
                get_window=lambda: self.game.window,
                slam_view=self.slam_view,
                debug_info=self.debug_info,
                get_control_center=lambda: self.control_center,
                get_drones=lambda: self.drones,
                get_rovers=lambda: self.rovers,
                presentation=self.presentation,
                is_paused=lambda: self.is_paused,
                is_music_enabled=self.music_enabled,
                sector_renderer=self.sector_renderer,
                get_sector_snapshot=lambda: self.exploration_sectors.snapshot(),
            )
        )
        
        # Set the starting position for drones
        self.start_point = None
        self.set_start_point()
        self.exploration_sectors = ExplorationSectorCoordinator(
            (self.map_h, self.map_w),
            self.num_drones,
            self.start_point,
            cell_size=self.settings.frontier.global_cell_size,
            confidence_threshold=(
                self.settings.frontier.confidence_threshold
            ),
            minimum_frontier_component_cells=(
                self.settings.frontier.minimum_cluster_cells
            ),
            minimum_unknown_support_cells=(
                self.settings.frontier.minimum_unknown_support_cells
            ),
        )

    def _initialize_runtime(self) -> None:
        """Create window, agents, pathfinding resources, and first frame."""
        if self._runtime_initialized:
            return

        self.completed = False
        self.mission_event.clear()
        self.exploration_completion_event.clear()
        self.pause_event.set()
        self.is_paused = False
        self.game.display = self.game.to_maximised()
        self.control_center = ControlCenter(self.game)

        AgentFactory.build_drones(self)
        AgentFactory.build_rovers(self)

        # Reset presentation after agents exist so their path/vision toggles
        # start from a known default each time a mission is run.
        self.presentation.reset(self.drones)

        self.clock = pygame.time.Clock()
        self.pathfinding.start()
        self._runtime_initialized = True

        self.update_sensors()
        self.renderer.draw()
        pygame.display.update()

    def set_start_point(self) -> None:
        """Pick a viable start point from the map generator worm starts.

        Keeps sampling the list of candidate worm starts until a non-wall
        coordinate is found.
        """
        # Continuously search for a valid starting point until one is found
        while self.start_point is None or wall_hit(self.map_matrix, self.start_point):
            # Randomly select one of the initial points of the worms
            # Choose based on available worm starts (don't assume 4)
            i = rand.randrange(len(self.cartographer.worm_x))
            self.start_point = (self.cartographer.worm_x[i], self.cartographer.worm_y[i])
    

# =============================================================================
# Drone threads and pathfinding interface
# =============================================================================

    def drone_thread(self, drone_id: int) -> None:
        """
        Thread function that controls the movement of a single drone during the mission.
        This method runs in a separate thread for each drone and continuously moves the drone
        until either the mission is terminated (via mission_event) or the drone completes its
        assigned mission.
        Notes:
            - The method respects the global mission_event flag, which can stop all drones.
            - Movement speed is controlled by self.delay using an interruptible wait.
            - The wait mechanism allows for immediate response when mission_event is set.
        """
        label = ("drone", drone_id)
        if not self.pause_coordinator.register_current_worker(label):
            return
        try:
            while (
                not self.mission_event.is_set()
                and not self.drones[drone_id].mission_completed()
            ):
                # Worker threads only pause at cooperative checkpoints, which
                # keeps shared drone state from being stopped mid-update.
                if not self.pause_checkpoint():
                    break
                self.drones[drone_id].move()

                if not self.pause_checkpoint():
                    break
                self.terrain_sharing.share_with_nearby_drones(drone_id)

                if not self.wait_simulation_delay(self.delay):
                    break
        finally:
            self.pause_coordinator.unregister_current_worker()


    def compute_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute an A* escape or homing path for a drone."""
        return self.pathfinding.compute_path(start, goal)


    def compute_path_segment(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> PathResult:
        """Compute a complete drone route or one capped progress segment."""
        return self.pathfinding.compute_path_segment(start, goal)

    def get_check_in_position(self) -> Tuple[int, int]:
        """Return the primary rover's current physical rendezvous point."""
        if self.rovers and self.rovers[0] is not None:
            return tuple(self.rovers[0].pos)
        return tuple(self.start_point)

    def sector_check_in(
        self,
        drone_id: int,
        completed_sector_id: int | None,
    ) -> SectorCheckInResult:
        """Exchange maps at the primary rover and advance the sector barrier."""
        if not self.rovers or self.rovers[0] is None:
            return SectorCheckInResult(arrived=False)
        arrived = self.terrain_sharing.check_in_with_rover(drone_id, 0)
        if not arrived:
            return SectorCheckInResult(
                arrived=False,
                generation=self.exploration_sectors.generation,
            )

        if self.exploration_completion_event.is_set():
            self.exploration_sectors.stop()
            return SectorCheckInResult(
                arrived=True,
                mission_exhausted=True,
                generation=self.exploration_sectors.generation,
            )

        outcome_report = None
        if 0 <= int(drone_id) < len(self.drones):
            movement = getattr(
                self.drones[int(drone_id)],
                "movement_controller",
                None,
            )
            report_outcome = getattr(
                movement,
                "sector_outcome_report",
                None,
            )
            if callable(report_outcome):
                outcome_report = report_outcome(completed_sector_id)
        rover_slam = self.rovers[0].slam_map.snapshot(point_limit=0)
        result = self.exploration_sectors.check_in(
            drone_id,
            completed_sector_id,
            rover_slam,
            outcome_report,
        )
        if self.exploration_completion_event.is_set():
            self.exploration_sectors.stop()
            result = SectorCheckInResult(
                arrived=True,
                mission_exhausted=True,
                generation=self.exploration_sectors.generation,
            )
        diagnostics = result.frontier_diagnostics
        if diagnostics is not None:
            component_sizes = diagnostics.component_sizes
            self.runtime_trace.record(
                "rover_sector_frontiers_filtered",
                sim_time=self.simulation_time(),
                generation=result.generation,
                rover_slam_version=rover_slam.version,
                raw_frontier_pixels=diagnostics.raw_frontier_pixels,
                raw_component_count=diagnostics.raw_component_count,
                component_size_min=(
                    0 if not component_sizes else min(component_sizes)
                ),
                component_size_max=(
                    0 if not component_sizes else max(component_sizes)
                ),
                component_size_sample=component_sizes[:32],
                significant_frontier_pixels=(
                    diagnostics.significant_frontier_pixels
                ),
                significant_component_count=(
                    diagnostics.significant_component_count
                ),
                large_component_count=diagnostics.large_component_count,
                unknown_supported_component_count=(
                    diagnostics.unknown_supported_component_count
                ),
                unknown_supported_candidate_count=(
                    diagnostics.unknown_supported_candidate_count
                ),
                redundant_unknown_supported_component_count=(
                    diagnostics
                    .redundant_unknown_supported_component_count
                ),
                frontier_unknown_basin_count=(
                    diagnostics.frontier_unknown_basin_count
                ),
                significant_unknown_basin_count=(
                    diagnostics.significant_unknown_basin_count
                ),
                border_connected_unknown_basin_count=(
                    diagnostics.border_connected_unknown_basin_count
                ),
                border_connected_unknown_supported_candidate_count=(
                    diagnostics
                    .border_connected_unknown_supported_candidate_count
                ),
                discarded_component_count=(
                    diagnostics.discarded_component_count
                ),
                discarded_frontier_pixels=(
                    diagnostics.discarded_frontier_pixels
                ),
                minimum_component_cells=(
                    diagnostics.minimum_component_cells
                ),
                minimum_unknown_support_cells=(
                    diagnostics.minimum_unknown_support_cells
                ),
                component_diagnostics=diagnostics.components,
                mission_exhausted=result.mission_exhausted,
            )
        workload = result.workload_diagnostics
        if workload is not None:
            initial = workload.initial_frontier_workloads
            balanced = workload.balanced_frontier_workloads
            self.runtime_trace.record(
                "rover_sector_workload_balanced",
                sim_time=self.simulation_time(),
                generation=result.generation,
                initial_frontier_workloads=initial,
                balanced_frontier_workloads=balanced,
                initial_workload_spread=(
                    0 if not initial else max(initial) - min(initial)
                ),
                balanced_workload_spread=(
                    0 if not balanced else max(balanced) - min(balanced)
                ),
                moved_coarse_cell_count=(
                    workload.moved_coarse_cell_count
                ),
                initial_estimated_efforts=(
                    workload.initial_estimated_efforts
                ),
                balanced_estimated_efforts=(
                    workload.balanced_estimated_efforts
                ),
                initial_estimated_effort_spread=(
                    0.0
                    if not workload.initial_estimated_efforts
                    else max(workload.initial_estimated_efforts)
                    - min(workload.initial_estimated_efforts)
                ),
                balanced_estimated_effort_spread=(
                    0.0
                    if not workload.balanced_estimated_efforts
                    else max(workload.balanced_estimated_efforts)
                    - min(workload.balanced_estimated_efforts)
                ),
            )
        outcome = result.outcome_diagnostics
        if outcome is not None:
            self.runtime_trace.record(
                "rover_sector_frontier_outcomes",
                sim_time=self.simulation_time(),
                generation=result.generation,
                previous_generation=outcome.previous_generation,
                confident_occupied_gain=outcome.confident_occupied_gain,
                remembered_component_count=(
                    outcome.remembered_component_count
                ),
                suppressed_component_count=(
                    outcome.suppressed_component_count
                ),
                suppressed_frontier_pixels=(
                    outcome.suppressed_frontier_pixels
                ),
                remaining_component_count=(
                    outcome.remaining_component_count
                ),
                remaining_frontier_pixels=(
                    outcome.remaining_frontier_pixels
                ),
                evaluated_component_count=(
                    outcome.evaluated_component_count
                ),
                locally_unchanged_component_count=(
                    outcome.locally_unchanged_component_count
                ),
                reported_zero_gain_component_count=(
                    outcome.reported_zero_gain_component_count
                ),
                productive_component_count=(
                    outcome.productive_component_count
                ),
                resolved_component_count=(
                    outcome.resolved_component_count
                ),
                component_outcomes=outcome.components,
                mission_exhausted=result.mission_exhausted,
            )
        suppressions = (
            () if outcome_report is None else outcome_report.suppressions
        )
        self.runtime_trace.record(
            "drone_sector_check_in",
            sim_time=self.simulation_time(),
            drone_id=int(drone_id),
            completed_sector_id=completed_sector_id,
            generation=result.generation,
            waiting_for_team=result.waiting_for_team,
            mission_exhausted=result.mission_exhausted,
            assigned_sector_id=(
                None
                if result.assignment is None
                else result.assignment.sector_id
            ),
            assigned_cell_count=(
                0
                if result.assignment is None
                else len(result.assignment.cells)
            ),
            assigned_frontier_cells=(
                0
                if result.assignment is None
                else result.assignment.frontier_cells
            ),
            assigned_estimated_effort=(
                0.0
                if result.assignment is None
                else result.assignment.estimated_effort
            ),
            assigned_frontier_component_ids=(
                ()
                if result.assignment is None
                else tuple(
                    component.component_id
                    for component in result.assignment.frontier_components
                )
            ),
            local_suppression_component_count=len(suppressions),
            local_suppression_component_ids=tuple(
                suppression.component_id
                for suppression in suppressions
            ),
            local_suppression_reasons=tuple(sorted({
                reason
                for suppression in suppressions
                for reason in suppression.reasons
            })),
            rover_slam_version=rover_slam.version,
        )
        return result

    def sector_assignment(self, drone_id: int) -> SectorCheckInResult:
        """Deliver a signalled assignment after one final rover exchange."""
        if self.exploration_completion_event.is_set():
            self.exploration_sectors.stop()
            return SectorCheckInResult(
                arrived=True,
                mission_exhausted=True,
                generation=self.exploration_sectors.generation,
            )
        result = self.exploration_sectors.claim_assignment(drone_id)
        if result.assignment is None or result.mission_exhausted:
            return result
        if self.exploration_completion_event.is_set():
            self.exploration_sectors.stop()
            return SectorCheckInResult(
                arrived=True,
                mission_exhausted=True,
                generation=self.exploration_sectors.generation,
            )
        departed = self.terrain_sharing.share_on_departure(drone_id, 0)
        if not departed:
            return SectorCheckInResult(
                arrived=False,
                waiting_for_team=True,
                generation=result.generation,
            )
        if self.exploration_completion_event.is_set():
            self.exploration_sectors.stop()
            return SectorCheckInResult(
                arrived=True,
                mission_exhausted=True,
                generation=self.exploration_sectors.generation,
            )
        self.runtime_trace.record(
            "drone_sector_assignment_delivered",
            sim_time=self.simulation_time(),
            drone_id=int(drone_id),
            generation=result.assignment.generation,
            sector_id=result.assignment.sector_id,
            frontier_cells=result.assignment.frontier_cells,
            estimated_effort=result.assignment.estimated_effort,
            frontier_component_ids=tuple(
                component.component_id
                for component in result.assignment.frontier_components
            ),
        )
        return result


    def compute_rover_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute the disabled rover path using mission terrain telemetry.

        Rover motion is disabled until its policy is defined. Before enabling
        it, route planning must consume the rover's own received knowledge.
        """
        terrain = self.terrain_knowledge.snapshot()
        return self.pathfinding.compute_weighted_path(
            terrain.roughness,
            terrain.confidence,
            start,
            goal,
        )


    def rover_thread(self, rover_id: int) -> None:
        """Drive rover movement using the terrain-aware weighted planner."""
        label = ("rover", rover_id)
        if not self.pause_coordinator.register_current_worker(label):
            return
        try:
            while not self.mission_event.is_set():
                if not self.pause_checkpoint():
                    break
                self.rovers[rover_id].move()
                if not self.wait_simulation_delay(self.delay):
                    break
        finally:
            self.pause_coordinator.unregister_current_worker()

    def pause_checkpoint(self) -> bool:
        """Park the calling simulation worker at a safe pause boundary."""
        return self.pause_coordinator.checkpoint()

    def wait_simulation_delay(self, duration: float) -> bool:
        """Wait for active simulation time, excluding paused intervals."""
        return self.pause_coordinator.wait(duration)

    def simulation_time(self) -> float:
        """Return monotonic mission time with paused intervals removed."""
        return self.simulation_clock.now()

    def toggle_pause(self) -> None:
        """Atomically park or resume all mission workers and simulation time."""
        if not self.is_paused:
            self.is_paused = True
            self.pause_event.clear()
            self.simulation_clock.pause()
            if self.control_center is not None:
                self.control_center.pause_timer()
            self.pause_coordinator.pause()
            return

        self.simulation_clock.resume()
        if self.control_center is not None:
            self.control_center.resume_timer()
        self.is_paused = False
        self.pause_event.set()
        self.pause_coordinator.resume()

    def music_enabled(self) -> bool:
        """Return the menu-owned music state when available."""
        menu = getattr(self.game, "menu", None)
        if menu is None or not hasattr(menu, "music_enabled"):
            return True
        return bool(menu.music_enabled())

    def toggle_music(self) -> None:
        """Toggle persisted background music through the menu facade."""
        menu = getattr(self.game, "menu", None)
        if menu is not None and hasattr(menu, "toggle_music"):
            menu.toggle_music()

    def update_sensors(self) -> None:
        """Update local sensing and wall-based exploration progress."""
        for drone in self.drones:
            drone.update_sensors()
        self._update_wall_mapping_progress()

    def _update_wall_mapping_progress(self) -> WallMappingSnapshot:
        """Publish wall coverage and start homing at accepted completion."""
        previous = self.wall_mapping_progress
        was_complete = self.wall_mapping_progress.complete
        configured_tolerance = (
            self.settings.mission_config.wall_completion_tolerance_pixels
        )
        previous_tolerance = min(
            configured_tolerance,
            previous.total_wall_pixels // 100,
        )
        was_within_tolerance = bool(
            previous.total_wall_pixels > 0
            and previous.total_wall_pixels - previous.mapped_wall_pixels
            <= previous_tolerance
        )
        versions = tuple(drone.slam_map.version for drone in self.drones)
        if versions == self.wall_mapping_progress.slam_versions:
            progress = self.wall_mapping_progress
        else:
            progress = wall_mapping_snapshot(
                np.asarray(self.map_matrix),
                tuple(self.drones),
                confidence_threshold=(
                    self.settings.frontier.confidence_threshold
                ),
            )
        self.wall_mapping_progress = progress
        missing_wall_pixels = max(
            0,
            progress.total_wall_pixels - progress.mapped_wall_pixels,
        )
        tolerance = min(
            configured_tolerance,
            progress.total_wall_pixels // 100,
        )
        within_tolerance = bool(
            progress.total_wall_pixels > 0
            and missing_wall_pixels <= tolerance
        )
        newly_accepted = within_tolerance and not was_within_tolerance
        if newly_accepted:
            self.exploration_completion_event.set()
            self.exploration_sectors.stop()
            for drone in self.drones:
                drone.runtime_state.start_returning_home()
        if self.runtime_trace is not None:
            if progress.complete and not was_complete:
                self.runtime_trace.record(
                    "team_wall_mapping_complete",
                    mapped_wall_pixels=progress.mapped_wall_pixels,
                    total_wall_pixels=progress.total_wall_pixels,
                    missing_wall_pixels=missing_wall_pixels,
                    drone_count=len(self.drones),
                )
            elif newly_accepted:
                self.runtime_trace.record(
                    "team_wall_mapping_tolerance_reached",
                    mapped_wall_pixels=progress.mapped_wall_pixels,
                    total_wall_pixels=progress.total_wall_pixels,
                    missing_wall_pixels=missing_wall_pixels,
                    tolerance_pixels=tolerance,
                    configured_tolerance_pixels=configured_tolerance,
                    drone_count=len(self.drones),
                )
        now = self.simulation_time()
        if (
            self.control_center is not None
            and (
                self.last_explored_update == 0.0
                or now - self.last_explored_update
                >= self.explored_update_interval
            )
        ):
            displayed_percent = (
                100
                if progress.complete
                else min(99, int(progress.ratio * 100.0))
            )
            self.control_center.set_explored_percent(displayed_percent)
            self.last_explored_update = now
        return progress
