"""Mission orchestration and runtime resource setup.

Constructing `MissionControl` prepares mission state only. Calling `run()`
initializes the mission window, agents, pathfinding resources, and worker
threads before entering the main loop.
"""

import random as rand
from dataclasses import dataclass, field, replace
from queue import Empty, Queue
import math
import threading
import time
from pathlib import Path
from typing import List, Tuple, Any, Optional

import numpy as np
import cv2
import pygame

from asset_config.helpers import wall_hit
from agents.factory import AgentFactory
from ui.control_center.facade import ControlCenter
from mapping.rover_targets import RoverFrontierTarget, RoverTargetService
from mapping.exploration_sectors import (
    ExplorationSectorCoordinator,
    SectorCheckInResult,
)
from mapping.terrain_fusion import TerrainFusionService
from mapping.terrain_knowledge import TerrainKnowledge
from mapping.terrain_sharing import TerrainSharingService
from mapping.wall_mapping import WallMappingSnapshot, wall_mapping_snapshot
from mapping.slam_map import FREE
from mission.debug_info import MissionDebugInfo
from mission.frame_timing import FrameProfiler
from mission.energy import EnergyState
from mission.exploration_coordination import (
    CoordinationReport,
    CoordinationResult,
    DirectiveKind,
    FrontierTaskCoordinator,
)
from mission.objectives import build_mission_objective
from mission.pause_control import PauseCoordinator, SimulationClock
from mission.rendezvous import RendezvousProtocol
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
from navigation.highway import HighwayBuildResult, HighwayService
from mission.presentation_adapter import PresentationAdapter
from rendering.slam_renderer import SlamRenderer
from rendering.mission_renderer import MissionRenderer
from rendering.sector_renderer import SectorRenderer
from rendering.highway_renderer import HighwayRenderer
from rendering.slam_view import SlamViewService
from mission.lifecycle import MissionControlLifecycleMixin


@dataclass
class _PendingExplorationCheckIn:
    """One drone report queued for execution on the primary rover thread."""

    drone_id: int
    report: CoordinationReport | None
    ready: threading.Event
    enqueued_at: float
    result: CoordinationResult | None = None


@dataclass
class _DockedExplorationDrone:
    """One verified mechanical attachment to the primary rover."""

    drone_id: int
    rover_id: int
    source: str
    report_id: int | None
    started_at: float
    contact_position: Tuple[int, int]
    carried_distance: float = 0.0
    carried_steps: int = 0
    carried_path: list[Tuple[int, int]] = field(default_factory=list)
    learned_endpoint_epochs: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class _RoverStagingSelection:
    """One rover-local service point with focused-endgame distribution cost."""

    position: Tuple[int, int]
    rover_route_cost: float
    terrain_roughness: float
    wall_clearance: float
    remaining_max_distance: float = math.inf
    remaining_total_distance: float = math.inf
    current_max_distance: float = math.inf
    current_total_distance: float = math.inf


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
        # Rovers advance from their own received maps on dedicated workers.
        self.rover_motion_enabled = True
        self.rendezvous_protocol: RendezvousProtocol | None = None
        self._rover_staging_cache_key = None
        self._rover_staging_cache = None
        self._rover_failed_route_cache: set[
            tuple[
                int,
                Tuple[int, int],
                Tuple[int, int],
                int,
                float,
            ]
        ] = set()
        self._last_focused_endgame_trace_state = False
        self._last_focused_staging_hold_revision: int | None = None

        # Runtime resources are initialized explicitly by run().
        # Pathfinding owns external resources (shared memory and a process pool)
        # but does not allocate them until ``run`` calls ``start``.
        self.pathfinding = PathfindingService(
            self.map_matrix,
            self.settings.mission_config.num_drones,
        )
        self.highway = HighwayService(
            confidence_threshold=(
                self.settings.frontier.confidence_threshold
            ),
            macro_cell_size=self.settings.highway.macro_cell_size,
            maximum_access_distance_sensor_ranges=(
                self.settings.highway.maximum_access_distance_sensor_ranges
            ),
            maximum_build_ms=self.settings.highway.maximum_build_ms,
            maximum_query_ms=self.settings.highway.maximum_query_ms,
            maximum_connector_expansions=(
                self.settings.highway.maximum_connector_expansions
            ),
        )
        self._last_highway_build_version: int | None = None
        self._last_rover_knowledge_version: int | None = None
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
        # Report-stop reservations and rover steps share this lock so a rover
        # cannot leave between verified contact and the queued check-in.
        self._exploration_check_in_lock = threading.RLock()
        self._exploration_check_in_queue: Queue[
            _PendingExplorationCheckIn
        ] = Queue()
        self._exploration_check_ins: dict[
            int, _PendingExplorationCheckIn
        ] = {}
        self._exploration_report_stops: set[int] = set()
        self._exploration_docked: dict[
            int, _DockedExplorationDrone
        ] = {}
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
                "physical_team_quiescence_after_component_exhaustion"
            ),
            exploration_progress="discovered_floor_coverage_display_only",
            terrain_role="rover_checkpoint_and_navigation",
            frontier_policy=(
                "coordinated_scan_then_frontier_component_dfs"
            ),
            component_cost_cell_size=self.settings.frontier.global_cell_size,
            component_check_in="queued_moving_rover_rendezvous",
            component_check_in_path="astar_with_breadcrumb_fallback",
            component_waiting_state="physical_rover_dock",
            component_docked_sensing=False,
            component_assignment_delivery="rover_signalled_no_polling",
            component_task_routing=(
                "rover_known_free_connectivity_then_drone_astar"
            ),
            component_dfs_navigation=(
                "logical_unwind_direct_astar_with_breadcrumb_fallback"
            ),
            component_probe_routing=(
                "bootstrap_only_rover_known_free_then_drone_astar"
            ),
            component_follower_policy=(
                "spare_drone_follows_leader_then_reserves_first_split_branch"
            ),
            incidental_scan_mode=self.settings.incidental_scan.mode,
            incidental_scan_maximum_attempts=(
                self.settings.incidental_scan.maximum_attempts_per_directive
            ),
            incidental_scan_maximum_wait_seconds=(
                self.settings.incidental_scan.maximum_wait_seconds_per_directive
            ),
            incidental_scan_attempt_timeout_seconds=(
                self.settings.incidental_scan.attempt_timeout_seconds
            ),
            incidental_scan_maximum_rotation_degrees=(
                self.settings.incidental_scan
                .maximum_rotation_degrees_per_directive
            ),
            incidental_scan_distance_cooldown_ranges=(
                self.settings.incidental_scan.distance_cooldown_sensor_ranges
            ),
            focused_frontier_batch_mode=self.settings.focused_frontier_batch.mode,
            focused_frontier_batch_maximum_claimed_components=(
                self.settings.focused_frontier_batch.maximum_claimed_components
            ),
            focused_frontier_batch_maximum_total_components=(
                self.settings.focused_frontier_batch.maximum_total_components
            ),
            focused_frontier_batch_maximum_detour_sensor_ranges=(
                self.settings.focused_frontier_batch.maximum_detour_sensor_ranges
            ),
            focused_frontier_batch_maximum_service_seconds=(
                self.settings.focused_frontier_batch.maximum_service_seconds
            ),
            focused_frontier_batch_maximum_planning_ms=(
                self.settings.focused_frontier_batch.maximum_planning_ms
            ),
            focused_frontier_batch_maximum_route_queries=(
                self.settings.focused_frontier_batch.maximum_route_queries
            ),
            highway_mode=self.settings.highway.mode,
            highway_macro_cell_size=self.settings.highway.macro_cell_size,
            highway_topology="corridor_backbone",
            highway_maximum_access_distance_sensor_ranges=(
                self.settings.highway.maximum_access_distance_sensor_ranges
            ),
            highway_refresh_policy="every_rover_slam_version",
            highway_build_policy="throttled_full_rebuild_process",
            highway_minimum_rebuild_seconds=2.0,
            highway_minimum_version_delta=1,
            highway_configured_minimum_version_delta=(
                self.settings.highway.minimum_version_delta
            ),
            highway_maximum_build_ms=(
                self.settings.highway.maximum_build_ms
            ),
            highway_maximum_query_ms=(
                self.settings.highway.maximum_query_ms
            ),
            highway_maximum_connector_expansions=(
                self.settings.highway.maximum_connector_expansions
            ),
            component_zero_gain_memory="individual_anchor_or_subarc_plus_lineage_low_gain",
            component_focused_endgame_policy=(
                "max_parallel_then_round_trip_distribution"
            ),
            rover_failed_route_cache="slam_start_target_body",
            component_workload_estimate=(
                "frontier_cells_plus_unknown_pixels_per_cell_size_squared"
            ),
            component_zero_work_policy=(
                "bootstrap_probe_then_component_team_quiescence"
            ),
            component_claim_lease="token_fenced_explicit_release",
            component_energy_policy="unlimited_via_energy_contract",
            component_wide_classifier=(
                "multiple_footprints_and_wall_contact_count_ratio_run"
            ),
            frontier_unknown_basin_rescue="interior_basins_only",
            rover_periodic_sharing=True,
            rover_drone_pair_sharing=True,
            sharing_pause_policy="once_per_continuous_contact_non_extending",
            rover_count=1,
            rover_rendezvous_protocol=(
                "contact_carried_ack_departure_or_confirmed_target"
            ),
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
        self.exploration_complete = False

        # Initialize presentation adapter for UI state and map rendering
        self.presentation = PresentationAdapter()
        self.slam_renderer = SlamRenderer(
            self.map_w,
            self.map_h,
            frontier_confidence_threshold=(
                self.settings.frontier.confidence_threshold
            ),
        )
        self.sector_renderer = SectorRenderer(self.map_w, self.map_h)
        self.highway_renderer = HighwayRenderer(self.map_w, self.map_h)
        self.last_explored_update = 0.0
        self.wall_mapping_progress = WallMappingSnapshot(0, 0, 0.0, False)
        self.floor_exploration_ratio = 0.0
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
                periodic_rover_sharing_enabled=True,
                on_drone_contact=self._rendezvous_drone_contact,
                on_drone_rover_contact=self._rendezvous_drone_rover_contact,
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
                get_frontier_candidates=self._rover_frontier_candidates,
                should_hold_position=self._rover_should_hold_position,
                simulation_time=self.simulation_time,
                runtime_trace=self.runtime_trace,
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
                get_rovers=lambda: self.rovers,
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
                get_rovers=lambda: self.rovers,
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
                get_sector_snapshot=lambda: (
                    self.exploration_coordinator.snapshot(blocking=False)
                    if self.exploration_coordinator is not None
                    else self.exploration_sectors.snapshot()
                ),
                is_exploration_complete=lambda: self.exploration_complete,
                get_docked_drone_ids=self.exploration_docked_ids,
                highway_renderer=self.highway_renderer,
                get_highway_snapshot=lambda rover_id: (
                    self.highway.snapshot if rover_id == 0 else None
                ),
            )
        )
        
        # Set the starting position for drones
        self.start_point = None
        self.set_start_point()
        self.rendezvous_protocol = self._new_rendezvous_protocol()
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
        # Built after drones so the coordinator uses the actual sensor
        # footprint.  The legacy sector coordinator remains available to
        # existing focused tests while production drones use this replacement.
        self.exploration_coordinator: FrontierTaskCoordinator | None = None
        self._traced_discovery_rounds: set[int] = set()

    def _initialize_runtime(self) -> None:
        """Create window, agents, pathfinding resources, and first frame."""
        if self._runtime_initialized:
            return

        self.completed = False
        self.exploration_complete = False
        self.mission_event.clear()
        self.exploration_completion_event.clear()
        self.pause_event.set()
        self.is_paused = False
        self.game.display = self.game.to_maximised()
        self.control_center = ControlCenter(self.game)
        self.rendezvous_protocol = self._new_rendezvous_protocol()
        self._exploration_docked.clear()
        self._rover_failed_route_cache.clear()
        self._last_focused_endgame_trace_state = False
        self._last_focused_staging_hold_revision = None
        self._last_highway_build_version = None
        self._last_rover_knowledge_version = None

        AgentFactory.build_drones(self)
        AgentFactory.build_rovers(self)

        if not self.drones:
            raise RuntimeError("component exploration requires at least one drone")
        vision_sensor = self.drones[0].sensor_controller.vision_sensor
        self.highway.sensor_range = float(vision_sensor.max_range)
        if self.settings.highway.mode != "off" or self.settings.focused_frontier_batch.mode != "off":
            self.highway.start()
        self.exploration_coordinator = FrontierTaskCoordinator(
            (self.map_h, self.map_w),
            self.num_drones,
            self.start_point,
            sensor_range=float(vision_sensor.max_range),
            sensor_fov_deg=float(vision_sensor.fov_deg),
            confidence_threshold=(
                self.settings.frontier.confidence_threshold
            ),
            minimum_component_cells=(
                self.settings.frontier.minimum_cluster_cells
            ),
            minimum_unknown_support_cells=(
                self.settings.frontier.minimum_unknown_support_cells
            ),
            frontier_stride=self.settings.frontier.stride,
            global_cell_size=self.settings.frontier.global_cell_size,
            focused_frontier_batch_mode=self.settings.focused_frontier_batch.mode,
            focused_frontier_batch_maximum_claimed_components=(
                self.settings.focused_frontier_batch.maximum_claimed_components
            ),
            focused_frontier_batch_maximum_total_components=(
                self.settings.focused_frontier_batch.maximum_total_components
            ),
            focused_frontier_batch_lease_margin_sensor_ranges=(
                self.settings.focused_frontier_batch.lease_margin_sensor_ranges
            ),
            focused_frontier_batch_maximum_detour_sensor_ranges=(
                self.settings.focused_frontier_batch.maximum_detour_sensor_ranges
            ),
            focused_frontier_batch_minimum_avoided_round_trip_sensor_ranges=(
                self.settings.focused_frontier_batch
                .minimum_avoided_round_trip_sensor_ranges
            ),
            focused_frontier_batch_maximum_service_seconds=(
                self.settings.focused_frontier_batch.maximum_service_seconds
            ),
            focused_frontier_batch_maximum_total_dfs_nodes=(
                self.settings.focused_frontier_batch.maximum_total_dfs_nodes
            ),
            focused_frontier_batch_maximum_consecutive_low_gain_scans=(
                self.settings.focused_frontier_batch
                .maximum_consecutive_low_gain_scans
            ),
            focused_frontier_batch_low_gain_maximum_new_cells=(
                self.settings.focused_frontier_batch.low_gain_maximum_new_cells
            ),
            focused_frontier_batch_low_gain_maximum_confidence_gain=(
                self.settings.focused_frontier_batch.low_gain_maximum_confidence_gain
            ),
            focused_frontier_batch_maximum_planning_ms=(
                self.settings.focused_frontier_batch.maximum_planning_ms
            ),
            focused_frontier_batch_maximum_route_queries=(
                self.settings.focused_frontier_batch.maximum_route_queries
            ),
        )

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

    def _rover_frontier_candidates(self) -> tuple[RoverFrontierTarget, ...]:
        """Expose rover-local registry tasks as stable staging candidates."""
        coordinator = self.exploration_coordinator
        if coordinator is None:
            return ()
        snapshot = coordinator.published_snapshot()
        live_components = {
            component.component_id
            for component in snapshot.components
            if component.state.value == "active"
        }
        components_by_id = {
            component.component_id: component
            for component in snapshot.components
        }
        claimed_task_ids = {claim.task_id for claim in snapshot.claims}
        staging = self._rover_staging_context()
        tasks = tuple(
            task for task in snapshot.tasks
            if (
                task.component_id in live_components
                and task.state.value in {
                    "ready",
                    "claimed",
                    "active",
                    "suspended",
                }
            )
        )
        outstanding_entries = tuple(
            tuple(task.preferred_entry) for task in tasks
        )
        candidates = []
        for task in tasks:
            selected = self._select_rover_staging(
                task.preferred_entry,
                staging,
                outstanding_entries=(
                    outstanding_entries
                    if snapshot.focused_endgame
                    else ()
                ),
            )
            if selected is None:
                continue
            position = selected.position
            candidates.append(RoverFrontierTarget(
                position=position,
                component_id=task.component_id,
                task_id=task.task_id,
                claimed=task.task_id in claimed_task_ids,
                depth=task.depth,
                estimated_effort=task.estimated_effort,
                parent_component_ids=components_by_id[
                    task.component_id
                ].parent_ids,
                service_cost=(
                    selected.rover_route_cost
                    + math.dist(position, task.preferred_entry)
                ),
                rover_route_cost=selected.rover_route_cost,
                terrain_roughness=selected.terrain_roughness,
                wall_clearance=selected.wall_clearance,
                focused_endgame=snapshot.focused_endgame,
                remaining_max_distance=(
                    selected.remaining_max_distance
                ),
                remaining_total_distance=(
                    selected.remaining_total_distance
                ),
                current_max_distance=selected.current_max_distance,
                current_total_distance=selected.current_total_distance,
            ))
        if (
            snapshot.focused_endgame
            and tasks
            and not candidates
            and self._last_focused_staging_hold_revision
            != snapshot.revision
        ):
            self._last_focused_staging_hold_revision = snapshot.revision
            rover_position = (
                None
                if not self.rovers or self.rovers[0] is None
                else tuple(self.rovers[0].pos)
            )
            self.runtime_trace.record(
                "rover_focused_endgame_staging_held",
                sim_time=self.simulation_time(),
                registry_revision=snapshot.revision,
                rover_position=rover_position,
                task_ids=tuple(task.task_id for task in tasks),
                outstanding_entries=outstanding_entries,
                reason="no_distribution_safe_candidate",
            )
        return tuple(candidates)

    def _rover_staging_context(self):
        """Build clearance and terrain arrays from the primary rover's map."""
        if not self.rovers or self.rovers[0] is None:
            return None
        rover = self.rovers[0]
        slam = rover.slam_map.snapshot(point_limit=0)
        terrain_version = int(rover.terrain_knowledge.version)
        icon = getattr(rover, "icon", None)
        body_radius = max(
            1.0,
            float(max(icon.get_size())) / 2.0 if icon is not None else 1.0,
        )
        cache_key = (int(slam.version), terrain_version, body_radius)
        if self._rover_staging_cache_key == cache_key:
            labels, clearance, asperity, terrain_confidence = (
                self._rover_staging_cache
            )
        else:
            known_free = (
                (np.asarray(slam.occupancy) == FREE)
                & (
                    np.asarray(slam.confidence)
                    >= self.settings.frontier.confidence_threshold
                )
            )
            padded = np.pad(known_free.astype(np.uint8), 1)
            clearance = cv2.distanceTransform(
                padded,
                cv2.DIST_L2,
                5,
            )[1:-1, 1:-1]
            safe = known_free & (clearance >= body_radius)
            _count, labels = cv2.connectedComponents(
                safe.astype(np.uint8),
                # A* permits a diagonal only if an orthogonal side is free.
                # Such a move is already connected by orthogonal steps, so
                # four-connectivity exactly matches route existence here.
                connectivity=4,
            )
            terrain = rover.terrain_knowledge.snapshot()
            terrain_confidence = terrain.confidence
            roughness = np.where(
                terrain_confidence > 0.0,
                terrain.roughness,
                1.0,
            ).astype(np.float32)
            diameter = max(1, int(math.ceil(body_radius * 2.0)) + 1)
            asperity = cv2.dilate(
                roughness,
                np.ones((diameter, diameter), dtype=np.uint8),
            )
            self._rover_staging_cache_key = cache_key
            self._rover_staging_cache = (
                labels,
                clearance,
                asperity,
                terrain_confidence,
            )
        rover_x, rover_y = int(rover.pos[0]), int(rover.pos[1])
        if not (
            0 <= rover_y < labels.shape[0]
            and 0 <= rover_x < labels.shape[1]
        ):
            return None
        rover_label = int(labels[rover_y, rover_x])
        if rover_label <= 0:
            return None
        safe = labels == rover_label
        return (
            safe,
            clearance,
            asperity,
            terrain_confidence,
            tuple(rover.pos),
            body_radius,
        )

    def _select_rover_staging(
        self,
        entry,
        context,
        *,
        outstanding_entries=(),
    ) -> _RoverStagingSelection | None:
        """Choose a low-asperity, wall-clear service point near a task."""
        if context is None:
            return None
        safe, clearance, roughness, terrain_confidence, rover_pos, radius = context
        entry_x, entry_y = int(entry[0]), int(entry[1])
        service_radius = max(
            radius * 4.0,
            float(self.settings.frontier.global_cell_size) * 2.0,
        )
        left = max(0, int(math.floor(entry_x - service_radius)))
        right = min(safe.shape[1], int(math.ceil(entry_x + service_radius + 1)))
        top = max(0, int(math.floor(entry_y - service_radius)))
        bottom = min(safe.shape[0], int(math.ceil(entry_y + service_radius + 1)))
        local_safe = safe[top:bottom, left:right]
        ys, xs = np.nonzero(local_safe)
        if len(xs) == 0:
            return None
        outstanding_entries = tuple(
            (int(point[0]), int(point[1]))
            for point in outstanding_entries
        )
        current_distances = tuple(
            math.dist(rover_pos, point) for point in outstanding_entries
        )
        current_max_distance = max(current_distances, default=math.inf)
        current_total_distance = sum(current_distances)
        distribution_tolerance = max(
            float(radius),
            float(self.settings.frontier.global_cell_size),
        )
        ranked = []
        for local_x, local_y in zip(xs, ys):
            x, y = int(local_x) + left, int(local_y) + top
            point = int(x), int(y)
            service_distance = math.dist(point, entry)
            if service_distance > service_radius:
                continue
            if float(terrain_confidence[y, x]) <= 0.0:
                continue
            known_roughness = float(roughness[y, x])
            rover_distance = math.dist(rover_pos, point)
            remaining_distances = tuple(
                math.dist(point, outstanding)
                for outstanding in outstanding_entries
            )
            remaining_max_distance = max(
                remaining_distances,
                default=math.inf,
            )
            remaining_total_distance = sum(remaining_distances)
            if outstanding_entries and (
                remaining_max_distance
                > current_max_distance + distribution_tolerance
                or remaining_total_distance
                > current_total_distance
                + distribution_tolerance * len(outstanding_entries)
            ):
                continue
            ranked.append((
                (
                    remaining_max_distance
                    if outstanding_entries else 0.0
                ),
                (
                    remaining_total_distance
                    if outstanding_entries else 0.0
                ),
                service_distance + rover_distance,
                known_roughness,
                -float(clearance[y, x]),
                rover_distance,
                int(y),
                int(x),
                point,
            ))
        if not ranked:
            return None
        selected = min(ranked)
        point = selected[-1]
        return _RoverStagingSelection(
            position=point,
            rover_route_cost=selected[5],
            terrain_roughness=selected[3],
            wall_clearance=-selected[4],
            remaining_max_distance=(
                selected[0] if outstanding_entries else math.inf
            ),
            remaining_total_distance=(
                selected[1] if outstanding_entries else math.inf
            ),
            current_max_distance=(
                current_max_distance if outstanding_entries else math.inf
            ),
            current_total_distance=(
                current_total_distance if outstanding_entries else math.inf
            ),
        )

    def _rover_should_hold_position(self, rover_id: int) -> bool:
        """Hold reports, or a completed team physically beside the rover."""
        if int(rover_id) != 0:
            return False
        with self._exploration_check_in_lock:
            if self._exploration_report_stops:
                return True
            if any(
                drone_id not in self._exploration_docked
                for drone_id in self._exploration_check_ins
            ):
                return True
            coordinator = self.exploration_coordinator
            snapshot = getattr(coordinator, "snapshot", None)
            if not callable(snapshot) or not self.drones:
                return False
            if not snapshot(blocking=False).mission_exhausted:
                return False
            if len(self.drones) != coordinator.drone_count:
                return False
            return len(self._exploration_docked) == len(self.drones)

    def get_check_in_position(
        self,
        drone_id: int | None = None,
    ) -> Tuple[int, int]:
        """Return physical rover position or one drone's selected rendezvous."""
        if drone_id is not None and self.rendezvous_protocol is not None:
            return self.rendezvous_protocol.drone_endpoint(int(drone_id))
        if self.rovers and self.rovers[0] is not None:
            return tuple(self.rovers[0].pos)
        return tuple(self.start_point)

    def rendezvous_endpoint_missed(
        self,
        drone_id: int,
        position: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Advance one drone only after it observes its confirmed point empty."""
        protocol = self.rendezvous_protocol
        if protocol is None:
            return tuple(position)
        return protocol.drone_missed_endpoint(int(drone_id), position)

    def visible_drone_positions(
        self,
        observer_id: int,
    ) -> tuple[tuple[int, Tuple[int, int]], ...]:
        """Expose no peer position outside direct LOS/proximity."""
        return self.terrain_sharing.visible_drone_positions(observer_id)

    def physical_contact_checkpoint(self, drone_id: int) -> None:
        """Process peer contact reached during one physical translation step."""
        self.terrain_sharing.physical_contact_checkpoint(int(drone_id))

    def announce_rendezvous(self, position: Tuple[int, int]):
        protocol = self.rendezvous_protocol
        return None if protocol is None else protocol.propose(position)

    def rendezvous_departure_ready(self, position: Tuple[int, int]) -> bool:
        protocol = self.rendezvous_protocol
        return protocol is None or protocol.can_depart(position)

    def rendezvous_arrived(self, position: Tuple[int, int]) -> bool:
        protocol = self.rendezvous_protocol
        return protocol is None or protocol.rover_arrived(position)

    def rendezvous_departed(self, position: Tuple[int, int]) -> bool:
        protocol = self.rendezvous_protocol
        return protocol is None or protocol.rover_departed(position)

    def _rendezvous_drone_contact(self, first_id: int, second_id: int) -> None:
        if self.rendezvous_protocol is not None:
            self.rendezvous_protocol.drone_drone_contact(first_id, second_id)

    def _rendezvous_drone_rover_contact(self, drone_id: int) -> None:
        with self._exploration_check_in_lock:
            protocol = self.rendezvous_protocol
            if protocol is None:
                return
            normalized_id = int(drone_id)
            protocol.drone_rover_contact(normalized_id)
            self._record_docked_endpoint_epoch_locked(normalized_id)

    def _record_docked_endpoint_epoch_locked(self, drone_id: int) -> None:
        """Attach physically learned endpoint identity to one dock session."""
        protocol = self.rendezvous_protocol
        session = self._exploration_docked.get(int(drone_id))
        if protocol is None or session is None:
            return
        try:
            announcement = protocol.snapshot().drone_announcements[
                int(drone_id)
            ]
            session.learned_endpoint_epochs.add(int(announcement.epoch))
        except (AttributeError, IndexError, TypeError):
            # Lightweight controller doubles may implement only contact and
            # endpoint lookup. Runtime protocols always expose this snapshot.
            return

    def _reserve_exploration_report_stop_locked(
        self,
        drone_id: int,
    ) -> None:
        if (
            drone_id in self._exploration_report_stops
            or drone_id in self._exploration_check_ins
        ):
            return
        self._exploration_report_stops.add(drone_id)
        distance = math.dist(
            self.drones[drone_id].snapshot().position,
            tuple(self.rovers[0].pos),
        )
        self.runtime_trace.record(
            "rover_report_stop_reserved",
            sim_time=self.simulation_time(),
            drone_id=drone_id,
            distance=distance,
        )

    def request_exploration_report_stop(self, drone_id: int) -> bool:
        """Atomically verify rover contact and reserve a stop for a report."""
        normalized_id = int(drone_id)
        if (
            self.exploration_coordinator is None
            or not self.rovers
            or self.rovers[0] is None
        ):
            return False
        with self._exploration_check_in_lock:
            if not self.terrain_sharing.drone_at_rover(normalized_id, 0):
                return False
            self._reserve_exploration_report_stop_locked(normalized_id)
            if self.rendezvous_protocol is not None:
                self.rendezvous_protocol.drone_rover_contact(normalized_id)
            return True

    def is_exploration_docked(self, drone_id: int) -> bool:
        """Return whether one drone is mechanically attached to rover 0."""
        with self._exploration_check_in_lock:
            return int(drone_id) in self._exploration_docked

    def exploration_docked_ids(self) -> frozenset[int]:
        """Return one coherent render-safe snapshot of attached drones."""
        with self._exploration_check_in_lock:
            return frozenset(self._exploration_docked)

    def request_exploration_dock(self, drone_id: int) -> CoordinationResult:
        """Dock through the ordinary physical no-report check-in."""
        return self.exploration_check_in(int(drone_id), None)

    def _dock_exploration_drone_locked(
        self,
        drone_id: int,
        report: CoordinationReport | None,
    ) -> None:
        """Attach one physically verified drone while movement is serialized."""
        if drone_id in self._exploration_docked:
            return
        report_stop_reserved = drone_id in self._exploration_report_stops
        rover = self.rovers[0]
        drone = self.drones[drone_id]
        contact_position = tuple(drone.snapshot().position)
        rover_position = tuple(rover.pos)
        endpoint = (
            rover_position
            if self.rendezvous_protocol is None
            else tuple(self.rendezvous_protocol.drone_endpoint(drone_id))
        )
        encounter = contact_position != endpoint
        if report is None:
            source = "check_in_intercept" if encounter else "check_in_endpoint"
        else:
            source = "report_intercept" if encounter else "report_endpoint"
        drone.runtime_state.carry_to(rover_position)
        drone.runtime_state.clear_ray_points()
        self._exploration_docked[drone_id] = _DockedExplorationDrone(
            drone_id=drone_id,
            rover_id=0,
            source=source,
            report_id=None if report is None else report.report_id,
            started_at=self.simulation_time(),
            contact_position=contact_position,
            carried_path=[rover_position],
        )
        if self.rendezvous_protocol is not None:
            if not report_stop_reserved:
                self._rendezvous_drone_rover_contact(drone_id)
            else:
                self._record_docked_endpoint_epoch_locked(drone_id)
        self.runtime_trace.record(
            "drone_docked",
            sim_time=self.simulation_time(),
            drone_id=drone_id,
            rover_id=0,
            source=source,
            report_id=None if report is None else report.report_id,
            contact_position=contact_position,
            rover_position=rover_position,
            remembered_endpoint=endpoint,
        )

    def _move_docked_drones_locked(self, rover_id: int) -> None:
        """Carry attached drones through one serialized rover step."""
        rover_position = tuple(self.rovers[rover_id].pos)
        for drone_id, session in self._exploration_docked.items():
            if session.rover_id != rover_id:
                continue
            distance = self.drones[drone_id].runtime_state.carry_to(
                rover_position
            )
            if distance <= 0.0:
                continue
            session.carried_distance += distance
            session.carried_steps += 1
            session.carried_path.append(rover_position)

    def _undock_exploration_drone_locked(
        self,
        drone_id: int,
        *,
        reason: str,
        directive_id: int | None = None,
        directive_kind: str | None = None,
    ) -> None:
        """Release one attachment and summarize its carried movement."""
        session = self._exploration_docked.pop(int(drone_id), None)
        if session is None:
            return
        # A scan that began immediately before docking may have published its
        # old world-space rays while the attachment was active.  Keep the
        # first released frame clear until sensing publishes a fresh pose.
        self.drones[session.drone_id].runtime_state.clear_ray_points()
        self.runtime_trace.record(
            "drone_undocked",
            sim_time=self.simulation_time(),
            drone_id=session.drone_id,
            rover_id=session.rover_id,
            reason=reason,
            directive_id=directive_id,
            directive_kind=directive_kind,
            docking_source=session.source,
            report_id=session.report_id,
            docked_seconds=max(
                0.0, self.simulation_time() - session.started_at
            ),
            contact_position=session.contact_position,
            release_position=self.drones[session.drone_id].snapshot().position,
            carried_distance=session.carried_distance,
            carried_steps=session.carried_steps,
            carried_path=tuple(session.carried_path),
            learned_endpoint_epochs=tuple(sorted(
                session.learned_endpoint_epochs
            )),
        )

    def _clear_exploration_docks(self, *, reason: str) -> None:
        """Release all remaining attachments during lifecycle teardown."""
        with self._exploration_check_in_lock:
            for drone_id in tuple(self._exploration_docked):
                self._undock_exploration_drone_locked(
                    drone_id,
                    reason=reason,
                )

    def _new_rendezvous_protocol(self) -> RendezvousProtocol:
        return RendezvousProtocol(
            self.num_drones,
            tuple(self.start_point),
            trace=self._trace_rendezvous,
        )

    def _trace_rendezvous(self, event: str, **fields: object) -> None:
        self.runtime_trace.record(
            event,
            sim_time=self.simulation_time(),
            **fields,
        )

    def exploration_check_in(
        self,
        drone_id: int,
        report: CoordinationReport | None = None,
    ) -> CoordinationResult:
        """Queue a physical rendezvous for the primary rover worker."""
        coordinator = self.exploration_coordinator
        normalized_id = int(drone_id)
        if coordinator is None or not self.rovers or self.rovers[0] is None:
            return CoordinationResult(arrived=False)
        with self._exploration_check_in_lock:
            if not self.terrain_sharing.drone_at_rover(normalized_id, 0):
                return CoordinationResult(arrived=False)
            self._dock_exploration_drone_locked(normalized_id, report)
            pending = self._exploration_check_ins.get(normalized_id)
            if pending is None:
                pending = _PendingExplorationCheckIn(
                    drone_id=normalized_id,
                    report=report,
                    ready=threading.Event(),
                    enqueued_at=time.perf_counter(),
                )
                self._exploration_check_ins[normalized_id] = pending
                self._exploration_check_in_queue.put(pending)
                self.runtime_trace.record(
                    "drone_component_check_in_queued",
                    sim_time=self.simulation_time(),
                    drone_id=normalized_id,
                    report_id=None if report is None else report.report_id,
                    directive_id=(
                        None if report is None else report.directive_id
                    ),
                )
            # Docking now preserves contact while the queued work is handled.
            # The pre-queue report reservation is no longer needed.
            self._exploration_report_stops.discard(normalized_id)
        return CoordinationResult(
            arrived=True,
            report_accepted=report is None,
            waiting=True,
            directive_ready=pending.ready,
        )

    def exploration_contact(self, drone_id: int) -> bool:
        """Check direct rover contact without reserving a report stop."""
        with self._exploration_check_in_lock:
            normalized_id = int(drone_id)
            if normalized_id in self._exploration_docked:
                return True
            at_rover = bool(self.terrain_sharing.drone_at_rover(normalized_id, 0))
            coordinator = self.exploration_coordinator
            if not at_rover and coordinator is not None:
                coordinator.waiting_contact_lost(normalized_id)
            return at_rover

    def _prune_exploration_waiting_contacts(self) -> None:
        """Expire old check-ins before evaluating team quiescence."""
        coordinator = self.exploration_coordinator
        if coordinator is None:
            return
        waiting = coordinator.snapshot().waiting_drone_ids
        for drone_id in waiting:
            if drone_id in self._exploration_docked:
                continue
            if not self.terrain_sharing.drone_at_rover(drone_id, 0):
                coordinator.waiting_contact_lost(drone_id)
                self.runtime_trace.record(
                    "rover_waiting_contact_expired",
                    sim_time=self.simulation_time(),
                    drone_id=drone_id,
                )

    def _refresh_highway_if_due(
        self,
        rover_slam,
        *,
        refresh_source: str = "physical_check_in",
    ) -> HighwayBuildResult | None:
        """Request advice once per newly received rover SLAM version."""
        if (
            self.settings.highway.mode == "off"
            and self.settings.focused_frontier_batch.mode == "off"
        ):
            return None
        version = int(rover_slam.version)
        last_version = self._last_highway_build_version
        if last_version == version:
            return None
        self._last_highway_build_version = version
        result = self.highway.refresh(rover_slam)
        self._record_highway_build(result, version, refresh_source)
        return result

    def _record_highway_build(self, result, version, refresh_source):
        retained = self.highway.snapshot
        self.runtime_trace.record(
            "rover_highway_build_completed",
            sim_time=self.simulation_time(),
            mode=self.settings.highway.mode,
            status=result.status,
            requested_version=version,
            source_version=result.source_version,
            build_kind=result.build_kind,
            published_version=(
                None if retained is None else retained.version
            ),
            retained_previous=(
                result.snapshot is None and retained is not None
            ),
            elapsed_ms=result.elapsed_ms,
            area_count=result.area_count,
            edge_count=result.edge_count,
            known_free_cells=result.known_free_cells,
            connector_expansions=result.connector_expansions,
            topology="corridor_backbone",
            refresh_source=refresh_source,
            component_count=0 if retained is None else retained.component_count,
            maximum_access_distance=(
                None if retained is None else retained.maximum_access_distance
            ),
            measured_access_distance=(
                None if retained is None else retained.measured_access_distance
            ),
            pruned_branches=0 if retained is None else retained.pruned_branches,
            capillary_branches=0 if retained is None else retained.capillary_branches,
            skeleton_scale=None if retained is None else retained.skeleton_scale,
            skeleton_method=None if retained is None else retained.skeleton_method,
        )

    def _poll_highway(self):
        result = self.highway.poll()
        if result is not None:
            graph = result.snapshot
            version = result.source_version if result.source_version is not None else self._last_highway_build_version
            self._record_highway_build(result, version, "background_worker")
            if graph is not None and self.exploration_coordinator is not None:
                self.exploration_coordinator.update_highway_snapshot(graph)

    def _refresh_rover_knowledge_if_changed(self) -> None:
        """Refresh frontiers and highways together on the primary rover worker."""
        coordinator = self.exploration_coordinator
        if coordinator is None or not self.rovers or self.rovers[0] is None:
            return
        slam_map = self.rovers[0].slam_map
        if slam_map.version == self._last_rover_knowledge_version:
            return
        rover_slam = slam_map.snapshot(point_limit=0)
        self._refresh_highway_if_due(rover_slam, refresh_source="rover_knowledge_update")
        coordinator.update_highway_snapshot(self.highway.snapshot)
        reconcile = coordinator.refresh_rover_knowledge(rover_slam)
        if reconcile is not None:
            self._record_frontier_reconcile(
                reconcile, rover_slam, coordinator.published_snapshot(),
                refresh_source="rover_knowledge_update",
            )
        self._last_rover_knowledge_version = int(rover_slam.version)

    def _perform_exploration_check_in(
        self,
        drone_id: int,
        report: CoordinationReport | None = None,
    ) -> CoordinationResult:
        """Merge and schedule one queued report on the rover worker."""
        coordinator = self.exploration_coordinator
        if coordinator is None or not self.rovers or self.rovers[0] is None:
            return CoordinationResult(arrived=False)
        coordinator.update_rover_position(self.get_check_in_position())
        arrived = self.terrain_sharing.check_in_with_rover(drone_id, 0)
        if not arrived:
            return CoordinationResult(arrived=False)
        self._prune_exploration_waiting_contacts()
        rover_slam = self.rovers[0].slam_map.snapshot(point_limit=0)
        self._refresh_highway_if_due(rover_slam)
        coordinator.update_highway_snapshot(self.highway.snapshot)
        # These are rover-local cumulative knowledge counters sampled after a
        # verified physical exchange. Their differences include any rover
        # observations or physical proximity shares since the last sample;
        # the display-only mission floor map is not coordinator knowledge.
        rover_progress = self.rovers[0].slam_map.progress_snapshot()
        rover_terrain_known_floor_cells = int(np.count_nonzero(
            self.rovers[0].terrain_knowledge.known_mask()
        ))
        battery = 100.0
        if 0 <= int(drone_id) < len(self.drones):
            battery = float(self.drones[int(drone_id)].snapshot().battery)
        result = coordinator.check_in(
            drone_id,
            rover_slam,
            report=report,
            energy_state=EnergyState(
                remaining_energy=battery,
                capacity=100.0,
                unlimited=True,
            ),
        )
        self._last_rover_knowledge_version = int(rover_slam.version)
        if (
            self.settings.highway.mode in {"observe", "active"}
            and self.highway.snapshot is not None
        ):
            result = replace(
                result,
                highway_snapshot=self.highway.snapshot,
            )
        coordination_snapshot = coordinator.snapshot()
        planning = result.batch_planning_summary
        if planning is not None:
            self.runtime_trace.record(
                "rover_focused_frontier_batch_planning_completed",
                sim_time=self.simulation_time(),
                planning_id=planning.planning_id,
                mode=planning.mode,
                elapsed_ms=planning.elapsed_ms,
                route_queries=planning.route_queries,
                route_cache_hits=planning.route_cache_hits,
                candidate_count=planning.candidate_count,
                planned_batch_count=planning.planned_batch_count,
                status=planning.status,
                highway_version=planning.highway_version,
                registry_revision=coordination_snapshot.revision,
            )
        for evaluation in result.batch_evaluations:
            self.runtime_trace.record(
                "rover_focused_frontier_batch_evaluated",
                sim_time=self.simulation_time(),
                evaluation_id=evaluation.evaluation_id,
                mode=evaluation.mode,
                focused_revision=coordination_snapshot.revision,
                drone_id=evaluation.drone_id,
                seed_task_id=evaluation.seed_task_id,
                candidate_task_id=evaluation.candidate_task_id,
                member_task_ids=evaluation.member_task_ids,
                separate_cost=evaluation.separate_cost,
                combined_cost=evaluation.combined_cost,
                avoided_round_trip=evaluation.avoided_round_trip,
                detour_cost=evaluation.detour_cost,
                accepted=evaluation.accepted,
                reason=evaluation.reason,
            )
        if (
            coordination_snapshot.focused_endgame
            != self._last_focused_endgame_trace_state
        ):
            self._last_focused_endgame_trace_state = (
                coordination_snapshot.focused_endgame
            )
            active_components = tuple(
                component
                for component in coordination_snapshot.components
                if component.state.value == "active"
            )
            self.runtime_trace.record(
                "rover_focused_endgame_changed",
                sim_time=self.simulation_time(),
                active=coordination_snapshot.focused_endgame,
                registry_revision=coordination_snapshot.revision,
                active_component_count=len(active_components),
                active_modes={
                    mode: sum(
                        component.exploration_mode.value == mode
                        for component in active_components
                    )
                    for mode in ("focused", "wall_follow", "sweep")
                },
            )
        self.runtime_trace.record(
            "drone_component_check_in",
            sim_time=self.simulation_time(),
            drone_id=int(drone_id),
            report_id=None if report is None else report.report_id,
            directive_id=None if report is None else report.directive_id,
            directive_kind=None if report is None else report.kind.value,
            claim_token=None if report is None else report.claim_token,
            claim_tokens=(
                () if report is None
                else tuple(
                    member.claim_token
                    for member in report.batch_member_reports
                )
            ),
            report_accepted=result.report_accepted,
            waiting=result.waiting,
            mission_exhausted=result.mission_exhausted,
            coordination_phase=coordination_snapshot.phase.value,
            rover_slam_version=rover_slam.version,
            rover_slam_newly_known_cells=rover_progress.newly_known_cells,
            rover_terrain_known_floor_cells=rover_terrain_known_floor_cells,
        )
        if report is not None and result.report_accepted:
            if report.kind == DirectiveKind.COMPONENT_BATCH:
                self.runtime_trace.record(
                    "rover_focused_frontier_batch_report_accepted",
                    sim_time=self.simulation_time(),
                    drone_id=int(drone_id),
                    report_id=report.report_id,
                    directive_id=report.directive_id,
                    lease_id=report.lease_id,
                    member_statuses=result.batch_member_statuses,
                    member_task_ids=tuple(
                        member.task_id
                        for member in report.batch_member_reports
                    ),
                    claim_tokens=tuple(
                        member.claim_token
                        for member in report.batch_member_reports
                    ),
                    member_dispositions=tuple(
                        member.disposition
                        for member in report.batch_member_reports
                    ),
                    provisional_observation_count=len(
                        report.provisional_observations
                    ),
                    provisional_newly_known_cells=sum(
                        item.sensor_newly_known_cells
                        for item in report.provisional_observations
                    ),
                    provisional_confidence_gain=sum(
                        item.sensor_confidence_gain
                        for item in report.provisional_observations
                    ),
                    provisional_retired_work_unit_ids=(
                        result.provisional_retired_work_unit_ids
                    ),
                    replayed=result.report_replayed,
                    outbound_distance=report.outbound_distance,
                    service_distance=report.service_distance,
                    return_distance=report.return_distance,
                )
            elif report.kind in {
                DirectiveKind.COMPONENT_TASK,
                DirectiveKind.COMPONENT_FOLLOW,
            }:
                self.runtime_trace.record(
                    (
                        "rover_task_completed"
                        if report.kind == DirectiveKind.COMPONENT_TASK
                        else "rover_branch_follow_completed"
                    ),
                    sim_time=self.simulation_time(),
                    drone_id=int(drone_id),
                    report_id=report.report_id,
                    task_id=report.task_id,
                    component_id=report.component_id,
                    claim_token=report.claim_token,
                    work_unit_outcomes=report.work_unit_outcomes,
                    causal_transition_count=len(report.causal_transitions),
                    suspended=report.suspension is not None,
                    suspension_reason=(
                        None if report.suspension is None
                        else report.suspension.reason
                    ),
                    outbound_distance=report.outbound_distance,
                    service_distance=report.service_distance,
                    return_distance=report.return_distance,
                    return_path_source=report.return_path_source,
                )
            elif report.kind == DirectiveKind.RADIAL_PROBE:
                self.runtime_trace.record(
                    "drone_probe_returned",
                    sim_time=self.simulation_time(),
                    drone_id=int(drone_id),
                    report_id=report.report_id,
                    completed_scan_headings=(
                        report.completed_scan_headings
                    ),
                    timed_out_scan_headings=(
                        report.timed_out_scan_headings
                    ),
                    sensor_newly_known_cells=(
                        report.sensor_newly_known_cells
                    ),
                    sensor_confidence_gain=report.sensor_confidence_gain,
                    outbound_distance=self._trace_path_distance(
                        report.outbound_actual_path
                    ),
                    return_distance=self._trace_path_distance(
                        report.return_actual_path
                    ),
                    return_path_source=report.return_path_source,
                )
            if (
                report.kind in {
                    DirectiveKind.ROVER_SCAN,
                    DirectiveKind.RADIAL_PROBE,
                }
                and result.reconcile_result is not None
            ):
                self.runtime_trace.record(
                    "rover_discovery_round_completed",
                    sim_time=self.simulation_time(),
                    round_id=report.round_id,
                    round_kind=report.kind.value,
                    registry_revision=result.reconcile_result.revision,
                    active_component_count=len(
                        result.reconcile_result.active_component_ids
                    ),
                    ready_work_unit_count=len(
                        result.reconcile_result.ready_work_unit_ids
                    ),
                    coordination_phase=coordination_snapshot.phase.value,
                )
        elif (
            report is not None
            and report.kind == DirectiveKind.COMPONENT_BATCH
        ):
            self.runtime_trace.record(
                "rover_focused_frontier_batch_report_rejected",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                report_id=report.report_id,
                directive_id=report.directive_id,
                lease_id=report.lease_id,
                member_task_ids=tuple(
                    member.task_id for member in report.batch_member_reports
                ),
                claim_tokens=tuple(
                    member.claim_token
                    for member in report.batch_member_reports
                ),
                reason="structural_validation",
                replayed=False,
            )
        if result.reconcile_result is not None:
            self._record_frontier_reconcile(
                result.reconcile_result, rover_slam, coordination_snapshot,
            )
        return result

    def _record_frontier_reconcile(
        self, reconcile, rover_slam, snapshot, *, refresh_source="physical_check_in",
    ) -> None:
        """Trace both report-driven and proximity-driven registry refreshes."""
        coordinator = self.exploration_coordinator
        for unit_id in reconcile.low_gain_deferred_work_unit_ids:
            unit = coordinator.registry.work_units[unit_id]
            self.runtime_trace.record(
                "rover_frontier_low_gain_deferred",
                sim_time=self.simulation_time(),
                component_id=unit.component_id,
                work_unit_id=unit_id,
                anchor=unit.anchor_position,
                revision=reconcile.revision,
            )
        self.runtime_trace.record(
            "rover_frontier_registry_reconciled",
            sim_time=self.simulation_time(),
            revision=reconcile.revision,
            elapsed_ms=reconcile.elapsed_ms,
            rover_slam_version=rover_slam.version,
            refresh_source=refresh_source,
            active_component_ids=reconcile.active_component_ids,
            active_component_count=len(reconcile.active_component_ids),
            ready_work_unit_ids=reconcile.ready_work_unit_ids,
            ready_work_unit_count=len(reconcile.ready_work_unit_ids),
            dormant_component_count=sum(
                component.state.value == "dormant"
                for component in snapshot.components
            ),
            resolved_component_count=sum(
                component.state.value == "resolved"
                for component in snapshot.components
            ),
            exploration_modes={
                mode: sum(
                    component.exploration_mode.value == mode
                    for component in snapshot.components
                    if component.state.value == "active"
                )
                for mode in ("focused", "wall_follow", "sweep")
            },
            focused_endgame=snapshot.focused_endgame,
        )
        for transition in reconcile.transitions:
            self.runtime_trace.record(
                "rover_frontier_lineage_changed",
                sim_time=self.simulation_time(),
                revision=reconcile.revision,
                transition_kind=transition.kind.value,
                parent_ids=transition.parent_ids,
                child_ids=transition.child_ids,
                evidence=transition.evidence,
            )

    def _process_exploration_check_ins(self) -> None:
        """Drain queued component reports without involving drone workers."""
        while True:
            try:
                pending = self._exploration_check_in_queue.get_nowait()
            except Empty:
                return
            started_at = time.perf_counter()
            pending.result = self._perform_exploration_check_in(
                pending.drone_id,
                pending.report,
            )
            pending.ready.set()
            self.runtime_trace.record(
                "rover_component_check_in_processed",
                sim_time=self.simulation_time(),
                drone_id=pending.drone_id,
                queue_wait_ms=(started_at - pending.enqueued_at) * 1000.0,
                processing_ms=(time.perf_counter() - started_at) * 1000.0,
                report_accepted=pending.result.report_accepted,
            )

    def exploration_energy_return(
        self,
        drone_id: int,
        route_home_cost: float,
        next_action_cost: float,
        safety_reserve: float,
    ):
        """Evaluate a movement checkpoint through the coordinator policy."""
        coordinator = self.exploration_coordinator
        if coordinator is None:
            return None
        battery = 100.0
        if 0 <= int(drone_id) < len(self.drones):
            battery = float(self.drones[int(drone_id)].snapshot().battery)
        return coordinator.evaluate_return(
            drone_id,
            EnergyState(
                remaining_energy=battery,
                capacity=100.0,
                unlimited=True,
            ),
            route_home_cost=route_home_cost,
            next_action_cost=next_action_cost,
            safety_reserve=safety_reserve,
        )

    def exploration_assignment(self, drone_id: int) -> CoordinationResult:
        """Deliver one signalled component, scan, probe, or home directive."""
        coordinator = self.exploration_coordinator
        if coordinator is None:
            return CoordinationResult(arrived=False)
        with self._exploration_check_in_lock:
            normalized_id = int(drone_id)
            if normalized_id not in self._exploration_docked:
                if normalized_id not in coordinator.snapshot().waiting_drone_ids:
                    self._exploration_check_ins.pop(normalized_id, None)
                return CoordinationResult(arrived=False)
            if normalized_id not in coordinator.snapshot().waiting_drone_ids:
                # Its earlier waiting check-in was invalidated by lost contact.
                # A fresh rover-worker check-in must restore that membership.
                self._exploration_check_ins.pop(normalized_id, None)
                return CoordinationResult(arrived=False)
            pending = self._exploration_check_ins.get(normalized_id)
            if pending is not None and not pending.ready.is_set():
                return CoordinationResult(
                    arrived=True,
                    waiting=True,
                    directive_ready=pending.ready,
                )
            if pending is not None:
                completed = self._exploration_check_ins.pop(normalized_id)
                processed = completed.result or CoordinationResult(arrived=False)
                if not processed.arrived or (
                    completed.report is not None
                    and not processed.report_accepted
                ):
                    return processed
                if (
                    processed.directive_ready is None
                    or not processed.directive_ready.is_set()
                ):
                    return processed
                claimed = coordinator.claim_directive(drone_id)
                result = replace(
                    claimed,
                    report_accepted=processed.report_accepted,
                    reconcile_result=processed.reconcile_result,
                    published_directives=processed.published_directives,
                    batch_evaluations=processed.batch_evaluations,
                    batch_member_statuses=processed.batch_member_statuses,
                    provisional_retired_work_unit_ids=(
                        processed.provisional_retired_work_unit_ids
                    ),
                    report_replayed=processed.report_replayed,
                    batch_planning_summary=(
                        processed.batch_planning_summary
                    ),
                    highway_snapshot=processed.highway_snapshot,
                )
            else:
                result = coordinator.claim_directive(drone_id)
            departure_kinds = {
                DirectiveKind.COMPONENT_TASK,
                DirectiveKind.COMPONENT_BATCH,
                DirectiveKind.COMPONENT_FOLLOW,
                DirectiveKind.RADIAL_PROBE,
            }
            release_kinds = departure_kinds | {DirectiveKind.ROVER_SCAN}
            if (
                result.directive is not None
                and result.directive.kind in release_kinds
            ):
                if result.directive.kind in departure_kinds:
                    self.terrain_sharing.share_on_departure(drone_id, 0)
                self._undock_exploration_drone_locked(
                    normalized_id,
                    reason="directive_assigned",
                    directive_id=result.directive.directive_id,
                    directive_kind=result.directive.kind.value,
                )
        directive = result.directive
        if directive is None:
            return result
        if (
            directive.round_id is not None
            and directive.round_id not in self._traced_discovery_rounds
        ):
            self._traced_discovery_rounds.add(directive.round_id)
            round_state = coordinator.snapshot().discovery_round
            self.runtime_trace.record(
                "rover_discovery_round_started",
                sim_time=self.simulation_time(),
                round_id=directive.round_id,
                round_kind=directive.kind.value,
                participant_ids=(
                    () if round_state is None
                    else round_state.participant_ids
                ),
                scan_plans=(
                    {} if round_state is None else round_state.scan_plans
                ),
                probe_targets=(
                    {} if round_state is None
                    else round_state.probe_targets
                ),
                ring=0 if round_state is None else round_state.ring,
            )
        if directive.kind == DirectiveKind.COMPONENT_TASK:
            self.runtime_trace.record(
                "rover_task_claimed",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                directive_id=directive.directive_id,
                task_id=directive.task.task_id,
                component_id=directive.task.component_id,
                component_revision=directive.task.component_revision,
                parent_task_id=directive.task.parent_task_id,
                dfs_depth=directive.task.depth,
                work_unit_ids=directive.claim.work_unit_ids,
                claim_token=directive.claim.token,
                entry=directive.task.preferred_entry,
                estimated_effort=directive.task.estimated_effort,
                assignment_policy=directive.assignment_policy,
                estimated_outbound_cost=(
                    directive.estimated_outbound_cost
                ),
                estimated_round_trip_cost=(
                    directive.estimated_round_trip_cost
                ),
                route_distance=self._trace_path_distance(
                    directive.outbound_route
                ),
            )
            if directive.assignment_policy == "focused_endgame_round_trip":
                self.runtime_trace.record(
                    "rover_focused_endgame_assignment",
                    sim_time=self.simulation_time(),
                    drone_id=int(drone_id),
                    directive_id=directive.directive_id,
                    task_id=directive.task.task_id,
                    component_id=directive.task.component_id,
                    entry=directive.task.preferred_entry,
                    estimated_effort=directive.task.estimated_effort,
                    estimated_outbound_cost=(
                        directive.estimated_outbound_cost
                    ),
                    estimated_round_trip_cost=(
                        directive.estimated_round_trip_cost
                    ),
                )
        elif directive.kind == DirectiveKind.COMPONENT_BATCH:
            lease = directive.spatial_lease
            self.runtime_trace.record(
                "rover_focused_frontier_batch_issued",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                directive_id=directive.directive_id,
                lease_id=None if lease is None else lease.lease_id,
                member_task_ids=tuple(
                    member.task.task_id for member in directive.batch_members
                ),
                component_ids=tuple(
                    member.task.component_id for member in directive.batch_members
                ),
                work_unit_ids=tuple(
                    unit_id
                    for member in directive.batch_members
                    for unit_id in member.claim.work_unit_ids
                ),
                claim_tokens=tuple(
                    member.claim.token for member in directive.batch_members
                ),
                lease_span_count=(
                    0 if lease is None else len(lease.cell_spans)
                ),
                lease_cell_count=0 if lease is None else lease.cell_count,
                estimated_separate_cost=directive.estimated_separate_cost,
                estimated_combined_cost=directive.estimated_combined_cost,
                estimated_avoided_round_trip=(
                    directive.estimated_avoided_round_trip
                ),
                estimated_detour_cost=directive.estimated_detour_cost,
                maximum_total_components=directive.maximum_total_components,
                maximum_detour_distance=directive.maximum_detour_distance,
                maximum_service_seconds=directive.maximum_service_seconds,
                maximum_total_dfs_nodes=directive.maximum_total_dfs_nodes,
            )
        elif directive.kind == DirectiveKind.COMPONENT_FOLLOW:
            self.runtime_trace.record(
                "rover_branch_follow_assigned",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                directive_id=directive.directive_id,
                task_id=directive.task.task_id,
                component_id=directive.task.component_id,
                leader_drone_id=directive.leader_drone_id,
                branch_index=directive.follow_branch_index,
            )
        elif directive.kind == DirectiveKind.RADIAL_PROBE:
            self.runtime_trace.record(
                "drone_probe_started",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                directive_id=directive.directive_id,
                round_id=directive.round_id,
                probe_target=directive.probe_target,
                scan_headings=directive.scan_headings,
                route_distance=self._trace_path_distance(
                    directive.outbound_route
                ),
            )
        elif directive.kind == DirectiveKind.HOME:
            snapshot = coordinator.snapshot()
            self.runtime_trace.record(
                "exploration_homing_started",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                reason=directive.reason,
            )
            self.runtime_trace.record(
                "rover_exploration_quiescence_evaluated",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                reason=directive.reason,
                mission_exhausted=snapshot.mission_exhausted,
                coordination_phase=snapshot.phase.value,
                ready_task_count=sum(
                    task.state.value in {"ready", "suspended", "blocked"}
                    for task in snapshot.tasks
                ),
                live_claim_count=len(snapshot.claims),
            )
        return result

    @staticmethod
    def _trace_path_distance(path: tuple[Tuple[int, int], ...]) -> float:
        return sum(
            float(np.hypot(
                current[0] - previous[0],
                current[1] - previous[1],
            ))
            for previous, current in zip(path, path[1:])
        )

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
        if result.published_assignments:
            self.runtime_trace.record(
                "rover_sector_epoch_published",
                sim_time=self.simulation_time(),
                generation=result.generation,
                rover_slam_version=rover_slam.version,
                active_drone_ids=tuple(
                    item.owner_drone_id for item in result.published_assignments
                    if not item.standby
                ),
                standby_drone_ids=tuple(
                    item.owner_drone_id for item in result.published_assignments
                    if item.standby
                ),
                assignments=tuple({
                    "drone_id": item.owner_drone_id,
                    "sector_id": item.sector_id,
                    "standby": item.standby,
                    "frontier_cells": item.frontier_cells,
                    "frontier_component_ids": tuple(
                        component.component_id for component in item.frontier_components
                    ),
                    "estimated_effort": item.estimated_effort,
                    "effort_breakdown": item.effort_breakdown,
                    "scope_pixels": (
                        None if item.exploration_mask is None
                        else int(np.count_nonzero(item.exploration_mask))
                    ),
                } for item in result.published_assignments),
            )
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
        if result.assignment.standby:
            return result
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


    def compute_rover_path(
        self,
        rover_id: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Plan only through traversable cells received by this rover."""
        normalized_id = int(rover_id)
        if not 0 <= normalized_id < len(self.rovers):
            return []
        rover = self.rovers[normalized_id]
        slam = rover.slam_map.snapshot(point_limit=0)
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        known_free = (
            (occupancy == FREE)
            & (
                confidence
                >= self.settings.frontier.confidence_threshold
            )
        )
        icon = getattr(rover, "icon", None)
        body_radius = max(
            1.0,
            float(max(icon.get_size())) / 2.0 if icon is not None else 1.0,
        )
        normalized_start = (int(start[0]), int(start[1]))
        normalized_goal = (int(goal[0]), int(goal[1]))
        normalized_radius = round(body_radius, 3)
        failed_key = (
            normalized_id,
            normalized_start,
            normalized_goal,
            int(slam.version),
            normalized_radius,
        )
        self._rover_failed_route_cache = {
            key for key in self._rover_failed_route_cache
            if key[0] != normalized_id
            or (
                key[1] == normalized_start
                and key[3] == int(slam.version)
                and key[4] == normalized_radius
            )
        }
        if failed_key in self._rover_failed_route_cache:
            self.runtime_trace.record(
                "rover_route_negative_cache_hit",
                sim_time=self.simulation_time(),
                rover_id=normalized_id,
                start=normalized_start,
                goal=normalized_goal,
                rover_slam_version=slam.version,
                body_radius=body_radius,
            )
            return []
        clearance = cv2.distanceTransform(
            np.pad(known_free.astype(np.uint8), 1),
            cv2.DIST_L2,
            5,
        )[1:-1, 1:-1]
        rover_safe = known_free & (clearance >= body_radius)
        start_x, start_y = int(start[0]), int(start[1])
        goal_x, goal_y = int(goal[0]), int(goal[1])
        height, width = known_free.shape
        if not (
            0 <= start_x < width
            and 0 <= start_y < height
            and 0 <= goal_x < width
            and 0 <= goal_y < height
            and rover_safe[start_y, start_x]
            and rover_safe[goal_y, goal_x]
        ):
            return []
        traversability = np.ones(rover_safe.shape, dtype=np.uint8)
        traversability[rover_safe] = 0
        terrain = rover.terrain_knowledge.snapshot()
        started_at = time.perf_counter()
        path = self.pathfinding.compute_weighted_path(
            terrain.roughness,
            terrain.confidence,
            start,
            goal,
            traversability_map=traversability,
        )
        self.runtime_trace.record(
            "rover_route_planned",
            sim_time=self.simulation_time(),
            rover_id=normalized_id,
            start=start,
            goal=goal,
            path_length=len(path),
            elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
            rover_slam_version=slam.version,
            local_slam_only=True,
            body_radius=body_radius,
            goal_clearance=float(clearance[goal_y, goal_x]),
        )
        if not path:
            self._rover_failed_route_cache.add(failed_key)
            self.runtime_trace.record(
                "rover_route_negative_cached",
                sim_time=self.simulation_time(),
                rover_id=normalized_id,
                start=normalized_start,
                goal=normalized_goal,
                rover_slam_version=slam.version,
                body_radius=body_radius,
            )
        return path


    def rover_thread(self, rover_id: int) -> None:
        """Drive rover movement using the terrain-aware weighted planner."""
        label = ("rover", rover_id)
        if not self.pause_coordinator.register_current_worker(label):
            return
        try:
            while not self.mission_event.is_set():
                if not self.pause_checkpoint():
                    break
                if rover_id == 0:
                    self._poll_highway()
                    self._process_exploration_check_ins()
                    with self._exploration_check_in_lock:
                        self.rovers[rover_id].move()
                        self._move_docked_drones_locked(rover_id)
                else:
                    self.rovers[rover_id].move()
                coordinator = self.exploration_coordinator
                if coordinator is not None and rover_id == 0:
                    coordinator.update_rover_position(
                        tuple(self.rovers[rover_id].pos)
                    )
                    self.terrain_sharing.share_with_rovers()
                    self._refresh_rover_knowledge_if_changed()
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
        if self.exploration_complete:
            return
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
        """Update local sensing and display-only floor exploration progress."""
        for drone in self.drones:
            drone.update_sensors()
        self._update_wall_mapping_progress()

    def _update_wall_mapping_progress(self) -> WallMappingSnapshot:
        """Publish wall diagnostics and discovered-floor UI progress.

        Component quiescence is the only exploration completion trigger.  The
        floor ratio deliberately remains observational so a complete mission
        can expose its small unexplored remainder.
        """
        was_complete = self.wall_mapping_progress.complete
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
        self.floor_exploration_ratio = self.terrain_knowledge.explored_ratio()
        if self.runtime_trace is not None:
            if progress.complete and not was_complete:
                self.runtime_trace.record(
                    "team_wall_mapping_complete",
                    mapped_wall_pixels=progress.mapped_wall_pixels,
                    total_wall_pixels=progress.total_wall_pixels,
                    missing_wall_pixels=missing_wall_pixels,
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
                100.0
                if self.floor_exploration_ratio >= 1.0
                else min(
                    99.99,
                    round(self.floor_exploration_ratio * 100.0, 2),
                )
            )
            self.control_center.set_explored_percent(displayed_percent)
            self.last_explored_update = now
        return progress
