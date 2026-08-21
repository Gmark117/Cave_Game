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
                "team_wall_mapping_or_local_border_exhaustion_and_home"
            ),
            exploration_progress="exposed_wall_slam_coverage",
            terrain_role="rover_navigation_only",
            frontier_policy=(
                "cached_global_wall_then_region_guidance_with_astar_escape"
            ),
            frontier_stride=self.settings.frontier.stride,
            frontier_minimum_cluster_cells=(
                self.settings.frontier.minimum_cluster_cells
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
            stagnation_distance=(
                self.settings.exploration.stagnation_distance
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
        self.slam_renderer = SlamRenderer(self.map_w, self.map_h)
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
            )
        )
        
        # Set the starting position for drones
        self.start_point = None
        self.set_start_point()

    def _initialize_runtime(self) -> None:
        """Create window, agents, pathfinding resources, and first frame."""
        if self._runtime_initialized:
            return

        self.completed = False
        self.mission_event.clear()
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
        """Publish wall coverage and start homing at exact completion."""
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
        if progress.complete and not was_complete:
            for drone in self.drones:
                drone.runtime_state.start_returning_home()
            if self.runtime_trace is not None:
                self.runtime_trace.record(
                    "team_wall_mapping_complete",
                    mapped_wall_pixels=progress.mapped_wall_pixels,
                    total_wall_pixels=progress.total_wall_pixels,
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
