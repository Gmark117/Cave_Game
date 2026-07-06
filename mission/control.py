"""Mission orchestration and runtime resource setup.

Constructing `MissionControl` prepares mission state only. Calling `run()`
initializes the mission window, agents, pathfinding resources, and worker
threads before entering the main loop.
"""

import random as rand
import threading
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
from mission.debug_info import MissionDebugInfo
from mission.frame_timing import FrameProfiler
from mission.objectives import build_mission_objective
from mission.pause_control import PauseCoordinator, SimulationClock
from contracts import (
    MissionDebugDependencies,
    MissionRendererDependencies,
    RoverTargetDependencies,
    SlamViewDependencies,
    TerrainFusionDependencies,
    TerrainSharingDependencies,
)
from navigation.pathfinding import PathfindingService
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
        
        self.delay = 1/15 # Set a delay for frame updates


        self.completed = False # Track whether the mission is completed

        # Initialize presentation adapter for UI state and map rendering
        self.presentation = PresentationAdapter(self.map_w, self.map_h)
        self.slam_renderer = SlamRenderer(self.map_w, self.map_h)
        self.last_explored_update = 0.0
        self.explored_update_interval = 0.5
        # Dependency bundles keep services decoupled from the large
        # MissionControl object while still giving them the callbacks they need.
        self.terrain_fusion_dependencies = TerrainFusionDependencies(
            terrain_knowledge=self.terrain_knowledge,
            get_control_center=lambda: self.control_center,
            presentation=self.presentation,
            simulation_time=self.simulation_time,
            explored_update_interval=self.explored_update_interval,
            last_explored_update=self.last_explored_update,
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
        """Compute a drone path through the pathfinding service."""
        return self.pathfinding.compute_path(start, goal)


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
        """Update every drone's SLAM and terrain knowledge."""
        for drone in self.drones:
            drone.update_sensors()
