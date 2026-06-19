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
from AgentFactory import AgentFactory
from ControlCenter import ControlCenter
from mapping.terrain_knowledge import TerrainKnowledge
from mission.frame_timing import FrameProfiler
from navigation.pathfinding import PathfindingService
from PresentationAdapter import PresentationAdapter
from SlamRenderer import SlamRenderer
from rendering.mission_renderer import MissionRenderer
from MissionControlTerrain import MissionControlTerrainMixin
from MissionControlLifecycle import MissionControlLifecycleMixin


class MissionControl(MissionControlTerrainMixin, MissionControlLifecycleMixin):
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
        # Set the seed from the settings
        rand.seed(game.sim_settings.seed)

        self.game         = game
        self.settings     = game.sim_settings 
        self.cartographer = game.cartographer
        self.map_matrix   = self.cartographer.bin_map # Get the binary map representation
        self.map_h, self.map_w = np.asarray(self.map_matrix).shape
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
        self.min_share_new_info_ratio = 0.04
        self.min_share_overlap_diff_ratio = 0.18
        self.min_share_roughness_delta = 0.12
        self.share_compare_stride = 8
        # Temporary switch: keep rovers stationary
        self.rover_motion_enabled = False

        # Runtime resources are initialized explicitly by run().
        self.pathfinding = PathfindingService(
            self.map_matrix,
            self.settings.num_drones,
        )
        self.mission_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.is_paused = False
        self.clock: Optional[pygame.time.Clock] = None
        self.drones = []
        self.rovers = []
        self.num_drones = self.settings.num_drones
        self.num_rovers = 0
        self.control_center: Optional[ControlCenter] = None
        self._runtime_initialized = False
        self._running = False
        self._has_run = False
        self.restart_requested = False
        self.frame_profiler = FrameProfiler()
        
        self.delay = 1/15 # Set a delay for frame updates


        # Initialize mission settings (0 for exploration, 1 for search & rescue)
        self.mission   = self.settings.mission
        self.completed = False # Track whether the mission is completed

        # Initialize presentation adapter for UI state and map rendering
        self.presentation = PresentationAdapter(self.map_w, self.map_h)
        self.slam_renderer = SlamRenderer(self.map_w, self.map_h)
        self._init_terrain_services()
        self.renderer = MissionRenderer(self)
        self.last_explored_update = 0.0
        self.explored_update_interval = 0.5
        
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
        self.control_center = ControlCenter(
            self.game,
            self.settings.num_drones,
        )

        AgentFactory.build_drones(self)
        AgentFactory.build_rovers(self)

        self.presentation.show_terrain_heatmap = False
        self.presentation.selected_drone_heatmap_id = None
        self.presentation.terrain_heatmap_dirty = True
        for drone in self.drones:
            drone.show_path = True
            drone.show_vision = True

        self.clock = pygame.time.Clock()
        self._setup_pathfinding_resources()
        self._runtime_initialized = True

        self.update_sensors()
        self.renderer.draw()
        pygame.display.update()
        pygame.time.wait(1000)

    def _setup_pathfinding_resources(self) -> None:
        """Compatibility wrapper for pathfinding resource initialization."""
        self.pathfinding.start()


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
        # Continue moving the drone until mission event is set or the drone completes its mission
        while not self.mission_event.is_set() and not self.drones[drone_id].mission_completed():
            if not self._wait_until_resumed():
                break
            self.drones[drone_id].move()  # Move the drone
            
            # Periodically share terrain data with nearby drones
            self._share_terrain_with_nearby_drones(drone_id)

            # Control the speed of movement
            # Use Event.wait so the sleep is interruptible when `mission_event` is set
            self.mission_event.wait(self.delay)


    def compute_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute a drone path through the pathfinding service."""
        return self.pathfinding.compute_path(start, goal)


    def compute_rover_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute a provisional rover path using mission terrain telemetry.

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


    def toggle_terrain_heatmap(self) -> None:
        """Toggle terrain heatmap visibility via presentation adapter."""
        self.presentation.toggle_terrain_heatmap()
        self._update_visibility_state()

    def toggle_drone_heatmap(self, drone_id: int) -> None:
        """Toggle per-drone heatmap visibility via presentation adapter."""
        if drone_id < 0 or drone_id >= len(self.drones):
            return
        self.presentation.toggle_drone_heatmap(drone_id)
        self._update_visibility_state()

    def rover_thread(self, rover_id: int) -> None:
        """Drive rover movement using the terrain-aware weighted planner."""
        while not self.mission_event.is_set():
            if not self._wait_until_resumed():
                break
            self.rovers[rover_id].move()
            self.mission_event.wait(self.delay)

    def _wait_until_resumed(self) -> bool:
        """Block an agent while paused and return False when shutting down."""
        while not self.mission_event.is_set():
            if self.pause_event.wait(0.05):
                return True
        return False

    def toggle_pause(self) -> None:
        """Toggle mission updates and agent movement between paused and running."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_event.clear()
            if self.control_center is not None:
                self.control_center.pause_timer()
            return

        self.pause_event.set()
        if self.control_center is not None:
            self.control_center.resume_timer()

    def update_sensors(self) -> None:
        """Update every drone's SLAM and terrain knowledge."""
        for drone in self.drones:
            drone.update_sensors()

    def draw_stop_button(self) -> None:
        """Compatibility wrapper for stop-button rendering."""
        self.renderer.draw_stop_button()

    def draw(self) -> None:
        """Compatibility wrapper for complete mission rendering."""
        self.renderer.draw()

    @property
    def stop_button_rect(self) -> pygame.Rect:
        """Legacy access to the renderer-owned stop-button hit area."""
        return self.renderer.stop_button_rect

    @property
    def restart_button_rect(self) -> pygame.Rect:
        """Expose the renderer-owned restart-button hit area."""
        return self.renderer.restart_button_rect

    @property
    def pause_button_rect(self) -> pygame.Rect:
        """Expose the renderer-owned pause/play-button hit area."""
        return self.renderer.pause_button_rect

    @property
    def map_shm(self) -> Any:
        """Legacy access to pathfinding shared memory."""
        return self.pathfinding.map_shm

    @property
    def map_shape(self) -> Optional[Tuple[int, int]]:
        """Legacy access to the shared pathfinding map shape."""
        return self.pathfinding.map_shape

    @property
    def pool(self) -> Any:
        """Legacy access to the pathfinding worker pool."""
        return self.pathfinding.pool

    @property
    def pool_sem(self) -> Any:
        """Legacy access to the pathfinding submission semaphore."""
        return self.pathfinding.pool_sem

    @property
    def known_roughness(self) -> np.ndarray:
        """Compatibility access to mission terrain roughness."""
        return self.terrain_knowledge.roughness

    @property
    def terrain_confidence(self) -> np.ndarray:
        """Compatibility access to mission terrain confidence."""
        return self.terrain_knowledge.confidence

    @property
    def terrain_lock(self) -> Any:
        """Compatibility access to the mission terrain lock."""
        return self.terrain_knowledge.lock

    @property
    def floor_mask(self) -> np.ndarray:
        """Compatibility access to the mission cave floor mask."""
        return self.terrain_knowledge.floor_mask

    @property
    def floor_cells(self) -> int:
        """Compatibility access to the mission floor-cell count."""
        return max(1, self.terrain_knowledge.floor_cells)
