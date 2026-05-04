"""Mission orchestration: spawn agents, manage threads and pathfinding pool.

`MissionControl` sets up shared memory for worker pathfinders, creates
the drone/rover agents, and runs the main loop that updates and draws
the simulation. Type hints clarify public method contracts.
"""

import math
import os
import random as rand
import threading
import logging
from typing import List, Tuple, Any, Optional

import numpy as np
import pygame
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

from asset_config.helpers import wall_hit
from asset_config.media import Images
from AgentFactory import AgentFactory
from ControlCenter import ControlCenter
from PresentationAdapter import PresentationAdapter
import AStarPathfinder
from MissionControlTerrain import MissionControlTerrainMixin
from MissionControlLifecycle import MissionControlLifecycleMixin


logger = logging.getLogger(__name__)


class MissionControl(MissionControlTerrainMixin, MissionControlLifecycleMixin):
    """Orchestrates the simulation mission.

    Responsible for creating agents (drones, rovers), preparing a
    shared-memory map for worker pathfinders, running per-drone threads,
    and driving the main rendering/update loop until the mission ends.
    """
    def __init__(self, game: Any) -> None:
        """Initialize mission control and start the mission loop.

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
        self.cave_png     = pygame.image.load(Images.CAVE_MAP.value).convert_alpha() # Load cave map image
        self.floor_cells = max(1, int(np.count_nonzero(np.asarray(self.map_matrix) == 0)))
        self.known_roughness = np.full(np.asarray(self.map_matrix).shape, -1.0, dtype=np.float32)
        self.terrain_confidence = np.zeros(np.asarray(self.map_matrix).shape, dtype=np.float32)
        self.terrain_lock = threading.Lock()
        self.floor_mask = (np.asarray(self.map_matrix) == 0)
        self.rover_assignment_lock = threading.Lock()
        self.rover_assignments = {}
        self.completed_rover_targets = set()
        self.min_share_new_info_ratio = 0.04
        self.min_share_overlap_diff_ratio = 0.18
        self.min_share_roughness_delta = 0.12
        self.share_compare_stride = 8
        self.pair_share_cooldown = 1.2
        self.last_pair_share: dict[Tuple[int, int], float] = {}
        # Temporary switch: keep rovers stationary
        self.rover_motion_enabled = False

        # Create shared-memory copy of the map for worker processes
        self.map_shm: Optional[shared_memory.SharedMemory] = None
        self.map_shape: Optional[Tuple[int, int]] = None
        try:
            arr = np.array(self.map_matrix, dtype=np.uint8)
            self.map_shape = arr.shape
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shm_arr[:] = arr[:]
            # SharedMemory object used by worker processes (A* tasks)
            self.map_shm = shm
        except (OSError, ValueError, BufferError) as exc:
            # If shared memory cannot be created, worker pathfinding will be disabled
            logger.warning("Shared memory setup failed; process-pool pathfinding disabled: %s", exc)
            self.map_shm = None
            self.map_shape = None
        
        self.delay = 1/15 # Set a delay for frame updates

        # Load cave wall images
        self.cave_walls_png = pygame.image.load(Images.CAVE_WALLS.value).convert_alpha()

        # Initialize mission settings (0 for exploration, 1 for search & rescue)
        self.mission   = self.settings.mission
        self.completed = False # Track whether the mission is completed

        # Initialise control center for displaying mission status
        self.control_center = ControlCenter(game, self.settings.num_drones)

        # Maximise the game window
        self.game.display = self.game.to_maximised()

        # Initialize stop button (top-left corner)
        self.stop_button_rect = pygame.Rect(10, 10, 80, 40)

        # Initialize presentation adapter for UI state and heatmap rendering
        self.presentation = PresentationAdapter(self.map_w, self.map_h)
        self.last_explored_update = 0.0
        self.explored_update_interval = 0.5
        
        # Set the starting position for drones
        self.start_point = None
        self.set_start_point()

        # Build the drones and the rovers
        AgentFactory.build_drones(self)
        AgentFactory.build_rovers(self)

        # Print them on the map
        self.draw()

        # Show the map and the robots at step 0 for 1 second
        pygame.display.update()
        pygame.time.wait(1000)

        # Create an event to stop the threads when the mission is complete
        self.mission_event = threading.Event()
        # Clock used to control main loop FPS
        self.clock = pygame.time.Clock()

        # Create process pool for pathfinding and start mission
        cpu = (os.cpu_count() or 1)
        # Reserve one CPU for the main process when possible
        if cpu > 1:
            max_workers = min(self.settings.num_drones, cpu - 1)
        else:
            max_workers = 1
        self.pool = ProcessPoolExecutor(max_workers=max_workers)
        # Semaphore to bound concurrent submissions to the pool (prevents over-submission)
        self.pool_sem = threading.Semaphore(max_workers)

        self.start_mission()


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
            self.drones[drone_id].move()  # Move the drone
            
            # Periodically share terrain data with nearby drones
            self._share_terrain_with_nearby_drones(drone_id)

            # Control the speed of movement
            # Use Event.wait so the sleep is interruptible when `mission_event` is set
            self.mission_event.wait(self.delay)


    def compute_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Submit pathfinding job to the process pool and return the path.

        If shared memory wasn't created at startup this returns an empty
        list. The method blocks until the worker returns the path.
        """
        if self.map_shm is None or self.map_shape is None:
            return []
        acquired = False
        try:
            # Block submission when the pool is saturated to avoid queue buildup
            self.pool_sem.acquire()
            acquired = True
            fut = self.pool.submit(AStarPathfinder.compute_path, self.map_shm.name, self.map_shape, start, goal)
            result = fut.result()
            return result
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Pathfinding pool request failed for %s -> %s: %s", start, goal, exc)
            return []
        finally:
            if acquired:
                self.pool_sem.release()


    def compute_rover_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute a terrain-aware path for rovers without affecting drone A*."""
        with self.terrain_lock:
            known_roughness = self.known_roughness.copy()
            terrain_confidence = self.terrain_confidence.copy()
        cave_map = np.asarray(self.map_matrix, dtype=np.uint8)
        return AStarPathfinder.compute_weighted_path(cave_map, known_roughness, terrain_confidence, start, goal)


    def toggle_terrain_heatmap(self) -> None:
        """Toggle terrain heatmap visibility via presentation adapter."""
        self.presentation.toggle_terrain_heatmap()

    def toggle_drone_heatmap(self, drone_id: int) -> None:
        """Toggle per-drone heatmap visibility via presentation adapter."""
        self.presentation.toggle_drone_heatmap(drone_id)

    def rover_thread(self, rover_id: int) -> None:
        """Drive rover movement using the terrain-aware weighted planner."""
        while not self.mission_event.is_set():
            self.rovers[rover_id].move()
            self.mission_event.wait(self.delay)

# =============================================================================
# Graph class for path validation and obstacle checking
# =============================================================================

    def draw_cave(self) -> None:
        """Draw the base cave map (underlays and floor)."""
        self.game.window.blit(self.cave_png, (0, 0))

    
    def draw_walls(self) -> None:
        """Draw the cave wall overlay (occludes floor but not icons)."""
        self.game.window.blit(self.cave_walls_png, (0, 0))

    def draw_stop_button(self) -> None:
        """Draw the stop button in the top-left corner."""
        from asset_config.rendering import Colors, Fonts
        
        # Draw button background
        pygame.draw.rect(self.game.window, Colors.RED.value, self.stop_button_rect)
        pygame.draw.rect(self.game.window, Colors.WHITE.value, self.stop_button_rect, 2)
        
        # Draw button text
        font = pygame.font.Font(Fonts.SMALL.value, 24)
        text_surface = font.render("STOP", True, Colors.WHITE.value)
        text_rect = text_surface.get_rect(center=self.stop_button_rect.center)
        self.game.window.blit(text_surface, text_rect)

   
    def draw(self) -> None:
        """Render full scene in layered order.

        Layers: floor -> drone paths -> walls -> drone vision -> icons -> UI
        """
        # Base map
        self.draw_cave()
        self.draw_terrain_heatmap()
        
        # Per-drone overlays: draw explored paths (under vision)
        for drone in self.drones:
            drone.draw_path()
        for rover in self.rovers:
            rover.draw_path()

        # Draw cave walls once
        self.draw_walls()
        
        # Draw drone visions on top of paths and icons
        for drone in self.drones:
            drone.draw_vision()

        # Draw all icons (drones and rovers)
        for i, drone in enumerate(self.drones):
            drone.draw_icon()
            if i < len(self.rovers):
                self.rovers[i].draw_icon()

        # Control center UI
        self.control_center.draw_control_center(
            self.drones,
            self.rovers,
            self.presentation.show_terrain_heatmap,
            self.presentation.selected_drone_heatmap_id
        )

        # Draw stop button
        self.draw_stop_button()