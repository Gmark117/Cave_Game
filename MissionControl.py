"""Mission orchestration: spawn agents, manage threads and pathfinding pool.

`MissionControl` sets up shared memory for worker pathfinders, creates
the drone/rover agents, and runs the main loop that updates and draws
the simulation. Type hints clarify public method contracts.
"""

import math
import os
import random as rand
import threading
from typing import List, Tuple, Any, Optional

import numpy as np
import pygame
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import Assets
from ControlCenter import ControlCenter
from Drone import Drone
from Rover import Rover
import AStarPathfinder


class MissionControl:
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
        self.terrain_roughness = np.array(getattr(self.cartographer, 'terrain_roughness', np.zeros_like(self.map_matrix)), dtype=np.float32)
        self.cave_png     = pygame.image.load(Assets.Images.CAVE_MAP.value).convert_alpha() # Load cave map image
        self.floor_cells = max(1, int(np.count_nonzero(np.asarray(self.map_matrix) == 0)))
        self.known_roughness = np.full(np.asarray(self.map_matrix).shape, -1.0, dtype=np.float32)
        self.terrain_confidence = np.zeros(np.asarray(self.map_matrix).shape, dtype=np.float32)
        self.terrain_lock = threading.Lock()
        self.rover_assignment_lock = threading.Lock()
        self.rover_assignments = {}
        self.completed_rover_targets = set()
        # Temporary switch: keep rovers stationary
        self.rover_motion_enabled = False

        # Create shared-memory copy of the map for worker processes
        try:
            arr = np.array(self.map_matrix, dtype=np.uint8)
            self.map_shape = arr.shape
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shm_arr[:] = arr[:]
            # SharedMemory object used by worker processes (A* tasks)
            self.map_shm: Optional[shared_memory.SharedMemory] = shm
        except Exception:
            # If shared memory cannot be created, worker pathfinding will be disabled
            self.map_shm = None
            self.map_shape = None
        
        self.delay = 1/15 # Set a delay for frame updates

        # Load cave wall images
        self.cave_walls_png = pygame.image.load(Assets.Images.CAVE_WALLS.value).convert_alpha()

        # Initialize mission settings (0 for exploration, 1 for search & rescue)
        self.mission   = self.settings.mission
        self.completed = False # Track whether the mission is completed

        # Initialise control center for displaying mission status
        self.control_center = ControlCenter(game, self.settings.num_drones)

        # Maximise the game window
        self.game.display = self.game.to_maximised()

        self.show_terrain_heatmap = False
        self.terrain_heatmap_dirty = True

        self.terrain_heatmap_surf = pygame.Surface((self.map_w, self.map_h), pygame.SRCALPHA)
        self.last_heatmap_refresh = 0.0
        self.heatmap_refresh_interval = 0.25
        self.last_explored_update = 0.0
        self.explored_update_interval = 0.5
        
        # Set the starting position for drones
        self.start_point = None
        self.set_start_point()

        # Build the drones and the rovers
        self.build_drones()
        self.build_rovers()

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


# ==============================================================================  
# Main mission loop and thread management
# =============================================================================

    def start_mission(self) -> None:
        """Run the mission: start per-drone threads and drive main loop.

        This method starts a thread per drone that runs `drone_thread`, then
        enters a rendering/update loop until `is_mission_over()` returns
        True. It handles shutdown of threads, the process pool, and shared
        memory cleanup.
        """
        # Start timer
        self.control_center.start_timer()
        # Compute FPS from delay (guard against zero)
        fps = max(1, round(1 / self.delay))

        # Create and start a thread for each drone's movement
        threads = [] # List to keep track of all threads
        for i in range(self.num_drones):
            t = threading.Thread(target=self.drone_thread, args=(i,))
            threads.append(t)
            t.start()

        if self.rover_motion_enabled:
            for i in range(self.num_rovers):
                t = threading.Thread(target=self.rover_thread, args=(i,))
                threads.append(t)
                t.start()

        # Main loop to keep moving drones until the mission is completed
        while not self.completed:
            # Cap frame rate and allow timely interrupt
            self.clock.tick(fps)

            for event in pygame.event.get():
                # If the window is closed
                if event.type == pygame.QUIT:
                    # Set the mission event to signal all threads to stop
                    self.mission_event.set()
                    # Quit and close the program
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    click_result = self.control_center.handle_click(event.pos, self.drones)
                    if click_result == 'terrain_heatmap':
                        self.toggle_terrain_heatmap()

            # Check if mission is over
            self.completed = self.is_mission_over()

            # Redraw the cave and the drones at each frame
            self.draw()
            pygame.display.update()

        # Signal all threads to stop when the mission is complete
        self.mission_event.set()

        # Wait for all threads to finish executing
        for t in threads:
            t.join()
        # Shutdown process pool and cleanup shared memory
        try:
            self.pool.shutdown(wait=True)
        except Exception:
            pass
        if getattr(self, 'map_shm', None):
            try:
                self.map_shm.close()
                self.map_shm.unlink()
            except Exception:
                pass
            

    def set_start_point(self) -> None:
        """Pick a viable start point from the map generator worm starts.

        Keeps sampling the list of candidate worm starts until a non-wall
        coordinate is found.
        """
        # Continuously search for a valid starting point until one is found
        while self.start_point is None or Assets.wall_hit(self.map_matrix, self.start_point):
            # Randomly select one of the initial points of the worms
            # Choose based on available worm starts (don't assume 4)
            i = rand.randrange(len(self.cartographer.worm_x))
            self.start_point = (self.cartographer.worm_x[i], self.cartographer.worm_y[i])
    

    def is_mission_over(self) -> bool:
        """Return True when all drones report mission completion.

        Side-effect: restores the windowed display mode when the mission
        completes.
        """
        # Check if all drones have completed their missions
        for drone in self.drones:
            if not drone.mission_completed():
                return False
        
        # =======================================================================================
        # Post-mission processing: display results, save data, etc. (Placeholder for future features)
        # =======================================================================================
        
        self.game.display = self.game.to_windowed() # Restore the game window to its original size

        return True # All drones are completed, mission is over


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

            # Control the speed of movement
            # Use Event.wait so the sleep is interruptible when `mission_event` is set
            self.mission_event.wait(self.delay)


    def compute_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Submit pathfinding job to the process pool and return the path.

        If shared memory wasn't created at startup this returns an empty
        list. The method blocks until the worker returns the path.
        """
        if not getattr(self, 'map_shm', None):
            return []
        try:
            # Block submission when the pool is saturated to avoid queue buildup
            self.pool_sem.acquire()
            fut = self.pool.submit(AStarPathfinder.compute_path, self.map_shm.name, self.map_shape, start, goal)
            result = fut.result()
            return result
        except Exception:
            return []
        finally:
            try:
                self.pool_sem.release()
            except Exception:
                pass


    def compute_rover_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute a terrain-aware path for rovers without affecting drone A*."""
        with self.terrain_lock:
            known_roughness = self.known_roughness.copy()
            terrain_confidence = self.terrain_confidence.copy()
        cave_map = np.asarray(self.map_matrix, dtype=np.uint8)
        return AStarPathfinder.compute_weighted_path(cave_map, known_roughness, terrain_confidence, start, goal)


    def record_terrain_scan(self, samples: List[Tuple[int, int, float, float]]) -> None:
        """Fuse drone terrain observations into the shared known-terrain maps."""
        if not samples:
            return

        map_updated = False
        with self.terrain_lock:
            for x, y, roughness, confidence in samples:
                xi = int(x)
                yi = int(y)
                if yi < 0 or yi >= self.known_roughness.shape[0] or xi < 0 or xi >= self.known_roughness.shape[1]:
                    continue
                if self.map_matrix[yi][xi] != 0:
                    continue

                obs_conf = float(np.clip(confidence, 0.05, 1.0))
                obs_roughness = float(np.clip(roughness, 0.0, 1.0))
                prev_conf = float(self.terrain_confidence[yi, xi])
                prev_value = float(self.known_roughness[yi, xi]) if prev_conf > 0 else obs_roughness
                total_conf = prev_conf + obs_conf
                blended = ((prev_value * prev_conf) + (obs_roughness * obs_conf)) / total_conf

                self.known_roughness[yi, xi] = blended
                self.terrain_confidence[yi, xi] = min(1.0, total_conf)
                map_updated = True

            now = pygame.time.get_ticks() / 1000.0
            if map_updated and (now - self.last_explored_update) >= self.explored_update_interval:
                explored_cells = int(np.count_nonzero(self.terrain_confidence > 0))
                self.control_center.explored_percent = round((explored_cells / self.floor_cells) * 100)
                self.last_explored_update = now

        if map_updated:
            self.terrain_heatmap_dirty = True


    def toggle_terrain_heatmap(self) -> None:
        """Toggle visibility of the scanned terrain heatmap overlay."""
        self.show_terrain_heatmap = not self.show_terrain_heatmap
        if self.show_terrain_heatmap:
            for drone in self.drones:
                drone.show_path = False
                drone.show_vision = False
        else:
            for drone in self.drones:
                drone.show_path = True
                drone.show_vision = True


    def _refresh_terrain_heatmap(self) -> None:
        """Rebuild the cached terrain heatmap surface from discovered scans."""
        with self.terrain_lock:
            roughness = np.clip(self.known_roughness.copy(), 0.0, 1.0)
            confidence = np.clip(self.terrain_confidence.copy(), 0.0, 1.0)

        valid_mask = confidence > 0.0
        if not np.any(valid_mask):
            self.terrain_heatmap_surf.fill((0, 0, 0, 0))
            self.terrain_heatmap_dirty = False
            return

        # 5-band discrete palette (Blue -> Green -> Yellow -> Orange -> Red)
        # Use nonlinear-stretched roughness to improve separation in mid-range values.
        ramp = np.clip(((roughness - 0.5) * 1.8) + 0.5, 0.0, 1.0)
        band = np.clip((ramp * 5.0).astype(np.int8), 0, 4)

        red = np.zeros_like(ramp, dtype=np.float32)
        green = np.zeros_like(ramp, dtype=np.float32)
        blue = np.zeros_like(ramp, dtype=np.float32)

        # Band colors (RGB):
        # 0: Blue, 1: Green, 2: Yellow, 3: Orange, 4: Red
        red[band == 0], green[band == 0], blue[band == 0] = 30.0, 80.0, 235.0
        red[band == 1], green[band == 1], blue[band == 1] = 45.0, 190.0, 70.0
        red[band == 2], green[band == 2], blue[band == 2] = 245.0, 225.0, 60.0
        red[band == 3], green[band == 3], blue[band == 3] = 245.0, 145.0, 40.0
        red[band == 4], green[band == 4], blue[band == 4] = 235.0, 45.0, 40.0

        alpha = np.where(valid_mask, 35.0 + (confidence * 125.0), 0.0)

        red = np.clip(red, 0.0, 255.0).astype(np.uint8)
        green = np.clip(green, 0.0, 255.0).astype(np.uint8)
        blue = np.clip(blue, 0.0, 255.0).astype(np.uint8)
        alpha = np.clip(alpha, 0.0, 160.0).astype(np.uint8)

        rgb_view = pygame.surfarray.pixels3d(self.terrain_heatmap_surf)
        alpha_view = pygame.surfarray.pixels_alpha(self.terrain_heatmap_surf)
        rgb_view[:, :, 0] = red.T
        rgb_view[:, :, 1] = green.T
        rgb_view[:, :, 2] = blue.T
        alpha_view[:, :] = alpha.T
        del rgb_view
        del alpha_view

        self.terrain_heatmap_dirty = False


    def draw_terrain_heatmap(self) -> None:
        """Blit the scanned terrain heatmap overlay when enabled."""
        if not self.show_terrain_heatmap:
            return
        now = pygame.time.get_ticks() / 1000.0
        if self.terrain_heatmap_dirty and (now - self.last_heatmap_refresh) >= self.heatmap_refresh_interval:
            self._refresh_terrain_heatmap()
            self.last_heatmap_refresh = now
        self.game.window.blit(self.terrain_heatmap_surf, (0, 0))


    def acquire_rover_target(self, rover_id: int, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Choose and reserve a discovered rough-terrain target for a rover."""
        with self.rover_assignment_lock, self.terrain_lock:
            assigned_targets = {target for rid, target in self.rover_assignments.items() if rid != rover_id}
            candidate_mask = (
                (np.asarray(self.map_matrix) == 0)
                & (self.terrain_confidence >= 0.25)
                & (self.known_roughness >= 0.35)
            )

            if not np.any(candidate_mask):
                return None

            ys, xs = np.where(candidate_mask)
            best_target = None
            best_score = float('-inf')
            norm = max(1.0, math.hypot(self.game.width, self.game.height))

            for x, y in zip(xs, ys):
                target = (int(x), int(y))
                if target in assigned_targets or target in self.completed_rover_targets:
                    continue

                distance_penalty = math.dist(current_pos, target) / norm
                score = (0.7 * float(self.known_roughness[y, x])) + (0.3 * float(self.terrain_confidence[y, x])) - distance_penalty
                if score > best_score:
                    best_score = score
                    best_target = target

            if best_target is not None:
                self.rover_assignments[rover_id] = best_target
            return best_target


    def release_rover_target(self, rover_id: int, completed: bool = False) -> None:
        """Release or mark complete a rover terrain target reservation."""
        with self.rover_assignment_lock:
            target = self.rover_assignments.pop(rover_id, None)
            if completed and target is not None:
                self.completed_rover_targets.add(target)


    def rover_thread(self, rover_id: int) -> None:
        """Drive rover movement using the terrain-aware weighted planner."""
        while not self.mission_event.is_set():
            self.rovers[rover_id].move()
            self.mission_event.wait(self.delay)

    
    def build_drones(self) -> None:
        """Instantiate `self.num_drones` `Drone` objects and load icons."""
        # Get the required number of drones from the settings
        self.num_drones = self.settings.num_drones

        # Set drone icon
        icon_size       = self.get_drone_icon_dim()
        self.drone_icon = pygame.image.load(Assets.Images.DRONE.value)
        self.drone_icon = pygame.transform.scale(self.drone_icon, icon_size)

        # List to store drone colors
        self.drone_colors = list(Assets.DroneColors)

        # Populate the swarm
        self.drones = []
        for i in range(self.num_drones):
            self.drones.append(Drone(self.game,
                                     self,
                                     i,
                                     self.start_point,
                                     self.drone_colors.pop(0).value,
                                     self.drone_icon,
                                     self.map_matrix))

    
    def get_drone_icon_dim(self) -> Tuple[int, int]:
        """Return the `(width,height)` for drone icons given map size."""
        match self.settings.map_dim:
            case 'SMALL' : return Assets.GameOptions.DRONE_ICON[0]
            case 'MEDIUM': return Assets.GameOptions.DRONE_ICON[1]
            case 'BIG'   : return Assets.GameOptions.DRONE_ICON[2]


    def pool_information(self) -> None:
        """Query optional informational methods on each drone.

        This method calls optional hooks (if present) on each drone to
        collect or update debugging/state information without assuming
        those methods exist on every drone implementation.
        """
        
        for drone in self.drones:
            if hasattr(drone, 'get_pos_history'):
                drone.get_pos_history()

        for drone in self.drones:
            if hasattr(drone, 'update_explored_map'):
                drone.update_explored_map()


# =============================================================================
# Rover setup and drawing methods
# =============================================================================

    def build_rovers(self) -> None:
        """Instantiate rover agents and prepare their icons.

        Rovers are fewer than drones; this creates `self.num_rovers` rover
        objects, scales the rover icon for the current map size and
        assigns a color from the rover color pool via
        `choose_rover_color()`.
        """
        # Number of rovers scales with drones (one rover per 4 drones)
        self.num_rovers = math.ceil(self.settings.num_drones / 4)

        # Prepare rover icon (scaled to map size)
        icon_size = self.get_rover_icon_dim()
        self.rover_icon = pygame.image.load(Assets.Images.ROVER.value)
        self.rover_icon = pygame.transform.scale(self.rover_icon, icon_size)

        # Pool of rover colors (enum members)
        self.rover_colors = list(Assets.RoverColors)

        # Instantiate rover objects
        self.rovers = []
        for i in range(self.num_rovers):
            color = self.choose_rover_color()
            self.rovers.append(Rover(self.game, self, i, self.start_point, color, self.rover_icon, self.map_matrix))
    

    def choose_rover_color(self) -> Tuple[int, int, int]:     
        """Return and remove a color tuple for a rover from the pool."""
        random_color = rand.choice(self.rover_colors)
        self.rover_colors.remove(random_color)
        return random_color.value


    def get_rover_icon_dim(self) -> Tuple[int, int]:
        """Return the `(width,height)` for rover icons given map size."""
        match self.settings.map_dim:
            case 'SMALL' : return Assets.GameOptions.ROVER_ICON[0]
            case 'MEDIUM': return Assets.GameOptions.ROVER_ICON[1]
            case 'BIG'   : return Assets.GameOptions.ROVER_ICON[2]

# =============================================================================
# Graph class for path validation and obstacle checking
# =============================================================================

    def draw_cave(self) -> None:
        """Draw the base cave map (underlays and floor)."""
        self.game.window.blit(self.cave_png, (0, 0))

    
    def draw_walls(self) -> None:
        """Draw the cave wall overlay (occludes floor but not icons)."""
        self.game.window.blit(self.cave_walls_png, (0, 0))

   
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
        self.control_center.draw_control_center(self.drones, self.rovers, self.show_terrain_heatmap)