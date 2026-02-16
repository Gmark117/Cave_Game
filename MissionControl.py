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
        self.cave_png     = pygame.image.load(Assets.Images.CAVE_MAP.value).convert_alpha() # Load cave map image

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
        # Per-drone overlays: draw explored paths (under vision)
        for drone in self.drones:
            drone.draw_path()

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
        self.control_center.draw_control_center()