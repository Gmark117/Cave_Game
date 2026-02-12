import random as rand
import pygame
import sys
import math
import threading
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import Assets
from ControlCenter import ControlCenter
from Drone import Drone
from Rover import Rover
import AStarPathfinder


class MissionControl():
    def __init__(self, game):
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
            self.map_shm = shm
        except Exception:
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

#  __  __  ___  ____   ____   ___   ___   _   _      ____   ___   _   _  _____  ____    ___   _     
# |  \/  ||_ _|/ ___| / ___| |_ _| / _ \ | \ | |    / ___| / _ \ | \ | ||_   _||  _ \  / _ \ | |
# | |\/| | | | \___ \ \___ \  | | | | | ||  \| |   | |    | | | ||  \| |  | |  | |_) || | | || |
# | |  | | | |  ___) | ___) | | | | |_| || |\  |   | |___ | |_| || |\  |  | |  |  _ < | |_| || |___
# |_|  |_||___||____/ |____/ |___| \___/ |_| \_|    \____| \___/ |_| \_|  |_|  |_| \_\ \___/ |_____|

    def start_mission(self):
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
    # Among the starting positions of the worms, find one that is viable
    def set_start_point(self):
         # Continuously search for a valid starting point until one is found
        while self.start_point is None or Assets.wall_hit(self.map_matrix, self.start_point):
            # Randomly select one of the initial points of the worms
            # Choose based on available worm starts (don't assume 4)
            i = rand.randrange(len(self.cartographer.worm_x))
            self.start_point = (self.cartographer.worm_x[i], self.cartographer.worm_y[i])
    
    # Check if the mission is completed
    def is_mission_over(self):
        # Check if all drones have completed their missions
        for drone in self.drones:
            if not drone.mission_completed():
                return False
        
        return True # All drones are completed, mission is over

#  ____   ____    ___   _   _  _____ 
# |  _ \ |  _ \  / _ \ | \ | || ____|
# | | | || |_) || | | ||  \| ||  _|
# | |_| ||  _ < | |_| || |\  || |___
# |____/ |_| \_\ \___/ |_| \_||_____|

    # Thread function for each drone's movement
    def drone_thread(self, drone_id):
        # Continue moving the drone until mission event is set or the drone completes its mission
        while not self.mission_event.is_set() and not self.drones[drone_id].mission_completed():
            self.drones[drone_id].move()  # Move the drone

            # Control the speed of movement
            # Use Event.wait so the sleep is interruptible when `mission_event` is set
            self.mission_event.wait(self.delay)

    def compute_path(self, start, goal):
        """Submit pathfinding job to worker pool and return result (blocking)."""
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

    # Instantiate the swarm of drones as a list
    def build_drones(self):
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

    # Return the dimension of the drone icon given the map dimension
    def get_drone_icon_dim(self):
        match self.settings.map_dim:
            case 'SMALL' : return Assets.GameOptions.DRONE_ICON[0]
            case 'MEDIUM': return Assets.GameOptions.DRONE_ICON[1]
            case 'BIG'   : return Assets.GameOptions.DRONE_ICON[2]

    def pool_information(self):
        # Iterate drone objects directly; guard optional methods
        for drone in self.drones:
            if hasattr(drone, 'get_pos_history'):
                drone.get_pos_history()

        for drone in self.drones:
            if hasattr(drone, 'update_explored_map'):
                drone.update_explored_map()

#  ____    ___  __     __ _____  ____  
# |  _ \  / _ \ \ \   / /| ____||  _ \
# | |_) || | | | \ \ / / |  _|  | |_) |
# |  _ < | |_| |  \ V /  | |___ |  _ <
# |_| \_\ \___/    \_/   |_____||_| \_\

    # Instantiate the fleet of rovers as a list
    def build_rovers(self):
        # Get the number of rovers depending on the number of drones
        self.num_rovers = math.ceil(self.settings.num_drones/4)

        # Set rover icon
        icon_size       = self.get_rover_icon_dim()
        self.rover_icon = pygame.image.load(Assets.Images.ROVER.value)
        self.rover_icon = pygame.transform.scale(self.rover_icon, icon_size)

        # List to store rover colors (Deprecated: Rovers don't need colors)
        self.rover_colors = list(Assets.RoverColors)

        # Populate the swarm
        self.rovers = []
        for i in range(self.num_rovers):
            self.rovers.append(Rover(self.game, self, i, self.start_point, self.choose_rover_color(), self.rover_icon, self.map_matrix))
    
    # Function to get a random color for each drone
    def choose_rover_color(self):     
        # Choose a random color from the list, then remove it
        random_color = rand.choice(self.rover_colors)
        self.rover_colors.remove(random_color)
        return random_color.value

    # Return the dimension of the drone icon given the map dimension
    def get_rover_icon_dim(self):
        match self.settings.map_dim:
            case 'SMALL' : return Assets.GameOptions.ROVER_ICON[0]
            case 'MEDIUM': return Assets.GameOptions.ROVER_ICON[1]
            case 'BIG'   : return Assets.GameOptions.ROVER_ICON[2]

#  ____   ____      _  __        __ ___  _   _   ____ 
# |  _ \ |  _ \    / \ \ \      / /|_ _|| \ | | / ___|
# | | | || |_) |  / _ \ \ \ /\ / /  | | |  \| || |  _
# | |_| ||  _ <  / ___ \ \ V  V /   | | | |\  || |_| |
# |____/ |_| \_\/_/   \_\ \_/\_/   |___||_| \_| \____|

    # Remove the icons drawn in the last positions
    def draw_cave(self):
        # Blit the cave map image onto the game window at (0, 0) position
        self.game.window.blit(self.cave_png, (0, 0))

    # Blit the cave walls
    def draw_walls(self):
        # The walls cover everything but the drone icon
        self.game.window.blit(self.cave_walls_png, (0, 0))

    # Draw all game elements in layers: (Lowest layer) 0 -> 3 (Highest layer)
    def draw(self):
        # Base map
        self.draw_cave()

        # Per-drone overlays: vision and explored path
        for drone in self.drones:
            drone.draw_vision()
            drone.draw_path()

        # Draw cave walls once
        self.draw_walls()

        # Draw all icons (drones and rovers)
        for i, drone in enumerate(self.drones):
            drone.draw_icon()
            if i < len(self.rovers):
                self.rovers[i].draw_icon()
                
        # Control center UI
        self.control_center.draw_control_center()