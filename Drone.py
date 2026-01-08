import pygame
import random as rand
import time
import math
from Assets import next_cell_coords, check_pixel_color, Colors
from Graph import Graph
from AStar import AStar


class Drone():
    def __init__(self, game, control, id, start_pos, color, icon, cave, strategy="random"):
        self.game     = game
        self.settings = game.sim_settings
        self.cave     = cave
        self.control  = control
        self.strategy = strategy
         
        self.id       = id # Unique identifier of the drone
        self.map_size = self.settings.map_dim # Map dimension
        self.radius   = self.calculate_radius() # Radius that represent the field of view # 39
        self.step     = 10 # Step of the drone
        self.dir      = rand.randint(0,359)

        self.color = color
        self.alpha = 150
        self.icon  = icon

        self.battery  = 100
        self.statuses = ['Ready', 'Deployed', 'Sharing', 'Homing', 'Charging', 'Done']
        
        self.explored = False # Flag to track if the exploration has started
        
        # Transparent surface used to track the explored path
        self.floor_surf = pygame.Surface((self.game.width,self.game.height), pygame.SRCALPHA)
        self.floor_surf.fill((*Colors.WHITE.value, 0))
        self.ray_points = []  # Initialize the list for rays
        self.delay      = self.control.delay

        self.show_path    = True
        self.speed_factor = 4
         
        self.border    = []
        self.start_pos = start_pos
        self.pos       = start_pos
        self.dir_log   = []
        self.graph     = Graph(*start_pos, cave)
        self.astar     = AStar(self.floor_surf, cave, self.color, self.game)

    # Define the radius based on the map size
    def calculate_radius(self):
        match self.map_size:
            case 'SMALL' : return 40
            case 'MEDIUM': return 20
            case 'BIG'   : return 10
        
    # Manage the movement of the drone
    def move(self):
        node_found = False
        while not node_found:
            try:
                # Find all valid directions
                valid_dirs, valid_targets, chosen_target = self.find_new_node()
            except AssertionError:
                # If no valid directions, update borders and try to reach the nearest border
                self.update_borders()
                node_found = self.reach_border()
            else:
                # Otherwise move in one of the valid directions
                node_found = self.explore(valid_dirs, valid_targets, chosen_target)
    
    # Find a valid direction around the drone
    def find_new_node(self):
        # 360-degree radar scan
        directions = 360

        # Initialize all possible directions and target positions
        all_dirs = list(range(directions)) 
        targets  = []
        dir_res  = int(360/len(all_dirs))  # Resolution for each direction

        # Initialize target positions
        for _ in range(len(all_dirs)):
            targets.append([0,0])

        # Blacklist directions that are not valid
        dir_blacklist = []
        for i in all_dirs:
            # Calculate the target pixel in the current direction
            targets[i][0], targets[i][1] = next_cell_coords(*self.pos, self.radius + 1, i*dir_res)
            
            # Check if the target is valid 
            if not self.graph.is_valid(self.floor_surf, self.pos, (*targets[i],)):
                # Add invalid directions to blacklist
                dir_blacklist.append(i)

        # Filter valid directions
        valid_dirs    = [dir for dir in all_dirs if dir not in dir_blacklist]
        valid_targets = [(*targets[valid_dir],) for valid_dir in valid_dirs]

        # Assert that there is at least one valid direction to proceed
        assert valid_dirs

        # Randomly choose a valid direction and target
        self.dir = rand.choice(valid_dirs)
        target = next_cell_coords(*self.pos, self.step, self.dir)
        while not self.graph.is_valid(self.floor_surf, self.pos, target, step=True):
            valid_dirs.remove(self.dir)
            valid_targets.remove((*targets[self.dir],))
            assert valid_dirs
            self.dir = rand.choice(valid_dirs)
            target = next_cell_coords(*self.pos, self.step, self.dir)
        return valid_dirs, valid_targets, target

    def explore(self, valid_dirs, valid_targets, chosen_target):
        # Flag to indicate whether the exploration has begun
        self.explored = True
        # Log the direction chosen
        self.dir_log.append(self.dir)
        # Add the target node to the graph
        self.graph.add_node(chosen_target)
        # Update the drone's position
        self.pos = chosen_target
        # Remove the explored direction
        valid_dirs.remove(self.dir)
        # Add unexplored pixels to the border list (each pixel only added once)
        self.border.extend(valid_targets)
        self.border = list(set(self.border))
        return True
    
    # If no valid directions are found, use A* algorithm to reach the nearest border pixel
    def reach_border(self):
        
        self.astar.clear() # Reset the A* state
        self.border.sort(key=self.get_distance) # Sort border pixels by distance

        # Use A* to find the optimal path to the closest border
        path = self.astar.find_path(self.pos, self.border)

        # Move the drone along the calculated path
        for node in path:
            self.pos = node
            self.graph.add_node(node) # Add the node to the graph
            time.sleep(self.delay/self.speed_factor) # Add delay for visualization
        return True
    
    # Update the border list, removing explored pixels
    def update_borders(self):
        self.border = [pixel for pixel in self.border if check_pixel_color(self.floor_surf, pixel, self.color, is_not=True)]

    # Check if the mission is completed (no border pixels left)
    def mission_completed(self):
        # Verify that the mission cannot be completed if it has never been explored
        if not self.explored:
            return False
        if not self.border:  # If the border list is empty, the mission is considered completed
            print(f"Drone {self.id} has completed the mission!")  
            return True  # Mission completed
        return False
    
     # Calculate distance from the current position to the target
    def get_distance(self, target):
        dist = math.dist(self.pos, target)
        # Discard targets within the current vision circle
        return self.game.width if dist <= self.radius else dist

    def update_explored_map(self):
        pass

#  ____   ____      _  __        __ ___  _   _   ____ 
# |  _ \ |  _ \    / \ \ \      / /|_ _|| \ | | / ___|
# | | | || |_) |  / _ \ \ \ /\ / /  | | |  \| || |  _
# | |_| ||  _ <  / ___ \ \ V  V /   | | | |\  || |_| |
# |____/ |_| \_\/_/   \_\ \_/\_/   |___||_| \_| \____|

    # Draw the explored path of the drone on the surface
    def draw_path(self):
        # Draw a filled polygon representing the explored area
        pygame.draw.polygon(self.floor_surf, (*self.color, int(2*self.alpha/3)), self.ray_points)

        # Draw the path calculated by the A* algorithm
        if self.show_path:
            for i in range(len(self.graph.pos)):
                if i>0:
                    # Draw a line connecting nodes in the path
                    pygame.draw.line(self.floor_surf, (*self.color, 255),
                                     self.graph.pos[i],
                                     self.graph.pos[i-1], 2)

        # Draw the starting point
        self.start_surf = pygame.Surface((12, 12), pygame.SRCALPHA)
        pygame.draw.circle(self.start_surf, (*Colors.BLUE.value, 255), (6,6), 6)
        
        # Blit the explored path surface and the starting point onto the game window
        self.game.window.blit(self.floor_surf, (0,0))
        self.game.window.blit(self.start_surf, (self.start_pos[0] - 6, self.start_pos[1] - 6))
    
    # Cast a ray from the drone's position to detect walls or obstacles        
    def cast_ray(self, start_pos, angle, max_length):
        step_size = 2 # Smaller step size for higher precision
        for length in range(0, max_length, step_size):
            end_x = start_pos[0] + length * math.cos(angle)
            end_y = start_pos[1] + length * math.sin(angle)

            # Ensure the ray stays within window bounds
            if 0 <= end_x < self.game.window.get_width() and 0 <= end_y < self.game.window.get_height():
                pixel_color = self.game.window.get_at((int(end_x), int(end_y)))
                if pixel_color == (0, 0, 0, 255):   # Check for black (wall) color
                    return (end_x, end_y)

            # Break if the ray goes out of bounds
            if not (0 <= end_x < self.game.window.get_width() and 0 <= end_y < self.game.window.get_height()):
                break
        return None

    # Draw the drone's field of view using sensor rays
    def draw_vision(self):
        num_rays = 100  # Number of rays for 360-degree vision
        angle_increment = 2 * math.pi / num_rays  # Incremental angle between rays
        self.ray_points.clear()  # Clear previous ray points

        # Loop through each ray to calculate its intersection with obstacles
        for i in range(num_rays):
            angle = i * angle_increment  # Calculate the current angle
            intersection = self.cast_ray(self.pos, angle, self.radius)  # Use the drone's position to cast a ray

            if intersection:
                # If the ray intersects an obstacle, add the intersection point to the ray points list
                self.ray_points.append(intersection)  
            else:
                # If there are no intersections, calculate the endpoint of the ray at maximum length
                end_x = self.pos[0] + self.radius * math.cos(angle)
                end_y = self.pos[1] + self.radius * math.sin(angle)
                self.ray_points.append((end_x, end_y))  # Add the final point of the ray to the list

        # Draw the field of view as a polygon if there are enough intersection points
        if len(self.ray_points) > 3:  # Ensure there are at least 3 points to form a polygon
            pygame.draw.polygon(self.game.window, (*self.color, int(2*self.alpha/3)), self.ray_points)  
        else:
             # If not enough points, draw a simple circle to indicate the field of view
            pygame.draw.circle(self.game.window, (*self.color, int(2*self.alpha/3)), (int(self.pos[0]), int(self.pos[1])), self.radius, 1)
      
    # Draw the drone icon
    def draw_icon(self):
        icon_width, icon_height = self.icon.get_size()  # Get the dimensions of the drone icon
        icon_position = (int(self.pos[0] - icon_width // 2), int(self.pos[1] - icon_height // 2))  # Center the icon
        # Blit (draw) the drone icon at the calculated position on the game window
        self.game.window.blit(self.icon, icon_position)