"""Drone agent for the Cave Explorer simulation.

Provides the `Drone` class which implements simple frontier exploration
and vision rendering. Type hints are added for clarity while keeping
runtime imports permissive to avoid circular dependencies.
"""

import math
import time
import random as rand
from typing import List, Tuple, Optional

import pygame

from Assets import next_cell_coords, check_pixel_color, Colors
from Graph import Graph


class Drone:
    """Autonomous aerial agent that explores the cave using frontier search.

    The `Drone` maintains a local exploration graph, a transparent surface
    storing explored areas, and methods to move, cast vision rays and
    render its state. Types are kept permissive for `game` and `control`
    to avoid circular imports.
    """

    def __init__(self, game, control, id: int, start_pos: Tuple[int, int],
                 color: Tuple[int, int, int], icon: 'pygame.Surface', cave: list,
                 strategy: str = "random") -> None:
        """Initialize runtime state for a drone.

        Args:
            game: Owner `Game` instance (used for surfaces and settings).
            control: `MissionControl` instance used for pathfinding calls.
            id: Unique integer id for the drone.
            start_pos: (x,y) starting coordinates.
            color: RGB color tuple used for drawing the drone overlays.
            icon: Pygame Surface for drawing the drone icon.
            cave: Binary map matrix used for collision checks.
            strategy: Strategy name (currently unused; reserved for extension).
        """
        self.game = game
        self.settings = game.sim_settings
        self.cave = cave
        self.control = control
        self.strategy = strategy

        # Identity and movement
        self.id = id
        self.map_size = self.settings.map_dim
        self.radius = self.calculate_radius()
        self.step = 10
        self.dir = rand.randint(0, 359)

        # Appearance / drawing
        self.color = color
        self.alpha = 150
        self.icon = icon

        # State and lifecycle
        self.battery = 100
        self.statuses = ['Ready', 'Deployed', 'Sharing', 'Homing', 'Charging', 'Done']
        self.explored = False

        # Transparent surface used to track the explored path
        self.floor_surf = pygame.Surface((self.game.width, self.game.height), pygame.SRCALPHA)
        self.floor_surf.fill((*Colors.WHITE.value, 0))
        self.ray_points = []
        self.delay = self.control.delay

        # Rendering / motion configuration
        self.show_path = True
        self.speed_factor = 4

        # Exploration bookkeeping
        self.border = []
        self.start_pos = start_pos
        self.pos = start_pos
        self.dir_log = []
        self.graph = Graph(*start_pos, cave)

    
    def calculate_radius(self) -> int:
        """Return vision radius in pixels derived from map size setting."""
        match self.map_size:
            case 'SMALL':
                return 40
            case 'MEDIUM':
                return 20
            case 'BIG':
                return 10
            case _:
                return 20
        
    
    def move(self) -> None:
        """Main movement loop invoked by `MissionControl` threads.

        Repeatedly attempts to find a new frontier direction; if none are
        available, tries to reach a stored border pixel using pathfinding.
        """
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
    
    
    def find_new_node(self) -> Tuple[List[int], List[Tuple[int, int]], Tuple[int, int]]:
        """Scan 360 degrees and return (valid_dirs, valid_targets, chosen_target).

        Raises `AssertionError` if no valid directions exist.
        """
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


    def explore(self, valid_dirs: List[int], valid_targets: List[Tuple[int, int]], chosen_target: Tuple[int, int]) -> bool:
        """Attempt exploration toward `chosen_target`.

        Uses `MissionControl.compute_path` when available. Returns True on
        successful traversal, False if no path was found.
        """
        # Flag to indicate whether the exploration has begun
        self.explored = True
        # Log the direction chosen
        self.dir_log.append(self.dir)
        # Add unexplored pixels to the border list (each pixel only added once)
        self.border.extend(valid_targets)
        # Use set to deduplicate; convert back to list of tuples
        self.border = list(set(self.border))
        # Remove the explored direction
        valid_dirs.remove(self.dir)

        # Compute path to chosen target using MissionControl worker pool
        path = []
        if hasattr(self.control, 'compute_path'):
            path = self.control.compute_path(self.pos, chosen_target)

        # If no path found, return False to trigger boundary handling
        if not path:
            return False

        # Follow the path step-by-step
        for node in path:
            self.pos = node
            self.graph.add_node(node)
            if hasattr(self.control, 'mission_event'):
                # Use Event.wait for interruptible sleeping
                self.control.mission_event.wait(self.delay / self.speed_factor)
            else:
                time.sleep(self.delay / self.speed_factor)

        return True
    
    
    def reach_border(self) -> bool:
        """Use A* to reach the nearest saved border pixel.

        Returns True if a path was followed to the border pixel, False
        otherwise.
        """
        self.border.sort(key=self.get_distance)  # Sort border pixels by distance

        # Use worker pool pathfinding to the closest border pixel
        if not self.border:
            return False
        target = self.border[0]
        path = []
        if hasattr(self.control, 'compute_path'):
            path = self.control.compute_path(self.pos, target)

        if not path:
            return False

        for node in path:
            self.pos = node
            self.graph.add_node(node)
            if hasattr(self.control, 'mission_event'):
                self.control.mission_event.wait(self.delay / self.speed_factor)
            else:
                time.sleep(self.delay / self.speed_factor)
        return True
    
    
    def update_borders(self) -> None:
        """Remove border pixels that are now explored (not matching `self.color`)."""
        self.border = [pixel for pixel in self.border if check_pixel_color(self.floor_surf, pixel, self.color, is_not=True)]

    
    def mission_completed(self) -> bool:
        # Verify that the mission cannot be completed if it has never been explored
        if not self.explored:
            return False
        if not self.border:  # If the border list is empty, the mission is considered completed
            print(f"Drone {self.id} has completed the mission!")
            return True  # Mission completed
        return False
    
    
    def get_distance(self, target: Tuple[int, int]) -> float:
        dist = math.dist(self.pos, target)
        # Discard targets within the current vision circle by returning a large value
        return float(self.game.width) if dist <= self.radius else dist


    def update_explored_map(self) -> None:
        """Placeholder for generating an explored map snapshot (optional).

        Currently a no-op; kept for compatibility with MissionControl calls.
        """
        return None


# =============================================================================
# Drone drawing methods (vision and path)
# =============================================================================
    
    def draw_path(self) -> None:
        """Render the explored-floor overlay and the A* path on `floor_surf`.

        - If `ray_points` contains a polygon, fill it to indicate explored area.
        - Draw A* path lines stored in `self.graph.pos` when `show_path` is True.
        - Draw a small starting-point marker and blit both overlays to the
          main window.
        """
        # Filled polygon of the last vision rays (gives a filled explored area)
        if len(self.ray_points) > 2:
            pygame.draw.polygon(self.floor_surf, (*self.color, int(2 * self.alpha / 3)), self.ray_points)

        # Draw the A* path as a polyline (if enabled)
        if self.show_path:
            for i in range(1, len(self.graph.pos)):
                pygame.draw.line(self.floor_surf, (*self.color, 255), self.graph.pos[i], self.graph.pos[i - 1], 2)

        # Draw the starting point marker
        self.start_surf = pygame.Surface((12, 12), pygame.SRCALPHA)
        pygame.draw.circle(self.start_surf, (*Colors.BLUE.value, 255), (6, 6), 6)

        # Blit the explored path surface and the starting point onto the game window
        self.game.window.blit(self.floor_surf, (0, 0))
        self.game.window.blit(self.start_surf, (self.start_pos[0] - 6, self.start_pos[1] - 6))
    
    
    def cast_ray(self, start_pos: Tuple[float, float], angle: float, max_length: int) -> Optional[Tuple[float, float]]:
        """Cast a radial ray and return the first collision point or None.

        Steps along the ray by `step_size` pixels and samples the window
        pixel color; returns the collision coordinates when a wall is
        detected (black pixel), otherwise None if no hit within range.
        """
        step_size = 2  # Smaller step size for higher precision
        for length in range(0, max_length, step_size):
            end_x = start_pos[0] + length * math.cos(angle)
            end_y = start_pos[1] + length * math.sin(angle)

            # Ensure the ray stays within window bounds
            if 0 <= end_x < self.game.window.get_width() and 0 <= end_y < self.game.window.get_height():
                pixel_color = self.game.window.get_at((int(end_x), int(end_y)))
                if pixel_color == (0, 0, 0, 255):  # Check for black (wall) color
                    return (end_x, end_y)

            # Break if the ray goes out of bounds
            if not (0 <= end_x < self.game.window.get_width() and 0 <= end_y < self.game.window.get_height()):
                break
        return None

    
    def draw_vision(self) -> None:
        """Render the drone's vision cone by casting multiple sensor rays.

        Uses `cast_ray` to collect hit points; if enough points are found a
        filled polygon is rendered to visually represent the agent's FOV.
        """
        num_rays = 100  # Number of rays for 360-degree vision
        angle_increment = 2 * math.pi / num_rays  # Incremental angle between rays
        self.ray_points.clear()  # Clear previous ray points

        # Loop through each ray to calculate its intersection with obstacles
        for i in range(num_rays):
            angle = i * angle_increment
            intersection = self.cast_ray(self.pos, angle, self.radius)

            if intersection:
                # If the ray intersects an obstacle, add the intersection point
                self.ray_points.append(intersection)
            else:
                # Otherwise add the far endpoint at max radius
                end_x = self.pos[0] + self.radius * math.cos(angle)
                end_y = self.pos[1] + self.radius * math.sin(angle)
                self.ray_points.append((end_x, end_y))

        # Draw the field of view as a polygon if there are enough intersection points
        if len(self.ray_points) > 3:
            pygame.draw.polygon(self.game.window, (*self.color, int(2 * self.alpha / 3)), self.ray_points)
        else:
            # Fallback: draw an outline circle indicating vision radius
            pygame.draw.circle(self.game.window, (*self.color, int(2 * self.alpha / 3)), (int(self.pos[0]), int(self.pos[1])), self.radius, 1)
      
    
    def draw_icon(self) -> None:
        """Blit the drone `icon` centered on the agent position."""
        icon_width, icon_height = self.icon.get_size()
        icon_position = (int(self.pos[0] - icon_width // 2), int(self.pos[1] - icon_height // 2))
        self.game.window.blit(self.icon, icon_position)