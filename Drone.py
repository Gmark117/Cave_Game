"""Drone agent for the Cave Explorer simulation.

Provides the `Drone` class which implements simple frontier exploration
and vision rendering. Type hints are added for clarity while keeping
runtime imports permissive to avoid circular dependencies.
"""

import math
import time
import random as rand
import threading
from typing import List, Tuple, Optional

import numpy as np
import pygame

from asset_config.helpers import next_cell_coords, check_pixel_color
from asset_config.rendering import Colors
from Graph import Graph


class Drone:
    """Autonomous aerial agent that explores the cave using frontier search.

    The `Drone` maintains a local exploration graph, a transparent surface
    storing explored areas, and methods to move, cast vision rays and
    render its state. Types are kept permissive for `game` and `control`
    to avoid circular imports.
    """

    def __init__(self, game, control, id: int, start_pos: Tuple[int, int],
                 color: Tuple[int, int, int], icon: 'pygame.Surface', cave: list) -> None:
        """Initialize runtime state for a drone.

        Args:
            game: Owner `Game` instance (used for surfaces and settings).
            control: `MissionControl` instance used for pathfinding calls.
            id: Unique integer id for the drone.
            start_pos: (x,y) starting coordinates.
            color: RGB color tuple used for drawing the drone overlays.
            icon: Pygame Surface for drawing the drone icon.
            cave: Binary map matrix used for collision checks.
        """
        self.game = game
        self.settings = game.sim_settings
        self.cave = cave
        self.control = control

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
        self.show_vision = True
        self.speed_factor = 4
        self.scan_interval = 0.25
        self.last_scan_time = 0.0
        self.scan_rays = 48
        self.border_retry_cooldown = 1.5
        self.border_retry_until: dict[Tuple[int, int], float] = {}

        # Local terrain knowledge (distributed mapping)
        self.known_roughness = np.full(np.asarray(self.cave).shape, -1.0, dtype=np.float32)
        self.terrain_confidence = np.zeros(np.asarray(self.cave).shape, dtype=np.float32)
        self.terrain_lock = threading.Lock()
        self.exploration_lock = threading.Lock()
        self.last_share_time = 0.0
        self.share_interval = 0.5

        # Exploration bookkeeping
        self.border = []
        self.start_pos = start_pos
        self.pos = start_pos
        self.dir_log = []
        self.graph = Graph(*start_pos, cave)
        self.returning_home = False
        self.done = False

    
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
        if self.done:
            return

        # Once exploration is exhausted, return to the start point
        if self.returning_home or (self.explored and not self.border):
            self.returning_home = True
            if self.reach_start_point():
                self.done = True
            return

        node_found = False
        while not node_found:
            try:
                # Find all valid directions
                valid_dirs, valid_targets, chosen_target = self.find_new_node()
            except AssertionError:
                # If no valid directions, update borders and try to reach the nearest border
                self.update_borders()
                node_found = self.reach_border()
                if not node_found:
                    # Yield control to avoid busy-looping in dead-ends
                    return
            else:
                # Otherwise move in one of the valid directions
                node_found = self.explore(valid_dirs, valid_targets, chosen_target)


    def reach_start_point(self) -> bool:
        """Return the drone to `start_pos` using A* pathfinding.

        Returns True when the drone is back at the starting position,
        otherwise False.
        """
        if self.pos == self.start_pos:
            return True

        path: List[Tuple[int, int]] = []
        if hasattr(self.control, 'compute_path'):
            path = self.control.compute_path(self.pos, self.start_pos)

        if not path:
            return False

        for node in path:
            self.pos = node
            self.graph.add_node(node)
            if hasattr(self.control, 'mission_event'):
                self.control.mission_event.wait(self.delay / self.speed_factor)
            else:
                time.sleep(self.delay / self.speed_factor)

        return self.pos == self.start_pos
    
    
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

        if not self.border:
            return False

        # Try multiple candidate border pixels (nearest first)
        now = time.perf_counter()
        for target in list(self.border):
            if target == self.pos:
                continue
            retry_at = self.border_retry_until.get(target, 0.0)
            if now < retry_at:
                continue

            path: List[Tuple[int, int]] = []
            if hasattr(self.control, 'compute_path'):
                path = self.control.compute_path(self.pos, target)

            # Accept only meaningful paths (more than current position)
            if not path or len(path) <= 1:
                self.border_retry_until[target] = now + self.border_retry_cooldown
                continue

            for node in path:
                self.pos = node
                self.graph.add_node(node)
                if hasattr(self.control, 'mission_event'):
                    self.control.mission_event.wait(self.delay / self.speed_factor)
                else:
                    time.sleep(self.delay / self.speed_factor)

            # Reached candidate; remove it from border and continue exploration
            if target in self.border:
                self.border.remove(target)
            self.border_retry_until.pop(target, None)
            return True

        return False
    
    
    def update_borders(self) -> None:
        """Remove border pixels that are now explored.

        A border cell is considered explored if either:
        - it is already painted on `floor_surf`, or
        - local/shared terrain confidence marks it as known.
        """
        def is_explored(pixel: Tuple[int, int]) -> bool:
            x, y = int(pixel[0]), int(pixel[1])
            if y < 0 or y >= self.terrain_confidence.shape[0] or x < 0 or x >= self.terrain_confidence.shape[1]:
                return True
            if self.cave[y][x] != 0:
                return True
            explored_by_surface = self.floor_surf.get_at((x, y))[3] > 0
            explored_by_terrain = float(self.terrain_confidence[y, x]) > 0.0
            return explored_by_surface or explored_by_terrain

        with self.exploration_lock:
            self.border = [pixel for pixel in self.border if not is_explored(pixel)]
        valid = set(self.border)
        self.border_retry_until = {k: v for k, v in self.border_retry_until.items() if k in valid}

    
    def mission_completed(self) -> bool:
        """Check if the exploration mission is complete."""
        # Verify that the mission cannot be completed if it has never been explored
        if not self.explored:
            return False

        # Exploration finished: trigger homing, then complete only at start
        if not self.border and not self.done:
            self.returning_home = True
            return False

        if self.done:
            print(f"Drone {self.id} has completed the mission!")
            return True  # Mission completed

        return False
    
    
    def get_distance(self, target: Tuple[int, int]) -> float:
        dist = math.dist(self.pos, target)
        # Discard targets within the current vision circle by returning a large value
        return float(self.game.width) if dist <= self.radius else dist

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
        with self.exploration_lock:
            # Filled polygon of the last vision rays (gives a filled explored area)
            if len(self.ray_points) > 2:
                pygame.draw.polygon(self.floor_surf, (*self.color, int(2 * self.alpha / 3)), self.ray_points)

            # Draw the A* path as a polyline
            for i in range(1, len(self.graph.pos)):
                pygame.draw.line(self.floor_surf, (*self.color, 255), self.graph.pos[i], self.graph.pos[i - 1], 2)

        # Draw the starting point marker
        self.start_surf = pygame.Surface((12, 12), pygame.SRCALPHA)
        pygame.draw.circle(self.start_surf, (*Colors.BLUE.value, 255), (6, 6), 6)

        # Blit the explored path surface and the starting point onto the game window
        if self.show_path:
            with self.exploration_lock:
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


    def scan_terrain(self, num_rays: int = 24) -> None:
        """Sample terrain roughness along visible rays and update local terrain maps."""
        if not hasattr(self.control, 'terrain_roughness'):
            return
        terrain = self.control.terrain_roughness
        if terrain.shape != np.asarray(self.cave).shape:
            return

        now = time.perf_counter()
        if (now - self.last_scan_time) < self.scan_interval:
            return
        self.last_scan_time = now

        samples = []
        angle_increment = 2 * math.pi / num_rays
        sample_step = max(2, self.radius // 12)
        height = len(self.cave)
        width = len(self.cave[0]) if height else 0

        for i in range(num_rays):
            angle = i * angle_increment
            for length in range(0, self.radius + 1, sample_step):
                sample_x = int(self.pos[0] + length * math.cos(angle))
                sample_y = int(self.pos[1] + length * math.sin(angle))

                if not (0 <= sample_x < width and 0 <= sample_y < height):
                    break
                if self.cave[sample_y][sample_x] != 0:
                    break

                base_roughness = float(terrain[sample_y, sample_x])
                confidence = max(0.2, 1.0 - (length / max(1, self.radius)))
                noise = rand.uniform(-0.03, 0.03)
                samples.append((sample_x, sample_y, min(1.0, max(0.0, base_roughness + noise)), confidence))

        # Update local terrain knowledge
        self._record_local_terrain_scan(samples)


    def _record_local_terrain_scan(self, samples: List[Tuple[int, int, float, float]]) -> None:
        """Fuse terrain observations into this drone's local knowledge maps."""
        if not samples:
            return
        
        with self.terrain_lock:
            for x, y, roughness, confidence in samples:
                xi = int(x)
                yi = int(y)
                if yi < 0 or yi >= self.known_roughness.shape[0] or xi < 0 or xi >= self.known_roughness.shape[1]:
                    continue
                if self.cave[yi][xi] != 0:
                    continue
                
                obs_conf = float(np.clip(confidence, 0.05, 1.0))
                obs_roughness = float(np.clip(roughness, 0.0, 1.0))
                prev_conf = float(self.terrain_confidence[yi, xi])
                prev_value = float(self.known_roughness[yi, xi]) if prev_conf > 0 else obs_roughness
                total_conf = prev_conf + obs_conf
                blended = ((prev_value * prev_conf) + (obs_roughness * obs_conf)) / total_conf
                
                self.known_roughness[yi, xi] = blended
                self.terrain_confidence[yi, xi] = min(1.0, total_conf)
    
    def merge_terrain_data(self, other_roughness: np.ndarray, other_confidence: np.ndarray) -> None:
        """Merge terrain data from another drone into this drone's maps.
        
        Uses weighted averaging: cells with higher confidence in either map
        are weighted more heavily.
        """
        if other_roughness is None or other_confidence is None:
            return
        
        with self.terrain_lock:
            h = min(self.known_roughness.shape[0], other_roughness.shape[0])
            w = min(self.known_roughness.shape[1], other_roughness.shape[1])
            if h <= 0 or w <= 0:
                return

            target_rough = self.known_roughness[:h, :w]
            target_conf = self.terrain_confidence[:h, :w]
            source_rough = np.clip(other_roughness[:h, :w], 0.0, 1.0)
            source_conf = np.clip(other_confidence[:h, :w], 0.0, 1.0)
            floor = (np.asarray(self.cave)[:h, :w] == 0)

            valid = floor & (source_conf > 0.0)
            if not np.any(valid):
                return

            self_conf_vals = target_conf[valid]
            source_conf_vals = source_conf[valid]
            source_rough_vals = source_rough[valid]
            self_rough_vals = target_rough[valid]

            base_self = np.where(self_conf_vals > 0.0, self_rough_vals, source_rough_vals)
            total = self_conf_vals + source_conf_vals
            blended = ((base_self * self_conf_vals) + (source_rough_vals * source_conf_vals)) / np.maximum(total, 1e-6)

            target_rough[valid] = blended
            target_conf[valid] = np.minimum(1.0, total)

    def merge_exploration_data(
        self,
        other_explored_alpha: np.ndarray,
        other_border: List[Tuple[int, int]]
    ) -> None:
        """Merge highlighted explored area and border metadata from another drone.

        This allows frontiers (`border`) to converge when drones exchange data.
        """
        if other_explored_alpha is None and other_border is None:
            return

        with self.exploration_lock:
            if other_explored_alpha is not None:
                # Merge only explored alpha, but repaint with this drone color to avoid color bleed.
                rgb_view = pygame.surfarray.pixels3d(self.floor_surf)
                alpha_view = pygame.surfarray.pixels_alpha(self.floor_surf)

                h = min(alpha_view.shape[1], other_explored_alpha.shape[1])
                w = min(alpha_view.shape[0], other_explored_alpha.shape[0])
                if h > 0 and w > 0:
                    incoming_alpha = other_explored_alpha[:w, :h]
                    mask = incoming_alpha > 0
                    if np.any(mask):
                        alpha_view[:w, :h][mask] = np.maximum(alpha_view[:w, :h][mask], incoming_alpha[mask])
                        rgb_view[:w, :h, 0][mask] = self.color[0]
                        rgb_view[:w, :h, 1][mask] = self.color[1]
                        rgb_view[:w, :h, 2][mask] = self.color[2]

                del rgb_view
                del alpha_view

            if other_border:
                merged = set(self.border)
                for b in other_border:
                    p = (int(b[0]), int(b[1]))
                    if 0 <= p[1] < len(self.cave) and 0 <= p[0] < len(self.cave[0]) and self.cave[p[1]][p[0]] == 0:
                        merged.add(p)
                self.border = list(merged)

        self.update_borders()

    
    def draw_vision(self) -> None:
        """Render the drone's vision cone by casting multiple sensor rays.

        Uses `cast_ray` to collect hit points; if enough points are found a
        filled polygon is rendered to visually represent the agent's FOV.
        """
        num_rays = 72  # Number of rays for 360-degree vision
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

        self.scan_terrain(self.scan_rays)

        # Draw the field of view as a polygon if there are enough intersection points
        if not self.show_vision:
            return

        if len(self.ray_points) > 3:
            pygame.draw.polygon(self.game.window, (*self.color, int(2 * self.alpha / 3)), self.ray_points)
        else:
            # Fallback: draw an outline circle indicating vision radius
            pygame.draw.circle(self.game.window, (*self.color, int(2 * self.alpha / 3)), (int(self.pos[0]), int(self.pos[1])), self.radius, 1)


    def toggle_path(self) -> None:
        """Toggle rendering visibility for the drone path overlay."""
        self.show_path = not self.show_path


    def toggle_vision(self) -> None:
        """Toggle rendering visibility for the drone vision overlay."""
        self.show_vision = not self.show_vision
      
    
    def draw_icon(self) -> None:
        """Blit the drone `icon` centered on the agent position."""
        icon_width, icon_height = self.icon.get_size()
        icon_position = (int(self.pos[0] - icon_width // 2), int(self.pos[1] - icon_height // 2))
        self.game.window.blit(self.icon, icon_position)
