"""Drone agent for the Cave Explorer simulation.

Provides the `Drone` state object and compatibility facade. Exploration,
sensing, and drawing are delegated to focused controller/renderer classes.
"""

import random as rand
import threading
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

from Graph import Graph
from SlamMap import SlamMap
from agents.drone_movement import DroneMovementController
from mapping.drone_sensor import DroneSensorController
from mapping.terrain_knowledge import TerrainKnowledge, TerrainSnapshot
from rendering.agent_renderer import DroneRenderer

if TYPE_CHECKING:
    import pygame


class Drone:
    """Autonomous agent state shared by movement, mapping, and rendering."""

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
        # Vision cone alpha set to 128 for semi-transparency
        self.alpha = 128
        self.icon = icon

        # State and lifecycle
        self.battery = 100
        self.statuses = ['Ready', 'Deployed', 'Sharing', 'Homing', 'Charging', 'Done']
        self.explored = False

        self.ray_points = []
        self.delay = self.control.delay

        # Presentation and traversal configuration
        self.show_path = True
        self.show_vision = True
        self.speed_factor = 4

        # Local terrain knowledge (distributed mapping)
        self.terrain_knowledge = TerrainKnowledge(self.cave)
        self.slam_lock = threading.Lock()
        self.exploration_lock = threading.Lock()

        # Exploration bookkeeping
        self.border = []
        self.start_pos = start_pos
        self.pos = start_pos
        self.dir_log = []
        self.graph = Graph(*start_pos, cave)
        self.returning_home = False
        self.done = False
        self.heading_deg = 0.0

        # SLAM state
        map_h = len(self.cave)
        map_w = len(self.cave[0]) if map_h else 0
        max_points = int(getattr(self.settings, 'slam_point_cloud_max_points', 6000))
        self.slam_map = SlamMap(map_h, map_w, max_points=max_points)
        self.movement_controller = DroneMovementController(self)
        self.sensor_controller = DroneSensorController(self)
        self.renderer = DroneRenderer(self)

    
    def calculate_radius(self) -> int:
        """Return vision radius in pixels derived from map size setting."""
        match self.map_size:
            case 'SMALL':
                return 40
            case 'MEDIUM':
                return 20
            case 'LARGE':
                return 10
            case _:
                return 20
        
    
    def move(self) -> None:
        """Compatibility wrapper for the exploration state machine."""
        self.movement_controller.move()


    def reach_start_point(self) -> bool:
        """Compatibility wrapper for homing behavior."""
        return self.movement_controller.reach_start_point()
    
    
    def find_new_node(self) -> Tuple[List[int], List[Tuple[int, int]], Tuple[int, int]]:
        """Compatibility wrapper for local direction selection."""
        return self.movement_controller.find_new_node()


    def explore(self, valid_dirs: List[int], valid_targets: List[Tuple[int, int]], chosen_target: Tuple[int, int]) -> bool:
        """Compatibility wrapper for exploration traversal."""
        return self.movement_controller.explore(
            valid_dirs,
            valid_targets,
            chosen_target,
        )
    
    
    def reach_border(self) -> bool:
        """Compatibility wrapper for frontier navigation."""
        return self.movement_controller.reach_border()
    
    
    def update_borders(self) -> None:
        """Compatibility wrapper for frontier rebuilding."""
        self.movement_controller.update_borders()

    def _maybe_rebuild_frontiers(self) -> bool:
        """Compatibility wrapper for cooldown-limited frontier rebuilding."""
        return self.movement_controller.maybe_rebuild_frontiers()

    def _rebuild_frontiers(self, stride: int = 4, confidence_threshold: float = 0.6) -> None:
        """Compatibility wrapper for frontier extraction."""
        self.movement_controller.rebuild_frontiers(
            stride,
            confidence_threshold,
        )

    
    def mission_completed(self) -> bool:
        """Compatibility wrapper for movement completion state."""
        return self.movement_controller.mission_completed()
    
    
    def get_distance(self, target: Tuple[int, int]) -> float:
        """Compatibility wrapper for frontier distance ordering."""
        return self.movement_controller.get_distance(target)

    def draw_path(self) -> None:
        """Compatibility wrapper for path rendering."""
        self.renderer.draw_path()
    
    
    def update_sensors(self) -> None:
        """Update local SLAM and terrain knowledge without rendering."""
        self.sensor_controller.update()

    def scan_terrain(self, ray_hits: List[object]) -> None:
        """Compatibility wrapper for terrain sampling from supplied ray hits."""
        self.sensor_controller.scan_terrain(ray_hits)

    def _record_local_terrain_scan(
        self, samples: List[Tuple[int, int, float, float]]
    ) -> None:
        """Compatibility wrapper for local terrain fusion."""
        self.sensor_controller.record_local_scan(samples)
    
    def merge_terrain_data(self, other_roughness: np.ndarray, other_confidence: np.ndarray) -> None:
        """Compatibility wrapper for terrain-knowledge merging."""
        if other_roughness is None or other_confidence is None:
            return
        self.terrain_knowledge.merge_from(
            TerrainSnapshot(
                np.asarray(other_roughness, dtype=np.float32),
                np.asarray(other_confidence, dtype=np.float32),
            )
        )

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
            if other_border:
                merged = set(self.border)
                for b in other_border:
                    p = (int(b[0]), int(b[1]))
                    if 0 <= p[1] < len(self.cave) and 0 <= p[0] < len(self.cave[0]) and self.cave[p[1]][p[0]] == 0:
                        merged.add(p)
                self.border = list(merged)

        self.update_borders()

    
    def draw_vision_overlay(self) -> None:
        """Compatibility wrapper for vision-overlay rendering."""
        self.renderer.draw_vision_overlay()

    def draw_vision(self) -> None:
        """Compatibility alias for the pure vision-overlay draw method."""
        self.draw_vision_overlay()


    def toggle_path(self) -> None:
        """Toggle rendering visibility for the drone path overlay."""
        self.show_path = not self.show_path


    def toggle_vision(self) -> None:
        """Toggle rendering visibility for the drone vision overlay."""
        self.show_vision = not self.show_vision
      
    
    def draw_icon(self) -> None:
        """Compatibility wrapper for icon rendering."""
        self.renderer.draw_icon()

    @property
    def floor_surf(self):
        """Legacy access to the renderer-owned path surface."""
        return self.renderer.path_surface

    @property
    def vision_overlay(self):
        """Legacy access to the renderer-owned vision surface."""
        return self.renderer.vision_surface

    @property
    def known_roughness(self) -> np.ndarray:
        """Compatibility access to local terrain roughness."""
        return self.terrain_knowledge.roughness

    @property
    def terrain_confidence(self) -> np.ndarray:
        """Compatibility access to local terrain confidence."""
        return self.terrain_knowledge.confidence

    @property
    def terrain_lock(self):
        """Compatibility access to the local terrain lock."""
        return self.terrain_knowledge.lock

    @property
    def border_retry_cooldown(self) -> float:
        return self.movement_controller.border_retry_cooldown

    @border_retry_cooldown.setter
    def border_retry_cooldown(self, value: float) -> None:
        self.movement_controller.border_retry_cooldown = value

    @property
    def border_retry_until(self) -> dict[Tuple[int, int], float]:
        return self.movement_controller.border_retry_until

    @border_retry_until.setter
    def border_retry_until(
        self,
        value: dict[Tuple[int, int], float],
    ) -> None:
        self.movement_controller.border_retry_until = value

    @property
    def frontier_rebuild_cooldown(self) -> float:
        return self.movement_controller.frontier_rebuild_cooldown

    @frontier_rebuild_cooldown.setter
    def frontier_rebuild_cooldown(self, value: float) -> None:
        self.movement_controller.frontier_rebuild_cooldown = value

    @property
    def last_frontier_rebuild(self) -> float:
        return self.movement_controller.last_frontier_rebuild

    @last_frontier_rebuild.setter
    def last_frontier_rebuild(self, value: float) -> None:
        self.movement_controller.last_frontier_rebuild = value

    @property
    def frontier_stride(self) -> int:
        return self.movement_controller.frontier_stride

    @frontier_stride.setter
    def frontier_stride(self, value: int) -> None:
        self.movement_controller.frontier_stride = value

    @property
    def frontier_confidence_threshold(self) -> float:
        return self.movement_controller.frontier_confidence_threshold

    @frontier_confidence_threshold.setter
    def frontier_confidence_threshold(self, value: float) -> None:
        self.movement_controller.frontier_confidence_threshold = value

    def merge_slam_map(self, other_map: SlamMap) -> None:
        """Merge another drone's SLAM map into this one."""
        if other_map is None:
            return
        with self.slam_lock:
            self.slam_map.merge_from(other_map)

    def _update_heading(self, prev: Tuple[int, int], curr: Tuple[int, int]) -> None:
        """Compatibility wrapper for heading updates."""
        self.movement_controller.update_heading(prev, curr)
