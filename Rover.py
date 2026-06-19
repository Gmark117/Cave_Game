"""Rover agent for the Cave Explorer simulation.

This module defines rover movement and mission state. Pygame drawing is
delegated to `RoverRenderer`.
"""

import random as rand
from typing import Tuple, List, Optional, TYPE_CHECKING

from Graph import Graph
from mapping.terrain_knowledge import TerrainKnowledge
from rendering.agent_renderer import RoverRenderer

if TYPE_CHECKING:
    import pygame


class Rover:
    """Simple ground rover agent used for map exploration visualization.

    The rover stores runtime state and delegates Pygame drawing to its
    renderer. Types are intentionally permissive to avoid circular imports.
    """

    def __init__(self, game: object, control: object, id: int, start_pos: Tuple[int, int],
                 color: Tuple[int, int, int], icon: 'pygame.Surface', cave: list) -> None:
        self.game     = game
        self.settings = game.sim_settings
        self.cave     = cave
        self.control  = control
         
        self.id       = id # Unique identifier of the drone
        self.map_size = self.settings.map_dim # Map dimension
        self.radius   = self.calculate_radius() # Radius that represent the field of view # 39
        self.step     = 10 # Step of the drone
        self.dir      = rand.randint(0,359)

        self.color = color
        self.alpha = 150
        self.icon  = icon

        self.battery  = 2400
        self.status = 'Ready'
        
        self.ray_points = []  # Initialize the list for rays
        self.delay      = self.control.delay

        self.show_path    = True
        self.speed_factor = 4
        self.current_path: List[Tuple[int, int]] = []
        self.target: Optional[Tuple[int, int]] = None
         
        self.border    = []
        self.start_pos = start_pos
        self.pos       = start_pos
        self.dir_log   = []
        self.graph     = Graph(*start_pos, cave)
        self.terrain_knowledge = TerrainKnowledge(cave)
        self.renderer  = RoverRenderer(self)

    # Define the radius based on the map size
    def calculate_radius(self) -> int:
        """Return vision radius (pixels) based on chosen map size."""
        match self.map_size:
            case 'SMALL' : return 40
            case 'MEDIUM': return 20
            case 'LARGE'   : return 10
            case _       : return 20


    def move(self) -> None:
        """Run the provisional rover policy while rover motion is disabled.

        This implementation predates the distributed-knowledge contract.
        Replace its mission-global target and routing inputs with rover-local
        received knowledge before enabling rover worker threads.
        """
        if self.current_path:
            self.status = 'Advancing'
            self.pos = self.current_path.pop(0)
            self.graph.add_node(self.pos)
            self.battery = max(0, self.battery - 1)

            if not self.current_path:
                self.status = 'Done'
                if hasattr(self.control, 'release_rover_target'):
                    self.control.release_rover_target(self.id, completed=True)
                self.target = None
            return

        self.status = 'Updating'
        if not hasattr(self.control, 'acquire_rover_target') or not hasattr(self.control, 'compute_rover_path'):
            return

        target = self.control.acquire_rover_target(self.id, self.pos)
        if target is None:
            self.status = 'Ready'
            return

        path = self.control.compute_rover_path(self.pos, target)
        if len(path) <= 1:
            self.control.release_rover_target(self.id, completed=False)
            self.status = 'Ready'
            return

        self.target = target
        self.current_path = path[1:]
        self.status = 'Advancing'


    def draw_path(self) -> None:
        """Compatibility wrapper for path rendering."""
        self.renderer.draw_path()

    def draw_icon(self) -> None:
        """Compatibility wrapper for icon rendering."""
        self.renderer.draw_icon()

    @property
    def floor_surf(self):
        """Legacy access to the renderer-owned path surface."""
        return self.renderer.path_surface

    @property
    def known_roughness(self):
        """Compatibility access to local terrain roughness."""
        return self.terrain_knowledge.roughness

    @property
    def terrain_confidence(self):
        """Compatibility access to local terrain confidence."""
        return self.terrain_knowledge.confidence

    @property
    def terrain_lock(self):
        """Compatibility access to the local terrain lock."""
        return self.terrain_knowledge.lock
