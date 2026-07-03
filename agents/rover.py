"""Rover agent for the Cave Explorer simulation.

This module defines rover movement and mission state. Pygame drawing is
delegated to `RoverRenderer`.
"""

import random as rand
from typing import Tuple, List, Optional, TYPE_CHECKING

from agents.graph import Graph
from mapping.terrain_knowledge import TerrainKnowledge
from mission.service_dependencies import RoverNavigationDependencies
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
        self.navigation = RoverNavigationDependencies(
            rover_targets=control.rover_targets,
            compute_rover_path=control.compute_rover_path,
        )
         
        self.id       = id # Unique identifier of the drone
        self.map_size = self.settings.mission_config.map_dim # Map dimension
        self.radius   = self.calculate_radius() # Radius that represent the field of view # 39
        self.step     = 10 # Step of the drone
        self.dir      = rand.randint(0,359)

        self.color = color
        self.alpha = 150
        self.icon  = icon

        self.battery  = 2400
        self.status = 'Ready'
        
        self.ray_points = []  # Initialize the list for rays
        self.delay      = control.delay

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
                self.navigation.rover_targets.release(
                    self.id,
                    completed=True,
                )
                self.target = None
            return

        self.status = 'Updating'
        target = self.navigation.rover_targets.acquire(self.id, self.pos)
        if target is None:
            self.status = 'Ready'
            return

        path = self.navigation.compute_rover_path(self.pos, target)
        if len(path) <= 1:
            self.navigation.rover_targets.release(self.id, completed=False)
            self.status = 'Ready'
            return

        self.target = target
        self.current_path = path[1:]
        self.status = 'Advancing'
