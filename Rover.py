"""Rover agent for the Cave Explorer simulation.

This module defines the `Rover` class which behaves similarly to a
`Drone` but with different battery and visualization settings. The
class provides movement helpers and simple drawing utilities used by
`MissionControl` when rendering and stepping the simulation.
"""

import math
import time
import random as rand
from typing import Tuple

import pygame

from Assets import next_cell_coords, check_pixel_color, Colors
from Graph import Graph


class Rover:
    """Simple ground rover agent used for map exploration visualization.

    The rover stores runtime state (position, battery, path surface)
    and provides drawing helpers. Types are intentionally permissive
    to avoid circular imports with `Game`/`MissionControl`.
    """

    def __init__(self, game: object, control: object, id: int, start_pos: Tuple[int, int],
                 color: Tuple[int, int, int], icon: pygame.Surface, cave: list, strategy: str = "random") -> None:
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

        self.battery  = 2400
        self.statuses = ['Ready', 'Updating', 'Advancing', 'Done']
        
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

    # Define the radius based on the map size
    def calculate_radius(self) -> int:
        """Return vision radius (pixels) based on chosen map size."""
        match self.map_size:
            case 'SMALL' : return 40
            case 'MEDIUM': return 20
            case 'BIG'   : return 10
            case _       : return 20


#  ____   ____      _  __        __ ___  _   _   ____ 
# |  _ \ |  _ \    / \ \ \      / /|_ _|| \ | | / ___|
# | | | || |_) |  / _ \ \ \ /\ / /  | | |  \| || |  _
# | |_| ||  _ <  / ___ \ \ V  V /   | | | |\  || |_| |
# |____/ |_| \_\/_/   \_\ \_/\_/   |___||_| \_| \____|

    # Draw the rover icon
    def draw_icon(self) -> None:
        """Blit the rover icon centered at current position onto the window."""
        icon_width, icon_height = self.icon.get_size()  # Get dimensions of the icon
        icon_position = (int(self.pos[0] - icon_width // 2), int(self.pos[1] - icon_height // 2))  # Center the icon

        # Blit the drone icon at the calculated position
        self.game.window.blit(self.icon, icon_position)