"""Agent construction helpers for Cave Game.

This module centralizes the creation and initialization of drones and
rovers so MissionControl can remain focused on mission orchestration.
"""

import math
import random as rand
from typing import Tuple

import numpy as np
import pygame

from asset_config.gameplay import GameOptions
from asset_config.media import Images
from asset_config.rendering import DroneColors, RoverColors
from Drone import Drone
from Rover import Rover


class AgentFactory:
    """Factory for constructing and initializing mission agents."""

    @staticmethod
    def build_drones(control) -> None:
        """Create drones, load the shared icon, and attach them to control."""
        control.num_drones = control.settings.num_drones
        icon_size = AgentFactory.get_drone_icon_dim(control.settings.map_dim)
        control.drone_icon = pygame.transform.scale(
            pygame.image.load(Images.DRONE.value),
            icon_size
        )

        drone_colors = list(DroneColors)
        control.drones = []
        for i in range(control.num_drones):
            control.drones.append(
                Drone(
                    control.game,
                    control,
                    i,
                    control.start_point,
                    drone_colors.pop(0).value,
                    control.drone_icon,
                    control.map_matrix,
                )
            )

    @staticmethod
    def build_rovers(control) -> None:
        """Create rovers, load the shared icon, and attach them to control."""
        control.num_rovers = math.ceil(control.settings.num_drones / 4)
        icon_size = AgentFactory.get_rover_icon_dim(control.settings.map_dim)
        control.rover_icon = pygame.transform.scale(
            pygame.image.load(Images.ROVER.value),
            icon_size
        )

        rover_colors = list(RoverColors)
        control.rovers = []
        for i in range(control.num_rovers):
            color = AgentFactory.choose_rover_color(rover_colors)
            rover = Rover(
                control.game,
                control,
                i,
                control.start_point,
                color,
                control.rover_icon,
                control.map_matrix,
            )
            rover.known_roughness = np.full(np.asarray(control.map_matrix).shape, -1.0, dtype=np.float32)
            rover.terrain_confidence = np.zeros(np.asarray(control.map_matrix).shape, dtype=np.float32)
            control.rovers.append(rover)

    @staticmethod
    def choose_rover_color(rover_colors) -> Tuple[int, int, int]:
        """Return and remove a rover color from a mutable pool."""
        random_color = rand.choice(rover_colors)
        rover_colors.remove(random_color)
        return random_color.value

    @staticmethod
    def get_drone_icon_dim(map_dim: str) -> Tuple[int, int]:
        """Return the `(width, height)` for drone icons given the map size."""
        match map_dim:
            case 'SMALL':
                return GameOptions.DRONE_ICON[0]
            case 'MEDIUM':
                return GameOptions.DRONE_ICON[1]
            case 'BIG':
                return GameOptions.DRONE_ICON[2]
            case _:
                return GameOptions.DRONE_ICON[1]

    @staticmethod
    def get_rover_icon_dim(map_dim: str) -> Tuple[int, int]:
        """Return the `(width, height)` for rover icons given the map size."""
        match map_dim:
            case 'SMALL':
                return GameOptions.ROVER_ICON[0]
            case 'MEDIUM':
                return GameOptions.ROVER_ICON[1]
            case 'BIG':
                return GameOptions.ROVER_ICON[2]
            case _:
                return GameOptions.ROVER_ICON[1]
