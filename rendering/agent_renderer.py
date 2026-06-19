"""Pygame renderers for drone and rover agents."""

from typing import Any

import pygame

from asset_config.rendering import Colors


class DroneRenderer:
    """Render a drone's path, vision overlay, start marker, and icon."""

    def __init__(self, drone: Any) -> None:
        self.drone = drone
        game = drone.game

        self.path_surface = pygame.Surface(
            (game.width, game.height),
            pygame.SRCALPHA,
        )
        self.path_surface.fill((*Colors.WHITE.value, 0))

        self.vision_surface = pygame.Surface(
            game.window.get_size(),
            pygame.SRCALPHA,
        )

        self.start_surface = pygame.Surface((12, 12), pygame.SRCALPHA)
        pygame.draw.circle(
            self.start_surface,
            (*Colors.BLUE.value, 255),
            (6, 6),
            6,
        )

    def draw_path(self) -> None:
        """Render route history and the shared starting-point marker."""
        drone = self.drone
        with drone.exploration_lock:
            for i in range(1, len(drone.graph.pos)):
                pygame.draw.line(
                    self.path_surface,
                    (*drone.color, 255),
                    drone.graph.pos[i],
                    drone.graph.pos[i - 1],
                    2,
                )

            if drone.show_path:
                drone.game.window.blit(self.path_surface, (0, 0))

        drone.game.window.blit(
            self.start_surface,
            (drone.start_pos[0] - 6, drone.start_pos[1] - 6),
        )

    def draw_vision_overlay(self) -> None:
        """Render the latest sensor-ray endpoints as a vision cone."""
        drone = self.drone
        if not drone.show_vision:
            return

        self.vision_surface.fill((0, 0, 0, 0))
        if len(drone.ray_points) > 1:
            points = [drone.pos] + drone.ray_points
            pygame.draw.polygon(
                self.vision_surface,
                (*drone.color, drone.alpha),
                points,
            )
        else:
            pygame.draw.circle(
                self.vision_surface,
                (*drone.color, drone.alpha),
                (int(drone.pos[0]), int(drone.pos[1])),
                12,
                1,
            )

        drone.game.window.blit(self.vision_surface, (0, 0))

    def draw_icon(self) -> None:
        """Blit the drone icon centered at its current position."""
        drone = self.drone
        icon_width, icon_height = drone.icon.get_size()
        icon_position = (
            int(drone.pos[0] - icon_width // 2),
            int(drone.pos[1] - icon_height // 2),
        )
        drone.game.window.blit(drone.icon, icon_position)


class RoverRenderer:
    """Render a rover's path history and icon."""

    def __init__(self, rover: Any) -> None:
        self.rover = rover
        game = rover.game

        self.path_surface = pygame.Surface(
            (game.width, game.height),
            pygame.SRCALPHA,
        )
        self.path_surface.fill((*Colors.WHITE.value, 0))

    def draw_path(self) -> None:
        """Render the rover route history."""
        rover = self.rover
        if not rover.show_path:
            return

        for i in range(1, len(rover.graph.pos)):
            pygame.draw.line(
                self.path_surface,
                (*rover.color, 180),
                rover.graph.pos[i],
                rover.graph.pos[i - 1],
                2,
            )
        rover.game.window.blit(self.path_surface, (0, 0))

    def draw_icon(self) -> None:
        """Blit the rover icon centered at its current position."""
        rover = self.rover
        icon_width, icon_height = rover.icon.get_size()
        icon_position = (
            int(rover.pos[0] - icon_width // 2),
            int(rover.pos[1] - icon_height // 2),
        )
        rover.game.window.blit(rover.icon, icon_position)

