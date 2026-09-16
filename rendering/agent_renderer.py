"""Pygame renderers for drone and rover agents."""

from typing import Any

import pygame

from agents.drone_runtime_state import DroneSnapshot
from asset_config.rendering import Colors


class DroneRenderer:
    """Render a drone's path, vision overlay, start marker, and icon."""

    def __init__(self, drone: Any) -> None:
        """Create reusable transparent surfaces for one drone."""
        self.drone = drone
        game = drone.game

        self.path_surface = pygame.Surface(
            (game.width, game.height),
            pygame.SRCALPHA,
        )
        self.path_surface.fill((*Colors.WHITE.value, 0))
        self._rendered_path_points = 0

        self.vision_surface = pygame.Surface(
            game.window.get_size(),
            pygame.SRCALPHA,
        )
        self._vision_signature = None

        self.start_surface = pygame.Surface((12, 12), pygame.SRCALPHA)
        pygame.draw.circle(
            self.start_surface,
            (*Colors.BLUE.value, 255),
            (6, 6),
            6,
        )

    def draw_path(self, snapshot: DroneSnapshot) -> None:
        """Render route history and the shared starting-point marker."""
        drone = self.drone
        self._draw_new_path_segments(
            snapshot.path_history,
            (*drone.color, 255),
        )

        if snapshot.show_path:
            drone.game.window.blit(self.path_surface, (0, 0))

        drone.game.window.blit(
            self.start_surface,
            (drone.start_pos[0] - 6, drone.start_pos[1] - 6),
        )

    def _draw_new_path_segments(
        self,
        path_history: tuple[tuple[int, int], ...],
        color: tuple[int, int, int, int],
    ) -> None:
        """Draw only path segments that have not already reached the cache."""
        if len(path_history) < self._rendered_path_points:
            self.path_surface.fill((*Colors.WHITE.value, 0))
            self._rendered_path_points = 0

        start_index = max(1, self._rendered_path_points)
        for i in range(start_index, len(path_history)):
            pygame.draw.line(
                self.path_surface,
                color,
                path_history[i],
                path_history[i - 1],
                2,
            )
        self._rendered_path_points = len(path_history)

    def draw_vision_overlay(self, snapshot: DroneSnapshot) -> None:
        """Render the latest sensor-ray endpoints as a vision cone."""
        drone = self.drone
        if not snapshot.show_vision:
            return

        signature = (
            snapshot.position,
            snapshot.ray_points,
            drone.color,
            drone.alpha,
        )
        if not snapshot.ray_points:
            if signature != self._vision_signature:
                self.vision_surface.fill((0, 0, 0, 0))
                self._vision_signature = signature
            return
        if signature != self._vision_signature:
            self.vision_surface.fill((0, 0, 0, 0))
            if len(snapshot.ray_points) > 1:
                points = [snapshot.position, *snapshot.ray_points]
                pygame.draw.polygon(
                    self.vision_surface,
                    (*drone.color, drone.alpha),
                    points,
                )
            else:
                pygame.draw.circle(
                    self.vision_surface,
                    (*drone.color, drone.alpha),
                    (
                        int(snapshot.position[0]),
                        int(snapshot.position[1]),
                    ),
                    12,
                    1,
                )
            self._vision_signature = signature

        drone.game.window.blit(self.vision_surface, (0, 0))

    def draw_icon(self, snapshot: DroneSnapshot) -> None:
        """Blit the drone icon centered at its current position."""
        drone = self.drone
        icon_width, icon_height = drone.icon.get_size()
        icon_position = (
            int(snapshot.position[0] - icon_width // 2),
            int(snapshot.position[1] - icon_height // 2),
        )
        drone.game.window.blit(drone.icon, icon_position)

    def draw_sharing_cue(
        self,
        snapshot: DroneSnapshot,
        target_position: tuple[int, int],
        target_radius: int | None = None,
    ) -> None:
        """Draw a yellow physical link around actively sharing agents."""
        window = self.drone.game.window
        color = Colors.YELLOW.value
        start = tuple(map(int, snapshot.position))
        end = tuple(map(int, target_position))
        start_radius = max(self.drone.icon.get_size()) // 2 + 4
        end_radius = start_radius if target_radius is None else target_radius
        pygame.draw.line(window, color, start, end, 2)
        pygame.draw.circle(window, color, start, start_radius, 2)
        pygame.draw.circle(window, color, end, end_radius, 2)


class RoverRenderer:
    """Render a rover's path history and icon."""

    def __init__(self, rover: Any) -> None:
        """Create reusable transparent surfaces for one rover."""
        self.rover = rover
        game = rover.game

        self.path_surface = pygame.Surface(
            (game.width, game.height),
            pygame.SRCALPHA,
        )
        self.path_surface.fill((*Colors.WHITE.value, 0))
        self._rendered_path_points = 0
        self._rotated_icons: dict[int, pygame.Surface] = {}

    def draw_path(self) -> None:
        """Render the rover route history."""
        rover = self.rover
        path_history = tuple(rover.graph.pos)
        if len(path_history) < self._rendered_path_points:
            self.path_surface.fill((*Colors.WHITE.value, 0))
            self._rendered_path_points = 0

        start_index = max(1, self._rendered_path_points)
        for i in range(start_index, len(path_history)):
            pygame.draw.line(
                self.path_surface,
                (*rover.color, 180),
                path_history[i],
                path_history[i - 1],
                2,
            )
        self._rendered_path_points = len(path_history)

        if not rover.show_path:
            return

        rover.game.window.blit(self.path_surface, (0, 0))

    def draw_icon(self) -> None:
        """Blit the rover icon centered at its current position."""
        rover = self.rover
        snapshot_method = getattr(rover, "snapshot", None)
        snapshot = snapshot_method() if callable(snapshot_method) else None
        position = snapshot.position if snapshot is not None else rover.pos
        heading = float(
            getattr(snapshot, "heading_deg", getattr(rover, "heading_deg", 0.0))
        )
        angle = int(round(heading)) % 360
        icon = self._rotated_icons.get(angle)
        if icon is None:
            # Pygame uses counter-clockwise angles; mission headings are
            # clockwise from north, matching the source sprite orientation.
            icon = pygame.transform.rotate(rover.icon, -angle)
            self._rotated_icons[angle] = icon
        icon_width, icon_height = icon.get_size()
        icon_position = (
            int(position[0] - icon_width // 2),
            int(position[1] - icon_height // 2),
        )
        rover.game.window.blit(icon, icon_position)
