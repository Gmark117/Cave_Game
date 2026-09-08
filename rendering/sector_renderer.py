"""Cached visualization of rover-coordinated exploration sectors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pygame

from mapping.exploration_sectors import ExplorationSectorSnapshot


class SectorRenderer:
    """Draw stable sector ownership without rebuilding it every frame."""

    ACTIVE_FILL_ALPHA = 24
    ACTIVE_EDGE_ALPHA = 150
    WAITING_FILL_ALPHA = 8
    WAITING_EDGE_ALPHA = 65

    def __init__(self, map_width: int, map_height: int) -> None:
        width = int(map_width)
        height = int(map_height)
        if width <= 0 or height <= 0:
            raise ValueError("sector rendering dimensions must be positive")
        self.map_width = width
        self.map_height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._cache_key: tuple[Any, ...] | None = None

    def draw(
        self,
        window: Any,
        snapshot: ExplorationSectorSnapshot | None,
        drone_colors: Mapping[int, tuple[int, int, int]],
    ) -> bool:
        """Blit the current epoch and report whether anything was drawn."""
        if snapshot is None or not snapshot.assignments:
            return False

        normalized_colors = tuple(sorted(
            (int(owner), tuple(int(channel) for channel in color[:3]))
            for owner, color in drone_colors.items()
        ))
        cache_key = (
            snapshot.generation,
            tuple(assignment.sector_id for assignment in snapshot.assignments),
            tuple(sorted(snapshot.waiting_drone_ids)),
            snapshot.mission_exhausted,
            normalized_colors,
        )
        if cache_key != self._cache_key:
            self._rebuild(snapshot, dict(normalized_colors))
            self._cache_key = cache_key
        window.blit(self.surface, (0, 0))
        return True

    def _rebuild(
        self,
        snapshot: ExplorationSectorSnapshot,
        drone_colors: Mapping[int, tuple[int, int, int]],
    ) -> None:
        self.surface.fill((0, 0, 0, 0))
        waiting = snapshot.waiting_drone_ids

        for assignment in snapshot.assignments:
            owner = assignment.owner_drone_id
            color = drone_colors.get(owner, (180, 180, 180))
            is_waiting = owner in waiting or snapshot.mission_exhausted
            fill_alpha = (
                self.WAITING_FILL_ALPHA
                if is_waiting
                else self.ACTIVE_FILL_ALPHA
            )
            edge_alpha = (
                self.WAITING_EDGE_ALPHA
                if is_waiting
                else self.ACTIVE_EDGE_ALPHA
            )
            cells = assignment.cells
            cell_size = assignment.cell_size
            for cell_x, cell_y in cells:
                left = cell_x * cell_size
                top = cell_y * cell_size
                right = min(self.map_width, left + cell_size)
                bottom = min(self.map_height, top + cell_size)
                if left >= self.map_width or top >= self.map_height:
                    continue
                pygame.draw.rect(
                    self.surface,
                    (*color, fill_alpha),
                    (left, top, right - left, bottom - top),
                )
                self._draw_external_edges(
                    cells,
                    cell_x,
                    cell_y,
                    left,
                    top,
                    right,
                    bottom,
                    (*color, edge_alpha),
                )

            pygame.draw.circle(
                self.surface,
                (*color, 235 if not is_waiting else 110),
                assignment.seed,
                7,
                width=2,
            )

        # Every assignment in an epoch shares the same physical gateway.
        gateway = snapshot.assignments[0].gateway
        marker_color = (255, 255, 255, 220)
        pygame.draw.circle(self.surface, marker_color, gateway, 8, width=2)
        pygame.draw.line(
            self.surface,
            marker_color,
            (gateway[0] - 5, gateway[1]),
            (gateway[0] + 5, gateway[1]),
            width=2,
        )
        pygame.draw.line(
            self.surface,
            marker_color,
            (gateway[0], gateway[1] - 5),
            (gateway[0], gateway[1] + 5),
            width=2,
        )

    def _draw_external_edges(
        self,
        cells: frozenset[tuple[int, int]],
        cell_x: int,
        cell_y: int,
        left: int,
        top: int,
        right: int,
        bottom: int,
        color: tuple[int, int, int, int],
    ) -> None:
        """Draw only ownership boundaries, omitting internal cell lines."""
        if (cell_x, cell_y - 1) not in cells:
            pygame.draw.line(self.surface, color, (left, top), (right, top))
        if (cell_x - 1, cell_y) not in cells:
            pygame.draw.line(self.surface, color, (left, top), (left, bottom))
        if (cell_x + 1, cell_y) not in cells:
            pygame.draw.line(
                self.surface,
                color,
                (max(left, right - 1), top),
                (max(left, right - 1), bottom),
            )
        if (cell_x, cell_y + 1) not in cells:
            pygame.draw.line(
                self.surface,
                color,
                (left, max(top, bottom - 1)),
                (right, max(top, bottom - 1)),
            )
