"""Reusable gameplay/map helper functions."""

from typing import Tuple


def next_cell_coords(x: int, y: int, step_len: int, direction: int) -> Tuple[int, int]:
    """Calculate next cell coordinates based on position, step, and direction."""
    import math

    rad = math.radians(direction)
    dx = round(step_len * math.sin(rad))
    dy = -round(step_len * math.cos(rad))
    return x + dx, y + dy


def wall_hit(map_matrix: list, pos: Tuple[int, int]) -> bool:
    """Return True if position corresponds to a wall in map_matrix."""

    return map_matrix[pos[1]][pos[0]] == 1


def check_pixel_color(surface, pixel: Tuple[int, int], color: Tuple[int, int, int], is_not: bool = False) -> bool:
    """Check if pixel color matches (or does not match) the given color."""

    pixel_color = surface.get_at(pixel)[:3]
    return pixel_color != color if is_not else pixel_color == color
