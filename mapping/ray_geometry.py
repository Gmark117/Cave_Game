"""Shared grid-line helpers for ray-based sensing."""

from typing import Tuple


Point = Tuple[int, int]


def bresenham_line_points(x0: int, y0: int, x1: int, y1: int) -> list[Point]:
    """Return integer cells along a line using Bresenham's algorithm."""
    points: list[Point] = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy

    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        doubled_error = 2 * error
        if doubled_error > -dy:
            error -= dy
            x += sx
        if doubled_error < dx:
            error += dx
            y += sy
    return points
