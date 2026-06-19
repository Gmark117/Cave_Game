"""Simple graph utilities for tracking explored node positions.

`Graph` stores a list of positions visited by an agent and provides
validation helpers (line-of-sight checks) used during exploration.
"""

from typing import Tuple, List

from asset_config.helpers import wall_hit

class Graph:
    def __init__(self, x_start: int, y_start: int, cave_mat: list) -> None:
        """Initialize graph with starting node and a reference to cave matrix.

        `cave_mat` is the binary map matrix where `1` indicates a wall.
        """
        self.cave_mat = cave_mat
        # List of visited node positions (x,y)
        self.pos = [(x_start, y_start)]
    
    
    def add_node(self, pos: Tuple[int, int]) -> None:
        """Append `pos` (x,y) to the graph's position list."""
        self.pos.append(pos)
    
    
    def is_valid(
        self,
        curr_pos: Tuple[int, int],
        candidate_pos: Tuple[int, int],
    ) -> bool:
        """Return True when the target and route do not intersect walls."""
        return (
            not wall_hit(self.cave_mat, candidate_pos)
            and not self.cross_obs(*curr_pos, *candidate_pos)
        )
    
    
    def cross_obs(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Return True if the line from (x1,y1) to (x2,y2) crosses any wall.

        Uses an integer Bresenham line-walking algorithm. Checks the starting
        and end pixels and every pixel along the line using `wall_hit`.
        """
        # Localize for speed
        cave = self.cave_mat
        wallHit = wall_hit

        x0, y0 = int(x1), int(y1)
        x1, y1 = int(x2), int(y2)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        x, y = x0, y0

        # Check start point
        if wallHit(cave, (x, y)):
            return True

        # Walk the line using Bresenham's algorithm. The implementation
        # chooses the driving axis (x or y) based on the greater delta to
        # guarantee single-step increments along the major axis.
        if dx > dy:
            err = dx // 2
            while x != x1:
                x += sx
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                if wallHit(cave, (x, y)):
                    return True
            return False
        else:
            err = dy // 2
            while y != y1:
                y += sy
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                if wallHit(cave, (x, y)):
                    return True
            return False
