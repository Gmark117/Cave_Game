"""Distance-weighted terrain roughness sampling for SLAM rays."""

from typing import Iterable, List, Tuple
import math

import numpy as np


class RoughnessSampler:
    """Samples terrain roughness along rays with distance-based confidence."""

    def __init__(self, terrain_roughness: np.ndarray, map_matrix: list) -> None:
        self.terrain_roughness = terrain_roughness
        self.map_matrix = map_matrix
        self.map_h = len(map_matrix)
        self.map_w = len(map_matrix[0]) if self.map_h else 0
        self.max_range = max(1.0, float(math.hypot(self.map_w, self.map_h)))

    def sample_from_rays(
        self,
        origin: Tuple[float, float],
        ray_hits: Iterable[object],
        step: int = 4
    ) -> List[Tuple[int, int, float, float]]:
        """Generate roughness samples along rays until wall hits."""
        if self.map_w <= 0 or self.map_h <= 0:
            return []

        samples: List[Tuple[int, int, float, float]] = []
        for hit in ray_hits:
            end = getattr(hit, 'end', None)
            if end is None:
                continue
            ex, ey = int(end[0]), int(end[1])
            if ex < 0 or ey < 0 or ex >= self.map_w or ey >= self.map_h:
                continue

            points = self._line_points(int(origin[0]), int(origin[1]), ex, ey, step)
            for x, y in points:
                if y < 0 or y >= self.map_h or x < 0 or x >= self.map_w:
                    break
                if self.map_matrix[y][x] != 0:
                    break

                base = float(self.terrain_roughness[y, x])
                dist = math.dist(origin, (x, y))
                confidence = max(0.2, 1.0 - (dist / self.max_range))
                noise = float(np.random.uniform(-0.03, 0.03))
                samples.append((x, y, min(1.0, max(0.0, base + noise)), confidence))
        return samples

    def _line_points(self, x0: int, y0: int, x1: int, y1: int, step: int) -> List[Tuple[int, int]]:
        """Return points along a line at fixed step intervals."""
        points: List[Tuple[int, int]] = []

        dx = x1 - x0
        dy = y1 - y0
        length = max(1, int(math.hypot(dx, dy)))
        for i in range(0, length + 1, max(1, step)):
            t = i / max(1, length)
            x = int(round(x0 + dx * t))
            y = int(round(y0 + dy * t))
            points.append((x, y))
        return points
