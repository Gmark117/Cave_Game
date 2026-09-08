"""Distance-weighted terrain roughness sampling for SLAM rays."""

from typing import Iterable, List, Tuple
import math

import numpy as np

from mapping.vision_sensor import RayHit


class RoughnessSampler:
    """Samples terrain roughness along rays with distance-based confidence."""

    def __init__(self, terrain_roughness: np.ndarray, map_matrix: list) -> None:
        """Store the generated roughness layer and cave collision map."""
        self.terrain_roughness = terrain_roughness
        self.map_matrix = map_matrix
        self.map_h = len(map_matrix)
        self.map_w = len(map_matrix[0]) if self.map_h else 0
        self.max_range = max(1.0, float(math.hypot(self.map_w, self.map_h)))

    def sample_from_rays(
        self,
        origin: Tuple[float, float],
        ray_hits: Iterable[RayHit],
        step: int = 2
    ) -> List[Tuple[int, int, float, float]]:
        """Generate roughness samples along rays until wall hits."""
        if self.map_w <= 0 or self.map_h <= 0:
            return []

        coordinates: list[Tuple[int, int]] = []
        for hit in ray_hits:
            ex, ey = int(hit.end[0]), int(hit.end[1])
            if ex < 0 or ey < 0 or ex >= self.map_w or ey >= self.map_h:
                continue

            points = list(hit.points)[::max(1, step)]
            for x, y in points:
                if y < 0 or y >= self.map_h or x < 0 or x >= self.map_w:
                    break
                if self.map_matrix[y][x] != 0:
                    break
                coordinates.append((x, y))
        if not coordinates:
            return []

        points = np.asarray(coordinates, dtype=np.intp)
        x = points[:, 0]
        y = points[:, 1]
        base = np.asarray(self.terrain_roughness)[y, x].astype(np.float64)
        distance = np.hypot(
            x.astype(np.float64) - float(origin[0]),
            y.astype(np.float64) - float(origin[1]),
        )
        confidence = np.maximum(0.2, 1.0 - distance / self.max_range)
        roughness = np.clip(
            base + np.random.uniform(-0.03, 0.03, size=len(points)),
            0.0,
            1.0,
        )
        return [
            (int(xi), int(yi), float(value), float(weight))
            for xi, yi, value, weight in zip(
                x,
                y,
                roughness,
                confidence,
            )
        ]
