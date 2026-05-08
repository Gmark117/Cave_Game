"""Vision sensor utilities for SLAM-style raycasting."""

from dataclasses import dataclass
from typing import List, Tuple
import math

from asset_config.helpers import wall_hit


@dataclass(frozen=True)
class RayHit:
    end: Tuple[int, int]
    hit: bool
    distance: float
    angle_deg: float


class VisionSensor:
    """Cast rays within a narrow FOV to detect walls using the map matrix."""

    def __init__(self, map_matrix: list, fov_deg: float = 60.0, num_rays: int = 60, step: int = 2) -> None:
        self.map_matrix = map_matrix
        self.map_h = len(map_matrix)
        self.map_w = len(map_matrix[0]) if self.map_h else 0
        self.fov_deg = float(fov_deg)
        self.num_rays = max(1, int(num_rays))
        self.step = max(1, int(step))
        self.max_range = int(math.hypot(self.map_w, self.map_h))

    def cast_cone(self, origin: Tuple[float, float], heading_deg: float) -> List[RayHit]:
        """Cast rays in a cone centered at heading_deg; returns ray hits."""
        if self.map_w <= 0 or self.map_h <= 0:
            return []

        half_fov = self.fov_deg / 2.0
        hits: List[RayHit] = []
        for i in range(self.num_rays):
            if self.num_rays == 1:
                angle_deg = heading_deg
            else:
                frac = i / (self.num_rays - 1)
                angle_deg = heading_deg - half_fov + frac * self.fov_deg
            hits.append(self._cast_single_ray(origin, angle_deg))
        return hits

    def _cast_single_ray(self, origin: Tuple[float, float], angle_deg: float) -> RayHit:
        rad = math.radians(angle_deg)
        dx = math.sin(rad)
        dy = -math.cos(rad)

        last_valid = (int(round(origin[0])), int(round(origin[1])))
        for length in range(0, self.max_range + 1, self.step):
            x = origin[0] + length * dx
            y = origin[1] + length * dy
            xi = int(round(x))
            yi = int(round(y))

            if xi < 0 or yi < 0 or xi >= self.map_w or yi >= self.map_h:
                break

            last_valid = (xi, yi)
            if wall_hit(self.map_matrix, last_valid):
                dist = math.dist(origin, last_valid)
                return RayHit(last_valid, True, dist, angle_deg)

        dist = math.dist(origin, last_valid)
        return RayHit(last_valid, False, dist, angle_deg)
