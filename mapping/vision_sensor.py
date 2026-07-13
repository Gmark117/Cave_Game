"""Vision sensor utilities for SLAM-style raycasting."""

from dataclasses import dataclass
from typing import List, Tuple
import math

from asset_config.helpers import wall_hit
from mapping.ray_geometry import bresenham_line_points


@dataclass(frozen=True)
class RayHit:
    """One raycast result plus the grid cells traversed to its endpoint."""

    end: Tuple[int, int]
    hit: bool
    distance: float
    angle_deg: float
    points: Tuple[Tuple[int, int], ...]


class VisionSensor:
    """Cast rays within a narrow FOV to detect walls using the map matrix."""

    def __init__(
        self,
        map_matrix: list,
        fov_deg: float = 60.0,
        num_rays: int = 60,
        step: int = 2,
        max_range: int | None = None,
    ) -> None:
        """Configure a simple grid raycaster over the cave matrix."""
        self.map_matrix = map_matrix
        self.map_h = len(map_matrix)
        self.map_w = len(map_matrix[0]) if self.map_h else 0
        self.fov_deg = float(fov_deg)
        self.num_rays = max(1, int(num_rays))
        self.step = max(1, int(step))
        map_range = int(math.hypot(self.map_w, self.map_h))
        self.max_range = max(
            1,
            int(max_range) if max_range is not None else map_range,
        )

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
        """Walk one ray until it reaches a wall or leaves the map."""
        rad = math.radians(angle_deg)
        dx = math.sin(rad)
        dy = -math.cos(rad)

        start = (int(round(origin[0])), int(round(origin[1])))
        last_valid = start
        traversed: List[Tuple[int, int]] = []
        previous_sample = start
        sample_lengths = list(range(0, self.max_range + 1, self.step))
        if sample_lengths[-1] != self.max_range:
            sample_lengths.append(self.max_range)

        for length in sample_lengths:
            x = origin[0] + length * dx
            y = origin[1] + length * dy
            sample = (int(round(x)), int(round(y)))
            segment = bresenham_line_points(
                previous_sample[0],
                previous_sample[1],
                sample[0],
                sample[1],
            )
            for point in segment:
                if traversed and point == traversed[-1]:
                    continue
                xi, yi = point
                if xi < 0 or yi < 0 or xi >= self.map_w or yi >= self.map_h:
                    return RayHit(
                        last_valid,
                        False,
                        math.dist(origin, last_valid),
                        angle_deg,
                        tuple(traversed),
                    )

                traversed.append(point)
                last_valid = point
                if wall_hit(self.map_matrix, point):
                    return RayHit(
                        point,
                        True,
                        math.dist(origin, point),
                        angle_deg,
                        tuple(traversed),
                    )
            previous_sample = sample

        dist = math.dist(origin, last_valid)
        return RayHit(
            last_valid,
            False,
            dist,
            angle_deg,
            tuple(traversed),
        )
