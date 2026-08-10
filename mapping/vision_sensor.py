"""Dense vision-cone sensing plus sparse rays for terrain telemetry."""

from dataclasses import dataclass
from typing import List, Tuple
import math

import numpy as np

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


@dataclass(frozen=True)
class VisionScan:
    """One gap-free visibility result and its sparse presentation rays."""

    ray_hits: Tuple[RayHit, ...]
    free_cells: Tuple[Tuple[int, int], ...]
    occupied_cells: Tuple[Tuple[int, int], ...]


class VisionSensor:
    """Observe every visible cone cell while retaining bounded sample rays."""

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
        self._map_array = np.asarray(map_matrix, dtype=np.uint8)
        self.fov_deg = float(fov_deg)
        self.num_rays = max(1, int(num_rays))
        self.step = max(1, int(step))
        map_range = int(math.hypot(self.map_w, self.map_h))
        self.max_range = max(
            1,
            int(max_range) if max_range is not None else map_range,
        )
        relative_axis = np.arange(
            -self.max_range,
            self.max_range + 1,
            dtype=np.float32,
        )
        self._relative_y, self._relative_x = np.meshgrid(
            relative_axis,
            relative_axis,
            indexing="ij",
        )
        self._relative_distance = np.hypot(
            self._relative_x,
            self._relative_y,
        )
        self._relative_angle = np.degrees(np.arctan2(
            self._relative_x,
            -self._relative_y,
        ))

    def cast_cone(self, origin: Tuple[float, float], heading_deg: float) -> List[RayHit]:
        """Cast the sparse rays used by overlays and terrain sampling."""
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

    def scan_cone(
        self,
        origin: Tuple[float, float],
        heading_deg: float,
    ) -> VisionScan:
        """Return gap-free visible cells independently of sparse sample rays."""
        ray_hits = tuple(self.cast_cone(origin, heading_deg))
        if self.map_w <= 0 or self.map_h <= 0:
            return VisionScan(ray_hits, (), ())

        cone_cells = self._dense_visible_cells(origin, heading_deg)
        free_cells = tuple(
            point for point in cone_cells
            if not wall_hit(self.map_matrix, point)
        )
        occupied_cells = tuple(
            point for point in cone_cells
            if wall_hit(self.map_matrix, point)
        )
        return VisionScan(ray_hits, free_cells, occupied_cells)

    def _dense_visible_cells(
        self,
        origin: Tuple[float, float],
        heading_deg: float,
    ) -> Tuple[Tuple[int, int], ...]:
        """Rasterize every cell within the collision-bounded polar depth."""
        center_x = int(round(origin[0]))
        center_y = int(round(origin[1]))
        if not (0 <= center_x < self.map_w and 0 <= center_y < self.map_h):
            return ()

        field_of_view = min(360.0, max(0.0, self.fov_deg))
        dense_ray_count = max(
            2,
            int(math.ceil(
                math.radians(field_of_view) * self.max_range * 2.0
            )) + 1,
        )
        angles = np.deg2rad(np.linspace(
            float(heading_deg) - field_of_view / 2.0,
            float(heading_deg) + field_of_view / 2.0,
            dense_ray_count,
            dtype=np.float32,
        ))
        lengths = np.arange(
            self.max_range + 1,
            dtype=np.float32,
        )
        ray_x = np.rint(
            float(origin[0]) + np.sin(angles)[:, None] * lengths
        ).astype(np.int32)
        ray_y = np.rint(
            float(origin[1]) - np.cos(angles)[:, None] * lengths
        ).astype(np.int32)
        in_bounds = (
            (ray_x >= 0)
            & (ray_x < self.map_w)
            & (ray_y >= 0)
            & (ray_y < self.map_h)
        )
        safe_x = np.clip(ray_x, 0, self.map_w - 1)
        safe_y = np.clip(ray_y, 0, self.map_h - 1)
        blocked = (~in_bounds) | (self._map_array[safe_y, safe_x] == 1)
        has_block = np.any(blocked, axis=1)
        first_block = np.argmax(blocked, axis=1)
        endpoint_index = np.where(
            has_block,
            first_block,
            self.max_range,
        )
        rows = np.arange(dense_ray_count)
        endpoint_out_of_bounds = ~in_bounds[rows, endpoint_index]
        endpoint_index = np.where(
            endpoint_out_of_bounds,
            np.maximum(0, endpoint_index - 1),
            endpoint_index,
        )
        endpoint_x = ray_x[rows, endpoint_index]
        endpoint_y = ray_y[rows, endpoint_index]

        left = max(0, center_x - self.max_range)
        right = min(self.map_w, center_x + self.max_range + 1)
        top = max(0, center_y - self.max_range)
        bottom = min(self.map_h, center_y + self.max_range + 1)
        grid_left = center_x - self.max_range
        grid_top = center_y - self.max_range
        x_slice = slice(left - grid_left, right - grid_left)
        y_slice = slice(top - grid_top, bottom - grid_top)
        delta_x = self._relative_x[y_slice, x_slice]
        delta_y = self._relative_y[y_slice, x_slice]
        distance = self._relative_distance[y_slice, x_slice]
        cell_angle = self._relative_angle[y_slice, x_slice]
        fractional_x = float(origin[0]) - center_x
        fractional_y = float(origin[1]) - center_y
        if abs(fractional_x) > 1e-6 or abs(fractional_y) > 1e-6:
            delta_x = delta_x - fractional_x
            delta_y = delta_y - fractional_y
            distance = np.hypot(delta_x, delta_y)
            cell_angle = np.degrees(np.arctan2(delta_x, -delta_y))
        inside = distance <= float(self.max_range) + 1e-6
        angle_delta = (
            cell_angle - float(heading_deg) + 180.0
        ) % 360.0 - 180.0
        if field_of_view < 360.0:
            inside &= (
                (distance == 0.0)
                | (np.abs(angle_delta) <= field_of_view / 2.0 + 1e-6)
            )

        angular_position = (
            (angle_delta + field_of_view / 2.0)
            / max(field_of_view, 1e-6)
            * (dense_ray_count - 1)
        )
        lower = np.clip(
            np.floor(angular_position).astype(np.int32),
            0,
            dense_ray_count - 1,
        )
        upper = np.clip(lower + 1, 0, dense_ray_count - 1)
        endpoint_distance = np.hypot(
            endpoint_x.astype(np.float32) - float(origin[0]),
            endpoint_y.astype(np.float32) - float(origin[1]),
        )
        visible_depth = np.minimum(
            endpoint_distance[lower],
            endpoint_distance[upper],
        )
        inside &= distance <= visible_depth + math.sqrt(0.5) + 1e-6
        visible_local_y, visible_local_x = np.nonzero(inside)
        visible_x = visible_local_x + left
        visible_y = visible_local_y + top
        return tuple(
            (int(x), int(y))
            for x, y in zip(visible_x, visible_y)
        )

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
