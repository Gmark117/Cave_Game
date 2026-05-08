"""Hybrid SLAM map state: occupancy grid + sparse point cloud."""

from typing import Iterable, List, Optional, Tuple
import math

import numpy as np

UNKNOWN = -1
FREE = 0
OCCUPIED = 1


class SlamMap:
    """Tracks local SLAM occupancy and sparse point observations."""

    def __init__(self, map_h: int, map_w: int, max_points: int = 6000) -> None:
        self.occupancy = np.full((map_h, map_w), UNKNOWN, dtype=np.int8)
        self.confidence = np.zeros((map_h, map_w), dtype=np.float32)
        self.max_range = max(1.0, float(math.hypot(map_w, map_h)))
        self.dirty = True

        self.point_cloud: List[Tuple[int, int]] = []
        self._point_set = set()
        self.max_points = max_points

    def update_from_rays(self, origin: Tuple[float, float], ray_hits: Iterable[object]) -> None:
        """Update occupancy and point cloud from ray hits."""
        ox = int(round(origin[0]))
        oy = int(round(origin[1]))
        updated = False

        for hit in ray_hits:
            end = getattr(hit, 'end', None)
            if end is None:
                continue
            ex, ey = int(end[0]), int(end[1])
            if ex < 0 or ey < 0 or ex >= self.occupancy.shape[1] or ey >= self.occupancy.shape[0]:
                continue

            points = self._line_points(ox, oy, ex, ey)
            if not points:
                continue

            dist = float(getattr(hit, 'distance', math.dist((ox, oy), (ex, ey))))
            base_conf = max(0.15, 1.0 - (dist / self.max_range))

            if getattr(hit, 'hit', False):
                free_points = points[:-1]
                hit_point = points[-1]
                updated |= self._mark_points(free_points, FREE, base_conf)
                updated |= self._mark_points([hit_point], OCCUPIED, min(1.0, base_conf + 0.25))
                self._add_point(hit_point)
            else:
                updated |= self._mark_points(points, FREE, base_conf)

        if updated:
            self.dirty = True

    def merge_from(self, other: 'SlamMap') -> None:
        """Merge another SlamMap into this one using confidence dominance."""
        if other is None:
            return

        h = min(self.occupancy.shape[0], other.occupancy.shape[0])
        w = min(self.occupancy.shape[1], other.occupancy.shape[1])
        if h <= 0 or w <= 0:
            return

        target_conf = self.confidence[:h, :w]
        source_conf = other.confidence[:h, :w]
        source_occ = other.occupancy[:h, :w]

        higher_conf = source_conf > target_conf
        if np.any(higher_conf):
            self.occupancy[:h, :w][higher_conf] = source_occ[higher_conf]
            target_conf[higher_conf] = source_conf[higher_conf]
            self.dirty = True

        for point in other.point_cloud:
            self._add_point(point)

    def merge_from_arrays(
        self,
        occupancy: np.ndarray,
        confidence: np.ndarray,
        point_cloud: Optional[List[Tuple[int, int]]] = None
    ) -> None:
        """Merge occupancy/confidence arrays into this map."""
        if occupancy is None or confidence is None:
            return

        h = min(self.occupancy.shape[0], occupancy.shape[0], confidence.shape[0])
        w = min(self.occupancy.shape[1], occupancy.shape[1], confidence.shape[1])
        if h <= 0 or w <= 0:
            return

        target_conf = self.confidence[:h, :w]
        source_conf = confidence[:h, :w]
        source_occ = occupancy[:h, :w]

        higher_conf = source_conf > target_conf
        if np.any(higher_conf):
            self.occupancy[:h, :w][higher_conf] = source_occ[higher_conf]
            target_conf[higher_conf] = source_conf[higher_conf]
            self.dirty = True

        if point_cloud:
            for point in point_cloud:
                self._add_point(point)

    def is_known(self, x: int, y: int, threshold: float = 0.6) -> bool:
        if y < 0 or y >= self.confidence.shape[0] or x < 0 or x >= self.confidence.shape[1]:
            return True
        return float(self.confidence[y, x]) >= threshold

    def _mark_points(self, points: Iterable[Tuple[int, int]], occ_value: int, conf: float) -> bool:
        updated = False
        for x, y in points:
            if y < 0 or y >= self.confidence.shape[0] or x < 0 or x >= self.confidence.shape[1]:
                continue
            prev_conf = float(self.confidence[y, x])
            if conf > prev_conf + 1e-4:
                self.occupancy[y, x] = occ_value
                self.confidence[y, x] = min(1.0, conf)
                updated = True
            elif self.occupancy[y, x] == occ_value:
                boosted = min(1.0, prev_conf + conf * 0.15)
                if boosted > prev_conf + 1e-4:
                    self.confidence[y, x] = boosted
                    updated = True
        return updated

    def _add_point(self, point: Tuple[int, int]) -> None:
        if point in self._point_set:
            return
        self.point_cloud.append(point)
        self._point_set.add(point)
        if len(self.point_cloud) > self.max_points:
            old = self.point_cloud.pop(0)
            self._point_set.discard(old)

    def _line_points(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Return integer points along a line using Bresenham's algorithm."""
        points: List[Tuple[int, int]] = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points
