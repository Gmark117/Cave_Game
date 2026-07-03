"""Thread-safe hybrid SLAM state with detached snapshots."""

from collections import deque
from dataclasses import dataclass
from itertools import islice
import math
import threading
from typing import Deque, Iterable, Optional, Tuple

import numpy as np

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

Point = Tuple[int, int]


@dataclass(frozen=True)
class SlamSnapshot:
    """Detached occupancy, confidence, point-cloud, and version state."""

    occupancy: np.ndarray
    confidence: np.ndarray
    point_cloud: Tuple[Point, ...] = ()
    version: int = 0

    def __post_init__(self) -> None:
        if self.occupancy.ndim != 2 or self.confidence.ndim != 2:
            raise ValueError("SLAM snapshot arrays must be two-dimensional")
        if self.occupancy.shape != self.confidence.shape:
            raise ValueError("SLAM snapshot arrays must have the same shape")


class SlamMap:
    """Own synchronized SLAM arrays, point observations, and merge rules."""

    def __init__(self, map_h: int, map_w: int, max_points: int = 6000) -> None:
        self._lock = threading.RLock()
        self._occupancy = np.full(
            (map_h, map_w),
            UNKNOWN,
            dtype=np.int8,
        )
        self._confidence = np.zeros(
            (map_h, map_w),
            dtype=np.float32,
        )
        self._point_cloud: Deque[Point] = deque()
        self._point_set: set[Point] = set()
        self._version = 0

        self.max_range = max(1.0, float(math.hypot(map_w, map_h)))
        self.max_points = max(0, int(max_points))

    @property
    def version(self) -> int:
        """Return the current map version."""
        with self._lock:
            return self._version

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the occupancy-grid shape."""
        return self._occupancy.shape

    def snapshot(self, point_limit: Optional[int] = None) -> SlamSnapshot:
        """Return detached arrays and recent points from one atomic version."""
        with self._lock:
            points = self._snapshot_points(point_limit)
            return SlamSnapshot(
                occupancy=self._occupancy.copy(),
                confidence=self._confidence.copy(),
                point_cloud=points,
                version=self._version,
            )

    def has_changed_since(self, version: int) -> bool:
        """Return whether this map has advanced beyond `version`."""
        with self._lock:
            return self._version != int(version)

    def update_from_rays(
        self,
        origin: Tuple[float, float],
        ray_hits: Iterable[object],
    ) -> bool:
        """Update occupancy and point state from rays, returning whether it changed."""
        ox = int(round(origin[0]))
        oy = int(round(origin[1]))

        with self._lock:
            updated = False
            for hit in ray_hits:
                end = getattr(hit, "end", None)
                if end is None:
                    continue
                ex, ey = int(end[0]), int(end[1])
                if (
                    ex < 0
                    or ey < 0
                    or ex >= self._occupancy.shape[1]
                    or ey >= self._occupancy.shape[0]
                ):
                    continue

                points = self._line_points(ox, oy, ex, ey)
                if not points:
                    continue

                distance = float(
                    getattr(
                        hit,
                        "distance",
                        math.dist((ox, oy), (ex, ey)),
                    )
                )
                base_confidence = max(
                    0.15,
                    1.0 - (distance / self.max_range),
                )

                if getattr(hit, "hit", False):
                    updated |= self._mark_points(
                        points[:-1],
                        FREE,
                        base_confidence,
                    )
                    updated |= self._mark_points(
                        [points[-1]],
                        OCCUPIED,
                        min(1.0, base_confidence + 0.25),
                    )
                    updated |= self._add_point(points[-1])
                else:
                    updated |= self._mark_points(
                        points,
                        FREE,
                        base_confidence,
                    )

            if updated:
                self._version += 1
            return updated

    def merge_from(self, source: SlamSnapshot) -> bool:
        """Merge a detached snapshot using confidence dominance."""
        source_occupancy = np.asarray(source.occupancy, dtype=np.int8)
        source_confidence = np.asarray(source.confidence, dtype=np.float32)
        if source_occupancy.shape != source_confidence.shape:
            raise ValueError("SLAM snapshot arrays must have the same shape")

        with self._lock:
            height = min(
                self._occupancy.shape[0],
                source_occupancy.shape[0],
            )
            width = min(
                self._occupancy.shape[1],
                source_occupancy.shape[1],
            )

            updated = False
            if height > 0 and width > 0:
                target_confidence = self._confidence[:height, :width]
                incoming_confidence = source_confidence[:height, :width]
                higher_confidence = incoming_confidence > target_confidence
                if np.any(higher_confidence):
                    target_occupancy = self._occupancy[:height, :width]
                    incoming_occupancy = source_occupancy[:height, :width]
                    target_occupancy[higher_confidence] = (
                        incoming_occupancy[higher_confidence]
                    )
                    target_confidence[higher_confidence] = (
                        incoming_confidence[higher_confidence]
                    )
                    updated = True

            for point in source.point_cloud:
                updated |= self._add_point(point)

            if updated:
                self._version += 1
            return updated

    def is_known(self, x: int, y: int, threshold: float = 0.6) -> bool:
        """Return whether a cell is known; out-of-bounds cells are unavailable."""
        with self._lock:
            if (
                y < 0
                or y >= self._confidence.shape[0]
                or x < 0
                or x >= self._confidence.shape[1]
            ):
                return True
            return float(self._confidence[y, x]) >= threshold

    def _snapshot_points(self, point_limit: Optional[int]) -> Tuple[Point, ...]:
        if point_limit is None:
            return tuple(self._point_cloud)

        limit = max(0, int(point_limit))
        if limit == 0 or not self._point_cloud:
            return ()
        recent = islice(reversed(self._point_cloud), limit)
        return tuple(recent)

    def _mark_points(
        self,
        points: Iterable[Point],
        occupancy_value: int,
        confidence: float,
    ) -> bool:
        updated = False
        for x, y in points:
            if (
                y < 0
                or y >= self._confidence.shape[0]
                or x < 0
                or x >= self._confidence.shape[1]
            ):
                continue

            previous_confidence = float(self._confidence[y, x])
            if confidence > previous_confidence + 1e-4:
                self._occupancy[y, x] = occupancy_value
                self._confidence[y, x] = min(1.0, confidence)
                updated = True
            elif self._occupancy[y, x] == occupancy_value:
                boosted = min(
                    1.0,
                    previous_confidence + confidence * 0.15,
                )
                if boosted > previous_confidence + 1e-4:
                    self._confidence[y, x] = boosted
                    updated = True
        return updated

    def _add_point(self, point: Point) -> bool:
        normalized = (int(point[0]), int(point[1]))
        if normalized in self._point_set or self.max_points == 0:
            return False

        self._point_cloud.append(normalized)
        self._point_set.add(normalized)
        if len(self._point_cloud) > self.max_points:
            old = self._point_cloud.popleft()
            self._point_set.discard(old)
        return True

    @staticmethod
    def _line_points(x0: int, y0: int, x1: int, y1: int) -> list[Point]:
        """Return integer points along a line using Bresenham's algorithm."""
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
