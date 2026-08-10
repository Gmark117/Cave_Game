"""Thread-safe hybrid SLAM state with detached snapshots."""

from collections import deque
from dataclasses import dataclass
from itertools import islice
import math
import threading
from typing import Deque, Iterable, Optional, Tuple

import numpy as np
from mapping.vision_sensor import RayHit

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
    origin: Point = (0, 0)
    full_shape: Tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Validate that occupancy and confidence describe the same grid."""
        if self.occupancy.ndim != 2 or self.confidence.ndim != 2:
            raise ValueError("SLAM snapshot arrays must be two-dimensional")
        if self.occupancy.shape != self.confidence.shape:
            raise ValueError("SLAM snapshot arrays must have the same shape")
        if self.full_shape is not None and (
            self.full_shape[0] < self.occupancy.shape[0]
            or self.full_shape[1] < self.occupancy.shape[1]
        ):
            raise ValueError("full SLAM shape cannot be smaller than its window")


@dataclass(frozen=True)
class SlamProgressSnapshot:
    """Detached cumulative progress counters for one local SLAM map."""

    version: int = 0
    completed_scan_sequence: int = 0
    sensor_newly_known_cells: int = 0
    sensor_confidence_gain: float = 0.0
    shared_newly_known_cells: int = 0
    shared_confidence_gain: float = 0.0
    collision_observations: int = 0
    collision_newly_known_cells: int = 0
    collision_confidence_gain: float = 0.0

    @property
    def newly_known_cells(self) -> int:
        """Return cumulative newly known cells from every evidence source."""
        return (
            self.sensor_newly_known_cells
            + self.shared_newly_known_cells
            + self.collision_newly_known_cells
        )

    @property
    def confidence_gain(self) -> float:
        """Return cumulative confidence gain from every evidence source."""
        return (
            self.sensor_confidence_gain
            + self.shared_confidence_gain
            + self.collision_confidence_gain
        )


@dataclass
class _ProgressDelta:
    """Mutable operation-local progress before it reaches public counters."""

    newly_known_cells: int = 0
    confidence_gain: float = 0.0


class SlamMap:
    """Own synchronized SLAM arrays, point observations, and merge rules."""

    def __init__(self, map_h: int, map_w: int, max_points: int = 6000) -> None:
        """Create an unknown occupancy grid and bounded point cloud."""
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
        self._completed_scan_sequence = 0
        self._sensor_newly_known_cells = 0
        self._sensor_confidence_gain = 0.0
        self._shared_newly_known_cells = 0
        self._shared_confidence_gain = 0.0
        self._collision_observations = 0
        self._collision_newly_known_cells = 0
        self._collision_confidence_gain = 0.0

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
                full_shape=self._occupancy.shape,
            )

    def snapshot_window(
        self,
        bounds: tuple[int, int, int, int],
    ) -> SlamSnapshot:
        """Return one bounded detached belief window from an atomic version."""
        with self._lock:
            return self._snapshot_window_locked(bounds)

    def try_snapshot_window(
        self,
        bounds: tuple[int, int, int, int],
    ) -> SlamSnapshot | None:
        """Return a bounded window immediately, or ``None`` if SLAM is busy."""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._snapshot_window_locked(bounds)
        finally:
            self._lock.release()

    def _snapshot_window_locked(
        self,
        bounds: tuple[int, int, int, int],
    ) -> SlamSnapshot:
        """Copy a normalized window while the caller owns ``_lock``."""
        left, top, right, bottom = (int(value) for value in bounds)
        height, width = self._occupancy.shape
        left = min(width, max(0, left))
        right = min(width, max(left, right))
        top = min(height, max(0, top))
        bottom = min(height, max(top, bottom))
        return SlamSnapshot(
            occupancy=self._occupancy[top:bottom, left:right].copy(),
            confidence=self._confidence[top:bottom, left:right].copy(),
            version=self._version,
            origin=(left, top),
            full_shape=self._occupancy.shape,
        )

    def progress_snapshot(self) -> SlamProgressSnapshot:
        """Return an immutable, atomic view of cumulative mapping progress."""
        with self._lock:
            return SlamProgressSnapshot(
                version=self._version,
                completed_scan_sequence=self._completed_scan_sequence,
                sensor_newly_known_cells=self._sensor_newly_known_cells,
                sensor_confidence_gain=self._sensor_confidence_gain,
                shared_newly_known_cells=self._shared_newly_known_cells,
                shared_confidence_gain=self._shared_confidence_gain,
                collision_observations=self._collision_observations,
                collision_newly_known_cells=(
                    self._collision_newly_known_cells
                ),
                collision_confidence_gain=self._collision_confidence_gain,
            )

    def has_changed_since(self, version: int) -> bool:
        """Return whether this map has advanced beyond `version`."""
        with self._lock:
            return self._version != int(version)

    def update_from_rays(
        self,
        origin: Tuple[float, float],
        ray_hits: Iterable[RayHit],
    ) -> bool:
        """Update occupancy and point state from rays, returning whether it changed."""
        _ = origin

        with self._lock:
            updated = False
            progress = _ProgressDelta()
            for hit in ray_hits:
                ex, ey = int(hit.end[0]), int(hit.end[1])
                if (
                    ex < 0
                    or ey < 0
                    or ex >= self._occupancy.shape[1]
                    or ey >= self._occupancy.shape[0]
                ):
                    continue

                points = list(hit.points)
                if not points:
                    continue

                distance = float(hit.distance)
                base_confidence = max(
                    0.15,
                    1.0 - (distance / self.max_range),
                )

                if hit.hit:
                    updated |= self._mark_points(
                        points[:-1],
                        FREE,
                        base_confidence,
                        progress,
                    )
                    updated |= self._mark_points(
                        [points[-1]],
                        OCCUPIED,
                        min(1.0, base_confidence + 0.25),
                        progress,
                    )
                    updated |= self._add_point(points[-1])
                else:
                    updated |= self._mark_points(
                        points,
                        FREE,
                        base_confidence,
                        progress,
                    )

            if updated:
                self._version += 1
            self._completed_scan_sequence += 1
            self._sensor_newly_known_cells += progress.newly_known_cells
            self._sensor_confidence_gain += progress.confidence_gain
            return updated

    def update_from_observations(
        self,
        origin: Tuple[float, float],
        *,
        free_cells: Iterable[Point],
        occupied_cells: Iterable[Point],
    ) -> bool:
        """Update SLAM from one gap-free visibility observation."""
        free = tuple(dict.fromkeys(
            (int(point[0]), int(point[1])) for point in free_cells
        ))
        occupied = tuple(dict.fromkeys(
            (int(point[0]), int(point[1])) for point in occupied_cells
        ))
        all_cells = (*free, *occupied)
        farthest = max(
            (math.dist(origin, point) for point in all_cells),
            default=0.0,
        )
        base_confidence = max(
            0.15,
            1.0 - (farthest / self.max_range),
        )

        with self._lock:
            progress = _ProgressDelta()
            updated = self._mark_points(
                free,
                FREE,
                base_confidence,
                progress,
            )
            updated |= self._mark_points(
                occupied,
                OCCUPIED,
                min(1.0, base_confidence + 0.25),
                progress,
            )
            for point in occupied:
                updated |= self._add_point(point)
            if updated:
                self._version += 1
            self._completed_scan_sequence += 1
            self._sensor_newly_known_cells += progress.newly_known_cells
            self._sensor_confidence_gain += progress.confidence_gain
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
                target_occupancy = self._occupancy[:height, :width]
                incoming_occupancy = source_occupancy[:height, :width]
                occupied_ties = (
                    (incoming_occupancy == OCCUPIED)
                    & (target_occupancy != OCCUPIED)
                    & (incoming_confidence >= target_confidence - 1e-4)
                )
                replace_cells = higher_confidence | occupied_ties
                if np.any(replace_cells):
                    previous_occupancy = target_occupancy[replace_cells].copy()
                    previous_confidence = target_confidence[replace_cells].copy()
                    replacement_occupancy = incoming_occupancy[replace_cells]
                    replacement_confidence = np.maximum(
                        previous_confidence,
                        incoming_confidence[replace_cells],
                    )
                    target_occupancy[replace_cells] = (
                        replacement_occupancy
                    )
                    target_confidence[replace_cells] = replacement_confidence
                    self._shared_newly_known_cells += int(
                        np.count_nonzero(
                            (previous_occupancy == UNKNOWN)
                            & (replacement_occupancy != UNKNOWN)
                        )
                    )
                    self._shared_confidence_gain += float(
                        np.sum(
                            replacement_confidence - previous_confidence,
                            dtype=np.float64,
                        )
                    )
                    updated = True

            for point in source.point_cloud:
                updated |= self._add_point(point)

            if updated:
                self._version += 1
            return updated

    def record_collision(
        self,
        point: Point,
        confidence: float = 1.0,
    ) -> bool:
        """Record a movement-confirmed obstacle in the local occupancy map."""
        normalized = (int(point[0]), int(point[1]))
        with self._lock:
            progress = _ProgressDelta()
            updated = self._mark_points(
                (normalized,),
                OCCUPIED,
                min(1.0, max(0.0, float(confidence))),
                progress,
            )
            updated |= self._add_point(normalized)
            if updated:
                self._version += 1
            self._collision_observations += 1
            self._collision_newly_known_cells += progress.newly_known_cells
            self._collision_confidence_gain += progress.confidence_gain
            return updated

    def _snapshot_points(self, point_limit: Optional[int]) -> Tuple[Point, ...]:
        """Return all points or only the newest ``point_limit`` points."""
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
        progress: _ProgressDelta | None = None,
    ) -> bool:
        """Apply one occupancy label to line points when confidence improves."""
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
            previous_occupancy = int(self._occupancy[y, x])
            if confidence > previous_confidence + 1e-4:
                self._occupancy[y, x] = occupancy_value
                self._confidence[y, x] = min(1.0, confidence)
                updated = True
            elif (
                occupancy_value == OCCUPIED
                and self._occupancy[y, x] != OCCUPIED
                and confidence >= previous_confidence - 1e-4
            ):
                # At equal confidence, the collision-safe interpretation wins.
                # This also lets a direct wall hit repair a saturated false-free
                # observation left behind by an earlier sparse ray.
                self._occupancy[y, x] = OCCUPIED
                self._confidence[y, x] = min(
                    1.0,
                    max(previous_confidence, confidence),
                )
                updated = True
            elif self._occupancy[y, x] == occupancy_value:
                boosted = min(
                    1.0,
                    previous_confidence + confidence * 0.15,
                )
                if boosted > previous_confidence + 1e-4:
                    self._confidence[y, x] = boosted
                    updated = True
            if progress is not None:
                current_occupancy = int(self._occupancy[y, x])
                current_confidence = float(self._confidence[y, x])
                if (
                    previous_occupancy == UNKNOWN
                    and current_occupancy != UNKNOWN
                ):
                    progress.newly_known_cells += 1
                progress.confidence_gain += max(
                    0.0,
                    current_confidence - previous_confidence,
                )
        return updated

    def _add_point(self, point: Point) -> bool:
        """Append a unique occupied point while enforcing the point limit."""
        normalized = (int(point[0]), int(point[1]))
        if normalized in self._point_set or self.max_points == 0:
            return False

        self._point_cloud.append(normalized)
        self._point_set.add(normalized)
        if len(self._point_cloud) > self.max_points:
            old = self._point_cloud.popleft()
            self._point_set.discard(old)
        return True
