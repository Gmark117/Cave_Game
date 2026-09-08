"""Thread-safe terrain roughness and confidence knowledge."""

import threading
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


TerrainSample = Tuple[int, int, float, float]


@dataclass(frozen=True)
class TerrainSnapshot:
    """Detached terrain arrays suitable for sharing or path planning."""

    roughness: np.ndarray
    confidence: np.ndarray
    version: int = 0

    def __post_init__(self) -> None:
        """Validate that paired terrain arrays describe the same grid."""
        if self.roughness.shape != self.confidence.shape:
            raise ValueError("Terrain snapshot arrays must have the same shape")


def fuse_terrain_samples(
    roughness: np.ndarray,
    confidence: np.ndarray,
    cave_map: np.ndarray,
    samples: Iterable[TerrainSample],
) -> bool:
    """Fuse one scan in bulk using confidence-weighted cell aggregates."""
    materialized = tuple(samples)
    if not materialized:
        return False
    observations = np.asarray(materialized, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[1] != 4:
        raise ValueError("terrain samples must contain x, y, value, confidence")

    x = observations[:, 0].astype(np.intp)
    y = observations[:, 1].astype(np.intp)
    valid = (
        (y >= 0)
        & (y < roughness.shape[0])
        & (x >= 0)
        & (x < roughness.shape[1])
    )
    if not np.any(valid):
        return False
    x = x[valid]
    y = y[valid]
    observed_roughness = np.clip(observations[valid, 2], 0.0, 1.0)
    observed_confidence = np.clip(
        observations[valid, 3],
        0.05,
        1.0,
    )

    floor = np.asarray(cave_map)[y, x] == 0
    if not np.any(floor):
        return False
    x = x[floor]
    y = y[floor]
    observed_roughness = observed_roughness[floor]
    observed_confidence = observed_confidence[floor]

    width = roughness.shape[1]
    flat_coordinates = y * width + x
    unique_coordinates, inverse = np.unique(
        flat_coordinates,
        return_inverse=True,
    )
    scan_confidence = np.bincount(
        inverse,
        weights=observed_confidence,
    )
    scan_weighted_roughness = np.bincount(
        inverse,
        weights=observed_roughness * observed_confidence,
    )
    unique_y, unique_x = np.divmod(unique_coordinates, width)

    previous_confidence = confidence[unique_y, unique_x].astype(
        np.float64,
    )
    scan_roughness = scan_weighted_roughness / np.maximum(
        scan_confidence,
        1e-12,
    )
    previous_roughness = np.where(
        previous_confidence > 0.0,
        roughness[unique_y, unique_x].astype(np.float64),
        scan_roughness,
    )
    total_confidence = previous_confidence + scan_confidence
    roughness[unique_y, unique_x] = (
        (previous_roughness * previous_confidence)
        + scan_weighted_roughness
    ) / np.maximum(total_confidence, 1e-12)
    confidence[unique_y, unique_x] = np.minimum(
        1.0,
        total_confidence,
    )
    return True


class TerrainKnowledge:
    """Own terrain arrays, synchronization, fusion, and merge rules."""

    def __init__(
        self,
        cave_map: np.ndarray,
        roughness: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
    ) -> None:
        """Create a synchronized terrain map shaped like the cave matrix."""
        cave = np.asarray(cave_map, dtype=np.uint8)
        if cave.ndim != 2:
            raise ValueError("Terrain cave map must be two-dimensional")

        self.cave_map = cave
        self.floor_mask = cave == 0
        self.floor_cells = int(np.count_nonzero(self.floor_mask))
        self.lock = threading.RLock()
        self._version = 0
        self.roughness = self._initial_array(
            roughness,
            fill=-1.0,
            name="roughness",
        )
        self.confidence = self._initial_array(
            confidence,
            fill=0.0,
            name="confidence",
        )

    def _initial_array(
        self,
        values: np.ndarray | None,
        fill: float,
        name: str,
    ) -> np.ndarray:
        """Create or validate one terrain array."""
        if values is None:
            return np.full(self.cave_map.shape, fill, dtype=np.float32)

        array = np.asarray(values, dtype=np.float32)
        if array.shape != self.cave_map.shape:
            raise ValueError(
                f"Terrain {name} shape {array.shape} does not match "
                f"cave shape {self.cave_map.shape}"
            )
        return array.copy()

    def record_samples(self, samples: Iterable[TerrainSample]) -> bool:
        """Fuse sensor samples into this knowledge map."""
        with self.lock:
            changed = fuse_terrain_samples(
                self.roughness,
                self.confidence,
                self.cave_map,
                samples,
            )
            if changed:
                self._version += 1
            return changed

    @property
    def version(self) -> int:
        """Return the monotonic terrain revision without copying map arrays."""
        with self.lock:
            return self._version

    def snapshot(self) -> TerrainSnapshot:
        """Return detached copies of roughness and confidence."""
        with self.lock:
            return TerrainSnapshot(
                self.roughness.copy(),
                self.confidence.copy(),
                version=self._version,
            )

    def merge_from(self, source: TerrainSnapshot) -> bool:
        """Merge a snapshot into this map using confidence-weighted values."""
        source_roughness = np.asarray(source.roughness, dtype=np.float32)
        source_confidence = np.asarray(source.confidence, dtype=np.float32)
        if source_roughness.shape != source_confidence.shape:
            raise ValueError("Terrain snapshot arrays must have the same shape")

        with self.lock:
            height = min(self.roughness.shape[0], source_roughness.shape[0])
            width = min(self.roughness.shape[1], source_roughness.shape[1])
            if height <= 0 or width <= 0:
                return False

            target_roughness = self.roughness[:height, :width]
            target_confidence = self.confidence[:height, :width]
            incoming_roughness = np.clip(
                source_roughness[:height, :width],
                0.0,
                1.0,
            )
            incoming_confidence = np.clip(
                source_confidence[:height, :width],
                0.0,
                1.0,
            )
            valid = (
                self.floor_mask[:height, :width]
                & (incoming_confidence > 0.0)
            )
            if not np.any(valid):
                return False

            # Merge only known incoming floor cells. Unknown target cells take
            # the incoming roughness as their baseline; known cells are averaged
            # by confidence.
            target_conf_values = target_confidence[valid]
            incoming_conf_values = incoming_confidence[valid]
            incoming_rough_values = incoming_roughness[valid]
            target_rough_values = target_roughness[valid]
            base_target = np.where(
                target_conf_values > 0.0,
                target_rough_values,
                incoming_rough_values,
            )
            total_confidence = target_conf_values + incoming_conf_values
            target_roughness[valid] = (
                (base_target * target_conf_values)
                + (incoming_rough_values * incoming_conf_values)
            ) / np.maximum(total_confidence, 1e-6)
            target_confidence[valid] = np.minimum(1.0, total_confidence)
            self._version += 1
            return True

    def known_mask(self, threshold: float = 0.0) -> np.ndarray:
        """Return a detached mask of known floor cells."""
        with self.lock:
            return (
                self.floor_mask
                & (self.confidence > float(threshold))
            ).copy()

    def explored_ratio(self, threshold: float = 0.0) -> float:
        """Return the fraction of floor cells known above `threshold`."""
        if self.floor_cells <= 0:
            return 0.0
        known_cells = int(np.count_nonzero(self.known_mask(threshold)))
        return known_cells / self.floor_cells
