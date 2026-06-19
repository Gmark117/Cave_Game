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

    def __post_init__(self) -> None:
        if self.roughness.shape != self.confidence.shape:
            raise ValueError("Terrain snapshot arrays must have the same shape")


def fuse_terrain_samples(
    roughness: np.ndarray,
    confidence: np.ndarray,
    cave_map: np.ndarray,
    samples: Iterable[TerrainSample],
) -> bool:
    """Fuse observations into supplied arrays using confidence weighting."""
    map_updated = False
    for x, y, observed_roughness, observed_confidence in samples:
        xi = int(x)
        yi = int(y)
        if (
            yi < 0
            or yi >= roughness.shape[0]
            or xi < 0
            or xi >= roughness.shape[1]
        ):
            continue
        if cave_map[yi, xi] != 0:
            continue

        obs_conf = float(np.clip(observed_confidence, 0.05, 1.0))
        obs_rough = float(np.clip(observed_roughness, 0.0, 1.0))
        previous_conf = float(confidence[yi, xi])
        previous_rough = (
            float(roughness[yi, xi]) if previous_conf > 0.0 else obs_rough
        )
        total_conf = previous_conf + obs_conf

        roughness[yi, xi] = (
            (previous_rough * previous_conf) + (obs_rough * obs_conf)
        ) / total_conf
        confidence[yi, xi] = min(1.0, total_conf)
        map_updated = True

    return map_updated


class TerrainKnowledge:
    """Own terrain arrays, synchronization, fusion, and merge rules."""

    def __init__(
        self,
        cave_map: np.ndarray,
        roughness: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
    ) -> None:
        cave = np.asarray(cave_map, dtype=np.uint8)
        if cave.ndim != 2:
            raise ValueError("Terrain cave map must be two-dimensional")

        self.cave_map = cave
        self.floor_mask = cave == 0
        self.floor_cells = int(np.count_nonzero(self.floor_mask))
        self.lock = threading.RLock()
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
            return fuse_terrain_samples(
                self.roughness,
                self.confidence,
                self.cave_map,
                samples,
            )

    def snapshot(self) -> TerrainSnapshot:
        """Return detached copies of roughness and confidence."""
        with self.lock:
            return TerrainSnapshot(
                self.roughness.copy(),
                self.confidence.copy(),
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
