"""Terrain roughness synthesis for generated cave floors."""

import cv2
import numpy as np


class TerrainRoughnessGenerator:
    """Create bounded floor-only roughness values for cave traversal."""

    def generate(
        self,
        bin_map: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return a float32 roughness map in `[0, 1]`."""

        height, width = bin_map.shape
        floor_mask = (bin_map == 0).astype(np.uint8)

        base_noise = rng.random((height, width), dtype=np.float32)
        base_noise = cv2.GaussianBlur(base_noise, (0, 0), sigmaX=18, sigmaY=18)
        base_noise = cv2.normalize(base_noise, None, 0.0, 1.0, cv2.NORM_MINMAX)

        cluster_noise = rng.random((height, width), dtype=np.float32)
        cluster_noise = cv2.GaussianBlur(
            cluster_noise,
            (0, 0),
            sigmaX=6,
            sigmaY=6,
        )
        cluster_noise = cv2.normalize(
            cluster_noise,
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX,
        )

        wall_bias = np.zeros((height, width), dtype=np.float32)
        if np.any(floor_mask):
            wall_distance = cv2.distanceTransform(floor_mask, cv2.DIST_L2, 5)
            max_distance = float(wall_distance.max()) or 1.0
            wall_bias = 1.0 - np.clip(wall_distance / max_distance, 0.0, 1.0)

        roughness = (
            (0.45 * base_noise)
            + (0.35 * wall_bias)
            + (0.20 * cluster_noise)
        )
        roughness = np.clip(roughness, 0.0, 1.0).astype(np.float32)
        roughness *= floor_mask.astype(np.float32)
        return roughness
