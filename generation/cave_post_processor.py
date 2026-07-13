"""OpenCV post-processing for raw cave maps."""

import cv2
import numpy as np

from asset_config.mapgen import MapGen
from generation.mapgen_helpers import (
    add_wall_transition_noise,
    remove_hermit_caves,
)


class CavePostProcessor:
    """Smooth and clean a raw worm-carved cave map."""

    def process(
        self,
        raw_map: np.ndarray,
        width: int,
        height: int,
        seed: int,
        worm_inputs: tuple[int, int, int],
    ) -> np.ndarray:
        """Return the final binary cave layout."""

        # Worm carving creates a rough binary image. Median filtering smooths
        # jagged tunnels before isolated disconnected caves are removed.
        kernel_dim = int(
            max(1, (worm_inputs[1] - MapGen.MEDIAN_FILTER_REDUCTION) | 1)
        )
        raw = raw_map.astype("uint8")
        smoothed = cv2.medianBlur(raw, kernel_dim)
        cleaned = remove_hermit_caves(smoothed)
        stalac = cv2.bitwise_or(raw, cleaned)
        _ = add_wall_transition_noise, width, height, seed
        # Wall-transition noise is disabled while exploration behavior is
        # re-baselined. The helper remains available for future tuning.
        return cv2.medianBlur(stalac, MapGen.BLUR_KERNEL_FINAL)
