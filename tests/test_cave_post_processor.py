import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from generation.cave_post_processor import CavePostProcessor


class CavePostProcessorTests(unittest.TestCase):
    def test_wall_transition_noise_is_disabled(self) -> None:
        cave = np.ones((9, 9), dtype=np.uint8)
        cave[2:7, 2:7] = 0
        processor = CavePostProcessor()

        with patch(
            "generation.cave_post_processor.cv2.medianBlur",
            side_effect=lambda image, kernel: image,
        ):
            with patch(
                "generation.cave_post_processor.add_wall_transition_noise",
                side_effect=AssertionError("noise should be disabled"),
            ) as noise:
                result = processor.process(cave, 9, 9, 5, (4, 1, 3))

        noise.assert_not_called()
        np.testing.assert_array_equal(result, cave)

    def test_process_returns_binary_map_with_noise_disabled(self) -> None:
        cave = np.ones((9, 9), dtype=np.uint8)
        cave[2:7, 2:7] = 0
        processor = CavePostProcessor()

        result = processor.process(cave, 9, 9, 5, (4, 4, 3))

        self.assertEqual(result.shape, cave.shape)
        self.assertTrue(np.all((result == 0) | (result == 1)))


if __name__ == "__main__":
    unittest.main()
