import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from generation.cave_post_processor import CavePostProcessor


class CavePostProcessorTests(unittest.TestCase):
    def test_noise_failure_keeps_cleaned_map_output(self) -> None:
        cave = np.ones((9, 9), dtype=np.uint8)
        cave[2:7, 2:7] = 0
        processor = CavePostProcessor()

        with self.assertLogs(
            "generation.cave_post_processor",
            level="WARNING",
        ) as logs:
            with patch(
                "generation.cave_post_processor.add_wall_transition_noise",
                side_effect=ValueError("bad noise"),
            ):
                result = processor.process(cave, 9, 9, 5, (4, 4, 3))

        self.assertEqual(result.shape, cave.shape)
        self.assertTrue(np.all((result == 0) | (result == 1)))
        self.assertIn("Wall-transition noise pass skipped", logs.output[0])


if __name__ == "__main__":
    unittest.main()
