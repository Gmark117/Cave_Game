import os
import unittest

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from generation.terrain_roughness_generator import TerrainRoughnessGenerator


class TerrainRoughnessGeneratorTests(unittest.TestCase):
    def test_terrain_roughness_is_floor_only_bounded_and_reproducible(self) -> None:
        cave = np.ones((8, 8), dtype=np.uint8)
        cave[1:7, 1:7] = 0
        generator = TerrainRoughnessGenerator()

        first = generator.generate(cave, np.random.default_rng(9))
        second = generator.generate(cave.copy(), np.random.default_rng(9))

        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(first[cave == 1] == 0.0))
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
