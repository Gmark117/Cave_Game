import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from RoughnessSampler import RoughnessSampler


class RoughnessSamplerTests(unittest.TestCase):
    def test_samples_floor_cells_and_stops_before_wall(self) -> None:
        cave = np.zeros((1, 5), dtype=np.uint8)
        cave[0, 3] = 1
        terrain = np.array([[0.1, 0.2, 0.3, 0.9, 0.5]], dtype=np.float32)
        sampler = RoughnessSampler(terrain, cave)
        hit = SimpleNamespace(end=(4, 0))

        with patch("RoughnessSampler.np.random.uniform", return_value=0.0):
            samples = sampler.sample_from_rays((0, 0), [hit], step=1)

        self.assertEqual(
            [(x, y) for x, y, _, _ in samples],
            [(0, 0), (1, 0), (2, 0)],
        )
        self.assertAlmostEqual(samples[0][2], 0.1)
        self.assertGreater(samples[0][3], samples[-1][3])
        self.assertGreaterEqual(samples[-1][3], 0.2)

    def test_ignores_hits_without_valid_endpoints(self) -> None:
        sampler = RoughnessSampler(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.uint8),
        )

        samples = sampler.sample_from_rays(
            (0, 0),
            [SimpleNamespace(), SimpleNamespace(end=(9, 9))],
        )

        self.assertEqual(samples, [])

    def test_empty_map_returns_no_samples(self) -> None:
        sampler = RoughnessSampler(np.empty((0, 0)), [])
        self.assertEqual(sampler.sample_from_rays((0, 0), []), [])


if __name__ == "__main__":
    unittest.main()
