import unittest
from unittest.mock import patch

import numpy as np

from mapping.roughness_sampler import RoughnessSampler
from mapping.vision_sensor import RayHit


class RoughnessSamplerTests(unittest.TestCase):
    def test_samples_floor_cells_and_stops_before_wall(self) -> None:
        cave = np.zeros((1, 5), dtype=np.uint8)
        cave[0, 3] = 1
        terrain = np.array([[0.1, 0.2, 0.3, 0.9, 0.5]], dtype=np.float32)
        sampler = RoughnessSampler(terrain, cave)
        hit = RayHit(
            end=(4, 0),
            hit=True,
            distance=4.0,
            angle_deg=90.0,
            points=((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        )

        with patch(
            "mapping.roughness_sampler.np.random.uniform",
            return_value=0.0,
        ):
            samples = sampler.sample_from_rays((0, 0), [hit], step=1)

        self.assertEqual(
            [(x, y) for x, y, _, _ in samples],
            [(0, 0), (1, 0), (2, 0)],
        )
        self.assertAlmostEqual(samples[0][2], 0.1)
        self.assertGreater(samples[0][3], samples[-1][3])
        self.assertGreaterEqual(samples[-1][3], 0.2)

    def test_uses_supplied_ray_points(self) -> None:
        cave = np.zeros((1, 5), dtype=np.uint8)
        terrain = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
        sampler = RoughnessSampler(terrain, cave)
        hit = RayHit(
            end=(4, 0),
            hit=False,
            distance=4.0,
            angle_deg=90.0,
            points=((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        )

        with patch(
            "mapping.roughness_sampler.np.random.uniform",
            return_value=0.0,
        ):
            samples = sampler.sample_from_rays((0, 0), [hit], step=1)

        self.assertEqual(
            [(x, y) for x, y, _, _ in samples],
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        )

    def test_default_sampling_keeps_every_second_visible_ray_cell(self) -> None:
        cave = np.zeros((1, 5), dtype=np.uint8)
        sampler = RoughnessSampler(
            np.full((1, 5), 0.4, dtype=np.float32),
            cave,
        )
        hit = RayHit(
            end=(4, 0),
            hit=False,
            distance=4.0,
            angle_deg=90.0,
            points=((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        )

        with patch(
            "mapping.roughness_sampler.np.random.uniform",
            return_value=0.0,
        ):
            samples = sampler.sample_from_rays((0, 0), [hit])

        self.assertEqual(
            [(x, y) for x, y, _, _ in samples],
            [(0, 0), (2, 0), (4, 0)],
        )

    def test_ignores_hits_with_out_of_bounds_endpoints(self) -> None:
        sampler = RoughnessSampler(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.uint8),
        )

        samples = sampler.sample_from_rays(
            (0, 0),
            [
                RayHit(
                    end=(9, 9),
                    hit=False,
                    distance=9.0,
                    angle_deg=90.0,
                    points=((0, 0), (9, 9)),
                )
            ],
        )

        self.assertEqual(samples, [])

    def test_empty_map_returns_no_samples(self) -> None:
        sampler = RoughnessSampler(np.empty((0, 0)), [])
        self.assertEqual(sampler.sample_from_rays((0, 0), []), [])


if __name__ == "__main__":
    unittest.main()
