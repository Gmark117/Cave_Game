import unittest

import numpy as np

from VisionSensor import VisionSensor


class VisionSensorTests(unittest.TestCase):
    def test_single_ray_reports_first_wall(self) -> None:
        cave = np.zeros((7, 7), dtype=np.uint8)
        cave[1, 3] = 1
        sensor = VisionSensor(cave, num_rays=1, step=1)

        hit = sensor.cast_cone((3, 5), heading_deg=0)[0]

        self.assertTrue(hit.hit)
        self.assertEqual(hit.end, (3, 1))
        self.assertEqual(hit.angle_deg, 0)
        self.assertAlmostEqual(hit.distance, 4.0)

    def test_single_ray_stops_at_map_edge_when_clear(self) -> None:
        cave = np.zeros((5, 5), dtype=np.uint8)
        sensor = VisionSensor(cave, num_rays=1, step=1)

        hit = sensor.cast_cone((2, 2), heading_deg=90)[0]

        self.assertFalse(hit.hit)
        self.assertEqual(hit.end, (4, 2))

    def test_cone_spans_requested_field_of_view(self) -> None:
        sensor = VisionSensor(
            np.zeros((5, 5), dtype=np.uint8),
            fov_deg=60,
            num_rays=3,
            step=1,
        )

        hits = sensor.cast_cone((2, 2), heading_deg=90)

        self.assertEqual([hit.angle_deg for hit in hits], [60.0, 90.0, 120.0])

    def test_empty_map_returns_no_rays(self) -> None:
        self.assertEqual(VisionSensor([]).cast_cone((0, 0), 0), [])


if __name__ == "__main__":
    unittest.main()
