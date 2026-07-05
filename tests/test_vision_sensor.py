import unittest

import numpy as np

from mapping.vision_sensor import VisionSensor


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
        self.assertEqual(hit.points, ((3, 5), (3, 4), (3, 3), (3, 2), (3, 1)))

    def test_single_ray_stops_at_map_edge_when_clear(self) -> None:
        cave = np.zeros((5, 5), dtype=np.uint8)
        sensor = VisionSensor(cave, num_rays=1, step=1)

        hit = sensor.cast_cone((2, 2), heading_deg=90)[0]

        self.assertFalse(hit.hit)
        self.assertEqual(hit.end, (4, 2))
        self.assertEqual(hit.points, ((2, 2), (3, 2), (4, 2)))

    def test_single_ray_honors_max_range_when_clear(self) -> None:
        cave = np.zeros((9, 9), dtype=np.uint8)
        sensor = VisionSensor(cave, num_rays=1, step=1, max_range=2)

        hit = sensor.cast_cone((4, 4), heading_deg=90)[0]

        self.assertFalse(hit.hit)
        self.assertEqual(hit.end, (6, 4))
        self.assertAlmostEqual(hit.distance, 2.0)
        self.assertEqual(hit.points, ((4, 4), (5, 4), (6, 4)))

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
