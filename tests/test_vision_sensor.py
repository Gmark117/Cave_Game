import math
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

    def test_sparse_ray_cannot_skip_wall_between_samples(self) -> None:
        cave = np.zeros((7, 7), dtype=np.uint8)
        cave[3, 3] = 1
        sensor = VisionSensor(cave, num_rays=1, step=2, max_range=4)

        hit = sensor.cast_cone((3, 4), heading_deg=0)[0]

        self.assertTrue(hit.hit)
        self.assertEqual(hit.end, (3, 3))
        self.assertEqual(hit.points, ((3, 4), (3, 3)))

    def test_cone_spans_requested_field_of_view(self) -> None:
        sensor = VisionSensor(
            np.zeros((5, 5), dtype=np.uint8),
            fov_deg=60,
            num_rays=3,
            step=1,
        )

        hits = sensor.cast_cone((2, 2), heading_deg=90)

        self.assertEqual([hit.angle_deg for hit in hits], [60.0, 90.0, 120.0])

    def test_dense_scan_observes_every_cell_center_inside_empty_cone(self) -> None:
        cave = np.zeros((11, 11), dtype=np.uint8)
        sensor = VisionSensor(
            cave,
            fov_deg=60,
            num_rays=2,
            step=2,
            max_range=4,
        )

        scan = sensor.scan_cone((5, 5), heading_deg=0)
        expected = set()
        for y in range(11):
            for x in range(11):
                dx = x - 5
                dy = y - 5
                if (dx * dx) + (dy * dy) > 16:
                    continue
                if dx == 0 and dy == 0:
                    expected.add((x, y))
                    continue
                angle = math.degrees(math.atan2(dx, -dy)) % 360.0
                difference = (angle + 180.0) % 360.0 - 180.0
                if abs(difference) <= 30.0:
                    expected.add((x, y))

        self.assertEqual(set(scan.free_cells), expected)
        self.assertEqual(scan.occupied_cells, ())
        self.assertEqual(len(scan.ray_hits), 2)
        self.assertIn((5, 2), scan.free_cells)
        self.assertNotIn(
            (5, 2),
            {point for hit in scan.ray_hits for point in hit.points},
        )

    def test_dense_scan_stops_visibility_behind_wall(self) -> None:
        cave = np.zeros((9, 9), dtype=np.uint8)
        cave[3, 4] = 1
        sensor = VisionSensor(
            cave,
            fov_deg=60,
            num_rays=3,
            max_range=6,
        )

        scan = sensor.scan_cone((4, 6), heading_deg=0)

        self.assertIn((4, 3), scan.occupied_cells)
        self.assertNotIn((4, 2), scan.free_cells)
        self.assertNotIn((4, 1), scan.free_cells)

    def test_empty_map_returns_no_rays(self) -> None:
        self.assertEqual(VisionSensor([]).cast_cone((0, 0), 0), [])


if __name__ == "__main__":
    unittest.main()
