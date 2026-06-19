import unittest

import numpy as np

from SlamMap import FREE, OCCUPIED, UNKNOWN, SlamMap
from VisionSensor import RayHit


class SlamMapTests(unittest.TestCase):
    def test_ray_hit_marks_free_route_and_occupied_endpoint(self) -> None:
        slam = SlamMap(5, 5)
        slam.dirty = False

        slam.update_from_rays(
            (0, 2),
            [RayHit(end=(4, 2), hit=True, distance=4.0, angle_deg=90.0)],
        )

        self.assertTrue(slam.dirty)
        self.assertTrue(np.all(slam.occupancy[2, :4] == FREE))
        self.assertEqual(int(slam.occupancy[2, 4]), OCCUPIED)
        self.assertGreater(float(slam.confidence[2, 4]), 0.0)
        self.assertIn((4, 2), slam.point_cloud)

    def test_merge_uses_higher_confidence_and_preserves_stronger_cells(self) -> None:
        target = SlamMap(2, 2)
        target.occupancy[0, 0] = FREE
        target.confidence[0, 0] = 0.8
        target.occupancy[1, 1] = FREE
        target.confidence[1, 1] = 0.2

        source = SlamMap(2, 2)
        source.occupancy[0, 0] = OCCUPIED
        source.confidence[0, 0] = 0.4
        source.occupancy[1, 1] = OCCUPIED
        source.confidence[1, 1] = 0.9
        source._add_point((1, 1))

        target.merge_from(source)

        self.assertEqual(int(target.occupancy[0, 0]), FREE)
        self.assertEqual(int(target.occupancy[1, 1]), OCCUPIED)
        self.assertAlmostEqual(float(target.confidence[1, 1]), 0.9)
        self.assertIn((1, 1), target.point_cloud)

    def test_point_cloud_is_unique_bounded_and_recent_first(self) -> None:
        slam = SlamMap(2, 4, max_points=3)
        for point in [(0, 0), (1, 0), (1, 0), (2, 0), (3, 0)]:
            slam._add_point(point)

        self.assertEqual(list(slam.point_cloud), [(1, 0), (2, 0), (3, 0)])
        self.assertEqual(slam.recent_points(2), [(3, 0), (2, 0)])

    def test_known_query_treats_out_of_bounds_as_unavailable(self) -> None:
        slam = SlamMap(2, 2)
        self.assertTrue(slam.is_known(-1, 0))
        self.assertFalse(slam.is_known(0, 0))
        self.assertEqual(int(slam.occupancy[0, 0]), UNKNOWN)


if __name__ == "__main__":
    unittest.main()
