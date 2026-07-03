import unittest
import threading

import numpy as np

from SlamMap import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    SlamMap,
    SlamSnapshot,
)
from VisionSensor import RayHit


def make_snapshot(
    shape: tuple[int, int],
    cells=(),
    points=(),
) -> SlamSnapshot:
    occupancy = np.full(shape, UNKNOWN, dtype=np.int8)
    confidence = np.zeros(shape, dtype=np.float32)
    for x, y, value, cell_confidence in cells:
        occupancy[y, x] = value
        confidence[y, x] = cell_confidence
    return SlamSnapshot(
        occupancy=occupancy,
        confidence=confidence,
        point_cloud=tuple(points),
    )


class SlamMapTests(unittest.TestCase):
    def test_ray_hit_marks_free_route_and_occupied_endpoint(self) -> None:
        slam = SlamMap(5, 5)

        changed = slam.update_from_rays(
            (0, 2),
            [RayHit(end=(4, 2), hit=True, distance=4.0, angle_deg=90.0)],
        )
        snapshot = slam.snapshot()

        self.assertTrue(changed)
        self.assertEqual(snapshot.version, 1)
        self.assertTrue(np.all(snapshot.occupancy[2, :4] == FREE))
        self.assertEqual(int(snapshot.occupancy[2, 4]), OCCUPIED)
        self.assertGreater(float(snapshot.confidence[2, 4]), 0.0)
        self.assertIn((4, 2), snapshot.point_cloud)

    def test_snapshot_is_detached_and_arrays_are_not_public(self) -> None:
        slam = SlamMap(2, 2)
        slam.merge_from(
            make_snapshot((2, 2), cells=[(0, 0, FREE, 0.8)])
        )

        snapshot = slam.snapshot()
        snapshot.occupancy[0, 0] = OCCUPIED
        snapshot.confidence[0, 0] = 0.0
        live = slam.snapshot()

        self.assertEqual(int(live.occupancy[0, 0]), FREE)
        self.assertAlmostEqual(float(live.confidence[0, 0]), 0.8)
        self.assertFalse(hasattr(slam, "occupancy"))
        self.assertFalse(hasattr(slam, "confidence"))
        self.assertFalse(hasattr(slam, "point_cloud"))
        self.assertFalse(hasattr(slam, "dirty"))

    def test_merge_uses_higher_confidence_and_preserves_stronger_cells(self) -> None:
        target = SlamMap(2, 2)
        target.merge_from(
            make_snapshot(
                (2, 2),
                cells=[
                    (0, 0, FREE, 0.8),
                    (1, 1, FREE, 0.2),
                ],
            )
        )
        source = make_snapshot(
            (2, 2),
            cells=[
                (0, 0, OCCUPIED, 0.4),
                (1, 1, OCCUPIED, 0.9),
            ],
            points=[(1, 1)],
        )

        changed = target.merge_from(source)
        snapshot = target.snapshot()

        self.assertTrue(changed)
        self.assertEqual(int(snapshot.occupancy[0, 0]), FREE)
        self.assertEqual(int(snapshot.occupancy[1, 1]), OCCUPIED)
        self.assertAlmostEqual(float(snapshot.confidence[1, 1]), 0.9)
        self.assertIn((1, 1), snapshot.point_cloud)

    def test_point_cloud_is_unique_bounded_and_snapshot_tail_is_recent_first(
        self,
    ) -> None:
        slam = SlamMap(2, 4, max_points=3)
        for point in [(0, 0), (1, 0), (1, 0), (2, 0), (3, 0)]:
            slam.merge_from(
                make_snapshot((2, 4), points=[point])
            )

        full = slam.snapshot()
        recent = slam.snapshot(point_limit=2)

        self.assertEqual(
            full.point_cloud,
            ((1, 0), (2, 0), (3, 0)),
        )
        self.assertEqual(recent.point_cloud, ((3, 0), (2, 0)))

    def test_version_changes_only_when_owned_state_changes(self) -> None:
        slam = SlamMap(2, 2)
        source = make_snapshot(
            (2, 2),
            cells=[(0, 0, FREE, 0.8)],
            points=[(0, 0)],
        )

        self.assertTrue(slam.merge_from(source))
        version = slam.version
        self.assertFalse(slam.merge_from(source))
        self.assertEqual(slam.version, version)
        self.assertFalse(slam.has_changed_since(version))
        self.assertTrue(slam.has_changed_since(version - 1))

    def test_known_query_treats_out_of_bounds_as_unavailable(self) -> None:
        slam = SlamMap(2, 2)
        self.assertTrue(slam.is_known(-1, 0))
        self.assertFalse(slam.is_known(0, 0))
        self.assertEqual(int(slam.snapshot().occupancy[0, 0]), UNKNOWN)

    def test_snapshot_validates_array_shapes(self) -> None:
        with self.assertRaises(ValueError):
            SlamSnapshot(
                occupancy=np.zeros((2, 2), dtype=np.int8),
                confidence=np.zeros((2, 3), dtype=np.float32),
            )

    def test_concurrent_updates_and_snapshots_remain_consistent(self) -> None:
        slam = SlamMap(8, 8)
        start = threading.Barrier(2)
        errors = []
        ray = RayHit(
            end=(7, 4),
            hit=True,
            distance=7.0,
            angle_deg=90.0,
        )

        def update() -> None:
            try:
                start.wait()
                for _ in range(100):
                    slam.update_from_rays((0, 4), [ray])
            except BaseException as exc:
                errors.append(exc)

        def read() -> None:
            try:
                start.wait()
                for _ in range(100):
                    snapshot = slam.snapshot(point_limit=10)
                    self.assertEqual(
                        snapshot.occupancy.shape,
                        snapshot.confidence.shape,
                    )
                    self.assertTrue(
                        np.all(
                            (snapshot.confidence >= 0.0)
                            & (snapshot.confidence <= 1.0)
                        )
                    )
            except BaseException as exc:
                errors.append(exc)

        updater = threading.Thread(target=update)
        reader = threading.Thread(target=read)
        updater.start()
        reader.start()
        updater.join(2.0)
        reader.join(2.0)

        self.assertFalse(updater.is_alive())
        self.assertFalse(reader.is_alive())
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
