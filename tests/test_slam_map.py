import unittest
import threading

import numpy as np

from mapping.slam_map import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    SlamMap,
    SlamProgressSnapshot,
    SlamSnapshot,
)
from mapping.vision_sensor import RayHit


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
    def test_snapshot_window_is_bounded_detached_and_keeps_global_coordinates(
        self,
    ) -> None:
        slam = SlamMap(20, 30)
        slam.merge_from(make_snapshot(
            (20, 30),
            cells=[(8, 7, FREE, 0.8), (11, 9, OCCUPIED, 0.9)],
        ))

        window = slam.snapshot_window((7, 6, 12, 10))

        self.assertEqual(window.origin, (7, 6))
        self.assertEqual(window.full_shape, (20, 30))
        self.assertEqual(window.occupancy.shape, (4, 5))
        self.assertEqual(int(window.occupancy[1, 1]), FREE)
        self.assertEqual(int(window.occupancy[3, 4]), OCCUPIED)
        version = window.version
        window.occupancy[:, :] = UNKNOWN
        current = slam.snapshot_window((7, 6, 12, 10))
        self.assertEqual(current.version, version)
        self.assertEqual(int(current.occupancy[1, 1]), FREE)

    def test_try_snapshot_window_never_waits_for_busy_writer(self) -> None:
        slam = SlamMap(20, 30)
        locked = threading.Event()
        release = threading.Event()

        def hold_writer_lock() -> None:
            with slam._lock:
                locked.set()
                release.wait(timeout=1.0)

        worker = threading.Thread(target=hold_writer_lock)
        worker.start()
        self.assertTrue(locked.wait(timeout=1.0))
        try:
            self.assertIsNone(slam.try_snapshot_window((7, 6, 12, 10)))
        finally:
            release.set()
            worker.join(timeout=1.0)

        self.assertIsNotNone(slam.try_snapshot_window((7, 6, 12, 10)))

    def test_progress_snapshot_tracks_sensor_gain_and_completed_scans(
        self,
    ) -> None:
        slam = SlamMap(5, 5)
        ray = RayHit(
            end=(4, 2),
            hit=True,
            distance=4.0,
            angle_deg=90.0,
            points=((0, 2), (1, 2), (2, 2), (3, 2), (4, 2)),
        )

        self.assertTrue(slam.update_from_rays((0, 2), [ray]))
        progress = slam.progress_snapshot()

        self.assertIsInstance(progress, SlamProgressSnapshot)
        self.assertEqual(progress.version, 1)
        self.assertEqual(progress.completed_scan_sequence, 1)
        self.assertEqual(progress.sensor_newly_known_cells, 5)
        self.assertGreater(progress.sensor_confidence_gain, 0.0)
        self.assertEqual(progress.shared_newly_known_cells, 0)
        self.assertEqual(progress.collision_observations, 0)
        self.assertEqual(progress.newly_known_cells, 5)
        self.assertAlmostEqual(
            progress.confidence_gain,
            progress.sensor_confidence_gain,
        )

    def test_zero_gain_scan_advances_completed_scan_sequence(self) -> None:
        slam = SlamMap(3, 3)

        self.assertFalse(slam.update_from_rays((1, 1), []))
        first = slam.progress_snapshot()
        self.assertFalse(slam.update_from_rays((1, 1), []))
        second = slam.progress_snapshot()

        self.assertEqual(first.completed_scan_sequence, 1)
        self.assertEqual(second.completed_scan_sequence, 2)
        self.assertEqual(second.version, 0)
        self.assertEqual(second.sensor_newly_known_cells, 0)
        self.assertEqual(second.sensor_confidence_gain, 0.0)

    def test_progress_snapshot_separates_shared_and_collision_gain(self) -> None:
        slam = SlamMap(3, 3)

        self.assertTrue(
            slam.merge_from(
                make_snapshot(
                    (3, 3),
                    cells=[(0, 0, FREE, 0.4), (1, 0, OCCUPIED, 0.8)],
                )
            )
        )
        self.assertTrue(slam.record_collision((2, 2), confidence=0.9))
        progress = slam.progress_snapshot()

        self.assertEqual(progress.completed_scan_sequence, 0)
        self.assertEqual(progress.sensor_newly_known_cells, 0)
        self.assertEqual(progress.shared_newly_known_cells, 2)
        self.assertAlmostEqual(progress.shared_confidence_gain, 1.2)
        self.assertEqual(progress.collision_observations, 1)
        self.assertEqual(progress.collision_newly_known_cells, 1)
        self.assertAlmostEqual(progress.collision_confidence_gain, 0.9)
        self.assertEqual(progress.newly_known_cells, 3)

    def test_ray_hit_marks_free_route_and_occupied_endpoint(self) -> None:
        slam = SlamMap(5, 5)

        changed = slam.update_from_rays(
            (0, 2),
            [
                RayHit(
                    end=(4, 2),
                    hit=True,
                    distance=4.0,
                    angle_deg=90.0,
                    points=((0, 2), (1, 2), (2, 2), (3, 2), (4, 2)),
                )
            ],
        )
        snapshot = slam.snapshot()

        self.assertTrue(changed)
        self.assertEqual(snapshot.version, 1)
        self.assertTrue(np.all(snapshot.occupancy[2, :4] == FREE))
        self.assertEqual(int(snapshot.occupancy[2, 4]), OCCUPIED)
        self.assertGreater(float(snapshot.confidence[2, 4]), 0.0)
        self.assertIn((4, 2), snapshot.point_cloud)

    def test_dense_observations_mark_all_seen_cells_in_one_scan(self) -> None:
        slam = SlamMap(5, 5)

        changed = slam.update_from_observations(
            (2, 4),
            free_cells=((1, 4), (2, 4), (3, 4), (2, 3), (1, 3)),
            occupied_cells=((2, 2),),
        )
        snapshot = slam.snapshot()
        progress = slam.progress_snapshot()

        self.assertTrue(changed)
        self.assertEqual(progress.completed_scan_sequence, 1)
        self.assertEqual(progress.sensor_newly_known_cells, 6)
        self.assertTrue(np.all(snapshot.occupancy[3, 1:3] == FREE))
        self.assertEqual(int(snapshot.occupancy[2, 2]), OCCUPIED)
        self.assertIn((2, 2), snapshot.point_cloud)

    def test_ray_hit_uses_supplied_points(self) -> None:
        slam = SlamMap(5, 5)
        ray = RayHit(
            end=(4, 2),
            hit=True,
            distance=4.0,
            angle_deg=90.0,
            points=((0, 2), (1, 2), (2, 2), (3, 2), (4, 2)),
        )

        changed = slam.update_from_rays((0, 2), [ray])

        self.assertTrue(changed)
        snapshot = slam.snapshot()
        self.assertTrue(np.all(snapshot.occupancy[2, :4] == FREE))
        self.assertEqual(int(snapshot.occupancy[2, 4]), OCCUPIED)

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

    def test_collision_repairs_equal_confidence_false_free_cell(self) -> None:
        slam = SlamMap(3, 3)
        slam.merge_from(
            make_snapshot((3, 3), cells=[(1, 1, FREE, 1.0)])
        )

        changed = slam.record_collision((1, 1))
        snapshot = slam.snapshot()

        self.assertTrue(changed)
        self.assertEqual(int(snapshot.occupancy[1, 1]), OCCUPIED)
        self.assertEqual(float(snapshot.confidence[1, 1]), 1.0)
        self.assertIn((1, 1), snapshot.point_cloud)

    def test_merge_prefers_occupied_cell_at_equal_confidence(self) -> None:
        target = SlamMap(2, 2)
        target.merge_from(
            make_snapshot((2, 2), cells=[(0, 0, FREE, 1.0)])
        )

        changed = target.merge_from(
            make_snapshot((2, 2), cells=[(0, 0, OCCUPIED, 1.0)])
        )
        snapshot = target.snapshot(point_limit=0)

        self.assertTrue(changed)
        self.assertEqual(int(snapshot.occupancy[0, 0]), OCCUPIED)
        self.assertEqual(float(snapshot.confidence[0, 0]), 1.0)

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
            points=(
                (0, 4),
                (1, 4),
                (2, 4),
                (3, 4),
                (4, 4),
                (5, 4),
                (6, 4),
                (7, 4),
            ),
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
