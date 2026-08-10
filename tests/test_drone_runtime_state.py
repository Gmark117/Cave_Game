import threading
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from agents.drone_runtime_state import DroneRuntimeState
from navigation.frontier_clusters import FrontierCluster


def make_cluster(cluster_id: int, representative: tuple[int, int]):
    x, y = representative
    return FrontierCluster(
        id=cluster_id,
        cells=frozenset({representative}),
        bounds=(x, y, x + 1, y + 1),
        representative=representative,
        expected_gain=1,
        known_by=frozenset({0}),
        lifecycle="active",
        first_seen_revision=1,
        last_seen_revision=1,
    )


class DroneRuntimeStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = DroneRuntimeState(
            start_position=(2, 2),
            cave=np.zeros((16, 16), dtype=np.uint8),
            direction=90,
            frontier_rebuild_cooldown=0.25,
        )

    def test_snapshot_is_immutable_and_detached(self) -> None:
        self.state.replace_frontier_clusters((
            make_cluster(4, (4, 4)),
            make_cluster(5, (5, 5)),
        ))
        self.state.begin_exploration(direction=45)
        self.state.set_ray_points([(6, 6), (7, 7)])
        initial = self.state.snapshot()

        self.state.replace_frontier_clusters((make_cluster(8, (8, 8)),))
        self.state.set_ray_points([(9, 9)])
        self.state.toggle_path()

        self.assertEqual(initial.frontiers, ((4, 4), (5, 5)))
        self.assertEqual(initial.frontier_cluster_ids, (4, 5))
        self.assertEqual(initial.ray_points, ((6, 6), (7, 7)))
        self.assertTrue(initial.show_path)
        with self.assertRaises(FrozenInstanceError):
            initial.position = (0, 0)

    def test_cluster_replacement_is_authoritative_and_id_aligned(self) -> None:
        self.state.replace_frontier_clusters((
            make_cluster(8, (8, 8)),
            make_cluster(3, (3, 4)),
        ))

        first = self.state.snapshot()

        self.assertEqual(first.frontier_cluster_ids, (3, 8))
        self.assertEqual(first.frontiers, ((3, 4), (8, 8)))

        self.state.replace_frontier_clusters((make_cluster(12, (9, 1)),))
        replaced = self.state.snapshot()

        self.assertEqual(replaced.frontier_cluster_ids, (12,))
        self.assertEqual(replaced.frontiers, ((9, 1),))

    def test_reconciliation_reports_registry_retirements_and_clears_snapshot(self) -> None:
        self.state.replace_frontier_clusters((
            make_cluster(3, (3, 4)),
            make_cluster(8, (8, 8)),
        ))

        retired = self.state.reconcile_frontier_clusters((
            make_cluster(8, (9, 8)),
        ))

        self.assertEqual(retired, (3,))
        self.assertEqual(self.state.snapshot().frontier_cluster_ids, (8,))
        self.assertEqual(self.state.snapshot().frontiers, ((9, 8),))

    def test_begin_exploration_does_not_mutate_authoritative_clusters(self) -> None:
        self.state.replace_frontier_clusters((make_cluster(7, (6, 6)),))

        self.state.begin_exploration(direction=45)
        snapshot = self.state.snapshot()

        self.assertTrue(snapshot.explored)
        self.assertEqual(snapshot.direction, 45)
        self.assertEqual(snapshot.direction_history, (45,))
        self.assertEqual(snapshot.frontier_cluster_ids, (7,))
        self.assertEqual(snapshot.frontiers, ((6, 6),))

    def test_move_updates_position_heading_and_path_atomically(self) -> None:
        self.state.move_to((4, 2))

        snapshot = self.state.snapshot()

        self.assertEqual(snapshot.position, (4, 2))
        self.assertEqual(snapshot.path_history[-1], (4, 2))
        self.assertAlmostEqual(snapshot.heading_deg, 90.0)

    def test_concurrent_snapshots_never_observe_torn_movement_state(self) -> None:
        start = threading.Barrier(2)
        writer_done = threading.Event()
        errors = []

        def writer() -> None:
            start.wait()
            for x in range(3, 15):
                self.state.move_to((x, 2))
            writer_done.set()

        def reader() -> None:
            start.wait()
            while not writer_done.is_set():
                snapshot = self.state.snapshot()
                if snapshot.path_history[-1] != snapshot.position:
                    errors.append(snapshot)
                    return

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)
        writer_thread.start()
        reader_thread.start()
        writer_thread.join(2.0)
        reader_thread.join(2.0)

        self.assertFalse(writer_thread.is_alive())
        self.assertFalse(reader_thread.is_alive())
        self.assertEqual(errors, [])

    def test_frontier_rebuild_reservation_is_atomic(self) -> None:
        self.assertTrue(self.state.reserve_frontier_rebuild(1.0))
        self.assertFalse(self.state.reserve_frontier_rebuild(1.1))
        self.assertTrue(self.state.reserve_frontier_rebuild(1.25))

    def test_empty_frontiers_do_not_implicitly_start_homing(self) -> None:
        self.state.begin_exploration(direction=45)

        done, homing = self.state.evaluate_mission_state()

        self.assertFalse(done)
        self.assertFalse(homing)

        self.state.start_returning_home()
        done, homing = self.state.evaluate_mission_state()

        self.assertFalse(done)
        self.assertTrue(homing)


if __name__ == "__main__":
    unittest.main()
