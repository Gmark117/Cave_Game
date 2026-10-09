from concurrent.futures import Future
import unittest
from unittest.mock import patch

import numpy as np

from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from navigation.highway import HighwayBuildResult, HighwayService, build_highway_graph
from navigation.highway_worker import HighwayWorker


class Executor:
    def __init__(self):
        self.jobs = []
        self.stopped = False

    def submit(self, callback, slam, settings):
        future = Future()
        self.jobs.append((future, slam))
        return future

    def shutdown(self, **kwargs):
        self.stopped = True


class HighwayWorkerTests(unittest.TestCase):
    def setUp(self):
        self.clock = 0.0
        self.executor = Executor()
        self.worker = HighwayWorker(dict(confidence_threshold=0.6, macro_cell_size=4,
                                         maximum_build_ms=250, maximum_access_distance=20),
                                    executor=self.executor, now=lambda: self.clock)
        self.addCleanup(self.worker.shutdown)
        self.occupancy = np.full((32, 48), FREE, np.int8)
        self.slam = self.snapshot(1)
        self.result = build_highway_graph(self.slam, confidence_threshold=0.6,
                                         macro_cell_size=4, maximum_build_ms=1000,
                                         maximum_connector_expansions=256,
                                         maximum_access_distance=20)
        self.assertIsNotNone(self.result.snapshot)

    def snapshot(self, version, origin=(0, 0)):
        return SlamSnapshot(self.occupancy.copy(), np.ones(self.occupancy.shape, np.float32),
                            version=version, origin=origin, full_shape=self.occupancy.shape)

    def finish(self, result=None):
        self.executor.jobs[-1][0].set_result(result or self.result)
        self.clock += self.worker.MINIMUM_INTERVAL
        return self.worker.poll()

    def test_uploads_coalesce_and_unchanged_geometry_rebases_version(self):
        self.worker.request(self.slam, None)
        for version in range(2, 8):
            self.worker.request(self.snapshot(version), None)
        self.assertEqual(len(self.executor.jobs), 1)
        result = self.finish()
        self.assertEqual(result.snapshot.version, 7)
        self.assertEqual((result.source_version, result.build_kind), (1, "full"))
        self.assertEqual(len(self.executor.jobs), 1)

    def test_changed_wall_cannot_publish_stale_graph(self):
        self.worker.request(self.slam, None)
        self.occupancy[10:20, 20] = OCCUPIED
        self.worker.request(self.snapshot(2), None)
        result = self.finish()
        self.assertEqual(result.status, "superseded")
        self.assertIsNone(result.snapshot)
        self.assertEqual(len(self.executor.jobs), 2)
        self.assertEqual(self.executor.jobs[-1][1].version, 2)

    def test_new_free_cells_allow_coherent_older_graph_without_starvation(self):
        self.occupancy[:, 30:] = UNKNOWN
        initial = self.snapshot(1)
        result = build_highway_graph(initial, confidence_threshold=0.6, macro_cell_size=4,
                                     maximum_build_ms=1000, maximum_connector_expansions=256,
                                     maximum_access_distance=20)
        self.worker.request(initial, None)
        self.occupancy[:, 30:] = FREE
        self.worker.request(self.snapshot(2), None)
        published = self.finish(result)
        self.assertEqual(published.snapshot.version, 1)
        self.assertEqual(len(self.executor.jobs), 2)
        self.assertFalse(self.worker.published_free[0, 40])

    def test_origin_change_cannot_relabel_a_graph(self):
        self.worker.request(self.slam, None)
        self.worker.request(self.snapshot(2, origin=(5, 10)), None)
        self.assertEqual(self.finish().status, "superseded")

    def test_full_rebuild_coalesces_new_inputs_until_throttle_expires(self):
        self.worker.request(self.slam, None)
        graph = self.finish().snapshot
        self.clock = 2.1
        self.worker.poll()
        self.assertEqual(len(self.executor.jobs), 1)
        self.occupancy[10, 20] = OCCUPIED
        self.worker.request(self.snapshot(2), graph)
        self.assertEqual(len(self.executor.jobs), 2)
        graph = self.finish().snapshot
        submitted = self.worker.last_submission
        self.occupancy[11, 20] = OCCUPIED
        self.clock = submitted + 1
        self.worker.request(self.snapshot(3), graph)
        self.worker.request(self.snapshot(4), graph)
        self.assertEqual(len(self.executor.jobs), 2)
        self.clock = submitted + 2.01
        self.worker.poll()
        self.assertEqual(self.executor.jobs[-1][1].version, 4)
        self.assertEqual(len(self.executor.jobs), 3)

    def test_unchanged_geometry_does_not_trigger_periodic_rebuild(self):
        self.worker.request(self.slam, None)
        graph = self.finish().snapshot
        self.clock = 120
        self.worker.poll()
        self.assertEqual(len(self.executor.jobs), 1)

    def test_failed_build_retries_only_after_throttle_interval(self):
        self.worker.request(self.slam, None)
        self.executor.jobs[-1][0].set_result(HighwayBuildResult("budget_exhausted", None, 250))
        self.clock = .25
        result = self.worker.poll()
        self.assertIsNone(result.snapshot)
        self.assertEqual(len(self.executor.jobs), 1)
        self.clock = 2
        self.worker.poll()
        self.assertEqual(len(self.executor.jobs), 2)

    def test_worker_exception_disables_optional_advice(self):
        self.worker.request(self.slam, None)
        self.executor.jobs[-1][0].set_exception(RuntimeError("worker stopped"))
        result = self.worker.poll()
        self.assertEqual(result.status, "unavailable")
        self.assertTrue(self.worker.closed)
        self.assertEqual(self.worker.request(self.snapshot(2), None).status, "unavailable")
        self.assertEqual(len(self.executor.jobs), 1)

    def test_budget_retry_finishes_same_input_before_new_additions(self):
        self.occupancy[:, 30:] = UNKNOWN
        initial = self.snapshot(1)
        initial_result = build_highway_graph(
            initial, confidence_threshold=.6, macro_cell_size=4,
            maximum_build_ms=1000, maximum_connector_expansions=256,
            maximum_access_distance=20,
        )
        self.worker.request(initial, None)
        self.occupancy[:, 30:] = FREE
        self.worker.request(self.snapshot(2), None)
        self.finish(HighwayBuildResult("budget_exhausted", None, 250))
        self.assertEqual(self.executor.jobs[-1][1].version, 1)
        published = self.finish(initial_result)
        self.assertEqual(published.snapshot.version, 1)
        self.assertFalse(self.worker.published_free[0, 40])
        self.assertEqual(self.executor.jobs[-1][1].version, 2)

    def test_wall_correction_cancels_queued_budget_retry(self):
        self.worker.request(self.slam, None)
        self.executor.jobs[-1][0].set_result(HighwayBuildResult("budget_exhausted", None, 250))
        self.clock = .25
        self.worker.poll()
        self.occupancy[10, 20] = OCCUPIED
        self.worker.request(self.snapshot(2), None)
        self.clock = 2
        self.worker.poll()
        self.assertEqual(self.executor.jobs[-1][1].version, 2)
        self.assertFalse(self.worker.job_retry)

    def test_repeated_budget_miss_releases_retry_to_latest_input(self):
        self.occupancy[:, 30:] = UNKNOWN
        self.worker.request(self.snapshot(1), None)
        self.occupancy[:, 30:] = FREE
        self.worker.request(self.snapshot(2), None)
        miss = HighwayBuildResult("budget_exhausted", None, 250)
        self.finish(miss)
        self.assertEqual(self.executor.jobs[-1][1].version, 1)
        self.finish(miss)
        self.assertEqual(self.executor.jobs[-1][1].version, 2)
        self.assertFalse(self.worker.job_retry)

    def test_service_starts_lazily_publishes_only_completed_work_and_closes_worker(self):
        with patch("navigation.highway_worker.HighwayWorker", return_value=self.worker) as worker_type:
            service = HighwayService(confidence_threshold=.6, macro_cell_size=4,
                                     maximum_build_ms=250, maximum_query_ms=50,
                                     maximum_connector_expansions=256)
            worker_type.assert_not_called()
            service.start()
            self.assertEqual(service.refresh(self.slam).status, "pending")
            self.assertIsNone(service.snapshot)
            self.executor.jobs[-1][0].set_result(self.result)
            service.poll()
            self.assertEqual(service.snapshot.version, 1)
            service.shutdown()
        self.assertTrue(self.executor.stopped)


if __name__ == "__main__":
    unittest.main()
