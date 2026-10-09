import math
import unittest

import numpy as np

from mapping.slam_map import FREE, OCCUPIED, SlamMap, SlamSnapshot
from navigation.local_slam_routes import LocalSlamRoutePlanner


class LocalSlamRoutePlannerTests(unittest.TestCase):
    def setUp(self):
        self.map = SlamMap(64, 64)
        self.map.merge_from(SlamSnapshot(np.full((64, 64), FREE, np.int8),
                                        np.ones((64, 64), np.float32)))
        self.events = []
        self.clock = 0
        self.planner = LocalSlamRoutePlanner(self.map, .6, now=lambda: self.clock,
                                           trace=lambda name, **fields: self.events.append((name, fields)))

    def test_complete_and_reverse_queries_share_cache_with_exact_cost(self):
        with self.planner.planning("test"):
            first = self.planner.route((2, 2), (40, 20))
            reverse = self.planner.route((40, 20), (2, 2))
            again = self.planner.route((2, 2), (40, 20))
        self.assertTrue(first.complete)
        self.assertIs(first, again)
        self.assertEqual(reverse.path, tuple(reversed(first.path)))
        self.assertAlmostEqual(first.cost, sum(math.dist(a, b) for a, b in zip(first.path, first.path[1:])))
        self.assertEqual((self.events[-1][1]["route_queries"], self.events[-1][1]["route_cache_hits"]), (1, 2))

    def test_new_local_wall_invalidates_previously_cached_path(self):
        first = self.planner.route((2, 2), (40, 2))
        occupancy = np.full((64, 64), -1, np.int8)
        confidence = np.zeros((64, 64), np.float32)
        occupancy[2, 20], confidence[2, 20] = OCCUPIED, 1
        self.map.merge_from(SlamSnapshot(occupancy, confidence))
        second = self.planner.route((2, 2), (40, 2))
        self.assertTrue(second.complete)
        self.assertNotEqual(second.snapshot_version, first.snapshot_version)
        self.assertNotIn((20, 2), second.path)

    def test_query_cap_and_pass_deadline_apply_to_cache_hits_too(self):
        self.planner.MAXIMUM_QUERIES = 2
        with self.planner.planning("test"):
            self.assertTrue(self.planner.route((2, 2), (3, 2)).complete)
            self.assertTrue(self.planner.route((2, 2), (4, 2)).complete)
            self.assertEqual(self.planner.route((2, 2), (5, 2)).status, "budget_exhausted")
            self.clock = .251
            self.assertEqual(self.planner.route((2, 2), (3, 2)).status, "budget_exhausted")
        self.assertEqual(self.events[-1][1]["status"], "budget_exhausted")

    def test_limited_search_returns_safe_progress_without_complete_cost(self):
        occupancy = np.full((64, 64), -1, np.int8)
        confidence = np.zeros((64, 64), np.float32)
        occupancy[:40, 28], confidence[:40, 28] = OCCUPIED, 1
        self.map.merge_from(SlamSnapshot(occupancy, confidence))
        self.planner.MAXIMUM_EXPANSIONS = 4
        route = self.planner.route((16, 16), (40, 16))
        self.assertEqual(route.status, "partial_limit")
        self.assertFalse(route.complete)
        self.assertEqual(route.path[0], (16, 16))
        self.assertNotEqual(route.path[-1], (16, 16))
        self.assertNotEqual(route.path[-1], (40, 16))
        belief = self.map.snapshot(point_limit=0)
        self.assertTrue(all(belief.occupancy[y, x] == FREE for x, y in route.path))
        with self.planner.planning("cached_partial"):
            self.assertIs(self.planner.route((16, 16), (40, 16)), route)
        self.assertEqual(self.events[-1][1]["status"], "budget_exhausted")
        self.assertEqual(self.events[-1][1]["route_cache_hits"], 1)


if __name__ == "__main__":
    unittest.main()
