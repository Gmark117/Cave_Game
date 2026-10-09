import unittest

import numpy as np

from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from navigation.return_route import local_return_alternative


class LocalReturnRouteTests(unittest.TestCase):
    def route(self, occupancy, *, cached_path=(), maximum_expansions=1000,
              incumbent_cost=100, origin=(0, 0), maximum_ms=1000):
        return local_return_alternative(
            SlamSnapshot(occupancy, np.ones(occupancy.shape, np.float32), origin=origin),
            (2 + origin[0], 2 + origin[1]), (17 + origin[0], 2 + origin[1]),
            confidence_threshold=0.6, incumbent_cost=incumbent_cost,
            maximum_ms=maximum_ms, maximum_expansions=maximum_expansions,
            cached_path=cached_path,
        )

    def test_known_direct_route_is_selected_only_when_strictly_shorter(self):
        occupancy = np.full((20, 20), FREE, np.int8)
        route = self.route(occupancy, origin=(10, 20))
        self.assertTrue(route.complete)
        self.assertEqual((route.path[0], route.path[-1], route.cost), ((12, 22), (27, 22), 15))
        self.assertEqual(self.route(occupancy, incumbent_cost=15).status, "no_shorter_route")

    def test_cached_route_is_revalidated_without_a_new_search(self):
        occupancy = np.full((20, 20), FREE, np.int8)
        occupancy[:10, 10] = OCCUPIED
        cached = ((2, 2), (2, 11), (17, 11), (17, 2))
        route = self.route(occupancy, cached_path=cached, maximum_expansions=0)
        self.assertTrue(route.complete)
        self.assertEqual(route.expanded_nodes, 0)
        occupancy[11, 10] = OCCUPIED
        route = self.route(occupancy, cached_path=cached, maximum_expansions=0)
        self.assertEqual(route.status, "budget_exhausted")
        self.assertFalse(route.path)

    def test_unknown_cells_and_diagonal_corners_cannot_be_crossed(self):
        occupancy = np.full((20, 20), FREE, np.int8)
        occupancy[:, 10] = UNKNOWN
        occupancy[10, 10] = FREE
        occupancy[10, 9] = OCCUPIED
        occupancy[9, 10] = OCCUPIED
        occupancy[11, 10] = OCCUPIED
        self.assertFalse(self.route(occupancy).complete)

    def test_bounded_search_around_wall_returns_safe_known_cells(self):
        occupancy = np.full((20, 20), FREE, np.int8)
        occupancy[:10, 10] = OCCUPIED
        route = self.route(occupancy)
        self.assertTrue(route.complete)
        for x, y in route.path:
            self.assertEqual(occupancy[y, x], FREE)
        for (x, y), (nx, ny) in zip(route.path, route.path[1:]):
            if x != nx and y != ny:
                self.assertEqual(occupancy[y, nx], FREE)
                self.assertEqual(occupancy[ny, x], FREE)
        self.assertEqual(self.route(occupancy, maximum_expansions=1).status, "budget_exhausted")
        self.assertEqual(self.route(occupancy, maximum_ms=0).status, "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
