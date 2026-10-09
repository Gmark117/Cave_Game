import unittest
import math
from unittest.mock import patch

import numpy as np

from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from navigation.highway import (
    HIGHWAY_BUDGET_EXHAUSTED,
    HIGHWAY_COMPLETE,
    HIGHWAY_STALE,
    HIGHWAY_UNREACHABLE,
    HighwayService,
    build_highway_graph,
    build_tiled_highway_graph,
)


class HighwayGraphTests(unittest.TestCase):
    def assert_safe_route(self, occupancy, route, origin=(0, 0)):
        self.assertTrue(route.complete)
        local = [(x - origin[0], y - origin[1]) for x, y in route.path]
        self.assertTrue(all(occupancy[y, x] == FREE for x, y in local))
        for (x, y), (nx, ny) in zip(local, local[1:]):
            self.assertLessEqual(max(abs(nx - x), abs(ny - y)), 1)
            if x != nx and y != ny:
                self.assertEqual(occupancy[y, nx], FREE)
                self.assertEqual(occupancy[ny, x], FREE)
        self.assertAlmostEqual(
            route.cost, sum(math.dist(a, b) for a, b in zip(route.path, route.path[1:])),
        )

    def snapshot(
        self,
        occupancy: np.ndarray,
        *,
        confidence: np.ndarray | None = None,
        version: int = 1,
        origin: tuple[int, int] = (0, 0),
    ) -> SlamSnapshot:
        if confidence is None:
            confidence = np.where(occupancy == UNKNOWN, 0.0, 1.0)
        return SlamSnapshot(
            occupancy=np.asarray(occupancy, dtype=np.int8),
            confidence=np.asarray(confidence, dtype=np.float32),
            version=version,
            origin=origin,
            full_shape=occupancy.shape,
        )

    def build(self, occupancy: np.ndarray, **kwargs):
        return build_highway_graph(
            self.snapshot(occupancy, **kwargs),
            confidence_threshold=0.6,
            macro_cell_size=4,
            maximum_build_ms=1000.0,
            maximum_connector_expansions=256,
        )

    def test_routes_across_safe_tile_portals_with_exact_polyline(self) -> None:
        occupancy = np.full((8, 12), FREE, dtype=np.int8)
        occupancy[:7, 6] = OCCUPIED
        result = self.build(occupancy, origin=(10, 20))

        self.assertEqual(result.status, HIGHWAY_COMPLETE)
        snapshot = result.snapshot
        self.assertIsNotNone(snapshot)
        route = snapshot.route(
            (11, 21),
            (20, 26),
            maximum_query_ms=100.0,
            maximum_connector_expansions=256,
            required_version=1,
        )

        self.assertTrue(route.complete)
        self.assertEqual(route.path[0], (11, 21))
        self.assertEqual(route.path[-1], (20, 26))
        self.assertGreater(route.cost, math.dist(route.path[0], route.path[-1]))
        self.assertGreater(route.cost, 0.0)
        self.assertTrue(all(
            occupancy[y - 20, x - 10] == FREE for x, y in route.path
        ))

    def test_disconnected_regions_inside_one_tile_remain_distinct(self) -> None:
        occupancy = np.full((4, 4), OCCUPIED, dtype=np.int8)
        occupancy[:, 0] = FREE
        occupancy[:, 3] = FREE
        result = self.build(occupancy)
        snapshot = result.snapshot

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.component_count, 2)
        route = snapshot.route(
            (0, 0),
            (3, 0),
            maximum_query_ms=100.0,
            maximum_connector_expansions=64,
        )
        self.assertEqual(route.status, HIGHWAY_UNREACHABLE)

    def test_unknown_and_low_confidence_cells_are_closed(self) -> None:
        occupancy = np.full((4, 8), FREE, dtype=np.int8)
        occupancy[:, 4] = UNKNOWN
        confidence = np.ones_like(occupancy, dtype=np.float32)
        confidence[:, 2] = 0.2
        result = build_highway_graph(
            self.snapshot(occupancy, confidence=confidence),
            confidence_threshold=0.6,
            macro_cell_size=4,
            maximum_build_ms=1000.0,
            maximum_connector_expansions=128,
        )
        snapshot = result.snapshot

        self.assertIsNotNone(snapshot)
        self.assertEqual(int(snapshot.area_labels[0, 2]), 0)
        self.assertEqual(int(snapshot.area_labels[0, 4]), 0)

    def test_version_fence_and_immutable_labels(self) -> None:
        result = self.build(np.full((4, 8), FREE, dtype=np.int8), version=7)
        snapshot = result.snapshot
        self.assertIsNotNone(snapshot)

        route = snapshot.route(
            (0, 0),
            (7, 3),
            maximum_query_ms=100.0,
            maximum_connector_expansions=128,
            required_version=8,
        )
        self.assertEqual(route.status, HIGHWAY_STALE)
        with self.assertRaises(ValueError):
            snapshot.area_labels[0, 0] = 99

    def test_build_budget_discards_partial_snapshot(self) -> None:
        result = build_highway_graph(
            self.snapshot(np.full((64, 64), FREE, dtype=np.int8)),
            confidence_threshold=0.6,
            macro_cell_size=4,
            maximum_build_ms=0.000001,
            maximum_connector_expansions=64,
        )

        self.assertEqual(result.status, HIGHWAY_BUDGET_EXHAUSTED)
        self.assertIsNone(result.snapshot)

    def test_service_retains_previous_snapshot_after_failed_rebuild(self) -> None:
        service = HighwayService(
            confidence_threshold=0.6,
            macro_cell_size=4,
            maximum_build_ms=1000.0,
            maximum_query_ms=100.0,
            maximum_connector_expansions=128,
        )
        first = service.refresh(
            self.snapshot(np.full((4, 8), FREE, dtype=np.int8), version=2)
        )
        self.assertEqual(first.status, HIGHWAY_COMPLETE)
        retained = service.snapshot

        service.maximum_build_ms = 0.000001
        failed = service.refresh(
            self.snapshot(np.full((64, 64), FREE, dtype=np.int8), version=3)
        )
        self.assertEqual(failed.status, HIGHWAY_BUDGET_EXHAUSTED)
        self.assertIs(service.snapshot, retained)

    def test_branching_backbone_shortens_obstacle_route_against_tiled_baseline(self):
        occupancy = np.full((160, 224), FREE, np.int8)
        occupancy[:120, 104:120] = OCCUPIED
        slam = self.snapshot(occupancy)
        graphs = [builder(
            slam, confidence_threshold=0.6, macro_cell_size=32,
            maximum_build_ms=250, maximum_connector_expansions=4096,
        ).snapshot for builder in (build_tiled_highway_graph, build_highway_graph)]
        self.assertTrue(all(graph is not None for graph in graphs))
        routes = [graph.route(
            (20, 20), (200, 20), maximum_query_ms=50,
            maximum_connector_expansions=4096,
        ) for graph in graphs]
        self.assert_safe_route(occupancy, routes[1])
        self.assertLess(routes[1].cost, routes[0].cost * 0.85)
        self.assertLess(graphs[1].area_count, graphs[0].area_count)

    def test_capillarity_bounds_geodesic_access_and_changes_network_density(self):
        occupancy = np.full((100, 140), OCCUPIED, np.int8)
        occupancy[2:-2, 2:-2] = FREE
        graphs = [build_highway_graph(
            self.snapshot(occupancy), confidence_threshold=0.6, macro_cell_size=32,
            maximum_build_ms=250, maximum_connector_expansions=4096,
            maximum_access_distance=reach,
        ).snapshot for reach in (12.0, 45.0)]
        for graph, reach in zip(graphs, (12.0, 45.0)):
            self.assertIsNotNone(graph)
            self.assertLessEqual(graph.measured_access_distance, reach)
        self.assertGreater(graphs[0].edge_count, graphs[1].edge_count)
        self.assertGreater(graphs[0].capillary_branches, graphs[1].capillary_branches)

    def test_backbone_keeps_loop_and_routes_both_sides_of_an_obstacle(self):
        occupancy = np.full((100, 140), OCCUPIED, np.int8)
        occupancy[10:90, 10:130] = FREE
        occupancy[30:70, 40:100] = OCCUPIED
        graph = build_highway_graph(
            self.snapshot(occupancy), confidence_threshold=0.6, macro_cell_size=32,
            maximum_build_ms=250, maximum_connector_expansions=4096,
        ).snapshot
        self.assertIsNotNone(graph)
        self.assertGreaterEqual(graph.edge_count, graph.area_count)
        for start, goal in (((20, 20), (120, 80)), ((20, 80), (120, 20))):
            route = graph.route(start, goal, maximum_query_ms=50, maximum_connector_expansions=4096)
            self.assert_safe_route(occupancy, route)

    def test_one_cell_elbow_survives_skeleton_diagonal_repair(self):
        occupancy = np.full((80, 120), OCCUPIED, np.int8)
        occupancy[10:70, 20] = FREE
        occupancy[69, 20:110] = FREE
        graph = build_highway_graph(
            self.snapshot(occupancy), confidence_threshold=0.6, macro_cell_size=32,
            maximum_build_ms=250, maximum_connector_expansions=4096,
        ).snapshot
        self.assertIsNotNone(graph)
        route = graph.route((20, 10), (109, 69), maximum_query_ms=50, maximum_connector_expansions=4096)
        self.assert_safe_route(occupancy, route)

    def test_diagonal_only_contact_does_not_join_free_components(self):
        occupancy = np.full((4, 4), OCCUPIED, np.int8)
        occupancy[1, 1] = occupancy[2, 2] = FREE
        graph = self.build(occupancy).snapshot
        self.assertIsNotNone(graph)
        self.assertEqual(graph.component_count, 2)
        self.assertEqual(graph.route(
            (1, 1), (2, 2), maximum_query_ms=50, maximum_connector_expansions=32,
        ).status, HIGHWAY_UNREACHABLE)

    def test_one_cell_bridge_between_chambers_is_not_pruned(self):
        occupancy = np.full((34, 74), OCCUPIED, np.int8)
        occupancy[2:32, 2:32] = FREE
        occupancy[2:32, 42:72] = FREE
        occupancy[16, 31:43] = FREE
        graph = self.build(occupancy).snapshot
        self.assertIsNotNone(graph)
        self.assertEqual(graph.component_count, 1)
        route = graph.route((4, 4), (68, 28), maximum_query_ms=50, maximum_connector_expansions=4096)
        self.assert_safe_route(occupancy, route)
        self.assertIn((36, 16), route.path)

    def test_large_map_advances_past_former_size_guard_with_safe_routes(self):
        service = HighwayService(
            confidence_threshold=0.6, macro_cell_size=32,
            maximum_build_ms=250, maximum_query_ms=50,
            maximum_connector_expansions=4096,
            maximum_access_distance_sensor_ranges=2, sensor_range=160,
        )
        service.refresh(self.snapshot(np.full((32, 32), FREE, np.int8), version=1))
        occupancy = np.full((600, 600), FREE, np.int8)
        occupancy[:450, 290:310] = OCCUPIED
        result = service.refresh(self.snapshot(occupancy, version=2))
        self.assertEqual(result.status, HIGHWAY_COMPLETE)
        graph = service.snapshot
        self.assertEqual(graph.version, 2)
        self.assertGreater(graph.skeleton_scale, 1)
        self.assertLessEqual(graph.measured_access_distance, 320)
        route = graph.route((20, 20), (580, 20), maximum_query_ms=50,
                            maximum_connector_expansions=4096)
        self.assert_safe_route(occupancy, route)

    def test_skeleton_reduction_retains_narrow_bridges_and_small_holes(self):
        from navigation.highway_backbone import _skeleton_grid
        import cv2
        import time

        free = np.zeros((600, 600), bool)
        free[10:590, 10:280] = True
        free[10:590, 320:590] = True
        free[301, 279:321] = True  # This bridge cannot survive 2x reduction.
        count, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
        _, scale = _skeleton_grid(free, labels, count - 1, 320, time.perf_counter() + 1)
        self.assertEqual(scale, 1)

        free[:] = True
        free[300, 300] = free[302, 300] = False  # Two holes merge at 2x.
        count, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
        _, scale = _skeleton_grid(free, labels, count - 1, 320, time.perf_counter() + 1)
        self.assertEqual(scale, 1)

    def test_oversized_native_work_still_retains_previous_complete_graph(self):
        service = HighwayService(
            confidence_threshold=0.6, macro_cell_size=32,
            maximum_build_ms=250, maximum_query_ms=50,
            maximum_connector_expansions=4096,
        )
        service.refresh(self.snapshot(np.full((32, 32), FREE, np.int8), version=1))
        previous = service.snapshot
        result = service.refresh(self.snapshot(np.full((1000, 1000), FREE, np.int8), version=2))
        self.assertEqual(result.status, HIGHWAY_BUDGET_EXHAUSTED)
        self.assertIs(service.snapshot, previous)

    def test_access_field_and_network_indices_are_immutable(self):
        graph = self.build(np.full((4, 8), FREE, np.int8)).snapshot
        with self.assertRaises(ValueError):
            graph.access_predecessors[0, 0] = 0
        with self.assertRaises(TypeError):
            graph.network_nodes[(0, 0)] = 5

    def test_whole_map_preparation_survives_budget_retry_and_invalidates_on_change(self):
        from navigation.highway_backbone import _BudgetExceeded, build_backbone
        occupancy = np.full((64, 96), FREE, np.int8)
        occupancy[:50, 45:48] = OCCUPIED
        cache = {}
        settings = dict(confidence_threshold=.6, macro_cell_size=4,
                        maximum_build_ms=1000, maximum_access_distance=80,
                        preparation_cache=cache)
        with patch("navigation.highway_backbone._cover", side_effect=_BudgetExceeded):
            result = build_backbone(self.snapshot(occupancy), **settings)
        self.assertEqual(result.status, HIGHWAY_BUDGET_EXHAUSTED)
        self.assertIsNone(result.snapshot)
        self.assertIn("smooth", cache)
        with patch("navigation.highway_backbone.medial_axis", side_effect=AssertionError("recomputed")):
            graph = build_backbone(self.snapshot(occupancy, version=2), **settings).snapshot
        self.assertIsNotNone(graph)
        self.assertEqual(graph.version, 2)
        self.assert_safe_route(occupancy, graph.route((10, 10), (80, 10),
                               maximum_query_ms=50, maximum_connector_expansions=4096))
        occupancy[50:60, 45:48] = OCCUPIED
        with patch("navigation.highway_backbone.medial_axis", wraps=__import__(
                "skimage.morphology", fromlist=["medial_axis"]).medial_axis) as skeleton:
            result = build_backbone(self.snapshot(occupancy, version=3), **settings)
        skeleton.assert_called_once()
        self.assertEqual(result.status, HIGHWAY_COMPLETE)

    def test_full_graph_process_round_trip_is_immutable(self):
        import pickle
        graph = pickle.loads(pickle.dumps(self.build(np.full((32, 64), FREE, np.int8)).snapshot))
        self.assertFalse(graph.area_labels.flags.writeable)
        self.assertFalse(graph.access_predecessors.flags.writeable)
        with self.assertRaises(TypeError):
            graph.network_nodes[(0, 0)] = 5

    def test_large_unreducible_map_uses_thinning_and_keeps_loops(self):
        from navigation.highway_backbone import build_backbone
        occupancy = np.full((600, 900), FREE, np.int8)
        occupancy[200:260, 300:360] = OCCUPIED
        occupancy[300, 449] = occupancy[302, 449] = OCCUPIED
        graph = build_backbone(self.snapshot(occupancy), confidence_threshold=.6,
                               macro_cell_size=32, maximum_build_ms=2000,
                               maximum_access_distance=320).snapshot
        self.assertIsNotNone(graph)
        self.assertEqual(graph.skeleton_method, "thinning")
        self.assertEqual(graph.skeleton_scale, 1)
        self.assertGreaterEqual(graph.edge_count, graph.area_count)
        self.assertLessEqual(graph.measured_access_distance, 320)
        self.assert_safe_route(occupancy, graph.route((20, 300), (880, 300),
                               maximum_query_ms=100, maximum_connector_expansions=4096))

    def test_query_budget_applies_to_direct_routes_and_connectors(self):
        occupancy = np.full((80, 120), FREE, np.int8)
        occupancy[:60, 50:60] = OCCUPIED
        graph = self.build(occupancy).snapshot
        self.assertEqual(graph.route(
            (10, 10), (100, 10), maximum_query_ms=0,
            maximum_connector_expansions=4096,
        ).status, HIGHWAY_BUDGET_EXHAUSTED)
        self.assertEqual(graph.route(
            (10, 10), (100, 10), maximum_query_ms=50,
            maximum_connector_expansions=1,
        ).status, HIGHWAY_BUDGET_EXHAUSTED)


if __name__ == "__main__":
    unittest.main()
