import unittest

import numpy as np

from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from navigation.highway import (
    HIGHWAY_BUDGET_EXHAUSTED,
    HIGHWAY_COMPLETE,
    HIGHWAY_STALE,
    HIGHWAY_UNREACHABLE,
    HighwayService,
    build_highway_graph,
)


class HighwayGraphTests(unittest.TestCase):
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
        self.assertGreater(route.graph_edges, 0)
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
        self.assertEqual(snapshot.area_count, 2)
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


if __name__ == "__main__":
    unittest.main()
