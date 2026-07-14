import concurrent.futures
import math
import unittest
from unittest.mock import patch

import numpy as np

from navigation.waypoint_graph import (
    EDGE_KNOWN_FREE_CORRIDOR,
    EDGE_KNOWN_FREE_CONNECTOR,
    EDGE_SLAM_LOS,
    EDGE_TRAVELLED,
    ROUTE_DISCONNECTED,
    ROUTE_GOAL_UNKNOWN,
    ROUTE_NO_GOAL_CONNECTOR,
    ROUTE_NO_START_CONNECTOR,
    ROUTE_OK,
    WaypointGraph,
    bresenham_path,
    known_free_path,
    validate_known_free_path,
)


class KnownFreePathTests(unittest.TestCase):
    def test_builds_oriented_bresenham_path_and_rejects_unknown_cell(self) -> None:
        known_free = np.ones((8, 8), dtype=bool)

        path = known_free_path((1, 2), (6, 4), known_free)

        self.assertEqual(path[0], (1, 2))
        self.assertEqual(path[-1], (6, 4))
        self.assertEqual(tuple(reversed(path)), bresenham_path((6, 4), (1, 2)))
        blocked = path[len(path) // 2]
        known_free[blocked[1], blocked[0]] = False
        self.assertEqual(known_free_path((1, 2), (6, 4), known_free), ())

    def test_diagonal_corner_rule_requires_one_known_free_side(self) -> None:
        squeezed = np.array([[True, False], [False, True]], dtype=bool)
        one_open_side = np.array([[True, True], [False, True]], dtype=bool)

        self.assertFalse(
            validate_known_free_path(((0, 0), (1, 1)), squeezed)
        )
        self.assertTrue(
            validate_known_free_path(((0, 0), (1, 1)), one_open_side)
        )

    def test_rejects_uint8_occupancy_map_as_navigation_knowledge(self) -> None:
        cave_map = np.zeros((4, 4), dtype=np.uint8)

        with self.assertRaises(TypeError):
            known_free_path((0, 0), (3, 3), cave_map)


class WaypointGraphInsertionTests(unittest.TestCase):
    def test_exact_duplicates_merge_without_map_but_near_duplicates_do_not(self) -> None:
        graph = WaypointGraph(merge_radius=4)

        first, first_added = graph.add_waypoint((5, 5), source="home")
        duplicate, duplicate_added = graph.add_waypoint((5, 5), source="other")
        near, near_added = graph.add_waypoint((7, 5), source="travelled")

        self.assertEqual(first, (5, 5))
        self.assertTrue(first_added)
        self.assertEqual(duplicate, (5, 5))
        self.assertFalse(duplicate_added)
        self.assertEqual(near, (7, 5))
        self.assertTrue(near_added)
        self.assertEqual(graph.node_count, 2)

    def test_near_duplicate_merges_only_with_known_free_line_of_sight(self) -> None:
        graph = WaypointGraph(merge_radius=4)
        graph.add_waypoint((5, 5))
        visible = np.ones((12, 12), dtype=bool)

        canonical, added = graph.add_waypoint(
            (8, 5), source="gateway", known_free=visible
        )

        self.assertEqual(canonical, (5, 5))
        self.assertFalse(added)
        hidden_graph = WaypointGraph(merge_radius=4)
        hidden_graph.add_waypoint((5, 5))
        hidden = visible.copy()
        hidden[5, 6] = False
        hidden_canonical, hidden_added = hidden_graph.add_waypoint(
            (8, 5), source="gateway", known_free=hidden
        )
        self.assertEqual(hidden_canonical, (8, 5))
        self.assertTrue(hidden_added)

    def test_snapshot_and_update_records_are_detached_and_traceable(self) -> None:
        graph = WaypointGraph(spacing=5)
        update = graph.register_travelled_path(((0, 0), (10, 0)))
        snapshot = graph.snapshot()

        self.assertEqual(
            tuple(node.position for node in update.added_waypoints),
            ((0, 0), (5, 0), (10, 0)),
        )
        self.assertEqual(update.added_nodes, update.added_waypoints)
        self.assertTrue(update.changed)
        self.assertEqual(snapshot.node_count, 3)
        self.assertEqual(snapshot.edge_count, 2)
        self.assertEqual(graph.counts(), (3, 2))
        self.assertTrue(all(edge.source == EDGE_TRAVELLED for edge in snapshot.edges))


class WaypointGraphTravelledPathTests(unittest.TestCase):
    def test_densifies_coarse_path_samples_cumulative_spacing_and_endpoints(self) -> None:
        graph = WaypointGraph(spacing=4, merge_radius=0)

        update = graph.register_travelled_path(((0, 0), (10, 0)))

        self.assertEqual(
            update.sampled_waypoints,
            ((0, 0), (4, 0), (8, 0), (10, 0)),
        )
        self.assertEqual(graph.node_count, 4)
        self.assertEqual(graph.edge_count, 3)
        self.assertEqual(update.added_edges[0].path, bresenham_path((0, 0), (4, 0)))
        self.assertEqual(update.added_edges[-1].path[-1], (10, 0))

    def test_stores_turning_travelled_polyline_in_original_orientation(self) -> None:
        graph = WaypointGraph(spacing=20)

        update = graph.register_travelled_path(((2, 2), (6, 2), (6, 6)))

        self.assertEqual(update.sampled_waypoints, ((2, 2), (6, 6)))
        edge = update.added_edges[0]
        expected = (
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
        )
        if edge.start == (2, 2):
            self.assertEqual(edge.path, expected)
        else:
            self.assertEqual(edge.path, tuple(reversed(expected)))
        self.assertAlmostEqual(edge.cost, 8.0)


class WaypointGraphGatewayTests(unittest.TestCase):
    def test_gateway_connects_only_bounded_nearest_visible_neighbors(self) -> None:
        graph = WaypointGraph(
            merge_radius=0,
            connector_distance=20,
            connector_limit=2,
        )
        for position in ((0, 0), (0, 4), (0, 8), (18, 18)):
            graph.add_waypoint(position, source="travelled")
        known_free = np.ones((24, 24), dtype=bool)

        update = graph.connect_known_free_waypoint((4, 4), known_free)

        self.assertEqual(update.status, ROUTE_OK)
        self.assertEqual(len(update.added_waypoints), 1)
        self.assertEqual(len(update.added_edges), 2)
        self.assertEqual(
            {edge.start if edge.end == (4, 4) else edge.end for edge in update.added_edges},
            {(0, 4), (0, 0)},
        )
        self.assertTrue(all(edge.source == EDGE_SLAM_LOS for edge in update.added_edges))

    def test_unknown_gateway_is_not_persisted_or_connected(self) -> None:
        graph = WaypointGraph()
        graph.add_waypoint((1, 1))
        known_free = np.ones((8, 8), dtype=bool)
        known_free[5, 5] = False

        update = graph.connect_known_free_waypoint((5, 5), known_free)

        self.assertEqual(update.status, "position_unknown")
        self.assertFalse(update.changed)
        self.assertEqual(graph.counts(), (1, 0))

    def test_visible_but_isolated_gateway_is_rolled_back(self) -> None:
        graph = WaypointGraph(connector_distance=3)
        graph.add_waypoint((1, 1), source="home")
        known_free = np.ones((16, 16), dtype=bool)

        update = graph.connect_known_free_waypoint((12, 12), known_free)

        self.assertEqual(update.status, "no_connector")
        self.assertFalse(update.changed)
        self.assertEqual(update.added_waypoints, ())
        self.assertEqual(graph.counts(), (1, 0))

    def test_gateway_degree_stays_bounded_across_reconnections(self) -> None:
        graph = WaypointGraph(
            merge_radius=0,
            connector_distance=20,
            connector_limit=2,
        )
        for position in ((0, 4), (0, 8)):
            graph.add_waypoint(position, source="travelled")
        known_free = np.ones((24, 24), dtype=bool)
        graph.connect_known_free_waypoint((4, 4), known_free)
        for position in ((3, 3), (3, 5)):
            graph.add_waypoint(position, source="travelled")

        update = graph.connect_known_free_waypoint((4, 4), known_free)

        self.assertFalse(update.changed)
        gateway_edges = [
            edge
            for edge in graph.snapshot().edges
            if (4, 4) in {edge.start, edge.end}
            and edge.source == EDGE_SLAM_LOS
        ]
        self.assertEqual(len(gateway_edges), 2)

    def test_corridor_bridge_samples_gap_beyond_connector_distance(self) -> None:
        graph = WaypointGraph(
            spacing=8,
            merge_radius=0,
            connector_distance=4,
        )
        graph.add_waypoint((2, 4), source="travelled")
        known_free = np.zeros((12, 40), dtype=bool)
        known_free[4, 2:35] = True

        before = graph.version
        self.assertEqual(
            graph.find_route((2, 4), (34, 4), known_free).status,
            ROUTE_NO_GOAL_CONNECTOR,
        )

        update = graph.connect_known_free_corridor(
            (34, 4),
            known_free,
            search_distance=33,
        )

        self.assertEqual(update.status, ROUTE_OK)
        self.assertTrue(update.changed)
        self.assertGreater(graph.version, before)
        self.assertEqual(
            update.sampled_waypoints,
            (
                (2, 4),
                (6, 4),
                (10, 4),
                (14, 4),
                (18, 4),
                (22, 4),
                (26, 4),
                (30, 4),
                (34, 4),
            ),
        )
        self.assertEqual(len(update.added_edges), 8)
        self.assertTrue(
            all(
                edge.source == EDGE_KNOWN_FREE_CORRIDOR
                and edge.cost <= graph.connector_distance + 1e-9
                for edge in update.added_edges
            )
        )
        self.assertTrue(
            graph.find_route((2, 4), (34, 4), known_free).found
        )

    def test_corridor_bridge_follows_curved_known_free_passage(self) -> None:
        graph = WaypointGraph(
            spacing=5,
            merge_radius=0,
            connector_distance=3,
        )
        graph.add_waypoint((2, 2), source="travelled")
        known_free = np.zeros((24, 24), dtype=bool)
        known_free[2, 2:9] = True
        known_free[2:17, 8] = True
        known_free[16, 8:19] = True
        target = (18, 16)

        self.assertEqual(known_free_path((2, 2), target, known_free), ())

        update = graph.connect_known_free_corridor(
            target,
            known_free,
            search_distance=24,
        )

        self.assertEqual(update.status, ROUTE_OK)
        self.assertEqual(update.sampled_waypoints[0], (2, 2))
        self.assertEqual(update.sampled_waypoints[-1], target)
        self.assertTrue(update.added_edges)
        self.assertTrue(
            all(
                known_free[y, x]
                for edge in update.added_edges
                for x, y in edge.path
            )
        )
        self.assertTrue(graph.find_route((2, 2), target, known_free).found)

    def test_failed_corridor_bridge_does_not_mutate_graph_or_version(self) -> None:
        graph = WaypointGraph(
            spacing=5,
            merge_radius=0,
            connector_distance=3,
        )
        graph.add_waypoint((2, 2), source="travelled")
        disconnected = np.zeros((12, 24), dtype=bool)
        disconnected[1:5, 1:6] = True
        disconnected[1:5, 17:22] = True
        before = graph.snapshot()

        disconnected_update = graph.connect_known_free_corridor(
            (20, 2),
            disconnected,
            search_distance=20,
        )

        self.assertEqual(disconnected_update.status, "no_connector")
        self.assertFalse(disconnected_update.changed)
        self.assertEqual(graph.snapshot(), before)

        unknown_update = graph.connect_known_free_corridor(
            (12, 8),
            disconnected,
            search_distance=20,
        )

        self.assertEqual(unknown_update.status, "position_unknown")
        self.assertFalse(unknown_update.changed)
        self.assertEqual(graph.snapshot(), before)

    def test_repeated_corridor_bridge_is_a_version_stable_noop(self) -> None:
        graph = WaypointGraph(
            spacing=8,
            merge_radius=0,
            connector_distance=4,
        )
        graph.add_waypoint((2, 4), source="travelled")
        known_free = np.zeros((12, 40), dtype=bool)
        known_free[4, 2:35] = True

        first = graph.connect_known_free_corridor(
            (34, 4),
            known_free,
            search_distance=33,
        )
        changed_version = graph.version
        second = graph.connect_known_free_corridor(
            (34, 4),
            known_free,
            search_distance=33,
        )

        self.assertTrue(first.changed)
        self.assertFalse(second.changed)
        self.assertEqual(graph.version, changed_version)


class WaypointGraphRoutingTests(unittest.TestCase):
    def test_missing_goal_connector_returns_before_edge_revalidation(self) -> None:
        graph = WaypointGraph(connector_distance=4, merge_radius=0)
        graph.add_waypoint((2, 2), source="home")
        known_free = np.ones((12, 40), dtype=bool)
        graph.connect_known_free_waypoint((6, 2), known_free)

        with patch.object(
            graph,
            "_append_route_edge",
            wraps=graph._append_route_edge,
        ) as append_edge:
            route = graph.find_route((2, 2), (35, 2), known_free)

        self.assertEqual(route.status, ROUTE_NO_GOAL_CONNECTOR)
        append_edge.assert_not_called()

    def test_dijkstra_routes_over_travelled_highway_and_returns_first_segment(self) -> None:
        graph = WaypointGraph(spacing=10, connector_distance=5)
        graph.register_travelled_path(((5, 5), (15, 5), (25, 5)))
        known_free = np.ones((16, 32), dtype=bool)

        route = graph.find_route((3, 5), (27, 5), known_free)

        self.assertEqual(route.status, ROUTE_OK)
        self.assertTrue(route.found)
        self.assertEqual(route.waypoints, ((3, 5), (5, 5), (15, 5), (25, 5), (27, 5)))
        self.assertEqual(route.first_segment_path, ((3, 5), (4, 5), (5, 5)))
        self.assertEqual(route.first_segment_source, EDGE_KNOWN_FREE_CONNECTOR)
        self.assertAlmostEqual(route.first_segment_cost, 2.0)
        self.assertAlmostEqual(route.cost, 24.0)

    def test_exact_start_node_skips_zero_connector_for_first_segment(self) -> None:
        graph = WaypointGraph(spacing=10, connector_distance=3)
        graph.register_travelled_path(((5, 5), (15, 5), (25, 5)))
        known_free = np.ones((16, 32), dtype=bool)

        route = graph.find_route((5, 5), (27, 5), known_free)

        self.assertEqual(route.status, ROUTE_OK)
        self.assertEqual(route.first_segment_source, EDGE_TRAVELLED)
        self.assertEqual(route.first_segment_path[0], (5, 5))
        self.assertEqual(route.first_segment_path[-1], (15, 5))

    def test_travelled_edge_remains_usable_when_requester_mask_hides_interior(self) -> None:
        graph = WaypointGraph(spacing=10, connector_distance=2)
        graph.register_travelled_path(((2, 3), (12, 3), (22, 3)))
        requester = np.zeros((8, 26), dtype=bool)
        requester[3, 2] = True
        requester[3, 22:25] = True

        route = graph.find_route((2, 3), (24, 3), requester)

        self.assertEqual(route.status, ROUTE_OK)
        self.assertEqual(route.first_segment_source, EDGE_TRAVELLED)

    def test_slam_los_edge_is_revalidated_and_can_disconnect_route(self) -> None:
        graph = WaypointGraph(connector_distance=7, merge_radius=0)
        graph.add_waypoint((2, 2), source="home")
        registration_mask = np.ones((8, 16), dtype=bool)
        graph.connect_known_free_waypoint((8, 2), registration_mask)
        requester = registration_mask.copy()
        requester[2, 5] = False

        route = graph.find_route((2, 2), (8, 2), requester)

        self.assertEqual(route.status, ROUTE_DISCONNECTED)

    def test_reports_unknown_and_missing_connectors(self) -> None:
        graph = WaypointGraph(connector_distance=3)
        graph.add_waypoint((10, 2))
        known_free = np.ones((8, 20), dtype=bool)
        known_free[2, 18] = False

        unknown_goal = graph.find_route((10, 2), (18, 2), known_free)
        missing_start = graph.find_route((1, 6), (10, 2), known_free)

        self.assertEqual(unknown_goal.status, ROUTE_GOAL_UNKNOWN)
        self.assertEqual(missing_start.status, ROUTE_NO_START_CONNECTOR)


class WaypointGraphConcurrencyTests(unittest.TestCase):
    def test_concurrent_registration_keeps_unique_nodes_and_edges(self) -> None:
        graph = WaypointGraph(spacing=5)
        path = ((0, 0), (20, 0))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            updates = tuple(executor.map(graph.register_travelled_path, [path] * 64))

        self.assertEqual(graph.counts(), (5, 4))
        self.assertEqual(sum(len(update.added_waypoints) for update in updates), 5)
        self.assertEqual(sum(len(update.added_edges) for update in updates), 4)
        snapshot = graph.snapshot()
        self.assertEqual(len({node.position for node in snapshot.nodes}), 5)
        self.assertTrue(
            all(math.isfinite(edge.cost) and edge.cost > 0 for edge in snapshot.edges)
        )


if __name__ == "__main__":
    unittest.main()
