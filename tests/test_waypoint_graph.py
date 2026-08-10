import concurrent.futures
import math
import unittest
from unittest.mock import patch

import numpy as np

import navigation.waypoint_graph as waypoint_graph_module
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
    GraphDelta,
    WaypointGraph,
    WaypointRole,
    bresenham_path,
    known_free_path,
    validate_known_free_path,
)
from navigation.trail_accumulator import StrategicTrailAccumulator


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
        graph = WaypointGraph()
        update = graph.register_travelled_section(((0, 0), (10, 0)))
        snapshot = graph.snapshot()

        self.assertEqual(
            tuple(node.position for node in update.added_waypoints),
            ((0, 0), (10, 0)),
        )
        self.assertEqual(update.added_nodes, update.added_waypoints)
        self.assertTrue(update.changed)
        self.assertEqual(snapshot.node_count, 2)
        self.assertEqual(snapshot.edge_count, 1)
        self.assertEqual(graph.counts(), (2, 1))
        self.assertTrue(all(edge.source == EDGE_TRAVELLED for edge in snapshot.edges))


class WaypointGraphStrategicTrailTests(unittest.TestCase):
    def test_accumulator_confirms_slam_clearance_local_minimum(self) -> None:
        accumulator = StrategicTrailAccumulator((20, 32))
        known_free = np.zeros((65, 101), dtype=bool)
        for x in range(101):
            half_width = min(20, 6 + abs(x - 50))
            known_free[32 - half_width : 33 + half_width, x] = True

        sections = accumulator.append(((20, 32), (80, 32)), known_free)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].path[-1], (50, 32))
        self.assertEqual(sections[0].end_roles, {WaypointRole.CHOKEPOINT})

    def test_short_section_has_only_strategic_endpoints_and_exact_edge(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        path = ((2, 2), (8, 2), (8, 8), (14, 8))

        update = graph.register_travelled_section(
            path,
            start_role=WaypointRole.HOME,
            end_role=WaypointRole.TURN,
        )

        self.assertEqual(graph.counts(), (2, 1))
        self.assertEqual(update.sampled_waypoints, ((2, 2), (14, 8)))
        self.assertEqual(update.added_edges[0].path, tuple(
            (point[0], point[1]) for point in (
                (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
                (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
                (8, 8), (9, 8), (10, 8), (11, 8), (12, 8),
                (13, 8), (14, 8),
            )
        ))
        self.assertEqual(graph.snapshot().nodes[-1].roles, {WaypointRole.TURN})

    def test_replaying_exact_section_is_revision_stable(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        path = ((0, 0), (8, 0), (8, 8))
        graph.register_travelled_section(
            path, end_role=WaypointRole.RECOVERY_ANCHOR,
        )
        revision = graph.topology_revision

        replay = graph.register_travelled_section(
            path, end_role=WaypointRole.RECOVERY_ANCHOR,
        )

        self.assertFalse(replay.changed)
        self.assertEqual(graph.topology_revision, revision)
        self.assertEqual(graph.counts(), (2, 1))

    def test_overlapping_section_splits_and_reuses_travelled_geometry(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        graph.register_travelled_section(((0, 4), (20, 4)))

        overlap = graph.register_travelled_section(
            ((5, 4), (15, 4)),
            end_role=WaypointRole.RECOVERY_ANCHOR,
        )

        snapshot = graph.snapshot()
        self.assertEqual({node.position for node in snapshot.nodes}, {
            (0, 4), (5, 4), (15, 4), (20, 4),
        })
        self.assertEqual(snapshot.edge_count, 3)
        self.assertEqual(
            sum(edge.cost for edge in snapshot.edges),
            20.0,
        )
        active_edge_ids = {edge.id for edge in snapshot.edges}
        self.assertTrue(set(overlap.delta.added_edge_ids) <= active_edge_ids)
        roles = {node.position: node.roles for node in snapshot.nodes}
        self.assertEqual(roles[(5, 4)], {WaypointRole.JUNCTION})
        self.assertEqual(roles[(15, 4)], {
            WaypointRole.JUNCTION,
            WaypointRole.RECOVERY_ANCHOR,
        })

    def test_near_parallel_trails_remain_separate(self) -> None:
        graph = WaypointGraph(merge_radius=8, spatial_hash_cell=32)
        graph.register_travelled_section(((2, 5), (18, 5)))

        graph.register_travelled_section(((2, 7), (18, 7)))

        self.assertEqual(graph.counts(), (4, 2))
        self.assertEqual(
            {edge.path for edge in graph.snapshot().edges},
            {
                bresenham_path((2, 5), (18, 5)),
                bresenham_path((2, 7), (18, 7)),
            },
        )

    def test_existing_home_acquires_role_without_changing_id(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        graph.add_waypoint((1, 1), source="home")
        home_id = graph.snapshot().nodes[0].id

        graph.register_travelled_section(
            ((1, 1), (10, 1)),
            start_role=WaypointRole.JUNCTION,
            end_role=WaypointRole.RECOVERY_ANCHOR,
        )

        home = next(node for node in graph.snapshot().nodes if node.id == home_id)
        self.assertEqual(home.roles, {WaypointRole.HOME, WaypointRole.JUNCTION})


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

    def test_corridor_bridge_stores_one_complete_scoped_polyline(self) -> None:
        graph = WaypointGraph(
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
            ((2, 4), (34, 4)),
        )
        self.assertEqual(len(update.added_edges), 1)
        self.assertEqual(update.added_edges[0].source, EDGE_KNOWN_FREE_CORRIDOR)
        self.assertEqual(update.added_edges[0].path, bresenham_path((2, 4), (34, 4)))
        self.assertEqual(
            update.added_waypoints[0].roles,
            {WaypointRole.FRONTIER_GATEWAY},
        )
        self.assertTrue(
            graph.find_route((2, 4), (34, 4), known_free).found
        )

    def test_corridor_bridge_follows_curved_known_free_passage(self) -> None:
        graph = WaypointGraph(
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
    def test_close_non_los_connector_budget_fits_route_lookup_gate(self) -> None:
        self.assertLessEqual(
            waypoint_graph_module.LOCAL_CONNECTOR_TIME_BUDGET_SECONDS,
            0.004,
        )

    def test_failed_close_non_los_search_reports_connector_astar_attempt(self) -> None:
        graph = WaypointGraph(connector_distance=8, merge_radius=0)
        graph.add_waypoint((2, 4), source="home")
        known_free = np.ones((9, 12), dtype=bool)
        known_free[:, 5] = False

        route = graph.find_route((2, 4), (8, 4), known_free)

        self.assertEqual(route.status, ROUTE_NO_GOAL_CONNECTOR)
        self.assertEqual(route.connector_astar_calls, 1)

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
        graph = WaypointGraph(connector_distance=5)
        graph.register_travelled_section(((5, 5), (15, 5), (25, 5)))
        known_free = np.ones((16, 32), dtype=bool)

        route = graph.find_route((3, 5), (27, 5), known_free)

        self.assertEqual(route.status, ROUTE_OK)
        self.assertTrue(route.found)
        self.assertEqual(route.waypoints, ((3, 5), (5, 5), (25, 5), (27, 5)))
        self.assertEqual(route.first_segment_path, ((3, 5), (4, 5), (5, 5)))
        self.assertEqual(route.first_segment_source, EDGE_KNOWN_FREE_CONNECTOR)
        self.assertAlmostEqual(route.first_segment_cost, 2.0)
        self.assertAlmostEqual(route.cost, 24.0)

    def test_exact_start_node_skips_zero_connector_for_first_segment(self) -> None:
        graph = WaypointGraph(connector_distance=3)
        graph.register_travelled_section(((5, 5), (15, 5), (25, 5)))
        known_free = np.ones((16, 32), dtype=bool)

        route = graph.find_route((5, 5), (27, 5), known_free)

        self.assertEqual(route.status, ROUTE_OK)
        self.assertEqual(route.first_segment_source, EDGE_TRAVELLED)
        self.assertEqual(route.first_segment_path[0], (5, 5))
        self.assertEqual(route.first_segment_path[-1], (25, 5))

    def test_far_known_free_line_of_sight_shortcuts_graph_detour(self) -> None:
        graph = WaypointGraph(connector_distance=4, merge_radius=0)
        graph.register_travelled_section(
            ((2, 2), (2, 10), (30, 10), (30, 2))
        )
        known_free = np.ones((14, 34), dtype=bool)

        route = graph.find_route((2, 2), (30, 2), known_free)

        self.assertEqual(route.status, ROUTE_OK)
        self.assertEqual(route.waypoints, ((2, 2), (30, 2)))
        self.assertEqual(route.first_segment_source, EDGE_SLAM_LOS)
        self.assertEqual(route.segment_sources, (EDGE_SLAM_LOS,))
        self.assertEqual(route.first_segment_path, bresenham_path((2, 2), (30, 2)))
        self.assertAlmostEqual(route.cost, 28.0)

    def test_far_blocked_line_of_sight_keeps_known_graph_route(self) -> None:
        graph = WaypointGraph(connector_distance=4, merge_radius=0)
        graph.register_travelled_section(
            ((2, 2), (2, 10), (30, 10), (30, 2))
        )
        known_free = np.ones((14, 34), dtype=bool)
        known_free[2, 16] = False

        route = graph.find_route((2, 2), (30, 2), known_free)

        self.assertEqual(route.status, ROUTE_OK)
        self.assertEqual(route.first_segment_source, EDGE_TRAVELLED)
        self.assertGreater(route.cost, 28.0)

    def test_close_non_los_target_uses_one_bounded_belief_only_path(self) -> None:
        graph = WaypointGraph(connector_distance=12, merge_radius=0)
        known_free = np.ones((12, 12), dtype=bool)
        known_free[1:9, 4] = False

        route = graph.find_route((2, 5), (7, 5), known_free)

        self.assertTrue(route.found)
        self.assertEqual(route.waypoints, ((2, 5), (7, 5)))
        self.assertEqual(route.segment_paths, (route.first_segment_path,))
        self.assertEqual(route.first_segment_path[0], (2, 5))
        self.assertEqual(route.first_segment_path[-1], (7, 5))
        self.assertTrue(validate_known_free_path(route.first_segment_path, known_free))
        self.assertGreater(route.cost, 5.0)
        self.assertEqual(route.connector_astar_calls, 1)

    def test_travelled_edge_remains_usable_when_requester_mask_hides_interior(self) -> None:
        graph = WaypointGraph(connector_distance=2)
        graph.register_travelled_section(((2, 3), (12, 3), (22, 3)))
        requester = np.zeros((8, 26), dtype=bool)
        requester[3, 2] = True
        requester[3, 22:25] = True

        route = graph.find_route((2, 3), (24, 3), requester)

        self.assertEqual(route.status, ROUTE_OK)
        self.assertEqual(route.first_segment_source, EDGE_TRAVELLED)

    def test_invalid_slam_los_edge_can_be_replaced_by_close_belief_connector(self) -> None:
        graph = WaypointGraph(connector_distance=7, merge_radius=0)
        graph.add_waypoint((2, 2), source="home")
        registration_mask = np.ones((8, 16), dtype=bool)
        graph.connect_known_free_waypoint((8, 2), registration_mask)
        requester = registration_mask.copy()
        requester[2, 5] = False

        route = graph.find_route((2, 2), (8, 2), requester)

        self.assertEqual(route.status, ROUTE_OK)
        self.assertEqual(route.first_segment_source, EDGE_KNOWN_FREE_CONNECTOR)
        self.assertNotIn((5, 2), route.first_segment_path)

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
        graph = WaypointGraph()
        path = ((0, 0), (20, 0))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            updates = tuple(
                executor.map(graph.register_travelled_section, [path] * 64)
            )

        self.assertEqual(graph.counts(), (2, 1))
        self.assertEqual(sum(len(update.added_waypoints) for update in updates), 2)
        self.assertEqual(sum(len(update.added_edges) for update in updates), 1)
        snapshot = graph.snapshot()
        self.assertEqual(len({node.position for node in snapshot.nodes}), 2)
        self.assertTrue(
            all(math.isfinite(edge.cost) and edge.cost > 0 for edge in snapshot.edges)
        )


class WaypointGraphStrategicCoreTests(unittest.TestCase):
    def test_ids_are_stable_monotonic_and_one_operation_has_one_delta(self) -> None:
        graph = WaypointGraph(merge_radius=0)

        update = graph.register_travelled_section(((0, 0), (10, 0)))
        snapshot = graph.snapshot()

        self.assertIsInstance(update.delta, GraphDelta)
        self.assertEqual(update.delta.revision, 1)
        self.assertEqual(graph.topology_revision, 1)
        self.assertEqual(tuple(node.id for node in snapshot.nodes), (1, 2))
        self.assertEqual(tuple(edge.id for edge in snapshot.edges), (1,))
        self.assertEqual(update.delta.added_node_ids, (1, 2))
        self.assertEqual(update.delta.added_edge_ids, (1,))

        old_ids = tuple(node.id for node in snapshot.nodes)
        graph.add_waypoint((20, 0))
        self.assertEqual(tuple(node.id for node in graph.snapshot().nodes[:2]), old_ids)
        self.assertEqual(graph.snapshot().nodes[-1].id, 3)

    def test_wall_separated_nodes_in_same_spatial_bucket_do_not_merge(self) -> None:
        graph = WaypointGraph(merge_radius=8, spatial_hash_cell=32)
        graph.add_waypoint((8, 8))
        known_free = np.ones((32, 32), dtype=bool)
        known_free[:, 11] = False

        canonical, added = graph.add_waypoint((14, 8), known_free=known_free)

        self.assertTrue(added)
        self.assertEqual(canonical, (14, 8))
        self.assertEqual(graph.node_count, 2)

    def test_split_and_collapse_preserve_exact_polyline_and_cost(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        update = graph.register_travelled_section(
            ((0, 0), (4, 0), (4, 4), (8, 4))
        )
        original = update.added_edges[0]

        split = graph.split_edge(original.id, (4, 2))
        split_edges = tuple(
            edge for edge in graph.snapshot().edges
            if edge.id in split.added_edge_ids
        )

        self.assertEqual(split.revision, 2)
        self.assertEqual(split.edge_replacements[original.id], tuple(edge.id for edge in split_edges))
        first = next(edge for edge in split_edges if original.start_id in {edge.start_id, edge.end_id})
        second = next(edge for edge in split_edges if edge.id != first.id)
        first_path = first.path if first.start_id == original.start_id else tuple(reversed(first.path))
        junction_id = first.end_id if first.start_id == original.start_id else first.start_id
        second_path = second.path if second.start_id == junction_id else tuple(reversed(second.path))
        reconstructed = first_path + second_path[1:]
        self.assertEqual(reconstructed, original.path)
        self.assertAlmostEqual(sum(edge.cost for edge in split_edges), original.cost)

        junction_id = next(
            node.id for node in graph.snapshot().nodes if node.position == (4, 2)
        )
        protected = graph.collapse_node(junction_id)
        self.assertEqual(protected.added_edge_ids, ())

        collapse_graph = WaypointGraph(merge_radius=0)
        first_section = collapse_graph.register_travelled_section(
            ((0, 0), (4, 0))
        )
        second_section = collapse_graph.register_travelled_section(
            ((4, 0), (8, 0))
        )
        middle_id = next(
            node.id for node in collapse_graph.snapshot().nodes
            if node.position == (4, 0)
        )
        collapse = collapse_graph.collapse_node(middle_id)
        collapsed = next(
            edge for edge in collapse_graph.snapshot().edges
            if edge.id in collapse.added_edge_ids
        )
        self.assertEqual(collapsed.path, bresenham_path((0, 0), (8, 0)))
        self.assertAlmostEqual(
            collapsed.cost,
            first_section.added_edges[0].cost + second_section.added_edges[0].cost,
        )

    def test_batch_collapse_removes_inactive_turn_but_protects_route_and_anchors(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        graph.register_travelled_section(
            ((0, 0), (4, 0)), end_role=WaypointRole.TURN,
        )
        graph.register_travelled_section(
            ((4, 0), (8, 0)), end_role=WaypointRole.RECOVERY_ANCHOR,
        )
        graph.register_travelled_section(((8, 0), (12, 0)))
        snapshot = graph.snapshot()
        turn_id = next(node.id for node in snapshot.nodes if node.position == (4, 0))
        anchor_id = next(node.id for node in snapshot.nodes if node.position == (8, 0))

        protected = graph.collapse_inactive_degree_two_nodes(
            active_route_node_ids=(turn_id,),
        )
        self.assertEqual(protected, ())

        collapsed = graph.collapse_inactive_degree_two_nodes()

        self.assertTrue(collapsed)
        remaining = {node.id for node in graph.snapshot().nodes}
        self.assertNotIn(turn_id, remaining)
        self.assertIn(anchor_id, remaining)

    def test_nearby_connected_junction_contracts_into_home_with_exact_paths(self) -> None:
        graph = WaypointGraph(merge_radius=4)
        graph.add_waypoint((0, 0), source="home")
        graph.register_travelled_section(
            ((0, 0), (1, 0)), end_role=WaypointRole.JUNCTION,
        )
        graph.register_travelled_section(((1, 0), (6, 0)))
        graph.register_travelled_section(((1, 0), (1, 6)))
        before = graph.snapshot()
        home_id = next(
            node.id for node in before.nodes
            if WaypointRole.HOME in node.roles
        )
        junction_id = next(
            node.id for node in before.nodes
            if node.position == (1, 0)
        )

        deltas = graph.collapse_inactive_nearby_nodes(radius=8.0)

        self.assertTrue(deltas)
        self.assertIn(junction_id, {
            node_id for delta in deltas for node_id in delta.retired_node_ids
        })
        after = graph.snapshot()
        self.assertIn(home_id, {node.id for node in after.nodes})
        self.assertEqual(after.node_count, before.node_count - 1)
        node_by_position = {node.position: node.id for node in after.nodes}
        for target in ((6, 0), (1, 6)):
            edge = next(
                edge for edge in after.edges
                if {edge.start_id, edge.end_id}
                == {home_id, node_by_position[target]}
            )
            path = (
                edge.path
                if edge.start_id == home_id else tuple(reversed(edge.path))
            )
            self.assertEqual(path[0], (0, 0))
            self.assertEqual(path[-1], target)
            self.assertIn((1, 0), path)

    def test_nearby_contraction_protects_active_routes_and_separate_anchors(self) -> None:
        graph = WaypointGraph(merge_radius=4)
        graph.register_travelled_section(
            ((0, 0), (1, 0)),
            start_role=WaypointRole.HOME,
            end_role=WaypointRole.RECOVERY_ANCHOR,
        )

        self.assertEqual(graph.collapse_inactive_nearby_nodes(radius=8.0), ())
        self.assertEqual(graph.node_count, 2)

        active_graph = WaypointGraph(merge_radius=4)
        active_graph.register_travelled_section(
            ((0, 0), (1, 0)), end_role=WaypointRole.JUNCTION,
        )
        active = active_graph.snapshot()
        active_junction = next(
            node.id for node in active.nodes
            if WaypointRole.JUNCTION in node.roles
        )
        self.assertEqual(active_graph.collapse_inactive_nearby_nodes(
            radius=8.0,
            active_route_node_ids=(active_junction,),
            active_route_edge_ids=(active.edges[0].id,),
        ), ())
        self.assertEqual(active_graph.node_count, 2)

    def test_nearby_unconnected_nodes_never_consolidate_by_proximity(self) -> None:
        graph = WaypointGraph(merge_radius=4)
        graph.add_waypoint((4, 4), source="turn")
        graph.add_waypoint((5, 4), source="junction")

        self.assertEqual(graph.collapse_inactive_nearby_nodes(radius=8.0), ())
        self.assertEqual(graph.node_count, 2)

    def test_roleless_inactive_trail_leaf_is_retired_without_touching_anchor(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        update = graph.register_travelled_section(
            ((2, 2), (12, 2)),
            end_role=WaypointRole.TURN,
        )
        snapshot = graph.snapshot()
        leaf = next(node for node in snapshot.nodes if not node.roles)
        anchor = next(node for node in snapshot.nodes if node.roles)
        edge = snapshot.edges[0]

        deltas = graph.retire_inactive_orphan_trail_leaves()

        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0].retired_node_ids, (leaf.id,))
        self.assertEqual(deltas[0].retired_edge_ids, (edge.id,))
        self.assertEqual(deltas[0].edge_replacements[edge.id], ())
        remaining = graph.snapshot()
        self.assertEqual(tuple(node.id for node in remaining.nodes), (anchor.id,))
        self.assertEqual(remaining.edges, ())
        self.assertTrue(update.changed)

    def test_orphan_trail_leaf_retirement_protects_active_identity(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        graph.register_travelled_section(
            ((2, 2), (12, 2)),
            end_role=WaypointRole.TURN,
        )
        snapshot = graph.snapshot()
        leaf = next(node for node in snapshot.nodes if not node.roles)
        edge = snapshot.edges[0]

        self.assertEqual(
            graph.retire_inactive_orphan_trail_leaves(
                active_route_node_ids=(leaf.id,),
            ),
            (),
        )
        self.assertEqual(
            graph.retire_inactive_orphan_trail_leaves(
                active_route_edge_ids=(edge.id,),
            ),
            (),
        )
        self.assertEqual(graph.counts(), (2, 1))

    def test_retiring_orphan_gateway_removes_corridor_but_respects_active_route(self) -> None:
        graph = WaypointGraph(connector_distance=8, merge_radius=0)
        graph.add_waypoint((1, 2), source="home")
        known_free = np.ones((8, 40), dtype=bool)
        update = graph.connect_known_free_corridor(
            (30, 2), known_free, search_distance=32.0,
        )
        gateway = next(
            node for node in graph.snapshot().nodes
            if WaypointRole.FRONTIER_GATEWAY in node.roles
        )
        corridor = next(
            edge for edge in graph.snapshot().edges
            if edge.source == EDGE_KNOWN_FREE_CORRIDOR
        )

        protected = graph.retire_frontier_gateway(
            gateway.id,
            active_route_edge_ids=(corridor.id,),
        )
        self.assertEqual(protected.retired_node_ids, ())

        retired = graph.retire_frontier_gateway(gateway.id)

        self.assertEqual(retired.retired_node_ids, (gateway.id,))
        self.assertIn(corridor.id, retired.retired_edge_ids)
        self.assertNotIn(gateway.id, {node.id for node in graph.snapshot().nodes})

    def test_travelled_crossing_splits_and_connects_in_one_revision(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        original = graph.register_travelled_section(
            ((0, 5), (10, 5))
        ).added_edges[0]

        crossing = graph.register_travelled_section(((5, 0), (5, 10)))

        self.assertEqual(crossing.delta.revision, 2)
        self.assertIn(original.id, crossing.delta.retired_edge_ids)
        self.assertEqual(len(crossing.delta.edge_replacements[original.id]), 2)
        junction = next(node for node in graph.snapshot().nodes if node.position == (5, 5))
        incident = [
            edge for edge in graph.snapshot().edges
            if junction.id in {edge.start_id, edge.end_id}
        ]
        self.assertEqual(len(incident), 4)

    def test_component_rejection_precedes_shortest_path_tree(self) -> None:
        graph = WaypointGraph(connector_distance=2, merge_radius=0)
        graph.register_travelled_section(((2, 2), (12, 2)))
        graph.register_travelled_section(((30, 2), (40, 2)))
        known_free = np.ones((8, 44), dtype=bool)

        with patch.object(graph, "_build_reverse_tree_locked") as build_tree:
            route = graph.find_route((2, 2), (40, 2), known_free)

        self.assertEqual(route.status, ROUTE_DISCONNECTED)
        build_tree.assert_not_called()

    def test_repeated_routes_hit_revision_keyed_lru_cache(self) -> None:
        graph = WaypointGraph(connector_distance=3, merge_radius=0)
        graph.register_travelled_section(((2, 2), (12, 2), (22, 2)))
        known_free = np.ones((8, 28), dtype=bool)

        first = graph.find_route(
            (2, 2), (24, 2), known_free,
            requester_id="d0", requester_knowledge_revision=7,
        )
        second = graph.find_route(
            (2, 2), (24, 2), known_free,
            requester_id="d0", requester_knowledge_revision=7,
        )

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(graph.route_tree_builds, 1)
        self.assertEqual(graph.route_cache_hits, 1)

        graph.add_waypoint((27, 6))
        third = graph.find_route(
            (2, 2), (24, 2), known_free,
            requester_id="d0", requester_knowledge_revision=7,
        )
        self.assertFalse(third.cache_hit)
        self.assertEqual(graph.route_tree_builds, 2)

    def test_multiple_goal_connectors_build_one_reverse_tree(self) -> None:
        graph = WaypointGraph(connector_distance=14, merge_radius=0)
        graph.register_travelled_section(
            ((2, 2), (12, 2)),
            start_role=WaypointRole.HOME,
            end_role=WaypointRole.TURN,
        )
        graph.register_travelled_section(
            ((12, 2), (22, 2)),
            start_role=WaypointRole.TURN,
            end_role=WaypointRole.TURN,
        )
        graph.register_travelled_section(
            ((22, 2), (32, 2)),
            start_role=WaypointRole.TURN,
            end_role=WaypointRole.TURN,
        )
        known_free = np.ones((8, 40), dtype=bool)

        route = graph.find_route(
            (0, 2),
            (35, 2),
            known_free,
            requester_id="d0",
        )

        self.assertTrue(route.found)
        self.assertEqual(route.cost, 35.0)
        self.assertEqual(graph.route_tree_builds, 1)

    def test_requester_revision_change_reuses_tree_after_scoped_revalidation(self) -> None:
        graph = WaypointGraph(connector_distance=3, merge_radius=0)
        graph.register_travelled_section(((2, 2), (12, 2), (22, 2)))
        known_free = np.ones((8, 28), dtype=bool)

        first = graph.find_route(
            (2, 2), (24, 2), known_free,
            requester_id="d0", requester_knowledge_revision=7,
        )
        second = graph.find_route(
            (2, 2), (24, 2), known_free,
            requester_id="d0", requester_knowledge_revision=8,
        )

        self.assertTrue(first.found)
        self.assertTrue(second.cache_hit)
        self.assertEqual(graph.route_tree_builds, 1)

    def test_route_ids_are_stable_nonzero_and_never_reused(self) -> None:
        graph = WaypointGraph(connector_distance=3, merge_radius=0)
        graph.add_waypoint((2, 2), source="home")
        graph.register_travelled_section(((2, 2), (12, 2), (22, 2)))
        known_free = np.ones((8, 28), dtype=bool)

        first = graph.find_route((2, 2), (24, 2), known_free)
        second = graph.find_route((2, 2), (24, 2), known_free)
        failed = graph.find_route((2, 2), (27, 7), known_free)

        self.assertGreater(first.id, 0)
        self.assertGreater(second.id, first.id)
        self.assertGreater(failed.id, second.id)

    def test_cached_tree_revalidates_only_selected_requester_edges(self) -> None:
        graph = WaypointGraph(connector_distance=8, merge_radius=0)
        graph.add_waypoint((2, 2), source="home")
        registration = np.ones((8, 20), dtype=bool)
        graph.connect_known_free_waypoint((8, 2), registration)
        graph.connect_known_free_waypoint((14, 2), registration)

        first = graph.find_route(
            (2, 2), (14, 2), registration,
            requester_id="d0", requester_knowledge_revision=3,
        )
        unrelated_change = registration.copy()
        unrelated_change[7, 19] = False
        reused = graph.find_route(
            (2, 2), (14, 2), unrelated_change,
            requester_id="d0", requester_knowledge_revision=3,
        )
        builds_before_invalidation = graph.route_tree_builds
        blocked = registration.copy()
        blocked[2, 5] = False
        invalidated = graph.find_route(
            (2, 2), (14, 2), blocked,
            requester_id="d0", requester_knowledge_revision=3,
        )

        self.assertTrue(first.found)
        self.assertTrue(reused.cache_hit)
        self.assertEqual(invalidated.status, ROUTE_DISCONNECTED)
        self.assertGreater(graph.route_tree_builds, builds_before_invalidation)

    def test_concurrent_snapshot_contains_coherent_id_adjacency(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        paths = [((0, y), (20, y)) for y in range(8)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            snapshots = tuple(executor.map(
                lambda path: (
                    graph.register_travelled_section(path), graph.snapshot()
                )[1],
                paths,
            ))

        for snapshot in snapshots:
            node_ids = {node.id for node in snapshot.nodes}
            self.assertEqual(len(node_ids), len(snapshot.nodes))
            self.assertTrue(all(
                edge.start_id in node_ids and edge.end_id in node_ids
                for edge in snapshot.edges
            ))

    def test_concurrent_insertion_routing_split_and_snapshots_stay_coherent(self) -> None:
        graph = WaypointGraph(connector_distance=2, merge_radius=0)
        graph.register_travelled_section(((0, 5), (10, 5)))
        split_target = graph.register_travelled_section(
            ((0, 20), (10, 20))
        ).added_edges[0]
        known_free = np.ones((24, 14), dtype=bool)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    graph.register_travelled_section, ((5, 0), (5, 10))
                ),
                executor.submit(graph.split_edge, split_target.id, (5, 20)),
                executor.submit(graph.find_route, (0, 5), (10, 5), known_free),
                executor.submit(graph.snapshot),
            ]
            results = [future.result() for future in futures]

        self.assertTrue(results[2].found)
        snapshot = graph.snapshot()
        node_ids = {node.id for node in snapshot.nodes}
        self.assertTrue(all(
            edge.start_id in node_ids and edge.end_id in node_ids
            for edge in snapshot.edges
        ))


if __name__ == "__main__":
    unittest.main()
