import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from navigation.frontier_clusters import (
    AssignmentRegistry,
    FrontierComponent,
    FrontierClusterRegistry,
    FrontierExtractor,
    FrontierGatewayManager,
    select_accessible_frontier_waypoint,
)
from navigation.waypoint_graph import WaypointGraph, WaypointRole


def belief(shape=(12, 12), *, free=(), occupied=(), version=1):
    occupancy = np.full(shape, UNKNOWN, dtype=np.int8)
    confidence = np.zeros(shape, dtype=np.float32)
    for x, y in free:
        occupancy[y, x] = FREE
        confidence[y, x] = 0.9
    for x, y in occupied:
        occupancy[y, x] = OCCUPIED
        confidence[y, x] = 0.9
    return SlamSnapshot(occupancy, confidence, version=version)


class FrontierExtractorTests(unittest.TestCase):
    def test_mask_and_components_depend_only_on_slam_belief(self):
        snapshot = belief(free=((4, 4), (5, 4), (4, 5), (5, 5)))
        extractor = FrontierExtractor(confidence_threshold=0.6)

        first = extractor.refresh(snapshot)
        second = FrontierExtractor(0.6).refresh(snapshot)

        self.assertTrue(np.array_equal(first.masks.frontier, second.masks.frontier))
        self.assertEqual(first.components, second.components)
        self.assertEqual(
            {component.representative for component in first.components},
            {(4, 4)},
        )

    def test_confident_occupancy_defines_free_occupied_and_unknown(self):
        snapshot = belief(free=((2, 2),), occupied=((3, 2),))
        snapshot.confidence[4, 4] = 0.59
        snapshot.occupancy[4, 4] = FREE

        result = FrontierExtractor(0.6).refresh(snapshot)

        self.assertTrue(result.masks.free[2, 2])
        self.assertTrue(result.masks.occupied[2, 3])
        self.assertTrue(result.masks.unknown[4, 4])

    def test_dirty_refresh_is_equivalent_to_forced_full_refresh(self):
        initial = belief(free=tuple((x, 5) for x in range(2, 8)), version=1)
        changed = belief(
            free=tuple((x, 5) for x in range(2, 9)) + ((8, 6),),
            version=2,
        )
        incremental = FrontierExtractor(0.6)
        incremental.refresh(initial)

        dirty = incremental.refresh(changed, dirty_regions=((7, 4, 9, 7),))
        full = FrontierExtractor(0.6).refresh(changed, force_full=True)

        self.assertTrue(np.array_equal(dirty.masks.frontier, full.masks.frontier))
        self.assertEqual(dirty.components, full.components)
        self.assertFalse(dirty.full_rebuild)

    def test_same_version_reuses_cached_result(self):
        extractor = FrontierExtractor(0.6)
        first = extractor.refresh(belief(free=((2, 2),), version=4))
        second = extractor.refresh(belief(free=((9, 9),), version=4))

        self.assertIs(first, second)

    def test_isolated_unknown_sampling_gap_is_not_a_frontier(self):
        free = tuple(
            (x, y)
            for y in range(9)
            for x in range(9)
            if (x, y) != (4, 4)
        )

        result = FrontierExtractor(0.6).refresh(belief(
            shape=(9, 9),
            free=free,
        ))

        self.assertFalse(np.any(result.masks.frontier))
        self.assertEqual(result.components, ())

    def test_isolated_unknown_wall_tip_remains_actionable(self):
        occupied = ((4, 2), (4, 3), (4, 4))
        free = tuple(
            (x, y)
            for y in range(9)
            for x in range(9)
            if (x, y) not in occupied and (x, y) != (4, 5)
        )

        result = FrontierExtractor(0.6).refresh(belief(
            shape=(9, 9),
            free=free,
            occupied=occupied,
        ))

        self.assertEqual(len(result.components), 1)
        self.assertEqual(result.components[0].wall_gain, 1)
        self.assertTrue(result.components[0].wall_cells)

    def test_incremental_wall_extension_matches_a_full_refresh(self):
        initial_occupied = ((4, 2), (4, 3), (4, 4))
        changed_occupied = (*initial_occupied, (4, 5))

        def wall_belief(occupied, unknown, version):
            free = tuple(
                (x, y)
                for y in range(9)
                for x in range(9)
                if (x, y) not in occupied and (x, y) != unknown
            )
            return belief(
                shape=(9, 9),
                free=free,
                occupied=occupied,
                version=version,
            )

        extractor = FrontierExtractor(0.6)
        extractor.refresh(wall_belief(initial_occupied, (4, 5), 1))
        incremental = extractor.refresh(
            wall_belief(changed_occupied, (4, 6), 2),
            dirty_regions=((3, 4, 6, 7),),
        )
        full = FrontierExtractor(0.6).refresh(
            wall_belief(changed_occupied, (4, 6), 2),
            force_full=True,
        )

        self.assertTrue(np.array_equal(
            incremental.masks.frontier,
            full.masks.frontier,
        ))
        self.assertEqual(incremental.components, full.components)


class FrontierClusterRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = FrontierClusterRegistry(
            match_distance=32.0,
            missing_refresh_limit=3,
        )
        self.extractor = FrontierExtractor(0.6)

    def components(self, free, version):
        return self.extractor.refresh(
            belief(free=tuple(free), version=version),
            force_full=True,
        ).components

    def test_id_survives_motion_and_three_missing_refreshes(self):
        first = self.registry.refresh(
            1, self.components(((4, 4), (4, 5)), 1), slam_version=1
        )
        cluster_id = first[0].id
        moved = self.registry.refresh(
            1, self.components(((5, 4), (5, 5)), 2), slam_version=2
        )
        self.assertEqual(moved[0].id, cluster_id)

        for version in (3, 4, 5):
            visible = self.registry.refresh(1, (), slam_version=version)
            self.assertEqual(visible[0].id, cluster_id)
        self.assertEqual(self.registry.refresh(1, (), slam_version=6), ())
        self.assertEqual(self.registry.refresh(
            1, self.components(((5, 4), (5, 5)), 7), slam_version=7
        ), ())

    def test_zero_gain_tombstone_blocks_unchanged_regeneration(self):
        components = self.components(((4, 4), (4, 5)), 1)
        cluster = self.registry.refresh(0, components, slam_version=1)[0]

        self.registry.retire(cluster.id, reason="zero_gain")
        refreshed = self.registry.refresh(0, components, slam_version=2)

        self.assertEqual(refreshed, ())
        self.assertEqual(self.registry.lifecycle_events()[-1].reason, "zero_gain")

    def test_recovery_penalties_are_visible_without_replacing_local_geometry(self):
        components = self.components(((4, 4), (4, 5)), 1)
        cluster = self.registry.refresh(0, components, slam_version=1)[0]

        self.assertTrue(self.registry.penalize(
            cluster.id,
            revisit=1.0,
            stall=2.0,
        ))
        visible = self.registry.visible_to(0)[0]

        self.assertEqual(visible.representative, cluster.representative)
        self.assertEqual(visible.revisit_penalty, 1.0)
        self.assertEqual(visible.stall_penalty, 2.0)

    def test_canonical_identity_does_not_leak_geometry(self):
        components = self.components(((4, 4), (4, 5)), 1)
        first = self.registry.refresh(0, components, slam_version=1)[0]
        second = self.registry.refresh(1, components, slam_version=1)[0]

        self.assertEqual(first.id, second.id)
        self.assertEqual(self.registry.visible_to(2), ())
        self.assertEqual(self.registry.get(first.id).known_by, frozenset({0, 1}))

    def test_same_identity_keeps_geometry_local_until_explicit_share(self):
        first_components = self.components(((4, 4), (4, 5)), 1)
        cluster_id = self.registry.refresh(
            0, first_components, slam_version=1
        )[0].id
        moved_components = self.components(((6, 4), (6, 5)), 2)
        self.registry.refresh(1, moved_components, slam_version=2)

        self.assertEqual(
            self.registry.visible_to(0)[0].representative,
            first_components[0].representative,
        )
        self.assertEqual(
            self.registry.visible_to(1)[0].representative,
            moved_components[0].representative,
        )
        self.assertEqual(self.registry.visible_to(0)[0].id, cluster_id)

    def test_knowledge_transfer_is_explicit_and_visibility_limited(self):
        components = self.components(((4, 4), (4, 5)), 1)
        cluster = self.registry.refresh(0, components, slam_version=1)[0]

        transferred = self.registry.share(0, 1)

        self.assertEqual(transferred, (cluster.id,))
        self.assertEqual(self.registry.visible_to(1)[0].id, cluster.id)
        self.assertEqual(self.registry.visible_to(2), ())

    def test_stable_cluster_selects_a_locally_known_alternate_waypoint(self):
        component = FrontierComponent(
            cells=frozenset({(8, 4), (9, 4), (10, 4)}),
            bounds=(8, 4, 11, 5),
            representative=(9, 4),
            expected_gain=4,
        )
        cluster = self.registry.refresh(0, (component,), slam_version=1)[0]
        known_free = np.zeros((12, 12), dtype=bool)
        known_free[4, 8] = True

        waypoint = select_accessible_frontier_waypoint(
            cluster, known_free, origin=(2, 4)
        )

        self.assertEqual(waypoint, (8, 4))
        self.assertEqual(self.registry.get(cluster.id).representative, (9, 4))
        self.assertEqual(self.registry.get(cluster.id).id, cluster.id)

    def test_accessible_waypoint_prefers_the_wall_continuation_cells(self):
        component = FrontierComponent(
            cells=frozenset({(3, 4), (8, 4), (9, 4)}),
            bounds=(3, 4, 10, 5),
            representative=(8, 4),
            expected_gain=8,
            wall_gain=2,
            wall_cells=frozenset({(8, 4), (9, 4)}),
        )
        cluster = self.registry.refresh(0, (component,), slam_version=1)[0]
        known_free = np.zeros((12, 12), dtype=bool)
        known_free[4, 3] = True
        known_free[4, 8] = True

        waypoint = select_accessible_frontier_waypoint(
            cluster, known_free, origin=(2, 4)
        )

        self.assertEqual(waypoint, (8, 4))

    def test_retained_wall_waypoint_requires_batched_displacement(self):
        component = FrontierComponent(
            cells=frozenset({(3, 4), (8, 4), (12, 4)}),
            bounds=(3, 4, 13, 5),
            representative=(8, 4),
            expected_gain=8,
            wall_gain=3,
            wall_cells=frozenset({(3, 4), (8, 4), (12, 4)}),
        )
        cluster = self.registry.refresh(0, (component,), slam_version=1)[0]
        known_free = np.zeros((16, 16), dtype=bool)
        known_free[4, 3] = True
        known_free[4, 8] = True
        known_free[4, 12] = True

        waypoint = select_accessible_frontier_waypoint(
            cluster,
            known_free,
            origin=(2, 4),
            minimum_distance=6.0,
        )

        self.assertEqual(waypoint, (8, 4))

    def test_wall_waypoint_prefers_local_unknown_support(self):
        component = FrontierComponent(
            cells=frozenset({(8, 4), (12, 4)}),
            bounds=(8, 4, 13, 5),
            representative=(8, 4),
            expected_gain=8,
            wall_gain=2,
            wall_cells=frozenset({(8, 4), (12, 4)}),
        )
        cluster = self.registry.refresh(0, (component,), slam_version=1)[0]
        known_free = np.zeros((16, 16), dtype=bool)
        known_free[4, 8] = True
        known_free[4, 12] = True
        unknown = np.zeros((16, 16), dtype=bool)
        unknown[3:6, 11:14] = True

        waypoint = select_accessible_frontier_waypoint(
            cluster,
            known_free,
            origin=(2, 4),
            unknown=unknown,
        )

        self.assertEqual(waypoint, (12, 4))


class AssignmentRegistryTests(unittest.TestCase):
    def test_race_has_exactly_one_winner_and_release_is_owner_scoped(self):
        assignments = AssignmentRegistry()
        barrier = threading.Barrier(8)

        def compete(drone_id):
            barrier.wait()
            return assignments.reserve(cluster_id=7, drone_id=drone_id)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = tuple(pool.map(compete, range(8)))

        winners = [result for result in results if result is not None]
        self.assertEqual(len(winners), 1)
        winner = winners[0]
        self.assertFalse(assignments.release(winner.token, drone_id=99))
        self.assertTrue(assignments.release(winner.token, drone_id=winner.drone_id))
        replacement = assignments.reserve(cluster_id=7, drone_id=99)
        self.assertIsNotNone(replacement)
        self.assertGreater(replacement.token, winner.token)

    def test_lazy_gateway_attachment_blocks_consolidated_goal(self):
        assignments = AssignmentRegistry()
        first = assignments.reserve(cluster_id=4, drone_id=0)
        attached = assignments.attach_gateway(
            first.token, drone_id=0, gateway_id=12
        )

        self.assertEqual(attached.gateway_id, 12)
        self.assertIsNone(assignments.reserve(
            cluster_id=5, drone_id=1, gateway_id=12
        ))


class FrontierGatewayManagerTests(unittest.TestCase):
    def setUp(self):
        self.registry = FrontierClusterRegistry(match_distance=32.0)
        self.graph = WaypointGraph(connector_distance=64.0)
        self.graph.add_waypoint((0, 2), source="home")
        self.graph.register_travelled_section(((0, 2), (10, 2)))
        self.known_free = np.ones((8, 100), dtype=bool)
        self.manager = FrontierGatewayManager(
            self.registry, self.graph, minimum_separation=64.0
        )

    @staticmethod
    def component(position):
        x, y = position
        return FrontierComponent(
            cells=frozenset({position}),
            bounds=(x, y, x + 1, y + 1),
            representative=position,
            expected_gain=3,
        )

    def test_gateway_manager_does_not_create_a_speculative_protected_node(self):
        cluster = self.registry.refresh(
            0, (self.component((20, 2)),), slam_version=1
        )[0]
        self.assertIsNone(cluster.gateway_id)

        gateway = self.manager.ensure_gateway(cluster.id, self.known_free)

        self.assertIsNone(gateway)
        self.assertEqual(self.graph.counts(), (2, 1))

    def test_gateway_manager_adopts_only_an_existing_required_corridor(self):
        cluster = self.registry.refresh(
            0, (self.component((70, 2)),), slam_version=1
        )[0]
        update = self.graph.connect_known_free_corridor(
            (70, 2), self.known_free, search_distance=80.0,
        )
        self.assertEqual(update.status, "ok")

        first = self.manager.ensure_gateway(cluster.id, self.known_free)
        second = self.manager.ensure_gateway(cluster.id, self.known_free)

        self.assertEqual(first, second)
        node = next(node for node in self.graph.snapshot().nodes if node.id == first)
        self.assertIn(WaypointRole.FRONTIER_GATEWAY, node.roles)


if __name__ == "__main__":
    unittest.main()
