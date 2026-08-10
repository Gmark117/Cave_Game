import math
import os
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from agents.drone import Drone
from agents.exploration_policy import (
    ExplorationDecision,
    ExplorationDecisionKind,
    FrontierExplorationPolicy,
)
from agents.mcts_exploration_policy import MctsExplorationPolicy
from config.simulation_config import (
    ExplorationConfig,
    MissionConfig,
    SimulationConfig,
    SlamConfig,
)
from mapping.slam_map import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    SlamProgressSnapshot,
    SlamSnapshot,
)
from agents.drone_movement import DroneMovementController
from mapping.terrain_knowledge import TerrainKnowledge
from mission.exploration_coordination import TeamExplorationCoordinator
from mission.pause_control import PauseCoordinator
from navigation.frontier_clusters import FrontierComponent
from navigation.waypoint_graph import (
    EDGE_KNOWN_FREE_CONNECTOR,
    GraphUpdate,
    ROUTE_NO_START_CONNECTOR,
    WaypointGraph,
    WaypointRole,
    bresenham_path,
)
from navigation.navigation_intent import (
    MovementMode,
    MovementOutcome,
    NavigationIntent,
    NavigationWatchdog,
    TransitionReason,
)


class ImmediateEvent:
    def wait(self, timeout: float) -> bool:
        return False


class MovementControl:
    delay = 1 / 15
    terrain_roughness = np.full((64, 64), 0.4, dtype=np.float32)

    def __init__(self) -> None:
        self.mission_event = ImmediateEvent()
        self.paths = {}
        self.path_requests = []
        self.terrain_knowledge = TerrainKnowledge(
            np.zeros((64, 64), dtype=np.uint8)
        )
        self.terrain_fusion = SimpleNamespace(record_scan=lambda samples: None)

    def compute_path(self, start, goal):
        self.path_requests.append((start, goal))
        return list(self.paths.get((start, goal), []))

    def simulation_time(self) -> float:
        return time.perf_counter()

    def pause_checkpoint(self) -> bool:
        return True

    def wait_simulation_delay(self, duration: float) -> bool:
        self.mission_event.wait(duration)
        return True


class RecordingTrace:
    def __init__(self) -> None:
        self.events = []

    def record(self, event, **fields) -> None:
        self.events.append((event, fields))


class FixedDecisionPolicy:
    def __init__(self, decision: ExplorationDecision) -> None:
        self.decision = decision
        self.contexts = []

    def decide(self, context):
        self.contexts.append(context)
        return self.decision


class ExhaustedPolicy:
    def __init__(self) -> None:
        self.contexts = []

    def decide(self, context):
        self.contexts.append(context)
        return ExplorationDecision(kind=ExplorationDecisionKind.EXHAUSTED)


class LocalMctsSpyPolicy:
    """Fail loudly if a valid stored route reaches either policy entry point."""

    def __init__(self) -> None:
        self.global_contexts = []
        self.local_contexts = []

    def decide(self, context):
        self.global_contexts.append(context)
        raise AssertionError("valid TRAVEL must not globally reselect")

    def decide_local(self, context):
        self.local_contexts.append(context)
        raise AssertionError("valid TRAVEL must not invoke local MCTS")


class FixedLocalPolicy:
    """Return one explicit deviation while rejecting global reselection."""

    def __init__(self, decision) -> None:
        self.decision = decision
        self.local_contexts = []

    def decide(self, context):
        raise AssertionError("local deviation must not globally reselect")

    def decide_local(self, context):
        self.local_contexts.append(context)
        return self.decision


class DroneMovementTests(unittest.TestCase):
    def setUp(self) -> None:
        settings = SimulationConfig(
            mission_config=MissionConfig(map_dim="LARGE"),
            slam=SlamConfig(
                scan_interval=0.0,
                scan_rays=5,
                point_cloud_max_points=50,
            ),
        )
        self.window = pygame.Surface((64, 64), pygame.SRCALPHA)
        game = SimpleNamespace(
            sim_settings=settings,
            window=self.window,
            width=64,
            height=64,
        )
        self.control = MovementControl()
        cave = np.zeros((64, 64), dtype=np.uint8)
        icon = pygame.Surface((4, 4), pygame.SRCALPHA)
        self.drone = Drone(
            game,
            self.control,
            0,
            (16, 16),
            (255, 0, 0),
            icon,
            cave,
        )

    def test_drone_uses_configured_exploration_policy(self) -> None:
        self.assertIsInstance(self.drone.exploration_policy, MctsExplorationPolicy)

        settings = SimulationConfig(
            mission_config=MissionConfig(map_dim="LARGE"),
            exploration=ExplorationConfig(policy="frontier"),
        )
        game = SimpleNamespace(
            sim_settings=settings,
            window=self.window,
            width=64,
            height=64,
        )
        icon = pygame.Surface((4, 4), pygame.SRCALPHA)
        drone = Drone(
            game,
            self.control,
            1,
            (16, 16),
            (0, 255, 0),
            icon,
            np.zeros((64, 64), dtype=np.uint8),
        )

        self.assertIsInstance(
            drone.exploration_policy,
            FrontierExplorationPolicy,
        )

    def test_homing_follows_path_and_updates_heading(self) -> None:
        self.drone.runtime_state.move_to((20, 20))
        self._make_slam_known_free()

        reached_home = self.drone.movement_controller.reach_start_point()
        snapshot = self.drone.snapshot()

        self.assertTrue(reached_home)
        self.assertEqual(snapshot.position, (16, 16))
        self.assertEqual(snapshot.path_history[-1], (16, 16))
        self.assertNotEqual(snapshot.heading_deg, 0.0)
        self.assertEqual(self.control.path_requests, [])

    def test_path_execution_traces_actual_travelled_distance(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )

        followed = controller._follow_path(
            [(16, 16), (19, 20)]
        )

        self.assertTrue(followed)
        motion = next(
            fields
            for event, fields in trace.events
            if event == "drone_motion"
        )
        self.assertEqual(motion["source"], "path")
        self.assertTrue(motion["completed"])
        self.assertEqual(motion["start"], (16, 16))
        self.assertEqual(motion["end"], (19, 20))
        self.assertEqual(motion["point_count"], 2)
        self.assertAlmostEqual(motion["travelled_distance"], 5.0)
        self.assertLessEqual(
            motion["started_sim_time"],
            motion["ended_sim_time"],
        )

    def test_interrupted_policy_path_traces_partial_travelled_distance(
        self,
    ) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
            wait_simulation_delay=lambda _duration: False,
        )

        followed = controller._follow_policy_path(
            [(19, 20), (22, 24)]
        )

        self.assertFalse(followed)
        motion = next(
            fields
            for event, fields in trace.events
            if event == "drone_motion"
        )
        self.assertEqual(motion["source"], "policy_path")
        self.assertFalse(motion["completed"])
        self.assertEqual(motion["start"], (16, 16))
        self.assertEqual(motion["end"], (19, 20))
        self.assertAlmostEqual(motion["travelled_distance"], 5.0)

    def test_mcts_trace_includes_root_coverage_and_overrun_stage(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        self.drone.exploration_policy = SimpleNamespace(
            config=SimpleNamespace(iterations=8),
            last_search_diagnostics=SimpleNamespace(
                iterations=0,
                generated_nodes=1,
                selected_kind="follow_edge",
                selected_direction=None,
                selected_target=(16, 16),
                selected_reward=0.0,
                slam_version=3,
                elapsed_ms=36.0,
                root_visits=(),
                root_coverage_complete=False,
                overrun_stage="root_coverage",
            ),
        )

        controller._trace_decision(
            "drone_decision",
            ExplorationDecision(ExplorationDecisionKind.EXHAUSTED),
        )

        fields = trace.events[-1][1]
        self.assertFalse(fields["mcts"]["root_coverage_complete"])
        self.assertEqual(fields["mcts"]["overrun_stage"], "root_coverage")

    def test_trace_dual_writes_canonical_intent_and_watchdog_payloads(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        latched = self.drone.runtime_state.set_navigation_intent(
            NavigationIntent(
                mode=MovementMode.TRAVEL,
                route_id=51,
                cluster_id=17,
                gateway_id=9,
                assignment_token=31,
                target=(30, 16),
                topology_revision=7,
                requester_knowledge_revision=11,
                route_node_ids=(1, 2),
                route_edge_ids=(5,),
                route_paths=(((16, 16), (30, 16)),),
                route_sources=("travelled",),
                route_segment_edge_ids=(5,),
                edge_cursor=0,
                polyline_cursor=2,
                remaining_route_cost=12.0,
                selection_slam_version=11,
            ),
            reason=TransitionReason.SELECTED,
        )
        self.drone.runtime_state.update_navigation_watchdog(
            NavigationWatchdog(
                last_progress_time=2.0,
                distance_without_progress=8.0,
                recent_visits=(5, 6, 5),
                reversal_count=1,
            )
        )

        controller._trace(
            "drone_intent_result",
            replan_reason="selected",
            route_replanned=True,
            mode_transition={
                "from_mode": None,
                "to_mode": "travel",
                "reason": "selected",
            },
            movement_outcome={
                "actual_information_gain": 3.0,
                "route_progress_delta": 2.0,
            },
        )

        fields = trace.events[-1][1]
        expected = {
            "intent_id": latched.intent_id,
            "mode": "travel",
            "goal_cluster_id": 17,
            "gateway_id": 9,
            "assignment_id": 31,
            "route_id": 51,
            "topology_revision": 7,
            "requester_knowledge_revision": 11,
            "selection_slam_version": 11,
            "route_node_ids": (1, 2),
            "route_edge_ids": (5,),
            "edge_cursor": 0,
            "polyline_cursor": 2,
            "remaining_route_cost": 12.0,
        }
        self.assertEqual(fields["intent"], expected)
        self.assertEqual(fields["navigation_intent"], expected)
        self.assertEqual(fields["watchdog"]["reversal_count"], 1)
        self.assertAlmostEqual(fields["watchdog"]["revisit_ratio"], 1 / 3)
        self.assertEqual(fields["replan_reason"], "selected")
        self.assertTrue(fields["route_replanned"])

    def test_graph_mutation_trace_keeps_legacy_aliases_with_canonical_ids(
        self,
    ) -> None:
        trace = RecordingTrace()
        graph = WaypointGraph(merge_radius=0)
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
            waypoint_graph=graph,
        )
        update = graph.register_travelled_section(
            ((1, 1), (8, 1)),
            start_role=WaypointRole.HOME,
            end_role=WaypointRole.TURN,
        )

        controller._trace_waypoint_update(update)

        delta = next(
            fields for event, fields in trace.events
            if event == "waypoint_graph_delta"
        )
        self.assertEqual(delta["topology_revision"], update.delta.revision)
        self.assertEqual(
            tuple(node["node_id"] for node in delta["added_nodes"]),
            update.delta.added_node_ids,
        )
        self.assertEqual(
            tuple(edge["edge_id"] for edge in delta["added_edges"]),
            update.delta.added_edge_ids,
        )
        legacy_nodes = [
            fields for event, fields in trace.events if event == "waypoint_added"
        ]
        legacy_edges = [
            fields for event, fields in trace.events
            if event == "waypoint_edge_added"
        ]
        self.assertTrue(all("node_id" in fields for fields in legacy_nodes))
        self.assertTrue(all("edge_id" in fields for fields in legacy_edges))

    def test_frontier_rebuild_uses_local_slam_state(self) -> None:
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[20, 20] = FREE
        confidence[20, 20] = 1.0
        self.drone.slam_map.merge_from(
            SlamSnapshot(occupancy, confidence)
        )

        self.drone.movement_controller.rebuild_frontiers()

        self.assertEqual(self.drone.snapshot().frontiers, ((20, 20),))

    def test_frontier_rebuild_ignores_mission_terrain_telemetry(self) -> None:
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[20, 20] = FREE
        confidence[20, 20] = 0.1
        self.drone.slam_map.merge_from(
            SlamSnapshot(occupancy, confidence)
        )
        self.control.terrain_knowledge.confidence[20, 20] = 1.0

        self.drone.movement_controller.rebuild_frontiers()

        self.assertEqual(self.drone.snapshot().frontiers, ())

        self.drone.terrain_knowledge.confidence[20, 20] = 1.0
        self.drone.movement_controller.rebuild_frontiers()

        # Terrain confidence may value a SLAM frontier, but cannot turn a
        # low-confidence occupancy cell into confidently traversable floor.
        self.assertEqual(self.drone.snapshot().frontiers, ())

    def test_movement_configuration_is_controller_owned(self) -> None:
        self.assertIsInstance(
            self.drone.movement_controller,
            DroneMovementController,
        )
        controller = self.drone.movement_controller
        controller.border_retry_cooldown = 3.0

        self.assertEqual(controller.border_retry_cooldown, 3.0)

    def test_short_straight_policy_moves_keep_live_trail_tail_ephemeral(self) -> None:
        graph = WaypointGraph(
            merge_radius=4.0,
            connector_distance=64.0,
            connector_limit=8,
        )
        graph.add_waypoint((16, 16), source="home")
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        controller._trail_accumulator.reset((16, 16))

        for target in ((26, 16), (36, 16), (46, 16), (56, 16)):
            self.assertTrue(controller._follow_policy_path([target]))

        self.assertEqual(graph.counts(), (1, 0))
        self.assertEqual(
            (
                controller._trail_accumulator.tail[0],
                controller._trail_accumulator.tail[-1],
            ),
            ((16, 16), (56, 16)),
        )

    def test_long_straight_trail_creates_only_coarse_recovery_anchors(self) -> None:
        graph = WaypointGraph(
            merge_radius=4.0,
            connector_distance=64.0,
            connector_limit=8,
        )
        graph.add_waypoint((0, 0), source="home")
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        controller._trail_accumulator.reset((0, 0))

        for index in range(100):
            controller._ingest_travelled_motion(
                [(index * 10, 0), ((index + 1) * 10, 0)]
            )

        self.assertEqual(graph.counts(), (8, 7))
        nodes = graph.snapshot().nodes
        self.assertEqual(
            tuple(node.position for node in nodes),
            tuple((position, 0) for position in (0, 128, 256, 384, 512, 640, 768, 896)),
        )
        self.assertTrue(all(
            node.roles == {WaypointRole.RECOVERY_ANCHOR}
            for node in nodes[1:]
        ))
        self.assertEqual(
            (
                controller._trail_accumulator.tail[0],
                controller._trail_accumulator.tail[-1],
            ),
            ((896, 0), (1000, 0)),
        )

    def test_confirmed_turn_is_promoted_after_both_legs(self) -> None:
        graph = WaypointGraph(merge_radius=0)
        graph.add_waypoint((0, 0), source="home")
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        controller._trail_accumulator.reset((0, 0))

        controller._ingest_travelled_motion([(0, 0), (30, 0)])
        controller._ingest_travelled_motion([(30, 0), (30, 20)])
        self.assertEqual(graph.counts(), (1, 0))

        controller._ingest_travelled_motion([(30, 20), (30, 30)])

        turn = graph.snapshot().nodes[-1]
        self.assertEqual(turn.position, (30, 0))
        self.assertEqual(turn.roles, {WaypointRole.TURN})
        self.assertEqual(graph.snapshot().edges[0].path[0], (0, 0))

    def test_far_route_lookup_does_not_flush_current_pose(self) -> None:
        graph = WaypointGraph(connector_distance=8, merge_radius=0)
        graph.add_waypoint((16, 16), source="home")
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        controller._trail_accumulator.reset((16, 16))
        controller._ingest_travelled_motion([(16, 16), (36, 16)])
        self.drone.runtime_state.move_to((36, 16))
        known_free = np.ones((64, 64), dtype=bool)

        cluster_id = self._install_frontier_clusters((60, 16))[0]
        controller._advance_waypoint_segment(
            (36, 16), (60, 16), known_free, cluster_id=cluster_id,
        )

        self.assertNotIn(
            (36, 16),
            tuple(node.position for node in graph.snapshot().nodes),
        )
        self.assertEqual(controller._trail_accumulator.tail[-1], (36, 16))

    def test_executed_ephemeral_connector_extends_the_live_trail(self) -> None:
        self._make_slam_known_free()
        graph = WaypointGraph(connector_distance=15, merge_radius=0)
        graph.add_waypoint((16, 16), source="home")
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        controller._trail_accumulator.reset((16, 16))
        intent = self.drone.runtime_state.set_navigation_intent(
            NavigationIntent(
                mode=MovementMode.TRAVEL,
                target=(36, 16),
                topology_revision=graph.topology_revision,
                requester_knowledge_revision=self.drone.slam_map.version,
                route_paths=(((16, 16), (36, 16)),),
                route_sources=(EDGE_KNOWN_FREE_CONNECTOR,),
                route_segment_edge_ids=(None,),
                remaining_route_cost=20.0,
            )
        )

        outcome = controller._execute_navigation_intent(intent)

        self.assertEqual(outcome.travelled_distance, 10.0)
        self.assertEqual(self.drone.snapshot().position, (26, 16))
        self.assertEqual(
            (controller._trail_accumulator.tail[0],
             controller._trail_accumulator.tail[-1]),
            ((16, 16), (26, 16)),
        )
        self.assertEqual(graph.counts(), (1, 0))

    def test_uncommitted_travelled_tail_restores_start_connectivity(self) -> None:
        self._make_slam_known_free()
        graph = WaypointGraph(connector_distance=8, merge_radius=0)
        graph.add_waypoint((16, 16), source="home")
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        controller._trail_accumulator.reset((16, 16))
        controller._ingest_travelled_motion([(16, 16), (36, 16)])
        self.drone.runtime_state.move_to((36, 16))
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        cluster_id = self._install_frontier_clusters((16, 16))[0]

        direct = graph.find_route(
            (36, 16),
            (16, 16),
            known_free,
            requester_id=self.drone.id,
            requester_knowledge_revision=self.drone.slam_map.version,
        )
        self.assertEqual(direct.status, ROUTE_NO_START_CONNECTOR)

        moved = controller._advance_waypoint_segment(
            (36, 16),
            (16, 16),
            known_free,
            cluster_id=cluster_id,
        )

        self.assertTrue(moved)
        self.assertEqual(self.drone.snapshot().position, (26, 16))
        active = self.drone.snapshot().navigation_intent
        self.assertIsNotNone(active)
        self.assertEqual(active.route_sources[0], "travelled")
        self.assertEqual(graph.counts(), (1, 0))

    def test_confirmed_exhaustion_starts_homing_after_frontier_rebuild(self) -> None:
        self.drone.runtime_state.move_to((20, 16))
        self._make_slam_known_free()
        self.drone.exploration_policy = ExhaustedPolicy()

        self.drone.movement_controller.move()
        snapshot = self.drone.snapshot()

        self.assertEqual(snapshot.position, (16, 16))
        self.assertTrue(snapshot.returning_home)
        self.assertTrue(snapshot.done)
        self.assertEqual(len(self.drone.exploration_policy.contexts), 2)

    def test_stored_close_target_route_execution_never_invokes_astar(self) -> None:
        self._make_slam_known_free()
        controller = self.drone.movement_controller
        graph = controller.waypoint_graph
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        route = graph.find_route((16, 16), (24, 20), known_free)
        intent = self.drone.runtime_state.set_navigation_intent(
            NavigationIntent(
                mode=MovementMode.TRAVEL,
                target=(24, 20),
                topology_revision=route.topology_revision,
                requester_knowledge_revision=self.drone.slam_map.version,
                route_paths=route.segment_paths,
                route_sources=route.segment_sources,
                route_segment_edge_ids=route.segment_edge_ids,
                remaining_route_cost=route.cost,
            )
        )

        with patch(
            "navigation.waypoint_graph._bounded_known_free_path_to_any",
            side_effect=AssertionError("stored-route execution called A*"),
        ):
            outcome = controller._execute_navigation_intent(intent)

        self.assertTrue(outcome.arrived)
        self.assertEqual(self.drone.snapshot().position, (24, 20))

    def test_move_executes_policy_frontier_decision(self) -> None:
        target = (24, 16)
        self._make_slam_known_free()
        cluster_id = self._install_frontier_clusters(target)[0]
        self.drone.exploration_policy = FixedDecisionPolicy(
            ExplorationDecision(
                kind=ExplorationDecisionKind.FRONTIER,
                target=target,
                cluster_id=cluster_id,
                frontier_cluster_ids=(cluster_id,),
            )
        )

        self.drone.movement_controller.move()

        self.assertEqual(self.drone.snapshot().position, target)
        self.assertEqual(
            self.drone.snapshot().navigation_intent.cluster_id,
            cluster_id,
        )

    def test_waypoint_frontier_replans_and_consumes_only_at_target(self) -> None:
        target = (56, 16)
        graph = self._configured_highway_graph()
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
            runtime_trace=trace,
        )
        cluster_id = self._install_frontier_clusters(target)[0]

        for expected in ((26, 16), (36, 16), (46, 16)):
            self.assertTrue(controller._reach_frontier_clusters([cluster_id]))
            self.assertEqual(self.drone.snapshot().position, expected)
            self.assertIn(target, self.drone.snapshot().frontiers)

        self.assertTrue(controller._reach_frontier_clusters([cluster_id]))
        self.assertEqual(self.drone.snapshot().position, target)
        self.assertIn(target, self.drone.snapshot().frontiers)
        segment_events = [
            fields
            for event, fields in trace.events
            if event == "drone_waypoint_segment_path"
        ]
        reached_events = [
            fields
            for event, fields in trace.events
            if event == "drone_frontier_reached"
        ]
        self.assertEqual(len(segment_events), 1)
        self.assertEqual(len(reached_events), 1)

    def test_zero_distance_frontier_arrival_starts_scan_without_failure(
        self,
    ) -> None:
        """A retained cluster at the live pose is progress, not unreachable."""
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
        )
        target = self.drone.snapshot().position
        cluster_id = self._install_frontier_clusters(target)[0]
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )

        advanced = controller._advance_waypoint_segment(
            target,
            target,
            known_free,
            cluster_id=cluster_id,
        )

        intent = self.drone.snapshot().navigation_intent
        self.assertTrue(advanced)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.mode, MovementMode.SCAN)
        self.assertEqual(intent.cluster_id, cluster_id)
        self.assertNotIn(cluster_id, controller._unreachable_blacklist)

    def test_retained_fallback_cannot_bypass_accessible_wall_tier(self) -> None:
        self._make_slam_known_free()
        controller = self.drone.movement_controller
        fallback = FrontierComponent(
            cells=frozenset({(20, 16)}),
            bounds=(20, 16, 21, 17),
            representative=(20, 16),
            expected_gain=20,
        )
        wall = FrontierComponent(
            cells=frozenset({(56, 16)}),
            bounds=(56, 16, 57, 17),
            representative=(56, 16),
            expected_gain=1,
            wall_gain=1,
            wall_cells=frozenset({(56, 16)}),
        )
        clusters = self.drone.frontier_registry.refresh(
            self.drone.id,
            (fallback, wall),
            slam_version=self.drone.slam_map.version,
        )
        self.drone.runtime_state.replace_frontier_clusters(clusters)
        fallback_id = next(
            cluster.id for cluster in clusters
            if cluster.representative == (20, 16)
        )
        wall_id = next(
            cluster.id for cluster in clusters
            if cluster.representative == (56, 16)
        )
        controller._retained_cluster_id = fallback_id
        self.drone.exploration_policy = FrontierExplorationPolicy()

        self.assertEqual(
            controller._score_frontier_clusters(
                tuple(cluster.id for cluster in clusters)
            ),
            (wall_id,),
        )
        with patch.object(
            controller,
            "_reach_frontier_clusters",
            return_value=True,
        ) as reach:
            controller.move()

        self.assertIsNone(controller._retained_cluster_id)
        reach.assert_called_once_with(tuple(sorted(
            cluster.id for cluster in clusters
        )))

    def test_retained_wall_intent_batches_distance_and_shortens_scan(self) -> None:
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
        )
        component = FrontierComponent(
            cells=frozenset({(16, 16), (32, 16)}),
            bounds=(16, 16, 33, 17),
            representative=(24, 16),
            expected_gain=12,
            wall_gain=2,
            wall_cells=frozenset({(16, 16), (32, 16)}),
        )
        cluster = self.drone.frontier_registry.refresh(
            self.drone.id,
            (component,),
            slam_version=self.drone.slam_map.version,
        )[0]
        self.drone.runtime_state.replace_frontier_clusters((cluster,))
        controller._retained_cluster_id = cluster.id

        moved = controller._reach_frontier_clusters((cluster.id,))

        intent = self.drone.snapshot().navigation_intent
        self.assertTrue(moved)
        self.assertEqual(intent.target, (32, 16))
        self.assertEqual(intent.scan_heading_count, 3)

    def test_tiny_wall_continuation_is_suppressed_until_geometry_changes(
        self,
    ) -> None:
        self._make_slam_known_free()
        controller = self.drone.movement_controller
        component = FrontierComponent(
            cells=frozenset({(16, 16), (18, 16)}),
            bounds=(16, 16, 19, 17),
            representative=(17, 16),
            expected_gain=2,
            wall_gain=2,
            wall_cells=frozenset({(16, 16), (18, 16)}),
        )
        cluster = self.drone.frontier_registry.refresh(
            self.drone.id,
            (component,),
            slam_version=self.drone.slam_map.version,
        )[0]
        self.drone.runtime_state.replace_frontier_clusters((cluster,))
        intent = self.drone.runtime_state.set_navigation_intent(
            NavigationIntent(
                mode=MovementMode.SCAN,
                cluster_id=cluster.id,
                target=(16, 16),
                scan_heading_cursor=5,
                scan_sequence=5,
            )
        )
        progress = SlamProgressSnapshot(
            completed_scan_sequence=6,
            sensor_newly_known_cells=1,
            sensor_confidence_gain=1.0,
        )

        with patch.object(
            self.drone.slam_map,
            "progress_snapshot",
            return_value=progress,
        ), patch.object(controller, "rebuild_frontiers"):
            outcome = controller._execute_scan_intent(intent)

        self.assertTrue(outcome.scan_complete)
        self.assertIsNone(controller._retained_cluster_id)
        self.assertIn(cluster.id, controller._scan_suppressed_clusters)
        self.assertEqual(controller._score_frontier_clusters((cluster.id,)), ())

        expanded = FrontierComponent(
            cells=frozenset({(16, 16), (18, 16), (32, 16)}),
            bounds=(16, 16, 33, 17),
            representative=(18, 16),
            expected_gain=3,
            wall_gain=3,
            wall_cells=frozenset({(16, 16), (18, 16), (32, 16)}),
        )
        refreshed = self.drone.frontier_registry.refresh(
            self.drone.id,
            (expanded,),
            slam_version=self.drone.slam_map.version + 1,
        )[0]
        self.drone.runtime_state.replace_frontier_clusters((refreshed,))

        self.assertEqual(
            controller._score_frontier_clusters((cluster.id,)),
            (cluster.id,),
        )
        self.assertNotIn(cluster.id, controller._scan_suppressed_clusters)

    def test_persistent_route_executes_exact_prefixes_without_astar(self) -> None:
        target = (56, 16)
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
        )
        cluster_id = self._install_frontier_clusters(target)[0]
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )

        self.assertTrue(controller._advance_waypoint_segment(
            (16, 16), target, known_free, cluster_id=cluster_id,
        ))
        first = self.drone.snapshot()
        intent = first.navigation_intent

        self.assertEqual(first.position, (26, 16))
        self.assertIsNotNone(intent)
        self.assertEqual(self.control.path_requests, [])

        self.assertTrue(controller._advance_waypoint_segment(
            first.position, target, known_free, cluster_id=cluster_id,
        ))
        second = self.drone.snapshot()

        self.assertEqual(second.position, (36, 16))
        self.assertEqual(second.navigation_intent.cluster_id, intent.cluster_id)
        self.assertEqual(
            second.navigation_intent.assignment_token,
            intent.assignment_token,
        )
        self.assertEqual(second.navigation_intent.route_edge_ids, intent.route_edge_ids)
        self.assertEqual(self.control.path_requests, [])

    def test_paused_route_prefix_advances_only_the_physically_executed_cursor(
        self,
    ) -> None:
        from navigation.navigation_intent import MovementMode, NavigationIntent

        controller = self.drone.movement_controller
        intent = NavigationIntent(
            mode=MovementMode.TRAVEL,
            target=(26, 16),
            route_paths=(((16, 16), (26, 16)),),
            route_sources=("travelled",),
            route_segment_edge_ids=(None,),
            remaining_route_cost=10.0,
        )
        self.drone.runtime_state.set_navigation_intent(intent)
        waits = iter((True, False))
        controller.dependencies = replace(
            controller.dependencies,
            wait_simulation_delay=lambda _duration: next(waits),
        )

        outcome = controller._execute_navigation_intent(intent)
        paused = self.drone.snapshot().navigation_intent

        self.assertEqual(outcome.transition_reason, TransitionReason.PAUSED)
        self.assertEqual(outcome.travelled_distance, 1.0)
        self.assertEqual(outcome.route_progress_delta, 1.0)
        self.assertEqual(self.drone.snapshot().position, (17, 16))
        self.assertEqual(paused.edge_cursor, 0)
        self.assertEqual(paused.polyline_cursor, 1)
        self.assertEqual(paused.remaining_route_cost, 9.0)

    def test_valid_travel_fast_path_never_invokes_local_mcts_or_reselects(
        self,
    ) -> None:
        target = (56, 16)
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
        )
        cluster_id = self._install_frontier_clusters(target)[0]
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        self.assertTrue(controller._advance_waypoint_segment(
            (16, 16), target, known_free, cluster_id=cluster_id,
        ))
        before = self.drone.snapshot().navigation_intent
        policy = LocalMctsSpyPolicy()
        self.drone.exploration_policy = policy

        with patch.object(
            controller,
            "_score_frontier_clusters",
            side_effect=AssertionError("valid intent must not reselect a goal"),
        ):
            controller.move()

        after = self.drone.snapshot().navigation_intent
        self.assertEqual(self.drone.snapshot().position, (36, 16))
        self.assertEqual(policy.local_contexts, [])
        self.assertEqual(policy.global_contexts, [])
        self.assertEqual(after.cluster_id, before.cluster_id)
        self.assertEqual(after.gateway_id, before.gateway_id)
        self.assertEqual(after.assignment_token, before.assignment_token)
        self.assertEqual(after.route_edge_ids, before.route_edge_ids)
        self.assertGreater(
            (after.edge_cursor, after.polyline_cursor),
            (before.edge_cursor, before.polyline_cursor),
        )
        self.assertEqual(self.control.path_requests, [])

    def test_off_route_pose_uses_one_local_deviation_without_losing_goal(self) -> None:
        target = (56, 16)
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
        )
        cluster_id = self._install_frontier_clusters(target)[0]
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        self.assertTrue(controller._advance_waypoint_segment(
            (16, 16), target, known_free, cluster_id=cluster_id,
        ))
        latched = self.drone.snapshot().navigation_intent
        self.drone.runtime_state.move_to((26, 18))
        policy = FixedLocalPolicy(ExplorationDecision(
            kind=ExplorationDecisionKind.STEP,
            target=(30, 18),
            direction=90,
            planned_path=((26, 18), (30, 18)),
            local_primitive="deviate_right",
        ))
        self.drone.exploration_policy = policy

        with patch.object(
            controller,
            "_score_frontier_clusters",
            side_effect=AssertionError("local deviation must retain its goal"),
        ):
            controller.move()

        current = self.drone.snapshot()
        self.assertEqual(current.position, (30, 18))
        self.assertEqual(len(policy.local_contexts), 1)
        self.assertEqual(current.navigation_intent.cluster_id, latched.cluster_id)
        self.assertEqual(
            current.navigation_intent.assignment_token,
            latched.assignment_token,
        )
        self.assertEqual(
            current.navigation_intent.route_edge_ids,
            latched.route_edge_ids,
        )
        self.assertEqual(
            current.navigation_intent.previous_primitive,
            "deviate_right",
        )
        self.assertEqual(self.control.path_requests, [])

    def test_active_local_mcts_copies_only_a_bounded_slam_window(self) -> None:
        target = (56, 16)
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
        )
        cluster_id = self._install_frontier_clusters(target)[0]
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        self.assertTrue(controller._advance_waypoint_segment(
            (16, 16), target, known_free, cluster_id=cluster_id,
        ))
        intent = self.drone.snapshot().navigation_intent
        self.drone.runtime_state.move_to((26, 18))
        original_window = self.drone.slam_map.try_snapshot_window

        with (
            patch.object(
                self.drone.slam_map,
                "snapshot",
                side_effect=AssertionError("local MCTS must not copy the full map"),
            ),
            patch.object(
                self.drone.slam_map,
                "try_snapshot_window",
                wraps=original_window,
            ) as snapshot_window,
        ):
            controller._execute_active_navigation_intent(intent)

        snapshot_window.assert_called_once()
        left, top, right, bottom = snapshot_window.call_args.args[0]
        self.assertLessEqual(right - left, 193)
        self.assertLessEqual(bottom - top, 193)

    def test_frontier_mode_follows_stable_cluster_deterministically_without_mcts(
        self,
    ) -> None:
        target = (56, 16)
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
        )
        cluster_id = self._install_frontier_clusters(target)[0]
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        self.assertTrue(controller._advance_waypoint_segment(
            (16, 16), target, known_free, cluster_id=cluster_id,
        ))
        latched = self.drone.snapshot().navigation_intent
        self.drone.exploration_policy = FrontierExplorationPolicy()

        with (
            patch.object(
                controller,
                "choose_exploration_action",
                side_effect=AssertionError("stable cluster must not reselect"),
            ),
            patch.object(
                controller,
                "_score_frontier_clusters",
                side_effect=AssertionError("stable cluster must not be rescored"),
            ),
            patch.object(
                MctsExplorationPolicy,
                "decide_local",
                create=True,
                side_effect=AssertionError("frontier mode must not run MCTS"),
            ),
        ):
            controller.move()
            self.assertEqual(self.drone.snapshot().position, (36, 16))
            controller.move()

        current = self.drone.snapshot()
        self.assertEqual(current.position, (46, 16))
        self.assertEqual(current.navigation_intent.cluster_id, latched.cluster_id)
        self.assertEqual(
            current.navigation_intent.assignment_token,
            latched.assignment_token,
        )
        self.assertEqual(
            current.navigation_intent.route_edge_ids,
            latched.route_edge_ids,
        )
        self.assertEqual(self.control.path_requests, [])

    def test_unrelated_topology_revision_keeps_latched_route(self) -> None:
        target = (56, 16)
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        cluster_id = self._install_frontier_clusters(target)[0]
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        self.assertTrue(controller._advance_waypoint_segment(
            (16, 16), target, known_free, cluster_id=cluster_id,
        ))
        intent = self.drone.snapshot().navigation_intent

        graph.add_waypoint((60, 60), source="recovery_anchor")
        outcome = controller._execute_navigation_intent(intent)

        self.assertFalse(outcome.invalidated)
        self.assertEqual(self.drone.snapshot().navigation_intent.cluster_id, intent.cluster_id)

    def test_retired_remaining_edge_explicitly_invalidates_once(self) -> None:
        target = (56, 16)
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        cluster_id = self._install_frontier_clusters(target)[0]
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        self.assertTrue(controller._advance_waypoint_segment(
            (16, 16), target, known_free, cluster_id=cluster_id,
        ))
        intent = self.drone.snapshot().navigation_intent
        remaining_edge = next(edge_id for edge_id in intent.route_edge_ids)

        graph.split_edge(remaining_edge, (30, 16))
        outcome = controller._execute_navigation_intent(intent)

        self.assertTrue(outcome.invalidated)
        self.assertEqual(
            outcome.transition_reason,
            TransitionReason.ROUTE_EDGE_RETIRED,
        )
        self.assertIsNone(self.drone.snapshot().navigation_intent)

    def test_scan_waits_for_each_sensor_sequence_and_finishes_after_six(self) -> None:
        from navigation.navigation_intent import MovementMode, NavigationIntent

        controller = self.drone.movement_controller
        intent = NavigationIntent(
            mode=MovementMode.TRAVEL,
            target=self.drone.snapshot().position,
        )
        controller._begin_scan(intent)

        waiting = controller._execute_navigation_intent(
            self.drone.snapshot().navigation_intent
        )
        self.assertFalse(waiting.scan_complete)
        self.assertEqual(self.drone.snapshot().navigation_intent.scan_heading_cursor, 0)

        for expected_heading in range(1, 6):
            self.drone.slam_map.update_from_rays((16, 16), ())
            outcome = controller._execute_navigation_intent(
                self.drone.snapshot().navigation_intent
            )
            self.assertFalse(outcome.scan_complete)
            self.assertEqual(
                self.drone.snapshot().navigation_intent.scan_heading_cursor,
                expected_heading,
            )

        self.drone.slam_map.update_from_rays((16, 16), ())
        outcome = controller._execute_navigation_intent(
            self.drone.snapshot().navigation_intent
        )
        self.assertTrue(outcome.scan_complete)
        self.assertEqual(outcome.transition_reason, TransitionReason.ZERO_GAIN)
        self.assertIsNone(self.drone.snapshot().navigation_intent)

    def test_gateway_scan_starts_from_arrival_heading(self) -> None:
        controller = self.drone.movement_controller
        self.drone.runtime_state.begin_exploration(135)
        intent = NavigationIntent(
            mode=MovementMode.TRAVEL,
            target=self.drone.snapshot().position,
        )

        controller._begin_scan(intent)
        scanning = self.drone.snapshot().navigation_intent

        self.assertEqual(scanning.scan_base_heading, 135.0)
        self.assertEqual(self.drone.snapshot().heading_deg, 135.0)
        self.drone.slam_map.update_from_rays((16, 16), ())
        controller._execute_scan_intent(scanning)
        self.assertEqual(self.drone.snapshot().heading_deg, 195.0)

    def test_retained_wall_scan_uses_configured_directed_heading_count(self) -> None:
        controller = self.drone.movement_controller
        intent = NavigationIntent(
            mode=MovementMode.TRAVEL,
            target=self.drone.snapshot().position,
            scan_heading_count=3,
        )

        with patch.object(
            controller,
            "_wall_continuation_heading",
            return_value=90.0,
        ):
            controller._begin_scan(intent)

        scanning = self.drone.snapshot().navigation_intent
        self.assertEqual(scanning.scan_base_heading, 30.0)
        self.assertEqual(self.drone.snapshot().heading_deg, 30.0)
        for expected_cursor, expected_heading in ((1, 90.0), (2, 150.0)):
            self.drone.slam_map.update_from_rays((16, 16), ())
            outcome = controller._execute_scan_intent(
                self.drone.snapshot().navigation_intent
            )
            self.assertFalse(outcome.scan_complete)
            self.assertEqual(
                self.drone.snapshot().navigation_intent.scan_heading_cursor,
                expected_cursor,
            )
            self.assertEqual(self.drone.snapshot().heading_deg, expected_heading)
        self.drone.slam_map.update_from_rays((16, 16), ())
        outcome = controller._execute_scan_intent(
            self.drone.snapshot().navigation_intent
        )
        self.assertTrue(outcome.scan_complete)

    def test_retained_wall_scan_faces_unknown_surface_cells(self) -> None:
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[16, 16] = FREE
        confidence[16, 16] = 1.0
        occupancy[16, 18] = OCCUPIED
        confidence[16, 18] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        component = FrontierComponent(
            cells=frozenset({(16, 16)}),
            bounds=(16, 16, 17, 17),
            representative=(16, 16),
            expected_gain=1,
            wall_gain=1,
            wall_cells=frozenset({(16, 16)}),
        )
        cluster = self.drone.frontier_registry.refresh(
            self.drone.id,
            (component,),
            slam_version=self.drone.slam_map.version,
        )[0]
        self.drone.runtime_state.replace_frontier_clusters((cluster,))
        intent = NavigationIntent(
            mode=MovementMode.TRAVEL,
            cluster_id=cluster.id,
            target=(16, 16),
            scan_heading_count=3,
        )

        heading = self.drone.movement_controller._wall_continuation_heading(
            intent
        )

        self.assertAlmostEqual(heading, 90.0)

    def test_resolved_frontier_at_arrival_skips_redundant_rotation_scan(self) -> None:
        controller = self.drone.movement_controller
        target = self.drone.snapshot().position
        cluster_id = self._install_frontier_clusters(target)[0]
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        occupancy[0, 0] = OCCUPIED
        self.drone.slam_map.merge_from(SlamSnapshot(
            occupancy,
            np.ones((64, 64), dtype=np.float32),
        ))
        intent = self.drone.runtime_state.set_navigation_intent(
            NavigationIntent(
                mode=MovementMode.TRAVEL,
                cluster_id=cluster_id,
                target=target,
            )
        )

        outcome = controller._begin_scan(intent)

        self.assertTrue(outcome.scan_complete)
        self.assertEqual(
            outcome.transition_reason,
            TransitionReason.SCAN_COMPLETE,
        )
        self.assertIsNone(self.drone.snapshot().navigation_intent)
        self.assertEqual(
            self.drone.frontier_registry.get(cluster_id).lifecycle,
            "retired",
        )

    def test_gateway_arrival_scan_keeps_normalized_terminal_route_cursor(self) -> None:
        from navigation.navigation_intent import MovementMode, NavigationIntent

        intent = NavigationIntent(
            mode=MovementMode.TRAVEL,
            cluster_id=91,
            target=(18, 16),
            route_paths=(((16, 16), (18, 16)),),
            route_sources=("travelled",),
            route_segment_edge_ids=(None,),
            remaining_route_cost=2.0,
        )
        self.drone.runtime_state.set_navigation_intent(intent)

        outcome = self.drone.movement_controller._execute_navigation_intent(intent)
        scanning = self.drone.snapshot().navigation_intent

        self.assertEqual(outcome.transition_reason, TransitionReason.SCAN_STARTED)
        self.assertEqual(scanning.mode, MovementMode.SCAN)
        self.assertEqual(scanning.edge_cursor, 1)
        self.assertEqual(scanning.polyline_cursor, 0)
        self.assertEqual(scanning.remaining_route_cost, 0.0)

    def test_confidence_only_scan_gain_prevents_zero_gain_retirement(self) -> None:
        from navigation.navigation_intent import MovementMode, NavigationIntent

        intent = NavigationIntent(
            mode=MovementMode.SCAN,
            target=self.drone.snapshot().position,
            scan_heading_cursor=5,
            scan_sequence=5,
            scan_start_sensor_newly_known_cells=4,
            scan_start_sensor_confidence_gain=2.0,
        )
        self.drone.runtime_state.set_navigation_intent(intent)
        progress = SlamProgressSnapshot(
            completed_scan_sequence=6,
            sensor_newly_known_cells=4,
            sensor_confidence_gain=2.75,
        )

        with patch.object(
            self.drone.slam_map,
            "progress_snapshot",
            return_value=progress,
        ):
            outcome = self.drone.movement_controller._execute_scan_intent(intent)

        self.assertTrue(outcome.scan_complete)
        self.assertAlmostEqual(outcome.actual_information_gain, 0.75)
        self.assertEqual(
            outcome.transition_reason,
            TransitionReason.SCAN_COMPLETE,
        )

    def test_shared_progress_does_not_count_as_local_scan_gain(self) -> None:
        intent = NavigationIntent(
            mode=MovementMode.SCAN,
            target=self.drone.snapshot().position,
            scan_heading_cursor=5,
            scan_sequence=5,
            scan_start_sensor_newly_known_cells=4,
            scan_start_sensor_confidence_gain=2.0,
        )
        self.drone.runtime_state.set_navigation_intent(intent)
        progress = SlamProgressSnapshot(
            completed_scan_sequence=6,
            sensor_newly_known_cells=4,
            sensor_confidence_gain=2.0,
            shared_newly_known_cells=100,
            shared_confidence_gain=75.0,
        )

        with patch.object(
            self.drone.slam_map,
            "progress_snapshot",
            return_value=progress,
        ):
            outcome = self.drone.movement_controller._execute_scan_intent(intent)

        self.assertEqual(outcome.actual_information_gain, 0.0)
        self.assertEqual(outcome.transition_reason, TransitionReason.ZERO_GAIN)

    def test_recovery_replacement_does_not_fall_through_to_global_selection(
        self,
    ) -> None:
        from navigation.navigation_intent import MovementMode, NavigationIntent

        controller = self.drone.movement_controller
        self.drone.runtime_state.move_to((17, 16))
        self.drone.runtime_state.move_to((18, 16))
        intent = NavigationIntent(
            mode=MovementMode.TRAVEL,
            target=(30, 16),
            route_paths=(((18, 16), (30, 16)),),
            route_sources=("travelled",),
            route_segment_edge_ids=(None,),
            remaining_route_cost=12.0,
        )
        self.drone.runtime_state.set_navigation_intent(intent)

        with (
            patch.object(
                controller,
                "_execute_active_navigation_intent",
                side_effect=lambda active: controller._start_recovery_intent(
                    active,
                    TransitionReason.STALLED,
                ),
            ),
            patch.object(
                controller,
                "choose_exploration_action",
                side_effect=AssertionError(
                    "new recovery intent must run on the next move"
                ),
            ),
        ):
            controller.move()

        recovery = self.drone.snapshot().navigation_intent
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.mode, MovementMode.RECOVERY)
        self.assertEqual(recovery.route_paths[0][0], (18, 16))

    def test_completed_recovery_consumes_the_watchdog_trigger(self) -> None:
        self._make_slam_known_free()
        controller = self.drone.movement_controller
        self.drone.runtime_state.move_to((17, 16))
        self.drone.runtime_state.move_to((18, 16))
        original = self.drone.runtime_state.set_navigation_intent(
            NavigationIntent(
                mode=MovementMode.TRAVEL,
                target=(36, 16),
                route_paths=(((18, 16), (36, 16)),),
                route_sources=("travelled",),
                route_segment_edge_ids=(None,),
                remaining_route_cost=18.0,
            )
        )
        self.drone.runtime_state.update_navigation_watchdog(
            NavigationWatchdog(
                last_progress_time=1.0,
                recent_visits=(1, 2, 1),
                reversal_count=2,
            )
        )

        controller._start_recovery_intent(
            original,
            TransitionReason.REVERSAL,
        )
        recovery = self.drone.snapshot().navigation_intent
        controller._execute_navigation_intent(recovery)
        next_intent = self.drone.runtime_state.set_navigation_intent(
            NavigationIntent(
                mode=MovementMode.TRAVEL,
                target=(36, 16),
                route_paths=(((16, 16), (36, 16)),),
                route_sources=("travelled",),
                route_segment_edge_ids=(None,),
                remaining_route_cost=20.0,
            )
        )

        outcome = controller._execute_navigation_intent(next_intent)

        self.assertEqual(outcome.transition_reason, TransitionReason.PROGRESS)
        active = self.drone.snapshot().navigation_intent
        self.assertIsNotNone(active)
        self.assertEqual(active.mode, MovementMode.TRAVEL)
        self.assertEqual(active.target, (36, 16))
        self.assertEqual(
            self.drone.snapshot().navigation_watchdog.reversal_count,
            0,
        )

    def test_watchdog_recovery_does_not_blacklist_reachable_cluster(self) -> None:
        target = (56, 16)
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        cluster_id = self._install_frontier_clusters(target)[0]
        self.drone.runtime_state.update_navigation_watchdog(
            NavigationWatchdog(
                last_progress_time=1.0,
                recent_visits=(1, 2, 1),
                reversal_count=2,
            )
        )

        moved = controller._reach_frontier_clusters((cluster_id,))

        self.assertTrue(moved)
        recovery = self.drone.snapshot().navigation_intent
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.mode, MovementMode.RECOVERY)
        self.assertNotIn(cluster_id, controller._unreachable_blacklist)
        self.assertNotIn(target, controller.border_retry_until)

    def test_scan_invokes_local_control_but_remains_sensor_sequence_gated(self) -> None:
        from navigation.navigation_intent import MovementMode, NavigationIntent

        controller = self.drone.movement_controller
        controller._begin_scan(NavigationIntent(
            mode=MovementMode.TRAVEL,
            target=self.drone.snapshot().position,
        ))
        policy = FixedLocalPolicy(ExplorationDecision(
            kind=ExplorationDecisionKind.ROTATE,
            target=self.drone.snapshot().position,
            direction=60,
            local_primitive="rotate_scan",
        ))
        self.drone.exploration_policy = policy

        controller.move()
        waiting = self.drone.snapshot().navigation_intent
        self.assertEqual(len(policy.local_contexts), 1)
        self.assertEqual(waiting.scan_heading_cursor, 0)

        self.drone.slam_map.update_from_rays((16, 16), ())
        controller.move()
        advanced = self.drone.snapshot().navigation_intent

        self.assertEqual(len(policy.local_contexts), 2)
        self.assertEqual(advanced.scan_heading_cursor, 1)
        self.assertEqual(advanced.previous_primitive, "rotate_scan")

    def test_off_route_local_rotation_waits_for_one_sensor_sequence(self) -> None:
        from navigation.navigation_intent import MovementMode, NavigationIntent

        controller = self.drone.movement_controller
        intent = NavigationIntent(
            mode=MovementMode.TRAVEL,
            target=(30, 16),
            route_paths=(((16, 16), (30, 16)),),
            route_sources=("travelled",),
            route_segment_edge_ids=(None,),
            remaining_route_cost=14.0,
        )
        self.drone.runtime_state.set_navigation_intent(intent)
        self.drone.runtime_state.move_to((16, 18))
        policy = FixedLocalPolicy(ExplorationDecision(
            kind=ExplorationDecisionKind.ROTATE,
            target=(16, 18),
            direction=60,
            local_primitive="rotate_scan",
        ))
        self.drone.exploration_policy = policy

        controller.move()
        pending = self.drone.snapshot().navigation_intent
        self.assertTrue(pending.local_scan_pending)
        self.assertEqual(pending.mode, MovementMode.TRAVEL)
        self.assertEqual(len(policy.local_contexts), 1)

        controller.move()
        self.assertTrue(self.drone.snapshot().navigation_intent.local_scan_pending)
        self.assertEqual(len(policy.local_contexts), 1)

        self.drone.slam_map.update_from_rays((16, 18), ())
        controller.move()
        completed = self.drone.snapshot().navigation_intent
        self.assertFalse(completed.local_scan_pending)
        self.assertEqual(completed.mode, MovementMode.TRAVEL)
        self.assertEqual(len(policy.local_contexts), 1)

    def test_scan_reservation_loss_invalidates_before_local_rotation(self) -> None:
        from navigation.navigation_intent import MovementMode, NavigationIntent

        assignment = self.drone.frontier_assignments.reserve(
            cluster_id=91,
            drone_id=self.drone.id,
            gateway_id=7,
        )
        intent = NavigationIntent(
            mode=MovementMode.SCAN,
            cluster_id=91,
            gateway_id=7,
            assignment_token=assignment.token,
            target=self.drone.snapshot().position,
        )
        self.drone.runtime_state.set_navigation_intent(intent)
        self.drone.frontier_assignments.release(
            assignment.token,
            drone_id=self.drone.id,
        )

        outcome = self.drone.movement_controller._execute_navigation_intent(
            intent
        )

        self.assertTrue(outcome.invalidated)
        self.assertEqual(
            outcome.transition_reason,
            TransitionReason.RESERVATION_LOST,
        )
        self.assertIsNone(self.drone.snapshot().navigation_intent)

    def test_unroutable_far_frontier_falls_back_to_near_direct_target(self) -> None:
        far_target = (56, 56)
        near_target = (20, 16)
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
        )
        far_id, near_id = self._install_frontier_clusters(
            far_target, near_target,
        )
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        known_free[far_target[1], far_target[0]] = False

        with (
            patch.object(controller, "_known_free_mask", return_value=known_free),
            patch.object(
                controller,
                "_score_frontier_clusters",
                return_value=(far_id, near_id),
            ),
        ):
            moved = controller._reach_frontier_clusters([far_id, near_id])

        self.assertTrue(moved)
        self.assertEqual(self.drone.snapshot().position, near_target)
        self.assertIn(far_target, self.drone.snapshot().frontiers)
        self.assertIn(far_target, controller.border_retry_until)
        self.assertEqual(self.control.path_requests, [])

    def test_bridge_advances_far_frontier_before_near_direct_target(self) -> None:
        far_target = (56, 40)
        near_target = (20, 16)
        graph = self._configured_highway_graph()
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
            runtime_trace=trace,
        )
        far_id, near_id = self._install_frontier_clusters(
            far_target, near_target,
        )
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )

        self.assertGreater(
            math.dist((52, 16), far_target),
            graph.connector_distance,
        )
        self.assertLessEqual(
            math.dist((52, 16), far_target),
            controller._waypoint_bridge_distance(),
        )
        self.assertEqual(
            graph.find_route((16, 16), far_target, known_free).status,
            "no_goal_connector",
        )

        with patch.object(
            controller,
            "_score_frontier_clusters",
            return_value=(far_id, near_id),
        ):
            moved = controller._reach_frontier_clusters([far_id, near_id])

        self.assertTrue(moved)
        self.assertEqual(self.drone.snapshot().position, (23, 20))
        self.assertIn(far_target, self.drone.snapshot().frontiers)
        self.assertIn(near_target, self.drone.snapshot().frontiers)
        self.assertEqual(
            self.control.path_requests,
            [],
        )
        bridge_event = next(
            fields
            for event, fields in trace.events
            if event == "drone_waypoint_bridge"
        )
        self.assertEqual(bridge_event["status"], "ok")
        self.assertGreater(bridge_event["added_edge_count"], 0)
        route_event = next(
            fields
            for event, fields in trace.events
            if event == "drone_waypoint_route"
        )
        self.assertGreaterEqual(route_event["route_lookup_calls"], 1)
        self.assertGreaterEqual(route_event["route_lookup_elapsed_ms"], 0.0)
        self.assertGreaterEqual(route_event["route_repair_elapsed_ms"], 0.0)
        self.assertAlmostEqual(
            route_event["route_elapsed_ms"],
            route_event["route_lookup_elapsed_ms"]
            + route_event["route_repair_elapsed_ms"],
        )

    def test_bridge_repairs_disconnected_shared_gateway(self) -> None:
        far_target = (56, 40)
        graph = self._configured_highway_graph()
        graph.add_waypoint(far_target, source="gateway")
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
            runtime_trace=trace,
        )
        far_id = self._install_frontier_clusters(far_target)[0]
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )

        self.assertEqual(
            graph.find_route((16, 16), far_target, known_free).status,
            "disconnected",
        )

        with patch.object(
            controller, "_score_frontier_clusters", return_value=(far_id,),
        ):
            moved = controller._reach_frontier_clusters([far_id])

        self.assertTrue(moved)
        self.assertEqual(self.drone.snapshot().position, (23, 20))
        bridge_event = next(
            fields
            for event, fields in trace.events
            if event == "drone_waypoint_bridge"
        )
        self.assertEqual(
            bridge_event["initial_route_status"],
            "disconnected",
        )
        self.assertEqual(bridge_event["status"], "ok")

    def test_non_bridgeable_target_does_not_consume_bridge_budget(self) -> None:
        unknown_target = (56, 56)
        bridge_target = (56, 40)
        graph = self._configured_highway_graph()
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
            runtime_trace=trace,
        )
        unknown_id, bridge_id = self._install_frontier_clusters(
            unknown_target, bridge_target,
        )
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        known_free[unknown_target[1], unknown_target[0]] = False

        with (
            patch.object(
                controller,
                "_known_free_mask",
                return_value=known_free,
            ),
            patch.object(
                controller,
                "_score_frontier_clusters",
                return_value=(unknown_id, bridge_id),
            ),
        ):
            moved = controller._reach_frontier_clusters(
                [unknown_id, bridge_id]
            )

        self.assertTrue(moved)
        self.assertEqual(self.drone.snapshot().position, (23, 20))
        bridge_events = [
            fields
            for event, fields in trace.events
            if event == "drone_waypoint_bridge"
        ]
        self.assertEqual(len(bridge_events), 1)
        self.assertEqual(bridge_events[0]["target"], bridge_target)
        self.assertEqual(bridge_events[0]["status"], "ok")

    def test_nonempty_frontier_view_without_actionable_waypoint_waits_boundedly(
        self,
    ) -> None:
        """Shared canonical work must not become a hot stalled-action loop."""
        target = (40, 40)
        cluster_id = self._install_frontier_clusters(target)[0]
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        coordinator = TeamExplorationCoordinator(
            registry=self.drone.frontier_registry,
            assignments=self.drone.frontier_assignments,
            get_drones=lambda: (self.drone,),
        )
        self.drone.exploration_coordinator = coordinator
        policy = FixedDecisionPolicy(ExplorationDecision(
            kind=ExplorationDecisionKind.FRONTIER,
            target=target,
            cluster_id=cluster_id,
            frontier_cluster_ids=(cluster_id,),
        ))
        self.drone.exploration_policy = policy

        with patch.object(
            controller,
            "_score_frontier_clusters",
            side_effect=((), ()),
        ) as score:
            controller.move()
            controller.move()

        self.assertEqual(score.call_count, 2)
        self.assertEqual(len(policy.contexts), 1)
        self.assertFalse(self.drone.snapshot().returning_home)
        rebuilt = [
            fields for event, fields in trace.events
            if event == "drone_frontiers_rebuilt"
        ]
        self.assertEqual(len(rebuilt), 1)
        waiting = [
            fields for event, fields in trace.events
            if event == "drone_waiting_for_team_frontier"
        ]
        self.assertEqual(len(waiting), 1)
        self.assertEqual(waiting[0]["reason"], "no_actionable_frontier")
        self.assertEqual(waiting[0]["requested_cluster_ids"], (cluster_id,))
        action = next(
            fields for event, fields in trace.events
            if event == "drone_action_result"
        )
        self.assertEqual(
            action["movement_outcome"]["transition_reason"],
            "no_actionable_frontier",
        )

    def test_frontier_wait_wakes_immediately_when_shared_slam_changes(self) -> None:
        target = (40, 40)
        cluster_id = self._install_frontier_clusters(target)[0]
        controller = self.drone.movement_controller
        coordinator = TeamExplorationCoordinator(
            registry=self.drone.frontier_registry,
            assignments=self.drone.frontier_assignments,
            get_drones=lambda: (self.drone,),
        )
        self.drone.exploration_coordinator = coordinator
        policy = FixedDecisionPolicy(ExplorationDecision(
            kind=ExplorationDecisionKind.FRONTIER,
            target=target,
            cluster_id=cluster_id,
            frontier_cluster_ids=(cluster_id,),
        ))
        self.drone.exploration_policy = policy

        with patch.object(
            controller,
            "_score_frontier_clusters",
            side_effect=((), (), (), ()),
        ) as score:
            controller.move()
            self._make_slam_known_free()
            controller.move()

        self.assertEqual(score.call_count, 4)
        self.assertEqual(len(policy.contexts), 2)

    def test_bridge_budget_skip_does_not_blacklist_unattempted_target(self) -> None:
        attempted_target = (56, 40)
        skipped_target = (56, 44)
        graph = self._configured_highway_graph()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
        )
        attempted_id, skipped_id = self._install_frontier_clusters(
            attempted_target, skipped_target,
        )
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        start = self.drone.snapshot().position
        for target in (attempted_target, skipped_target):
            direct = bresenham_path(start, target)
            blocked = direct[len(direct) // 2]
            known_free[blocked[1], blocked[0]] = False

        with (
            patch.object(
                controller,
                "_score_frontier_clusters",
                return_value=(attempted_id, skipped_id),
            ),
            patch.object(
                graph,
                "connect_known_free_corridor",
                return_value=GraphUpdate(status="no_connector"),
            ) as connect_corridor,
            patch.object(
                controller,
                "_known_free_mask",
                return_value=known_free,
            ),
        ):
            moved = controller._reach_frontier_clusters(
                [attempted_id, skipped_id]
            )

        self.assertFalse(moved)
        connect_corridor.assert_called_once()
        self.assertIn(attempted_id, controller._unreachable_blacklist)
        self.assertIn(attempted_target, controller.border_retry_until)
        self.assertNotIn(skipped_id, controller._unreachable_blacklist)
        self.assertNotIn(skipped_target, controller.border_retry_until)

    def _install_frontier_clusters(self, *targets):
        """Install authoritative stable clusters for movement integration tests."""
        components = tuple(
            FrontierComponent(
                cells=frozenset({target}),
                bounds=(target[0], target[1], target[0] + 1, target[1] + 1),
                representative=target,
                expected_gain=1,
            )
            for target in targets
        )
        clusters = self.drone.frontier_registry.refresh(
            self.drone.id,
            components,
            slam_version=self.drone.slam_map.version,
        )
        self.drone.runtime_state.replace_frontier_clusters(clusters)
        cluster_by_target = {
            cluster.representative: cluster.id for cluster in clusters
        }
        return tuple(cluster_by_target[target] for target in targets)

    def _configured_highway_graph(self) -> WaypointGraph:
        """Install a known-free travelled chain for waypoint integration tests."""
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        self.drone.slam_map.merge_from(
            SlamSnapshot(occupancy, confidence)
        )
        graph = WaypointGraph(
            merge_radius=0.0,
            connector_distance=15.0,
            connector_limit=4,
        )
        graph.register_travelled_section(
            [(16, 16), (52, 16)],
            start_role=WaypointRole.HOME,
            end_role=WaypointRole.RECOVERY_ANCHOR,
        )
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        return graph

    def test_move_executes_policy_homing_decision_and_marks_done(self) -> None:
        self.drone.runtime_state.move_to((20, 20))
        self._make_slam_known_free()
        self.drone.exploration_policy = FixedDecisionPolicy(
            ExplorationDecision(
                kind=ExplorationDecisionKind.HOMING,
                target=(16, 16),
            )
        )

        self.drone.movement_controller.move()
        snapshot = self.drone.snapshot()

        self.assertEqual(snapshot.position, (16, 16))
        self.assertTrue(snapshot.done)
        self.assertEqual(self.control.path_requests, [])

    def _make_slam_known_free(self) -> None:
        """Install a detached belief corridor for belief-only route tests."""
        self.drone.slam_map.merge_from(SlamSnapshot(
            np.full((64, 64), FREE, dtype=np.int8),
            np.ones((64, 64), dtype=np.float32),
        ))

    def test_mutable_runtime_fields_are_owned_by_state(self) -> None:
        self.assertFalse(hasattr(self.drone, "pos"))
        self.assertFalse(hasattr(self.drone, "border"))
        self.assertFalse(hasattr(self.drone, "graph"))
        self.assertFalse(hasattr(self.drone, "exploration_lock"))
        self.assertEqual(
            self.drone.snapshot().position,
            self.drone.start_pos,
        )

    def test_path_traversal_is_motionless_after_pause_barrier_returns(self) -> None:
        stop_event = threading.Event()
        coordinator = PauseCoordinator(stop_event)
        self.control.pause_checkpoint = coordinator.checkpoint
        self.control.wait_simulation_delay = coordinator.wait
        self.drone.movement_controller.dependencies = replace(
            self.drone.movement_controller.dependencies,
            pause_checkpoint=coordinator.checkpoint,
            wait_simulation_delay=coordinator.wait,
        )
        self.drone.delay = 0.2
        self.drone.speed_factor = 1
        first_node = threading.Event()
        original_move_to = self.drone.runtime_state.move_to

        def record_node(node) -> None:
            original_move_to(node)
            first_node.set()

        self.drone.runtime_state.move_to = record_node
        path = [(17, 16), (18, 16), (19, 16), (20, 16)]

        def follow_path() -> None:
            coordinator.register_current_worker(("drone", 0))
            try:
                self.drone.movement_controller._follow_path(path)
            finally:
                coordinator.unregister_current_worker()

        worker = threading.Thread(target=follow_path)
        worker.start()
        self.assertTrue(first_node.wait(2.0))

        coordinator.pause()
        paused_position = self.drone.snapshot().position
        time.sleep(0.25)
        paused_snapshot = self.drone.snapshot()

        self.assertEqual(paused_snapshot.position, paused_position)
        self.assertNotEqual(paused_snapshot.position, path[-1])

        coordinator.resume()
        worker.join(2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.drone.snapshot().position, path[-1])


if __name__ == "__main__":
    unittest.main()
