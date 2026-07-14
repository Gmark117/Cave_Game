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
from asset_config.helpers import next_cell_coords
from config.simulation_config import (
    ExplorationConfig,
    MissionConfig,
    SimulationConfig,
    SlamConfig,
)
from mapping.slam_map import FREE, UNKNOWN, SlamSnapshot
from agents.drone_movement import DroneMovementController
from mapping.terrain_knowledge import TerrainKnowledge
from mission.pause_control import PauseCoordinator
from navigation.waypoint_graph import WaypointGraph


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

    def decide(self, context, is_segment_valid):
        self.contexts.append(context)
        return self.decision


class VersionRecordingPolicy:
    def __init__(self) -> None:
        self.versions = []

    def decide(self, context, is_segment_valid):
        self.versions.append(context.slam_snapshot.version)
        return ExplorationDecision(
            kind=ExplorationDecisionKind.ROTATE,
            target=context.pose_estimate.position,
            direction=0,
            frontier_targets=(context.pose_estimate.position,),
        )


class ExhaustedPolicy:
    def __init__(self) -> None:
        self.contexts = []

    def decide(self, context, is_segment_valid):
        self.contexts.append(context)
        return ExplorationDecision(kind=ExplorationDecisionKind.EXHAUSTED)

    def extract_frontiers(
        self,
        context,
        *,
        stride=None,
        confidence_threshold=None,
    ):
        return ()


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
        self.control.paths[((20, 20), (16, 16))] = [
            (20, 20),
            (18, 18),
            (16, 16),
        ]

        reached_home = self.drone.movement_controller.reach_start_point()
        snapshot = self.drone.snapshot()

        self.assertTrue(reached_home)
        self.assertEqual(snapshot.position, (16, 16))
        self.assertEqual(snapshot.path_history[-1], (16, 16))
        self.assertNotEqual(snapshot.heading_deg, 0.0)

    def test_frontier_rebuild_uses_local_slam_state(self) -> None:
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[20, 20] = FREE
        confidence[20, 20] = 1.0
        self.drone.slam_map.merge_from(
            SlamSnapshot(occupancy, confidence)
        )

        self.drone.movement_controller.rebuild_frontiers(
            stride=1,
            confidence_threshold=0.6,
        )

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

        self.drone.movement_controller.rebuild_frontiers(
            stride=1,
            confidence_threshold=0.6,
        )

        self.assertEqual(self.drone.snapshot().frontiers, ())

        self.drone.terrain_knowledge.confidence[20, 20] = 1.0
        self.drone.movement_controller.rebuild_frontiers(
            stride=1,
            confidence_threshold=0.6,
        )

        self.assertEqual(self.drone.snapshot().frontiers, ((20, 20),))

    def test_movement_configuration_is_controller_owned(self) -> None:
        self.assertIsInstance(
            self.drone.movement_controller,
            DroneMovementController,
        )
        controller = self.drone.movement_controller
        self.assertEqual(controller.frontier_stride, 4)

        controller.frontier_stride = 2
        controller.border_retry_cooldown = 3.0

        self.assertEqual(controller.frontier_stride, 2)
        self.assertEqual(controller.border_retry_cooldown, 3.0)

    def test_short_policy_moves_share_waypoint_spacing_across_actions(self) -> None:
        graph = WaypointGraph(
            spacing=32.0,
            merge_radius=4.0,
            connector_distance=64.0,
            connector_limit=8,
        )
        graph.add_waypoint((16, 16), source="home")
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        controller._waypoint_pending_path = [(16, 16)]

        for target in ((26, 16), (36, 16), (46, 16), (56, 16)):
            self.assertTrue(controller._follow_policy_path([target]))

        self.assertEqual(graph.counts(), (2, 1))
        self.assertEqual(
            tuple(node.position for node in graph.snapshot().nodes),
            ((16, 16), (48, 16)),
        )
        self.assertEqual(
            (
                controller._waypoint_pending_path[0],
                controller._waypoint_pending_path[-1],
            ),
            ((48, 16), (56, 16)),
        )

    def test_waypoint_spacing_survives_many_registration_chunks(self) -> None:
        graph = WaypointGraph(
            spacing=32.0,
            merge_radius=4.0,
            connector_distance=64.0,
            connector_limit=8,
        )
        graph.add_waypoint((0, 0), source="home")
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        controller._waypoint_pending_path = [(0, 0)]

        for index in range(100):
            controller._register_travelled_path(
                [(index * 10, 0), ((index + 1) * 10, 0)]
            )

        self.assertEqual(graph.counts(), (32, 31))
        self.assertEqual(
            (
                controller._waypoint_pending_path[0],
                controller._waypoint_pending_path[-1],
            ),
            ((992, 0), (1000, 0)),
        )
        controller._flush_pending_waypoint_path(force=True)
        self.assertEqual(graph.counts(), (33, 32))
        self.assertEqual(controller._waypoint_pending_path, [(1000, 0)])

    def test_move_executes_policy_step_decision(self) -> None:
        target = (18, 16)
        self.control.paths[((16, 16), target)] = [target]
        self.drone.exploration_policy = FixedDecisionPolicy(
            ExplorationDecision(
                kind=ExplorationDecisionKind.STEP,
                target=target,
                direction=90,
                valid_directions=(90,),
                frontier_targets=((24, 16),),
            )
        )

        self.drone.movement_controller.move()
        snapshot = self.drone.snapshot()

        self.assertEqual(snapshot.position, target)
        self.assertEqual(snapshot.direction, 90)
        self.assertEqual(snapshot.frontiers, ((24, 16),))
        self.assertEqual(
            self.drone.exploration_policy.contexts[0].pose_estimate.position,
            (16, 16),
        )

    def test_empty_cached_frontiers_do_not_force_homing_before_policy_decides(self) -> None:
        target = (18, 16)
        self.control.paths[((16, 16), target)] = [target]
        self.drone.runtime_state.begin_exploration(0, [])
        self.drone.exploration_policy = FixedDecisionPolicy(
            ExplorationDecision(
                kind=ExplorationDecisionKind.STEP,
                target=target,
                direction=90,
                valid_directions=(90,),
                frontier_targets=((24, 16),),
            )
        )

        self.drone.movement_controller.move()
        first_context = self.drone.exploration_policy.contexts[0]

        self.assertFalse(first_context.runtime_snapshot.returning_home)
        self.assertEqual(self.drone.snapshot().position, target)
        self.assertFalse(self.drone.snapshot().returning_home)

    def test_confirmed_exhaustion_starts_homing_after_frontier_rebuild(self) -> None:
        self.drone.runtime_state.move_to((20, 16))
        self.control.paths[((20, 16), (16, 16))] = [
            (20, 16),
            (16, 16),
        ]
        self.drone.exploration_policy = ExhaustedPolicy()

        self.drone.movement_controller.move()
        snapshot = self.drone.snapshot()

        self.assertEqual(snapshot.position, (16, 16))
        self.assertTrue(snapshot.returning_home)
        self.assertTrue(snapshot.done)
        self.assertEqual(len(self.drone.exploration_policy.contexts), 2)

    def test_move_executes_policy_planned_path_with_safety_validation(self) -> None:
        target = (18, 16)
        self.drone.exploration_policy = FixedDecisionPolicy(
            ExplorationDecision(
                kind=ExplorationDecisionKind.STEP,
                target=target,
                direction=90,
                valid_directions=(90,),
                frontier_targets=((24, 16),),
                planned_path=((17, 16), target),
            )
        )
        calls = []
        original_is_valid = self.drone.runtime_state.graph_is_valid

        def record_validation(current, candidate):
            calls.append((current, candidate))
            return original_is_valid(current, candidate)

        self.drone.runtime_state.graph_is_valid = record_validation

        self.drone.movement_controller.move()

        self.assertEqual(self.drone.snapshot().position, target)
        self.assertEqual(
            calls,
            [((16, 16), (17, 16)), ((17, 16), target)],
        )
        self.assertEqual(len(self.drone.exploration_policy.contexts), 1)

    def test_rejected_policy_step_records_collision_in_slam(self) -> None:
        blocked = (17, 16)
        self.drone.cave[blocked[1], blocked[0]] = 1
        self.drone.exploration_policy = FixedDecisionPolicy(
            ExplorationDecision(
                kind=ExplorationDecisionKind.STEP,
                target=(18, 16),
                direction=90,
                valid_directions=(90,),
                planned_path=(blocked, (18, 16)),
            )
        )

        self.drone.movement_controller.move()
        slam = self.drone.slam_map.snapshot(point_limit=0)

        self.assertEqual(self.drone.snapshot().position, (16, 16))
        self.assertEqual(int(slam.occupancy[blocked[1], blocked[0]]), 1)
        self.assertEqual(float(slam.confidence[blocked[1], blocked[0]]), 1.0)

    def test_move_executes_policy_rotate_decision(self) -> None:
        self.drone.exploration_policy = FixedDecisionPolicy(
            ExplorationDecision(
                kind=ExplorationDecisionKind.ROTATE,
                target=(16, 16),
                direction=135,
                frontier_targets=((24, 16),),
            )
        )

        self.drone.movement_controller.move()
        snapshot = self.drone.snapshot()

        self.assertEqual(snapshot.position, (16, 16))
        self.assertEqual(snapshot.direction, 135)
        self.assertEqual(snapshot.heading_deg, 135.0)
        self.assertEqual(snapshot.frontiers, ((24, 16),))

    def test_choose_action_replans_from_newer_slam_snapshot(self) -> None:
        policy = VersionRecordingPolicy()
        self.drone.exploration_policy = policy

        self.drone.movement_controller.move()

        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[20, 20] = FREE
        confidence[20, 20] = 1.0
        self.drone.slam_map.merge_from(
            SlamSnapshot(occupancy, confidence)
        )
        self.drone.movement_controller.move()

        self.assertEqual(policy.versions, [0, 1])

    def test_move_executes_policy_frontier_decision(self) -> None:
        target = (24, 16)
        self.control.paths[((16, 16), target)] = [(16, 16), target]
        self.drone.runtime_state.merge_frontiers([target])
        self.drone.exploration_policy = FixedDecisionPolicy(
            ExplorationDecision(
                kind=ExplorationDecisionKind.FRONTIER,
                target=target,
                frontier_targets=(target,),
            )
        )

        self.drone.movement_controller.move()

        self.assertEqual(self.drone.snapshot().position, target)
        self.assertEqual(self.drone.snapshot().frontiers, ())

    def test_failed_direct_frontier_path_uses_one_waypoint_segment(self) -> None:
        target = (56, 16)
        graph = self._configured_highway_graph(direct_path_limit=100.0)
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
            runtime_trace=trace,
        )
        self.control.paths[((16, 16), (36, 16))] = [
            (16, 16),
            (36, 16),
        ]
        self.drone.runtime_state.merge_frontiers([target])

        moved = controller._reach_frontier_targets([target])

        self.assertTrue(moved)
        self.assertEqual(self.drone.snapshot().position, (36, 16))
        self.assertIn(target, self.drone.snapshot().frontiers)
        self.assertEqual(
            self.control.path_requests,
            [((16, 16), target), ((16, 16), (36, 16))],
        )
        event_names = [event for event, _fields in trace.events]
        self.assertIn("drone_frontier_direct_path_failed", event_names)
        self.assertIn("drone_waypoint_route", event_names)
        self.assertIn("drone_waypoint_segment_path", event_names)
        self.assertNotIn("drone_frontier_reached", event_names)
        route_event = next(
            fields
            for event, fields in trace.events
            if event == "drone_waypoint_route"
        )
        self.assertEqual(route_event["status"], "ok")
        self.assertEqual(route_event["gateway_status"], "ok")
        self.assertGreaterEqual(route_event["route_elapsed_ms"], 0.0)
        self.assertGreater(route_event["graph_nodes"], 0)

    def test_waypoint_frontier_replans_and_consumes_only_at_target(self) -> None:
        target = (56, 16)
        graph = self._configured_highway_graph(direct_path_limit=1.0)
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
            runtime_trace=trace,
        )
        self.control.paths.update(
            {
                ((16, 16), (36, 16)): [(16, 16), (36, 16)],
                ((36, 16), (52, 16)): [(36, 16), (52, 16)],
                ((52, 16), target): [(52, 16), target],
            }
        )
        self.drone.runtime_state.merge_frontiers([target])

        self.assertTrue(controller._reach_frontier_targets([target]))
        self.assertEqual(self.drone.snapshot().position, (36, 16))
        self.assertIn(target, self.drone.snapshot().frontiers)

        self.assertTrue(controller._reach_frontier_targets([target]))
        self.assertEqual(self.drone.snapshot().position, (52, 16))
        self.assertIn(target, self.drone.snapshot().frontiers)

        self.assertTrue(controller._reach_frontier_targets([target]))
        self.assertEqual(self.drone.snapshot().position, target)
        self.assertNotIn(target, self.drone.snapshot().frontiers)
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
        self.assertEqual(len(segment_events), 3)
        self.assertEqual(len(reached_events), 1)

    def test_unroutable_far_frontier_falls_back_to_near_direct_target(self) -> None:
        far_target = (56, 56)
        near_target = (20, 16)
        graph = self._configured_highway_graph(direct_path_limit=10.0)
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
        )
        self.control.paths[((16, 16), near_target)] = [
            (16, 16),
            near_target,
        ]
        self.drone.runtime_state.merge_frontiers(
            [far_target, near_target]
        )

        moved = controller._reach_frontier_targets(
            [far_target, near_target]
        )

        self.assertTrue(moved)
        self.assertEqual(self.drone.snapshot().position, near_target)
        self.assertIn(far_target, self.drone.snapshot().frontiers)
        self.assertNotIn(near_target, self.drone.snapshot().frontiers)
        self.assertIn(far_target, controller.border_retry_until)
        self.assertEqual(
            self.control.path_requests,
            [((16, 16), near_target)],
        )

    def test_bridge_advances_far_frontier_before_near_direct_target(self) -> None:
        far_target = (56, 40)
        near_target = (20, 16)
        graph = self._configured_highway_graph(direct_path_limit=10.0)
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
            runtime_trace=trace,
        )
        self.control.paths[((16, 16), (36, 16))] = [
            (16, 16),
            (36, 16),
        ]
        self.control.paths[((16, 16), near_target)] = [
            (16, 16),
            near_target,
        ]
        self.drone.runtime_state.merge_frontiers(
            [far_target, near_target]
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

        moved = controller._reach_frontier_targets(
            [far_target, near_target]
        )

        self.assertTrue(moved)
        self.assertEqual(self.drone.snapshot().position, (36, 16))
        self.assertIn(far_target, self.drone.snapshot().frontiers)
        self.assertIn(near_target, self.drone.snapshot().frontiers)
        self.assertEqual(
            self.control.path_requests,
            [((16, 16), (36, 16))],
        )
        bridge_event = next(
            fields
            for event, fields in trace.events
            if event == "drone_waypoint_bridge"
        )
        self.assertEqual(bridge_event["status"], "ok")
        self.assertGreater(bridge_event["added_edge_count"], 0)

    def test_bridge_repairs_disconnected_shared_gateway(self) -> None:
        far_target = (56, 40)
        graph = self._configured_highway_graph(direct_path_limit=10.0)
        graph.add_waypoint(far_target, source="gateway")
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
            runtime_trace=trace,
        )
        self.control.paths[((16, 16), (36, 16))] = [
            (16, 16),
            (36, 16),
        ]
        self.drone.runtime_state.merge_frontiers([far_target])
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )

        self.assertEqual(
            graph.find_route((16, 16), far_target, known_free).status,
            "disconnected",
        )

        moved = controller._reach_frontier_targets([far_target])

        self.assertTrue(moved)
        self.assertEqual(self.drone.snapshot().position, (36, 16))
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
        graph = self._configured_highway_graph(direct_path_limit=10.0)
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
            runtime_trace=trace,
        )
        self.control.paths[((16, 16), (36, 16))] = [
            (16, 16),
            (36, 16),
        ]
        self.drone.runtime_state.merge_frontiers(
            [unknown_target, bridge_target]
        )
        known_free = controller._known_free_mask(
            self.drone.slam_map.snapshot(point_limit=0)
        )
        known_free[unknown_target[1], unknown_target[0]] = False

        with patch.object(
            controller,
            "_known_free_mask",
            return_value=known_free,
        ):
            moved = controller._reach_frontier_targets(
                [unknown_target, bridge_target]
            )

        self.assertTrue(moved)
        self.assertEqual(self.drone.snapshot().position, (36, 16))
        bridge_events = [
            fields
            for event, fields in trace.events
            if event == "drone_waypoint_bridge"
        ]
        self.assertEqual(len(bridge_events), 1)
        self.assertEqual(bridge_events[0]["target"], bridge_target)
        self.assertEqual(bridge_events[0]["status"], "ok")

    def test_unsafe_astar_shortcut_uses_trusted_route_shape(self) -> None:
        target = (36, 16)
        graph = WaypointGraph(
            spacing=100.0,
            merge_radius=0.0,
            connector_distance=4.0,
            connector_limit=4,
        )
        travelled = (
            (16, 16),
            (16, 24),
            (36, 24),
            target,
        )
        graph.register_travelled_path(travelled)
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        for x, y in ((16, 16), target):
            occupancy[y, x] = FREE
            confidence[y, x] = 1.0
        self.drone.slam_map.merge_from(
            SlamSnapshot(occupancy, confidence)
        )
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        controller.waypoint_config = replace(
            self.drone.settings.waypoints,
            direct_path_limit=1.0,
        )
        controller.dependencies = replace(
            controller.dependencies,
            waypoint_graph=graph,
            runtime_trace=trace,
        )
        self.control.paths[((16, 16), target)] = [
            (16, 16),
            (26, 16),
            target,
        ]
        self.drone.runtime_state.merge_frontiers([target])

        moved = controller._reach_frontier_targets([target])

        self.assertTrue(moved)
        self.assertEqual(self.drone.snapshot().position, target)
        self.assertNotIn(target, self.drone.snapshot().frontiers)
        self.assertIn((16, 24), self.drone.snapshot().path_history)
        segment_event = next(
            fields
            for event, fields in trace.events
            if event == "drone_waypoint_segment_path"
        )
        self.assertEqual(
            segment_event["path_source"],
            "trusted_route_fallback",
        )
        self.assertEqual(segment_event["astar_path_len"], 3)
        self.assertGreater(segment_event["path_len"], 3)

    def _configured_highway_graph(
        self,
        *,
        direct_path_limit: float,
    ) -> WaypointGraph:
        """Install a known-free travelled chain for waypoint integration tests."""
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        self.drone.slam_map.merge_from(
            SlamSnapshot(occupancy, confidence)
        )
        graph = WaypointGraph(
            spacing=20.0,
            merge_radius=0.0,
            connector_distance=15.0,
            connector_limit=4,
        )
        graph.register_travelled_path([(16, 16), (52, 16)])
        controller = self.drone.movement_controller
        controller.waypoint_graph = graph
        controller.waypoint_config = replace(
            self.drone.settings.waypoints,
            direct_path_limit=direct_path_limit,
        )
        return graph

    def test_move_executes_policy_homing_decision_and_marks_done(self) -> None:
        self.drone.runtime_state.move_to((20, 20))
        self.control.paths[((20, 20), (16, 16))] = [
            (20, 20),
            (16, 16),
        ]
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

    def test_find_new_node_keeps_integer_degree_candidates(self) -> None:
        current_position = self.drone.snapshot().position
        calls = []

        def graph_is_valid(current, candidate) -> bool:
            calls.append((current, candidate))
            return True

        self.drone.runtime_state.graph_is_valid = graph_is_valid

        with patch("agents.exploration_policy.rand.choice", return_value=90):
            valid_dirs, valid_targets, target = (
                self.drone.movement_controller.find_new_node()
            )

        expected_target = next_cell_coords(
            *current_position,
            self.drone.step,
            90,
        )
        self.assertEqual(valid_dirs, list(range(360)))
        self.assertEqual(len(valid_targets), 360)
        self.assertEqual(target, expected_target)
        self.assertEqual(self.drone.snapshot().direction, 90)
        self.assertEqual(len(calls), 361)
        self.assertEqual(
            calls[0],
            (
                current_position,
                next_cell_coords(
                    *current_position,
                    self.drone.radius + 1,
                    0,
                ),
            ),
        )
        self.assertEqual(
            calls[359],
            (
                current_position,
                next_cell_coords(
                    *current_position,
                    self.drone.radius + 1,
                    359,
                ),
            ),
        )
        self.assertEqual(calls[-1], (current_position, expected_target))

    def test_find_new_node_removes_rejected_short_step(self) -> None:
        current_position = self.drone.snapshot().position
        rejected_step = next_cell_coords(
            *current_position,
            self.drone.step,
            90,
        )
        calls = 0

        def graph_is_valid(current, candidate) -> bool:
            nonlocal calls
            calls += 1
            return not (calls > 360 and candidate == rejected_step)

        self.drone.runtime_state.graph_is_valid = graph_is_valid

        with patch(
            "agents.exploration_policy.rand.choice",
            side_effect=[90, 180],
        ):
            valid_dirs, valid_targets, target = (
                self.drone.movement_controller.find_new_node()
            )

        self.assertNotIn(90, valid_dirs)
        self.assertEqual(len(valid_dirs), 359)
        self.assertEqual(len(valid_targets), 359)
        self.assertEqual(
            target,
            next_cell_coords(
                *current_position,
                self.drone.step,
                180,
            ),
        )
        self.assertEqual(self.drone.snapshot().direction, 180)

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
