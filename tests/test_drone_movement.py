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
)
from asset_config.helpers import next_cell_coords
from config.simulation_config import MissionConfig, SimulationConfig, SlamConfig
from mapping.slam_map import FREE, UNKNOWN, SlamSnapshot
from agents.drone_movement import DroneMovementController
from mapping.terrain_knowledge import TerrainKnowledge
from mission.pause_control import PauseCoordinator


class ImmediateEvent:
    def wait(self, timeout: float) -> bool:
        return False


class MovementControl:
    delay = 1 / 15
    terrain_roughness = np.full((64, 64), 0.4, dtype=np.float32)

    def __init__(self) -> None:
        self.mission_event = ImmediateEvent()
        self.paths = {}
        self.terrain_knowledge = TerrainKnowledge(
            np.zeros((64, 64), dtype=np.uint8)
        )
        self.terrain_fusion = SimpleNamespace(record_scan=lambda samples: None)

    def compute_path(self, start, goal):
        return list(self.paths.get((start, goal), []))

    def simulation_time(self) -> float:
        return time.perf_counter()

    def pause_checkpoint(self) -> bool:
        return True

    def wait_simulation_delay(self, duration: float) -> bool:
        self.mission_event.wait(duration)
        return True


class FixedDecisionPolicy:
    def __init__(self, decision: ExplorationDecision) -> None:
        self.decision = decision
        self.contexts = []

    def decide(self, context, is_segment_valid):
        self.contexts.append(context)
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
