import os
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from Drone import Drone
from SimulationConfig import MissionConfig, SimulationConfig, SlamConfig
from SlamMap import FREE, UNKNOWN, SlamSnapshot
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
