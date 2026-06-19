import os
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from Drone import Drone
from SlamMap import FREE
from agents.drone_movement import DroneMovementController
from mapping.terrain_knowledge import TerrainKnowledge


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

    def compute_path(self, start, goal):
        return list(self.paths.get((start, goal), []))

    def record_terrain_scan(self, samples) -> None:
        pass


class DroneMovementTests(unittest.TestCase):
    def setUp(self) -> None:
        settings = SimpleNamespace(
            map_dim="LARGE",
            slam_scan_interval=0.0,
            slam_scan_rays=5,
            slam_point_cloud_max_points=50,
            frontier_rebuild_cooldown=0.25,
            frontier_stride=4,
            frontier_confidence_threshold=0.6,
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
        self.drone.pos = (20, 20)
        self.control.paths[((20, 20), (16, 16))] = [
            (20, 20),
            (18, 18),
            (16, 16),
        ]

        reached_home = self.drone.reach_start_point()

        self.assertTrue(reached_home)
        self.assertEqual(self.drone.pos, (16, 16))
        self.assertEqual(self.drone.graph.pos[-1], (16, 16))
        self.assertNotEqual(self.drone.heading_deg, 0.0)

    def test_frontier_rebuild_uses_local_slam_state(self) -> None:
        self.drone.slam_map.occupancy.fill(-1)
        self.drone.slam_map.confidence.fill(0.0)
        self.drone.slam_map.occupancy[20, 20] = FREE
        self.drone.slam_map.confidence[20, 20] = 1.0

        self.drone._rebuild_frontiers(
            stride=1,
            confidence_threshold=0.6,
        )

        self.assertEqual(self.drone.border, [(20, 20)])

    def test_frontier_rebuild_ignores_mission_terrain_telemetry(self) -> None:
        self.drone.slam_map.occupancy.fill(-1)
        self.drone.slam_map.confidence.fill(0.0)
        self.drone.slam_map.occupancy[20, 20] = FREE
        self.control.terrain_knowledge.confidence[20, 20] = 1.0

        self.drone._rebuild_frontiers(
            stride=1,
            confidence_threshold=0.6,
        )

        self.assertEqual(self.drone.border, [])

        self.drone.terrain_knowledge.confidence[20, 20] = 1.0
        self.drone._rebuild_frontiers(
            stride=1,
            confidence_threshold=0.6,
        )

        self.assertEqual(self.drone.border, [(20, 20)])

    def test_movement_configuration_is_controller_owned(self) -> None:
        self.assertIsInstance(
            self.drone.movement_controller,
            DroneMovementController,
        )
        self.assertEqual(self.drone.frontier_stride, 4)

        self.drone.frontier_stride = 2
        self.drone.border_retry_cooldown = 3.0

        self.assertEqual(self.drone.movement_controller.frontier_stride, 2)
        self.assertEqual(
            self.drone.movement_controller.border_retry_cooldown,
            3.0,
        )


if __name__ == "__main__":
    unittest.main()
