import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from agents.rover import Rover
from config.simulation_config import MissionConfig, SimulationConfig


class RoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rover_targets = SimpleNamespace(
            acquire=Mock(return_value=(2, 0)),
            release=Mock(),
            reject_failed_route=Mock(),
            should_hold=Mock(return_value=False),
            target_is_current=Mock(return_value=True),
        )
        self.control = SimpleNamespace(
            delay=1 / 15,
            rover_targets=self.rover_targets,
            compute_rover_path=Mock(
                return_value=[(0, 0), (1, 0), (2, 0)]
            ),
        )
        game = SimpleNamespace(
            sim_settings=SimulationConfig(
                mission_config=MissionConfig(map_dim="SMALL")
            ),
            window=pygame.Surface((16, 16), pygame.SRCALPHA),
            width=16,
            height=16,
        )
        self.rover = Rover(
            game,
            self.control,
            0,
            (0, 0),
            (255, 0, 0),
            pygame.Surface((2, 2), pygame.SRCALPHA),
            np.zeros((4, 4), dtype=np.uint8),
        )

    def test_rover_plans_advances_and_stages_at_frontier(self) -> None:
        self.rover.move()
        self.assertEqual(self.rover.target, (2, 0))
        self.assertEqual(self.rover.status, "Announcing")
        self.rover.move()
        self.assertEqual(self.rover.current_path, [(1, 0), (2, 0)])

        self.rover.move()
        self.assertEqual(self.rover.pos, (1, 0))
        self.rover.move()

        self.assertEqual(self.rover.pos, (2, 0))
        self.assertEqual(self.rover.status, "Staging")
        self.assertEqual(self.rover.target, (2, 0))
        self.rover_targets.release.assert_not_called()
        self.control.compute_rover_path.assert_called_once_with(
            0, (0, 0), (2, 0)
        )

    def test_unreachable_target_is_released_for_retry(self) -> None:
        self.control.compute_rover_path.return_value = []

        self.rover.move()

        self.rover_targets.release.assert_called_once_with(
            0,
            completed=False,
        )
        self.rover_targets.reject_failed_route.assert_called_once_with(
            0, (0, 0), (2, 0), sim_time=0.0,
        )
        self.assertEqual(self.rover.status, "Ready")
        self.assertEqual(self.rover.current_path, [])

    def test_rover_does_not_move_until_endpoint_is_acknowledged(self) -> None:
        self.rover.navigation = replace(
            self.rover.navigation,
            rendezvous_departure_ready=lambda _position: False,
        )

        self.rover.move()
        self.rover.move()

        self.assertEqual(self.rover.status, "Announcing")
        self.assertEqual(self.rover.pos, (0, 0))
        self.assertEqual(self.rover.current_path, [])
        self.control.compute_rover_path.assert_called_once_with(
            0, (0, 0), (2, 0)
        )

    def test_rover_holds_position_for_pending_rendezvous(self) -> None:
        self.rover_targets.should_hold.return_value = True

        self.rover.move()

        self.assertEqual(self.rover.pos, (0, 0))
        self.assertEqual(self.rover.status, "Rendezvous")
        self.rover_targets.acquire.assert_not_called()

    def test_report_stop_pauses_an_authorized_route_without_losing_it(
        self,
    ) -> None:
        self.rover.current_path = [(1, 0), (2, 0)]
        self.rover.target = (2, 0)
        self.rover_targets.should_hold.return_value = True

        self.rover.move()

        self.assertEqual(self.rover.pos, (0, 0))
        self.assertEqual(self.rover.current_path, [(1, 0), (2, 0)])
        self.assertEqual(self.rover.status, "Rendezvous")
        self.rover_targets.should_hold.return_value = False
        self.rover.move()
        self.assertEqual(self.rover.pos, (1, 0))
        self.assertEqual(self.rover.current_path, [(2, 0)])

    def test_rover_owns_local_terrain_knowledge(self) -> None:
        knowledge = self.rover.terrain_knowledge
        self.assertEqual(knowledge.roughness.shape, (4, 4))
        self.assertTrue(np.all(knowledge.roughness == -1.0))
        self.assertTrue(np.all(knowledge.confidence == 0.0))
        self.assertFalse(
            any(
                hasattr(self.rover, name)
                for name in (
                    "known_roughness",
                    "terrain_confidence",
                    "terrain_lock",
                )
            )
        )

    def test_rover_owns_team_slam_checkpoint(self) -> None:
        snapshot = self.rover.slam_map.snapshot()

        self.assertEqual(snapshot.occupancy.shape, (4, 4))
        self.assertTrue(np.all(snapshot.occupancy == -1))
        self.assertTrue(np.all(snapshot.confidence == 0.0))

    def test_navigation_snapshot_is_detached_for_control_center(self) -> None:
        self.rover.target = (2, 0)
        self.rover.current_path = [(1, 0), (2, 0)]
        snapshot = self.rover.snapshot()

        self.rover.current_path.pop(0)
        self.rover.pos = (1, 0)

        self.assertEqual(snapshot.position, (0, 0))
        self.assertEqual(snapshot.target, (2, 0))
        self.assertEqual(snapshot.path_remaining, 2)


if __name__ == "__main__":
    unittest.main()
