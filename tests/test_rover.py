import os
import unittest
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

    def test_rover_plans_advances_and_completes_target(self) -> None:
        self.rover.move()
        self.assertEqual(self.rover.target, (2, 0))
        self.assertEqual(self.rover.current_path, [(1, 0), (2, 0)])

        self.rover.move()
        self.assertEqual(self.rover.pos, (1, 0))
        self.rover.move()

        self.assertEqual(self.rover.pos, (2, 0))
        self.assertEqual(self.rover.status, "Done")
        self.assertIsNone(self.rover.target)
        self.rover_targets.release.assert_called_once_with(
            0,
            completed=True,
        )

    def test_unreachable_target_is_released_for_retry(self) -> None:
        self.control.compute_rover_path.return_value = []

        self.rover.move()

        self.rover_targets.release.assert_called_once_with(
            0,
            completed=False,
        )
        self.assertEqual(self.rover.status, "Ready")
        self.assertEqual(self.rover.current_path, [])

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


if __name__ == "__main__":
    unittest.main()
