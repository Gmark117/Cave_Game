import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from Rover import Rover


class RoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control = SimpleNamespace(
            delay=1 / 15,
            acquire_rover_target=Mock(return_value=(2, 0)),
            compute_rover_path=Mock(
                return_value=[(0, 0), (1, 0), (2, 0)]
            ),
            release_rover_target=Mock(),
        )
        game = SimpleNamespace(
            sim_settings=SimpleNamespace(map_dim="SMALL"),
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
        self.control.release_rover_target.assert_called_once_with(
            0,
            completed=True,
        )

    def test_unreachable_target_is_released_for_retry(self) -> None:
        self.control.compute_rover_path.return_value = []

        self.rover.move()

        self.control.release_rover_target.assert_called_once_with(
            0,
            completed=False,
        )
        self.assertEqual(self.rover.status, "Ready")
        self.assertEqual(self.rover.current_path, [])

    def test_rover_owns_local_terrain_knowledge(self) -> None:
        self.assertEqual(self.rover.known_roughness.shape, (4, 4))
        self.assertTrue(np.all(self.rover.known_roughness == -1.0))
        self.assertTrue(np.all(self.rover.terrain_confidence == 0.0))
        self.assertIs(
            self.rover.terrain_lock,
            self.rover.terrain_knowledge.lock,
        )


if __name__ == "__main__":
    unittest.main()
