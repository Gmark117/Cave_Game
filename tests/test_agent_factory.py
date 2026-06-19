import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from AgentFactory import AgentFactory


class AgentFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control = SimpleNamespace(
            settings=SimpleNamespace(num_drones=5, map_dim="SMALL"),
            game=SimpleNamespace(),
            start_point=(3, 4),
            map_matrix=np.zeros((6, 7), dtype=np.uint8),
        )
        self.icon = pygame.Surface((8, 8), pygame.SRCALPHA)

    def test_build_drones_creates_configured_agents(self) -> None:
        created = []

        def make_drone(*args):
            drone = SimpleNamespace(args=args)
            created.append(drone)
            return drone

        with patch("AgentFactory.pygame.image.load", return_value=self.icon):
            with patch("AgentFactory.Drone", side_effect=make_drone) as drone_cls:
                AgentFactory.build_drones(self.control)

        self.assertEqual(self.control.num_drones, 5)
        self.assertEqual(len(self.control.drones), 5)
        self.assertEqual(drone_cls.call_count, 5)
        self.assertEqual(created[0].args[2], 0)
        self.assertEqual(created[0].args[3], (3, 4))
        self.assertIs(created[0].args[-1], self.control.map_matrix)
        self.assertEqual(self.control.drone_icon.get_size(), (25, 25))

    def test_build_rovers_delegates_complete_agent_construction(self) -> None:
        with patch("AgentFactory.pygame.image.load", return_value=self.icon):
            with patch(
                "AgentFactory.AgentFactory.choose_rover_color",
                return_value=(1, 2, 3),
            ):
                with patch(
                    "AgentFactory.Rover",
                    side_effect=lambda *args: SimpleNamespace(args=args),
                ) as rover_cls:
                    AgentFactory.build_rovers(self.control)

        self.assertEqual(self.control.num_rovers, 2)
        self.assertEqual(rover_cls.call_count, 2)
        self.assertEqual(self.control.rover_icon.get_size(), (40, 40))
        self.assertIs(
            self.control.rovers[0].args[-1],
            self.control.map_matrix,
        )

    def test_icon_dimensions_follow_map_size(self) -> None:
        self.assertEqual(AgentFactory.get_drone_icon_dim("LARGE"), (10, 10))
        self.assertEqual(AgentFactory.get_rover_icon_dim("MEDIUM"), (25, 25))


if __name__ == "__main__":
    unittest.main()
