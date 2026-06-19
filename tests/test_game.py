import os
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from Game import Game


class GameTests(unittest.TestCase):
    def test_check_events_maps_navigation_keys_to_flags(self) -> None:
        game = object.__new__(Game)
        game.UP_KEY = False
        game.DOWN_KEY = False
        game.START_KEY = False
        game.BACK_KEY = False
        game.LEFT_KEY = False
        game.RIGHT_KEY = False
        events = [
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT),
        ]

        with patch("Game.pygame.event.get", return_value=events):
            game.check_events()

        self.assertTrue(game.START_KEY)
        self.assertTrue(game.UP_KEY)
        self.assertTrue(game.LEFT_KEY)
        self.assertFalse(game.DOWN_KEY)

    def test_reset_keys_clears_all_navigation_flags(self) -> None:
        game = object.__new__(Game)
        for name in (
            "UP_KEY",
            "DOWN_KEY",
            "START_KEY",
            "BACK_KEY",
            "LEFT_KEY",
            "RIGHT_KEY",
        ):
            setattr(game, name, True)

        game.reset_keys()

        self.assertTrue(
            all(
                not getattr(game, name)
                for name in (
                    "UP_KEY",
                    "DOWN_KEY",
                    "START_KEY",
                    "BACK_KEY",
                    "LEFT_KEY",
                    "RIGHT_KEY",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
