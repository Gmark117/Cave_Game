import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pygame

from MenuController import MenuController
from MenuModels import (
    ButtonItem,
    MenuAction,
    MenuScreen,
    SelectorItem,
    SliderItem,
    TextInputItem,
    TitleItem,
)


def game_flags(**overrides):
    values = {
        "UP_KEY": False,
        "DOWN_KEY": False,
        "START_KEY": False,
        "LEFT_KEY": False,
        "RIGHT_KEY": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PressedKeys:
    def __init__(self, *pressed):
        self.pressed = set(pressed)

    def __getitem__(self, key):
        return key in self.pressed


class MenuControllerTests(unittest.TestCase):
    def make_controller(self):
        self.action_handler = Mock()
        self.play_button = Mock()
        self.audio_changed = Mock()
        self.selector = SelectorItem(
            "Selector",
            (0, 0),
            options=("one", "two"),
        )
        self.slider = SliderItem(
            "Volume",
            (0, 0),
            value=0,
            minimum=0,
            maximum=100,
            step=20,
        )
        self.seed = TextInputItem("Seed", (0, 0))
        screens = {
            MenuScreen.MAIN: [
                TitleItem("Title", (0, 0)),
                ButtonItem(
                    "Open",
                    (0, 0),
                    action=MenuAction.OPEN_SIMULATION,
                ),
            ],
            MenuScreen.SIMULATION: [
                TitleItem("Title", (0, 0)),
                self.selector,
                self.seed,
            ],
            MenuScreen.AUDIO: [
                TitleItem("Title", (0, 0)),
                self.slider,
            ],
            MenuScreen.CREDITS: [
                TitleItem("Title", (0, 0)),
                ButtonItem(
                    "Back",
                    (0, 0),
                    action=MenuAction.BACK_TO_MAIN,
                ),
            ],
        }
        return MenuController(
            screens,
            self.action_handler,
            self.play_button,
            self.audio_changed,
        )

    def test_navigation_wraps_and_skips_non_selectable_rows(self) -> None:
        controller = self.make_controller()

        self.assertEqual(controller.current_index, 1)
        self.assertEqual(controller.next_selectable("down"), 1)
        self.assertEqual(controller.next_selectable("up"), 1)

    def test_button_dispatches_named_action(self) -> None:
        controller = self.make_controller()

        controller.handle_input(game_flags(START_KEY=True))

        self.play_button.assert_called_once_with()
        self.action_handler.assert_called_once_with(MenuAction.OPEN_SIMULATION)

    def test_selector_changes_without_wrapping(self) -> None:
        controller = self.make_controller()
        controller.open_screen(MenuScreen.SIMULATION)

        controller.handle_input(game_flags(RIGHT_KEY=True))
        controller.handle_input(game_flags(RIGHT_KEY=True))

        self.assertEqual(self.selector.value, 1)
        self.assertEqual(self.play_button.call_count, 1)

    def test_slider_clamps_and_notifies_audio(self) -> None:
        controller = self.make_controller()
        controller.open_screen(MenuScreen.AUDIO)

        controller.handle_input(game_flags(LEFT_KEY=True))
        controller.handle_input(game_flags(RIGHT_KEY=True))

        self.assertEqual(self.slider.value, 20)
        self.assertEqual(self.audio_changed.call_count, 2)

    def test_numpad_enters_seed_and_is_detected_while_held(self) -> None:
        controller = self.make_controller()
        controller.open_screen(MenuScreen.SIMULATION)
        controller.current_index = 2

        with patch(
            "MenuController.pygame.key.get_pressed",
            return_value=PressedKeys(pygame.K_KP1),
        ):
            controller.handle_input(game_flags())
            held = controller.number_key_pressed()

        self.assertEqual(self.seed.text, "1")
        self.assertTrue(held)


if __name__ == "__main__":
    unittest.main()
