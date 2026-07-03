"""Navigation and input state for the Cave Explorer menu."""

from __future__ import annotations

from typing import Callable, Dict, List

import pygame

from ui.menu.models import (
    ButtonItem,
    MenuAction,
    MenuRow,
    MenuScreen,
    SelectorItem,
    SliderItem,
    TextInputItem,
)

NUMPAD_DIGIT_KEYS = (
    pygame.K_KP0,
    pygame.K_KP1,
    pygame.K_KP2,
    pygame.K_KP3,
    pygame.K_KP4,
    pygame.K_KP5,
    pygame.K_KP6,
    pygame.K_KP7,
    pygame.K_KP8,
    pygame.K_KP9,
)


class MenuController:
    """Own menu selection, navigation, and item mutations."""

    def __init__(
        self,
        screens: Dict[MenuScreen, List[MenuRow]],
        action_handler: Callable[[MenuAction], None],
        play_button: Callable[[], None],
        audio_changed: Callable[[], None],
    ) -> None:
        self.screens = screens
        self._action_handler = action_handler
        self._play_button = play_button
        self._audio_changed = audio_changed
        self.current_screen = MenuScreen.MAIN
        self.current_index = self.first_selectable()

    @property
    def current_items(self) -> List[MenuRow]:
        return self.screens[self.current_screen]

    def first_selectable(self) -> int:
        for index, item in enumerate(self.current_items):
            if item.selectable:
                return index
        return 0

    def next_selectable(self, direction: str) -> int:
        step = 1 if direction == "down" else -1
        index = self.current_index
        for _ in range(len(self.current_items)):
            index = (index + step) % len(self.current_items)
            if self.current_items[index].selectable:
                return index
        return self.current_index

    def open_screen(
        self,
        screen: MenuScreen,
        *,
        select_last: bool = False,
    ) -> None:
        self.current_screen = screen
        self.current_index = (
            len(self.current_items) - 1 if select_last else self.first_selectable()
        )

    def handle_input(self, game: object) -> None:
        if game.UP_KEY:
            self.current_index = self.next_selectable("up")
            return
        if game.DOWN_KEY:
            self.current_index = self.next_selectable("down")
            return

        item = self.current_items[self.current_index]
        if game.START_KEY:
            if not isinstance(item, ButtonItem):
                return
            self._play_button()
            self._action_handler(item.action)
            return

        if self._handle_item_input(item, game) and self.current_screen is MenuScreen.AUDIO:
            self._audio_changed()

    def _handle_item_input(self, item: MenuRow, game: object) -> bool:
        if isinstance(item, SelectorItem):
            if game.LEFT_KEY and item.value > 0:
                item.value -= 1
                try:
                    self._play_button()
                except (AttributeError, pygame.error):
                    pass
                return True
            if game.RIGHT_KEY and item.value < len(item.options) - 1:
                item.value += 1
                try:
                    self._play_button()
                except (AttributeError, pygame.error):
                    pass
                return True
            return False

        if isinstance(item, SliderItem):
            if game.LEFT_KEY:
                item.value = max(item.minimum, item.value - item.step)
                self._play_button()
                return True
            if game.RIGHT_KEY:
                item.value = min(item.maximum, item.value + item.step)
                self._play_button()
                return True
            return False

        if isinstance(item, TextInputItem):
            return self._handle_text_input(item)

        return False

    @staticmethod
    def _handle_text_input(item: TextInputItem) -> bool:
        keys = pygame.key.get_pressed()
        modified = False
        for key in range(pygame.K_0, pygame.K_9 + 1):
            if keys[key]:
                item.text += chr(key)
                modified = True
        for digit, key in enumerate(NUMPAD_DIGIT_KEYS):
            if keys[key]:
                item.text += str(digit)
                modified = True
        if keys[pygame.K_BACKSPACE] and item.text:
            item.text = item.text[:-1]
            modified = True
        return modified

    @staticmethod
    def number_key_pressed() -> bool:
        """Return whether seed-entry keys are still held down."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_BACKSPACE] or any(
            keys[key] for key in range(pygame.K_0, pygame.K_9 + 1)
        ):
            return True
        return any(keys[key] for key in NUMPAD_DIGIT_KEYS)
