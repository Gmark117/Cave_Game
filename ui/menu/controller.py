"""Navigation and input state for the Cave Explorer menu."""

from __future__ import annotations

from typing import Callable, Dict, List

import pygame

from ui.menu.models import (
    ButtonItem,
    KeyHint,
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
        """Store menu data and callbacks without depending on rendering."""
        self.screens = screens
        self._action_handler = action_handler
        self._play_button = play_button
        self._audio_changed = audio_changed
        self.current_screen = MenuScreen.MAIN
        self.current_index = self.first_selectable()

    @property
    def current_items(self) -> List[MenuRow]:
        """Return the rows for the currently open screen."""
        return self.screens[self.current_screen]

    def first_selectable(self) -> int:
        """Return the first selectable row index for the active screen."""
        for index, item in enumerate(self.current_items):
            if item.selectable:
                return index
        return 0

    def next_selectable(self, direction: str) -> int:
        """Move selection up/down, wrapping around non-selectable rows."""
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
        """Switch screens and choose the initial selected row."""
        self.current_screen = screen
        self.current_index = (
            len(self.current_items) - 1 if select_last else self.first_selectable()
        )

    def key_hints(self) -> tuple[KeyHint, ...]:
        """Return the key images that apply to the current menu state."""
        item = self.current_items[self.current_index]
        hints = []
        if self._arrow_keys_available(item):
            hints.append(KeyHint.MOVE)
        if isinstance(item, TextInputItem):
            hints.append(KeyHint.NUMBERS)
        if isinstance(item, ButtonItem) and item.selectable:
            hints.append(KeyHint.ENTER)
        if self._backspace_available(item):
            hints.append(KeyHint.BACKSPACE)
        return tuple(hints)

    def _arrow_keys_available(self, item: MenuRow) -> bool:
        """Return whether any arrow key can change the current menu state."""
        selectable_count = sum(1 for row in self.current_items if row.selectable)
        if selectable_count > 1:
            return True
        return self._left_right_available(item)

    @staticmethod
    def _left_right_available(item: MenuRow) -> bool:
        """Return whether left/right can mutate the selected row."""
        if isinstance(item, SelectorItem):
            return len(item.options) > 1 and (
                item.value > 0 or item.value < len(item.options) - 1
            )
        if isinstance(item, SliderItem):
            return item.value > item.minimum or item.value < item.maximum
        return False

    def _backspace_available(self, item: MenuRow) -> bool:
        """Return whether backspace can delete text or use a screen back action."""
        if isinstance(item, TextInputItem):
            return bool(item.text)
        return self._screen_back_action() is not None

    def _screen_back_action(self) -> MenuAction | None:
        """Return the Backspace action for the active screen, if any."""
        if self.current_screen is MenuScreen.SIMULATION:
            return MenuAction.BACK_TO_MAIN
        if self.current_screen is MenuScreen.AUDIO:
            return MenuAction.SAVE_AUDIO_AND_BACK
        if self.current_screen is MenuScreen.CREDITS:
            return MenuAction.BACK_TO_MAIN
        return None

    def handle_input(self, game: object) -> None:
        """Apply one frame of keyboard flags to the current menu state."""
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
        if game.BACK_KEY:
            back_action = (
                None if isinstance(item, TextInputItem) else self._screen_back_action()
            )
            if back_action is not None:
                self._play_button()
                self._action_handler(back_action)
                return

        if self._handle_item_input(item, game) and self.current_screen is MenuScreen.AUDIO:
            self._audio_changed()

    def _handle_item_input(self, item: MenuRow, game: object) -> bool:
        """Mutate selectors, sliders, or text inputs from left/right/keys."""
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
        """Append digits or remove one character from a text-input row."""
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
