import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pygame

from ui.menu.models import ButtonItem, KeyHint, TitleItem
from ui.menu.renderer import KEY_HINT_GAP, KEY_HINT_HEIGHT, KEY_HINT_MARGIN, MenuRenderer


class FakeSurface:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(0, 0, self.width, self.height)


class MenuRendererTests(unittest.TestCase):
    def test_draw_uses_renderer_owned_background_and_selection(self) -> None:
        renderer = object.__new__(MenuRenderer)
        renderer.game = SimpleNamespace(display=Mock())
        renderer.background = Mock()
        renderer._draw_item = Mock()
        items = [
            TitleItem("Title", (0, 0)),
            ButtonItem("Button", (0, 0)),
        ]

        renderer.draw(items, 1)

        renderer.game.display.blit.assert_called_once_with(
            renderer.background,
            (0, 0),
        )
        renderer._draw_item.assert_any_call(items[0], False)
        renderer._draw_item.assert_any_call(items[1], True)

    def test_draw_renders_key_hints_after_menu_items(self) -> None:
        renderer = object.__new__(MenuRenderer)
        renderer.game = SimpleNamespace(display=Mock())
        renderer.background = Mock()
        renderer._draw_item = Mock()
        renderer._draw_key_hints = Mock()
        items = [ButtonItem("Button", (0, 0))]
        hints = (KeyHint.MOVE, KeyHint.ENTER)

        renderer.draw(items, 0, hints)

        renderer._draw_key_hints.assert_called_once_with(hints)

    def test_key_hints_are_blitted_as_bottom_right_strip(self) -> None:
        renderer = object.__new__(MenuRenderer)
        display = Mock()
        display.get_rect.return_value = pygame.Rect(0, 0, 500, 300)
        renderer.game = SimpleNamespace(display=display)
        move_surface = FakeSurface(100, 40)
        enter_surface = FakeSurface(50, 20)
        renderer.key_hint_images = {
            KeyHint.MOVE: move_surface,
            KeyHint.ENTER: enter_surface,
        }

        renderer._draw_key_hints((KeyHint.MOVE, KeyHint.ENTER))

        total_width = 100 + 50 + KEY_HINT_GAP
        expected_x = 500 - total_width - KEY_HINT_MARGIN
        expected_center_y = 300 - KEY_HINT_MARGIN - (40 / 2)
        first_rect = display.blit.call_args_list[0].args[1]
        second_rect = display.blit.call_args_list[1].args[1]

        self.assertEqual(display.blit.call_count, 2)
        self.assertEqual(first_rect.left, expected_x)
        self.assertEqual(first_rect.centery, expected_center_y)
        self.assertEqual(second_rect.left, expected_x + 100 + KEY_HINT_GAP)
        self.assertEqual(second_rect.centery, expected_center_y)

    def test_non_arrow_hints_match_arrow_visible_artwork_height(self) -> None:
        renderer = object.__new__(MenuRenderer)
        move_surface = self._artwork_surface(
            height=10,
            artwork_start=3,
            artwork_height=4,
        )
        enter_surface = self._artwork_surface(
            height=10,
            artwork_start=0,
            artwork_height=8,
        )

        scaled = renderer._scale_key_hint_images(
            {
                KeyHint.MOVE: move_surface,
                KeyHint.ENTER: enter_surface,
            }
        )

        self.assertEqual(scaled[KeyHint.MOVE].get_height(), KEY_HINT_HEIGHT)
        self.assertEqual(scaled[KeyHint.ENTER].get_height(), 36)

    @staticmethod
    def _artwork_surface(
        height: int,
        artwork_start: int,
        artwork_height: int,
    ) -> pygame.Surface:
        surface = pygame.Surface((10, height), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 255))
        for y in range(artwork_start, artwork_start + artwork_height):
            for x in range(10):
                surface.set_at((x, y), (255, 255, 255, 255))
        return surface


if __name__ == "__main__":
    unittest.main()
