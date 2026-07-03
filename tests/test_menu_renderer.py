import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ui.menu.models import ButtonItem, TitleItem
from ui.menu.renderer import MenuRenderer


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


if __name__ == "__main__":
    unittest.main()
