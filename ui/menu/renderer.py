"""Pygame resources and drawing for the Cave Explorer menu."""

from __future__ import annotations

from typing import Sequence, Tuple

import pygame

from ui.menu.models import (
    ButtonItem,
    KeyHint,
    MenuRow,
    SelectorItem,
    SliderItem,
    TextInputItem,
    TitleItem,
)
from asset_config.gameplay import Display
from asset_config.media import Images
from asset_config.rendering import Colors, Fonts


KEY_HINT_HEIGHT = 72
KEY_HINT_GAP = 12
KEY_HINT_MARGIN = 24
KEY_HINT_ARTWORK_THRESHOLD = 80
KEY_HINT_ARTWORK_ROW_MIN_PIXELS = 8


class MenuRenderer:
    """Own backgrounds, fonts, layout, and menu drawing."""

    def __init__(self, game: object) -> None:
        """Load background images used by menu screens."""
        self.game = game
        self.background = pygame.image.load(Images.CAVE.value)
        self.dark_background = pygame.image.load(Images.DARK_CAVE.value)
        self.key_hint_images = self._load_key_hint_images()

    def draw(
        self,
        items: Sequence[MenuRow],
        selected_index: int,
        key_hints: Sequence[KeyHint] = (),
    ) -> None:
        """Draw all rows in one menu screen."""
        self.game.display.blit(self.background, (0, 0))
        for index, item in enumerate(items):
            self._draw_item(item, index == selected_index)
        self._draw_key_hints(key_hints)

    def _load_key_hint_images(self) -> dict[KeyHint, pygame.Surface]:
        """Load key hint images and match their visible artwork scale."""
        images = {
            KeyHint.MOVE: Images.KEY_HINTS,
            KeyHint.NUMBERS: Images.NUMBERS_HINT,
            KeyHint.ENTER: Images.ENTER_HINT,
            KeyHint.BACKSPACE: Images.BACKSPACE_HINT,
        }
        surfaces = {
            hint: pygame.image.load(image.value)
            for hint, image in images.items()
        }
        return self._scale_key_hint_images(surfaces)

    def _scale_key_hint_images(
        self,
        surfaces: dict[KeyHint, pygame.Surface],
    ) -> dict[KeyHint, pygame.Surface]:
        """Scale non-arrow hints to the arrow icon's visible artwork height."""
        move_surface = surfaces[KeyHint.MOVE]
        move_artwork_height = self._visible_artwork_height(move_surface)
        move_canvas_height = max(1, move_surface.get_height())
        target_artwork_height = (
            KEY_HINT_HEIGHT * move_artwork_height / move_canvas_height
        )

        scaled = {}
        for hint, surface in surfaces.items():
            if hint is KeyHint.MOVE:
                target_height = KEY_HINT_HEIGHT
            else:
                artwork_height = max(1, self._visible_artwork_height(surface))
                target_height = round(
                    target_artwork_height * surface.get_height() / artwork_height
                )
            scaled[hint] = self._scale_to_height(
                surface,
                max(1, target_height),
            )
        return scaled

    @staticmethod
    def _visible_artwork_height(surface: pygame.Surface) -> int:
        """Measure bright artwork rows while ignoring isolated noise pixels."""
        width, height = surface.get_size()
        active_rows = []
        for y in range(height):
            active_pixels = 0
            for x in range(width):
                red, green, blue, alpha = surface.get_at((x, y))
                if (
                    alpha > 0
                    and max(red, green, blue) >= KEY_HINT_ARTWORK_THRESHOLD
                ):
                    active_pixels += 1
            if active_pixels >= KEY_HINT_ARTWORK_ROW_MIN_PIXELS:
                active_rows.append(y)

        if not active_rows:
            return height
        return active_rows[-1] - active_rows[0] + 1

    @staticmethod
    def _scale_to_height(surface: pygame.Surface, height: int) -> pygame.Surface:
        """Scale a surface proportionally to a fixed height."""
        width, original_height = surface.get_size()
        if original_height == height:
            return surface
        scale = height / original_height
        scaled_size = (max(1, round(width * scale)), height)
        return pygame.transform.smoothscale(surface, scaled_size)

    def _draw_key_hints(self, key_hints: Sequence[KeyHint]) -> None:
        """Draw the active keyboard hints as a bottom-right strip."""
        if not key_hints:
            return

        surfaces = [
            self.key_hint_images[hint]
            for hint in key_hints
            if hint in self.key_hint_images
        ]
        if not surfaces:
            return

        display_rect = self.game.display.get_rect()
        total_width = sum(surface.get_width() for surface in surfaces)
        total_width += KEY_HINT_GAP * (len(surfaces) - 1)
        x = display_rect.right - total_width - KEY_HINT_MARGIN
        row_height = max(surface.get_height() for surface in surfaces)
        center_y = display_rect.bottom - KEY_HINT_MARGIN - row_height / 2

        for surface in surfaces:
            rect = surface.get_rect()
            rect.left = x
            rect.centery = round(center_y)
            self.game.display.blit(surface, rect)
            x += rect.width + KEY_HINT_GAP

    def draw_loading(self, text: Sequence[str]) -> None:
        """Draw centered loading text over the dark cave background."""
        font_size = 100
        line_offset = 100
        self.game.display.blit(self.dark_background, (0, 0))
        font = pygame.font.Font(Fonts.BIG.value, font_size)
        first_line_y = Display.CENTER_H - line_offset * (len(text) - 1) / 2
        for index, line_text in enumerate(text):
            text_surface = font.render(line_text, True, Colors.WHITE.value)
            rect = text_surface.get_rect()
            rect.center = (
                Display.CENTER_W,
                first_line_y + line_offset * index,
            )
            self.game.display.blit(text_surface, rect)
        self.game.blit_screen()

    def _draw_item(self, item: MenuRow, selected: bool) -> None:
        """Draw one menu row and its optional value control."""
        x, y = item.position
        color = (
            Colors.EUCALYPTUS.value
            if isinstance(item, TitleItem)
            else Colors.RED.value
            if selected
            else Colors.WHITE.value
        )
        value_offset = 350
        self._draw_text(
            item.label,
            x,
            y,
            item.size,
            Fonts.BIG.value if item.font_big else Fonts.SMALL.value,
            color,
            item.alignment,
        )

        if isinstance(item, SelectorItem) and item.options:
            self._draw_selector(item, value_offset)
        elif isinstance(item, TextInputItem):
            if not item.text:
                display_text = (
                    "Enter value" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
                )
                value_color = Colors.RED.value
            else:
                display_text = item.text
                value_color = Colors.GREENDARK.value
            self._draw_text(
                display_text,
                x + value_offset,
                y,
                item.size,
                Fonts.SMALL.value,
                value_color,
                item.alignment,
            )
        elif isinstance(item, SliderItem):
            self._draw_slider(item, value_offset)
        elif isinstance(item, (TitleItem, ButtonItem)):
            return

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        size: int,
        font: str,
        color: Tuple[int, int, int],
        align: str,
    ) -> None:
        """Draw text using a Pygame rect anchor."""
        style = pygame.font.Font(font, size)
        text_surface = style.render(text, True, color)
        rect = text_surface.get_rect()
        if align == "midright":
            rect.midright = (x, y)
        elif align == "midleft":
            rect.midleft = (x, y)
        elif align == "center":
            rect.center = (x, y)
        self.game.display.blit(text_surface, rect)

    def _draw_arrow(
        self,
        center_x: int,
        center_y: int,
        size: int,
        direction: str,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a simple triangular selector arrow."""
        center_x = int(center_x)
        center_y = int(center_y)
        half_width = max(3, int(size // 2))
        half_height = max(3, int(size // 2))
        if direction == "left":
            points = [
                (center_x + half_width, center_y - half_height),
                (center_x + half_width, center_y + half_height),
                (center_x - half_width, center_y),
            ]
        else:
            points = [
                (center_x - half_width, center_y - half_height),
                (center_x - half_width, center_y + half_height),
                (center_x + half_width, center_y),
            ]
        pygame.draw.polygon(self.game.display, color, points)

    def _draw_selector(self, item: SelectorItem[object], value_offset: int) -> None:
        """Draw a selector value with arrows for available directions."""
        x, y = item.position
        value_text = str(item.options[item.value])
        value_x = x + value_offset
        arrow_size = max(10, int(item.size * 0.4))
        font = pygame.font.Font(str(Fonts.SMALL.value), item.size)
        text_surface = font.render(value_text, True, Colors.GREENDARK.value)
        text_rect = text_surface.get_rect()
        if item.alignment == "midright":
            text_rect.midright = (value_x, y)
        elif item.alignment == "midleft":
            text_rect.midleft = (value_x, y)
        else:
            text_rect.center = (value_x, y)
        self.game.display.blit(text_surface, text_rect)

        left_offset = max(18, item.size)
        left_x = text_rect.left - left_offset
        right_x = text_rect.right + left_offset - 5
        if len(item.options) > 1:
            if item.value > 0:
                self._draw_arrow(
                    left_x,
                    text_rect.centery,
                    arrow_size,
                    "left",
                    Colors.GREY.value,
                )
            if item.value < len(item.options) - 1:
                self._draw_arrow(
                    right_x,
                    text_rect.centery,
                    arrow_size,
                    "right",
                    Colors.GREY.value,
                )

    def _draw_slider(self, item: SliderItem, value_offset: int) -> None:
        """Draw the volume-style slider as five filled bars."""
        x, y = item.position
        slider_x = x + value_offset
        max_width = 200
        bar_count = 5
        bar_width = max_width / bar_count
        try:
            filled_bars = round(
                (item.value - item.minimum)
                / ((item.maximum - item.minimum) / bar_count)
            )
        except (TypeError, ValueError, ZeroDivisionError):
            filled_bars = 0

        for index in range(bar_count):
            color = (
                Colors.GREEN.value
                if index < filled_bars
                else Colors.WHITE.value
            )
            pygame.draw.rect(
                self.game.display,
                color,
                (slider_x + index * bar_width, y - 8, bar_width - 2, 20),
            )

        arrow_size = max(10, int(item.size * 0.4))
        left_offset = max(18, item.size)
        center_y = int((y - 8) + (20 / 2))
        if item.value > item.minimum:
            self._draw_arrow(
                slider_x - left_offset,
                center_y,
                arrow_size,
                "left",
                Colors.GREY.value,
            )
        if item.value < item.maximum:
            self._draw_arrow(
                slider_x + max_width + left_offset - 5,
                center_y,
                arrow_size,
                "right",
                Colors.GREY.value,
            )
