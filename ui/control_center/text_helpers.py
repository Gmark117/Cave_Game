"""Text composition helpers for the control-center renderer."""

import time
from typing import Any, Iterable, Optional

import pygame

from asset_config.gameplay import Display
from asset_config.rendering import Colors


class ControlCenterTextMixin:
    """Render cached labels, wrapped text, and status text surfaces."""

    def _wrap_text_surfaces(
        self,
        text: str,
        font_obj: pygame.font.Font,
        max_w: int,
    ) -> list[pygame.Surface]:
        """Split text into rendered surfaces that fit within ``max_w``."""
        words = text.split()
        if not words:
            return [
                font_obj.render(
                    "",
                    True,
                    Colors.WHITE.value,
                ).convert_alpha()
            ]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            test = f"{current} {word}"
            if font_obj.render(
                test,
                True,
                Colors.WHITE.value,
            ).get_width() <= max_w:
                current = test
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return [
            font_obj.render(
                line,
                True,
                Colors.WHITE.value,
            ).convert_alpha()
            for line in lines
        ]

    def _draw_label_value_entry(
        self,
        prefix: str,
        index: int,
        label: str,
        value: str,
        font_obj: pygame.font.Font,
        max_w: int,
        ypos: int,
        line_gap: int,
        wrap_gap: int,
    ) -> tuple[int, int, bool]:
        """Draw a label/value pair, wrapping the value when it is too wide."""
        label_surf = font_obj.render(
            label + " ",
            True,
            Colors.GREY.value,
        ).convert_alpha()
        value_surf = font_obj.render(
            value,
            True,
            Colors.WHITE.value,
        ).convert_alpha()
        if label_surf.get_width() + value_surf.get_width() <= max_w:
            total_w = label_surf.get_width() + value_surf.get_width()
            height = max(
                label_surf.get_height(),
                value_surf.get_height(),
            )
            surf = pygame.Surface((total_w, height), pygame.SRCALPHA)
            surf.blit(
                label_surf,
                (0, (height - label_surf.get_height()) // 2),
            )
            surf.blit(
                value_surf,
                (
                    label_surf.get_width(),
                    (height - value_surf.get_height()) // 2,
                ),
            )
            key = f"{prefix}_{index}_{label}_{value}"
            self._dynamic_cache.setdefault(
                key,
                {"value": None, "time": 0, "surf": surf},
            )
            self._blit_cached_surface(
                surf,
                self.origin_x,
                ypos,
                "midleft",
            )
            return ypos + height + line_gap, index + 1, False

        self._blit_cached_surface(
            label_surf,
            self.origin_x,
            ypos,
            "midleft",
        )
        ypos += label_surf.get_height() + wrap_gap
        for surface in self._wrap_text_surfaces(
            value,
            font_obj,
            max_w,
        ):
            key = f"{prefix}_{index}_{label}_{value[:40]}"
            self._dynamic_cache.setdefault(
                key,
                {"value": None, "time": 0, "surf": surface},
            )
            self._blit_cached_surface(
                surface,
                self.origin_x,
                ypos,
                "midleft",
            )
            ypos += surface.get_height() + wrap_gap
            index += 1
            if ypos > Display.FULL_H - self.CONTENT_BOTTOM_MARGIN:
                return ypos, index, True
        return ypos, index, False

    def _draw_wrapped_text_lines(
        self,
        prefix: str,
        index: int,
        text: str,
        font_obj: pygame.font.Font,
        max_w: int,
        ypos: int,
        line_gap: int,
    ) -> tuple[int, int, bool]:
        """Draw preformatted text with wrapping and bottom-boundary checks."""
        for surface in self._wrap_text_surfaces(
            text,
            font_obj,
            max_w,
        ):
            key = f"{prefix}_{index}_{text[:40]}"
            self._dynamic_cache.setdefault(
                key,
                {"value": None, "time": 0, "surf": surface},
            )
            self._blit_cached_surface(
                surface,
                self.origin_x,
                ypos,
                "midleft",
            )
            ypos += surface.get_height() + line_gap
            index += 1
            if ypos > Display.FULL_H - self.CONTENT_BOTTOM_MARGIN:
                return ypos, index, True
        return ypos, index, False

    def _compose_text_surface(
        self,
        texts: Iterable[
            tuple[str, tuple[int, int, int], int]
        ],
        size: int,
        font_path: Any,
    ) -> pygame.Surface:
        """Combine differently colored text fragments into one surface."""
        font_obj = self._get_font(font_path, size)
        parts: list[pygame.Surface] = []
        total_w = 0
        max_h = 0
        for substring, color, alpha in texts:
            surface = font_obj.render(
                substring,
                True,
                color,
            ).convert_alpha()
            if alpha != 255:
                surface.set_alpha(alpha)
            parts.append(surface)
            total_w += surface.get_width()
            max_h = max(max_h, surface.get_height())
        composed = pygame.Surface((total_w, max_h), pygame.SRCALPHA)
        x = 0
        for surface in parts:
            composed.blit(
                surface,
                (x, (max_h - surface.get_height()) // 2),
            )
            x += surface.get_width()
        return composed

    def _get_cached_text_surface(
        self,
        key: str,
        texts: list[tuple[str, tuple[int, int, int], int]],
        size: int,
        font_path: Any,
        ttl: Optional[float] = None,
    ) -> pygame.Surface:
        """Return a cached composed text surface until value or TTL changes."""
        now = time.perf_counter()
        value = tuple(text[0] for text in texts)
        entry = self._dynamic_cache.get(key)
        if (
            entry
            and entry.get("value") == value
            and (
                ttl is None
                or now - entry.get("time", 0) < ttl
            )
        ):
            return entry["surf"]
        surface = self._compose_text_surface(
            texts,
            size,
            font_path,
        )
        self._dynamic_cache[key] = {
            "value": value,
            "time": now,
            "surf": surface,
        }
        return surface

    def _get_cached_status_surface(
        self,
        key: str,
        battery: int,
        status: str,
        battery_color: tuple[int, int, int],
        status_color: tuple[int, int, int],
        size: int,
        font_path: Any,
    ) -> pygame.Surface:
        """Return a cached battery/status surface for one agent row."""
        value = (str(battery), status)
        entry = self._dynamic_cache.get(key)
        if entry and entry.get("value") == value:
            return entry["surf"]
        font_obj = self._get_font(font_path, size)
        surface = self._compose_status_surface(
            font_obj,
            battery,
            status,
            battery_color,
            status_color,
        )
        self._dynamic_cache[key] = {
            "value": value,
            "time": time.perf_counter(),
            "surf": surface,
        }
        return surface

    def _compose_status_surface(
        self,
        font_obj: pygame.font.Font,
        battery: int,
        status: str,
        battery_color: tuple[int, int, int],
        status_color: tuple[int, int, int],
    ) -> pygame.Surface:
        """Compose battery and status text, wrapping long statuses if needed."""
        battery_surf = font_obj.render(
            f"{battery}%",
            True,
            battery_color,
        ).convert_alpha()
        battery_surf.set_alpha(128)
        column_width = font_obj.render(
            "00000%",
            True,
            battery_color,
        ).get_width()
        separator = font_obj.render(
            "|",
            True,
            Colors.WHITE.value,
        ).convert_alpha()
        separator.set_alpha(128)
        status_surf = font_obj.render(
            status,
            True,
            status_color,
        ).convert_alpha()
        status_surf.set_alpha(128)
        gap = 8
        max_allowed = Display.LEGEND_WIDTH - (
            column_width
            + gap
            + separator.get_width()
            + gap
            + 24
        )
        if (
            status_surf.get_width() <= max_allowed
            or max_allowed <= 32
        ):
            return self._compose_inline_status_surface(
                battery_surf,
                separator,
                status_surf,
                column_width,
                gap,
            )
        return self._compose_wrapped_status_surface(
            battery_surf,
            separator,
            self._wrap_text_surfaces(
                status,
                font_obj,
                Display.LEGEND_WIDTH - 16,
            ),
            column_width,
            gap,
        )

    def _compose_inline_status_surface(
        self,
        battery_surf: pygame.Surface,
        separator: pygame.Surface,
        status_surf: pygame.Surface,
        column_width: int,
        gap: int,
    ) -> pygame.Surface:
        """Compose a one-line battery/status surface."""
        total_w = (
            column_width
            + gap
            + separator.get_width()
            + gap
            + status_surf.get_width()
        )
        max_h = max(
            battery_surf.get_height(),
            separator.get_height(),
            status_surf.get_height(),
        )
        surface = pygame.Surface((total_w, max_h), pygame.SRCALPHA)
        surface.blit(
            battery_surf,
            (
                column_width - battery_surf.get_width(),
                (max_h - battery_surf.get_height()) // 2,
            ),
        )
        separator_x = column_width + gap
        surface.blit(
            separator,
            (
                separator_x,
                (max_h - separator.get_height()) // 2,
            ),
        )
        surface.blit(
            status_surf,
            (
                separator_x + separator.get_width() + gap,
                (max_h - status_surf.get_height()) // 2,
            ),
        )
        return surface

    def _compose_wrapped_status_surface(
        self,
        battery_surf: pygame.Surface,
        separator: pygame.Surface,
        status_lines: list[pygame.Surface],
        column_width: int,
        gap: int,
    ) -> pygame.Surface:
        """Compose a multi-line battery/status surface for narrow rows."""
        first_h = max(
            battery_surf.get_height(),
            separator.get_height(),
        )
        line_gap = 6
        status_h = (
            sum(
                surface.get_height() + line_gap
                for surface in status_lines
            )
            - line_gap
        )
        total_w = min(
            Display.LEGEND_WIDTH - 16,
            max(
                column_width + gap + separator.get_width(),
                max(
                    (
                        surface.get_width()
                        for surface in status_lines
                    ),
                    default=0,
                )
                + column_width
                + gap
                + separator.get_width(),
            ),
        )
        surface = pygame.Surface(
            (total_w, first_h + line_gap + status_h),
            pygame.SRCALPHA,
        )
        surface.blit(
            battery_surf,
            (
                column_width - battery_surf.get_width(),
                (first_h - battery_surf.get_height()) // 2,
            ),
        )
        separator_x = column_width + gap
        surface.blit(
            separator,
            (
                separator_x,
                (first_h - separator.get_height()) // 2,
            ),
        )
        status_x = separator_x + separator.get_width() + gap
        ypos = first_h + line_gap
        for status_surf in status_lines:
            surface.blit(status_surf, (status_x, ypos))
            ypos += status_surf.get_height() + line_gap
        return surface

    def _blit_cached_surface(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        handle: Any,
    ) -> None:
        """Blit a cached surface using one of the configured rect handles."""
        rect = surface.get_rect()
        attribute = self._handle_map.get(
            str(handle).lower(),
            "midleft",
        )
        setattr(rect, attribute, (int(x - self.origin_x), y))
        self.control_surf.blit(surface, rect)

    def _get_font(
        self,
        font_path: Any,
        size: int,
    ) -> pygame.font.Font:
        """Load each font/size pair once and reuse it across frames."""
        key = (font_path, size)
        if key not in self._font_cache:
            self._font_cache[key] = pygame.font.Font(
                font_path,
                size,
            )
        return self._font_cache[key]
