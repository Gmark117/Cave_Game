"""Widget drawing helpers for the control-center renderer."""

from pathlib import Path
from typing import Any, Optional

import pygame

from asset_config.gameplay import Display
from asset_config.media import Images
from asset_config.rendering import Colors, Fonts


class ControlCenterWidgetMixin:
    """Draw tabs, toggles, and renderer-owned hit rectangles."""

    def draw_heatmap_toggle(self, enabled: bool) -> None:
        """Draw the global terrain heatmap toggle and save its hit rect."""
        rect = pygame.Rect(Display.LEGEND_WIDTH - 46, 138, 34, 24)
        self.draw_toggle_button(
            rect,
            "H",
            enabled,
            Colors.EUCALYPTUS.value,
        )
        self._heatmap_toggle = self._absolute_rect(rect)

    def draw_tabs(self, active_tab: str) -> None:
        """Draw tab buttons and collect their absolute hit rectangles."""
        total_w = (
            self.TAB_BUTTON_W * len(self.TAB_ORDER)
            + self.TAB_BUTTON_GAP * (len(self.TAB_ORDER) - 1)
        )
        start_x = (Display.LEGEND_WIDTH - total_w) // 2
        for index, tab_name in enumerate(self.TAB_ORDER):
            rect = pygame.Rect(
                start_x
                + index * (self.TAB_BUTTON_W + self.TAB_BUTTON_GAP),
                self.TAB_Y,
                self.TAB_BUTTON_W,
                self.TAB_BUTTON_H,
            )
            self.draw_tab_button(
                rect,
                tab_name,
                active_tab == tab_name,
            )
            self._tabs.append((tab_name, self._absolute_rect(rect)))

    def draw_tab_button(
        self,
        rect: pygame.Rect,
        tab_name: str,
        active: bool,
    ) -> None:
        """Draw one tab button with an icon and active/inactive styling."""
        bg_color = Colors.WHITE.value if active else Colors.GREY.value
        alpha_bg = 230 if active else 128
        button_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            button_surf,
            (*bg_color, alpha_bg),
            button_surf.get_rect(),
            border_radius=8,
        )
        border_color = (
            Colors.EUCALYPTUS.value
            if active
            else Colors.GREY.value
        )
        pygame.draw.rect(
            button_surf,
            (*border_color, 220),
            button_surf.get_rect(),
            width=2 if active else 1,
            border_radius=8,
        )
        icon_color = Colors.BLACK.value if active else Colors.GREY.value
        self.draw_tab_icon(
            button_surf,
            tab_name,
            icon_color,
        )
        self.control_surf.blit(button_surf, rect.topleft)

    def draw_tab_icon(
        self,
        target: pygame.Surface,
        tab_name: str,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a loaded tab sprite, or a simple fallback icon."""
        width, height = target.get_size()
        center_x = width // 2
        center_y = height // 2

        if tab_name in self._tab_sprites:
            image = self._tab_sprites[tab_name]
            target.blit(
                image,
                (
                    (width - image.get_width()) // 2,
                    (height - image.get_height()) // 2,
                ),
            )
            return

        if tab_name == "drones":
            pygame.draw.circle(
                target,
                color,
                (center_x, center_y + 2),
                5,
                width=2,
            )
            pygame.draw.line(
                target,
                color,
                (center_x - 10, center_y - 6),
                (center_x + 10, center_y - 6),
                2,
            )
            pygame.draw.line(
                target,
                color,
                (center_x, center_y - 11),
                (center_x, center_y - 1),
                2,
            )
        elif tab_name == "rovers":
            pygame.draw.rect(
                target,
                color,
                pygame.Rect(center_x - 10, center_y - 6, 20, 10),
                width=2,
                border_radius=2,
            )
            pygame.draw.circle(
                target,
                color,
                (center_x - 7, center_y + 8),
                3,
                width=1,
            )
            pygame.draw.circle(
                target,
                color,
                (center_x + 7, center_y + 8),
                3,
                width=1,
            )
        elif tab_name == "debug":
            pygame.draw.circle(
                target,
                color,
                (center_x - 2, center_y - 2),
                6,
                width=2,
            )
            pygame.draw.line(
                target,
                color,
                (center_x + 3, center_y + 3),
                (center_x + 10, center_y + 10),
                2,
            )
            pygame.draw.line(
                target,
                color,
                (center_x - 2, center_y - 6),
                (center_x - 2, center_y + 2),
                1,
            )
            pygame.draw.line(
                target,
                color,
                (center_x - 6, center_y - 2),
                (center_x + 2, center_y - 2),
                1,
            )
        else:
            for index, bar_height in enumerate((6, 11, 8)):
                pygame.draw.rect(
                    target,
                    color,
                    pygame.Rect(
                        center_x - 9 + index * 7,
                        center_y + 6 - bar_height,
                        4,
                        bar_height,
                    ),
                    border_radius=1,
                )

    def _draw_drone_toggles(
        self,
        status: Any,
        y_center: int,
        selected_drone_heatmap_id: Optional[int],
    ) -> None:
        """Draw path/vision/terrain buttons for one drone row."""
        button_width = 34
        button_height = 24
        gap = 8
        start_x = Display.LEGEND_WIDTH - (
            button_width * 3 + gap * 2 + 12
        )
        top = y_center - button_height // 2
        path_rect = pygame.Rect(
            start_x,
            top,
            button_width,
            button_height,
        )
        vision_rect = pygame.Rect(
            start_x + button_width + gap,
            top,
            button_width,
            button_height,
        )
        terrain_rect = pygame.Rect(
            start_x + (button_width + gap) * 2,
            top,
            button_width,
            button_height,
        )
        self.draw_toggle_button(
            path_rect,
            "P",
            status.show_path,
            status.color,
        )
        self.draw_toggle_button(
            vision_rect,
            "V",
            status.show_vision,
            status.color,
        )
        self.draw_toggle_button(
            terrain_rect,
            "T",
            selected_drone_heatmap_id == status.id,
            status.color,
        )
        self._drone_toggles.extend(
            (
                (
                    status.id,
                    "path",
                    self._absolute_rect(path_rect),
                ),
                (
                    status.id,
                    "vision",
                    self._absolute_rect(vision_rect),
                ),
                (
                    status.id,
                    "terrain",
                    self._absolute_rect(terrain_rect),
                ),
            )
        )

    def draw_toggle_button(
        self,
        rect: pygame.Rect,
        label: str,
        enabled: bool,
        accent_color: tuple[int, int, int],
    ) -> None:
        """Draw a small labeled or square toggle button."""
        bg_color = accent_color if enabled else Colors.GREY.value
        button_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        if label.upper() != "T":
            pygame.draw.rect(
                button_surf,
                (*bg_color, 128),
                button_surf.get_rect(),
                border_radius=6,
            )
            pygame.draw.rect(
                button_surf,
                (*Colors.WHITE.value, 128),
                button_surf.get_rect(),
                width=1,
                border_radius=6,
            )

        if label.upper() == "T":
            width, height = button_surf.get_size()
            pad = max(2, min(width, height) // 8)
            side = min(width, height) - pad * 2
            square_rect = pygame.Rect(0, 0, side, side)
            square_rect.center = (width // 2, height // 2)
            pygame.draw.rect(
                button_surf,
                (*Colors.WHITE.value, 128),
                square_rect,
                width=1,
            )
            if enabled:
                inner_inset = max(3, side // 6)
                inner_rect = pygame.Rect(
                    0,
                    0,
                    side - inner_inset * 2,
                    side - inner_inset * 2,
                )
                inner_rect.center = square_rect.center
                pygame.draw.rect(
                    button_surf,
                    (*accent_color, 128),
                    inner_rect,
                )
        else:
            text_color = (
                Colors.BLACK.value
                if enabled
                else Colors.WHITE.value
            )
            text_surf = self._get_font(
                Fonts.BIG.value,
                18,
            ).render(
                label,
                True,
                text_color,
            ).convert_alpha()
            text_surf.set_alpha(128)
            button_surf.blit(
                text_surf,
                text_surf.get_rect(
                    center=button_surf.get_rect().center
                ),
            )
        self.control_surf.blit(button_surf, rect.topleft)

    def _load_tab_sprites(self) -> None:
        """Load optional bitmap icons for tab buttons."""
        self._load_tab_sprite(
            "drones",
            Images.DRONE.value,
            (28, 28),
        )
        self._load_tab_sprite(
            "rovers",
            Images.ROVER.value,
            (28, 28),
        )
        self._load_tab_sprite(
            "debug",
            Images.DEBUG_ICON.value,
            (34, 34),
        )
        self._load_tab_sprite(
            "system",
            Images.SYSTEM_ICON.value,
            (36, 36),
        )

    def _load_tab_sprite(
        self,
        tab_name: str,
        image_path: Any,
        size: tuple[int, int],
    ) -> None:
        """Load a tab icon, preferring the outlined variant when present."""
        try:
            path = Path(str(image_path))
            outlined = path.with_name(
                path.stem + "_outlined" + path.suffix
            )
            load_path = outlined if outlined.exists() else path
            image = pygame.image.load(
                str(load_path)
            ).convert_alpha()
        except Exception:
            return
        self._tab_sprites[tab_name] = pygame.transform.smoothscale(
            image,
            size,
        )

    def _absolute_rect(
        self,
        rect: pygame.Rect,
    ) -> tuple[int, int, int, int]:
        """Convert panel-local rectangle coordinates to window coordinates."""
        return (
            rect.x + self.origin_x,
            rect.y + self.origin_y,
            rect.width,
            rect.height,
        )
