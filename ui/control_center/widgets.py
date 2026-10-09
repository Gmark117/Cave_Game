"""Widget drawing helpers for the control-center renderer."""

from pathlib import Path
from typing import Any, Optional

import pygame

from asset_config.gameplay import Display


BUTTON_ASSET_DIR = (
    Path(__file__).resolve().parents[2] / "Assets" / "Images" / "Buttons"
)
BUTTON_SIZE = 40
BUTTON_ICON_SIZE = 28
BUTTON_RADIUS = 8
DRONE_BUTTON_SIZE = 26
DRONE_BUTTON_ICON_SIZE = 18
DRONE_BUTTON_RADIUS = 5
BUTTON_BG = (86, 86, 86)
BUTTON_BG_ACTIVE = (104, 104, 104)
BUTTON_BORDER = (64, 64, 64)
BUTTON_ACTIVE_BORDER = (86, 120, 106)
BUTTON_SEPARATOR = (112, 112, 112, 190)
BUTTON_SEPARATOR_RADIUS = 3


class ControlCenterWidgetMixin:
    """Draw tabs, toggles, and renderer-owned hit rectangles."""

    def draw_mission_controls(
        self,
        is_paused: bool,
        music_enabled: bool,
    ) -> None:
        """Draw mission-level controls in the first button row."""
        buttons = (
            ("stop", "stop_button.png"),
            ("restart", "restart_button.png"),
            (
                "pause",
                "play_button.png" if is_paused else "pause_button.png",
            ),
            (
                "music",
                (
                    "music_ON_button.png"
                    if music_enabled
                    else "music_OFF_button.png"
                ),
            ),
            ("exit", "exit_button.png"),
        )
        for (action, asset_name), rect in zip(
            buttons,
            self._button_row_rects(len(buttons), self.MISSION_BUTTON_Y),
        ):
            self.draw_image_button(rect, asset_name)
            self._mission_controls.append(
                (action, self._absolute_rect(rect))
            )
        self._draw_separator_between_buttons(
            self._mission_controls[2][1],
            self._mission_controls[3][1],
        )

    def draw_tabs(
        self,
        active_tab: str,
        heatmap_enabled: bool,
        full_map_enabled: bool,
    ) -> None:
        """Draw tab buttons plus the map/LIDAR view-control group."""
        tab_rects, map_rect, heatmap_rect = self._view_control_rects()
        for tab_name, rect in zip(self.TAB_ORDER, tab_rects):
            self.draw_tab_button(
                rect,
                tab_name,
                active_tab == tab_name,
            )
            self._tabs.append((tab_name, self._absolute_rect(rect)))

        self._draw_separator_dot(
            (tab_rects[-1].right + map_rect.left) // 2,
            self.TAB_Y + self.TAB_BUTTON_H // 2,
        )
        self.draw_image_button(
            map_rect,
            self._state_asset_name("map", full_map_enabled),
            active=full_map_enabled,
        )
        self._map_toggle = self._absolute_rect(map_rect)
        self.draw_image_button(
            heatmap_rect,
            self._state_asset_name("lidar_view", heatmap_enabled),
            active=heatmap_enabled,
        )
        self._heatmap_toggle = self._absolute_rect(heatmap_rect)

    def draw_tab_button(
        self,
        rect: pygame.Rect,
        tab_name: str,
        active: bool,
    ) -> None:
        """Draw one tab button with an icon and active/inactive styling."""
        asset_name = {
            "drones": "drone_button.png",
            "rovers": "rover_button.png",
            "debug": "debug_button.png",
            "system": "system_button.png",
        }.get(tab_name)
        if asset_name is None:
            return
        self.draw_image_button(rect, asset_name, active=active)

    def _draw_drone_toggles(
        self,
        status: Any,
        y_center: int,
        selected_drone_heatmap_id: Optional[int],
    ) -> None:
        """Draw path, vision, and selected-drone buttons for one row."""
        button_width = DRONE_BUTTON_SIZE
        button_height = DRONE_BUTTON_SIZE
        gap = 5
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
        selected_rect = pygame.Rect(
            start_x + (button_width + gap) * 2,
            top,
            button_width,
            button_height,
        )
        self.draw_toggle_button(
            path_rect,
            "P",
            status.show_path,
        )
        self.draw_toggle_button(
            vision_rect,
            "V",
            status.show_vision,
        )
        self.draw_toggle_button(
            selected_rect,
            "T",
            selected_drone_heatmap_id == status.id,
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
                    "selected",
                    self._absolute_rect(selected_rect),
                ),
            )
        )

    def _draw_rover_toggle(
        self,
        status: Any,
        y_center: int,
        selected_rover_heatmap_id: Optional[int],
    ) -> None:
        """Draw highway visibility and rover-local SLAM/terrain buttons."""
        highway_rect = pygame.Rect(
            Display.LEGEND_WIDTH - DRONE_BUTTON_SIZE * 2 - 5 - 12,
            y_center - DRONE_BUTTON_SIZE // 2,
            DRONE_BUTTON_SIZE,
            DRONE_BUTTON_SIZE,
        )
        self.draw_toggle_button(
            highway_rect,
            "P",
            status.show_highway,
        )
        self._rover_toggles.append((
            status.id,
            "highway",
            self._absolute_rect(highway_rect),
        ))
        rect = pygame.Rect(
            Display.LEGEND_WIDTH - DRONE_BUTTON_SIZE - 12,
            y_center - DRONE_BUTTON_SIZE // 2,
            DRONE_BUTTON_SIZE,
            DRONE_BUTTON_SIZE,
        )
        self.draw_toggle_button(
            rect,
            "T",
            selected_rover_heatmap_id == status.id,
        )
        self._rover_toggles.append((
            status.id,
            "selected",
            self._absolute_rect(rect),
        ))

    def draw_toggle_button(
        self,
        rect: pygame.Rect,
        label: str,
        enabled: bool,
    ) -> None:
        """Draw a small labeled or square toggle button."""
        self.draw_image_button(
            rect,
            self._toggle_asset_name(label, enabled),
            active=enabled,
            icon_size=DRONE_BUTTON_ICON_SIZE,
            border_radius=DRONE_BUTTON_RADIUS,
            border_width=1,
        )

    def draw_image_button(
        self,
        rect: pygame.Rect,
        asset_name: str,
        active: bool = False,
        icon_size: int = BUTTON_ICON_SIZE,
        border_radius: int = BUTTON_RADIUS,
        border_width: int = 2,
    ) -> None:
        """Draw a rounded image button with a centered bitmap icon."""
        button_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        bg_color = BUTTON_BG_ACTIVE if active else BUTTON_BG
        border_color = BUTTON_ACTIVE_BORDER if active else BUTTON_BORDER
        pygame.draw.rect(
            button_surf,
            bg_color,
            button_surf.get_rect(),
            border_radius=border_radius,
        )
        pygame.draw.rect(
            button_surf,
            border_color,
            button_surf.get_rect(),
            width=max(1, border_width),
            border_radius=border_radius,
        )

        icon = self._get_button_sprite(asset_name, icon_size)
        if icon is not None:
            button_surf.blit(
                icon,
                icon.get_rect(center=button_surf.get_rect().center),
            )
        self.control_surf.blit(button_surf, rect.topleft)

    def _get_button_sprite(
        self,
        asset_name: str,
        icon_size: int = BUTTON_ICON_SIZE,
    ) -> Optional[pygame.Surface]:
        """Load and cache one button icon scaled to the shared icon size."""
        if not hasattr(self, "_button_sprites"):
            self._button_sprites = {}
        cache_key = f"{asset_name}:{icon_size}"
        if cache_key in self._button_sprites:
            return self._button_sprites[cache_key]

        try:
            image = pygame.image.load(
                str(BUTTON_ASSET_DIR / asset_name)
            )
            try:
                image = image.convert_alpha()
            except pygame.error:
                image = image.copy()
        except Exception:
            return None

        self._button_sprites[cache_key] = pygame.transform.smoothscale(
            image,
            (icon_size, icon_size),
        )
        return self._button_sprites[cache_key]

    def _button_row_rects(
        self,
        count: int,
        top: int,
    ) -> tuple[pygame.Rect, ...]:
        """Return centered same-size button rectangles for one row."""
        total_w = (
            self.TAB_BUTTON_W * count
            + self.TAB_BUTTON_GAP * (count - 1)
        )
        start_x = (Display.LEGEND_WIDTH - total_w) // 2
        return tuple(
            pygame.Rect(
                start_x + index * (self.TAB_BUTTON_W + self.TAB_BUTTON_GAP),
                top,
                self.TAB_BUTTON_W,
                self.TAB_BUTTON_H,
            )
            for index in range(count)
        )

    def _view_control_rects(
        self,
    ) -> tuple[tuple[pygame.Rect, ...], pygame.Rect, pygame.Rect]:
        """Return compact tab/map/LIDAR rectangles for the grouped row."""
        tab_start_x = 7
        tab_gap = 7
        map_left = 204
        view_gap = 8
        tab_rects = tuple(
            pygame.Rect(
                tab_start_x
                + index * (self.TAB_BUTTON_W + tab_gap),
                self.TAB_Y,
                self.TAB_BUTTON_W,
                self.TAB_BUTTON_H,
            )
            for index in range(len(self.TAB_ORDER))
        )
        map_rect = pygame.Rect(
            map_left,
            self.TAB_Y,
            self.TAB_BUTTON_W,
            self.TAB_BUTTON_H,
        )
        heatmap_rect = pygame.Rect(
            map_rect.right + view_gap,
            self.TAB_Y,
            self.TAB_BUTTON_W,
            self.TAB_BUTTON_H,
        )
        return tab_rects, map_rect, heatmap_rect

    def _draw_separator_between_buttons(
        self,
        left_abs_rect: tuple[int, int, int, int],
        right_abs_rect: tuple[int, int, int, int],
    ) -> None:
        """Draw a small separator dot in the gap between two buttons."""
        left = pygame.Rect(left_abs_rect).move(-self.origin_x, -self.origin_y)
        right = pygame.Rect(right_abs_rect).move(
            -self.origin_x,
            -self.origin_y,
        )
        self._draw_separator_dot(
            (left.right + right.left) // 2,
            (left.centery + right.centery) // 2,
        )

    def _draw_separator_dot(self, x: int, y: int) -> None:
        """Draw a muted dot separator between compact button groups."""
        pygame.draw.circle(
            self.control_surf,
            BUTTON_SEPARATOR,
            (int(x), int(y)),
            BUTTON_SEPARATOR_RADIUS,
        )

    @staticmethod
    def _toggle_asset_name(label: str, enabled: bool) -> str:
        """Resolve a row-toggle label to its approved ON/OFF asset."""
        mapping = {
            "P": "path",
            "V": "vision",
            "T": "selected",
        }
        return ControlCenterWidgetMixin._state_asset_name(
            mapping.get(label.upper(), "selected"),
            enabled,
        )

    @staticmethod
    def _state_asset_name(prefix: str, enabled: bool) -> str:
        """Resolve an ON/OFF button asset name from a prefix."""
        state = "ON" if enabled else "OFF"
        return f"{prefix}_{state}_button.png"

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
