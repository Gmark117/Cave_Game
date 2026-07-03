"""Pygame resources, layout, drawing, and hit-map production for ControlCenter."""

import time
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import pygame

from ui.control_center.controller import ControlHitMap
from asset_config.gameplay import Display
from asset_config.media import Images
from asset_config.rendering import Colors, Fonts, RectHandle
from ui.control_center.view_model import (
    DRONE_ROSTER,
    ROVER_ROSTER,
    AgentRosterEntry,
    ControlCenterViewModel,
    DroneStatusView,
    RoverStatusView,
)


class ControlCenterRenderer:
    """Own every Pygame and layout concern for the mission control panel."""

    TAB_ORDER = ("drones", "rovers", "debug", "system")
    TITLE_Y = 70
    MET_Y = 120
    EXPLORED_Y = 150
    TAB_Y = 178
    SECTION_HEADER_Y = 240
    DRONE_NAME_Y = 275
    DRONE_DATA_Y = 305
    CONTENT_BOTTOM_MARGIN = 40
    TAB_BUTTON_W = 54
    TAB_BUTTON_H = 34
    TAB_BUTTON_GAP = 10

    def __init__(self, game: Any) -> None:
        self.game = game
        self.origin_x = Display.FULL_W - Display.LEGEND_WIDTH
        self.origin_y = 0
        self.origin = (self.origin_x, self.origin_y)
        self.control_surf = pygame.Surface(
            (Display.LEGEND_WIDTH, Display.FULL_H),
            pygame.SRCALPHA,
        )
        self.control_surf.fill((*Colors.BLACK.value, 255))

        self._font_cache: dict[
            tuple[Any, int],
            pygame.font.Font,
        ] = {}
        self._static_surfaces: dict[Any, pygame.Surface] = {}
        self._static_fragments: dict[str, pygame.Surface] = {}
        self._dynamic_cache: dict[str, dict[str, Any]] = {}
        self._tab_sprites: dict[str, pygame.Surface] = {}
        self._handle_map = {
            "center": "center",
            "midtop": "midtop",
            "midright": "midright",
            "midleft": "midleft",
        }
        self._tabs: list[tuple[str, tuple[int, int, int, int]]] = []
        self._drone_toggles: list[
            tuple[int, str, tuple[int, int, int, int]]
        ] = []
        self._heatmap_toggle: Optional[tuple[int, int, int, int]] = None

        self._pre_render_statics()
        self._load_tab_sprites()

    def render(self, view: ControlCenterViewModel) -> ControlHitMap:
        """Draw one immutable frame and return its detached hit geometry."""
        self.control_surf.fill((*Colors.BLACK.value, 255))
        self._tabs = []
        self._drone_toggles = []
        self._heatmap_toggle = None

        self.draw_title()
        self.draw_statistics(view)
        self.draw_tabs(view.active_tab)

        if view.active_tab == "drones":
            self.draw_drone_section(
                view.drone_statuses,
                view.selected_drone_heatmap_id,
            )
        elif view.active_tab == "rovers":
            self.draw_rover_section(view.rover_statuses)
        elif view.active_tab == "debug":
            self.draw_debug_panel(view.debug_lines)
        else:
            self.draw_system_panel(view)

        self.game.window.blit(self.control_surf, self.origin)
        return ControlHitMap(
            heatmap_toggle=self._heatmap_toggle,
            tabs=self._tabs,
            drone_toggles=self._drone_toggles,
        )

    def draw_title(self) -> None:
        surf = self._static_surfaces.get("title")
        if surf is None:
            return
        legend_w = Display.LEGEND_WIDTH
        max_w = max(legend_w - 16, 8)
        if surf.get_width() > max_w:
            scale = max_w / surf.get_width()
            surf = pygame.transform.smoothscale(
                surf,
                (
                    int(surf.get_width() * scale),
                    int(surf.get_height() * scale),
                ),
            ).convert_alpha()

        rect = surf.get_rect()
        rect.centerx = legend_w // 2
        rect.centery = self.TITLE_Y
        if rect.left < 8:
            rect.left = 8
        self.control_surf.blit(surf, rect)

    def draw_statistics(self, view: ControlCenterViewModel) -> None:
        met_texts = [
            ("M.E.T.: ", Colors.GREY.value, 255),
            (view.elapsed_time, Colors.WHITE.value, 255),
        ]
        met_surf = self._get_cached_text_surface(
            "met",
            met_texts,
            25,
            Fonts.BIG.value,
            ttl=1.0,
        )
        self._blit_cached_surface(
            met_surf,
            self.origin_x,
            self.MET_Y,
            RectHandle.MIDLEFT.value,
        )

        explored_texts = [
            ("Explored: ", Colors.GREY.value, 255),
            (
                f"{view.explored_percent}%",
                self.percent_color(view.explored_percent),
                255,
            ),
        ]
        explored_surf = self._get_cached_text_surface(
            f"explored_{view.explored_percent}",
            explored_texts,
            25,
            Fonts.BIG.value,
        )
        self._blit_cached_surface(
            explored_surf,
            self.origin_x,
            self.EXPLORED_Y,
            RectHandle.MIDLEFT.value,
        )
        self.draw_heatmap_toggle(view.show_terrain_heatmap)

    def draw_heatmap_toggle(self, enabled: bool) -> None:
        rect = pygame.Rect(Display.LEGEND_WIDTH - 46, 138, 34, 24)
        self.draw_toggle_button(
            rect,
            "H",
            enabled,
            Colors.EUCALYPTUS.value,
        )
        self._heatmap_toggle = self._absolute_rect(rect)

    def draw_tabs(self, active_tab: str) -> None:
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

    def draw_drone_section(
        self,
        statuses: tuple[DroneStatusView, ...],
        selected_drone_heatmap_id: Optional[int],
    ) -> None:
        self.draw_section_header("drones")
        statuses_by_id = {status.id: status for status in statuses}
        for roster_entry in DRONE_ROSTER:
            self.draw_status(
                roster_entry,
                status_view=statuses_by_id.get(roster_entry.id),
                selected_drone_heatmap_id=selected_drone_heatmap_id,
                name_height=self.DRONE_NAME_Y,
                data_height=self.DRONE_DATA_Y,
            )

    def draw_rover_section(
        self,
        statuses: tuple[RoverStatusView, ...],
    ) -> None:
        self.draw_section_header("rovers")
        statuses_by_id = {status.id: status for status in statuses}
        for roster_entry in ROVER_ROSTER:
            self.draw_status(
                roster_entry,
                rover=True,
                status_view=statuses_by_id.get(roster_entry.id),
                name_height=self.DRONE_NAME_Y,
                data_height=self.DRONE_DATA_Y,
            )

    def draw_debug_panel(self, debug_lines: tuple[str, ...]) -> None:
        self.draw_section_header("debug")
        lines = debug_lines or ("No debug lines available",)
        font_obj = self._get_font(Fonts.BIG.value, 20)
        max_w = Display.LEGEND_WIDTH - 16
        ypos = self.DRONE_NAME_Y
        index = 0
        line_gap = 8
        for line in lines:
            if ":" in line:
                label, value = line.split(":", 1)
                ypos, index, done = self._draw_label_value_entry(
                    "debug",
                    index,
                    label.strip() + ":",
                    value.strip(),
                    font_obj,
                    max_w,
                    ypos,
                    line_gap,
                    line_gap,
                )
            else:
                ypos, index, done = self._draw_wrapped_text_lines(
                    "debug",
                    index,
                    line,
                    font_obj,
                    max_w,
                    ypos,
                    line_gap,
                )
            if done:
                return

    def draw_system_panel(self, view: ControlCenterViewModel) -> None:
        self.draw_section_header("system")
        total_drones = len(view.drone_statuses)
        active_vision = sum(
            1 for drone in view.drone_statuses if drone.show_vision
        )
        active_paths = sum(
            1 for drone in view.drone_statuses if drone.show_path
        )
        avg_battery = (
            0
            if total_drones == 0
            else int(
                sum(
                    drone.battery
                    for drone in view.drone_statuses
                )
                / total_drones
            )
        )
        pairs = [
            ("Active tab:", view.active_tab.upper()),
            ("Drones online:", str(total_drones)),
            ("Rovers online:", str(len(view.rover_statuses))),
            (
                "Vision overlays:",
                f"{active_vision}/{total_drones}",
            ),
            ("Path overlays:", f"{active_paths}/{total_drones}"),
            ("Avg drone battery:", f"{avg_battery}%"),
            ("Debug lines:", str(len(view.debug_lines))),
        ]
        font_obj = self._get_font(Fonts.BIG.value, 20)
        max_w = Display.LEGEND_WIDTH - 16
        ypos = self.DRONE_DATA_Y - 25
        index = 0
        for label, value in pairs:
            ypos, index, done = self._draw_label_value_entry(
                "system",
                index,
                label,
                value,
                font_obj,
                max_w,
                ypos,
                8,
                6,
            )
            if done:
                return

    def draw_status(
        self,
        roster_entry: AgentRosterEntry,
        rover: bool = False,
        status_view: Optional[
            DroneStatusView | RoverStatusView
        ] = None,
        selected_drone_heatmap_id: Optional[int] = None,
        name_height: Optional[int] = None,
        data_height: Optional[int] = None,
    ) -> None:
        deployed = status_view is not None
        number = roster_entry.id
        label = roster_entry.name
        color = (
            roster_entry.color
            if status_view is None
            else status_view.color
        )
        display_label = (
            label
            if status_view is None
            else status_view.name
        )
        if name_height is None:
            name_height = 760 if rover else 230
        if data_height is None:
            data_height = 790 if rover else 260
        max_battery = 2400 if rover else 100

        key = (
            f"rover_{label}"
            if rover
            else f"drone_{label}"
        )
        if deployed:
            key = ("live_agent_name", display_label, color)
            if key not in self._static_surfaces:
                self._static_surfaces[key] = self._get_font(
                    Fonts.BIG.value,
                    25,
                ).render(
                    display_label,
                    True,
                    color,
                ).convert_alpha()
        name_surf = self._static_surfaces[key]
        y_center = name_height + 60 * number
        name_rect = name_surf.get_rect()
        name_rect.midleft = (8, y_center)
        self.control_surf.blit(name_surf, name_rect)

        if isinstance(status_view, DroneStatusView):
            self._draw_drone_toggles(
                status_view,
                y_center,
                selected_drone_heatmap_id,
            )

        if status_view is None:
            na_surf = self._static_fragments["N/A"]
            na_rect = na_surf.get_rect()
            na_rect.midleft = (8, data_height + 60 * number)
            self.control_surf.blit(na_surf, na_rect)
            return

        status_color = self._status_color(status_view.status)
        battery_color = self.percent_color(
            status_view.battery,
            max_battery,
        )
        cache_key = (
            f"status_{'rover_' if rover else ''}"
            f"{label}_{status_view.battery}_{status_view.status}"
        )
        data_surf = self._get_cached_status_surface(
            cache_key,
            status_view.battery,
            status_view.status,
            battery_color,
            status_color,
            25,
            Fonts.BIG.value,
        )
        self._blit_cached_surface(
            data_surf,
            self.origin_x,
            data_height + 60 * number,
            RectHandle.MIDLEFT.value,
        )

    def _draw_drone_toggles(
        self,
        status: DroneStatusView,
        y_center: int,
        selected_drone_heatmap_id: Optional[int],
    ) -> None:
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

    def draw_section_header(self, key: str) -> None:
        surf = self._static_surfaces.get(key)
        if surf is None:
            return
        rect = surf.get_rect()
        rect.centerx = Display.LEGEND_WIDTH // 2
        rect.centery = self.SECTION_HEADER_Y
        self.control_surf.blit(surf, rect)

    def _wrap_text_surfaces(
        self,
        text: str,
        font_obj: pygame.font.Font,
        max_w: int,
    ) -> list[pygame.Surface]:
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
                RectHandle.MIDLEFT.value,
            )
            return ypos + height + line_gap, index + 1, False

        self._blit_cached_surface(
            label_surf,
            self.origin_x,
            ypos,
            RectHandle.MIDLEFT.value,
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
                RectHandle.MIDLEFT.value,
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
                RectHandle.MIDLEFT.value,
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
        key = (font_path, size)
        if key not in self._font_cache:
            self._font_cache[key] = pygame.font.Font(
                font_path,
                size,
            )
        return self._font_cache[key]

    def _load_tab_sprites(self) -> None:
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

    def _pre_render_statics(self) -> None:
        title_font = self._get_font(Fonts.BIG.value, 35)
        self._static_surfaces["title"] = title_font.render(
            "Control Center",
            True,
            Colors.RED.value,
        ).convert_alpha()
        section_font = self._get_font(Fonts.BIG.value, 30)
        for key, label in (
            ("drones", "Drones"),
            ("rovers", "Rovers"),
            ("debug", "Debug"),
            ("system", "System"),
        ):
            self._static_surfaces[key] = section_font.render(
                label,
                True,
                Colors.EUCALYPTUS.value,
            ).convert_alpha()

        name_font = self._get_font(Fonts.BIG.value, 25)
        for entry in DRONE_ROSTER:
            self._static_surfaces[
                f"drone_{entry.name}"
            ] = name_font.render(
                entry.name,
                True,
                entry.color,
            ).convert_alpha()
        for entry in ROVER_ROSTER:
            self._static_surfaces[
                f"rover_{entry.name}"
            ] = name_font.render(
                entry.name,
                True,
                entry.color,
            ).convert_alpha()

        na_surface = name_font.render(
            "N/A",
            True,
            Colors.GREY.value,
        ).convert_alpha()
        na_surface.set_alpha(128)
        self._static_fragments["N/A"] = na_surface

    @staticmethod
    def percent_color(
        value: int,
        max_value: int = 100,
    ) -> tuple[int, int, int]:
        if value < max_value * 20 / 100:
            return Colors.RED.value
        if value < max_value * 80 / 100:
            return Colors.YELLOW.value
        return Colors.GREEN.value

    @staticmethod
    def _status_color(status: str) -> tuple[int, int, int]:
        if status in ("Ready", "Done"):
            return Colors.GREEN.value
        if status in (
            "Updating",
            "Advancing",
            "Sharing",
            "Charging",
        ):
            return Colors.YELLOW.value
        if status in ("Deployed", "Homing"):
            return Colors.WHITE.value
        return Colors.RED.value

    def _absolute_rect(
        self,
        rect: pygame.Rect,
    ) -> tuple[int, int, int, int]:
        return (
            rect.x + self.origin_x,
            rect.y + self.origin_y,
            rect.width,
            rect.height,
        )
