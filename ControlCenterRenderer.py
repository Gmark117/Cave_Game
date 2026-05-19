"""Dedicated renderer for ControlCenter orchestration.

This module contains a thin renderer class that drives the existing
ControlCenter drawing helpers. Moving orchestration here keeps
`ControlCenter` focused on state while the renderer handles draw order.
"""

from typing import Any, List, Optional, Tuple
import pygame
from asset_config.gameplay import Display
from asset_config.rendering import Colors, Fonts, RectHandle


class ControlCenterRenderer:
    """Orchestrates ControlCenter drawing using extracted helpers.

    The renderer implements the higher-level draw steps while still
    relying on a few low-level text/cache helpers present on the
    `ControlCenter` instance to avoid duplicating cache logic.
    """

    def render(
        self,
        cc: Any,
        drone_objects: List[Any],
        rover_objects: Optional[List[Any]] = None,
        show_terrain_heatmap: bool = True,
        selected_drone_heatmap_id: Optional[int] = None,
        debug_lines: Optional[List[str]] = None,
    ) -> None:
        # Clear control surface, draw content there, then blit once to window
        cc.control_surf.fill((*Colors.BLACK.value, 255))
        cc.drone_toggle_rects.clear()
        cc._tab_state.clear()
        cc.heatmap_toggle_rect = None

        if rover_objects is not None:
            for name, rover_info in cc.rovers.items():
                rover_id = rover_info["id"]
                if rover_id < len(rover_objects):
                    rover = rover_objects[rover_id]
                    rover_info["battery"] = rover.battery
                    rover_info["status"] = rover.status

        self.draw_title(cc)
        self.draw_statistics(cc, show_terrain_heatmap)
        self.draw_tabs(cc)

        if cc.active_tab == "drones":
            self.draw_drone_section(cc, drone_objects, selected_drone_heatmap_id)
        elif cc.active_tab == "rovers":
            self.draw_rover_section(cc)
        elif cc.active_tab == "debug":
            self.draw_debug_panel(cc, debug_lines)
        else:
            self.draw_system_panel(cc, drone_objects, debug_lines)

        cc.game.window.blit(cc.control_surf, cc.origin)

    # --- High level draw steps (moved from ControlCenter) ---
    def draw_title(self, cc: Any) -> None:
        surf = cc._static_surfaces.get("title")
        if surf:
            legend_w = Display.LEGEND_WIDTH
            max_w = max(legend_w - 16, 8)
            if surf.get_width() > max_w:
                scale = max_w / surf.get_width()
                new_w = int(surf.get_width() * scale)
                new_h = int(surf.get_height() * scale)
                surf = pygame.transform.smoothscale(
                    surf, (new_w, new_h)
                ).convert_alpha()

            rect = surf.get_rect()
            rect.centerx = legend_w // 2
            rect.centery = cc.TITLE_Y
            if rect.left < 8:
                rect.left = 8
            cc.control_surf.blit(surf, rect)

    def draw_statistics(self, cc: Any, show_terrain_heatmap: bool) -> None:
        met_texts = [
            ("M.E.T.: ", Colors.GREY.value, 255),
            (cc.format_timer(), Colors.WHITE.value, 255),
        ]
        met_surf = cc._get_cached_text_surface(
            "met", met_texts, 25, Fonts.BIG.value, ttl=1.0
        )
        if met_surf:
            cc._blit_cached_surface(
                met_surf, cc.origin_x, cc.MET_Y, RectHandle.MIDLEFT.value
            )

        explored_texts = [
            ("Explored: ", Colors.GREY.value, 255),
            (f"{cc.explored_percent}%", cc.percent_color(cc.explored_percent), 255),
        ]
        explored_surf = cc._get_cached_text_surface(
            f"explored_{cc.explored_percent}", explored_texts, 25, Fonts.BIG.value
        )
        if explored_surf:
            cc._blit_cached_surface(
                explored_surf, cc.origin_x, cc.EXPLORED_Y, RectHandle.MIDLEFT.value
            )

        self.draw_heatmap_toggle(cc, show_terrain_heatmap)

    def draw_heatmap_toggle(self, cc: Any, enabled: bool) -> None:
        rect = pygame.Rect(Display.LEGEND_WIDTH - 46, 138, 34, 24)
        self.draw_toggle_button(cc, rect, "H", enabled, Colors.EUCALYPTUS.value)
        cc.heatmap_toggle_rect = rect.move(cc.origin_x, cc.origin_y)

    def draw_tabs(self, cc: Any) -> None:
        button_w = cc.TAB_BUTTON_W
        button_h = cc.TAB_BUTTON_H
        gap = cc.TAB_BUTTON_GAP
        total_w = (button_w * len(cc.TAB_ORDER)) + (gap * (len(cc.TAB_ORDER) - 1))
        start_x = (Display.LEGEND_WIDTH - total_w) // 2
        y = cc.TAB_Y

        for idx, tab_name in enumerate(cc.TAB_ORDER):
            x = start_x + idx * (button_w + gap)
            rect = pygame.Rect(x, y, button_w, button_h)
            is_active = cc.active_tab == tab_name
            self.draw_tab_button(cc, rect, tab_name, is_active)
            cc._tab_state.register(tab_name, rect.move(cc.origin_x, cc.origin_y))

    def draw_tab_button(
        self, cc: Any, rect: "pygame.Rect", tab_name: str, active: bool
    ) -> None:
        # Match other toggles: use white when active, grey when inactive
        bg_color = Colors.WHITE.value if active else Colors.GREY.value
        # Slightly more transparent when inactive to match toggle buttons
        alpha_bg = 230 if active else 128

        button_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            button_surf, (*bg_color, alpha_bg), button_surf.get_rect(), border_radius=8
        )
        border_color = Colors.EUCALYPTUS.value if active else Colors.GREY.value
        pygame.draw.rect(
            button_surf,
            (*border_color, 220),
            button_surf.get_rect(),
            width=2 if active else 1,
            border_radius=8,
        )

        icon_color = Colors.BLACK.value if active else Colors.GREY.value
        self.draw_tab_icon(cc, button_surf, tab_name, icon_color)
        cc.control_surf.blit(button_surf, rect.topleft)

    def draw_tab_icon(
        self,
        cc: Any,
        target: "pygame.Surface",
        tab_name: str,
        color: Tuple[int, int, int],
    ) -> None:
        w, h = target.get_size()
        cx = w // 2
        cy = h // 2

        if tab_name in cc._tab_sprites:
            img = cc._tab_sprites[tab_name]
            iw, ih = img.get_size()
            x = (w - iw) // 2
            y = (h - ih) // 2
            # Blit the image only; outlined variants are baked into PNGs.
            target.blit(img, (x, y))
            return

        if tab_name == "drones":
            pygame.draw.circle(target, color, (cx, cy + 2), 5, width=2)
            pygame.draw.line(target, color, (cx - 10, cy - 6), (cx + 10, cy - 6), 2)
            pygame.draw.line(target, color, (cx, cy - 11), (cx, cy - 1), 2)
        elif tab_name == "rovers":
            pygame.draw.rect(
                target,
                color,
                pygame.Rect(cx - 10, cy - 6, 20, 10),
                width=2,
                border_radius=2,
            )
            pygame.draw.circle(target, color, (cx - 7, cy + 8), 3, width=1)
            pygame.draw.circle(target, color, (cx + 7, cy + 8), 3, width=1)
        elif tab_name == "debug":
            pygame.draw.circle(target, color, (cx - 2, cy - 2), 6, width=2)
            pygame.draw.line(target, color, (cx + 3, cy + 3), (cx + 10, cy + 10), 2)
            pygame.draw.line(target, color, (cx - 2, cy - 6), (cx - 2, cy + 2), 1)
            pygame.draw.line(target, color, (cx - 6, cy - 2), (cx + 2, cy - 2), 1)
        else:
            bars = [6, 11, 8]
            start_x = cx - 9
            for i, bar_h in enumerate(bars):
                bar_rect = pygame.Rect(start_x + (i * 7), cy + 6 - bar_h, 4, bar_h)
                pygame.draw.rect(target, color, bar_rect, border_radius=1)

    def draw_drone_section(
        self,
        cc: Any,
        drone_objects: List[Any],
        selected_drone_heatmap_id: Optional[int],
    ) -> None:
        self.draw_section_header(cc, "drones")

        for drone in cc.drones:
            if cc.drones[drone]["id"] < cc.num_drones:
                drone_id = cc.drones[drone]["id"]
                self.draw_status(
                    cc,
                    drone,
                    drone_obj=drone_objects[drone_id],
                    selected_drone_heatmap_id=selected_drone_heatmap_id,
                    name_height=cc.DRONE_NAME_Y,
                    data_height=cc.DRONE_DATA_Y,
                )
            else:
                self.draw_status(
                    cc,
                    drone,
                    deployed=False,
                    name_height=cc.DRONE_NAME_Y,
                    data_height=cc.DRONE_DATA_Y,
                )

    def draw_rover_section(self, cc: Any) -> None:
        self.draw_section_header(cc, "rovers")

        for rover in cc.rovers:
            if cc.rovers[rover]["id"] < cc.num_rovers:
                self.draw_status(
                    cc,
                    rover,
                    rover=True,
                    name_height=cc.DRONE_NAME_Y,
                    data_height=cc.DRONE_DATA_Y,
                )
            else:
                self.draw_status(
                    cc,
                    rover,
                    rover=True,
                    deployed=False,
                    name_height=cc.DRONE_NAME_Y,
                    data_height=cc.DRONE_DATA_Y,
                )

    def draw_debug_panel(self, cc: Any, debug_lines: Optional[List[str]]) -> None:
        self.draw_section_header(cc, "debug")

        lines = debug_lines or ["No debug lines available"]
        font_obj = cc._get_font(Fonts.BIG.value, 20)
        max_w = Display.LEGEND_WIDTH - 16
        ypos = cc.DRONE_NAME_Y
        idx = 0
        line_gap = 8
        for line in lines:
            if ":" in line:
                label, val = line.split(":", 1)
                label = label.strip() + ":"
                val = val.strip()
                ypos, idx, done = cc._draw_label_value_entry(
                    "debug", idx, label, val, font_obj, max_w, ypos, line_gap, line_gap
                )
                if done:
                    return
            else:
                ypos, idx, done = cc._draw_wrapped_text_lines(
                    "debug", idx, line, font_obj, max_w, ypos, line_gap
                )
                if done:
                    return

    def draw_system_panel(
        self, cc: Any, drone_objects: List[Any], debug_lines: Optional[List[str]]
    ) -> None:
        self.draw_section_header(cc, "system")

        total_drones = len(drone_objects)
        active_vision = sum(
            1 for drone in drone_objects if getattr(drone, "show_vision", False)
        )
        active_paths = sum(
            1 for drone in drone_objects if getattr(drone, "show_path", False)
        )
        avg_battery = (
            0
            if total_drones == 0
            else int(
                sum(int(getattr(drone, "battery", 0)) for drone in drone_objects)
                / total_drones
            )
        )

        system_pairs = [
            ("Active tab:", cc.active_tab.upper()),
            ("Drones online:", str(total_drones)),
            ("Rovers online:", str(cc.num_rovers)),
            ("Vision overlays:", f"{active_vision}/{total_drones}"),
            ("Path overlays:", f"{active_paths}/{total_drones}"),
            ("Avg drone battery:", f"{avg_battery}%"),
            ("Debug lines:", str(0 if not debug_lines else len(debug_lines))),
        ]

        font_obj = cc._get_font(Fonts.BIG.value, 20)
        max_w = Display.LEGEND_WIDTH - 16
        ypos = cc.DRONE_DATA_Y - 25
        idx = 0
        for label, val in system_pairs:
            ypos, idx, done = cc._draw_label_value_entry(
                "system", idx, label, val, font_obj, max_w, ypos, 8, 6
            )
            if done:
                return

    def draw_toggle_button(
        self,
        cc: Any,
        rect: "pygame.Rect",
        label: str,
        enabled: bool,
        accent_color: Tuple[int, int, int],
    ) -> None:
        bg_color = accent_color if enabled else Colors.GREY.value

        button_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        # For terrain toggle ('T') we don't draw the rounded background;
        # render a standalone square icon instead. For other toggles, keep
        # the existing rounded background style.
        if label.upper() != "T":
            pygame.draw.rect(
                button_surf, (*bg_color, 128), button_surf.get_rect(), border_radius=6
            )
            pygame.draw.rect(
                button_surf,
                (*Colors.WHITE.value, 128),
                button_surf.get_rect(),
                width=1,
                border_radius=6,
            )

        # Special-case terrain toggle: draw an outer square with optional inner square
        if label.upper() == "T":
            w, h = button_surf.get_size()
            # Draw a centered square icon inside the rectangular button area.
            # Determine square side length and center it.
            pad = max(2, min(w, h) // 8)
            side = min(w, h) - (pad * 2)
            cx = w // 2
            cy = h // 2
            square_rect = pygame.Rect(0, 0, side, side)
            square_rect.center = (cx, cy)

            # Draw outer square outline (no rounded corners) with semi-transparency
            pygame.draw.rect(button_surf, (*Colors.WHITE.value, 128), square_rect, width=1)

            if enabled:
                # Inner filled square when toggled ON, inset slightly, semi-transparent
                inner_inset = max(3, side // 6)
                inner_side = side - (inner_inset * 2)
                inner_rect = pygame.Rect(0, 0, inner_side, inner_side)
                inner_rect.center = (cx, cy)
                pygame.draw.rect(button_surf, (*accent_color, 128), inner_rect)
        else:
            # Default behavior: draw label text
            text_color = Colors.BLACK.value if enabled else Colors.WHITE.value
            font = cc._get_font(Fonts.BIG.value, 18)
            text_surf = font.render(label, True, text_color).convert_alpha()
            text_surf.set_alpha(128)
            button_surf.blit(
                text_surf, text_surf.get_rect(center=button_surf.get_rect().center)
            )

        cc.control_surf.blit(button_surf, rect.topleft)

    def draw_section_header(self, cc: Any, key: str) -> None:
        surf = cc._static_surfaces.get(key)
        if surf is None:
            return
        rect = surf.get_rect()
        rect.centerx = Display.LEGEND_WIDTH // 2
        rect.centery = cc.SECTION_HEADER_Y
        cc.control_surf.blit(surf, rect)

    def draw_status(
        self,
        cc: Any,
        label: str,
        rover: bool = False,
        deployed: bool = True,
        drone_obj: Any = None,
        selected_drone_heatmap_id: Optional[int] = None,
        name_height: Optional[int] = None,
        data_height: Optional[int] = None,
    ) -> None:
        if not rover:
            number = cc.drones[label]["id"]
            color = cc.drones[label]["color"]
            battery = cc.drones[label]["battery"]
            status = cc.drones[label]["status"]

            if name_height is None:
                name_height = 230
            if data_height is None:
                data_height = 260
            max_battery = 100
        else:
            number = cc.rovers[label]["id"]
            color = cc.rovers[label]["color"]
            battery = cc.rovers[label]["battery"]
            status = cc.rovers[label]["status"]

            if name_height is None:
                name_height = 760
            if data_height is None:
                data_height = 790
            max_battery = 2400

        key = ("rover_" + label) if rover else ("drone_" + label)
        name_surf = cc._static_surfaces.get(key)
        y_center = name_height + 60 * number
        rect = name_surf.get_rect()
        rect.midleft = (8, y_center)
        cc.control_surf.blit(name_surf, rect)

        if deployed and not rover and drone_obj is not None:
            # draw toggles using existing helper on ControlCenter
            cc._draw_drone_toggles(
                number, y_center, drone_obj, selected_drone_heatmap_id
            )

        if deployed:
            match status:
                case "Ready" | "Done":
                    status_color = Colors.GREEN.value
                case "Updating" | "Advancing" | "Sharing" | "Charging":
                    status_color = Colors.YELLOW.value
                case "Deployed" | "Homing":
                    status_color = Colors.WHITE.value
                case _:
                    status_color = Colors.RED.value

            battery_color = cc.percent_color(battery, max_battery)

            key = f'status_{"rover_"+label if rover else label}_{battery}_{status}'
            data_surf = cc._get_cached_status_surface(
                key,
                battery,
                status,
                battery_color,
                status_color,
                25,
                Fonts.BIG.value,
                max_battery,
            )
            if data_surf:
                cc._blit_cached_surface(
                    data_surf,
                    cc.origin_x,
                    data_height + 60 * number,
                    RectHandle.MIDLEFT.value,
                )
        else:
            na_surf = cc._static_fragments["N/A"]
            rect = na_surf.get_rect()
            rect.midleft = (8, data_height + 60 * number)
            cc.control_surf.blit(na_surf, rect)

