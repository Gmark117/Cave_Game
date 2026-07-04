"""Top-level Pygame renderer for the mission control-center panel."""

from typing import Any, Optional

import pygame

from asset_config.gameplay import Display
from asset_config.rendering import Colors, Fonts, RectHandle
from ui.control_center.controller import ControlHitMap
from ui.control_center.panels import ControlCenterPanelMixin
from ui.control_center.text_helpers import ControlCenterTextMixin
from ui.control_center.view_model import (
    DRONE_ROSTER,
    ROVER_ROSTER,
    ControlCenterViewModel,
)
from ui.control_center.widgets import ControlCenterWidgetMixin


class ControlCenterRenderer(
    ControlCenterPanelMixin,
    ControlCenterWidgetMixin,
    ControlCenterTextMixin,
):
    """Own frame-level drawing while helpers own widgets, panels, and text."""

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
        """Create surfaces, caches, and static text for one control panel."""
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
        # Static surfaces never change; dynamic cache entries are keyed by
        # changing values like timer text or agent status.
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

        # Draw order matters: header/tabs first, then the active tab content,
        # then blit the entire panel into the mission window.
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
        """Draw the static control-center title."""
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
        """Draw mission elapsed time, explored percentage, and heatmap toggle."""
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

    def _pre_render_statics(self) -> None:
        """Cache labels that do not change between frames."""
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
        # Roster names are static labels; live agent status is rendered through
        # the dynamic text cache in the panel mixin.
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
