"""Control center UI for mission status and runtime statistics.

Provides a compact, cached-rendering surface used by `MissionControl`
to display drone/rover health, elapsed time, and simple status lines.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any

import pygame
from asset_config.gameplay import Display
from asset_config.rendering import Colors, DroneColors, Fonts, RectHandle, RoverColors
from asset_config.media import Images
from ControlCenterRenderer import ControlCenterRenderer


@dataclass
class ControlCenterTabState:
    """Small state holder for the visible tab and its hit areas."""

    active_tab: str = "drones"
    rects: dict[str, "pygame.Rect"] = field(default_factory=dict)

    def clear(self) -> None:
        self.rects.clear()

    def register(self, tab_name: str, rect: "pygame.Rect") -> None:
        self.rects[tab_name] = rect

    def hit_test(self, mouse_pos: Tuple[int, int]) -> Optional[str]:
        for tab_name, rect in self.rects.items():
            if rect.collidepoint(mouse_pos):
                self.active_tab = tab_name
                return tab_name
        return None


@dataclass
class ControlCenterRenderState:
    """Cached render data and assets used by ControlCenter."""

    font_cache: dict[Tuple[Any, int], "pygame.font.Font"] = field(default_factory=dict)
    static_surfaces: dict[Any, "pygame.Surface"] = field(default_factory=dict)
    static_fragments: dict[str, "pygame.Surface"] = field(default_factory=dict)
    dynamic_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    drone_toggle_rects: dict[Tuple[int, str], "pygame.Rect"] = field(
        default_factory=dict
    )
    heatmap_toggle_rect: Optional["pygame.Rect"] = None
    tab_sprites: dict[str, "pygame.Surface"] = field(default_factory=dict)


class ControlCenter:
    """UI component that renders mission status, timers and agent stats."""

    TAB_ORDER = ("drones", "rovers", "debug", "system")
    DEFAULT_DRONES = (
        ("Blinky", 0, DroneColors.RED.value, 10, "Ready"),
        ("Pinky", 1, DroneColors.PINK.value, 50, "Homing"),
        ("Inky", 2, DroneColors.L_BLUE.value, 100, "Charging"),
        ("Clyde", 3, DroneColors.ORANGE.value, 100, "Ready"),
        ("Sue", 4, DroneColors.PURPLE.value, 100, "Ready"),
        ("Tim", 5, DroneColors.BROWN.value, 100, "Ready"),
        ("Funky", 6, DroneColors.GREEN.value, 100, "Ready"),
        ("Kinky", 7, DroneColors.GOLD.value, 100, "Ready"),
    )
    DEFAULT_ROVERS = (
        ("Huey", 0, RoverColors.RED.value, 2400, "Ready"),
        ("Dewey", 1, RoverColors.BLUE.value, 1400, "Updating"),
        ("Louie", 2, RoverColors.GREEN.value, 240, "Ready"),
    )
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

    def __init__(self, game: Any, num_drones: int) -> None:
        """Create control center UI state for the running `game`.

        Args:
            game: Owner `Game` instance (used to blit the control surface).
            num_drones: Number of drones to display in the status panel.
        """
        self.game = game
        self.tic = None
        self.explored_percent = 100

        # Get number of deployed drones and rovers
        self.num_drones = num_drones
        self.num_rovers = 1 + (4 % num_drones)

        self._configure_geometry()
        self._initialize_agent_defaults()
        self._initialize_render_state()
        self._initialize_tab_state()
        self._initialize_handle_map()

    # =============================================================================
    # Utility methods (Dictionaries)
    # =============================================================================

    def drone_dict(self):
        """Populate `self.drones` with default drone status entries."""
        self.drones = {
            name: {
                "id": agent_id,
                "color": color,
                "battery": battery,
                "status": status,
            }
            for name, agent_id, color, battery, status in self.DEFAULT_DRONES
        }

    # Create rover dictionary
    def rover_dict(self):
        """Populate `self.rovers` with default rover status entries."""
        self.rovers = {
            name: {
                "id": agent_id,
                "color": color,
                "battery": battery,
                "status": status,
            }
            for name, agent_id, color, battery, status in self.DEFAULT_ROVERS
        }

    def _configure_geometry(self) -> None:
        """Compute the base geometry and control surface."""
        self.origin_x = Display.FULL_W - Display.LEGEND_WIDTH
        self.origin_y = 0
        self.origin = (self.origin_x, self.origin_y)
        self.mid_x = self.origin_x + (Display.LEGEND_WIDTH / 2)
        self.mid_y = Display.FULL_H / 2
        self.control_surf = pygame.Surface(
            (Display.LEGEND_WIDTH, Display.FULL_H), pygame.SRCALPHA
        )
        self.control_surf.fill((*Colors.BLACK.value, 255))

    def _initialize_agent_defaults(self) -> None:
        """Populate the built-in drone and rover dictionaries."""
        self.drone_dict()
        self.rover_dict()

    def _initialize_render_state(self) -> None:
        """Create font caches, rendered statics, and image caches."""
        self._render_state = ControlCenterRenderState()
        # Renderer orchestrates drawing using the component's helpers
        self._renderer = ControlCenterRenderer()
        self._pre_render_statics()
        self._load_tab_sprites()

    def _initialize_tab_state(self) -> None:
        """Create the tab state object used by drawing and hit testing."""
        self._tab_state = ControlCenterTabState()

    def _initialize_handle_map(self) -> None:
        """Create a small cache for text anchor handling."""
        self._handle_map = {
            "center": "center",
            "midtop": "midtop",
            "midright": "midright",
            "midleft": "midleft",
        }

    @property
    def active_tab(self) -> str:
        return self._tab_state.active_tab

    @active_tab.setter
    def active_tab(self, value: str) -> None:
        self._tab_state.active_tab = value

    @property
    def tab_rects(self) -> dict[str, "pygame.Rect"]:
        return self._tab_state.rects

    @property
    def _font_cache(self) -> dict[Tuple[Any, int], "pygame.font.Font"]:
        return self._render_state.font_cache

    @_font_cache.setter
    def _font_cache(self, value: dict[Tuple[Any, int], "pygame.font.Font"]) -> None:
        self._render_state.font_cache = value

    @property
    def _static_surfaces(self) -> dict[Any, "pygame.Surface"]:
        return self._render_state.static_surfaces

    @_static_surfaces.setter
    def _static_surfaces(self, value: dict[Any, "pygame.Surface"]) -> None:
        self._render_state.static_surfaces = value

    @property
    def _static_fragments(self) -> dict[str, "pygame.Surface"]:
        return self._render_state.static_fragments

    @_static_fragments.setter
    def _static_fragments(self, value: dict[str, "pygame.Surface"]) -> None:
        self._render_state.static_fragments = value

    @property
    def _dynamic_cache(self) -> dict[str, dict[str, Any]]:
        return self._render_state.dynamic_cache

    @_dynamic_cache.setter
    def _dynamic_cache(self, value: dict[str, dict[str, Any]]) -> None:
        self._render_state.dynamic_cache = value

    @property
    def drone_toggle_rects(self) -> dict[Tuple[int, str], "pygame.Rect"]:
        return self._render_state.drone_toggle_rects

    @drone_toggle_rects.setter
    def drone_toggle_rects(self, value: dict[Tuple[int, str], "pygame.Rect"]) -> None:
        self._render_state.drone_toggle_rects = value

    @property
    def heatmap_toggle_rect(self) -> Optional["pygame.Rect"]:
        return self._render_state.heatmap_toggle_rect

    @heatmap_toggle_rect.setter
    def heatmap_toggle_rect(self, value: Optional["pygame.Rect"]) -> None:
        self._render_state.heatmap_toggle_rect = value

    @property
    def _tab_sprites(self) -> dict[str, "pygame.Surface"]:
        return self._render_state.tab_sprites

    @_tab_sprites.setter
    def _tab_sprites(self, value: dict[str, "pygame.Surface"]) -> None:
        self._render_state.tab_sprites = value

    def _load_tab_sprites(self) -> None:
        """Load all tab sprites without letting one failure disable the others."""
        self._load_tab_sprite("drones", Images.DRONE.value, (28, 28))
        self._load_tab_sprite("rovers", Images.ROVER.value, (28, 28))
        self._load_tab_sprite("debug", Images.DEBUG_ICON.value, (34, 34))
        self._load_tab_sprite("system", Images.SYSTEM_ICON.value, (36, 36))

    # =============================================================================
    # Timer methods
    # =============================================================================

    def start_timer(self) -> None:
        self.tic = time.perf_counter()

    def format_timer(self) -> str:
        """
        Format the elapsed time as a string in MM:SS format.
        Returns:
            str: A string representing the elapsed time in the format "MM:SS".
                 Returns "00:00" if the timer has not been started yet.
                 Minutes and seconds are zero-padded to always show two digits.
        Example:
            >>> # Timer not started
            >>> format_timer()
            '00:00'
            >>> # After 65 seconds
            >>> format_timer()
            '01:05'
        """
        # If timer not started yet, show 00:00
        if not self.tic:
            return "00:00"
        # Use integer seconds to avoid rounding artifacts during the first second
        elapsed = int(time.perf_counter() - self.tic)
        if elapsed < 0:
            elapsed = 0
        minutes, seconds = divmod(elapsed, 60)

        str_minutes = "0" + str(minutes) if (minutes < 10) else str(minutes)
        str_seconds = "0" + str(seconds) if (seconds < 10) else str(seconds)

        return str_minutes + ":" + str_seconds

    # =============================================================================
    # Drawing methods
    # =============================================================================

    def draw_control_center(
        self,
        drone_objects: List[Any],
        rover_objects: Optional[List[Any]] = None,
        show_terrain_heatmap: bool = True,
        selected_drone_heatmap_id: Optional[int] = None,
        debug_lines: Optional[List[str]] = None,
    ) -> None:
        """Render the entire control center onto `self.control_surf` and blit it."""
        # Delegate drawing orchestration to the renderer
        self._renderer.render(
            self,
            drone_objects,
            rover_objects=rover_objects,
            show_terrain_heatmap=show_terrain_heatmap,
            selected_drone_heatmap_id=selected_drone_heatmap_id,
            debug_lines=debug_lines,
        )

    def _wrap_text_surfaces(
        self, text: str, font_obj: "pygame.font.Font", max_w: int
    ) -> List["pygame.Surface"]:
        """Wrap a long text string into multiple surfaces that fit within `max_w` pixels."""
        words = text.split()
        if not words:
            return [font_obj.render("", True, Colors.WHITE.value).convert_alpha()]

        lines = []
        cur = words[0]
        for w in words[1:]:
            test = cur + " " + w
            test_surf = font_obj.render(test, True, Colors.WHITE.value)
            if test_surf.get_width() <= max_w:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)

        return [
            font_obj.render(l, True, Colors.WHITE.value).convert_alpha() for l in lines
        ]

    def _draw_label_value_entry(
        self,
        prefix: str,
        idx: int,
        label: str,
        val: str,
        font_obj: "pygame.font.Font",
        max_w: int,
        ypos: int,
        line_gap: int,
        wrap_gap: int,
    ) -> Tuple[int, int, bool]:
        """Draw a label/value row and return updated cursor state."""
        label_s = font_obj.render(label + " ", True, Colors.GREY.value).convert_alpha()
        val_s = font_obj.render(val, True, Colors.WHITE.value).convert_alpha()

        if label_s.get_width() + val_s.get_width() <= max_w:
            total_w = label_s.get_width() + val_s.get_width()
            h = max(label_s.get_height(), val_s.get_height())
            surf = pygame.Surface((total_w, h), pygame.SRCALPHA)
            surf.blit(label_s, (0, (h - label_s.get_height()) // 2))
            surf.blit(val_s, (label_s.get_width(), (h - val_s.get_height()) // 2))
            key = f"{prefix}_{idx}_{label}_{val}"
            self._dynamic_cache.setdefault(
                key, {"value": None, "time": 0, "surf": surf}
            )
            self._blit_cached_surface(
                surf, self.origin_x, ypos, RectHandle.MIDLEFT.value
            )
            return ypos + h + line_gap, idx + 1, False

        self._blit_cached_surface(
            label_s, self.origin_x, ypos, RectHandle.MIDLEFT.value
        )
        ypos += label_s.get_height() + wrap_gap
        for vs in self._wrap_text_surfaces(val, font_obj, max_w):
            key = f"{prefix}_{idx}_{label}_{val[:40]}"
            self._dynamic_cache.setdefault(key, {"value": None, "time": 0, "surf": vs})
            self._blit_cached_surface(vs, self.origin_x, ypos, RectHandle.MIDLEFT.value)
            ypos += vs.get_height() + wrap_gap
            idx += 1
            if ypos > Display.FULL_H - self.CONTENT_BOTTOM_MARGIN:
                return ypos, idx, True
        return ypos, idx, False

    def _draw_wrapped_text_lines(
        self,
        prefix: str,
        idx: int,
        text: str,
        font_obj: "pygame.font.Font",
        max_w: int,
        ypos: int,
        line_gap: int,
    ) -> Tuple[int, int, bool]:
        """Draw wrapped plain text lines and return updated cursor state."""
        for surf in self._wrap_text_surfaces(text, font_obj, max_w):
            key = f"{prefix}_{idx}_{text[:40]}"
            self._dynamic_cache.setdefault(
                key, {"value": None, "time": 0, "surf": surf}
            )
            self._blit_cached_surface(
                surf, self.origin_x, ypos, RectHandle.MIDLEFT.value
            )
            ypos += surf.get_height() + line_gap
            idx += 1
            if ypos > Display.FULL_H - self.CONTENT_BOTTOM_MARGIN:
                return ypos, idx, True
        return ypos, idx, False

    def _draw_drone_toggles(
        self,
        drone_id: int,
        y_center: int,
        drone_obj: Any,
        selected_drone_heatmap_id: Optional[int],
    ) -> None:
        """Draw clickable path/vision/terrain toggle buttons for one drone row."""
        button_width = 34
        button_height = 24
        gap = 8
        start_x = Display.LEGEND_WIDTH - ((button_width * 3) + (gap * 2) + 12)
        top = y_center - (button_height // 2)

        path_rect = pygame.Rect(start_x, top, button_width, button_height)
        vision_rect = pygame.Rect(
            start_x + button_width + gap, top, button_width, button_height
        )
        terrain_rect = pygame.Rect(
            start_x + (button_width + gap) * 2, top, button_width, button_height
        )

        self._renderer.draw_toggle_button(
            self, path_rect, "P", drone_obj.show_path, drone_obj.color
        )
        self._renderer.draw_toggle_button(
            self, vision_rect, "V", drone_obj.show_vision, drone_obj.color
        )
        self._renderer.draw_toggle_button(
            self,
            terrain_rect,
            "T",
            selected_drone_heatmap_id == drone_id,
            drone_obj.color,
        )

        self.drone_toggle_rects[(drone_id, "path")] = path_rect.move(
            self.origin_x, self.origin_y
        )
        self.drone_toggle_rects[(drone_id, "vision")] = vision_rect.move(
            self.origin_x, self.origin_y
        )
        self.drone_toggle_rects[(drone_id, "terrain")] = terrain_rect.move(
            self.origin_x, self.origin_y
        )

    def handle_click(
        self, mouse_pos: Tuple[int, int], drone_objects: List[Any]
    ) -> Optional[Tuple[str, Optional[int]]]:
        """Handle click events and return an action token with optional drone id.

        Returns:
            ('terrain_heatmap', None) when the global heatmap toggle is clicked,
            ('drone_heatmap', drone_id) when a per-drone heatmap toggle is clicked,
            ('drone_overlay', drone_id) when a drone path/vision toggle is clicked,
            None when no control was clicked.
        """
        if (
            self.heatmap_toggle_rect is not None
            and self.heatmap_toggle_rect.collidepoint(mouse_pos)
        ):
            return ("terrain_heatmap", None)

        if self._tab_state.hit_test(mouse_pos) is not None:
            return ("control_tab", None)

        for (drone_id, overlay_type), rect in self.drone_toggle_rects.items():
            if rect.collidepoint(mouse_pos):
                drone = drone_objects[drone_id]
                if overlay_type == "path":
                    drone.toggle_path()
                    return ("drone_overlay", drone_id)
                if overlay_type == "vision":
                    drone.toggle_vision()
                    return ("drone_overlay", drone_id)
                else:
                    return ("drone_heatmap", drone_id)
        return None

    def _compose_text_surface(
        self, texts: List[Tuple[str, Tuple[int, int, int], int]], size: int, font
    ) -> "pygame.Surface":
        """Compose and return a Surface for the given text fragments.

        `texts` is a list of tuples `(substring, color, alpha)`.
        """
        font_obj = self._get_font(font, size)
        parts = []
        total_w = 0
        max_h = 0
        for substring, color, alpha in texts:
            # Reuse pre-rendered fragment if available (exact-match)
            s = self._static_fragments.get(substring)
            if s is None:
                s = font_obj.render(substring, True, color).convert_alpha()
            else:
                # avoid mutating shared static fragments when applying alpha
                s = s.copy()
            if alpha != 255:
                s.set_alpha(alpha)
            parts.append(s)
            w, h = s.get_size()
            total_w += w
            max_h = max(max_h, h)

        surf = pygame.Surface((total_w, max_h), pygame.SRCALPHA)
        x = 0
        for p in parts:
            surf.blit(p, (x, (max_h - p.get_height()) // 2))
            x += p.get_width()
        return surf

    def _get_cached_text_surface(
        self,
        key: str,
        texts: List[Tuple[str, Tuple[int, int, int], int]],
        size: int,
        font,
        ttl: float = None,
    ) -> Optional["pygame.Surface"]:
        """Return cached composed surface for texts; re-render when value changes or ttl expires."""
        now = time.perf_counter()
        value = tuple(t[0] for t in texts)
        entry = self._dynamic_cache.get(key)
        if entry:
            if entry.get("value") == value:
                if ttl is not None and (now - entry.get("time", 0)) < ttl:
                    return entry["surf"]
            # value changed -> re-render
        # Render new surface
        surf = self._compose_text_surface(texts, size, font)
        self._dynamic_cache[key] = {"value": value, "time": now, "surf": surf}
        return surf

    def _get_cached_status_surface(
        self,
        key: str,
        battery: int,
        status: str,
        battery_color: Tuple[int, int, int],
        status_color: Tuple[int, int, int],
        size: int,
        font,
        max_battery: int = 100,
    ) -> "pygame.Surface":
        """Compose and cache a status surface with fixed-width battery column."""
        now = time.perf_counter()
        value = (str(battery), status)
        entry = self._dynamic_cache.get(key)
        if entry and entry.get("value") == value:
            return entry["surf"]

        font_obj = self._get_font(font, size)
        surf = self._compose_status_surface(
            font_obj, battery, status, battery_color, status_color
        )
        self._dynamic_cache[key] = {"value": value, "time": now, "surf": surf}
        return surf

    def _compose_status_surface(
        self,
        font_obj: "pygame.font.Font",
        battery: int,
        status: str,
        battery_color: Tuple[int, int, int],
        status_color: Tuple[int, int, int],
    ) -> "pygame.Surface":
        """Build a status surface, using a single line when it fits and wrapping otherwise."""
        bat_s, sep_s, stat_s, col_w, gap = self._build_status_surfaces(
            font_obj, battery, status, battery_color, status_color
        )
        sep_w = sep_s.get_width()
        max_allowed = self._status_inline_width(col_w, sep_w, gap)

        if stat_s.get_width() <= max_allowed or max_allowed <= 32:
            return self._compose_inline_status_surface(bat_s, sep_s, stat_s, col_w, gap)

        status_lines = self._wrap_text_surfaces(
            status, font_obj, Display.LEGEND_WIDTH - 16
        )
        return self._compose_wrapped_status_surface(
            bat_s, sep_s, status_lines, col_w, gap
        )

    def _build_status_surfaces(
        self,
        font_obj: "pygame.font.Font",
        battery: int,
        status: str,
        battery_color: Tuple[int, int, int],
        status_color: Tuple[int, int, int],
    ) -> Tuple["pygame.Surface", "pygame.Surface", "pygame.Surface", int, int]:
        """Render the atomic surfaces and return their shared measurements."""
        battery_text = f"{battery}%"
        bat_s = font_obj.render(battery_text, True, battery_color).convert_alpha()
        bat_s.set_alpha(128)

        sample = font_obj.render("00000%", True, battery_color).convert_alpha()
        col_w = sample.get_width()

        sep_s = font_obj.render("|", True, Colors.WHITE.value).convert_alpha()
        sep_s.set_alpha(128)

        stat_s = font_obj.render(status, True, status_color).convert_alpha()
        stat_s.set_alpha(128)

        return bat_s, sep_s, stat_s, col_w, 8

    def _status_inline_width(self, col_w: int, sep_w: int, gap: int) -> int:
        """Return the max width available for the inline status text."""
        return Display.LEGEND_WIDTH - (col_w + gap + sep_w + gap + 24)

    def _compose_inline_status_surface(
        self,
        bat_s: "pygame.Surface",
        sep_s: "pygame.Surface",
        stat_s: "pygame.Surface",
        col_w: int,
        gap: int,
    ) -> "pygame.Surface":
        """Compose the compact one-line status layout."""
        sep_w = sep_s.get_width()
        total_w = col_w + gap + sep_w + gap + stat_s.get_width()
        max_h = max(bat_s.get_height(), sep_s.get_height(), stat_s.get_height())
        surf = pygame.Surface((total_w, max_h), pygame.SRCALPHA)

        bat_x = col_w - bat_s.get_width()
        bat_y = (max_h - bat_s.get_height()) // 2
        surf.blit(bat_s, (bat_x, bat_y))

        sep_x = col_w + gap
        sep_y = (max_h - sep_s.get_height()) // 2
        surf.blit(sep_s, (sep_x, sep_y))

        stat_x = sep_x + sep_w + gap
        stat_y = (max_h - stat_s.get_height()) // 2
        surf.blit(stat_s, (stat_x, stat_y))
        return surf

    def _compose_wrapped_status_surface(
        self,
        bat_s: "pygame.Surface",
        sep_s: "pygame.Surface",
        status_lines: List["pygame.Surface"],
        col_w: int,
        gap: int,
    ) -> "pygame.Surface":
        """Compose the two-line-or-more status layout when the value does not fit inline."""
        sep_w = sep_s.get_width()
        first_h = max(bat_s.get_height(), sep_s.get_height())
        line_gap = 6
        status_h = sum(s.get_height() + line_gap for s in status_lines) - line_gap
        total_w = min(
            Display.LEGEND_WIDTH - 16,
            max(
                col_w + gap + sep_w,
                max((s.get_width() for s in status_lines), default=0)
                + col_w
                + gap
                + sep_w,
            ),
        )
        total_h = first_h + line_gap + status_h
        surf = pygame.Surface((total_w, total_h), pygame.SRCALPHA)

        bat_x = col_w - bat_s.get_width()
        bat_y = (first_h - bat_s.get_height()) // 2
        surf.blit(bat_s, (bat_x, bat_y))

        sep_x = col_w + gap
        sep_y = (first_h - sep_s.get_height()) // 2
        surf.blit(sep_s, (sep_x, sep_y))

        status_x = col_w + gap + sep_w + gap
        y = first_h + line_gap
        for s in status_lines:
            surf.blit(s, (status_x, y))
            y += s.get_height() + line_gap

        return surf

    def _blit_cached_surface(
        self, surf: "pygame.Surface", x: int, y: int, handle: Any
    ) -> None:
        """Blit a prepared surface onto `control_surf` using handle and absolute x,y."""
        x_local = x - self.origin_x
        rect = surf.get_rect()
        handle_l = str(handle).lower() if handle is not None else ""
        attr = self._handle_map.get(handle_l, "midleft")
        setattr(rect, attr, (int(x_local), y))
        self.control_surf.blit(surf, rect)

    def _get_font(self, font_path: Any, size: int) -> "pygame.font.Font":
        key = (font_path, size)
        f = self._font_cache.get(key)
        if f is None:
            f = pygame.font.Font(font_path, size)
            self._font_cache[key] = f
        return f

    def _load_tab_sprite(
        self, tab_name: str, image_path: Any, size: Tuple[int, int]
    ) -> None:
        """Load one tab sprite without failing the other icons."""
        try:
            image = pygame.image.load(str(image_path)).convert_alpha()
        except Exception:
            return
        self._tab_sprites[tab_name] = pygame.transform.smoothscale(image, size)

    def _pre_render_statics(self) -> None:
        """Prepare pre-rendered static surfaces used by the control panel."""
        # Pre-render title and section labels
        title_font = self._get_font(Fonts.BIG.value, 35)
        self._static_surfaces["title"] = title_font.render(
            "Control Center", True, Colors.RED.value
        ).convert_alpha()
        sub_font = self._get_font(Fonts.BIG.value, 30)
        self._static_surfaces["drones"] = sub_font.render(
            "Drones", True, Colors.EUCALYPTUS.value
        ).convert_alpha()
        self._static_surfaces["rovers"] = sub_font.render(
            "Rovers", True, Colors.EUCALYPTUS.value
        ).convert_alpha()
        # Debug subtitle uses the same style as other section subtitles for consistency
        self._static_surfaces["debug"] = sub_font.render(
            "Debug", True, Colors.EUCALYPTUS.value
        ).convert_alpha()
        self._static_surfaces["system"] = sub_font.render(
            "System", True, Colors.EUCALYPTUS.value
        ).convert_alpha()

        # Pre-render drone and rover name labels
        name_font = self._get_font(Fonts.BIG.value, 25)
        for name, info in self.drones.items():
            surf = name_font.render(name, True, info["color"]).convert_alpha()
            self._static_surfaces[("drone_" + name)] = surf
        for name, info in self.rovers.items():
            surf = name_font.render(name, True, info["color"]).convert_alpha()
            self._static_surfaces[("rover_" + name)] = surf
        # Pre-render small static fragments used in dynamic lines
        small_font = self._get_font(Fonts.BIG.value, 25)
        self._static_fragments["M.E.T.:           "] = small_font.render(
            "M.E.T.:           ", True, Colors.GREY.value
        ).convert_alpha()
        self._static_fragments["Explored:     "] = small_font.render(
            "Explored:     ", True, Colors.GREY.value
        ).convert_alpha()
        self._static_fragments["  |  "] = small_font.render(
            "  |  ", True, Colors.WHITE.value
        ).convert_alpha()
        s = small_font.render("N/A", True, Colors.GREY.value).convert_alpha()
        s.set_alpha(128)
        self._static_fragments["N/A"] = s

    def percent_color(self, val: int, max_val: int = 100) -> Tuple[int, int, int]:
        """Return a color tuple representing a percentage (red/yellow/green)."""
        if val < max_val * 20 / 100:
            return Colors.RED.value
        elif val < max_val * 80 / 100:
            return Colors.YELLOW.value
        else:
            return Colors.GREEN.value
