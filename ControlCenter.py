"""Control center UI for mission status and runtime statistics.

Provides a compact, cached-rendering surface used by `MissionControl`
to display drone/rover health, elapsed time, and simple status lines.
"""

import time
from typing import Optional, Tuple, List, Any

import pygame
from asset_config.gameplay import Display
from asset_config.rendering import Colors, DroneColors, Fonts, RectHandle, RoverColors

class ControlCenter:
    """UI component that renders mission status, timers and agent stats."""

    def __init__(self, game: Any, num_drones: int) -> None:
        """Create control center UI state for the running `game`.

        Args:
            game: Owner `Game` instance (used to blit the control surface).
            num_drones: Number of drones to display in the status panel.
        """
        self.game = game
        self.tic = None
        self.explored_percent = 100 # TO BE CALCULATED OUTSIDE AND PASSED AS ARGUMENT

        # Get number of deployed drones and rovers
        self.num_drones = num_drones
        self.num_rovers = 1 + (4 % num_drones)

        # Calculate surface origin
        self.origin_x = Display.FULL_W - Display.LEGEND_WIDTH
        self.origin_y = 0
        self.origin   = (self.origin_x,self.origin_y)

        # Calculate surface mid points
        self.mid_x = self.origin_x + (Display.LEGEND_WIDTH / 2)
        self.mid_y = Display.FULL_H / 2

        # Define surface
        self.control_surf = pygame.Surface((Display.LEGEND_WIDTH, Display.FULL_H), pygame.SRCALPHA)
        self.control_surf.fill((*Colors.BLACK.value, 255))

        # Create dictionaries
        self.drone_dict()
        self.rover_dict()
        
        # Caches for fonts and pre-rendered static surfaces
        self._font_cache = {}
        self._static_surfaces = {}
        # Static text fragments (substring -> surface) to avoid re-rendering common labels
        self._static_fragments = {}
        self._pre_render_statics()
        # Cache for dynamic text surfaces: {key: {'value': tuple, 'time': float, 'surf': Surface}}
        self._dynamic_cache = {}
        self.drone_toggle_rects = {}
        self.heatmap_toggle_rect = None
        # Debug flags removed in production
        # Handle mapping to avoid repeated string comparisons
        self._handle_map = {
            'center': 'center',
            'midtop': 'midtop',
            'midright': 'midright',
            'midleft': 'midleft'
        }


# =============================================================================
# Utility methods (Dictionaries)
# =============================================================================

    def drone_dict(self):
        """Populate `self.drones` with default drone status entries."""
        self.drones = {
            'Blinky': {
                'id': 0,
                'color': DroneColors.RED.value,
                'battery': 10,
                'status': 'Ready'
            },
            'Pinky': {
                'id': 1,
                'color': DroneColors.PINK.value,
                'battery': 50,
                'status': 'Homing'
            },
            'Inky': {
                'id': 2,
                'color': DroneColors.L_BLUE.value,
                'battery': 100,
                'status': 'Charging'
            },
            'Clyde': {
                'id': 3,
                'color': DroneColors.ORANGE.value,
                'battery': 100,
                'status': 'Ready'
            },
            'Sue': {
                'id': 4,
                'color': DroneColors.PURPLE.value,
                'battery': 100,
                'status': 'Ready'
            },
            'Tim': {
                'id': 5,
                'color': DroneColors.BROWN.value,
                'battery': 100,
                'status': 'Ready'
            },
            'Funky': {
                'id': 6,
                'color': DroneColors.GREEN.value,
                'battery': 100,
                'status': 'Ready'
            },
            'Kinky': {
                'id': 7,
                'color': DroneColors.GOLD.value,
                'battery': 100,
                'status': 'Ready'
            }
        }

    # Create rover dictionary
    def rover_dict(self):
        """Populate `self.rovers` with default rover status entries."""
        self.rovers = {
            'Huey' : {
                'id': 0,
                'color': RoverColors.RED.value,
                'battery': 2400,
                'status': 'Ready'
            },
            'Dewey' : {
                'id': 1,
                'color': RoverColors.BLUE.value,
                'battery': 1400,
                'status': 'Updating'
            },
            'Louie' : {
                'id': 2,
                'color': RoverColors.GREEN.value,
                'battery': 240,
                'status': 'Ready'
            }
        }


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
            return '00:00'
        # Use integer seconds to avoid rounding artifacts during the first second
        elapsed = int(time.perf_counter() - self.tic)
        if elapsed < 0:
            elapsed = 0
        minutes, seconds = divmod(elapsed, 60)

        str_minutes = '0' + str(minutes) if (minutes < 10) else str(minutes)
        str_seconds = '0' + str(seconds) if (seconds < 10) else str(seconds)

        return str_minutes + ':' + str_seconds


# =============================================================================
# Drawing methods
# =============================================================================

    def draw_control_center(
        self,
        drone_objects: List[Any],
        rover_objects: Optional[List[Any]] = None,
        show_terrain_heatmap: bool = True,
        selected_drone_heatmap_id: Optional[int] = None
    ) -> None:
        """Render the entire control center onto `self.control_surf` and blit it."""
        # Clear control surface, draw content there, then blit once to window
        self.control_surf.fill((*Colors.BLACK.value, 255))
        self.drone_toggle_rects.clear()
        self.heatmap_toggle_rect = None

        if rover_objects is not None:
            for name, rover_info in self.rovers.items():
                rover_id = rover_info['id']
                if rover_id < len(rover_objects):
                    rover = rover_objects[rover_id]
                    rover_info['battery'] = rover.battery
                    rover_info['status'] = rover.status

        self._draw_title()
        self._draw_statistics(show_terrain_heatmap)
        self._draw_drone_section(drone_objects, selected_drone_heatmap_id)
        self._draw_rover_section()

        self.game.window.blit(self.control_surf, self.origin)
        # Debug rectangles removed
        

    def _draw_title(self) -> None:
        """Render the title at the top of the control panel."""
        surf = self._static_surfaces.get('title')
        if surf:
            # Ensure the title fits the legend: clamp or scale if necessary
            legend_w = Display.LEGEND_WIDTH
            max_w = max(legend_w - 16, 8)
            if surf.get_width() > max_w:
                scale = max_w / surf.get_width()
                new_w = int(surf.get_width() * scale)
                new_h = int(surf.get_height() * scale)
                surf = pygame.transform.smoothscale(surf, (new_w, new_h)).convert_alpha()

            rect = surf.get_rect()
            rect.centerx = legend_w // 2
            rect.centery = 70
            # If centering would push the left edge off-surface, clamp to a small margin
            if rect.left < 8:
                rect.left = 8
            self.control_surf.blit(surf, rect)
        

    def _draw_statistics(self, show_terrain_heatmap: bool) -> None:
        """Render timer and overall statistics in the control panel."""
        # Draw time (update at most once per second)
        met_texts = [('M.E.T.: ', Colors.GREY.value, 255),
                     (self.format_timer(), Colors.WHITE.value, 255)]
        met_surf = self._get_cached_text_surface('met', met_texts, 25, Fonts.BIG.value, ttl=1.0)
        if met_surf:
            self._blit_cached_surface(met_surf, self.origin_x, 120, RectHandle.MIDLEFT.value)

        # Draw explored map percentage (only when value changes)
        explored_texts = [('Explored: ', Colors.GREY.value, 255),
                          (f'{self.explored_percent}%', self.percent_color(self.explored_percent), 255)]
        explored_surf = self._get_cached_text_surface(f'explored_{self.explored_percent}', explored_texts, 25, Fonts.BIG.value)
        if explored_surf:
            self._blit_cached_surface(explored_surf, self.origin_x, 150, RectHandle.MIDLEFT.value)

        self._draw_heatmap_toggle(show_terrain_heatmap)


    def _draw_heatmap_toggle(self, enabled: bool) -> None:
        """Draw the global terrain heatmap toggle button."""
        rect = pygame.Rect(Display.LEGEND_WIDTH - 46, 138, 34, 24)
        self._draw_toggle_button(rect, 'H', enabled, Colors.EUCALYPTUS.value)
        self.heatmap_toggle_rect = rect.move(self.origin_x, self.origin_y)


    def _draw_drone_section(self, drone_objects: List[Any], selected_drone_heatmap_id: Optional[int]) -> None:
        """Render the drone section header and individual drone statuses."""
        # Draw subtitle
        surf = self._static_surfaces.get('drones')
        if surf:
            rect = surf.get_rect()
            rect.centerx = Display.LEGEND_WIDTH // 2
            rect.centery = 195
            self.control_surf.blit(surf, rect)

        for drone in self.drones:
            if self.drones[drone]['id'] < self.num_drones:
                drone_id = self.drones[drone]['id']
                self._draw_status(
                    drone,
                    drone_obj=drone_objects[drone_id],
                    selected_drone_heatmap_id=selected_drone_heatmap_id
                )
            else:
                self._draw_status(drone, deployed=False)


    def _draw_rover_section(self) -> None:
        """Render the rover section header and individual rover statuses."""
        # Draw subtitle
        surf = self._static_surfaces.get('rovers')
        if surf:
            rect = surf.get_rect()
            rect.centerx = Display.LEGEND_WIDTH // 2
            rect.centery = 725
            self.control_surf.blit(surf, rect)

        for rover in self.rovers:
            if self.rovers[rover]['id'] < self.num_rovers:
                self._draw_status(rover, rover=True)
            else:
                self._draw_status(rover, rover=True, deployed=False)
    

    def _draw_status(
        self,
        label: str,
        rover: bool = False,
        deployed: bool = True,
        drone_obj: Any = None,
        selected_drone_heatmap_id: Optional[int] = None
    ) -> None:
        """Render a single status line for a drone or rover identified by `label`."""
        # Get data
        if not rover:
            number = self.drones[label]['id']
            color = self.drones[label]['color']
            battery = self.drones[label]['battery']
            status = self.drones[label]['status']

            name_height = 230
            data_height = 260
            max_battery = 100
        else:
            number = self.rovers[label]['id']
            color = self.rovers[label]['color']
            battery = self.rovers[label]['battery']
            status = self.rovers[label]['status']

            name_height = 760
            data_height = 790
            max_battery = 2400

        # Blit label (pre-rendered)
        key = ('rover_' + label) if rover else ('drone_' + label)
        name_surf = self._static_surfaces.get(key)
        y_center = name_height + 60*number
        rect = name_surf.get_rect()
        rect.midleft = (8, y_center)
        self.control_surf.blit(name_surf, rect)

        if deployed and not rover and drone_obj is not None:
            self._draw_drone_toggles(number, y_center, drone_obj, selected_drone_heatmap_id)
        
        if deployed:
            # Define Status color
            match status:
                case 'Ready'|'Done':
                    status_color = Colors.GREEN.value
                case 'Updating'|'Advancing'|'Sharing'|'Charging':
                    status_color = Colors.YELLOW.value
                case 'Deployed'|'Homing':
                    status_color = Colors.WHITE.value
                case _:
                    status_color = Colors.RED.value
            
            # Define Battery color
            battery_color = self.percent_color(battery, max_battery)

            # Blit data (cache and only re-render when battery/status changes)
            key = f'status_{"rover_"+label if rover else label}_{battery}_{status}'
            data_surf = self._get_cached_status_surface(key, battery, status, battery_color, status_color, 25, Fonts.BIG.value, max_battery)
            if data_surf:
                self._blit_cached_surface(data_surf, self.origin_x, data_height + 60*number, RectHandle.MIDLEFT.value)
        else:
            # Blit 'N/A'
            na_surf = self._static_fragments['N/A']
            rect = na_surf.get_rect()
            rect.midleft = (8, data_height + 60*number)
            self.control_surf.blit(na_surf, rect)


    def _draw_drone_toggles(self, drone_id: int, y_center: int, drone_obj: Any, selected_drone_heatmap_id: Optional[int]) -> None:
        """Draw clickable path/vision/terrain toggle buttons for one drone row."""
        button_width = 34
        button_height = 24
        gap = 8
        start_x = Display.LEGEND_WIDTH - ((button_width * 3) + (gap * 2) + 12)
        top = y_center - (button_height // 2)

        path_rect = pygame.Rect(start_x, top, button_width, button_height)
        vision_rect = pygame.Rect(start_x + button_width + gap, top, button_width, button_height)
        terrain_rect = pygame.Rect(start_x + (button_width + gap) * 2, top, button_width, button_height)

        self._draw_toggle_button(path_rect, 'P', drone_obj.show_path, drone_obj.color)
        self._draw_toggle_button(vision_rect, 'V', drone_obj.show_vision, drone_obj.color)
        self._draw_toggle_button(terrain_rect, 'T', selected_drone_heatmap_id == drone_id, drone_obj.color)

        self.drone_toggle_rects[(drone_id, 'path')] = path_rect.move(self.origin_x, self.origin_y)
        self.drone_toggle_rects[(drone_id, 'vision')] = vision_rect.move(self.origin_x, self.origin_y)
        self.drone_toggle_rects[(drone_id, 'terrain')] = terrain_rect.move(self.origin_x, self.origin_y)


    def _draw_toggle_button(self, rect: 'pygame.Rect', label: str, enabled: bool, accent_color: Tuple[int, int, int]) -> None:
        """Draw a single toggle button in the control panel."""
        bg_color = accent_color if enabled else Colors.GREY.value
        text_color = Colors.BLACK.value if enabled else Colors.WHITE.value

        button_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(button_surf, (*bg_color, 128), button_surf.get_rect(), border_radius=6)
        pygame.draw.rect(button_surf, (*Colors.WHITE.value, 128), button_surf.get_rect(), width=1, border_radius=6)

        font = self._get_font(Fonts.BIG.value, 18)
        text_surf = font.render(label, True, text_color).convert_alpha()
        text_surf.set_alpha(128)
        text_rect = text_surf.get_rect(center=rect.center)
        button_surf.blit(text_surf, text_surf.get_rect(center=button_surf.get_rect().center))
        self.control_surf.blit(button_surf, rect.topleft)


    def handle_click(self, mouse_pos: Tuple[int, int], drone_objects: List[Any]) -> Optional[Tuple[str, Optional[int]]]:
        """Handle click events and return an action token with optional drone id.

        Returns:
            ('terrain_heatmap', None) when the global heatmap toggle is clicked,
            ('drone_heatmap', drone_id) when a per-drone heatmap toggle is clicked,
            ('drone_overlay', drone_id) when a drone path/vision toggle is clicked,
            None when no control was clicked.
        """
        if self.heatmap_toggle_rect is not None and self.heatmap_toggle_rect.collidepoint(mouse_pos):
            return ('terrain_heatmap', None)

        for (drone_id, overlay_type), rect in self.drone_toggle_rects.items():
            if rect.collidepoint(mouse_pos):
                drone = drone_objects[drone_id]
                if overlay_type == 'path':
                    drone.toggle_path()
                    return ('drone_overlay', drone_id)
                if overlay_type == 'vision':
                    drone.toggle_vision()
                    return ('drone_overlay', drone_id)
                else:
                    return ('drone_heatmap', drone_id)
        return None
            

    def draw_text(self, texts: List[Tuple[str, Tuple[int, int, int], int]], size: int, x: int, y: int, font, handle) -> None:
        """Compose and blit a composed text surface at absolute `x,y` using `handle`."""
        # Compose a single surface for these texts (this applies alpha during composition)
        surf = self._compose_text_surface(texts, size, font)
        # Compute local coordinates and blit using cached handle mapping
        x_local = x - self.origin_x
        rect = surf.get_rect()
        handle_l = str(handle).lower() if handle is not None else ''
        attr = self._handle_map.get(handle_l, 'midleft')
        setattr(rect, attr, (int(x_local), y))
        self.control_surf.blit(surf, rect)


    def _compose_text_surface(self, texts: List[Tuple[str, Tuple[int, int, int], int]], size: int, font) -> 'pygame.Surface':
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


    def _get_cached_text_surface(self, key: str, texts: List[Tuple[str, Tuple[int, int, int], int]], size: int, font, ttl: float = None) -> Optional['pygame.Surface']:
        """Return cached composed surface for texts; re-render when value changes or ttl expires."""
        now = time.perf_counter()
        value = tuple(t[0] for t in texts)
        entry = self._dynamic_cache.get(key)
        if entry:
            if entry.get('value') == value:
                if ttl is not None and (now - entry.get('time', 0)) < ttl:
                    return entry['surf']
            # value changed -> re-render
        # Render new surface
        surf = self._compose_text_surface(texts, size, font)
        self._dynamic_cache[key] = {'value': value, 'time': now, 'surf': surf}
        return surf


    def _get_cached_status_surface(self, key: str, battery: int, status: str, battery_color: Tuple[int, int, int], status_color: Tuple[int, int, int], size: int, font, max_battery: int = 100) -> 'pygame.Surface':
        """Compose and cache a status surface with fixed-width battery column."""
        now = time.perf_counter()
        value = (str(battery), status)
        entry = self._dynamic_cache.get(key)
        if entry and entry.get('value') == value:
            return entry['surf']

        # Prepare font and individual surfaces
        font_obj = self._get_font(font, size)
        # Battery surface
        battery_text = f'{battery}%'
        bat_s = font_obj.render(battery_text, True, battery_color).convert_alpha()
        bat_s.set_alpha(128)

        # Determine fixed battery column width using a 5-digit standard '00000%'
        sample_text = '00000%'
        sample = font_obj.render(sample_text, True, battery_color).convert_alpha()
        col_w = sample.get_width()

        # Separator
        sep_s = font_obj.render('|', True, Colors.WHITE.value).convert_alpha()
        sep_s.set_alpha(128)

        # Status surface
        stat_s = font_obj.render(status, True, status_color).convert_alpha()
        stat_s.set_alpha(128)

        # Build composed surface: battery column (col_w), gap, separator, gap, status
        gap = 8
        total_w = col_w + gap + sep_s.get_width() + gap + stat_s.get_width()
        max_h = max(bat_s.get_height(), sep_s.get_height(), stat_s.get_height())
        surf = pygame.Surface((total_w, max_h), pygame.SRCALPHA)

        # Blit battery right-aligned inside column so digits line up
        bat_x = col_w - bat_s.get_width()
        bat_y = (max_h - bat_s.get_height()) // 2
        surf.blit(bat_s, (bat_x, bat_y))

        # Blit separator at column end + gap
        sep_x = col_w + gap
        sep_y = (max_h - sep_s.get_height()) // 2
        surf.blit(sep_s, (sep_x, sep_y))

        # Blit status after separator + gap
        stat_x = sep_x + sep_s.get_width() + gap
        stat_y = (max_h - stat_s.get_height()) // 2
        surf.blit(stat_s, (stat_x, stat_y))

        self._dynamic_cache[key] = {'value': value, 'time': now, 'surf': surf}
        return surf


    def _blit_cached_surface(self, surf: 'pygame.Surface', x: int, y: int, handle: Any) -> None:
        """Blit a prepared surface onto `control_surf` using handle and absolute x,y."""
        x_local = x - self.origin_x
        rect = surf.get_rect()
        handle_l = str(handle).lower() if handle is not None else ''
        attr = self._handle_map.get(handle_l, 'midleft')
        setattr(rect, attr, (int(x_local), y))
        self.control_surf.blit(surf, rect)


    def _get_font(self, font_path: Any, size: int) -> 'pygame.font.Font':
        key = (font_path, size)
        f = self._font_cache.get(key)
        if f is None:
            f = pygame.font.Font(font_path, size)
            self._font_cache[key] = f
        return f


    def _pre_render_statics(self) -> None:
        """Prepare pre-rendered static surfaces used by the control panel."""
        # Pre-render title and section labels
        title_font = self._get_font(Fonts.BIG.value, 35)
        self._static_surfaces['title'] = title_font.render('Control Center', True, Colors.RED.value).convert_alpha()
        sub_font = self._get_font(Fonts.BIG.value, 30)
        self._static_surfaces['drones'] = sub_font.render('Drones', True, Colors.EUCALYPTUS.value).convert_alpha()
        self._static_surfaces['rovers'] = sub_font.render('Rovers', True, Colors.EUCALYPTUS.value).convert_alpha()

        # Pre-render drone and rover name labels
        name_font = self._get_font(Fonts.BIG.value, 25)
        for name, info in self.drones.items():
            surf = name_font.render(name, True, info['color']).convert_alpha()
            self._static_surfaces[('drone_' + name)] = surf
        for name, info in self.rovers.items():
            surf = name_font.render(name, True, info['color']).convert_alpha()
            self._static_surfaces[('rover_' + name)] = surf
        # Pre-render small static fragments used in dynamic lines
        small_font = self._get_font(Fonts.BIG.value, 25)
        self._static_fragments['M.E.T.:           '] = small_font.render('M.E.T.:           ', True, Colors.GREY.value).convert_alpha()
        self._static_fragments['Explored:     '] = small_font.render('Explored:     ', True, Colors.GREY.value).convert_alpha()
        self._static_fragments['  |  '] = small_font.render('  |  ', True, Colors.WHITE.value).convert_alpha()
        s = small_font.render('N/A', True, Colors.GREY.value).convert_alpha()
        s.set_alpha(128)
        self._static_fragments['N/A'] = s
    

    def percent_color(self, val: int, max_val: int = 100) -> Tuple[int, int, int]:
        """Return a color tuple representing a percentage (red/yellow/green)."""
        if val < max_val*20/100:
            return Colors.RED.value
        elif val < max_val*80/100:
            return Colors.YELLOW.value
        else:
            return Colors.GREEN.value
