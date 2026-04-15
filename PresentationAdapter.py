"""Presentation layer adapter for UI state and event dispatch.

Separates UI concerns from simulation logic by managing heatmap visibility,
selected drone state, and click-event dispatch to control-center views.
"""

from typing import Optional, Tuple, List, Any

import pygame
import numpy as np


class PresentationAdapter:
    """Isolates UI state and click dispatch from simulation orchestration.

    Responsibilities:
    - Manage heatmap visibility flags (global terrain and per-drone)
    - Track which drone's heatmap is selected for detail view
    - Dispatch click events from control-center to internal state updates
    - Provide clean interface for MissionControl to render UI without owning state
    """

    def __init__(self, map_w: int, map_h: int) -> None:
        """Initialize presentation adapter with map dimensions.

        Args:
            map_w: Map width in pixels (for heatmap surface allocation).
            map_h: Map height in pixels (for heatmap surface allocation).
        """
        # Heatmap visibility state
        self.show_terrain_heatmap = False
        self.selected_drone_heatmap_id: Optional[int] = None

        # Heatmap rendering state
        self.terrain_heatmap_dirty = True
        self.terrain_heatmap_surf = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
        self.last_heatmap_refresh = 0.0
        self.heatmap_refresh_interval = 0.25

    def toggle_terrain_heatmap(self) -> None:
        """Toggle global terrain heatmap visibility on/off."""
        self.show_terrain_heatmap = not self.show_terrain_heatmap
        self.terrain_heatmap_dirty = True

    def toggle_drone_heatmap(self, drone_id: int) -> None:
        """Toggle per-drone heatmap: show if different drone, hide if same drone."""
        if self.selected_drone_heatmap_id == drone_id:
            # Clicking same drone toggles it off
            self.selected_drone_heatmap_id = None
        else:
            # Clicking different drone selects it
            self.selected_drone_heatmap_id = drone_id
        self.terrain_heatmap_dirty = True

    def handle_click(
        self,
        mouse_pos: Tuple[int, int],
        control_center: Any,
        drone_objects: List[Any]
    ) -> None:
        """Dispatch click event from control-center and update internal UI state.

        Args:
            mouse_pos: (x, y) pixel coordinates of mouse click.
            control_center: ControlCenter instance with drawable rectangles.
            drone_objects: List of drone objects for overlay toggling.
        """
        click_result = control_center.handle_click(mouse_pos, drone_objects)
        if click_result is None:
            return

        action, drone_id = click_result
        if action == 'terrain_heatmap':
            self.toggle_terrain_heatmap()
        elif action == 'drone_heatmap' and drone_id is not None:
            self.toggle_drone_heatmap(drone_id)
        # Note: 'drone_overlay' is handled by ControlCenter directly (it calls drone.toggle_path/toggle_vision)
