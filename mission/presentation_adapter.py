"""Presentation layer adapter for UI state and event dispatch.

Separates UI concerns from simulation logic by managing SLAM map visibility,
selected drone state, and click-event dispatch to control-center views.
"""

from typing import Any, List, Optional, Tuple


class PresentationAdapter:
    """Isolates UI state and click dispatch from simulation orchestration.

    Responsibilities:
    - Manage SLAM map visibility flags (global and per-drone)
    - Track which drone's SLAM map is selected for detail view
    - Dispatch click events from control-center to internal state updates
    - Provide clean interface for MissionControl to render UI without owning state
    """

    def __init__(self, map_w: int, map_h: int) -> None:
        """Initialize presentation adapter with map dimensions.

        Args:
            map_w: Map width in pixels.
            map_h: Map height in pixels.
        """
        # SLAM map visibility state
        self.show_terrain_heatmap = False
        self.show_full_map = False
        self.selected_drone_heatmap_id: Optional[int] = None

        # SLAM map rendering state
        self.terrain_heatmap_dirty = True

    def reset(self, drone_objects: List[Any]) -> None:
        """Restore the default combined occupancy and agent-overlay state."""
        self.show_terrain_heatmap = False
        self.show_full_map = False
        self.selected_drone_heatmap_id = None
        self.terrain_heatmap_dirty = True
        for drone in drone_objects:
            drone.set_overlay_visibility(
                show_path=True,
                show_vision=True,
            )

    def toggle_terrain_heatmap(self, drone_objects: List[Any]) -> None:
        """Toggle global terrain rendering and synchronize agent overlays."""
        self.show_terrain_heatmap = not self.show_terrain_heatmap
        self.terrain_heatmap_dirty = True
        self._apply_heatmap_visibility(drone_objects)

    def toggle_full_map(self) -> None:
        """Toggle the generated cave-map underlay below discovered data."""
        self.show_full_map = not self.show_full_map
        self.terrain_heatmap_dirty = True

    def toggle_drone_heatmap(
        self,
        drone_id: int,
        drone_objects: List[Any],
    ) -> None:
        """Toggle one drone's map and synchronize all agent overlays."""
        if not self._valid_drone_id(drone_id, drone_objects):
            return
        if self.selected_drone_heatmap_id == drone_id:
            self.selected_drone_heatmap_id = None
        else:
            self.selected_drone_heatmap_id = drone_id
        self.terrain_heatmap_dirty = True
        self._apply_heatmap_visibility(drone_objects)

    def toggle_drone_path(
        self,
        drone_id: int,
        drone_objects: List[Any],
    ) -> None:
        """Toggle one drone's path overlay."""
        if self._valid_drone_id(drone_id, drone_objects):
            drone_objects[drone_id].toggle_path()

    def toggle_drone_vision(
        self,
        drone_id: int,
        drone_objects: List[Any],
    ) -> None:
        """Toggle one drone's vision overlay."""
        if self._valid_drone_id(drone_id, drone_objects):
            drone_objects[drone_id].toggle_vision()

    def _apply_heatmap_visibility(self, drone_objects: List[Any]) -> None:
        """Derive every drone overlay from the current heatmap selection."""
        selected_id = self.selected_drone_heatmap_id
        if selected_id is None:
            show_overlays = not self.show_terrain_heatmap
            for drone in drone_objects:
                drone.set_overlay_visibility(
                    show_path=show_overlays,
                    show_vision=show_overlays,
                )
            return

        show_selected_path = not self.show_terrain_heatmap
        for drone_id, drone in enumerate(drone_objects):
            is_selected = drone_id == selected_id
            drone.set_overlay_visibility(
                show_path=is_selected and show_selected_path,
                show_vision=is_selected,
            )

    @staticmethod
    def _valid_drone_id(drone_id: int, drone_objects: List[Any]) -> bool:
        """Return whether an action targets a current mission drone."""
        return 0 <= drone_id < len(drone_objects)

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
        click_result = control_center.handle_click(mouse_pos)
        if click_result is None:
            return

        self.handle_control_action(click_result, drone_objects)

    def handle_control_action(
        self,
        click_result: Tuple[str, Optional[int]],
        drone_objects: List[Any],
    ) -> None:
        """Apply a semantic control-center action to presentation state."""
        action, drone_id = click_result
        if action == "terrain_heatmap":
            self.toggle_terrain_heatmap(drone_objects)
        elif action == "full_map":
            self.toggle_full_map()
        elif action == "drone_heatmap" and drone_id is not None:
            self.toggle_drone_heatmap(drone_id, drone_objects)
        elif action == "drone_path" and drone_id is not None:
            self.toggle_drone_path(drone_id, drone_objects)
        elif action == "drone_vision" and drone_id is not None:
            self.toggle_drone_vision(drone_id, drone_objects)
