"""Mission-facing facade for the control-center controller and renderer."""

from typing import Any, Iterable, Optional

from ui.control_center.controller import (
    ControlAction,
    ControlCenterController,
    ControlHitMap,
)
from ui.control_center.renderer import ControlCenterRenderer
from ui.control_center.view_model import (
    ControlCenterViewModel,
    DroneStatusView,
    RoverStatusView,
)


class ControlCenter:
    """Coordinate non-rendering UI state with the Pygame renderer."""

    def __init__(self, game: Any) -> None:
        """Create controller/renderer collaborators for one mission UI."""
        self.game = game
        self._controller = ControlCenterController()
        self._renderer = ControlCenterRenderer(game)
        self._hit_map = ControlHitMap()
        self._num_drones = 0
        self._num_rovers = 0

    @property
    def active_tab(self) -> str:
        """Currently selected control-center tab."""
        return self._controller.active_tab

    @active_tab.setter
    def active_tab(self, value: str) -> None:
        """Set the currently selected control-center tab."""
        self._controller.active_tab = str(value)

    @property
    def explored_percent(self) -> int:
        """Latest mission explored percentage shown in the header."""
        return self._controller.explored_percent

    @explored_percent.setter
    def explored_percent(self, value: int) -> None:
        """Update the mission explored percentage."""
        self.set_explored_percent(value)

    @property
    def num_drones(self) -> int:
        """Number of drones represented in the latest rendered frame."""
        return self._num_drones

    @property
    def num_rovers(self) -> int:
        """Number of rovers represented in the latest rendered frame."""
        return self._num_rovers

    def start_timer(self) -> None:
        """Start mission elapsed-time tracking."""
        self._controller.start_timer()

    def pause_timer(self) -> None:
        """Pause the displayed elapsed-time counter."""
        self._controller.pause_timer()

    def resume_timer(self) -> None:
        """Resume the displayed elapsed-time counter."""
        self._controller.resume_timer()

    def format_timer(self) -> str:
        """Return the mission elapsed time as ``MM:SS``."""
        return self._controller.format_timer()

    def set_explored_percent(self, value: int) -> None:
        """Update mission progress without exposing controller internals."""
        self._controller.set_explored_percent(value)

    def draw_control_center(
        self,
        drone_statuses: tuple[DroneStatusView, ...],
        rover_statuses: tuple[RoverStatusView, ...] = (),
        show_terrain_heatmap: bool = True,
        selected_drone_heatmap_id: Optional[int] = None,
        debug_lines: Optional[Iterable[str]] = None,
        is_paused: bool = False,
        music_enabled: bool = True,
        show_full_map: bool = True,
    ) -> None:
        """Build one detached frame model, render it, and retain its hit map."""
        self._num_drones = len(drone_statuses)
        self._num_rovers = len(rover_statuses)
        view = ControlCenterViewModel(
            elapsed_time=self._controller.format_timer(),
            explored_percent=self._controller.explored_percent,
            active_tab=self._controller.active_tab,
            drone_statuses=drone_statuses,
            rover_statuses=rover_statuses,
            show_terrain_heatmap=show_terrain_heatmap,
            selected_drone_heatmap_id=selected_drone_heatmap_id,
            debug_lines=debug_lines or (),
            is_paused=is_paused,
            music_enabled=music_enabled,
            show_full_map=show_full_map,
        )
        self._hit_map = self._renderer.render(view)

    def handle_click(
        self,
        mouse_pos: tuple[int, int],
    ) -> Optional[ControlAction]:
        """Interpret the latest renderer-produced hit geometry."""
        return self._controller.handle_click(
            mouse_pos,
            self._hit_map,
        )

    def percent_color(
        self,
        value: int,
        max_value: int = 100,
    ) -> tuple[int, int, int]:
        """Retain the public percentage-color helper."""
        return self._renderer.percent_color(value, max_value)
