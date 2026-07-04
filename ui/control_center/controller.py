"""Non-rendering state and input handling for the mission control center."""

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import pygame


RectValue = Tuple[int, int, int, int]
ControlAction = Tuple[str, Optional[int]]


def _rect_value(rect: Optional[Sequence[int]]) -> Optional[RectValue]:
    """Normalize a Pygame-like rect sequence into a plain tuple."""
    if rect is None:
        return None
    return tuple(int(value) for value in rect)  # type: ignore[return-value]


@dataclass(frozen=True)
class ControlHitMap:
    """Detached immutable hit rectangles produced by control-center layout."""

    heatmap_toggle: Optional[RectValue] = None
    tabs: tuple[tuple[str, RectValue], ...] = ()
    drone_toggles: tuple[tuple[int, str, RectValue], ...] = ()

    def __init__(
        self,
        heatmap_toggle: Optional[Sequence[int]] = None,
        tabs: Iterable[tuple[str, Sequence[int]]] = (),
        drone_toggles: Iterable[
            tuple[int, str, Sequence[int]]
        ] = (),
    ) -> None:
        """Copy mutable renderer rectangles into immutable tuple values."""
        object.__setattr__(
            self,
            "heatmap_toggle",
            _rect_value(heatmap_toggle),
        )
        object.__setattr__(
            self,
            "tabs",
            tuple(
                (str(name), _rect_value(rect))
                for name, rect in tabs
            ),
        )
        object.__setattr__(
            self,
            "drone_toggles",
            tuple(
                (int(drone_id), str(action), _rect_value(rect))
                for drone_id, action, rect in drone_toggles
            ),
        )


class ControlCenterController:
    """Own timer, progress, selected tab, and hit-test interpretation."""

    def __init__(self) -> None:
        """Initialize non-rendering state for the control panel."""
        self.active_tab = "drones"
        self.explored_percent = 100
        self._tic: Optional[float] = None
        self._paused_at: Optional[float] = None
        self._paused_duration = 0.0

    def start_timer(self) -> None:
        """Start mission elapsed-time tracking."""
        self._tic = time.perf_counter()
        self._paused_at = None
        self._paused_duration = 0.0

    def pause_timer(self) -> None:
        """Freeze elapsed-time display while the simulation is paused."""
        if self._tic is not None and self._paused_at is None:
            self._paused_at = time.perf_counter()

    def resume_timer(self) -> None:
        """Resume elapsed-time display after a pause."""
        if self._paused_at is None:
            return
        self._paused_duration += time.perf_counter() - self._paused_at
        self._paused_at = None

    def format_timer(self) -> str:
        """Return elapsed mission time as ``MM:SS``."""
        if self._tic is None:
            return "00:00"
        now = (
            self._paused_at
            if self._paused_at is not None
            else time.perf_counter()
        )
        elapsed = max(
            0,
            int(now - self._tic - self._paused_duration),
        )
        minutes, seconds = divmod(elapsed, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def set_explored_percent(self, value: int) -> None:
        """Store explored percentage for the next rendered frame."""
        self.explored_percent = int(value)

    def handle_click(
        self,
        mouse_pos: tuple[int, int],
        hit_map: ControlHitMap,
    ) -> Optional[ControlAction]:
        """Convert a mouse position and hit map into a semantic UI action."""
        if (
            hit_map.heatmap_toggle is not None
            and pygame.Rect(hit_map.heatmap_toggle).collidepoint(mouse_pos)
        ):
            return ("terrain_heatmap", None)

        for tab_name, rect in hit_map.tabs:
            if pygame.Rect(rect).collidepoint(mouse_pos):
                self.active_tab = tab_name
                return ("control_tab", None)

        for drone_id, overlay_type, rect in hit_map.drone_toggles:
            if not pygame.Rect(rect).collidepoint(mouse_pos):
                continue
            if overlay_type == "path":
                return ("drone_path", drone_id)
            if overlay_type == "vision":
                return ("drone_vision", drone_id)
            return ("drone_heatmap", drone_id)
        return None
