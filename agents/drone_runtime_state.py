"""Synchronized mutable runtime state for one drone."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from agents.graph import Graph


Position = Tuple[int, int]


@dataclass(frozen=True)
class DroneSnapshot:
    """Detached immutable state safe for rendering, sharing, and debugging."""

    position: Position
    direction: int
    direction_history: tuple[int, ...]
    path_history: tuple[Position, ...]
    frontiers: tuple[Position, ...]
    returning_home: bool
    done: bool
    explored: bool
    heading_deg: float
    ray_points: tuple[Position, ...]
    battery: int
    show_path: bool
    show_vision: bool
    frontier_rebuild_cooldown: float
    last_frontier_rebuild: float


class DroneRuntimeState:
    """Own all mutable drone values shared across mission threads."""

    def __init__(
        self,
        start_position: Position,
        cave: Any,
        direction: int,
        frontier_rebuild_cooldown: float,
    ) -> None:
        self._lock = threading.RLock()
        self._graph = Graph(*start_position, cave)
        self._position = start_position
        self._direction = int(direction)
        self._direction_history: list[int] = []
        self._frontiers: list[Position] = []
        self._returning_home = False
        self._done = False
        self._explored = False
        self._heading_deg = 0.0
        self._ray_points: list[Position] = []
        self._battery = 100
        self._show_path = True
        self._show_vision = True
        self._frontier_rebuild_cooldown = max(
            0.0,
            float(frontier_rebuild_cooldown),
        )
        self._last_frontier_rebuild = 0.0

    def snapshot(self) -> DroneSnapshot:
        """Return one coherent detached copy of current runtime state."""
        with self._lock:
            return DroneSnapshot(
                position=self._position,
                direction=self._direction,
                direction_history=tuple(self._direction_history),
                path_history=tuple(self._graph.pos),
                frontiers=tuple(self._frontiers),
                returning_home=self._returning_home,
                done=self._done,
                explored=self._explored,
                heading_deg=self._heading_deg,
                ray_points=tuple(self._ray_points),
                battery=self._battery,
                show_path=self._show_path,
                show_vision=self._show_vision,
                frontier_rebuild_cooldown=self._frontier_rebuild_cooldown,
                last_frontier_rebuild=self._last_frontier_rebuild,
            )

    def graph_is_valid(self, current: Position, candidate: Position) -> bool:
        """Validate a movement segment against the simulator collision map."""
        with self._lock:
            return self._graph.is_valid(current, candidate)

    def move_to(self, position: Position) -> None:
        """Update position, heading, and the rendered breadcrumb history."""
        normalized = (int(position[0]), int(position[1]))
        with self._lock:
            previous = self._position
            delta_x = normalized[0] - previous[0]
            delta_y = normalized[1] - previous[1]
            self._position = normalized
            if delta_x != 0 or delta_y != 0:
                self._heading_deg = math.degrees(
                    math.atan2(delta_x, -delta_y)
                ) % 360.0
            self._graph.add_node(normalized)

    def set_direction(self, direction: int) -> None:
        """Store the selected exploration heading in degrees."""
        with self._lock:
            self._direction = int(direction) % 360

    def reorient(self, direction: int) -> None:
        """Rotate in place to a recovery heading without changing position."""
        normalized = int(direction) % 360
        with self._lock:
            self._direction = normalized
            self._heading_deg = float(normalized)
            self._direction_history.append(normalized)

    def begin_exploration(
        self,
        direction: int,
        frontiers: Iterable[Position] = (),
    ) -> None:
        """Record one random heading and merge its visible border targets."""
        normalized = int(direction) % 360
        with self._lock:
            self._explored = True
            self._direction = normalized
            self._heading_deg = float(normalized)
            self._direction_history.append(normalized)
            merged = set(self._frontiers)
            merged.update(
                (int(position[0]), int(position[1]))
                for position in frontiers
            )
            self._frontiers = sorted(merged)

    def replace_frontiers(self, frontiers: Iterable[Position]) -> None:
        """Replace all known border targets with a fresh SLAM frontier set."""
        with self._lock:
            self._frontiers = sorted({
                (int(position[0]), int(position[1]))
                for position in frontiers
            })

    def merge_frontiers(self, frontiers: Iterable[Position]) -> None:
        """Merge border coordinates shared by another drone."""
        with self._lock:
            merged = set(self._frontiers)
            merged.update(
                (int(position[0]), int(position[1]))
                for position in frontiers
            )
            self._frontiers = sorted(merged)

    def remove_frontier(self, target: Position) -> None:
        """Remove one reached or stale border target."""
        normalized = (int(target[0]), int(target[1]))
        with self._lock:
            if normalized in self._frontiers:
                self._frontiers.remove(normalized)

    def evaluate_mission_state(self) -> tuple[bool, bool]:
        """Return `(done, homing)` and home after explored borders exhaust."""
        with self._lock:
            if self._explored and not self._frontiers and not self._done:
                self._returning_home = True
            return self._done, self._returning_home

    def start_returning_home(self) -> None:
        """Explicitly start homing for mission/UI compatibility."""
        with self._lock:
            self._explored = True
            if not self._done:
                self._returning_home = True

    def mark_done(self) -> None:
        """Mark this drone as fully complete."""
        with self._lock:
            self._done = True

    def set_ray_points(self, ray_points: Iterable[Position]) -> None:
        """Store the latest sparse ray endpoints for vision rendering."""
        with self._lock:
            self._ray_points = [
                (int(position[0]), int(position[1]))
                for position in ray_points
            ]

    def toggle_path(self) -> None:
        with self._lock:
            self._show_path = not self._show_path

    def toggle_vision(self) -> None:
        with self._lock:
            self._show_vision = not self._show_vision

    def set_overlay_visibility(
        self,
        *,
        show_path: bool,
        show_vision: bool,
    ) -> None:
        with self._lock:
            self._show_path = bool(show_path)
            self._show_vision = bool(show_vision)

    def reserve_frontier_rebuild(self, now: float) -> bool:
        """Reserve a frontier rebuild after its configured cooldown."""
        with self._lock:
            if (
                float(now) - self._last_frontier_rebuild
            ) < self._frontier_rebuild_cooldown:
                return False
            self._last_frontier_rebuild = float(now)
            return True
