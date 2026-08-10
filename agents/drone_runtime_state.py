"""Synchronized mutable runtime state for one drone."""

import math
import threading
from dataclasses import dataclass, replace
from typing import Any, Iterable, Tuple

from agents.graph import Graph
from navigation.navigation_intent import (
    MovementMode,
    MovementOutcome,
    NavigationIntent,
    NavigationWatchdog,
    TransitionReason,
)


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
    frontier_cluster_ids: tuple[int, ...] = ()
    movement_mode: MovementMode = MovementMode.TRAVEL
    navigation_intent: NavigationIntent | None = None
    last_movement_outcome: MovementOutcome = MovementOutcome()
    transition_reason: TransitionReason = TransitionReason.NONE
    navigation_watchdog: NavigationWatchdog = NavigationWatchdog()


class DroneRuntimeState:
    """Own all mutable drone values shared across mission threads."""

    def __init__(
        self,
        start_position: Position,
        cave: Any,
        direction: int,
        frontier_rebuild_cooldown: float,
    ) -> None:
        """Initialize synchronized drone state at the mission start position."""
        self._lock = threading.RLock()
        # Graph owns the route history and collision checks; every access is
        # guarded by this runtime-state lock.
        self._graph = Graph(*start_position, cave)
        self._position = start_position
        self._direction = int(direction)
        self._direction_history: list[int] = []
        self._frontiers: list[Position] = []
        self._frontier_clusters: dict[int, Any] = {}
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
        self._movement_mode = MovementMode.TRAVEL
        self._navigation_intent: NavigationIntent | None = None
        self._next_intent_id = 1
        self._last_movement_outcome = MovementOutcome()
        self._transition_reason = TransitionReason.NONE
        self._navigation_watchdog = NavigationWatchdog()

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
                frontier_cluster_ids=tuple(sorted(self._frontier_clusters)),
                movement_mode=self._movement_mode,
                navigation_intent=self._navigation_intent,
                last_movement_outcome=self._last_movement_outcome,
                transition_reason=self._transition_reason,
                navigation_watchdog=self._navigation_watchdog,
            )

    def navigation_intent(self) -> NavigationIntent | None:
        """Return the immutable active intent without copying path history."""
        with self._lock:
            return self._navigation_intent

    def set_navigation_intent(
        self,
        intent: NavigationIntent,
        *,
        reason: TransitionReason = TransitionReason.SELECTED,
    ) -> NavigationIntent:
        """Atomically latch a strategic goal, route, and movement mode."""
        with self._lock:
            intent_id = int(intent.intent_id)
            if intent_id <= 0:
                intent_id = self._next_intent_id
                self._next_intent_id += 1
            else:
                self._next_intent_id = max(
                    self._next_intent_id,
                    intent_id + 1,
                )
            latched = replace(intent, intent_id=intent_id)
            self._navigation_intent = latched
            self._movement_mode = latched.mode
            self._transition_reason = reason
            return latched

    def advance_navigation_intent(
        self,
        *,
        edge_cursor: int,
        polyline_cursor: int,
        remaining_route_cost: float,
    ) -> NavigationIntent | None:
        """Advance only route cursors while preserving the selected identity."""
        with self._lock:
            if self._navigation_intent is None:
                return None
            self._navigation_intent = self._navigation_intent.advanced(
                edge_cursor=edge_cursor,
                polyline_cursor=polyline_cursor,
                remaining_route_cost=remaining_route_cost,
            )
            return self._navigation_intent

    def replace_navigation_intent(
        self,
        intent: NavigationIntent,
        *,
        reason: TransitionReason = TransitionReason.NONE,
    ) -> NavigationIntent:
        """Replace an active intent while retaining synchronized ownership."""
        with self._lock:
            active = self._navigation_intent
            intent_id = int(intent.intent_id)
            route_id = int(intent.route_id)
            if active is not None:
                intent_id = active.intent_id
                if route_id <= 0:
                    route_id = active.route_id
            elif intent_id <= 0:
                intent_id = self._next_intent_id
                self._next_intent_id += 1
            else:
                self._next_intent_id = max(
                    self._next_intent_id,
                    intent_id + 1,
                )
            latched = replace(
                intent,
                intent_id=intent_id,
                route_id=route_id,
            )
            self._navigation_intent = latched
            self._movement_mode = latched.mode
            if reason != TransitionReason.NONE:
                self._transition_reason = reason
            return latched

    def record_movement_outcome(self, outcome: MovementOutcome) -> None:
        """Store one explicit movement result and its transition reason."""
        with self._lock:
            self._last_movement_outcome = outcome
            if outcome.transition_reason != TransitionReason.NONE:
                self._transition_reason = outcome.transition_reason

    def clear_navigation_intent(
        self,
        outcome: MovementOutcome,
    ) -> NavigationIntent | None:
        """Clear and return the previous intent while preserving its mode."""
        with self._lock:
            previous = self._navigation_intent
            self._navigation_intent = None
            self._last_movement_outcome = outcome
            if outcome.transition_reason != TransitionReason.NONE:
                self._transition_reason = outcome.transition_reason
            return previous

    def set_movement_mode(
        self,
        mode: MovementMode,
        *,
        reason: TransitionReason,
    ) -> None:
        """Change mode explicitly, updating an active intent if one exists."""
        with self._lock:
            self._movement_mode = mode
            self._transition_reason = reason
            if self._navigation_intent is not None:
                self._navigation_intent = replace(
                    self._navigation_intent,
                    mode=mode,
                )

    def update_navigation_watchdog(
        self,
        watchdog: NavigationWatchdog,
    ) -> None:
        """Replace the detached watchdog state under the runtime lock."""
        with self._lock:
            self._navigation_watchdog = watchdog

    def graph_is_valid(
        self,
        current: Position,
        candidate: Position,
    ) -> bool:
        """Validate a movement segment through the owned graph."""
        with self._lock:
            return self._graph.is_valid(current, candidate)

    def move_to(self, position: Position) -> None:
        """Atomically update position, heading, and path history."""
        with self._lock:
            previous = self._position
            dx = position[0] - previous[0]
            dy = position[1] - previous[1]
            self._position = position
            if dx != 0 or dy != 0:
                self._heading_deg = math.degrees(math.atan2(dx, -dy))
            self._graph.add_node(position)

    def set_direction(self, direction: int) -> None:
        """Store the current exploration heading in degrees."""
        with self._lock:
            self._direction = int(direction)

    def begin_exploration(
        self,
        direction: int,
    ) -> None:
        """Record one exploration heading without mutating cluster state."""
        with self._lock:
            self._explored = True
            self._direction = int(direction)
            self._heading_deg = float(direction)
            self._direction_history.append(self._direction)

    def replace_frontier_clusters(self, clusters: Iterable[Any]) -> None:
        """Replace authoritative stable clusters and derive coordinates."""
        self.reconcile_frontier_clusters(clusters)

    def reconcile_frontier_clusters(
        self,
        clusters: Iterable[Any],
    ) -> tuple[int, ...]:
        """Apply a registry view and report stable IDs removed from runtime.

        Canonical retirement is shared mission state.  Returning removed IDs
        makes that transition observable while ensuring stale representatives
        cannot survive in a drone snapshot.
        """
        with self._lock:
            previous_ids = set(self._frontier_clusters)
            detached = {int(cluster.id): cluster for cluster in clusters}
            self._frontier_clusters = detached
            self._frontiers = [
                (
                    int(detached[cluster_id].representative[0]),
                    int(detached[cluster_id].representative[1]),
                )
                for cluster_id in sorted(detached)
            ]
            return tuple(sorted(previous_ids - set(detached)))

    def evaluate_mission_state(self) -> tuple[bool, bool]:
        """Atomically return `(done, homing)` without changing mission phase."""
        with self._lock:
            return self._done, self._returning_home

    def start_returning_home(self) -> None:
        """Latch the drone into homing after exploration is confirmed exhausted."""
        with self._lock:
            self._explored = True
            if not self._done:
                self._returning_home = True

    def mark_done(self) -> None:
        """Mark this drone as fully complete for mission objective checks."""
        with self._lock:
            self._done = True

    def set_ray_points(self, ray_points: Iterable[Position]) -> None:
        """Store the latest vision-ray endpoints for rendering the cone."""
        with self._lock:
            self._ray_points = [
                (int(position[0]), int(position[1]))
                for position in ray_points
            ]

    def toggle_path(self) -> None:
        """Toggle path overlay visibility for this drone."""
        with self._lock:
            self._show_path = not self._show_path

    def toggle_vision(self) -> None:
        """Toggle vision overlay visibility for this drone."""
        with self._lock:
            self._show_vision = not self._show_vision

    def set_overlay_visibility(
        self,
        *,
        show_path: bool,
        show_vision: bool,
    ) -> None:
        """Set both overlay flags together to avoid mixed UI states."""
        with self._lock:
            self._show_path = bool(show_path)
            self._show_vision = bool(show_vision)

    def reserve_frontier_rebuild(self, now: float) -> bool:
        """Reserve a rebuild when the configured cooldown has elapsed."""
        with self._lock:
            if (
                now - self._last_frontier_rebuild
            ) < self._frontier_rebuild_cooldown:
                return False
            self._last_frontier_rebuild = now
            return True
