"""Persistent strategic navigation state and explicit movement outcomes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Tuple


Position = Tuple[int, int]
Polyline = Tuple[Position, ...]


class MovementMode(str, Enum):
    """High-level movement modes shared by planning and runtime state."""

    TRAVEL = "travel"
    SCAN = "scan"
    RECOVERY = "recovery"
    HOME = "home"


class TransitionReason(str, Enum):
    """Reasons that can create, retain, or terminate an intent."""

    NONE = "none"
    SELECTED = "selected"
    PROGRESS = "progress"
    REACHED = "reached"
    INVALIDATED = "invalidated"
    ROUTE_EDGE_RETIRED = "route_edge_retired"
    BELIEF_CORRIDOR_INVALID = "belief_corridor_invalid"
    STALLED = "stalled"
    REVERSAL = "reversal"
    RESERVATION_LOST = "reservation_lost"
    GOAL_RETIRED = "goal_retired"
    NO_ACTIONABLE_FRONTIER = "no_actionable_frontier"
    SCAN_STARTED = "scan_started"
    SCAN_COMPLETE = "scan_complete"
    ZERO_GAIN = "zero_gain"
    RECOVERY_COMPLETE = "recovery_complete"
    HOME = "home"
    COLLISION = "collision"
    PAUSED = "paused"


@dataclass(frozen=True)
class NavigationIntent:
    """A latched goal and the exact oriented route currently being executed."""

    intent_id: int = 0
    route_id: int = 0
    mode: MovementMode = MovementMode.TRAVEL
    cluster_id: int | None = None
    gateway_id: int | None = None
    assignment_token: int | None = None
    target: Position | None = None
    topology_revision: int = 0
    requester_knowledge_revision: int = 0
    route_node_ids: tuple[int, ...] = ()
    route_edge_ids: tuple[int, ...] = ()
    route_paths: tuple[Polyline, ...] = ()
    route_sources: tuple[str, ...] = ()
    route_segment_edge_ids: tuple[int | None, ...] = ()
    edge_cursor: int = 0
    polyline_cursor: int = 0
    remaining_route_cost: float = math.inf
    selection_slam_version: int = 0
    scan_heading_cursor: int = 0
    scan_heading_count: int = 6
    scan_base_heading: float = 0.0
    scan_sequence: int = 0
    scan_start_sensor_newly_known_cells: int = 0
    scan_start_sensor_confidence_gain: float = 0.0
    local_scan_pending: bool = False
    previous_primitive: str | None = None

    def advanced(
        self,
        *,
        edge_cursor: int,
        polyline_cursor: int,
        remaining_route_cost: float,
    ) -> "NavigationIntent":
        """Return a cursor-advanced copy without changing latched identity."""
        return replace(
            self,
            edge_cursor=max(0, int(edge_cursor)),
            polyline_cursor=max(0, int(polyline_cursor)),
            remaining_route_cost=max(0.0, float(remaining_route_cost)),
        )


@dataclass(frozen=True)
class MovementOutcome:
    """Observable result of one bounded movement/state-machine action."""

    travelled_distance: float = 0.0
    route_progress_delta: float = 0.0
    arrived: bool = False
    collision: bool = False
    scan_complete: bool = False
    actual_information_gain: float = 0.0
    invalidated: bool = False
    transition_reason: TransitionReason = TransitionReason.NONE

    @property
    def made_progress(self) -> bool:
        """Only information gain or monotonic route reduction is progress."""
        return self.actual_information_gain > 0.0 or self.route_progress_delta > 0.0

    def __bool__(self) -> bool:
        """Preserve legacy truth tests while callers migrate to rich fields."""
        return bool(
            self.arrived
            or self.scan_complete
            or self.travelled_distance > 0.0
            or self.route_progress_delta > 0.0
        )


@dataclass(frozen=True)
class NavigationWatchdog:
    """Synchronized no-progress and short-window revisit state."""

    last_progress_time: float = 0.0
    distance_without_progress: float = 0.0
    recent_visits: tuple[object, ...] = ()
    reversal_count: int = 0

    @property
    def revisit_ratio(self) -> float:
        """Return the repeated-visit share of the bounded watchdog window."""
        if not self.recent_visits:
            return 0.0
        return 1.0 - (
            len(set(self.recent_visits)) / len(self.recent_visits)
        )

    def observe(
        self,
        outcome: MovementOutcome,
        *,
        now: float,
        visit: object | None = None,
        window: int = 32,
    ) -> "NavigationWatchdog":
        """Fold one action into progress, revisit, and reversal accounting."""
        visits = self.recent_visits
        reversals = self.reversal_count
        if visit is not None and (not visits or visits[-1] != visit):
            if (
                len(visits) >= 2
                and visits[-2] == visit
                and visits[-1] != visit
            ):
                reversals += 1
            visits = (visits + (visit,))[-max(1, int(window)):]
        if outcome.made_progress:
            return NavigationWatchdog(
                last_progress_time=float(now),
                distance_without_progress=0.0,
                recent_visits=visits,
                reversal_count=reversals,
            )
        started = self.last_progress_time or float(now)
        return NavigationWatchdog(
            last_progress_time=started,
            distance_without_progress=(
                self.distance_without_progress + outcome.travelled_distance
            ),
            recent_visits=visits,
            reversal_count=reversals,
        )

    def recovery_reason(self, *, now: float) -> TransitionReason | None:
        """Return the deterministic watchdog transition, if any."""
        if self.reversal_count >= 2:
            return TransitionReason.REVERSAL
        if self.revisit_ratio >= 0.60:
            return TransitionReason.STALLED
        if self.distance_without_progress >= 64.0:
            return TransitionReason.STALLED
        if (
            self.last_progress_time > 0.0
            and float(now) - self.last_progress_time >= 10.0
        ):
            return TransitionReason.STALLED
        return None
