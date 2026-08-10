"""Deterministic bounded global frontier candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Tuple


Position = Tuple[int, int]


@dataclass(frozen=True)
class StrategicCandidate:
    cluster_id: int
    representative: Position
    expected_gain: float
    route_cost: float
    waypoint: Position | None = None
    revisit_penalty: float = 0.0
    stall_penalty: float = 0.0
    reserved_by_other: bool = False
    reachable: bool = True
    blacklisted: bool = False
    premature_switch: float = 0.0
    wall_gain: float = 0.0


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: StrategicCandidate
    score: float


def _normalized(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return ()
    low, high = min(values), max(values)
    if high - low <= 1e-9:
        return tuple(0.0 for _value in values)
    return tuple((value - low) / (high - low) for value in values)


def select_strategic_candidates(
    candidates: Iterable[StrategicCandidate],
    *,
    position: Position,
    per_group_limit: int = 16,
) -> tuple[ScoredCandidate, ...]:
    """Filter, bound to gain/near groups, normalize, and score candidates."""
    eligible = tuple(
        candidate for candidate in candidates
        if candidate.reachable
        and not candidate.blacklisted
        and not candidate.reserved_by_other
        and math.isfinite(candidate.route_cost)
    )
    wall_continuations = tuple(
        candidate for candidate in eligible if candidate.wall_gain > 0.0
    )
    active_tier = wall_continuations or eligible
    limit = max(0, int(per_group_limit))
    highest_gain = sorted(
        active_tier,
        key=lambda item: (
            -item.wall_gain,
            -item.expected_gain,
            item.cluster_id,
        ),
    )[:limit]
    nearest = sorted(
        active_tier,
        key=lambda item: (
            math.dist(position, item.waypoint or item.representative),
            item.cluster_id,
        ),
    )[:limit]
    bounded = tuple({
        item.cluster_id: item for item in highest_gain + nearest
    }.values())
    gains = _normalized(tuple(item.expected_gain for item in bounded))
    wall_gains = _normalized(tuple(item.wall_gain for item in bounded))
    costs = _normalized(tuple(item.route_cost for item in bounded))
    revisits = _normalized(tuple(item.revisit_penalty for item in bounded))
    stalls = _normalized(tuple(item.stall_penalty for item in bounded))
    scored = tuple(
        ScoredCandidate(
            item,
            (4.0 * wall_gain if wall_continuations else 0.0)
            + (1.0 if wall_continuations else 3.0) * gain
            - cost
            - 1.5 * revisit
            - 2.0 * stall
            - 2.5 * item.premature_switch,
        )
        for item, gain, wall_gain, cost, revisit, stall in zip(
            bounded, gains, wall_gains, costs, revisits, stalls
        )
    )
    return tuple(sorted(
        scored,
        key=lambda item: (-item.score, item.candidate.cluster_id),
    ))
