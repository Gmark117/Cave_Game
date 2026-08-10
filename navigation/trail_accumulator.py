"""Per-drone strategic trail accumulation without persistent pose breadcrumbs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple

import numpy as np

from navigation.waypoint_graph import Position, WaypointPath, WaypointRole, bresenham_path


@dataclass(frozen=True)
class TrailSection:
    """One confirmed physical section ready for persistent graph ingestion."""

    path: WaypointPath
    end_roles: frozenset[WaypointRole]


class StrategicTrailAccumulator:
    """Retain a live trail tail and emit only confirmed strategic sections."""

    def __init__(
        self,
        start: Sequence[int],
        *,
        turn_threshold_degrees: float = 45.0,
        minimum_turn_leg: float = 24.0,
        recovery_interval: float = 128.0,
        chokepoint_narrow_clearance: float = 8.0,
        chokepoint_shoulder_clearance: float = 16.0,
        chokepoint_shoulder_length: float = 24.0,
    ) -> None:
        self.turn_threshold_degrees = float(turn_threshold_degrees)
        self.minimum_turn_leg = float(minimum_turn_leg)
        self.recovery_interval = float(recovery_interval)
        self.chokepoint_narrow_clearance = float(chokepoint_narrow_clearance)
        self.chokepoint_shoulder_clearance = float(chokepoint_shoulder_clearance)
        self.chokepoint_shoulder_length = float(chokepoint_shoulder_length)
        if min(
            self.turn_threshold_degrees,
            self.minimum_turn_leg,
            self.recovery_interval,
            self.chokepoint_narrow_clearance,
            self.chokepoint_shoulder_clearance,
            self.chokepoint_shoulder_length,
        ) <= 0.0:
            raise ValueError("strategic trail thresholds must be positive")
        self._tail: list[Position] = []
        self.reset(start)

    @property
    def tail(self) -> WaypointPath:
        """Return the ephemeral tail, including its last strategic anchor."""
        return tuple(self._tail)

    def reset(self, start: Sequence[int]) -> None:
        """Discard an unconnectable tail and begin at an observed pose."""
        self._tail = [(int(start[0]), int(start[1]))]

    def append(
        self,
        path: Iterable[Sequence[int]],
        known_free: np.ndarray | None = None,
    ) -> Tuple[TrailSection, ...]:
        """Append physical motion and return newly confirmed sections."""
        normalized = self._densify(path)
        if len(normalized) <= 1:
            return ()
        if self._tail[-1] != normalized[0]:
            self.reset(normalized[0])
        self._tail.extend(normalized[1:])

        sections: list[TrailSection] = []
        while len(self._tail) > 1:
            cumulative = self._cumulative_distances(self._tail)
            candidates: dict[int, set[WaypointRole]] = {}
            turn_index = self._confirmed_turn_index(cumulative)
            if turn_index is not None:
                candidates.setdefault(turn_index, set()).add(WaypointRole.TURN)
            if known_free is not None:
                choke_index = self._confirmed_chokepoint_index(
                    cumulative, known_free,
                )
                if choke_index is not None:
                    candidates.setdefault(choke_index, set()).add(
                        WaypointRole.CHOKEPOINT
                    )
            recovery_index = self._index_at_distance(
                cumulative, self.recovery_interval,
            )
            if recovery_index is not None:
                candidates.setdefault(recovery_index, set()).add(
                    WaypointRole.RECOVERY_ANCHOR
                )
            if not candidates:
                break

            index = min(candidates, key=lambda candidate: cumulative[candidate])
            section_path = tuple(self._tail[: index + 1])
            sections.append(TrailSection(
                path=section_path,
                end_roles=frozenset(candidates[index]),
            ))
            self._tail = self._tail[index:]
        return tuple(sections)

    @staticmethod
    def _densify(path: Iterable[Sequence[int]]) -> list[Position]:
        points: list[Position] = []
        for raw in path:
            point = (int(raw[0]), int(raw[1]))
            if not points or points[-1] != point:
                points.append(point)
        if len(points) <= 1:
            return points
        dense = [points[0]]
        for start, end in zip(points, points[1:]):
            dense.extend(bresenham_path(start, end)[1:])
        return dense

    @staticmethod
    def _cumulative_distances(path: Sequence[Position]) -> list[float]:
        result = [0.0]
        for start, end in zip(path, path[1:]):
            result.append(result[-1] + math.dist(start, end))
        return result

    @staticmethod
    def _index_before_distance(
        cumulative: Sequence[float], index: int, distance: float,
    ) -> int | None:
        target = cumulative[index] - distance
        if target < -1e-9:
            return None
        return max(i for i in range(index + 1) if cumulative[i] <= target + 1e-9)

    @staticmethod
    def _index_after_distance(
        cumulative: Sequence[float], index: int, distance: float,
    ) -> int | None:
        target = cumulative[index] + distance
        for i in range(index, len(cumulative)):
            if cumulative[i] + 1e-9 >= target:
                return i
        return None

    @staticmethod
    def _index_at_distance(
        cumulative: Sequence[float], distance: float,
    ) -> int | None:
        for index, value in enumerate(cumulative[1:], start=1):
            if value + 1e-9 >= distance:
                return index
        return None

    def _confirmed_turn_index(self, cumulative: Sequence[float]) -> int | None:
        best: tuple[float, float, int] | None = None
        for index in range(1, len(self._tail) - 1):
            immediate_incoming = (
                self._tail[index][0] - self._tail[index - 1][0],
                self._tail[index][1] - self._tail[index - 1][1],
            )
            immediate_outgoing = (
                self._tail[index + 1][0] - self._tail[index][0],
                self._tail[index + 1][1] - self._tail[index][1],
            )
            immediate_cosine = max(-1.0, min(1.0, (
                immediate_incoming[0] * immediate_outgoing[0]
                + immediate_incoming[1] * immediate_outgoing[1]
            ) / (
                math.hypot(*immediate_incoming)
                * math.hypot(*immediate_outgoing)
            )))
            if math.degrees(math.acos(immediate_cosine)) + 1e-9 < self.turn_threshold_degrees:
                continue
            before = self._index_before_distance(
                cumulative, index, self.minimum_turn_leg,
            )
            after = self._index_after_distance(
                cumulative, index, self.minimum_turn_leg,
            )
            if before is None or after is None:
                continue
            incoming = (
                self._tail[index][0] - self._tail[before][0],
                self._tail[index][1] - self._tail[before][1],
            )
            outgoing = (
                self._tail[after][0] - self._tail[index][0],
                self._tail[after][1] - self._tail[index][1],
            )
            first_length = math.hypot(*incoming)
            second_length = math.hypot(*outgoing)
            if first_length <= 1e-9 or second_length <= 1e-9:
                continue
            cosine = max(-1.0, min(1.0, (
                incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
            ) / (first_length * second_length)))
            angle = math.degrees(math.acos(cosine))
            if angle + 1e-9 < self.turn_threshold_degrees:
                continue
            score = (angle, -cumulative[index], -index)
            if best is None or score > best:
                best = score
        return None if best is None else -best[2]

    def _confirmed_chokepoint_index(
        self, cumulative: Sequence[float], known_free: np.ndarray,
    ) -> int | None:
        mask = np.asarray(known_free)
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise TypeError("known_free must be a two-dimensional boolean mask")
        clearances: dict[int, float] = {}

        def clearance(index: int) -> float:
            if index not in clearances:
                clearances[index] = self._clearance(
                    self._tail[index], mask,
                    self.chokepoint_shoulder_clearance,
                )
            return clearances[index]

        for index in range(1, len(self._tail) - 1):
            before = self._index_before_distance(
                cumulative, index, self.chokepoint_shoulder_length,
            )
            after = self._index_after_distance(
                cumulative, index, self.chokepoint_shoulder_length,
            )
            if before is None or after is None:
                continue
            center = clearance(index)
            if center > self.chokepoint_narrow_clearance + 1e-9:
                continue
            if (
                clearance(before) + 1e-9 < self.chokepoint_shoulder_clearance
                or clearance(after) + 1e-9 < self.chokepoint_shoulder_clearance
            ):
                continue
            if center < clearance(index - 1) and center < clearance(index + 1):
                return index
        return None

    @staticmethod
    def _clearance(
        point: Position, known_free: np.ndarray, cap: float,
    ) -> float:
        x, y = point
        height, width = known_free.shape
        if not (0 <= x < width and 0 <= y < height) or not known_free[y, x]:
            return 0.0
        radius = int(math.ceil(cap))
        best = cap
        for other_y in range(max(0, y - radius), min(height, y + radius + 1)):
            for other_x in range(max(0, x - radius), min(width, x + radius + 1)):
                if known_free[other_y, other_x]:
                    continue
                best = min(best, math.hypot(other_x - x, other_y - y))
        # Map boundaries are also the edge of known free evidence.
        best = min(best, x + 1.0, y + 1.0, width - x, height - y)
        return best
