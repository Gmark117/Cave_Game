"""Small exploration-policy primitives for drone movement."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Iterable

from asset_config.helpers import next_cell_coords
from config.simulation_config import ExplorationConfig


Position = tuple[int, int]
CoverageCell = tuple[int, int]
CoverageEdge = tuple[CoverageCell, CoverageCell]


class RandomDirectionPolicy:
    """Choose uniformly or by supplied weights with a private RNG."""

    def __init__(self, *, seed: int) -> None:
        self._random = random.Random(int(seed))

    def choose_direction(self, valid_directions: Iterable[int]) -> int:
        """Return one valid integer heading.

        A private seeded generator keeps each drone reproducible without
        coupling choices to thread scheduling or unrelated global randomness.
        """
        choices = tuple(int(direction) for direction in valid_directions)
        if not choices:
            raise ValueError("valid_directions must not be empty")
        return self._random.choice(choices)

    def choose_weighted_direction(
        self,
        direction_weights: Mapping[int, float],
    ) -> int:
        """Choose reproducibly from positive per-heading weights."""
        weighted = tuple(
            (int(direction), max(0.0, float(weight)))
            for direction, weight in direction_weights.items()
        )
        if not weighted:
            raise ValueError("direction_weights must not be empty")
        total = sum(weight for _direction, weight in weighted)
        if total <= 0.0:
            return self.choose_direction(
                direction for direction, _weight in weighted
            )

        sample = self._random.random() * total
        cumulative = 0.0
        for direction, weight in weighted:
            cumulative += weight
            if sample < cumulative:
                return direction
        return weighted[-1][0]


@dataclass(frozen=True)
class _CoverageRecord:
    """Exponentially decaying traversal pressure for one cell or edge."""

    value: float
    updated_at: float


class CoverageMemory:
    """Drone-local traversal pressure, independent of movement and mission state."""

    def __init__(
        self, exploration: ExplorationConfig, position: Position, now: float,
    ) -> None:
        self.cell_size = max(
            1,
            int(exploration.coverage_memory_cell_size),
        )
        self.decay_seconds = max(
            1e-9,
            float(exploration.coverage_memory_decay_seconds),
        )
        self.visit_weight = max(
            0.0,
            float(exploration.coverage_visit_weight),
        )
        self.edge_weight = max(
            0.0,
            float(exploration.coverage_edge_weight),
        )
        initial_cell = self.cell(position)
        self.cell_visits: dict[CoverageCell, _CoverageRecord] = {
            initial_cell: _CoverageRecord(value=1.0, updated_at=now)
        }
        self.edge_visits: dict[CoverageEdge, _CoverageRecord] = {}
        self._last_cell = initial_cell

    def cell(self, position: Position) -> CoverageCell:
        """Return the coarse coverage-memory cell for a map position."""
        return (
            int(position[0]) // self.cell_size,
            int(position[1]) // self.cell_size,
        )

    @staticmethod
    def edge(
        first: CoverageCell,
        second: CoverageCell,
    ) -> CoverageEdge:
        """Return one direction-independent coarse traversal edge."""
        return (first, second) if first <= second else (second, first)

    def _record_value(
        self,
        record: _CoverageRecord | None,
        now: float,
    ) -> float:
        """Return an exponentially decayed visit pressure."""
        if record is None:
            return 0.0
        elapsed = max(0.0, float(now) - record.updated_at)
        return record.value * math.exp(
            -elapsed / self.decay_seconds
        )

    def _increment_record(
        self,
        records: dict[Any, _CoverageRecord],
        key: Any,
        now: float,
    ) -> None:
        """Add one visit after lazily decaying the previous pressure."""
        records[key] = _CoverageRecord(
            value=self._record_value(records.get(key), now) + 1.0,
            updated_at=float(now),
        )

    def heading_penalties(
        self,
        directions: Iterable[int],
        *,
        current: Position,
        apply_penalty: bool,
        now: float,
        width: int,
        height: int,
    ) -> tuple[
        dict[int, CoverageCell],
        dict[int, float],
        dict[int, float],
        dict[int, float],
    ]:
        """Score projected cells and repeated edges for candidate headings."""
        current_cell = self.cell(current)
        width = max(1, int(width))
        height = max(1, int(height))
        cells: dict[int, CoverageCell] = {}
        visit_pressure: dict[int, float] = {}
        edge_pressure: dict[int, float] = {}
        penalty_factor: dict[int, float] = {}
        for raw_direction in directions:
            direction = int(raw_direction)
            projected = next_cell_coords(
                *current,
                self.cell_size,
                direction,
            )
            projected = (
                min(max(int(projected[0]), 0), width - 1),
                min(max(int(projected[1]), 0), height - 1),
            )
            cell = self.cell(projected)
            edge = self.edge(current_cell, cell)
            if cell == current_cell:
                # A coarse cell does not distinguish headings within itself;
                # penalizing it would arbitrarily overpower other evidence
                # near map edges and cell centers.
                visits = 0.0
                traversals = 0.0
            else:
                visits = self._record_value(
                    self.cell_visits.get(cell),
                    now,
                )
                traversals = self._record_value(
                    self.edge_visits.get(edge),
                    now,
                )
            pressure = (
                self.visit_weight * math.log1p(visits)
                + self.edge_weight * math.log1p(traversals)
            )
            cells[direction] = cell
            visit_pressure[direction] = visits
            edge_pressure[direction] = traversals
            penalty_factor[direction] = (
                1.0 if not apply_penalty else 1.0 / (1.0 + pressure)
            )
        return cells, visit_pressure, edge_pressure, penalty_factor

    def record_transition(
        self,
        previous: Position,
        current: Position,
        now: float,
    ) -> tuple[int, int, int, int]:
        """Record one coarse cell crossing and return trace counters."""
        previous_cell = self.cell(previous)
        current_cell = self.cell(current)
        if self._last_cell != previous_cell:
            self._increment_record(
                self.cell_visits,
                previous_cell,
                now,
            )
            self._last_cell = previous_cell
        if current_cell == previous_cell:
            return 0, 0, 0, 0

        edge = self.edge(previous_cell, current_cell)
        revisited = current_cell in self.cell_visits
        repeated_edge = edge in self.edge_visits
        self._increment_record(
            self.cell_visits,
            current_cell,
            now,
        )
        self._increment_record(
            self.edge_visits,
            edge,
            now,
        )
        self._last_cell = current_cell
        return (
            1,
            0 if revisited else 1,
            1 if revisited else 0,
            1 if repeated_edge else 0,
        )
