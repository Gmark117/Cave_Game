"""Exploration decision policies for drone agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random as rand
from typing import Callable, Tuple

import numpy as np

from agents.drone_runtime_state import DroneSnapshot
from asset_config.helpers import next_cell_coords
from mapping.localization import PoseEstimate
from mapping.slam_map import FREE, UNKNOWN, SlamSnapshot
from mapping.terrain_knowledge import TerrainSnapshot


Position = Tuple[int, int]

LARGE_UNEXPLORED_MIN_CELLS = 8


@dataclass(frozen=True)
class FrontierPriorityMasks:
    """Precomputed masks used to rank frontier targets."""

    unexplored: np.ndarray
    low_confidence: np.ndarray


class ExplorationDecisionKind(Enum):
    """Kinds of exploration decisions understood by movement controllers."""

    STEP = "step"
    ROTATE = "rotate"
    FRONTIER = "frontier"
    HOMING = "homing"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class ExplorationDecision:
    """Policy decision returned to the movement controller."""

    kind: ExplorationDecisionKind
    target: Position | None = None
    direction: int | None = None
    valid_directions: tuple[int, ...] = ()
    frontier_targets: tuple[Position, ...] = ()
    planned_path: tuple[Position, ...] = ()


@dataclass(frozen=True)
class ExplorationContext:
    """Detached decision inputs for one exploration-policy evaluation."""

    pose_estimate: PoseEstimate
    runtime_snapshot: DroneSnapshot
    cave_map: np.ndarray
    start_position: Position
    step: int
    radius: int
    map_width: int
    frontier_stride: int
    frontier_confidence_threshold: float
    battery: int
    slam_snapshot: SlamSnapshot | None = None
    terrain_snapshot: TerrainSnapshot | None = None


class FrontierExplorationPolicy:
    """Current frontier/random-step exploration behavior behind a boundary."""

    def decide(
        self,
        context: ExplorationContext,
        is_segment_valid: Callable[[Position, Position], bool],
    ) -> ExplorationDecision:
        """Choose the next high-level exploration action."""
        if context.runtime_snapshot.returning_home:
            return ExplorationDecision(
                kind=ExplorationDecisionKind.HOMING,
                target=context.start_position,
            )

        step_decision = self.choose_next_step(context, is_segment_valid)
        if step_decision.kind == ExplorationDecisionKind.STEP:
            return step_decision

        frontiers = self.prioritize_frontiers(context)
        if frontiers:
            return ExplorationDecision(
                kind=ExplorationDecisionKind.FRONTIER,
                target=frontiers[0],
                frontier_targets=frontiers,
            )

        return ExplorationDecision(ExplorationDecisionKind.EXHAUSTED)

    def choose_next_step(
        self,
        context: ExplorationContext,
        is_segment_valid: Callable[[Position, Position], bool],
    ) -> ExplorationDecision:
        """Choose one locally valid exploration step."""
        current_position = context.pose_estimate.position
        valid_dirs: list[int] = []
        valid_targets: list[Position] = []

        for direction in range(360):
            frontier_target = next_cell_coords(
                *current_position,
                context.radius + 1,
                direction,
            )
            if is_segment_valid(current_position, frontier_target):
                valid_dirs.append(direction)
                valid_targets.append(frontier_target)

        if not valid_dirs:
            return ExplorationDecision(ExplorationDecisionKind.EXHAUSTED)

        chosen_direction = rand.choice(valid_dirs)
        target = next_cell_coords(
            *current_position,
            context.step,
            chosen_direction,
        )
        while not is_segment_valid(current_position, target):
            rejected_index = valid_dirs.index(chosen_direction)
            valid_dirs.pop(rejected_index)
            valid_targets.pop(rejected_index)
            if not valid_dirs:
                return ExplorationDecision(ExplorationDecisionKind.EXHAUSTED)
            chosen_direction = rand.choice(valid_dirs)
            target = next_cell_coords(
                *current_position,
                context.step,
                chosen_direction,
            )

        return ExplorationDecision(
            kind=ExplorationDecisionKind.STEP,
            target=target,
            direction=chosen_direction,
            valid_directions=tuple(valid_dirs),
            frontier_targets=tuple(valid_targets),
        )

    def prioritize_frontiers(
        self,
        context: ExplorationContext,
    ) -> tuple[Position, ...]:
        """Return frontier targets ordered by current exploration priority."""
        masks = self._build_frontier_priority_masks(context)
        return tuple(
            sorted(
                context.runtime_snapshot.frontiers,
                key=lambda target: self._frontier_priority_key(
                    context,
                    target,
                    masks=masks,
                ),
            )
        )

    def frontier_priority(
        self,
        context: ExplorationContext,
        target: Position,
    ) -> tuple[int, float, int, int]:
        """Return a hierarchical priority key for one frontier target."""
        return self._frontier_priority_key(
            context,
            target,
            masks=self._build_frontier_priority_masks(context),
        )

    def update_priority_frontier_registry(
        self,
        context: ExplorationContext,
        frontiers: tuple[Position, ...],
    ) -> None:
        """Remember high-value frontiers across capped fallback decisions."""
        _ = context, frontiers

    def frontier_distance(
        self,
        context: ExplorationContext,
        target: Position,
    ) -> float:
        """Return target distance while deprioritizing already-visible cells."""
        distance = math.dist(context.pose_estimate.position, target)
        if distance <= context.radius:
            return float(context.map_width)
        return distance

    def _frontier_priority_key(
        self,
        context: ExplorationContext,
        target: Position,
        *,
        masks: FrontierPriorityMasks | None,
    ) -> tuple[int, float, int, int]:
        """Rank targets by information class, then travel cost."""
        distance = self.frontier_distance(context, target)
        tiebreak = self._coordinate_tiebreak(target)
        if masks is None:
            return (2, distance, 0, tiebreak)

        large_threshold = self._large_unexplored_threshold(context.radius)
        unexplored_size = self._nearby_mask_count(
            masks.unexplored,
            target,
            radius=self._unexplored_score_radius(context.radius),
        )
        low_confidence_count = self._nearby_mask_count(
            masks.low_confidence,
            target,
            radius=self._unexplored_score_radius(context.radius),
        )
        if unexplored_size >= large_threshold:
            return (0, distance, -unexplored_size, tiebreak)
        if low_confidence_count > 0:
            return (1, distance, -low_confidence_count, tiebreak)
        return (2, distance, -unexplored_size, tiebreak)

    def _build_frontier_priority_masks(
        self,
        context: ExplorationContext,
    ) -> FrontierPriorityMasks | None:
        """Build masks for unexplored space and low-confidence known cells."""
        if context.slam_snapshot is None:
            return None

        occupancy = context.slam_snapshot.occupancy
        confidence = context.slam_snapshot.confidence
        floor_mask = self._floor_mask(context, occupancy.shape)
        terrain_known = self._terrain_known_mask(context, occupancy.shape)
        unexplored = floor_mask & (occupancy == UNKNOWN) & (~terrain_known)
        low_confidence = (
            floor_mask
            & (occupancy != UNKNOWN)
            & (confidence < context.frontier_confidence_threshold)
        )
        return FrontierPriorityMasks(
            unexplored=unexplored,
            low_confidence=low_confidence,
        )

    def _floor_mask(
        self,
        context: ExplorationContext,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Return floor cells when the cave map matches the SLAM shape."""
        cave = np.asarray(context.cave_map)
        if cave.shape == shape:
            return cave == 0
        return np.ones(shape, dtype=bool)

    @staticmethod
    def _terrain_known_mask(
        context: ExplorationContext,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Return cells whose terrain has already been sampled."""
        if (
            context.terrain_snapshot is None
            or context.terrain_snapshot.confidence.shape != shape
        ):
            return np.zeros(shape, dtype=bool)
        return context.terrain_snapshot.confidence > 0.0

    @staticmethod
    def _nearby_mask_count(
        mask: np.ndarray,
        target: Position,
        *,
        radius: int,
    ) -> int:
        """Return the number of matching cells in a square local window."""
        height, width = mask.shape
        x, y = int(target[0]), int(target[1])
        if x < 0 or y < 0 or x >= width or y >= height:
            return 0
        radius = max(1, int(radius))
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        return int(np.count_nonzero(mask[y0:y1, x0:x1]))

    @staticmethod
    def _large_unexplored_threshold(radius: int) -> int:
        """Return the minimum local unknown size treated as a large cluster."""
        radius = max(1, int(radius))
        return max(LARGE_UNEXPLORED_MIN_CELLS, (radius * radius) // 4)

    @staticmethod
    def _unexplored_score_radius(radius: int) -> int:
        """Return the local window radius used for frontier hierarchy scoring."""
        return max(3, int(radius) // 2)

    @staticmethod
    def _coordinate_tiebreak(target: Position) -> int:
        """Return a stable tie-break without absolute row/column bias."""
        x, y = int(target[0]), int(target[1])
        return ((x * 73_856_093) ^ (y * 19_349_663)) & 0xFFFFFFFF

    def extract_frontiers(
        self,
        context: ExplorationContext,
        *,
        stride: int | None = None,
        confidence_threshold: float | None = None,
    ) -> tuple[Position, ...]:
        """Extract frontier cells from local SLAM and terrain confidence."""
        if context.slam_snapshot is None or context.terrain_snapshot is None:
            raise ValueError(
                "frontier extraction requires SLAM and terrain snapshots"
            )

        occupancy = context.slam_snapshot.occupancy
        slam_confidence = context.slam_snapshot.confidence
        terrain_confidence = context.terrain_snapshot.confidence
        cave = np.asarray(context.cave_map)

        height, width = occupancy.shape
        floor_mask = cave == 0
        threshold = (
            context.frontier_confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )

        known_mask = (
            (slam_confidence >= threshold)
            | (terrain_confidence > 0.0)
        )
        free_known = floor_mask & known_mask & (occupancy == FREE)
        unknown = floor_mask & (~known_mask)

        neighbor_unknown = np.zeros_like(unknown, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ys_src = slice(max(0, -dy), height - max(0, dy))
                ys_dst = slice(max(0, dy), height - max(0, -dy))
                xs_src = slice(max(0, -dx), width - max(0, dx))
                xs_dst = slice(max(0, dx), width - max(0, -dx))
                neighbor_unknown[ys_dst, xs_dst] |= unknown[ys_src, xs_src]

        frontier_mask = free_known & neighbor_unknown
        sample_stride = context.frontier_stride if stride is None else int(stride)
        sample_stride = max(1, sample_stride)
        if sample_stride > 1:
            sampled = frontier_mask[::sample_stride, ::sample_stride]
            ys, xs = np.where(sampled)
            ys = ys * sample_stride
            xs = xs * sample_stride
        else:
            ys, xs = np.where(frontier_mask)

        return tuple(
            (int(x), int(y))
            for y, x in zip(ys, xs)
        )
