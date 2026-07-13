"""Monte Carlo Tree Search exploration policy for drone agents."""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
import random
import time
from typing import Callable

import numpy as np

from agents.exploration_policy import (
    ExplorationContext,
    ExplorationDecision,
    ExplorationDecisionKind,
    FrontierExplorationPolicy,
    FrontierPriorityMasks,
    Position,
)
from asset_config.helpers import next_cell_coords
from config.simulation_config import ExplorationConfig
from mapping.drone_sensor import LIDAR_RANGE_RADIUS_MULTIPLIER
from mapping.ray_geometry import bresenham_line_points
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN


TRANSLATE = "translate"
ROTATE = "rotate"
SENSOR_FOV_DEG = 60.0
NEAR_DUPLICATE_DEG = 15.0
STRICT_TURN_LIMIT_DEG = 135.0
LARGE_UNEXPLORED_CELL_GAIN = 10.0
LOW_CONFIDENCE_CELL_GAIN = 2.0
SMALL_UNEXPLORED_CELL_GAIN = 0.15
ROOT_FRONTIER_MIN_CANDIDATES = 128
ROOT_FRONTIER_CANDIDATES_PER_RESULT = 48
FALLBACK_FRONTIER_MIN_CANDIDATES = 64
FALLBACK_FRONTIER_CANDIDATES_PER_RESULT = 16
LARGE_FRONTIER_REGISTRY_LIMIT = 48


@dataclass(frozen=True)
class RootVisitDiagnostic:
    """Immutable summary for one root child in the latest MCTS search."""

    kind: str
    direction: int
    target: Position
    visits: int
    mean_reward: float


@dataclass(frozen=True)
class MctsSearchDiagnostics:
    """Immutable diagnostics exposed after each MCTS decision."""

    iterations: int
    root_visits: tuple[RootVisitDiagnostic, ...]
    selected_reward: float
    generated_nodes: int
    slam_version: int
    selected_kind: str = ""
    selected_direction: int | None = None
    selected_target: Position | None = None
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class DecisionGrid:
    """Precomputed SLAM masks for one MCTS decision."""

    occupancy: np.ndarray
    confidence: np.ndarray
    known_free: np.ndarray
    known_occupied: np.ndarray
    unexplored: np.ndarray
    low_confidence_known: np.ndarray
    unknown: np.ndarray
    height: int
    width: int
    max_range: int
    slam_version: int


@dataclass
class DecisionCache:
    """Caches for repeated state/action evaluations in one decision."""

    directions: dict[tuple, list[int]]
    paths: dict[tuple[Position, Position], tuple[Position, ...]]
    visible: dict[tuple[int, int, int], frozenset[Position]]
    cell_gain: dict[Position, float]


@dataclass(frozen=True)
class SearchState:
    """Planning state used inside one rebuilt MCTS tree."""

    position: Position
    heading: int
    depth: int
    observed: frozenset[Position]


@dataclass(frozen=True)
class SearchAction:
    """One simulated root, tree, or rollout action."""

    kind: str
    direction: int
    target: Position
    path: tuple[Position, ...]
    observed_cells: frozenset[Position]
    immediate_gain: float
    order: int


class MctsNode:
    """Mutable MCTS node for one decision-local search tree."""

    def __init__(
        self,
        state: SearchState,
        *,
        parent: MctsNode | None,
        incoming_action: SearchAction | None,
        untried_actions: tuple[SearchAction, ...],
        discounted_reward_from_root: float = 0.0,
    ) -> None:
        """Create a tree node with mutable MCTS statistics."""
        self.state = state
        self.parent = parent
        self.incoming_action = incoming_action
        self.untried_actions = list(untried_actions)
        self.children: list[MctsNode] = []
        self.visits = 0
        self.accumulated_reward = 0.0
        self.discounted_reward_from_root = discounted_reward_from_root

    @property
    def mean_reward(self) -> float:
        """Return the average backed-up reward."""
        if self.visits <= 0:
            return 0.0
        return self.accumulated_reward / self.visits


class MctsExplorationPolicy(FrontierExplorationPolicy):
    """Single-drone MCTS exploration over local SLAM belief only."""

    def __init__(
        self,
        config: ExplorationConfig,
        *,
        seed: int,
    ) -> None:
        """Store search settings and the drone-specific base seed."""
        self.config = config
        self.seed = int(seed)
        self._last_search_diagnostics: MctsSearchDiagnostics | None = None
        self._ray_offset_cache: dict[
            tuple[int, int, int],
            tuple[tuple[tuple[int, int], ...], ...],
        ] = {}
        self._large_frontier_registry: dict[Position, int] = {}

    @property
    def last_search_diagnostics(self) -> MctsSearchDiagnostics | None:
        """Return immutable diagnostics for the most recent search."""
        return self._last_search_diagnostics

    def decide(
        self,
        context: ExplorationContext,
        is_segment_valid: Callable[[Position, Position], bool],
    ) -> ExplorationDecision:
        """Choose one root MCTS action without using ground-truth planning."""
        started_at = time.perf_counter()
        deadline = self._decision_deadline(started_at)
        search_deadline = self._search_deadline(deadline, context)
        _ = is_segment_valid
        if context.runtime_snapshot.returning_home:
            return ExplorationDecision(
                kind=ExplorationDecisionKind.HOMING,
                target=context.start_position,
            )

        if context.slam_snapshot is None:
            self._last_search_diagnostics = self._empty_diagnostics(
                context,
                elapsed_ms=self._elapsed_ms(started_at),
            )
            return ExplorationDecision(ExplorationDecisionKind.EXHAUSTED)

        grid = self._build_grid(context)
        cache = self._build_cache()
        rng = random.Random(self._decision_seed(context))
        heading = int(round(context.pose_estimate.heading_deg)) % 360
        root_state = SearchState(
            position=context.pose_estimate.position,
            heading=heading,
            depth=0,
            observed=frozenset(),
        )
        root_frontiers = self._frontier_cluster_centroids(
            context,
            root_state.position,
            grid=grid,
            deadline=search_deadline,
        )
        root_actions = self._generate_actions(
            context,
            grid,
            cache,
            root_state,
            rng,
            root_frontiers=root_frontiers,
            deadline=search_deadline,
        )
        if not root_actions:
            diagnostics = self._empty_diagnostics(
                context,
                elapsed_ms=self._elapsed_ms(started_at),
            )
            fallback = self._fallback_frontier_decision(
                context,
                root_frontiers,
            )
            if fallback is not None:
                self._last_search_diagnostics = (
                    self._diagnostics_for_frontier(
                        diagnostics,
                        fallback,
                        elapsed_ms=self._elapsed_ms(started_at),
                    )
                )
                return fallback

            self._last_search_diagnostics = diagnostics
            return ExplorationDecision(ExplorationDecisionKind.EXHAUSTED)

        root = MctsNode(
            root_state,
            parent=None,
            incoming_action=None,
            untried_actions=root_actions,
        )
        generated_nodes = 1
        iterations = 0
        for _ in range(self.config.iterations):
            if (
                iterations > 0
                and search_deadline is not None
                and time.perf_counter() >= search_deadline
            ):
                break
            iterations += 1
            node = self._select(root)
            if node.state.depth < self.config.horizon and node.untried_actions:
                action = node.untried_actions.pop(0)
                child_state = self._apply_action(node.state, action)
                discounted_reward = (
                    node.discounted_reward_from_root
                    + (self.config.discount ** node.state.depth)
                    * action.immediate_gain
                )
                child = MctsNode(
                    child_state,
                    parent=node,
                    incoming_action=action,
                    untried_actions=self._generate_actions(
                        context,
                        grid,
                        cache,
                        child_state,
                        rng,
                        root_frontiers=(),
                        deadline=search_deadline,
                    ),
                    discounted_reward_from_root=discounted_reward,
                )
                node.children.append(child)
                generated_nodes += 1
                node = child

            rollout_reward = self._rollout(
                context,
                grid,
                cache,
                node.state,
                rng,
                deadline=search_deadline,
            )
            total_reward = (
                node.discounted_reward_from_root + rollout_reward
            )
            self._backpropagate(node, total_reward)

        selected = self._select_root_child(root)
        selected_reward = selected.mean_reward if selected is not None else 0.0
        diagnostics = self._diagnostics(
            context,
            root,
            iterations=iterations,
            selected_reward=selected_reward,
            generated_nodes=generated_nodes,
            selected_action=(
                None if selected is None else selected.incoming_action
            ),
            elapsed_ms=self._elapsed_ms(started_at),
        )
        self._last_search_diagnostics = diagnostics
        if (
            selected is None
            or selected.incoming_action is None
            or selected_reward <= 0.0
        ):
            fallback = self._fallback_frontier_decision(
                context,
                root_frontiers,
            )
            if fallback is not None:
                self._last_search_diagnostics = (
                    self._diagnostics_for_frontier(
                        diagnostics,
                        fallback,
                        elapsed_ms=self._elapsed_ms(started_at),
                    )
                )
                return fallback
            return ExplorationDecision(ExplorationDecisionKind.EXHAUSTED)

        action = selected.incoming_action
        if action.kind == ROTATE:
            return ExplorationDecision(
                kind=ExplorationDecisionKind.ROTATE,
                target=root_state.position,
                direction=action.direction,
                frontier_targets=root_frontiers,
            )

        translation_directions = tuple(
            child.incoming_action.direction
            for child in root.children
            if (
                child.incoming_action is not None
                and child.incoming_action.kind == TRANSLATE
            )
        )
        return ExplorationDecision(
            kind=ExplorationDecisionKind.STEP,
            target=action.target,
            direction=action.direction,
            valid_directions=translation_directions or (action.direction,),
            frontier_targets=root_frontiers,
            planned_path=action.path,
        )

    def _fallback_frontier_decision(
        self,
        context: ExplorationContext,
        root_frontiers: tuple[Position, ...],
    ) -> ExplorationDecision | None:
        """Route toward known frontier candidates when local gain is exhausted."""
        current_position = context.pose_estimate.position
        prioritized = (*root_frontiers, *context.runtime_snapshot.frontiers)
        seen: set[Position] = set()
        frontiers = []
        for frontier in prioritized:
            if frontier == current_position or frontier in seen:
                continue
            seen.add(frontier)
            frontiers.append(frontier)
        if not frontiers:
            return None

        root_frontier_set = set(root_frontiers)
        masks = self._build_frontier_priority_masks(context)
        self._remember_large_frontiers(context, root_frontiers, masks=masks)
        registry_frontiers = self._large_frontier_registry_candidates(
            context,
            frontiers,
            masks=masks,
        )
        registry_set = set(registry_frontiers)
        frontiers = self._fallback_frontier_candidates(
            context,
            frontiers,
            root_frontier_set | registry_set,
        )
        frontiers = self._merge_frontier_candidates(
            (*registry_frontiers, *frontiers)
        )
        frontiers.sort(
            key=lambda frontier: (
                *self._frontier_priority_key(
                    context,
                    frontier,
                    masks=masks,
                )[:3],
                0 if frontier in registry_set else 1,
                0 if frontier in root_frontier_set else 1,
                self._coordinate_tiebreak(frontier),
            )
        )

        return ExplorationDecision(
            kind=ExplorationDecisionKind.FRONTIER,
            target=frontiers[0],
            frontier_targets=tuple(frontiers),
        )

    def update_priority_frontier_registry(
        self,
        context: ExplorationContext,
        frontiers: tuple[Position, ...],
    ) -> None:
        """Remember large unexplored frontier regions across decisions."""
        masks = self._build_frontier_priority_masks(context)
        if masks is None:
            return

        available = set(frontiers)
        self._large_frontier_registry = {
            frontier: version
            for frontier, version in self._large_frontier_registry.items()
            if frontier in available
        }

        large_frontiers = [
            frontier
            for frontier in frontiers
            if self._frontier_priority_key(
                context,
                frontier,
                masks=masks,
            )[0]
            == 0
        ]
        self._remember_large_frontiers(
            context,
            large_frontiers,
            masks=masks,
        )

    def _remember_large_frontiers(
        self,
        context: ExplorationContext,
        frontiers: tuple[Position, ...] | list[Position],
        *,
        masks: FrontierPriorityMasks | None,
    ) -> None:
        """Store bounded, spatially diverse representatives of large frontiers."""
        if masks is None or not frontiers:
            return

        version = (
            0
            if context.slam_snapshot is None
            else int(context.slam_snapshot.version)
        )
        existing = dict(self._large_frontier_registry)
        bins: dict[tuple[int, int], Position] = {}
        bin_size = self._frontier_spatial_bin_size(context.radius)

        for frontier in (*existing.keys(), *frontiers):
            if (
                self._frontier_priority_key(context, frontier, masks=masks)[0]
                != 0
            ):
                continue
            key = (
                int(frontier[0]) // bin_size,
                int(frontier[1]) // bin_size,
            )
            current = bins.get(key)
            if current is None or self._frontier_priority_key(
                context,
                frontier,
                masks=masks,
            ) < self._frontier_priority_key(context, current, masks=masks):
                bins[key] = frontier

        candidates = sorted(
            bins.values(),
            key=lambda frontier: (
                self._spatial_bin_tiebreak(
                    (
                        int(frontier[0]) // bin_size,
                        int(frontier[1]) // bin_size,
                    )
                ),
                self._frontier_priority_key(context, frontier, masks=masks),
            ),
        )[:LARGE_FRONTIER_REGISTRY_LIMIT]
        self._large_frontier_registry = {
            frontier: version for frontier in candidates
        }

    def _large_frontier_registry_candidates(
        self,
        context: ExplorationContext,
        frontiers: list[Position],
        *,
        masks: FrontierPriorityMasks | None,
    ) -> tuple[Position, ...]:
        """Return valid registered large frontiers that bypass fallback caps."""
        if masks is None or not self._large_frontier_registry:
            return ()

        available = set(frontiers)
        valid = []
        for frontier in self._large_frontier_registry:
            if frontier not in available:
                continue
            if self._frontier_priority_key(context, frontier, masks=masks)[0] != 0:
                continue
            valid.append(frontier)

        self._large_frontier_registry = {
            frontier: self._large_frontier_registry[frontier]
            for frontier in valid
        }
        return tuple(
            sorted(
                valid,
                key=lambda frontier: self._frontier_priority_key(
                    context,
                    frontier,
                    masks=masks,
                ),
            )
        )

    @staticmethod
    def _merge_frontier_candidates(
        frontiers: tuple[Position, ...] | list[Position],
    ) -> list[Position]:
        """Return unique frontier candidates while preserving first sighting."""
        merged: list[Position] = []
        seen: set[Position] = set()
        for frontier in frontiers:
            if frontier in seen:
                continue
            seen.add(frontier)
            merged.append(frontier)
        return merged

    def _fallback_frontier_candidates(
        self,
        context: ExplorationContext,
        frontiers: list[Position],
        root_frontier_set: set[Position],
    ) -> list[Position]:
        """Return a bounded frontier set for budget-safe fallback scoring."""
        limit = max(
            FALLBACK_FRONTIER_MIN_CANDIDATES,
            max(1, self.config.frontier_cluster_limit)
            * FALLBACK_FRONTIER_CANDIDATES_PER_RESULT,
        )
        if len(frontiers) <= limit:
            return list(frontiers)

        candidates = [
            frontier
            for frontier in frontiers
            if frontier in root_frontier_set
        ][:limit]
        selected = set(candidates)
        remaining_limit = max(0, limit - len(candidates))
        if remaining_limit <= 0:
            return candidates

        remaining = [frontier for frontier in frontiers if frontier not in selected]
        nearest_limit = max(1, remaining_limit // 2)
        nearest = heapq.nsmallest(
            min(nearest_limit, len(remaining)),
            remaining,
            key=lambda frontier: (
                self.frontier_distance(context, frontier),
                self._coordinate_tiebreak(frontier),
            ),
        )
        candidates.extend(nearest)
        selected.update(nearest)

        diverse_limit = max(0, limit - len(candidates))
        if diverse_limit > 0:
            candidates.extend(
                self._spatially_diverse_frontiers(
                    context,
                    remaining,
                    selected,
                    limit=diverse_limit,
                )
            )
        return candidates

    def _spatially_diverse_frontiers(
        self,
        context: ExplorationContext,
        frontiers: list[Position],
        selected: set[Position],
        *,
        limit: int,
    ) -> list[Position]:
        """Return one representative from diverse coarse map regions."""
        if limit <= 0:
            return []

        bin_size = self._frontier_spatial_bin_size(context.radius)
        bins: dict[tuple[int, int], tuple[tuple[float, int], Position]] = {}
        for frontier in frontiers:
            if frontier in selected:
                continue
            key = (
                int(frontier[0]) // bin_size,
                int(frontier[1]) // bin_size,
            )
            score = (
                self.frontier_distance(context, frontier),
                self._coordinate_tiebreak(frontier),
            )
            current = bins.get(key)
            if current is None or score < current[0]:
                bins[key] = (score, frontier)

        representatives = sorted(
            (
                (
                    self._spatial_bin_tiebreak(key),
                    score,
                    frontier,
                )
                for key, (score, frontier) in bins.items()
            ),
            key=lambda item: (item[0], item[1]),
        )
        return [frontier for _, _, frontier in representatives[:limit]]

    def _select(self, root: MctsNode) -> MctsNode:
        """Select a leaf by UCT until expansion is possible."""
        node = root
        while (
            not node.untried_actions
            and node.children
            and node.state.depth < self.config.horizon
        ):
            node = self._select_uct_child(node)
        return node

    def _select_uct_child(self, node: MctsNode) -> MctsNode:
        """Choose the child with the highest UCT score."""
        parent_visits = max(1, node.visits)

        def score(child: MctsNode) -> tuple[float, int, int]:
            if child.visits <= 0:
                exploitation = float("inf")
            else:
                exploration = self.config.uct_exploration * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                exploitation = child.mean_reward + exploration
            direction, kind_order = self._stable_action_order(
                child.incoming_action
            )
            return exploitation, -direction, -kind_order

        return max(node.children, key=score)

    def _select_root_child(self, root: MctsNode) -> MctsNode | None:
        """Choose the executed root child by visits, reward, then direction."""
        if not root.children:
            return None

        def key(child: MctsNode) -> tuple[int, float, int, int]:
            direction, kind_order = self._stable_action_order(
                child.incoming_action
            )
            return child.visits, child.mean_reward, -direction, -kind_order

        return max(root.children, key=key)

    def _rollout(
        self,
        context: ExplorationContext,
        grid: DecisionGrid,
        cache: DecisionCache,
        state: SearchState,
        rng: random.Random,
        *,
        deadline: float | None = None,
    ) -> float:
        """Roll out stochastically until the configured horizon."""
        total = 0.0
        rollout_state = state
        while rollout_state.depth < self.config.horizon:
            if self._deadline_reached(deadline):
                break
            actions = self._generate_actions(
                context,
                grid,
                cache,
                rollout_state,
                rng,
                root_frontiers=(),
                deadline=deadline,
            )
            if not actions:
                break
            action = self._choose_rollout_action(actions, rng)
            total += (
                self.config.discount ** rollout_state.depth
            ) * action.immediate_gain
            rollout_state = self._apply_action(rollout_state, action)
        return total

    def _choose_rollout_action(
        self,
        actions: tuple[SearchAction, ...],
        rng: random.Random,
    ) -> SearchAction:
        """Choose rollout actions with a gain-biased stochastic policy."""
        gains = [max(0.0, action.immediate_gain) for action in actions]
        if not any(gains):
            return rng.choice(actions)
        if self.config.rollout_temperature <= 0.0:
            best_gain = max(gains)
            best_actions = [
                action
                for action, gain in zip(actions, gains)
                if gain == best_gain
            ]
            return best_actions[0]

        max_gain = max(gains)
        scale = max(1.0, max_gain) * self.config.rollout_temperature
        weights = [
            math.exp((gain - max_gain) / max(1e-6, scale))
            for gain in gains
        ]
        return rng.choices(actions, weights=weights, k=1)[0]

    def _backpropagate(self, node: MctsNode, reward: float) -> None:
        """Backpropagate one rollout reward to the root."""
        current: MctsNode | None = node
        while current is not None:
            current.visits += 1
            current.accumulated_reward += reward
            current = current.parent

    def _generate_actions(
        self,
        context: ExplorationContext,
        grid: DecisionGrid,
        cache: DecisionCache,
        state: SearchState,
        rng: random.Random,
        *,
        root_frontiers: tuple[Position, ...],
        deadline: float | None = None,
    ) -> tuple[SearchAction, ...]:
        """Generate translation and rotation actions from one search state."""
        if state.depth > 0 and self._deadline_reached(deadline):
            return ()
        candidate_directions = self._candidate_directions(
            state,
            rng,
            cache=cache,
            root_frontiers=root_frontiers,
        )
        if not candidate_directions:
            return ()

        strict_directions = [
            direction
            for direction in candidate_directions
            if (
                self._angular_distance(direction, state.heading)
                <= STRICT_TURN_LIMIT_DEG
            )
        ]
        actions = self._translation_actions(
            context,
            grid,
            cache,
            state,
            strict_directions,
            deadline=deadline,
        )
        if not actions:
            actions = self._translation_actions(
                context,
                grid,
                cache,
                state,
                candidate_directions,
                deadline=deadline,
            )

        translation_dirs = {
            action.direction
            for action in actions
            if action.kind == TRANSLATE
        }
        remaining_capacity = self.config.branching_factor - len(actions)
        if remaining_capacity > 0:
            actions.extend(
                self._rotation_actions(
                    context,
                    grid,
                    cache,
                    state,
                    candidate_directions,
                    translation_dirs,
                    limit=remaining_capacity,
                    start_order=len(actions),
                    deadline=deadline,
                )
            )

        return tuple(actions[: self.config.branching_factor])

    def _candidate_directions(
        self,
        state: SearchState,
        rng: random.Random,
        *,
        cache: DecisionCache | None = None,
        root_frontiers: tuple[Position, ...],
    ) -> list[int]:
        """Build frontier-biased directions, then seeded random headings."""
        cache_key = (
            state.position,
            state.heading,
            state.depth,
            root_frontiers,
        )
        if cache is not None and cache_key in cache.directions:
            return list(cache.directions[cache_key])

        directions: list[int] = []

        if state.depth == 0:
            for target in root_frontiers[: self.config.frontier_cluster_limit]:
                self._append_direction(
                    directions,
                    self._direction_to(state.position, target),
                )
        else:
            self._append_direction(directions, state.heading)

        attempts = 0
        max_attempts = max(360, self.config.branching_factor * 30)
        while (
            len(directions) < self.config.branching_factor
            and attempts < max_attempts
        ):
            attempts += 1
            self._append_direction(directions, rng.randrange(360))

        if cache is not None:
            cache.directions[cache_key] = list(directions)
        return directions

    def _translation_actions(
        self,
        context: ExplorationContext,
        grid: DecisionGrid,
        cache: DecisionCache,
        state: SearchState,
        directions: list[int],
        *,
        deadline: float | None = None,
    ) -> list[SearchAction]:
        """Return all belief-valid translations for the supplied headings."""
        actions: list[SearchAction] = []
        for index, direction in enumerate(directions):
            if index > 0 and self._deadline_reached(deadline):
                break
            target = next_cell_coords(
                state.position[0],
                state.position[1],
                context.step,
                direction,
            )
            path = self._known_free_path(
                context,
                state.position,
                target,
                grid=grid,
                cache=cache,
            )
            if not path:
                continue
            observed_cells = self._visible_unknown_cells(
                context,
                target,
                direction,
                state.observed,
                grid=grid,
                cache=cache,
                deadline=deadline,
            )
            immediate_gain = self._information_gain(
                context,
                grid,
                cache,
                observed_cells,
            )
            actions.append(
                SearchAction(
                    kind=TRANSLATE,
                    direction=direction,
                    target=target,
                    path=path,
                    observed_cells=observed_cells,
                    immediate_gain=immediate_gain,
                    order=len(actions),
                )
            )
        return actions

    def _rotation_actions(
        self,
        context: ExplorationContext,
        grid: DecisionGrid,
        cache: DecisionCache,
        state: SearchState,
        directions: list[int],
        translation_dirs: set[int],
        *,
        limit: int,
        start_order: int,
        deadline: float | None = None,
    ) -> list[SearchAction]:
        """Return rotation actions for blocked headings with sensing gain."""
        actions: list[SearchAction] = []
        for index, direction in enumerate(directions):
            if index > 0 and self._deadline_reached(deadline):
                break
            if direction in translation_dirs:
                continue
            if self._angular_distance(direction, state.heading) == 0.0:
                continue
            observed_cells = self._visible_unknown_cells(
                context,
                state.position,
                direction,
                state.observed,
                grid=grid,
                cache=cache,
                deadline=deadline,
            )
            if not observed_cells:
                continue
            immediate_gain = self._information_gain(
                context,
                grid,
                cache,
                observed_cells,
            )
            actions.append(
                SearchAction(
                    kind=ROTATE,
                    direction=direction,
                    target=state.position,
                    path=(),
                    observed_cells=observed_cells,
                    immediate_gain=immediate_gain,
                    order=start_order + len(actions),
                )
            )
            if len(actions) >= limit:
                break
        return actions

    def _known_free_path(
        self,
        context: ExplorationContext,
        start: Position,
        target: Position,
        *,
        grid: DecisionGrid | None = None,
        cache: DecisionCache | None = None,
    ) -> tuple[Position, ...]:
        """Return a straight path only when every cell is known free."""
        cache_key = (start, target)
        if cache is not None and cache_key in cache.paths:
            return cache.paths[cache_key]
        if grid is None:
            grid = self._build_grid(context)

        points = tuple(
            bresenham_line_points(
                int(start[0]),
                int(start[1]),
                int(target[0]),
                int(target[1]),
            )
        )
        if not points:
            return ()
        for point in points:
            x, y = point
            if (
                x < 0
                or y < 0
                or x >= grid.width
                or y >= grid.height
                or not bool(grid.known_free[y, x])
            ):
                if cache is not None:
                    cache.paths[cache_key] = ()
                return ()
        path = points[1:]
        if cache is not None:
            cache.paths[cache_key] = path
        return path

    def _apply_action(
        self,
        state: SearchState,
        action: SearchAction,
    ) -> SearchState:
        """Advance a search state by one simulated action."""
        return SearchState(
            position=action.target,
            heading=action.direction,
            depth=state.depth + 1,
            observed=state.observed | action.observed_cells,
        )

    def _visible_unknown_cells(
        self,
        context: ExplorationContext,
        position: Position,
        heading: int,
        already_observed: frozenset[Position],
        *,
        grid: DecisionGrid | None = None,
        cache: DecisionCache | None = None,
        deadline: float | None = None,
    ) -> frozenset[Position]:
        """Simulate a 60-degree cone and return deduped unknown cells."""
        if context.slam_snapshot is None and grid is None:
            return frozenset()

        if grid is None:
            grid = self._build_grid(context)
        cache_key = (
            int(position[0]),
            int(position[1]),
            int(round(heading)) % 360,
        )
        if cache is not None and cache_key in cache.visible:
            visible = cache.visible[cache_key]
        else:
            visible = self._scan_visible_unknown_cells(
                grid,
                (cache_key[0], cache_key[1]),
                cache_key[2],
                deadline=deadline,
            )
            if cache is not None:
                cache.visible[cache_key] = visible

        if not already_observed:
            return visible
        return visible.difference(already_observed)

    def _scan_visible_unknown_cells(
        self,
        grid: DecisionGrid,
        position: Position,
        heading: int,
        *,
        deadline: float | None = None,
    ) -> frozenset[Position]:
        """Return all currently unknown cells visible from a pose."""
        newly_observed: set[Position] = set()
        start = (int(position[0]), int(position[1]))
        if (
            start[0] < 0
            or start[1] < 0
            or start[0] >= grid.width
            or start[1] >= grid.height
        ):
            return frozenset()

        for ray in self._ray_offsets(heading, grid.max_range):
            if newly_observed and self._deadline_reached(deadline):
                break
            for dx, dy in ray:
                x = start[0] + dx
                y = start[1] + dy
                if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
                    break
                if bool(grid.known_occupied[y, x]):
                    break
                if bool(grid.unknown[y, x]):
                    newly_observed.add((x, y))

        return frozenset(newly_observed)

    def _information_gain(
        self,
        context: ExplorationContext,
        grid: DecisionGrid,
        cache: DecisionCache,
        cells: frozenset[Position],
    ) -> float:
        """Return weighted information gain using the exploration hierarchy."""
        return sum(
            self._cell_information_gain(context, grid, cache, cell)
            for cell in cells
        )

    def _cell_information_gain(
        self,
        context: ExplorationContext,
        grid: DecisionGrid,
        cache: DecisionCache,
        cell: Position,
    ) -> float:
        """Score one newly visible cell by information priority tier."""
        cached = cache.cell_gain.get(cell)
        if cached is not None:
            return cached

        x, y = int(cell[0]), int(cell[1])
        if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
            gain = 0.0
        elif bool(grid.unexplored[y, x]):
            nearby_unexplored = self._nearby_mask_count(
                grid.unexplored,
                (x, y),
                radius=self._unexplored_score_radius(context.radius),
            )
            if nearby_unexplored >= self._large_unexplored_threshold(
                context.radius
            ):
                gain = LARGE_UNEXPLORED_CELL_GAIN
            else:
                gain = SMALL_UNEXPLORED_CELL_GAIN
        elif bool(grid.low_confidence_known[y, x]):
            threshold = max(1e-6, context.frontier_confidence_threshold)
            confidence_gap = max(
                0.0,
                threshold - float(grid.confidence[y, x]),
            ) / threshold
            gain = LOW_CONFIDENCE_CELL_GAIN * max(0.25, confidence_gap)
        else:
            gain = 0.0

        cache.cell_gain[cell] = gain
        return gain

    def _ray_offsets(
        self,
        heading: int,
        max_range: int,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return cached relative cells for each planning ray."""
        normalized_heading = int(round(heading)) % 360
        key = (
            int(self.config.planning_rays),
            int(max_range),
            normalized_heading,
        )
        cached = self._ray_offset_cache.get(key)
        if cached is not None:
            return cached

        num_rays = max(1, int(self.config.planning_rays))
        half_fov = SENSOR_FOV_DEG / 2.0
        rays: list[tuple[tuple[int, int], ...]] = []
        for index in range(num_rays):
            if num_rays == 1:
                angle = float(normalized_heading)
            else:
                fraction = index / (num_rays - 1)
                angle = (
                    float(normalized_heading)
                    - half_fov
                    + fraction * SENSOR_FOV_DEG
                )
            radians = math.radians(angle)
            dx_unit = math.sin(radians)
            dy_unit = -math.cos(radians)
            previous: tuple[int, int] | None = None
            cells: list[tuple[int, int]] = []
            for length in range(max(1, int(max_range)) + 1):
                cell = (
                    int(round(length * dx_unit)),
                    int(round(length * dy_unit)),
                )
                if cell == previous:
                    continue
                previous = cell
                cells.append(cell)
            rays.append(tuple(cells))

        offsets = tuple(rays)
        self._ray_offset_cache[key] = offsets
        return offsets

    def _ray_endpoint(
        self,
        context: ExplorationContext,
        start: Position,
        angle_deg: float,
        max_range: int,
    ) -> Position:
        """Return the endpoint of a simulated planning ray."""
        radians = math.radians(angle_deg)
        dx = math.sin(radians)
        dy = -math.cos(radians)
        last = start
        for length in range(0, max_range + 1):
            point = (
                int(round(start[0] + length * dx)),
                int(round(start[1] + length * dy)),
            )
            if not self._inside(context, point):
                break
            last = point
            if self._is_known_occupied(context, point):
                break
        return last

    def _frontier_cluster_centroids(
        self,
        context: ExplorationContext,
        position: Position,
        *,
        grid: DecisionGrid | None = None,
        deadline: float | None = None,
    ) -> tuple[Position, ...]:
        """Return nearest representatives of 8-connected frontier clusters."""
        limit = self.config.frontier_cluster_limit
        if limit <= 0:
            return ()
        if context.slam_snapshot is None and grid is None:
            return ()
        if grid is None:
            grid = self._build_grid(context)
        known_free = grid.known_free
        unknown = grid.unknown
        height, width = grid.height, grid.width

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

        frontier_mask = known_free & neighbor_unknown
        visited = np.zeros_like(frontier_mask, dtype=bool)
        representatives: list[Position] = []
        masks = FrontierPriorityMasks(
            unexplored=grid.unexplored,
            low_confidence=grid.low_confidence_known,
        )
        candidate_limit = max(limit, limit * 4)
        ys, xs = np.where(frontier_mask)
        candidate_indexes = self._nearest_frontier_candidate_indexes(
            xs,
            ys,
            position,
            limit=limit,
            radius=context.radius,
        )
        frontier_cells = sorted(
            ((int(xs[index]), int(ys[index])) for index in candidate_indexes),
            key=lambda cell: (
                *self._frontier_priority_key(
                    context,
                    cell,
                    masks=masks,
                )[:3],
                self._frontier_cell_priority(position, cell),
            ),
        )
        for index, (x, y) in enumerate(frontier_cells):
            if len(representatives) >= candidate_limit:
                break
            if index > 0 and self._deadline_reached(deadline):
                break
            if visited[y, x]:
                continue
            cluster = self._collect_frontier_cluster(
                frontier_mask,
                visited,
                (int(x), int(y)),
                deadline=deadline,
            )
            if not cluster:
                continue
            centroid_x = sum(cell[0] for cell in cluster) / len(cluster)
            centroid_y = sum(cell[1] for cell in cluster) / len(cluster)
            representatives.append(
                min(
                    cluster,
                    key=lambda cell: (
                        math.dist(cell, (centroid_x, centroid_y)),
                        cell[1],
                        cell[0],
                    ),
                )
            )

        representatives.sort(
            key=lambda cell: (
                *self._frontier_priority_key(
                    context,
                    cell,
                    masks=masks,
                )[:3],
                self._frontier_cell_priority(position, cell),
            )
        )
        return tuple(representatives[:limit])

    def _nearest_frontier_candidate_indexes(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        position: Position,
        *,
        limit: int,
        radius: int,
    ) -> np.ndarray:
        """Return a bounded nearest-frontier candidate pool for root hints."""
        candidate_count = max(
            ROOT_FRONTIER_MIN_CANDIDATES,
            max(1, int(limit)) * ROOT_FRONTIER_CANDIDATES_PER_RESULT,
        )
        total = len(xs)
        if total <= candidate_count:
            return np.arange(total)

        dx = xs.astype(np.int64, copy=False) - int(position[0])
        dy = ys.astype(np.int64, copy=False) - int(position[1])
        distance_sq = dx * dx + dy * dy
        nearest_count = max(1, candidate_count // 2)
        nearest_indexes = np.argpartition(distance_sq, nearest_count - 1)[
            :nearest_count
        ]
        selected: set[int] = {int(index) for index in nearest_indexes}
        diverse_needed = max(0, candidate_count - len(selected))
        if diverse_needed <= 0:
            return np.fromiter(selected, dtype=np.int64)

        bin_size = self._frontier_spatial_bin_size(radius)
        sample_count = min(total, max(diverse_needed * 4, diverse_needed))
        step = max(1, total // max(1, sample_count))
        offset = self.seed % step
        sampled_indexes = np.arange(offset, total, step, dtype=np.int64)[
            :sample_count
        ]
        bins: dict[tuple[int, int], int] = {}
        for raw_index in sampled_indexes:
            index = int(raw_index)
            if index in selected:
                continue
            x = int(xs[index])
            y = int(ys[index])
            key = (x // bin_size, y // bin_size)
            if key not in bins:
                bins[key] = index

        diverse = sorted(
            (
                (self._spatial_bin_tiebreak(key), int(distance_sq[index]), index)
                for key, index in bins.items()
            ),
            key=lambda item: (item[0], item[1]),
        )
        for _, _, index in diverse:
            selected.add(int(index))
            if len(selected) >= candidate_count:
                break

        if len(selected) < candidate_count:
            for raw_index in sampled_indexes:
                selected.add(int(raw_index))
                if len(selected) >= candidate_count:
                    break
        return np.fromiter(selected, dtype=np.int64)

    def _collect_frontier_cluster(
        self,
        frontier_mask: np.ndarray,
        visited: np.ndarray,
        start: Position,
        *,
        deadline: float | None = None,
    ) -> list[Position]:
        """Collect one 8-connected frontier component."""
        height, width = frontier_mask.shape
        stack = [start]
        cluster: list[Position] = []
        visited[start[1], start[0]] = True
        while stack:
            if (
                cluster
                and len(cluster) % 256 == 0
                and self._deadline_reached(deadline)
            ):
                break
            x, y = stack.pop()
            cluster.append((x, y))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if visited[ny, nx] or not frontier_mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))
        return cluster

    def _append_direction(self, directions: list[int], direction: int) -> None:
        """Append a heading unless it is a near-duplicate."""
        normalized = int(round(direction)) % 360
        if all(
            self._angular_distance(normalized, existing)
            >= NEAR_DUPLICATE_DEG
            for existing in directions
        ):
            directions.append(normalized)

    def _build_grid(self, context: ExplorationContext) -> DecisionGrid:
        """Build reusable SLAM masks for one decision."""
        if context.slam_snapshot is None:
            raise ValueError("MCTS planning requires a SLAM snapshot")

        occupancy = context.slam_snapshot.occupancy
        confidence = context.slam_snapshot.confidence
        threshold = context.frontier_confidence_threshold
        known_free = (occupancy == FREE) & (confidence >= threshold)
        known_occupied = (occupancy == OCCUPIED) & (confidence >= threshold)
        unexplored = occupancy == UNKNOWN
        low_confidence_known = (occupancy != UNKNOWN) & (
            confidence < threshold
        )
        unknown = unexplored | low_confidence_known
        height, width = occupancy.shape
        return DecisionGrid(
            occupancy=occupancy,
            confidence=confidence,
            known_free=known_free,
            known_occupied=known_occupied,
            unexplored=unexplored,
            low_confidence_known=low_confidence_known,
            unknown=unknown,
            height=height,
            width=width,
            max_range=max(
                1,
                int(context.radius * LIDAR_RANGE_RADIUS_MULTIPLIER),
            ),
            slam_version=int(context.slam_snapshot.version),
        )

    @staticmethod
    def _build_cache() -> DecisionCache:
        """Build empty caches for one rebuilt MCTS tree."""
        return DecisionCache(
            directions={},
            paths={},
            visible={},
            cell_gain={},
        )

    def _is_known_free(
        self,
        context: ExplorationContext,
        point: Position,
    ) -> bool:
        """Return True if SLAM says a cell is confidently traversable."""
        if context.slam_snapshot is None or not self._inside(context, point):
            return False
        x, y = point
        return (
            int(context.slam_snapshot.occupancy[y, x]) == FREE
            and float(context.slam_snapshot.confidence[y, x])
            >= context.frontier_confidence_threshold
        )

    def _is_known_occupied(
        self,
        context: ExplorationContext,
        point: Position,
    ) -> bool:
        """Return True if SLAM says a cell is a confident wall."""
        if context.slam_snapshot is None or not self._inside(context, point):
            return False
        x, y = point
        return (
            int(context.slam_snapshot.occupancy[y, x]) == OCCUPIED
            and float(context.slam_snapshot.confidence[y, x])
            >= context.frontier_confidence_threshold
        )

    def _is_unknown(
        self,
        context: ExplorationContext,
        point: Position,
    ) -> bool:
        """Return True when a cell is not confidently known."""
        if context.slam_snapshot is None or not self._inside(context, point):
            return False
        x, y = point
        return (
            int(context.slam_snapshot.occupancy[y, x]) == UNKNOWN
            or float(context.slam_snapshot.confidence[y, x])
            < context.frontier_confidence_threshold
        )

    def _inside(self, context: ExplorationContext, point: Position) -> bool:
        """Return whether a point is inside the SLAM grid."""
        if context.slam_snapshot is None:
            return False
        x, y = point
        height, width = context.slam_snapshot.occupancy.shape
        return 0 <= x < width and 0 <= y < height

    def _decision_seed(self, context: ExplorationContext) -> int:
        """Return a deterministic per-snapshot RNG seed."""
        position = context.pose_estimate.position
        version = (
            0
            if context.slam_snapshot is None
            else int(context.slam_snapshot.version)
        )
        heading = int(round(context.pose_estimate.heading_deg)) % 360
        return (
            self.seed
            + version * 1_000_003
            + position[0] * 9_176
            + position[1] * 13_063
            + heading * 97
        ) & 0xFFFFFFFF

    @staticmethod
    def _direction_to(start: Position, target: Position) -> int:
        """Return a gameplay heading from start to target."""
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        if dx == 0 and dy == 0:
            return 0
        return int(round(math.degrees(math.atan2(dx, -dy)))) % 360

    @staticmethod
    def _angular_distance(left: int, right: int) -> float:
        """Return the smallest absolute angle between headings."""
        return abs((int(left) - int(right) + 180) % 360 - 180)

    def _frontier_cell_priority(
        self,
        position: Position,
        cell: Position,
    ) -> tuple[int, int]:
        """Return a direction-neutral priority for budgeted frontier scans."""
        dx = int(cell[0]) - int(position[0])
        dy = int(cell[1]) - int(position[1])
        distance_sq = dx * dx + dy * dy
        return (
            distance_sq,
            self._coordinate_tiebreak(cell),
        )

    def _coordinate_tiebreak(self, cell: Position) -> int:
        """Return a stable per-drone tie-break without absolute x/y bias."""
        x, y = int(cell[0]), int(cell[1])
        return (
            (x * 73_856_093)
            ^ (y * 19_349_663)
            ^ (self.seed * 83_492_791)
        ) & 0xFFFFFFFF

    @staticmethod
    def _frontier_spatial_bin_size(radius: int) -> int:
        """Return the coarse bin size used for global frontier diversity."""
        return max(16, int(radius) * 2)

    def _spatial_bin_tiebreak(self, key: tuple[int, int]) -> int:
        """Return a deterministic per-drone ordering for coarse map bins."""
        x, y = int(key[0]), int(key[1])
        return (
            (x * 73_856_093)
            ^ (y * 19_349_663)
            ^ (self.seed * 83_492_791)
        ) & 0xFFFFFFFF

    @staticmethod
    def _stable_action_order(
        action: SearchAction | None,
    ) -> tuple[int, int]:
        """Return stable ordering values for tie-breaking."""
        if action is None:
            return 360, 9
        kind_order = 0 if action.kind == TRANSLATE else 1
        return action.direction, kind_order

    def _empty_diagnostics(
        self,
        context: ExplorationContext,
        *,
        elapsed_ms: float = 0.0,
    ) -> MctsSearchDiagnostics:
        """Return diagnostics for a search that had no actions."""
        version = (
            0
            if context.slam_snapshot is None
            else int(context.slam_snapshot.version)
        )
        return MctsSearchDiagnostics(
            iterations=0,
            root_visits=(),
            selected_reward=0.0,
            generated_nodes=0,
            slam_version=version,
            elapsed_ms=elapsed_ms,
        )

    def _diagnostics(
        self,
        context: ExplorationContext,
        root: MctsNode,
        *,
        iterations: int,
        selected_reward: float,
        generated_nodes: int,
        selected_action: SearchAction | None = None,
        elapsed_ms: float = 0.0,
    ) -> MctsSearchDiagnostics:
        """Build immutable diagnostics from the completed root node."""
        root_visits = tuple(
            RootVisitDiagnostic(
                kind=child.incoming_action.kind,
                direction=child.incoming_action.direction,
                target=child.incoming_action.target,
                visits=child.visits,
                mean_reward=child.mean_reward,
            )
            for child in sorted(
                root.children,
                key=lambda child: self._stable_action_order(
                    child.incoming_action
                ),
            )
            if child.incoming_action is not None
        )
        version = (
            0
            if context.slam_snapshot is None
            else int(context.slam_snapshot.version)
        )
        return MctsSearchDiagnostics(
            iterations=iterations,
            root_visits=root_visits,
            selected_reward=selected_reward,
            generated_nodes=generated_nodes,
            slam_version=version,
            selected_kind="" if selected_action is None else selected_action.kind,
            selected_direction=(
                None if selected_action is None else selected_action.direction
            ),
            selected_target=(
                None if selected_action is None else selected_action.target
            ),
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _diagnostics_for_frontier(
        diagnostics: MctsSearchDiagnostics,
        decision: ExplorationDecision,
        *,
        elapsed_ms: float | None = None,
    ) -> MctsSearchDiagnostics:
        """Mark diagnostics with the executed frontier fallback decision."""
        return replace(
            diagnostics,
            selected_kind=decision.kind.value,
            selected_direction=decision.direction,
            selected_target=decision.target,
            elapsed_ms=(
                diagnostics.elapsed_ms if elapsed_ms is None else elapsed_ms
            ),
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        """Return elapsed milliseconds since a perf-counter timestamp."""
        return (time.perf_counter() - started_at) * 1000.0

    def _decision_deadline(self, started_at: float) -> float | None:
        """Return the perf-counter deadline for this decision, if enabled."""
        budget_ms = float(self.config.decision_time_budget_ms)
        if budget_ms <= 0.0:
            return None
        return started_at + (budget_ms / 1000.0)

    def _search_deadline(
        self,
        deadline: float | None,
        context: ExplorationContext,
    ) -> float | None:
        """Reserve a small slice for budgeted frontier fallback sorting."""
        if deadline is None or not context.runtime_snapshot.frontiers:
            return deadline
        budget_seconds = float(self.config.decision_time_budget_ms) / 1000.0
        reserve_seconds = min(0.010, budget_seconds * 0.25)
        return deadline - reserve_seconds

    @staticmethod
    def _deadline_reached(deadline: float | None) -> bool:
        """Return whether a cooperative search deadline has elapsed."""
        return deadline is not None and time.perf_counter() >= deadline
