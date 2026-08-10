"""Deadline-safe, goal-conditioned local MCTS navigation controller.

The controller consumes a detached SLAM snapshot and a persistent
``NavigationIntent``.  It never extracts frontiers or constructs whole-map
planning masks: every search is limited to a fixed-size window around the
current pose and the immediately relevant route prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import random
import time
from typing import Callable, Iterable

import numpy as np

from asset_config.helpers import next_cell_coords
from config.simulation_config import ExplorationConfig
from mapping.ray_geometry import bresenham_line_points
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from navigation.navigation_intent import MovementMode, NavigationIntent


Position = tuple[int, int]

DEVIATION_DEGREES = 15.0
SCAN_ROTATION_DEGREES = 60.0
DIAGNOSTIC_RESERVE_FRACTION = 0.30
LOCAL_WINDOW_RADIUS_CAP = 96
LOCAL_WINDOW_RADIUS_MIN = 16
LOCAL_PLANNING_RAY_CAP = 32


class LocalPrimitive(str, Enum):
    """The complete bounded root vocabulary for local navigation."""

    FOLLOW_EDGE = "follow_edge"
    DEVIATE_LEFT = "deviate_left"
    DEVIATE_RIGHT = "deviate_right"
    ROTATE_SCAN = "rotate_scan"
    RECOVERY = "recovery"


ROOT_PRIMITIVE_ORDER = (
    LocalPrimitive.FOLLOW_EDGE,
    LocalPrimitive.DEVIATE_LEFT,
    LocalPrimitive.DEVIATE_RIGHT,
    LocalPrimitive.ROTATE_SCAN,
    LocalPrimitive.RECOVERY,
)


@dataclass(frozen=True)
class LocalRewardComponents:
    """Normalized components of the locked Phase 5 reward function."""

    route_progress: float = 0.0
    information_gain: float = 0.0
    revisit: float = 0.0
    oscillation: float = 0.0
    target_switch: float = 0.0
    turn: float = 0.0
    time_energy: float = 0.0
    collision_risk: float = 0.0

    def __post_init__(self) -> None:
        """Clamp every component to its documented normalized interval."""
        for name in (
            "route_progress",
            "information_gain",
            "revisit",
            "oscillation",
            "target_switch",
            "turn",
            "time_energy",
            "collision_risk",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                value = 0.0
            object.__setattr__(self, name, min(1.0, max(0.0, value)))

    @property
    def score(self) -> float:
        """Return the locked normalized Phase 5 weighted reward."""
        return (
            2.0 * self.route_progress
            + 3.0 * self.information_gain
            - 1.5 * self.revisit
            - 2.0 * self.oscillation
            - 2.5 * self.target_switch
            - 0.25 * self.turn
            - 0.25 * self.time_energy
            - 4.0 * self.collision_risk
        )


@dataclass(frozen=True)
class LocalMctsRequest:
    """Detached inputs for one local, goal-conditioned decision."""

    position: Position
    heading_deg: float
    step: int
    radius: int
    intent: NavigationIntent
    slam_snapshot: SlamSnapshot | None = None
    slam_snapshot_provider: (
        Callable[[tuple[int, int, int, int]], SlamSnapshot] | None
    ) = None
    slam_shape: tuple[int, int] | None = None
    slam_version_hint: int = 0
    recent_visits: tuple[object, ...] = ()
    previous_primitive: LocalPrimitive | None = None
    stalled: bool = False
    confidence_threshold: float = 0.6
    recovery_path: tuple[Position, ...] = ()


@dataclass(frozen=True)
class LocalSearchState:
    """Goal, route, history, and stall state carried through simulation."""

    position: Position
    heading_deg: float
    goal_id: object
    route_edge_ids: tuple[int, ...]
    route_segment_edge_ids: tuple[int | None, ...]
    route_edge_cursor: int
    polyline_cursor: int
    remaining_route_cost: float
    recent_visits: tuple[object, ...]
    previous_primitive: LocalPrimitive | None
    stalled: bool
    local_route_cursor: int = 0
    local_recovery_cursor: int = 0
    observed_gain_cells: frozenset[Position] = frozenset()
    depth: int = 0


@dataclass(frozen=True)
class LocalAction:
    """One bounded primitive instantiated for a particular search state."""

    primitive: LocalPrimitive
    target: Position
    heading_deg: float
    path: tuple[Position, ...]
    order: int
    route_cursor_delta: int = 0
    recovery_cursor_delta: int = 0
    advances_active_route: bool = False
    result_edge_cursor: int = 0
    result_polyline_cursor: int = 0


@dataclass(frozen=True)
class LocalRootDiagnostic:
    """Stable diagnostics for one root arm."""

    primitive: LocalPrimitive
    target: Position
    heading_deg: float
    visits: int
    mean_reward: float
    initial_reward: float


@dataclass(frozen=True)
class LocalMctsDiagnostics:
    """Immutable search diagnostics suitable for runtime tracing."""

    iterations: int
    root_visits: tuple[LocalRootDiagnostic, ...]
    selected_primitive: LocalPrimitive
    selected_reward: float
    generated_nodes: int
    root_coverage_complete: bool
    overrun_stage: str | None
    fallback_primitive: LocalPrimitive | None
    elapsed_ms: float
    budget_ms: float
    search_budget_ms: float
    reserved_budget_ms: float
    window_bounds: tuple[int, int, int, int]
    preprocessing_cells: int
    deadline_checks: tuple[tuple[str, int], ...]
    slam_version: int


@dataclass(frozen=True)
class LocalMctsDecision:
    """Selected local primitive and its bounded execution geometry."""

    primitive: LocalPrimitive
    target: Position
    heading_deg: float
    path: tuple[Position, ...]
    diagnostics: LocalMctsDiagnostics


@dataclass(frozen=True)
class _LocalBeliefWindow:
    left: int
    top: int
    right: int
    bottom: int
    occupancy: np.ndarray
    confidence: np.ndarray
    route_prefix: tuple[Position, ...]
    route_segment_indexes: tuple[int, ...]
    route_polyline_cursors: tuple[int, ...]
    slam_version: int

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    @property
    def cell_count(self) -> int:
        return max(0, self.right - self.left) * max(0, self.bottom - self.top)

    def contains(self, position: Position) -> bool:
        x, y = position
        return self.left <= x < self.right and self.top <= y < self.bottom

    def values(self, position: Position) -> tuple[int, float] | None:
        if not self.contains(position):
            return None
        x, y = position
        return (
            int(self.occupancy[y - self.top, x - self.left]),
            float(self.confidence[y - self.top, x - self.left]),
        )


@dataclass
class _RootArm:
    action: LocalAction
    visits: int = 0
    total_reward: float = 0.0
    initial_reward: float = 0.0
    immediate_reward: float = 0.0
    initial_gain_cells: frozenset[Position] = frozenset()

    @property
    def mean_reward(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.total_reward / self.visits


class _DeadlineExceeded(RuntimeError):
    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


class _DeadlineGuard:
    def __init__(
        self,
        clock: Callable[[], float],
        deadline: float | None,
    ) -> None:
        self.clock = clock
        self.deadline = deadline
        self.counts: dict[str, int] = {}

    def check(self, stage: str) -> None:
        self.counts[stage] = self.counts.get(stage, 0) + 1
        if self.deadline is not None and self.clock() >= self.deadline:
            raise _DeadlineExceeded(stage)

    def snapshot(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self.counts.items()))


class LocalMctsController:
    """Small MCTS controller over five goal-conditioned local primitives."""

    def __init__(
        self,
        config: ExplorationConfig,
        *,
        seed: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self._clock = clock or time.perf_counter

    def build_search_state(self, request: LocalMctsRequest) -> LocalSearchState:
        """Build the required persistent goal-conditioned search state."""
        intent = request.intent
        goal_id: object
        if intent.cluster_id is not None:
            goal_id = int(intent.cluster_id)
        elif intent.mode == MovementMode.HOME:
            goal_id = ("home", intent.target)
        else:
            goal_id = (intent.mode.value, intent.target)
        previous = request.previous_primitive
        if previous is None and intent.previous_primitive:
            try:
                previous = LocalPrimitive(intent.previous_primitive)
            except ValueError:
                previous = None
        return LocalSearchState(
            position=(int(request.position[0]), int(request.position[1])),
            heading_deg=float(request.heading_deg) % 360.0,
            goal_id=goal_id,
            route_edge_ids=tuple(intent.route_edge_ids),
            route_segment_edge_ids=tuple(intent.route_segment_edge_ids),
            route_edge_cursor=max(0, int(intent.edge_cursor)),
            polyline_cursor=max(0, int(intent.polyline_cursor)),
            remaining_route_cost=max(0.0, float(intent.remaining_route_cost)),
            recent_visits=tuple(request.recent_visits),
            previous_primitive=previous,
            stalled=bool(request.stalled),
        )

    def decide(self, request: LocalMctsRequest) -> LocalMctsDecision:
        """Run one bounded search, falling back safely on incomplete coverage."""
        started_at = self._clock()
        budget_ms = max(0.0, float(self.config.decision_time_budget_ms))
        search_budget_ms = budget_ms * (1.0 - DIAGNOSTIC_RESERVE_FRACTION)
        reserved_budget_ms = budget_ms - search_budget_ms
        deadline = (
            None
            if budget_ms <= 0.0
            else started_at + search_budget_ms / 1000.0
        )
        guard = _DeadlineGuard(self._clock, deadline)
        state = self.build_search_state(request)
        safe = self._safe_primitive(request.intent.mode)
        window: _LocalBeliefWindow | None = None
        actions: tuple[LocalAction, ...] = ()
        arms: list[_RootArm] = []
        iterations = 0
        generated_nodes = 1
        overrun_stage: str | None = None
        coverage_complete = False
        selected_action = self._stationary_action(safe, state)

        try:
            guard.check("preprocessing")
            window = self._build_local_window(request, guard)
            guard.check("root_generation")
            actions = self._build_actions(request, window, state)
            arms = [_RootArm(action=action) for action in actions]
            generated_nodes += len(arms)
            selected_action = next(
                (action for action in actions if action.primitive == safe),
                actions[0] if actions else selected_action,
            )
            rng = random.Random(self._decision_seed(state))

            for arm in arms:
                guard.check("root_evaluation")
                components, gain_cells = self._evaluate_action(
                    request,
                    window,
                    state,
                    arm.action,
                    guard,
                )
                immediate_reward = components.score
                reward = immediate_reward
                simulation_state = self._transition_state(
                    state,
                    arm.action,
                    immediate_reward,
                    gain_cells=gain_cells,
                )
                discount = float(self.config.discount)
                for depth in range(1, self.config.horizon):
                    guard.check("simulation_depth")
                    simulated_actions = self._build_actions(
                        request,
                        window,
                        simulation_state,
                    )
                    if not simulated_actions:
                        break
                    guard.check("expansion")
                    simulated = self._choose_simulation_action(
                        simulated_actions,
                        rng,
                    )
                    rollout_components, rollout_gain = self._evaluate_action(
                        request,
                        window,
                        simulation_state,
                        simulated,
                        guard,
                    )
                    reward += (
                        discount ** depth
                    ) * rollout_components.score
                    simulation_state = self._transition_state(
                        simulation_state,
                        simulated,
                        rollout_components.score,
                        gain_cells=rollout_gain,
                    )
                    generated_nodes += 1
                arm.visits = 1
                arm.total_reward = reward
                arm.initial_reward = reward
                arm.immediate_reward = immediate_reward
                arm.initial_gain_cells = gain_cells
            coverage_complete = bool(arms) and all(
                arm.visits == 1 for arm in arms
            )

            # UCT only distinguishes competing roots.  Re-simulating the sole
            # scan root cannot alter the decision and used to consume nearly
            # the entire deadline on every sensor-gated rotation tick.
            while (
                coverage_complete
                and len(arms) > 1
                and iterations < self.config.iterations
            ):
                guard.check("expansion")
                arm = self._select_uct_arm(arms)
                guard.check("expansion")
                simulation_state = self._transition_state(
                    state,
                    arm.action,
                    arm.immediate_reward,
                    gain_cells=arm.initial_gain_cells,
                )
                reward = arm.immediate_reward
                discount = float(self.config.discount)
                for depth in range(1, self.config.horizon):
                    guard.check("simulation_depth")
                    simulated_actions = self._build_actions(
                        request,
                        window,
                        simulation_state,
                    )
                    if not simulated_actions:
                        break
                    guard.check("expansion")
                    simulated = self._choose_simulation_action(
                        simulated_actions,
                        rng,
                    )
                    components, gain_cells = self._evaluate_action(
                        request,
                        window,
                        simulation_state,
                        simulated,
                        guard,
                    )
                    reward += (discount ** depth) * components.score
                    simulation_state = self._transition_state(
                        simulation_state,
                        simulated,
                        components.score,
                        gain_cells=gain_cells,
                    )
                    generated_nodes += 1
                arm.visits += 1
                arm.total_reward += reward
                iterations += 1
        except _DeadlineExceeded as exc:
            overrun_stage = exc.stage

        if coverage_complete:
            selected_arm = self._select_best_arm(arms, safe)
            selected_action = selected_arm.action
            selected_reward = selected_arm.mean_reward
            fallback = None
        else:
            # Partial root evaluations are diagnostic only.  They must never
            # bias an incomplete search away from the deterministic safe path.
            iterations = 0
            selected_reward = 0.0
            fallback = safe

        elapsed_ms = max(0.0, (self._clock() - started_at) * 1000.0)
        bounds = window.bounds if window is not None else (0, 0, 0, 0)
        cells = window.cell_count if window is not None else 0
        root_diagnostics = tuple(
            LocalRootDiagnostic(
                primitive=arm.action.primitive,
                target=arm.action.target,
                heading_deg=arm.action.heading_deg,
                visits=arm.visits,
                mean_reward=arm.mean_reward,
                initial_reward=arm.initial_reward,
            )
            for arm in arms
        )
        diagnostics = LocalMctsDiagnostics(
            iterations=iterations,
            root_visits=root_diagnostics,
            selected_primitive=selected_action.primitive,
            selected_reward=selected_reward,
            generated_nodes=generated_nodes,
            root_coverage_complete=coverage_complete,
            overrun_stage=overrun_stage,
            fallback_primitive=fallback,
            elapsed_ms=elapsed_ms,
            budget_ms=budget_ms,
            search_budget_ms=search_budget_ms,
            reserved_budget_ms=reserved_budget_ms,
            window_bounds=bounds,
            preprocessing_cells=cells,
            deadline_checks=guard.snapshot(),
            slam_version=(
                window.slam_version
                if window is not None
                else (
                    int(request.slam_snapshot.version)
                    if request.slam_snapshot is not None
                    else int(request.slam_version_hint)
                )
            ),
        )
        return LocalMctsDecision(
            primitive=selected_action.primitive,
            target=selected_action.target,
            heading_deg=selected_action.heading_deg,
            path=selected_action.path,
            diagnostics=diagnostics,
        )

    def predict_gain_cells(
        self,
        request: LocalMctsRequest,
        *,
        position: Position,
        heading_deg: float,
    ) -> tuple[Position, ...]:
        """Return first-occluder gain cells for characterization and tooling."""
        guard = _DeadlineGuard(self._clock, None)
        window = self._build_local_window(request, guard)
        return self._predicted_gain_cells(
            request,
            window,
            position,
            heading_deg,
            frozenset(),
            guard,
        )

    def _build_local_window(
        self,
        request: LocalMctsRequest,
        guard: _DeadlineGuard,
    ) -> _LocalBeliefWindow:
        """Copy only a bounded pose-centered belief region and route prefix."""
        source = request.slam_snapshot
        shape = request.slam_shape
        if shape is None and source is not None:
            shape = source.full_shape or source.occupancy.shape
        if shape is None:
            raise ValueError("local MCTS requires a SLAM shape or snapshot")
        height, width = int(shape[0]), int(shape[1])
        window_radius = min(
            LOCAL_WINDOW_RADIUS_CAP,
            max(
                LOCAL_WINDOW_RADIUS_MIN,
                int(request.step) * (int(self.config.horizon) + 2),
                int(request.radius) * 2,
            ),
        )
        x, y = request.position
        left = max(0, int(x) - window_radius)
        top = max(0, int(y) - window_radius)
        right = min(width, int(x) + window_radius + 1)
        bottom = min(height, int(y) + window_radius + 1)

        if request.slam_snapshot_provider is not None:
            guard.check("preprocessing")
            source = request.slam_snapshot_provider((left, top, right, bottom))
            if source is None:
                raise _DeadlineExceeded("preprocessing_lock")
            guard.check("preprocessing")
        if source is None:
            raise ValueError("local MCTS requires a detached SLAM window")
        occupancy = np.asarray(source.occupancy)
        confidence = np.asarray(source.confidence)
        if occupancy.ndim != 2 or confidence.shape != occupancy.shape:
            raise ValueError("SLAM occupancy and confidence must be aligned 2-D arrays")
        source_left, source_top = source.origin
        source_right = source_left + occupancy.shape[1]
        source_bottom = source_top + occupancy.shape[0]
        left = max(left, source_left)
        top = max(top, source_top)
        right = min(right, source_right)
        bottom = min(bottom, source_bottom)
        source_x0 = left - source_left
        source_y0 = top - source_top
        source_x1 = right - source_left
        source_y1 = bottom - source_top

        route_prefix: list[Position] = [(int(x), int(y))]
        intent = request.intent
        remaining_budget = max(
            float(request.step) * (self.config.horizon + 2),
            float(request.step),
        )
        travelled = 0.0
        edge_cursor = max(0, int(intent.edge_cursor))
        polyline_cursor = max(0, int(intent.polyline_cursor))
        route_segment_indexes = [edge_cursor]
        route_polyline_cursors = [polyline_cursor]
        for path_index in range(edge_cursor, len(intent.route_paths)):
            guard.check("preprocessing")
            vertices = intent.route_paths[path_index]
            if path_index == edge_cursor:
                raster, raster_capped = self._raster_suffix_after_cursor(
                    vertices,
                    polyline_cursor,
                    request.position,
                    guard,
                )
            else:
                raster, raster_capped = self._rasterize(
                    vertices,
                    guard,
                    "preprocessing",
                )
            base_cursor = polyline_cursor if path_index == edge_cursor else 0
            truncated = raster_capped
            for local_cursor, point in enumerate(raster[1:], start=1):
                guard.check("preprocessing")
                step_distance = math.dist(route_prefix[-1], point)
                if travelled + step_distance > remaining_budget + 1e-9:
                    truncated = True
                    break
                if not (left <= point[0] < right and top <= point[1] < bottom):
                    truncated = True
                    break
                route_prefix.append(point)
                route_segment_indexes.append(path_index)
                route_polyline_cursors.append(base_cursor + local_cursor)
                travelled += step_distance
            if not truncated:
                # The executor normalizes a fully consumed segment immediately,
                # even when the step budget ends exactly on its endpoint.
                # Mirror `(next_edge, 0)` (or terminal `(len(paths), 0)`) so
                # simulated visit identity and completion match execution.
                route_segment_indexes[-1] = path_index + 1
                route_polyline_cursors[-1] = 0
            if truncated or travelled >= remaining_budget - 1e-9:
                break
            polyline_cursor = 0

        guard.check("preprocessing")
        local_occupancy = np.array(
            occupancy[source_y0:source_y1, source_x0:source_x1],
            dtype=np.int8,
            copy=True,
        )
        guard.check("preprocessing")
        local_confidence = np.array(
            confidence[source_y0:source_y1, source_x0:source_x1],
            dtype=np.float32,
            copy=True,
        )
        guard.check("preprocessing")
        return _LocalBeliefWindow(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            occupancy=local_occupancy,
            confidence=local_confidence,
            route_prefix=tuple(route_prefix),
            route_segment_indexes=tuple(route_segment_indexes),
            route_polyline_cursors=tuple(route_polyline_cursors),
            slam_version=int(source.version),
        )

    def _build_actions(
        self,
        request: LocalMctsRequest,
        window: _LocalBeliefWindow,
        state: LocalSearchState,
    ) -> tuple[LocalAction, ...]:
        """Instantiate only the fixed, mode-appropriate root primitives."""
        if request.intent.mode == MovementMode.SCAN:
            primitives = (LocalPrimitive.ROTATE_SCAN,)
        elif request.intent.mode == MovementMode.RECOVERY:
            # Both formerly followed the same stored recovery polyline and
            # produced identical rewards. Keep only the mode-safe identity so
            # the mandatory root evaluation completes without pointless UCT.
            primitives = (LocalPrimitive.RECOVERY,)
        else:
            primitives = ROOT_PRIMITIVE_ORDER
        return tuple(
            self._action_for_primitive(request, window, state, primitive, order)
            for order, primitive in enumerate(primitives)
        )

    def _action_for_primitive(
        self,
        request: LocalMctsRequest,
        window: _LocalBeliefWindow,
        state: LocalSearchState,
        primitive: LocalPrimitive,
        order: int,
    ) -> LocalAction:
        next_route_points = window.route_prefix[state.local_route_cursor + 1:]
        route_tail = (state.position, *next_route_points)
        route_heading = self._route_heading(route_tail, state.heading_deg)
        route_cursor_delta = 0
        recovery_cursor_delta = 0
        advances_active_route = False
        result_edge_cursor = state.route_edge_cursor
        result_polyline_cursor = state.polyline_cursor
        if primitive == LocalPrimitive.FOLLOW_EDGE:
            path = self._bounded_path_from_position(
                state.position,
                route_tail,
                float(request.step),
            )
            route_cursor_delta = max(0, len(path) - 1)
            advances_active_route = True
            target_route_cursor = min(
                len(window.route_prefix) - 1,
                state.local_route_cursor + route_cursor_delta,
            )
            result_edge_cursor = window.route_segment_indexes[
                target_route_cursor
            ]
            result_polyline_cursor = window.route_polyline_cursors[
                target_route_cursor
            ]
            target = path[-1] if path else state.position
            heading = self._path_heading(path, route_heading)
        elif primitive in {
            LocalPrimitive.DEVIATE_LEFT,
            LocalPrimitive.DEVIATE_RIGHT,
        }:
            sign = -1.0 if primitive == LocalPrimitive.DEVIATE_LEFT else 1.0
            heading = (route_heading + sign * DEVIATION_DEGREES) % 360.0
            target = self._bounded_step_target(
                state.position,
                max(1, int(request.step)),
                heading,
            )
            path = (state.position, target)
        elif primitive == LocalPrimitive.ROTATE_SCAN:
            heading = (state.heading_deg + SCAN_ROTATION_DEGREES) % 360.0
            target = state.position
            path = (state.position,)
        else:
            heading = state.heading_deg
            if request.intent.mode == MovementMode.RECOVERY:
                recovery = route_tail
                advances_active_route = True
            else:
                recovery_points = tuple(request.recovery_path)
                recovery = (
                    state.position,
                    *recovery_points[state.local_recovery_cursor + 1:],
                )
                if not recovery_points:
                    recovery = (state.position,)
            path = self._bounded_path_from_position(
                state.position,
                recovery or (state.position,),
                float(request.step),
            )
            recovery_cursor_delta = max(0, len(path) - 1)
            if advances_active_route:
                route_cursor_delta = recovery_cursor_delta
                target_route_cursor = min(
                    len(window.route_prefix) - 1,
                    state.local_route_cursor + route_cursor_delta,
                )
                result_edge_cursor = window.route_segment_indexes[
                    target_route_cursor
                ]
                result_polyline_cursor = window.route_polyline_cursors[
                    target_route_cursor
                ]
            target = path[-1] if path else state.position
            heading = self._path_heading(path, heading)
        return LocalAction(
            primitive=primitive,
            target=(int(target[0]), int(target[1])),
            heading_deg=float(heading) % 360.0,
            path=tuple(path) or (state.position,),
            order=order,
            route_cursor_delta=route_cursor_delta,
            recovery_cursor_delta=recovery_cursor_delta,
            advances_active_route=advances_active_route,
            result_edge_cursor=result_edge_cursor,
            result_polyline_cursor=result_polyline_cursor,
        )

    def _evaluate_action(
        self,
        request: LocalMctsRequest,
        window: _LocalBeliefWindow,
        state: LocalSearchState,
        action: LocalAction,
        guard: _DeadlineGuard,
    ) -> tuple[LocalRewardComponents, frozenset[Position]]:
        """Evaluate one primitive using normalized local belief-only terms."""
        guard.check("root_evaluation" if state.depth == 0 else "expansion")
        path_distance = self._path_distance(action.path)
        if action.advances_active_route:
            progress = min(
                1.0,
                path_distance / max(1.0, min(
                    float(request.step),
                    max(1.0, state.remaining_route_cost),
                )),
            )
        else:
            progress = 0.0

        gain_cells = frozenset(self._predicted_gain_cells(
            request,
            window,
            action.target,
            action.heading_deg,
            state.observed_gain_cells,
            guard,
        ))
        information_gain = len(gain_cells) / self._planning_ray_count()
        marker, coarse_marker = self._visit_markers(state, action)
        revisit_count = sum(
            visit in {marker, coarse_marker}
            for visit in state.recent_visits
        )
        revisit = revisit_count / max(1, len(state.recent_visits))
        opposite = {
            LocalPrimitive.DEVIATE_LEFT: LocalPrimitive.DEVIATE_RIGHT,
            LocalPrimitive.DEVIATE_RIGHT: LocalPrimitive.DEVIATE_LEFT,
        }
        prior_opposite = opposite.get(action.primitive)
        oscillation = float(
            prior_opposite is not None
            and prior_opposite == state.previous_primitive
        )
        if (
            len(state.recent_visits) >= 2
            and marker == state.recent_visits[-2]
            and marker != state.recent_visits[-1]
        ):
            oscillation = 1.0
        target_switch = float(
            action.primitive == LocalPrimitive.RECOVERY
            and request.intent.mode != MovementMode.RECOVERY
            and not state.stalled
            and state.previous_primitive != LocalPrimitive.RECOVERY
        )
        turn = self._angular_distance(state.heading_deg, action.heading_deg) / 180.0
        if action.primitive == LocalPrimitive.ROTATE_SCAN:
            time_energy = min(1.0, turn)
        else:
            time_energy = min(1.0, path_distance / max(1.0, float(request.step)))
        collision_risk = self._collision_risk(
            request,
            window,
            action,
            guard,
        )
        return LocalRewardComponents(
            route_progress=progress,
            information_gain=information_gain,
            revisit=revisit,
            oscillation=oscillation,
            target_switch=target_switch,
            turn=turn,
            time_energy=time_energy,
            collision_risk=collision_risk,
        ), gain_cells

    def _predicted_gain_cells(
        self,
        request: LocalMctsRequest,
        window: _LocalBeliefWindow,
        position: Position,
        heading_deg: float,
        already_observed: frozenset[Position],
        guard: _DeadlineGuard,
    ) -> tuple[Position, ...]:
        """Predict one gain cell per ray, stopping at its first occluder."""
        rays = self._planning_ray_count()
        start = float(heading_deg) - SCAN_ROTATION_DEGREES / 2.0
        spacing = 0.0 if rays == 1 else SCAN_ROTATION_DEGREES / (rays - 1)
        max_range = min(
            LOCAL_WINDOW_RADIUS_CAP,
            max(int(request.radius) * 4, int(request.step) * 2, 1),
        )
        gained: list[Position] = []
        seen = set(already_observed)
        for ray_index in range(rays):
            guard.check("ray_cell")
            ray_heading = (
                float(heading_deg)
                if rays == 1
                else start + spacing * ray_index
            )
            endpoint = next_cell_coords(
                *position,
                max_range,
                int(round(ray_heading)) % 360,
            )
            ray = bresenham_line_points(
                position[0], position[1], endpoint[0], endpoint[1]
            )
            for raw_cell in ray[1:]:
                guard.check("ray_cell")
                cell = (int(raw_cell[0]), int(raw_cell[1]))
                values = window.values(cell)
                if values is None:
                    break
                occupancy, confidence = values
                if occupancy == OCCUPIED and confidence >= request.confidence_threshold:
                    break
                if (
                    occupancy == UNKNOWN
                    or not math.isfinite(confidence)
                    or confidence < request.confidence_threshold
                ):
                    if cell not in seen:
                        gained.append(cell)
                        seen.add(cell)
                    break
        return tuple(gained)

    def _planning_ray_count(self) -> int:
        """Return the fixed upper bound for decision-local predicted rays."""
        return min(
            LOCAL_PLANNING_RAY_CAP,
            max(1, int(self.config.planning_rays)),
        )

    def _collision_risk(
        self,
        request: LocalMctsRequest,
        window: _LocalBeliefWindow,
        action: LocalAction,
        guard: _DeadlineGuard,
    ) -> float:
        if action.primitive == LocalPrimitive.ROTATE_SCAN:
            return 0.0
        risk = 0.0
        for start, end in zip(action.path, action.path[1:]):
            for point_index, raw_cell in enumerate(self._iter_line(start, end)):
                if point_index == 0:
                    continue
                guard.check("simulation_depth")
                cell = (int(raw_cell[0]), int(raw_cell[1]))
                values = window.values(cell)
                if values is None:
                    return 1.0
                occupancy, confidence = values
                if (
                    occupancy == OCCUPIED
                    and confidence >= request.confidence_threshold
                ):
                    return 1.0
                if (
                    occupancy == UNKNOWN
                    or not math.isfinite(confidence)
                    or confidence < request.confidence_threshold
                ):
                    risk = max(risk, 0.5)
        return risk

    def _transition_state(
        self,
        state: LocalSearchState,
        action: LocalAction,
        reward: float,
        *,
        gain_cells: Iterable[Position] = (),
    ) -> LocalSearchState:
        distance = self._path_distance(action.path)
        remaining = state.remaining_route_cost
        polyline_cursor = state.polyline_cursor
        route_edge_cursor = state.route_edge_cursor
        local_route_cursor = state.local_route_cursor
        local_recovery_cursor = state.local_recovery_cursor
        if action.advances_active_route:
            remaining = max(0.0, remaining - distance)
            route_edge_cursor = action.result_edge_cursor
            polyline_cursor = action.result_polyline_cursor
            local_route_cursor += action.route_cursor_delta
        if action.primitive == LocalPrimitive.RECOVERY:
            local_recovery_cursor += action.recovery_cursor_delta
        visit, _ = self._visit_markers(state, action)
        visits = state.recent_visits
        if not visits or visits[-1] != visit:
            visits = (*visits, visit)[-32:]
        return replace(
            state,
            position=action.target,
            heading_deg=action.heading_deg,
            route_edge_cursor=route_edge_cursor,
            polyline_cursor=polyline_cursor,
            local_route_cursor=local_route_cursor,
            local_recovery_cursor=local_recovery_cursor,
            remaining_route_cost=remaining,
            recent_visits=visits,
            previous_primitive=action.primitive,
            stalled=state.stalled and reward <= 0.0,
            observed_gain_cells=(
                state.observed_gain_cells | frozenset(gain_cells)
            ),
            depth=state.depth + 1,
        )

    @staticmethod
    def _visit_markers(
        state: LocalSearchState,
        action: LocalAction,
    ) -> tuple[object, tuple[int, int]]:
        """Mirror the runtime watchdog's edge-or-coarse-cell visit identity."""
        coarse = (action.target[0] // 32, action.target[1] // 32)
        cursor = max(0, int(action.result_edge_cursor))
        if state.route_segment_edge_ids:
            cursor = min(cursor, len(state.route_segment_edge_ids) - 1)
            edge_id = state.route_segment_edge_ids[cursor]
            if edge_id is not None:
                return edge_id, coarse
        return coarse, coarse

    def _select_uct_arm(self, arms: list[_RootArm]) -> _RootArm:
        total = max(1, sum(arm.visits for arm in arms))
        exploration = max(0.0, float(self.config.uct_exploration))
        return max(
            arms,
            key=lambda arm: (
                arm.mean_reward + exploration * math.sqrt(
                    math.log(total + 1.0) / max(1, arm.visits)
                ),
                -arm.action.order,
            ),
        )

    @staticmethod
    def _choose_simulation_action(
        actions: tuple[LocalAction, ...],
        rng: random.Random,
    ) -> LocalAction:
        # Seeded sampling keeps rollouts diverse without affecting root order.
        return actions[rng.randrange(len(actions))]

    @staticmethod
    def _select_best_arm(
        arms: list[_RootArm],
        safe: LocalPrimitive,
    ) -> _RootArm:
        return max(
            arms,
            key=lambda arm: (
                arm.mean_reward,
                int(arm.action.primitive == safe),
                -arm.action.order,
            ),
        )

    @staticmethod
    def _safe_primitive(mode: MovementMode) -> LocalPrimitive:
        if mode == MovementMode.SCAN:
            return LocalPrimitive.ROTATE_SCAN
        if mode == MovementMode.RECOVERY:
            return LocalPrimitive.RECOVERY
        return LocalPrimitive.FOLLOW_EDGE

    @staticmethod
    def _stationary_action(
        primitive: LocalPrimitive,
        state: LocalSearchState,
    ) -> LocalAction:
        heading = state.heading_deg
        if primitive == LocalPrimitive.ROTATE_SCAN:
            heading = (heading + SCAN_ROTATION_DEGREES) % 360.0
        return LocalAction(
            primitive=primitive,
            target=state.position,
            heading_deg=heading,
            path=(state.position,),
            order=ROOT_PRIMITIVE_ORDER.index(primitive),
        )

    def _decision_seed(
        self,
        state: LocalSearchState,
    ) -> int:
        """Seed rollouts only from bounded persistent decision state."""
        goal_hash = self._stable_value_hash(state.goal_id)
        return (
            self.seed * 1_000_003
            + state.position[0] * 7_919
            + state.position[1] * 1_543
            + state.route_edge_cursor * 389
            + state.polyline_cursor * 193
            + goal_hash
        ) & 0xFFFFFFFF

    @classmethod
    def _stable_value_hash(cls, value: object) -> int:
        if value is None:
            return 0
        if isinstance(value, (int, bool)):
            return int(value) & 0xFFFFFFFF
        if isinstance(value, str):
            result = 2_166_136_261
            for byte in value.encode("utf-8"):
                result = ((result ^ byte) * 16_777_619) & 0xFFFFFFFF
            return result
        if isinstance(value, tuple):
            result = 0
            for item in value:
                result = (result * 65_599 + cls._stable_value_hash(item)) & 0xFFFFFFFF
            return result
        return cls._stable_value_hash(repr(value))

    @staticmethod
    def _route_heading(
        route_prefix: tuple[Position, ...],
        fallback: float,
    ) -> float:
        if len(route_prefix) < 2:
            return float(fallback) % 360.0
        start, end = route_prefix[0], route_prefix[1]
        return math.degrees(math.atan2(end[0] - start[0], start[1] - end[1])) % 360.0

    @staticmethod
    def _path_heading(path: tuple[Position, ...], fallback: float) -> float:
        if len(path) < 2:
            return float(fallback) % 360.0
        start, end = path[-2], path[-1]
        return math.degrees(math.atan2(end[0] - start[0], start[1] - end[1])) % 360.0

    @staticmethod
    def _angular_distance(left: float, right: float) -> float:
        delta = abs((float(left) - float(right)) % 360.0)
        return min(delta, 360.0 - delta)

    @staticmethod
    def _path_distance(path: tuple[Position, ...]) -> float:
        return sum(math.dist(start, end) for start, end in zip(path, path[1:]))

    @classmethod
    def _bounded_step_target(
        cls,
        position: Position,
        step: int,
        heading_deg: float,
    ) -> Position:
        """Return the closest integer heading whose distance never exceeds step."""
        radius = max(1, int(step))
        target = next_cell_coords(
            *position,
            radius,
            int(round(heading_deg)) % 360,
        )
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        while math.hypot(dx, dy) > radius + 1e-9:
            candidates = []
            if dx:
                reduced_x = dx - (1 if dx > 0 else -1)
                candidates.append((reduced_x, dy))
            if dy:
                reduced_y = dy - (1 if dy > 0 else -1)
                candidates.append((dx, reduced_y))
            dx, dy = min(
                candidates,
                key=lambda delta: (
                    cls._angular_distance(
                        heading_deg,
                        math.degrees(math.atan2(delta[0], -delta[1])) % 360.0,
                    ),
                    radius - math.hypot(*delta),
                    delta,
                ),
            )
        return position[0] + dx, position[1] + dy

    @staticmethod
    def _bounded_path_from_position(
        position: Position,
        path: tuple[Position, ...],
        budget: float,
    ) -> tuple[Position, ...]:
        if not path:
            return (position,)
        points = list(path)
        if points[0] != position:
            points.insert(0, position)
        selected = [position]
        travelled = 0.0
        for point in points[1:]:
            distance = math.dist(selected[-1], point)
            if travelled + distance > budget + 1e-9:
                break
            selected.append(point)
            travelled += distance
        return tuple(selected)

    @staticmethod
    def _rasterize(
        vertices: tuple[Position, ...],
        guard: _DeadlineGuard,
        stage: str,
    ) -> tuple[tuple[Position, ...], bool]:
        if not vertices:
            return (), False
        if len(vertices) == 1:
            guard.check(stage)
            return ((int(vertices[0][0]), int(vertices[0][1])),), False
        raster: list[Position] = []
        point_cap = LOCAL_WINDOW_RADIUS_CAP * 3 + 1
        for start, end in zip(vertices, vertices[1:]):
            guard.check(stage)
            for point in LocalMctsController._iter_line(start, end):
                guard.check(stage)
                normalized = (int(point[0]), int(point[1]))
                if not raster or raster[-1] != normalized:
                    raster.append(normalized)
                if len(raster) >= point_cap:
                    return tuple(raster), True
        return tuple(raster), False

    @staticmethod
    def _raster_suffix_after_cursor(
        vertices: tuple[Position, ...],
        polyline_cursor: int,
        position: Position,
        guard: _DeadlineGuard,
    ) -> tuple[tuple[Position, ...], bool]:
        """Return the exact stored-raster suffix after ``polyline_cursor``.

        Restarting Bresenham at the live pose is not equivalent to consuming the
        suffix of the original oriented leg.  It can change both geometry and
        cursor identity.  Walk the original raster under the deadline guard and
        prepend the live pose, matching ``DroneMovementController._route_prefix``
        even when the pose is temporarily off route.  A very long spent prefix
        remains bounded by the enclosing search deadline instead of being copied.
        """
        current = (int(position[0]), int(position[1]))
        if not vertices:
            return (current,), False
        cursor = max(0, int(polyline_cursor))
        suffix: list[Position] = [current]
        raster_index = 0
        if len(vertices) == 1:
            guard.check("preprocessing")
            return tuple(suffix), False
        for leg_index, (start, end) in enumerate(zip(vertices, vertices[1:])):
            guard.check("preprocessing")
            for point_index, point in enumerate(
                LocalMctsController._iter_line(start, end)
            ):
                guard.check("preprocessing")
                if leg_index and point_index == 0:
                    continue
                if raster_index > cursor:
                    suffix.append((int(point[0]), int(point[1])))
                    if len(suffix) >= LOCAL_WINDOW_RADIUS_CAP * 3 + 1:
                        return tuple(suffix), True
                raster_index += 1
        return tuple(suffix), False

    @staticmethod
    def _iter_line(start: Position, end: Position) -> Iterable[Position]:
        """Yield a Bresenham line lazily so preprocessing stays deadline-safe."""
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                return
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy


__all__ = [
    "DEVIATION_DEGREES",
    "DIAGNOSTIC_RESERVE_FRACTION",
    "LocalAction",
    "LocalMctsController",
    "LocalMctsDecision",
    "LocalMctsDiagnostics",
    "LocalMctsRequest",
    "LocalPrimitive",
    "LocalRewardComponents",
    "LocalRootDiagnostic",
    "LocalSearchState",
]
