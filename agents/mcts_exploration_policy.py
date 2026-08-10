"""Goal-conditioned local MCTS policy adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agents.exploration_policy import (
    ExplorationContext,
    ExplorationDecision,
    ExplorationDecisionKind,
    FrontierExplorationPolicy,
    Position,
)
from agents.local_mcts_controller import (
    LocalMctsController,
    LocalMctsRequest,
    LocalPrimitive,
    SCAN_ROTATION_DEGREES,
)
from config.simulation_config import ExplorationConfig
from mapping.slam_map import SlamSnapshot
from navigation.navigation_intent import MovementMode


@dataclass(frozen=True)
class RootVisitDiagnostic:
    """Immutable summary for one evaluated local root primitive."""

    kind: str
    direction: int
    target: Position
    visits: int
    mean_reward: float


@dataclass(frozen=True)
class MctsSearchDiagnostics:
    """Immutable diagnostics exposed after each local MCTS decision."""

    iterations: int
    root_visits: tuple[RootVisitDiagnostic, ...]
    selected_reward: float
    generated_nodes: int
    slam_version: int
    selected_kind: str = ""
    selected_direction: int | None = None
    selected_target: Position | None = None
    elapsed_ms: float = 0.0
    root_coverage_complete: bool = False
    overrun_stage: str | None = None
    safe_fallback: str | None = None
    budget_ms: float = 0.0
    search_budget_ms: float = 0.0
    reserved_budget_ms: float = 0.0
    window_bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
    preprocessing_cells: int = 0
    deadline_checks: tuple[tuple[str, int], ...] = ()
    performed: bool = False


class MctsExplorationPolicy(FrontierExplorationPolicy):
    """Deterministic global selection plus bounded local MCTS."""

    def __init__(
        self,
        config: ExplorationConfig,
        *,
        seed: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.uses_local_mcts = True
        self._local_controller = LocalMctsController(
            config,
            seed=seed,
            clock=clock,
        )
        self._last_search_diagnostics: MctsSearchDiagnostics | None = None

    @property
    def last_search_diagnostics(self) -> MctsSearchDiagnostics | None:
        """Return diagnostics for the latest actual local search."""
        return self._last_search_diagnostics

    def decide(self, context: ExplorationContext) -> ExplorationDecision:
        """Select a stable global goal or adapt an active local intent."""
        snapshot = context.runtime_snapshot
        if snapshot.returning_home and snapshot.navigation_intent is None:
            self._last_search_diagnostics = self._empty_diagnostics(context)
            return ExplorationDecision(
                kind=ExplorationDecisionKind.HOMING,
                target=context.start_position,
            )
        if snapshot.navigation_intent is not None:
            return self.decide_local(context)
        self._last_search_diagnostics = self._empty_diagnostics(context)
        return super().decide(context)

    def decide_local(
        self,
        context: ExplorationContext,
        *,
        slam_snapshot_provider: Callable[
            [tuple[int, int, int, int]], SlamSnapshot
        ] | None = None,
        slam_shape: tuple[int, int] | None = None,
        slam_version_hint: int = 0,
    ) -> ExplorationDecision:
        """Adapt one active intent to the bounded local controller."""
        snapshot = context.runtime_snapshot
        intent = snapshot.navigation_intent
        if intent is None:
            self._last_search_diagnostics = self._empty_diagnostics(context)
            return ExplorationDecision(ExplorationDecisionKind.EXHAUSTED)
        if intent.mode == MovementMode.SCAN:
            position = (
                int(context.pose_estimate.position[0]),
                int(context.pose_estimate.position[1]),
            )
            direction = int(round(
                context.pose_estimate.heading_deg + SCAN_ROTATION_DEGREES
            )) % 360
            slam_version = (
                int(context.slam_snapshot.version)
                if context.slam_snapshot is not None
                else int(slam_version_hint)
            )
            self._last_search_diagnostics = MctsSearchDiagnostics(
                iterations=0,
                root_visits=(),
                selected_reward=0.0,
                generated_nodes=0,
                slam_version=slam_version,
                selected_kind=LocalPrimitive.ROTATE_SCAN.value,
                selected_direction=direction,
                selected_target=position,
                performed=False,
            )
            return ExplorationDecision(
                kind=ExplorationDecisionKind.ROTATE,
                target=position,
                cluster_id=intent.cluster_id,
                direction=direction,
                planned_path=(position,),
                local_primitive=LocalPrimitive.ROTATE_SCAN.value,
            )
        if intent.mode == MovementMode.RECOVERY:
            # Recovery has exactly one mode-safe action and the movement
            # controller follows the exact polyline already stored in the
            # intent. Preparing a SLAM window cannot alter that choice.
            position = (
                int(context.pose_estimate.position[0]),
                int(context.pose_estimate.position[1]),
            )
            direction = int(round(context.pose_estimate.heading_deg)) % 360
            slam_version = (
                int(context.slam_snapshot.version)
                if context.slam_snapshot is not None
                else int(slam_version_hint)
            )
            self._last_search_diagnostics = MctsSearchDiagnostics(
                iterations=0,
                root_visits=(),
                selected_reward=0.0,
                generated_nodes=0,
                slam_version=slam_version,
                selected_kind=LocalPrimitive.RECOVERY.value,
                selected_direction=direction,
                selected_target=position,
                performed=False,
            )
            return ExplorationDecision(
                kind=ExplorationDecisionKind.STEP,
                target=position,
                cluster_id=intent.cluster_id,
                direction=direction,
                planned_path=(position,),
                local_primitive=LocalPrimitive.RECOVERY.value,
            )
        if (
            context.slam_snapshot is None and slam_snapshot_provider is None
        ):
            self._last_search_diagnostics = self._empty_diagnostics(context)
            return ExplorationDecision(ExplorationDecisionKind.EXHAUSTED)

        previous = None
        if intent.previous_primitive:
            try:
                previous = LocalPrimitive(intent.previous_primitive)
            except ValueError:
                previous = None
        recovery_path = tuple(reversed(
            snapshot.path_history[-max(2, int(context.step) + 1):]
        ))
        stalled = (
            snapshot.transition_reason.value in {"stalled", "reversal"}
            or snapshot.navigation_watchdog.reversal_count >= 2
            or snapshot.navigation_watchdog.distance_without_progress >= 64.0
        )
        local = self._local_controller.decide(LocalMctsRequest(
            position=context.pose_estimate.position,
            heading_deg=context.pose_estimate.heading_deg,
            step=max(1, int(context.step)),
            radius=max(1, int(context.radius)),
            intent=intent,
            slam_snapshot=context.slam_snapshot,
            slam_snapshot_provider=slam_snapshot_provider,
            slam_shape=slam_shape,
            slam_version_hint=slam_version_hint,
            recent_visits=snapshot.navigation_watchdog.recent_visits,
            previous_primitive=previous,
            stalled=stalled,
            confidence_threshold=context.frontier_confidence_threshold,
            recovery_path=recovery_path,
        ))
        direction = int(round(local.heading_deg)) % 360
        self._last_search_diagnostics = MctsSearchDiagnostics(
            iterations=local.diagnostics.iterations,
            root_visits=tuple(
                RootVisitDiagnostic(
                    kind=root.primitive.value,
                    direction=int(round(root.heading_deg)) % 360,
                    target=root.target,
                    visits=root.visits,
                    mean_reward=root.mean_reward,
                )
                for root in local.diagnostics.root_visits
            ),
            selected_reward=local.diagnostics.selected_reward,
            generated_nodes=local.diagnostics.generated_nodes,
            slam_version=local.diagnostics.slam_version,
            selected_kind=local.primitive.value,
            selected_direction=direction,
            selected_target=local.target,
            elapsed_ms=local.diagnostics.elapsed_ms,
            root_coverage_complete=local.diagnostics.root_coverage_complete,
            overrun_stage=local.diagnostics.overrun_stage,
            safe_fallback=(
                None
                if local.diagnostics.fallback_primitive is None
                else local.diagnostics.fallback_primitive.value
            ),
            budget_ms=local.diagnostics.budget_ms,
            search_budget_ms=local.diagnostics.search_budget_ms,
            reserved_budget_ms=local.diagnostics.reserved_budget_ms,
            window_bounds=local.diagnostics.window_bounds,
            preprocessing_cells=local.diagnostics.preprocessing_cells,
            deadline_checks=local.diagnostics.deadline_checks,
            performed=True,
        )
        if local.primitive == LocalPrimitive.ROTATE_SCAN:
            return ExplorationDecision(
                kind=ExplorationDecisionKind.ROTATE,
                target=local.target,
                cluster_id=intent.cluster_id,
                direction=direction,
                planned_path=local.path,
                local_primitive=local.primitive.value,
            )
        return ExplorationDecision(
            kind=ExplorationDecisionKind.STEP,
            target=local.target,
            cluster_id=intent.cluster_id,
            direction=direction,
            planned_path=local.path,
            local_primitive=local.primitive.value,
        )

    @staticmethod
    def _empty_diagnostics(
        context: ExplorationContext,
    ) -> MctsSearchDiagnostics:
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
            performed=False,
        )
