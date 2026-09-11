"""Rover-owned frontier tasks, claims, and bounded discovery rounds."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
import math
import threading
from typing import Iterable

import cv2
import numpy as np

from mapping.frontier_registry import (
    CausalTransition,
    ComponentState,
    ComponentWorkUnit,
    FrontierComponentRecord,
    FrontierComponentRegistry,
    FrontierRegistrySnapshot,
    LineageKind,
    RegistryReconcileResult,
    SensorFootprint,
    WorkUnitState,
)
from mapping.slam_map import FREE, SlamSnapshot
from mission.energy import (
    EnergyReturnDecision,
    EnergyRequirement,
    EnergyState,
    UnlimitedEnergyPolicy,
)
from navigation.astar_pathfinder import PATH_COMPLETE


Position = tuple[int, int]
TaskId = int
ClaimToken = int


class TaskState(str, Enum):
    READY = "ready"
    CLAIMED = "claimed"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class DirectiveKind(str, Enum):
    ROVER_SCAN = "rover_scan"
    RADIAL_PROBE = "radial_probe"
    COMPONENT_TASK = "component_task"
    COMPONENT_FOLLOW = "component_follow"
    HOME = "home"


class DiscoveryKind(str, Enum):
    ROVER_SCAN = "rover_scan"
    RADIAL_PROBE = "radial_probe"


class DiscoveryState(str, Enum):
    PLANNED = "planned"
    DISPATCHED = "dispatched"
    COLLECTING = "collecting"
    RECONCILING = "reconciling"
    COMPLETE = "complete"
    ABORTED = "aborted"


class ExplorationPhase(str, Enum):
    """Monotonic mission phases; probing is confined to bootstrap."""

    INITIAL_SCAN = "initial_scan"
    BOOTSTRAP_PROBING = "bootstrap_probing"
    COMPONENT_EXPLORATION = "component_exploration"
    COMPLETE = "complete"


@dataclass
class ExplorationTask:
    task_id: TaskId
    component_id: int
    component_revision: int
    work_unit_ids: tuple[int, ...]
    parent_task_id: TaskId | None
    depth: int
    preferred_entry: Position
    estimated_effort: float
    state: TaskState = TaskState.READY
    suspension: "TaskSuspension | None" = None


@dataclass(frozen=True)
class ClaimLease:
    task_id: TaskId
    work_unit_ids: tuple[int, ...]
    owner_drone_id: int
    token: ClaimToken
    issued_revision: int


@dataclass(frozen=True)
class DFSFrame:
    task_id: TaskId
    component_id: int
    component_revision: int
    entry_position: Position
    outbound_actual_path: tuple[Position, ...]
    pending_work_unit_ids: tuple[int, ...]
    pending_child_task_ids: tuple[int, ...]
    active_work_unit_id: int | None


@dataclass(frozen=True)
class TaskSuspension:
    task_id: TaskId
    claim_token: ClaimToken
    drone_id: int
    reason: str
    dfs_stack: tuple[DFSFrame, ...]
    remaining_work_unit_ids: tuple[int, ...]
    position_at_suspension: Position
    actual_return_path: tuple[Position, ...]
    return_path_source: str
    local_slam_version: int
    energy_state: EnergyState


@dataclass(frozen=True)
class WorkUnitOutcome:
    work_unit_id: int
    disposition: str
    sensor_newly_known_cells: int = 0
    sensor_confidence_gain: float = 0.0


@dataclass(frozen=True)
class CoordinationReport:
    report_id: int
    directive_id: int
    kind: DirectiveKind
    round_id: int | None = None
    task_id: int | None = None
    component_id: int | None = None
    claim_token: ClaimToken | None = None
    work_unit_outcomes: tuple[WorkUnitOutcome, ...] = ()
    causal_transitions: tuple[CausalTransition, ...] = ()
    suspension: TaskSuspension | None = None
    completed_scan_headings: tuple[int, ...] = ()
    timed_out_scan_headings: tuple[int, ...] = ()
    sensor_newly_known_cells: int = 0
    sensor_confidence_gain: float = 0.0
    outbound_actual_path: tuple[Position, ...] = ()
    return_actual_path: tuple[Position, ...] = ()
    return_path_source: str = "none"


@dataclass(frozen=True)
class ExplorationDirective:
    directive_id: int
    kind: DirectiveKind
    round_id: int | None = None
    scan_headings: tuple[int, ...] = ()
    probe_target: Position | None = None
    outbound_route: tuple[Position, ...] = ()
    task: ExplorationTask | None = None
    work_units: tuple[ComponentWorkUnit, ...] = ()
    claim: ClaimLease | None = None
    leader_drone_id: int | None = None
    follow_branch_index: int | None = None
    reserved_branch_count: int = 0
    reason: str = ""


@dataclass
class DiscoveryRound:
    round_id: int
    kind: DiscoveryKind
    state: DiscoveryState
    participant_ids: tuple[int, ...]
    scan_plans: dict[int, tuple[int, ...]]
    probe_targets: dict[int, Position]
    reports: dict[int, CoordinationReport]
    rover_slam_version_before: int
    registry_revision_before: int
    ring: int


@dataclass(frozen=True)
class RouteQuote:
    drone_id: int
    task_id: int
    status: str
    route: tuple[Position, ...]
    route_cost: float
    route_home_cost: float
    can_accept: bool
    dfs_depth: int
    continuation_affinity: bool


@dataclass(frozen=True)
class ExplorationCoordinatorSnapshot:
    revision: int
    components: tuple[FrontierComponentRecord, ...]
    work_units: tuple[ComponentWorkUnit, ...]
    tasks: tuple[ExplorationTask, ...]
    claims: tuple[ClaimLease, ...]
    waiting_drone_ids: frozenset[int]
    mission_exhausted: bool
    discovery_round: DiscoveryRound | None
    phase: ExplorationPhase = ExplorationPhase.INITIAL_SCAN


@dataclass(frozen=True)
class CoordinationResult:
    arrived: bool
    report_accepted: bool = False
    waiting: bool = False
    mission_exhausted: bool = False
    directive: ExplorationDirective | None = None
    directive_ready: threading.Event | None = None
    reconcile_result: RegistryReconcileResult | None = None
    published_directives: tuple[ExplorationDirective, ...] = ()


class FrontierTaskCoordinator:
    """Assign component work asynchronously; never assign map territory."""

    def __init__(
        self,
        map_shape: tuple[int, int],
        drone_count: int,
        rover_position: Position,
        *,
        sensor_range: float,
        sensor_fov_deg: float,
        confidence_threshold: float = 0.6,
        minimum_component_cells: int = 12,
        minimum_unknown_support_cells: int = 64,
        frontier_stride: int = 4,
        global_cell_size: int = 32,
        energy_policy: object | None = None,
    ) -> None:
        if drone_count <= 0:
            raise ValueError("exploration drone_count must be positive")
        self.map_shape = tuple(int(value) for value in map_shape)
        self.drone_count = int(drone_count)
        self.rover_position = tuple(int(value) for value in rover_position)
        self.sensor_range = float(sensor_range)
        self.sensor_fov_deg = float(sensor_fov_deg)
        self.energy_policy = energy_policy or UnlimitedEnergyPolicy()
        self.registry = FrontierComponentRegistry(
            self.map_shape,
            SensorFootprint(
                max_range=self.sensor_range,
                fov_deg=self.sensor_fov_deg,
                frontier_stride=frontier_stride,
                global_cell_size=global_cell_size,
            ),
            confidence_threshold=confidence_threshold,
            minimum_component_cells=minimum_component_cells,
            minimum_unknown_support_cells=minimum_unknown_support_cells,
        )
        self._lock = threading.RLock()
        self._waiting: set[int] = set()
        self._ready_events = {
            drone_id: threading.Event()
            for drone_id in range(self.drone_count)
        }
        self._pending_directives: dict[int, ExplorationDirective] = {}
        self._active_directives: dict[int, ExplorationDirective] = {}
        self._tasks: dict[int, ExplorationTask] = {}
        self._task_by_unit: dict[int, int] = {}
        self._component_task_ids: dict[int, list[int]] = {}
        self._claims_by_task: dict[int, ClaimLease] = {}
        self._claims_by_token: dict[int, ClaimLease] = {}
        self._component_blockers: dict[int, set[int]] = {}
        self._energy_states = {
            drone_id: EnergyState(100.0, 100.0, unlimited=True)
            for drone_id in range(self.drone_count)
        }
        self._accepted_report_ids: set[int] = set()
        self._last_reported_task_by_drone: dict[int, int] = {}
        self._preferred_parent_task_by_component: dict[int, int] = {}
        self._route_rejections: dict[tuple[int, int], int] = {}
        self._next_task_id = 0
        self._next_claim_token = 0
        self._next_directive_id = 0
        self._next_round_id = 0
        self._initial_scan_complete = False
        self._phase = ExplorationPhase.INITIAL_SCAN
        self._bootstrap_followers_issued = False
        self._discovery_round: DiscoveryRound | None = None
        self._probe_radius = {drone_id: 0.0 for drone_id in range(drone_count)}
        self._probe_rings = {drone_id: 0 for drone_id in range(drone_count)}
        self._probe_exhausted: set[int] = set()
        self._maximum_probe_rings = max(
            1,
            int(math.ceil(
                math.hypot(self.map_shape[1], self.map_shape[0])
                / max(1.0, self.sensor_range * 0.75)
            )),
        )
        self._mission_exhausted = False
        self._last_rover_slam: SlamSnapshot | None = None
        self._last_reconcile_result: RegistryReconcileResult | None = None
        self._published_snapshot = self._snapshot_unlocked()

    def snapshot(
        self,
        *,
        blocking: bool = True,
    ) -> ExplorationCoordinatorSnapshot:
        """Return current state, or the last publication without blocking UI."""
        acquired = self._lock.acquire(blocking=blocking)
        if not acquired:
            return self._published_snapshot
        try:
            snapshot = self._snapshot_unlocked()
            self._published_snapshot = snapshot
            return snapshot
        finally:
            self._lock.release()

    def _snapshot_unlocked(self) -> ExplorationCoordinatorSnapshot:
        registry = self.registry.snapshot()
        round_copy = None
        if self._discovery_round is not None:
            current = self._discovery_round
            round_copy = DiscoveryRound(
                round_id=current.round_id,
                kind=current.kind,
                state=current.state,
                participant_ids=current.participant_ids,
                scan_plans=dict(current.scan_plans),
                probe_targets=dict(current.probe_targets),
                reports=dict(current.reports),
                rover_slam_version_before=current.rover_slam_version_before,
                registry_revision_before=current.registry_revision_before,
                ring=current.ring,
            )
        return ExplorationCoordinatorSnapshot(
            revision=registry.revision,
            components=registry.components,
            work_units=registry.work_units,
            tasks=tuple(
                replace(self._tasks[key]) for key in sorted(self._tasks)
            ),
            claims=tuple(
                self._claims_by_task[key]
                for key in sorted(self._claims_by_task)
            ),
            waiting_drone_ids=frozenset(self._waiting),
            mission_exhausted=self._mission_exhausted,
            discovery_round=round_copy,
            phase=self._phase,
        )

    def update_rover_position(self, position: Position) -> None:
        """Publish the rendezvous point used by routing and discovery."""
        with self._lock:
            self.rover_position = tuple(int(value) for value in position)

    def published_snapshot(self) -> ExplorationCoordinatorSnapshot:
        """Return the latest immutable publication without acquiring the lock."""
        return self._published_snapshot

    def evaluate_return(
        self,
        drone_id: int,
        energy_state: EnergyState,
        *,
        route_home_cost: float,
        next_action_cost: float,
        safety_reserve: float,
    ) -> EnergyReturnDecision:
        """Evaluate the executor checkpoint through the configured policy."""
        normalized_id = int(drone_id)
        if not 0 <= normalized_id < self.drone_count:
            raise ValueError("drone_id is outside the configured team")
        with self._lock:
            self._energy_states[normalized_id] = energy_state
            required = bool(self.energy_policy.must_return(
                energy_state,
                route_home_cost=float(route_home_cost),
                next_action_cost=float(next_action_cost),
                safety_reserve=float(safety_reserve),
            ))
            return EnergyReturnDecision(
                state=energy_state,
                route_home_cost=float(route_home_cost),
                next_action_cost=float(next_action_cost),
                safety_reserve=float(safety_reserve),
                must_return=required,
            )

    def check_in(
        self,
        drone_id: int,
        rover_slam: SlamSnapshot,
        *,
        report: CoordinationReport | None = None,
        energy_state: EnergyState | None = None,
    ) -> CoordinationResult:
        normalized_id = int(drone_id)
        if not 0 <= normalized_id < self.drone_count:
            raise ValueError("drone_id is outside the configured team")
        with self._lock:
            self._last_rover_slam = rover_slam
            if energy_state is not None:
                self._energy_states[normalized_id] = energy_state
            active = self._active_directives.get(normalized_id)
            reconcile_result = None
            report_accepted = report is None
            duplicate_report = bool(
                report is not None
                and report.report_id in self._accepted_report_ids
            )
            if report is not None:
                report_accepted, reconcile_result = self._accept_report(
                    normalized_id,
                    active,
                    report,
                    rover_slam,
                )
                if not report_accepted:
                    return CoordinationResult(
                        arrived=True,
                        report_accepted=False,
                        waiting=False,
                        mission_exhausted=self._mission_exhausted,
                    )
                if duplicate_report and active is not None:
                    return CoordinationResult(
                        arrived=True,
                        report_accepted=True,
                        waiting=False,
                        mission_exhausted=self._mission_exhausted,
                    )
                if not duplicate_report:
                    self._active_directives.pop(normalized_id, None)
            self._waiting.add(normalized_id)
            if normalized_id in self._pending_directives:
                self._ready_events[normalized_id].set()
            else:
                self._ready_events[normalized_id].clear()

            if self._mission_exhausted:
                self._issue_home((normalized_id,), reason="mission_exhausted")
            elif not self._initial_scan_complete:
                if (
                    self._discovery_round is None
                    and len(self._waiting) == self.drone_count
                ):
                    self._start_rover_scan_round(rover_slam)
            elif self._discovery_round is None:
                if reconcile_result is None and not duplicate_report:
                    reconcile_result = self._reconcile(rover_slam, ())
                self._schedule(rover_slam)

            return CoordinationResult(
                arrived=True,
                report_accepted=report_accepted,
                waiting=normalized_id in self._waiting,
                mission_exhausted=self._mission_exhausted,
                directive_ready=self._ready_events[normalized_id],
                reconcile_result=reconcile_result,
                published_directives=tuple(
                    self._pending_directives[key]
                    for key in sorted(self._pending_directives)
                    if self._pending_directives[key].directive_id
                    >= self._next_directive_id - self.drone_count
                ),
            )

    def claim_directive(self, drone_id: int) -> CoordinationResult:
        normalized_id = int(drone_id)
        if not 0 <= normalized_id < self.drone_count:
            raise ValueError("drone_id is outside the configured team")
        with self._lock:
            event = self._ready_events[normalized_id]
            directive = self._pending_directives.pop(normalized_id, None)
            event.clear()
            if directive is None:
                return CoordinationResult(
                    arrived=True,
                    waiting=True,
                    mission_exhausted=self._mission_exhausted,
                    directive_ready=event,
                )
            self._waiting.discard(normalized_id)
            self._active_directives[normalized_id] = directive
            if directive.claim is not None:
                task = self._tasks[directive.claim.task_id]
                task.state = TaskState.ACTIVE
                self.registry.activate_work_units(
                    directive.claim.work_unit_ids
                )
            return CoordinationResult(
                arrived=True,
                waiting=False,
                mission_exhausted=(directive.kind == DirectiveKind.HOME),
                directive=directive,
            )

    def stop(self) -> None:
        with self._lock:
            self._mission_exhausted = True
            if self._discovery_round is not None:
                self._discovery_round.state = DiscoveryState.ABORTED
            self._issue_home(
                tuple(range(self.drone_count)),
                reason="coordinator_stopped",
            )

    def _accept_report(
        self,
        drone_id: int,
        active: ExplorationDirective | None,
        report: CoordinationReport,
        rover_slam: SlamSnapshot,
    ) -> tuple[bool, RegistryReconcileResult | None]:
        if report.report_id in self._accepted_report_ids:
            return True, None
        if active is None or active.directive_id != report.directive_id:
            return False, None
        if active.kind != report.kind:
            return False, None
        if report.kind in {
            DirectiveKind.ROVER_SCAN,
            DirectiveKind.RADIAL_PROBE,
        }:
            current = self._discovery_round
            if current is None or active.round_id != current.round_id:
                return False, None
            self._accepted_report_ids.add(report.report_id)
            current.reports[drone_id] = report
            if report.kind == DirectiveKind.RADIAL_PROBE:
                target = active.probe_target
                if target is not None:
                    self._probe_radius[drone_id] = max(
                        self._probe_radius[drone_id],
                        math.dist(self.rover_position, target),
                    )
                    self._probe_rings[drone_id] += 1
                    if self._probe_rings[drone_id] >= self._maximum_probe_rings:
                        self._probe_exhausted.add(drone_id)
            if len(current.reports) == len(current.participant_ids):
                current.state = DiscoveryState.RECONCILING
                causal = tuple(
                    item
                    for accepted in current.reports.values()
                    for item in accepted.causal_transitions
                )
                result = self._reconcile(rover_slam, causal)
                current.state = DiscoveryState.COMPLETE
                if current.kind == DiscoveryKind.ROVER_SCAN:
                    self._initial_scan_complete = True
                if self._has_outstanding_work():
                    self._phase = ExplorationPhase.COMPONENT_EXPLORATION
                elif current.kind == DiscoveryKind.ROVER_SCAN:
                    self._phase = ExplorationPhase.BOOTSTRAP_PROBING
                self._discovery_round = None
                return True, result
            current.state = DiscoveryState.COLLECTING
            return True, None

        if report.kind == DirectiveKind.COMPONENT_FOLLOW:
            self._accepted_report_ids.add(report.report_id)
            return True, self._reconcile(
                rover_slam,
                report.causal_transitions,
            )

        if report.kind == DirectiveKind.COMPONENT_TASK:
            claim = active.claim
            if (
                claim is None
                or report.claim_token != claim.token
                or self._claims_by_token.get(claim.token) != claim
            ):
                return False, None
            self._accepted_report_ids.add(report.report_id)
            self._last_reported_task_by_drone[drone_id] = claim.task_id
            self._preferred_parent_task_by_component[
                self._tasks[claim.task_id].component_id
            ] = claim.task_id
            completed_ids = set()
            for outcome in report.work_unit_outcomes:
                if outcome.work_unit_id not in claim.work_unit_ids:
                    continue
                if self.registry.complete_work_unit(
                    outcome.work_unit_id,
                    reason=outcome.disposition,
                ):
                    completed_ids.add(outcome.work_unit_id)
            task = self._tasks[claim.task_id]
            rejection_key = (drone_id, claim.task_id)
            if (
                report.suspension is not None
                and report.suspension.reason == "route_unreachable"
            ):
                self._route_rejections[rejection_key] = (
                    task.component_revision
                )
            else:
                self._route_rejections.pop(rejection_key, None)
            self._release_claim(claim)
            remaining = tuple(
                unit_id for unit_id in claim.work_unit_ids
                if unit_id not in completed_ids
            )
            if report.suspension is not None and remaining:
                task.state = TaskState.SUSPENDED
                task.suspension = report.suspension
                self.registry.release_work_units(remaining)
            elif remaining:
                task.state = TaskState.SUSPENDED
                self.registry.release_work_units(remaining)
            else:
                task.state = TaskState.COMPLETE
                task.suspension = None
            return True, self._reconcile(
                rover_slam,
                report.causal_transitions,
            )
        return False, None

    def _release_claim(self, claim: ClaimLease) -> None:
        self._claims_by_task.pop(claim.task_id, None)
        self._claims_by_token.pop(claim.token, None)
        for component_id, blockers in tuple(
            self._component_blockers.items()
        ):
            blockers.discard(claim.token)
            if not blockers:
                self._component_blockers.pop(component_id, None)
                component = self.registry.components.get(component_id)
                if component is not None:
                    self.registry.unblock_work_units(
                        component.work_unit_ids
                    )

    def _reconcile(
        self,
        rover_slam: SlamSnapshot,
        causal: Iterable[CausalTransition],
    ) -> RegistryReconcileResult:
        result = self.registry.reconcile(
            rover_slam,
            causal_transitions=tuple(causal),
        )
        self._last_reconcile_result = result
        self._apply_lineage_blocks(result)
        self._refresh_component_blocks()
        self._sync_tasks()
        return result

    def _apply_lineage_blocks(self, result: RegistryReconcileResult) -> None:
        for transition in result.transitions:
            if transition.kind not in {
                LineageKind.SPLIT,
                LineageKind.MERGED,
                LineageKind.RESEGMENTED,
            }:
                continue
            blocker_tokens = {
                claim.token
                for task_id, claim in self._claims_by_task.items()
                if self._tasks[task_id].component_id
                in transition.parent_ids
            }
            for parent_id in transition.parent_ids:
                blocker_tokens.update(
                    self._component_blockers.get(parent_id, ())
                )
            if not blocker_tokens:
                continue
            for child_id in transition.child_ids:
                self._component_blockers.setdefault(child_id, set()).update(
                    blocker_tokens
                )
            for parent_id in transition.parent_ids:
                self._component_blockers.pop(parent_id, None)

    def _refresh_component_blocks(self) -> None:
        for component_id, blocker_tokens in tuple(
            self._component_blockers.items()
        ):
            component = self.registry.components.get(component_id)
            if component is None or component.state != ComponentState.ACTIVE:
                self._component_blockers.pop(component_id, None)
                continue
            live_tokens = {
                token for token in blocker_tokens
                if token in self._claims_by_token
            }
            if not live_tokens:
                self._component_blockers.pop(component_id, None)
                self.registry.unblock_work_units(component.work_unit_ids)
                continue
            self._component_blockers[component_id] = live_tokens
            self.registry.block_work_units(component.work_unit_ids)

    def _sync_tasks(self) -> None:
        for task in self._tasks.values():
            if task.state in {
                TaskState.CLAIMED,
                TaskState.ACTIVE,
                TaskState.COMPLETE,
                TaskState.CANCELLED,
            }:
                continue
            states = tuple(
                self.registry.work_units[unit_id].state
                for unit_id in task.work_unit_ids
            )
            if states and all(
                state == WorkUnitState.CANCELLED for state in states
            ):
                task.state = TaskState.CANCELLED
            elif any(state == WorkUnitState.BLOCKED for state in states):
                task.state = TaskState.BLOCKED
            elif (
                states
                and all(state == WorkUnitState.READY for state in states)
                and task.state == TaskState.BLOCKED
            ):
                task.state = TaskState.READY

        pending_by_component: dict[int, list[ComponentWorkUnit]] = {}
        for unit in self.registry.work_units.values():
            if unit.work_unit_id in self._task_by_unit:
                continue
            if unit.state not in {WorkUnitState.READY, WorkUnitState.BLOCKED}:
                continue
            pending_by_component.setdefault(unit.component_id, []).append(unit)

        for component_id in sorted(pending_by_component):
            units = tuple(sorted(
                pending_by_component[component_id],
                key=lambda item: item.work_unit_id,
            ))
            component = self.registry.components[component_id]
            parent_tasks = tuple(
                task_id
                for parent_id in component.parent_ids
                for task_id in self._component_task_ids.get(parent_id, ())
            )
            preferred_parent_tasks = tuple(
                self._preferred_parent_task_by_component[parent_id]
                for parent_id in component.parent_ids
                if parent_id in self._preferred_parent_task_by_component
            )
            parent_task_id = (
                min(preferred_parent_tasks)
                if preferred_parent_tasks
                else min(parent_tasks) if parent_tasks else None
            )
            depth = 0
            if parent_tasks:
                depth = max(self._tasks[item].depth for item in parent_tasks) + 1
            task_id = self._next_task_id
            self._next_task_id += 1
            task = ExplorationTask(
                task_id=task_id,
                component_id=component_id,
                component_revision=max(
                    unit.component_revision for unit in units
                ),
                work_unit_ids=tuple(
                    unit.work_unit_id for unit in units
                ),
                parent_task_id=parent_task_id,
                depth=depth,
                preferred_entry=units[0].anchor_position,
                estimated_effort=sum(
                    unit.estimated_effort for unit in units
                ),
                state=(
                    TaskState.BLOCKED
                    if any(
                        unit.state == WorkUnitState.BLOCKED for unit in units
                    )
                    else TaskState.READY
                ),
            )
            self._tasks[task_id] = task
            for unit in units:
                self._task_by_unit[unit.work_unit_id] = task_id
            self._component_task_ids.setdefault(component_id, []).append(
                task_id
            )

    def _schedule(self, rover_slam: SlamSnapshot) -> None:
        if self._mission_exhausted or self._discovery_round is not None:
            return
        self._sync_tasks()
        idle = tuple(sorted(
            drone_id for drone_id in self._waiting
            if drone_id not in self._pending_directives
        ))
        if not idle:
            return
        tasks = tuple(
            task for _task_id, task in sorted(self._tasks.items())
            if task.state in {TaskState.READY, TaskState.SUSPENDED}
            and not self._task_reserved_by_follower(task)
            and self.registry.components[task.component_id].state
            == ComponentState.ACTIVE
            and all(
                self.registry.work_units[unit_id].state == WorkUnitState.READY
                for unit_id in task.work_unit_ids
            )
        )
        reachable = self._known_free_reachable_mask(rover_slam)
        reachable_task_ids = frozenset(
            task.task_id
            for task in tasks
            if self._point_is_reachable(
                task.preferred_entry,
                reachable,
                rover_slam.origin,
            )
        )
        quotes, selected = self._assign_tasks(
            idle,
            tasks,
            reachable_task_ids=reachable_task_ids,
        )
        assigned_drones: set[int] = set()
        leader_tasks: list[tuple[int, ExplorationTask]] = []
        for drone_id, task_id, route in selected:
            task = self._tasks[task_id]
            if not self.registry.claim_work_units(task.work_unit_ids):
                continue
            token = self._next_claim_token
            self._next_claim_token += 1
            claim = ClaimLease(
                task_id=task_id,
                work_unit_ids=task.work_unit_ids,
                owner_drone_id=drone_id,
                token=token,
                issued_revision=self.registry.revision,
            )
            self._claims_by_task[task_id] = claim
            self._claims_by_token[token] = claim
            task.state = TaskState.CLAIMED
            units = tuple(
                replace(self.registry.work_units[unit_id])
                for unit_id in task.work_unit_ids
            )
            self._issue(
                drone_id,
                ExplorationDirective(
                    directive_id=self._allocate_directive_id(),
                    kind=DirectiveKind.COMPONENT_TASK,
                    outbound_route=route,
                    task=replace(task),
                    work_units=units,
                    claim=claim,
                    reason="reachable_component_work",
                ),
            )
            assigned_drones.add(drone_id)
            leader_tasks.append((drone_id, task))

        remaining = tuple(
            drone_id for drone_id in idle if drone_id not in assigned_drones
        )
        if not remaining:
            return
        reachable_outstanding = bool(self._claims_by_task or quotes)
        if (
            self._phase == ExplorationPhase.COMPONENT_EXPLORATION
            and leader_tasks
            and not self._bootstrap_followers_issued
        ):
            self._issue_bootstrap_followers(remaining, leader_tasks)
            self._bootstrap_followers_issued = True
            return
        if self._phase == ExplorationPhase.BOOTSTRAP_PROBING:
            if self._start_radial_round(
                remaining,
                rover_slam,
                reachable_mask=reachable,
            ):
                return
            if self._team_is_quiescent() and not reachable_outstanding:
                self._finish_mission("bootstrap_discovery_exhausted")
            return
        if (
            self._phase == ExplorationPhase.COMPONENT_EXPLORATION
            and self._team_is_quiescent()
            and not self._has_outstanding_work()
        ):
            self._finish_mission("component_work_exhausted")

    def _task_reserved_by_follower(self, task: ExplorationTask) -> bool:
        """Prevent rover reassignment of lineage held by an active follower."""
        followed_parents = {
            directive.task.component_id
            for directive in (
                *self._active_directives.values(),
                *self._pending_directives.values(),
            )
            if directive.kind == DirectiveKind.COMPONENT_FOLLOW
            and directive.task is not None
        }
        if not followed_parents:
            return False
        component = self.registry.components.get(task.component_id)
        return bool(
            component is not None
            and (
                component.component_id in followed_parents
                or any(
                    parent_id in followed_parents
                    for parent_id in component.parent_ids
                )
            )
        )

    def _issue_bootstrap_followers(
        self,
        drone_ids: tuple[int, ...],
        leaders: list[tuple[int, ExplorationTask]],
    ) -> None:
        """Use spare bootstrap drones to reserve distinct child branches."""
        if not drone_ids or not leaders:
            return
        followers_by_leader: dict[int, list[int]] = {
            leader_id: [] for leader_id, _task in leaders
        }
        for index, drone_id in enumerate(drone_ids):
            leader_id, _task = leaders[index % len(leaders)]
            followers_by_leader[leader_id].append(drone_id)

        for leader_id, task in leaders:
            followers = followers_by_leader[leader_id]
            if not followers:
                continue
            current = self._pending_directives.get(leader_id)
            if current is not None:
                self._pending_directives[leader_id] = replace(
                    current,
                    reserved_branch_count=len(followers),
                )
            units = tuple(
                replace(self.registry.work_units[unit_id])
                for unit_id in task.work_unit_ids
            )
            for branch_index, drone_id in enumerate(followers):
                self._issue(
                    drone_id,
                    ExplorationDirective(
                        directive_id=self._allocate_directive_id(),
                        kind=DirectiveKind.COMPONENT_FOLLOW,
                        task=replace(task),
                        work_units=units,
                        leader_drone_id=leader_id,
                        follow_branch_index=branch_index,
                        reason="bootstrap_component_capacity_deficit",
                    ),
                )

    def _team_is_quiescent(self) -> bool:
        """Require every physical rover check-in before declaring completion."""
        return bool(
            len(self._waiting) == self.drone_count
            and not self._pending_directives
            and not self._active_directives
            and self._discovery_round is None
        )

    def _finish_mission(self, reason: str) -> None:
        self._mission_exhausted = True
        self._phase = ExplorationPhase.COMPLETE
        self._issue_home(tuple(sorted(self._waiting)), reason=reason)

    def _assign_tasks(
        self,
        drones: tuple[int, ...],
        tasks: tuple[ExplorationTask, ...],
        *,
        reachable_task_ids: frozenset[int] | None = None,
    ) -> tuple[tuple[RouteQuote, ...], tuple[tuple[int, int, tuple[Position, ...]], ...]]:
        """Assign connected tasks; exact routes belong to drone workers."""
        quotes: dict[tuple[int, int], RouteQuote] = {}
        eligible_task_ids = (
            frozenset(task.task_id for task in tasks)
            if reachable_task_ids is None
            else reachable_task_ids
        )
        for drone_id in drones:
            state = self._energy_states[drone_id]
            for task in tasks:
                if task.task_id not in eligible_task_ids:
                    continue
                if self._route_rejections.get(
                    (drone_id, task.task_id)
                ) == task.component_revision:
                    continue
                route_cost = math.dist(
                    self.rover_position,
                    task.preferred_entry,
                )
                continuation_affinity = bool(
                    (
                        task.suspension is not None
                        and task.suspension.drone_id == drone_id
                    )
                    or (
                        task.parent_task_id is not None
                        and task.parent_task_id
                        == self._last_reported_task_by_drone.get(drone_id)
                    )
                )
                requirement = EnergyRequirement(
                    route_to_task_cost=route_cost,
                    next_action_cost=task.estimated_effort,
                    route_home_cost=route_cost,
                    safety_reserve=0.0,
                )
                can_accept = bool(self.energy_policy.can_accept(
                    state,
                    requirement,
                ))
                if not can_accept:
                    continue
                quotes[(drone_id, task.task_id)] = RouteQuote(
                    drone_id=drone_id,
                    task_id=task.task_id,
                    status=PATH_COMPLETE,
                    route=(),
                    route_cost=route_cost,
                    route_home_cost=route_cost,
                    can_accept=True,
                    dfs_depth=task.depth,
                    continuation_affinity=continuation_affinity,
                )

        task_ids = tuple(task.task_id for task in tasks)

        @lru_cache(maxsize=None)
        def solve(
            drone_index: int,
            used: tuple[int, ...],
        ) -> tuple[int, int, int, float, tuple[tuple[int, int], ...]]:
            if drone_index >= len(drones):
                return (0, 0, 0, 0.0, ())
            drone_id = drones[drone_index]
            used_set = set(used)
            candidates = [solve(drone_index + 1, used)]
            for task_id in task_ids:
                quote = quotes.get((drone_id, task_id))
                if quote is None or task_id in used_set:
                    continue
                tail = solve(
                    drone_index + 1,
                    tuple(sorted((*used, task_id))),
                )
                candidates.append((
                    tail[0] + 1,
                    tail[1] + quote.dfs_depth,
                    tail[2] + int(quote.continuation_affinity),
                    tail[3] + quote.route_cost,
                    ((drone_id, task_id), *tail[4]),
                ))
            return min(
                candidates,
                key=lambda item: (
                    -item[0],
                    -item[1],
                    -item[2],
                    item[3],
                    item[4],
                ),
            )

        _count, _depth, _affinity, _cost, pairs = solve(0, ())
        selected = tuple(
            (drone_id, task_id, quotes[(drone_id, task_id)].route)
            for drone_id, task_id in pairs
        )
        return tuple(quotes.values()), selected

    def _known_free_reachable_mask(
        self,
        rover_slam: SlamSnapshot,
    ) -> np.ndarray:
        """Return the rover's eight-connected, confidently free region."""
        occupancy = np.asarray(rover_slam.occupancy)
        confidence = np.asarray(rover_slam.confidence)
        known_free = (
            (occupancy == FREE)
            & (confidence >= self.registry.confidence_threshold)
        )
        reachable = np.zeros_like(known_free, dtype=bool)
        rover_x = self.rover_position[0] - int(rover_slam.origin[0])
        rover_y = self.rover_position[1] - int(rover_slam.origin[1])
        height, width = known_free.shape
        if not (
            0 <= rover_x < width
            and 0 <= rover_y < height
            and known_free[rover_y, rover_x]
        ):
            return reachable
        _count, labels = cv2.connectedComponents(
            known_free.astype(np.uint8),
            connectivity=8,
        )
        rover_label = int(labels[rover_y, rover_x])
        if rover_label > 0:
            reachable = labels == rover_label
        return reachable

    @staticmethod
    def _point_is_reachable(
        point: Position,
        reachable_mask: np.ndarray,
        origin: Position = (0, 0),
    ) -> bool:
        x = int(point[0]) - int(origin[0])
        y = int(point[1]) - int(origin[1])
        return bool(
            0 <= y < reachable_mask.shape[0]
            and 0 <= x < reachable_mask.shape[1]
            and reachable_mask[y, x]
        )

    def _start_rover_scan_round(self, rover_slam: SlamSnapshot) -> None:
        participants = tuple(sorted(self._waiting))
        headings = self._full_scan_headings()
        plans = {
            drone_id: tuple(
                heading for index, heading in enumerate(headings)
                if index % len(participants) == participant_index
            )
            for participant_index, drone_id in enumerate(participants)
        }
        round_id = self._next_round_id
        self._next_round_id += 1
        self._discovery_round = DiscoveryRound(
            round_id=round_id,
            kind=DiscoveryKind.ROVER_SCAN,
            state=DiscoveryState.DISPATCHED,
            participant_ids=participants,
            scan_plans=plans,
            probe_targets={},
            reports={},
            rover_slam_version_before=rover_slam.version,
            registry_revision_before=self.registry.revision,
            ring=0,
        )
        for drone_id in participants:
            self._issue(
                drone_id,
                ExplorationDirective(
                    directive_id=self._allocate_directive_id(),
                    kind=DirectiveKind.ROVER_SCAN,
                    round_id=round_id,
                    scan_headings=plans[drone_id],
                    reason="coordinated_initial_scan",
                ),
            )

    def _start_radial_round(
        self,
        candidates: tuple[int, ...],
        rover_slam: SlamSnapshot,
        *,
        reachable_mask: np.ndarray,
    ) -> bool:
        plans: dict[int, tuple[int, ...]] = {}
        targets: dict[int, Position] = {}
        routes: dict[int, tuple[Position, ...]] = {}
        reserved: list[Position] = []
        for drone_id in candidates:
            if drone_id in self._probe_exhausted:
                continue
            selected = self._probe_target(
                drone_id,
                rover_slam,
                reserved,
                reachable_mask=reachable_mask,
            )
            if selected is None:
                self._probe_exhausted.add(drone_id)
                continue
            target, route = selected
            targets[drone_id] = target
            routes[drone_id] = route
            plans[drone_id] = self._full_scan_headings()
            reserved.append(target)
        if not targets:
            return False
        participants = tuple(sorted(targets))
        round_id = self._next_round_id
        self._next_round_id += 1
        ring = max(self._probe_rings[drone_id] for drone_id in participants) + 1
        self._discovery_round = DiscoveryRound(
            round_id=round_id,
            kind=DiscoveryKind.RADIAL_PROBE,
            state=DiscoveryState.DISPATCHED,
            participant_ids=participants,
            scan_plans=plans,
            probe_targets=targets,
            reports={},
            rover_slam_version_before=rover_slam.version,
            registry_revision_before=self.registry.revision,
            ring=ring,
        )
        for drone_id in participants:
            self._issue(
                drone_id,
                ExplorationDirective(
                    directive_id=self._allocate_directive_id(),
                    kind=DirectiveKind.RADIAL_PROBE,
                    round_id=round_id,
                    scan_headings=plans[drone_id],
                    probe_target=targets[drone_id],
                    outbound_route=routes[drone_id],
                    reason="insufficient_component_capacity",
                ),
            )
        return True

    def _probe_target(
        self,
        drone_id: int,
        rover_slam: SlamSnapshot,
        reserved: Iterable[Position],
        *,
        reachable_mask: np.ndarray,
    ) -> tuple[Position, tuple[Position, ...]] | None:
        ys, xs = np.nonzero(reachable_mask)
        heading = 360.0 * drone_id / self.drone_count
        radians = math.radians(heading)
        axis_x, axis_y = math.sin(radians), -math.cos(radians)
        previous_radius = self._probe_radius[drone_id]
        target_radius = previous_radius + self.sensor_range * 0.75
        ranked: list[tuple[float, float, int, int, Position]] = []
        for x_value, y_value in zip(xs, ys):
            point = (
                int(x_value) + int(rover_slam.origin[0]),
                int(y_value) + int(rover_slam.origin[1]),
            )
            dx = point[0] - self.rover_position[0]
            dy = point[1] - self.rover_position[1]
            projection = dx * axis_x + dy * axis_y
            if projection <= previous_radius + self.sensor_range * 0.1:
                continue
            lateral = abs(dx * axis_y - dy * axis_x)
            distance = math.hypot(dx, dy)
            if distance <= 1e-9 or lateral / distance > 0.5:
                continue
            if any(
                math.dist(point, other) < self.sensor_range * 0.5
                for other in reserved
            ):
                continue
            ranked.append((
                abs(projection - target_radius),
                lateral,
                point[1],
                point[0],
                point,
            ))
        for _radial_error, _lateral, _y, _x, point in sorted(ranked):
            route_cost = math.dist(self.rover_position, point)
            requirement = EnergyRequirement(
                route_to_task_cost=route_cost,
                next_action_cost=float(len(self._full_scan_headings())),
                route_home_cost=route_cost,
                safety_reserve=0.0,
            )
            if not self.energy_policy.can_accept(
                self._energy_states[drone_id],
                requirement,
            ):
                continue
            return point, ()
        return None

    def _full_scan_headings(self) -> tuple[int, ...]:
        count = max(1, int(math.ceil(360.0 / self.sensor_fov_deg)))
        spacing = 360.0 / count
        return tuple(
            int(round(index * spacing)) % 360
            for index in range(count)
        )

    def _has_outstanding_work(self) -> bool:
        return bool(
            self._claims_by_task
            or any(
                task.state in {
                    TaskState.READY,
                    TaskState.CLAIMED,
                    TaskState.ACTIVE,
                    TaskState.SUSPENDED,
                    TaskState.BLOCKED,
                }
                and self.registry.components[task.component_id].state
                == ComponentState.ACTIVE
                for task in self._tasks.values()
            )
        )

    def _issue(self, drone_id: int, directive: ExplorationDirective) -> None:
        if drone_id in self._pending_directives:
            return
        self._pending_directives[drone_id] = directive
        self._ready_events[drone_id].set()

    def _issue_home(self, drone_ids: Iterable[int], *, reason: str) -> None:
        for drone_id in drone_ids:
            self._issue(
                int(drone_id),
                ExplorationDirective(
                    directive_id=self._allocate_directive_id(),
                    kind=DirectiveKind.HOME,
                    reason=reason,
                ),
            )

    def _allocate_directive_id(self) -> int:
        directive_id = self._next_directive_id
        self._next_directive_id += 1
        return directive_id

    @staticmethod
    def _path_distance(path: Iterable[Position]) -> float:
        points = tuple(path)
        return sum(
            math.dist(previous, current)
            for previous, current in zip(points, points[1:])
        )
