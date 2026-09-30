"""Rover-owned frontier tasks, claims, and bounded discovery rounds."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
import itertools
import math
import threading
import time
from typing import Iterable

import cv2
import numpy as np

from mapping.frontier_registry import (
    CausalTransition,
    ComponentState,
    ComponentWorkUnit,
    ExplorationMode,
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
from navigation.highway import (
    HIGHWAY_COMPLETE,
    HighwayGraphSnapshot,
    HighwayRoute,
)


Position = tuple[int, int]
TaskId = int
ClaimToken = int
_BATCH_LEASE_FINALIZATION_RESERVE_MS = 50.0


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
    COMPONENT_BATCH = "component_batch"
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
class BatchMember:
    """One unchanged component task carried by a batch directive."""

    task: ExplorationTask
    work_units: tuple[ComponentWorkUnit, ...]
    claim: ClaimLease
    estimated_service_cost: float


@dataclass(frozen=True)
class SpatialLease:
    """Exclusive raster envelope for provisional drone-local work."""

    lease_id: int
    owner_drone_id: int
    directive_id: int
    issued_revision: int
    cell_spans: tuple[tuple[int, int, int], ...]
    member_task_ids: tuple[int, ...]

    def contains(self, point: Position) -> bool:
        x, y = (int(value) for value in point)
        return any(
            row == y and left <= x <= right
            for row, left, right in self.cell_spans
        )

    @property
    def cell_count(self) -> int:
        return sum(right - left + 1 for _row, left, right in self.cell_spans)


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
    scan_position: Position | None = None
    scan_heading: int | None = None
    frontier_position: Position | None = None
    frontier_heading: int | None = None


@dataclass(frozen=True)
class BatchMemberReport:
    """Independent result and fencing token for one batch member."""

    task_id: int
    component_id: int
    component_revision: int
    claim_token: ClaimToken
    disposition: str
    work_unit_outcomes: tuple[WorkUnitOutcome, ...] = ()
    causal_transitions: tuple[CausalTransition, ...] = ()
    suspension: TaskSuspension | None = None


@dataclass(frozen=True)
class ProvisionalFrontierObservation:
    """Lease-authorized local work with no rover work-unit authority."""

    observation_id: int
    source_task_id: int
    cells: frozenset[Position]
    scan_position: Position
    scan_heading: int
    sensor_newly_known_cells: int
    sensor_confidence_gain: float
    local_route_distance: float
    causal_predecessor_id: int | None = None


@dataclass(frozen=True)
class FocusedFrontierBatchEvaluation:
    """One deterministic observe/active batching decision for telemetry."""

    evaluation_id: int
    mode: str
    drone_id: int
    seed_task_id: int
    candidate_task_id: int
    member_task_ids: tuple[int, ...]
    separate_cost: float
    combined_cost: float
    avoided_round_trip: float
    detour_cost: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class FocusedFrontierBatchPlanningSummary:
    """One bounded scheduling pass, separate from candidate evaluations."""

    planning_id: int
    mode: str
    elapsed_ms: float
    route_queries: int
    route_cache_hits: int
    candidate_count: int
    planned_batch_count: int
    status: str
    highway_version: int | None


@dataclass(frozen=True)
class _FocusedFrontierBatchPlan:
    """Prevalidated economics and lease preview reused at issue time."""

    drone_id: int
    task_ids: tuple[int, ...]
    ordered_task_ids: tuple[int, ...]
    lease_preview: SpatialLease
    separate_cost: float
    combined_cost: float
    avoided_round_trip: float
    detour_cost: float


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
    outbound_distance: float = 0.0
    service_distance: float = 0.0
    return_distance: float = 0.0
    lease_id: int | None = None
    batch_member_reports: tuple[BatchMemberReport, ...] = ()
    provisional_observations: tuple[ProvisionalFrontierObservation, ...] = ()


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
    assignment_policy: str = "ordinary"
    estimated_outbound_cost: float = 0.0
    estimated_round_trip_cost: float = 0.0
    batch_members: tuple[BatchMember, ...] = ()
    spatial_lease: SpatialLease | None = None
    estimated_separate_cost: float = 0.0
    estimated_combined_cost: float = 0.0
    estimated_avoided_round_trip: float = 0.0
    estimated_detour_cost: float = 0.0
    issued_work_unit_ids: tuple[int, ...] = ()
    maximum_total_components: int = 0
    maximum_detour_distance: float = 0.0
    minimum_avoided_round_trip: float = 0.0
    maximum_service_seconds: float = 0.0
    maximum_total_dfs_nodes: int = 0
    maximum_consecutive_low_gain_scans: int = 0
    low_gain_maximum_new_cells: int = 0
    low_gain_maximum_confidence_gain: float = 0.0
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
    focused_endgame: bool = False
    spatial_leases: tuple[SpatialLease, ...] = ()


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
    batch_evaluations: tuple[FocusedFrontierBatchEvaluation, ...] = ()
    batch_member_statuses: tuple[tuple[int, str], ...] = ()
    provisional_retired_work_unit_ids: tuple[int, ...] = ()
    report_replayed: bool = False
    batch_planning_summary: FocusedFrontierBatchPlanningSummary | None = None
    highway_snapshot: HighwayGraphSnapshot | None = None


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
        focused_frontier_batch_mode: str = "off",
        focused_frontier_batch_maximum_claimed_components: int = 3,
        focused_frontier_batch_maximum_total_components: int = 4,
        focused_frontier_batch_lease_margin_sensor_ranges: float = 1.0,
        focused_frontier_batch_maximum_detour_sensor_ranges: float = 2.0,
        focused_frontier_batch_minimum_avoided_round_trip_sensor_ranges: float = 0.25,
        focused_frontier_batch_maximum_service_seconds: float = 45.0,
        focused_frontier_batch_maximum_total_dfs_nodes: int = 48,
        focused_frontier_batch_maximum_consecutive_low_gain_scans: int = 2,
        focused_frontier_batch_low_gain_maximum_new_cells: int = 1,
        focused_frontier_batch_low_gain_maximum_confidence_gain: float = 1.0,
        focused_frontier_batch_maximum_planning_ms: float = 250.0,
        focused_frontier_batch_maximum_route_queries: int = 128,
    ) -> None:
        if drone_count <= 0:
            raise ValueError("exploration drone_count must be positive")
        self.map_shape = tuple(int(value) for value in map_shape)
        self.drone_count = int(drone_count)
        self.rover_position = tuple(int(value) for value in rover_position)
        self.sensor_range = float(sensor_range)
        self.sensor_fov_deg = float(sensor_fov_deg)
        self.energy_policy = energy_policy or UnlimitedEnergyPolicy()
        self.focused_frontier_batch_mode = str(focused_frontier_batch_mode).casefold()
        if self.focused_frontier_batch_mode not in {"off", "observe", "active"}:
            raise ValueError("invalid focused frontier batch mode")
        self.focused_frontier_batch_maximum_claimed_components = max(
            1, int(focused_frontier_batch_maximum_claimed_components),
        )
        self.focused_frontier_batch_maximum_total_components = max(
            self.focused_frontier_batch_maximum_claimed_components,
            int(focused_frontier_batch_maximum_total_components),
        )
        self.focused_frontier_batch_lease_margin = max(
            0.0,
            float(focused_frontier_batch_lease_margin_sensor_ranges) * self.sensor_range,
        )
        self.focused_frontier_batch_maximum_detour = max(
            0.0,
            float(focused_frontier_batch_maximum_detour_sensor_ranges) * self.sensor_range,
        )
        self.focused_frontier_batch_minimum_avoided_round_trip = max(
            0.0,
            float(focused_frontier_batch_minimum_avoided_round_trip_sensor_ranges)
            * self.sensor_range,
        )
        self.focused_frontier_batch_maximum_service_seconds = max(
            0.0, float(focused_frontier_batch_maximum_service_seconds),
        )
        self.focused_frontier_batch_maximum_total_dfs_nodes = max(
            1, int(focused_frontier_batch_maximum_total_dfs_nodes),
        )
        self.focused_frontier_batch_maximum_consecutive_low_gain_scans = max(
            1, int(focused_frontier_batch_maximum_consecutive_low_gain_scans),
        )
        self.focused_frontier_batch_low_gain_maximum_new_cells = max(
            0, int(focused_frontier_batch_low_gain_maximum_new_cells),
        )
        self.focused_frontier_batch_low_gain_maximum_confidence_gain = max(
            0.0, float(focused_frontier_batch_low_gain_maximum_confidence_gain),
        )
        self.focused_frontier_batch_maximum_planning_ms = max(
            0.0, float(focused_frontier_batch_maximum_planning_ms),
        )
        self.focused_frontier_batch_maximum_route_queries = max(
            1, int(focused_frontier_batch_maximum_route_queries),
        )
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
        self._leases_by_id: dict[int, SpatialLease] = {}
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
        self._next_lease_id = 0
        self._next_batch_evaluation_id = 0
        self._next_batch_planning_id = 0
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
        self._focused_endgame = False
        self._last_rover_slam: SlamSnapshot | None = None
        self._last_reconcile_result: RegistryReconcileResult | None = None
        self._pending_batch_evaluations: list[FocusedFrontierBatchEvaluation] = []
        self._last_batch_planning_summary: FocusedFrontierBatchPlanningSummary | None = None
        self._highway_snapshot: HighwayGraphSnapshot | None = None
        self._last_batch_member_statuses: tuple[tuple[int, str], ...] = ()
        self._last_provisional_retired_ids: tuple[int, ...] = ()
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
            focused_endgame=self._focused_endgame,
            spatial_leases=tuple(
                self._leases_by_id[key] for key in sorted(self._leases_by_id)
            ),
        )

    def update_rover_position(self, position: Position) -> None:
        """Publish the rendezvous point used by routing and discovery."""
        with self._lock:
            self.rover_position = tuple(int(value) for value in position)

    def update_highway_snapshot(
        self,
        snapshot: HighwayGraphSnapshot | None,
    ) -> None:
        """Install one immutable rover-owned routing snapshot."""
        with self._lock:
            self._highway_snapshot = snapshot

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
            self._last_batch_member_statuses = ()
            self._last_provisional_retired_ids = ()
            self._last_batch_planning_summary = None
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
                        report_replayed=True,
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
                batch_evaluations=self._take_batch_evaluations(),
                batch_planning_summary=self._last_batch_planning_summary,
                batch_member_statuses=self._last_batch_member_statuses,
                provisional_retired_work_unit_ids=(
                    self._last_provisional_retired_ids
                ),
                report_replayed=duplicate_report,
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
            if directive.kind == DirectiveKind.COMPONENT_BATCH:
                for member in directive.batch_members:
                    self._tasks[member.claim.task_id].state = TaskState.ACTIVE
                    self.registry.activate_work_units(
                        member.claim.work_unit_ids
                    )
            return CoordinationResult(
                arrived=True,
                waiting=False,
                mission_exhausted=(directive.kind == DirectiveKind.HOME),
                directive=directive,
            )

    def waiting_contact_lost(self, drone_id: int) -> None:
        """A past check-in no longer counts as physical team quiescence."""
        with self._lock:
            self._waiting.discard(int(drone_id))

    def stop(self) -> None:
        with self._lock:
            self._mission_exhausted = True
            self._leases_by_id.clear()
            if self._discovery_round is not None:
                self._discovery_round.state = DiscoveryState.ABORTED
            self._issue_home(
                tuple(range(self.drone_count)),
                reason="coordinator_stopped",
            )

    def _take_batch_evaluations(self) -> tuple[FocusedFrontierBatchEvaluation, ...]:
        evaluations = tuple(self._pending_batch_evaluations)
        self._pending_batch_evaluations.clear()
        return evaluations

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

        if report.kind == DirectiveKind.COMPONENT_BATCH:
            return self._accept_batch_report(
                drone_id,
                active,
                report,
                rover_slam,
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
                    if (
                        outcome.scan_position is not None
                        and outcome.scan_heading is not None
                    ):
                        self.registry.record_scan_result(
                            outcome.work_unit_id,
                            scan_position=outcome.scan_position,
                            scan_heading=outcome.scan_heading,
                            frontier_position=outcome.frontier_position,
                            frontier_heading=outcome.frontier_heading,
                            newly_known_cells=outcome.sensor_newly_known_cells,
                            confidence_gain=outcome.sensor_confidence_gain,
                            rover_slam=rover_slam,
                        )
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

    def _accept_batch_report(
        self,
        drone_id: int,
        active: ExplorationDirective,
        report: CoordinationReport,
        rover_slam: SlamSnapshot,
    ) -> tuple[bool, RegistryReconcileResult | None]:
        """Validate the full envelope, then commit independent members once."""
        validated = self._validate_batch_report(
            drone_id,
            active,
            report,
            rover_slam,
        )
        if validated is None:
            return False, None
        member_reports = validated
        statuses: list[tuple[int, str]] = []
        causal: list[CausalTransition] = []
        for member, member_report in member_reports:
            live_claim = self._claims_by_token.get(member.claim.token)
            task_claim = self._claims_by_task.get(member.task.task_id)
            applicable = bool(
                live_claim == member.claim
                and task_claim == member.claim
                and all(
                    (
                        unit := self.registry.work_units.get(unit_id)
                    ) is not None
                    and unit.component_id == member.task.component_id
                    and unit.component_revision
                    == member.task.component_revision
                    and unit.state in {
                        WorkUnitState.CLAIMED,
                        WorkUnitState.ACTIVE,
                    }
                    for unit_id in member.claim.work_unit_ids
                )
            )
            if not applicable:
                statuses.append((member.task.task_id, "stale"))
                if live_claim == member.claim or task_claim == member.claim:
                    self.registry.release_work_units(
                        member.claim.work_unit_ids
                    )
                    self._release_claim(member.claim)
                    task = self._tasks[member.task.task_id]
                    component = self.registry.components.get(
                        task.component_id
                    )
                    task.state = (
                        TaskState.CANCELLED
                        if component is None
                        or component.state in {
                            ComponentState.SPLIT,
                            ComponentState.MERGED,
                            ComponentState.RESOLVED,
                        }
                        else TaskState.SUSPENDED
                    )
                    task.suspension = None
                continue
            statuses.append((
                member.task.task_id,
                "suspended"
                if member_report.suspension is not None
                else "applicable",
            ))
            completed_ids: set[int] = set()
            for outcome in member_report.work_unit_outcomes:
                if self.registry.complete_work_unit(
                    outcome.work_unit_id,
                    reason=outcome.disposition,
                ):
                    completed_ids.add(outcome.work_unit_id)
                    if (
                        outcome.scan_position is not None
                        and outcome.scan_heading is not None
                    ):
                        self.registry.record_scan_result(
                            outcome.work_unit_id,
                            scan_position=outcome.scan_position,
                            scan_heading=outcome.scan_heading,
                            frontier_position=outcome.frontier_position,
                            frontier_heading=outcome.frontier_heading,
                            newly_known_cells=(
                                outcome.sensor_newly_known_cells
                            ),
                            confidence_gain=outcome.sensor_confidence_gain,
                            rover_slam=rover_slam,
                        )
            task = self._tasks[member.claim.task_id]
            self._last_reported_task_by_drone[drone_id] = task.task_id
            self._preferred_parent_task_by_component[
                task.component_id
            ] = task.task_id
            rejection_key = (drone_id, task.task_id)
            if (
                member_report.suspension is not None
                and member_report.suspension.reason == "route_unreachable"
            ):
                self._route_rejections[rejection_key] = task.component_revision
            else:
                self._route_rejections.pop(rejection_key, None)
            self._release_claim(member.claim)
            remaining = tuple(
                unit_id for unit_id in member.claim.work_unit_ids
                if unit_id not in completed_ids
            )
            if remaining:
                task.state = TaskState.SUSPENDED
                task.suspension = member_report.suspension
                self.registry.release_work_units(remaining)
            else:
                task.state = TaskState.COMPLETE
                task.suspension = None
            causal.extend(member_report.causal_transitions)

        reconcile = self._reconcile(rover_slam, causal)
        retired = self.registry.retire_provisional_work_units(
            (item.cells for item in report.provisional_observations),
            preexisting_work_unit_ids=active.issued_work_unit_ids,
        )
        if retired:
            self._sync_tasks()
            reconcile = replace(
                reconcile,
                ready_work_unit_ids=tuple(
                    unit_id for unit_id in reconcile.ready_work_unit_ids
                    if unit_id not in set(retired)
                ),
            )
            self._last_reconcile_result = reconcile
        if active.spatial_lease is not None:
            self._leases_by_id.pop(active.spatial_lease.lease_id, None)
        self._accepted_report_ids.add(report.report_id)
        self._last_batch_member_statuses = tuple(statuses)
        self._last_provisional_retired_ids = retired
        return True, reconcile

    def _validate_batch_report(
        self,
        drone_id: int,
        active: ExplorationDirective,
        report: CoordinationReport,
        rover_slam: SlamSnapshot,
    ) -> tuple[tuple[BatchMember, BatchMemberReport], ...] | None:
        """Return ordered member pairs only when the whole report is valid."""
        lease = active.spatial_lease
        occupancy = np.asarray(rover_slam.occupancy)
        confidence = np.asarray(rover_slam.confidence)
        if (
            not active.batch_members
            or lease is None
            or occupancy.shape != self.map_shape
            or confidence.shape != self.map_shape
            or occupancy.ndim != 2
            or confidence.ndim != 2
            or report.lease_id != lease.lease_id
            or lease.owner_drone_id != int(drone_id)
            or lease.directive_id != active.directive_id
            or self._leases_by_id.get(lease.lease_id) != lease
            or report.round_id is not None
            or report.task_id is not None
            or report.component_id is not None
            or report.claim_token is not None
            or report.work_unit_outcomes
            or report.causal_transitions
            or report.suspension is not None
            or len(report.provisional_observations)
            > active.maximum_total_dfs_nodes
            or report.sensor_newly_known_cells < 0
            or report.sensor_confidence_gain < 0.0
            or not all(math.isfinite(value) and value >= 0.0 for value in (
                report.sensor_confidence_gain,
                report.outbound_distance,
                report.service_distance,
                report.return_distance,
            ))
        ):
            return None
        expected = {member.task.task_id: member for member in active.batch_members}
        received = {item.task_id: item for item in report.batch_member_reports}
        if (
            len(received) != len(report.batch_member_reports)
            or set(received) != set(expected)
        ):
            return None
        seen_outcomes: set[int] = set()
        seen_tokens: set[int] = set()
        ordered: list[tuple[BatchMember, BatchMemberReport]] = []
        allowed_dispositions = {
            "complete",
            "suspended",
            "unreachable",
            "budget_exhausted",
            "sensor_timeout",
        }
        for member in active.batch_members:
            item = received[member.task.task_id]
            outcome_ids = tuple(
                outcome.work_unit_id for outcome in item.work_unit_outcomes
            )
            if (
                item.component_id != member.task.component_id
                or item.component_revision != member.task.component_revision
                or item.claim_token != member.claim.token
                or item.claim_token in seen_tokens
                or item.disposition not in allowed_dispositions
                or len(outcome_ids) != len(set(outcome_ids))
                or any(unit_id in seen_outcomes for unit_id in outcome_ids)
                or any(
                    unit_id not in member.claim.work_unit_ids
                    for unit_id in outcome_ids
                )
                or any(
                    outcome.sensor_newly_known_cells < 0
                    or outcome.sensor_confidence_gain < 0.0
                    or not math.isfinite(outcome.sensor_confidence_gain)
                    for outcome in item.work_unit_outcomes
                )
                or any(
                    transition.predecessor_id != member.task.component_id
                    or transition.report_id != report.report_id
                    for transition in item.causal_transitions
                )
            ):
                return None
            seen_tokens.add(item.claim_token)
            seen_outcomes.update(outcome_ids)
            remaining = tuple(
                unit_id for unit_id in member.claim.work_unit_ids
                if unit_id not in set(outcome_ids)
            )
            suspension = item.suspension
            if suspension is not None and (
                suspension.task_id != member.task.task_id
                or suspension.claim_token != member.claim.token
                or suspension.drone_id != int(drone_id)
                or tuple(suspension.remaining_work_unit_ids) != remaining
                or item.disposition not in {
                    "suspended",
                    "unreachable",
                    "budget_exhausted",
                    "sensor_timeout",
                }
            ):
                return None
            if suspension is None and remaining:
                return None
            if suspension is None and item.disposition != "complete":
                return None
            ordered.append((member, item))

        observation_ids: set[int] = set()
        member_task_ids = set(expected)
        component_by_task = {
            member.task.task_id: member.task.component_id
            for member in active.batch_members
        }
        for observation in report.provisional_observations:
            if (
                observation.observation_id in observation_ids
                or observation.source_task_id not in member_task_ids
                or observation.causal_predecessor_id not in {
                    None,
                    component_by_task[observation.source_task_id],
                }
                or not observation.cells
                or not lease.contains(observation.scan_position)
                or any(not lease.contains(point) for point in observation.cells)
                or observation.sensor_newly_known_cells < 0
                or observation.sensor_confidence_gain < 0.0
                or observation.local_route_distance < 0.0
                or not all(math.isfinite(value) for value in (
                    observation.sensor_confidence_gain,
                    observation.local_route_distance,
                ))
            ):
                return None
            observation_ids.add(observation.observation_id)
        return tuple(ordered)

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
        self._focused_endgame = self._focused_endgame_is_active()
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
            focused_endgame=self._focused_endgame,
        )
        quote_by_pair = {
            (quote.drone_id, quote.task_id): quote for quote in quotes
        }
        batch_plans: dict[int, _FocusedFrontierBatchPlan] = {}
        if self._focused_endgame and self.focused_frontier_batch_mode != "off":
            batch_plans = self._plan_focused_frontier_batches(
                selected,
                tasks,
                quote_by_pair,
                rover_slam,
                reachable,
            )
        assigned_drones: set[int] = set()
        leader_tasks: list[tuple[int, ExplorationTask]] = []
        for drone_id, task_id, route in selected:
            task = self._tasks[task_id]
            quote = quote_by_pair[(drone_id, task_id)]
            batch_plan = batch_plans.get(drone_id)
            if (
                self.focused_frontier_batch_mode == "active"
                and batch_plan is not None
                and len(batch_plan.task_ids) > 1
                and self._issue_component_batch(
                    drone_id,
                    batch_plan,
                )
            ):
                assigned_drones.add(drone_id)
                leader_tasks.append((drone_id, task))
                continue
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
                    assignment_policy=(
                        "focused_endgame_round_trip"
                        if self._focused_endgame
                        else "ordinary_depth_continuation"
                    ),
                    estimated_outbound_cost=quote.route_cost,
                    estimated_round_trip_cost=quote.route_cost * 2.0,
                    reason="reachable_component_work",
                ),
            )
            assigned_drones.add(drone_id)
            leader_tasks.append((drone_id, task))

        remaining = tuple(
            drone_id for drone_id in idle if drone_id not in assigned_drones
        )
        # Bootstrap followers belong to the first component dispatch. If all
        # drones received work then, do not first launch followers at endgame.
        bootstrap_dispatch = bool(
            self._phase == ExplorationPhase.COMPONENT_EXPLORATION
            and leader_tasks
            and not self._bootstrap_followers_issued
        )
        if bootstrap_dispatch:
            self._bootstrap_followers_issued = True
        if not remaining:
            return
        reachable_outstanding = bool(self._claims_by_task or quotes)
        if bootstrap_dispatch:
            self._issue_bootstrap_followers(remaining, leader_tasks)
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

    def _plan_focused_frontier_batches(
        self,
        selected: tuple[tuple[int, int, tuple[Position, ...]], ...],
        tasks: tuple[ExplorationTask, ...],
        quotes: dict[tuple[int, int], RouteQuote],
        rover_slam: SlamSnapshot,
        reachable: np.ndarray,
    ) -> dict[int, _FocusedFrontierBatchPlan]:
        """Attach leftovers using bounded highway queries after seed matching."""
        started = time.perf_counter()
        deadline = (
            started + self.focused_frontier_batch_maximum_planning_ms / 1000.0
        )
        groups = {
            drone_id: [task_id]
            for drone_id, task_id, _route in selected
        }
        task_by_id = {task.task_id: task for task in tasks}
        seeded = {task_id for _drone_id, task_id, _route in selected}
        remaining = {
            task.task_id for task in tasks if task.task_id not in seeded
        }
        route_cache: dict[tuple[Position, Position], HighwayRoute] = {}
        route_queries = 0
        route_cache_hits = 0
        candidate_count = 0
        planning_status = "complete"

        def planning_budget_exhausted() -> bool:
            nonlocal planning_status
            if planning_status == "planning_budget":
                return True
            if time.perf_counter() >= deadline:
                planning_status = "planning_budget"
                return True
            return False

        def route(first: Position, second: Position) -> HighwayRoute:
            nonlocal route_queries, route_cache_hits, planning_status
            key = (tuple(first), tuple(second))
            reverse = (key[1], key[0])
            # Cached routes still participate in the wall-clock budget.  A
            # cache-heavy tour must not run past the deadline merely because
            # it no longer enters the graph query below.
            if planning_budget_exhausted():
                return HighwayRoute("budget_exhausted")
            if key in route_cache:
                route_cache_hits += 1
                return route_cache[key]
            if route_queries >= self.focused_frontier_batch_maximum_route_queries:
                planning_status = "planning_budget"
                return HighwayRoute("budget_exhausted")
            route_queries += 1
            snapshot = self._highway_snapshot
            if snapshot is None:
                path = self._direct_reachable_path(
                    key[0], key[1], reachable, rover_slam.origin,
                    deadline=deadline,
                )
                result = HighwayRoute(
                    status=HIGHWAY_COMPLETE if path else "unavailable",
                    path=path,
                    cost=self._path_distance(path) if path else math.inf,
                )
            else:
                result = snapshot.route(
                    key[0],
                    key[1],
                    maximum_query_ms=max(
                        0.0,
                        min(
                            50.0,
                            (deadline - time.perf_counter()) * 1000.0,
                        ),
                    ),
                    maximum_connector_expansions=4096,
                )
                if result.complete and not self._path_is_reachable(
                    result.path,
                    reachable,
                    rover_slam.origin,
                    deadline=deadline,
                ):
                    result = HighwayRoute(
                        "stale",
                        snapshot_version=result.snapshot_version,
                        elapsed_ms=result.elapsed_ms,
                        expanded_nodes=result.expanded_nodes,
                    )
            route_cache[key] = result
            route_cache[reverse] = replace(
                result,
                path=tuple(reversed(result.path)),
            )
            planning_budget_exhausted()
            return result

        def tour(task_ids: Iterable[int]) -> tuple[float, tuple[int, ...]]:
            ids = tuple(task_ids)
            best = math.inf
            best_order: tuple[int, ...] = ()
            for order in itertools.permutations(ids):
                if planning_budget_exhausted():
                    break
                points = (
                    self.rover_position,
                    *(task_by_id[item].preferred_entry for item in order),
                    self.rover_position,
                )
                legs: list[HighwayRoute] = []
                for first, second in zip(points, points[1:]):
                    legs.append(route(first, second))
                    if planning_status == "planning_budget":
                        break
                if planning_status == "planning_budget":
                    break
                cost = sum(item.cost for item in legs)
                if (cost, order) < (best, best_order or order):
                    best = cost
                    best_order = order
            return best, best_order

        while remaining and not planning_budget_exhausted():
            best: tuple[float, int, int, float, float, float] | None = None
            for drone_id in sorted(groups):
                if planning_budget_exhausted():
                    break
                member_ids = tuple(groups[drone_id])
                if len(member_ids) >= self.focused_frontier_batch_maximum_claimed_components:
                    continue
                base_cost, _base_order = tour(member_ids)
                for candidate_id in sorted(remaining):
                    if planning_budget_exhausted():
                        break
                    candidate_count += 1
                    if (drone_id, candidate_id) not in quotes:
                        continue
                    candidate_cost, _candidate_order = tour((candidate_id,))
                    separate_cost = base_cost + candidate_cost
                    combined_ids = (*member_ids, candidate_id)
                    combined_cost, _combined_order = tour(combined_ids)
                    avoided = separate_cost - combined_cost
                    seed_cost, _seed_order = tour((member_ids[0],))
                    detour = combined_cost - seed_cost
                    reason = "accepted"
                    if planning_status == "planning_budget":
                        reason = "planning_budget"
                    elif not all(math.isfinite(value) for value in (
                        separate_cost, combined_cost, avoided, detour,
                    )):
                        reason = "unreachable_pair"
                    elif avoided <= 1e-9:
                        reason = "nonpositive_avoided_round_trip"
                    elif avoided + 1e-9 < (
                        self.focused_frontier_batch_minimum_avoided_round_trip
                    ):
                        reason = "insufficient_avoided_round_trip"
                    elif detour > self.focused_frontier_batch_maximum_detour + 1e-9:
                        reason = "detour_limit"
                    else:
                        effort = sum(
                            task_by_id[item].estimated_effort
                            for item in combined_ids
                        )
                        if effort > (
                            self.focused_frontier_batch_maximum_service_seconds + 1e-9
                        ):
                            reason = "service_time_limit"
                        else:
                            requirement = EnergyRequirement(
                                route_to_task_cost=combined_cost / 2.0,
                                next_action_cost=effort,
                                route_home_cost=combined_cost / 2.0,
                                safety_reserve=0.0,
                            )
                            if not self.energy_policy.can_accept(
                                self._energy_states[drone_id], requirement,
                            ):
                                reason = "energy_limit"
                    if reason != "accepted":
                        self._record_batch_evaluation(
                            drone_id,
                            member_ids[0],
                            candidate_id,
                            combined_ids,
                            separate_cost,
                            combined_cost,
                            avoided,
                            detour,
                            False,
                            reason,
                        )
                        if planning_status == "planning_budget":
                            break
                        continue
                    candidate = (
                        -avoided,
                        drone_id,
                        candidate_id,
                        separate_cost,
                        combined_cost,
                        detour,
                    )
                    if best is None or candidate < best:
                        best = candidate
                if planning_status == "planning_budget":
                    break
            if planning_status == "planning_budget":
                break
            if best is None:
                break
            negative_avoided, drone_id, candidate_id, separate, combined, detour = best
            groups[drone_id].append(candidate_id)
            remaining.remove(candidate_id)
            self._record_batch_evaluation(
                drone_id,
                groups[drone_id][0],
                candidate_id,
                tuple(groups[drone_id]),
                separate,
                combined,
                -negative_avoided,
                detour,
                True,
                "accepted",
            )

        plans: dict[int, _FocusedFrontierBatchPlan] = {}
        if planning_status == "planning_budget":
            groups = {
                drone_id: [task_ids[0]]
                for drone_id, task_ids in groups.items()
            }
        reserved_previews: list[SpatialLease] = []
        batch_group_ids = tuple(
            drone_id for drone_id in sorted(groups)
            if len(groups[drone_id]) >= 2
        )
        for group_index, drone_id in enumerate(batch_group_ids):
            if planning_budget_exhausted():
                break
            remaining_group_count = len(batch_group_ids) - group_index
            remaining_ms = max(
                0.0,
                (deadline - time.perf_counter()) * 1000.0,
            )
            if remaining_ms < (
                _BATCH_LEASE_FINALIZATION_RESERVE_MS
                * remaining_group_count
            ):
                planning_status = "planning_budget"
                break
            member_ids = tuple(groups[drone_id])
            combined_cost, best_order = tour(member_ids)
            separate_cost = sum(
                tour((task_id,))[0] for task_id in member_ids
            )
            seed_cost = tour((member_ids[0],))[0]
            connectors = tuple(
                route(
                    task_by_id[first_id].preferred_entry,
                    task_by_id[second_id].preferred_entry,
                ).path
                for first_id, second_id in zip(best_order, best_order[1:])
            )
            if (
                planning_status == "planning_budget"
                or not best_order
                or not all(connectors)
            ):
                planning_status = (
                    "planning_budget"
                    if planning_status == "planning_budget"
                    else "route_unavailable"
                )
                break
            remaining_ms = max(
                0.0,
                (deadline - time.perf_counter()) * 1000.0,
            )
            if remaining_ms < (
                _BATCH_LEASE_FINALIZATION_RESERVE_MS
                * remaining_group_count
            ):
                planning_status = "planning_budget"
                break
            preview = self._build_spatial_lease(
                drone_id,
                -1,
                member_ids,
                rover_slam,
                ordered_task_ids=best_order,
                connector_paths=connectors,
                extra_reserved_leases=reserved_previews,
                deadline=deadline,
            )
            if planning_budget_exhausted():
                break
            if preview is None:
                self._record_batch_evaluation(
                    drone_id,
                    member_ids[0],
                    member_ids[-1],
                    member_ids,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    False,
                    "lease_conflict",
                )
                continue
            reserved_previews.append(preview)
            plans[drone_id] = _FocusedFrontierBatchPlan(
                drone_id=drone_id,
                task_ids=member_ids,
                ordered_task_ids=best_order,
                lease_preview=preview,
                separate_cost=separate_cost,
                combined_cost=combined_cost,
                avoided_round_trip=max(0.0, separate_cost - combined_cost),
                detour_cost=max(0.0, combined_cost - seed_cost),
            )
        if planning_status == "planning_budget":
            plans.clear()
        elif not plans:
            planning_status = "no_batch"
        self._last_batch_planning_summary = FocusedFrontierBatchPlanningSummary(
            planning_id=self._next_batch_planning_id,
            mode=self.focused_frontier_batch_mode,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            route_queries=route_queries,
            route_cache_hits=route_cache_hits,
            candidate_count=candidate_count,
            planned_batch_count=len(plans),
            status=planning_status,
            highway_version=(
                None if self._highway_snapshot is None
                else self._highway_snapshot.version
            ),
        )
        self._next_batch_planning_id += 1
        return plans

    def _record_batch_evaluation(
        self,
        drone_id: int,
        seed_task_id: int,
        candidate_task_id: int,
        member_task_ids: tuple[int, ...],
        separate_cost: float,
        combined_cost: float,
        avoided_round_trip: float,
        detour_cost: float,
        accepted: bool,
        reason: str,
    ) -> None:
        evaluation = FocusedFrontierBatchEvaluation(
            evaluation_id=self._next_batch_evaluation_id,
            mode=self.focused_frontier_batch_mode,
            drone_id=int(drone_id),
            seed_task_id=int(seed_task_id),
            candidate_task_id=int(candidate_task_id),
            member_task_ids=tuple(int(value) for value in member_task_ids),
            separate_cost=float(separate_cost),
            combined_cost=float(combined_cost),
            avoided_round_trip=float(avoided_round_trip),
            detour_cost=float(detour_cost),
            accepted=bool(accepted),
            reason=str(reason),
        )
        self._next_batch_evaluation_id += 1
        self._pending_batch_evaluations.append(evaluation)

    def _issue_component_batch(
        self,
        drone_id: int,
        plan: _FocusedFrontierBatchPlan,
    ) -> bool:
        task_ids = plan.task_ids
        tasks = tuple(self._tasks[task_id] for task_id in task_ids)
        directive_id = self._allocate_directive_id()
        lease = replace(
            plan.lease_preview,
            lease_id=self._next_lease_id,
            directive_id=directive_id,
        )
        self._next_lease_id += 1
        if lease is None or not self.registry.claim_work_unit_groups(
            (task.work_unit_ids for task in tasks),
            expected_components=(
                (task.component_id, task.component_revision)
                for task in tasks
            ),
        ):
            return False
        claims: list[ClaimLease] = []
        members: list[BatchMember] = []
        for task in tasks:
            token = self._next_claim_token
            self._next_claim_token += 1
            claim = ClaimLease(
                task_id=task.task_id,
                work_unit_ids=task.work_unit_ids,
                owner_drone_id=drone_id,
                token=token,
                issued_revision=self.registry.revision,
            )
            claims.append(claim)
            self._claims_by_task[task.task_id] = claim
            self._claims_by_token[token] = claim
            task.state = TaskState.CLAIMED
            members.append(BatchMember(
                task=replace(task),
                work_units=tuple(
                    replace(self.registry.work_units[unit_id])
                    for unit_id in task.work_unit_ids
                ),
                claim=claim,
                estimated_service_cost=task.estimated_effort,
            ))
        self._leases_by_id[lease.lease_id] = lease
        self._issue(
            drone_id,
            ExplorationDirective(
                directive_id=directive_id,
                kind=DirectiveKind.COMPONENT_BATCH,
                batch_members=tuple(members),
                spatial_lease=lease,
                assignment_policy="focused_frontier_batch",
                estimated_outbound_cost=math.dist(
                    self.rover_position, tasks[0].preferred_entry,
                ),
                estimated_round_trip_cost=plan.combined_cost,
                estimated_separate_cost=plan.separate_cost,
                estimated_combined_cost=plan.combined_cost,
                estimated_avoided_round_trip=plan.avoided_round_trip,
                estimated_detour_cost=plan.detour_cost,
                issued_work_unit_ids=tuple(sorted(self.registry.work_units)),
                maximum_total_components=(
                    self.focused_frontier_batch_maximum_total_components
                ),
                maximum_detour_distance=self.focused_frontier_batch_maximum_detour,
                minimum_avoided_round_trip=(
                    self.focused_frontier_batch_minimum_avoided_round_trip
                ),
                maximum_service_seconds=(
                    self.focused_frontier_batch_maximum_service_seconds
                ),
                maximum_total_dfs_nodes=(
                    self.focused_frontier_batch_maximum_total_dfs_nodes
                ),
                maximum_consecutive_low_gain_scans=(
                    self.focused_frontier_batch_maximum_consecutive_low_gain_scans
                ),
                low_gain_maximum_new_cells=(
                    self.focused_frontier_batch_low_gain_maximum_new_cells
                ),
                low_gain_maximum_confidence_gain=(
                    self.focused_frontier_batch_low_gain_maximum_confidence_gain
                ),
                reason="focused_frontier_batch",
            ),
        )
        return True

    def _build_spatial_lease(
        self,
        drone_id: int,
        directive_id: int,
        task_ids: tuple[int, ...],
        rover_slam: SlamSnapshot,
        *,
        ordered_task_ids: tuple[int, ...] | None = None,
        connector_paths: tuple[tuple[Position, ...], ...] | None = None,
        extra_reserved: Iterable[Position] = (),
        extra_reserved_leases: Iterable[SpatialLease] = (),
        deadline: float | None = None,
    ) -> SpatialLease | None:
        """Build a deterministic non-overlapping rover-SLAM envelope.

        Raster work is restricted to the required cells' radius-padded region.
        A precise Euclidean distance transform applies the configured margin
        without a large-kernel dilation across the entire cave raster during
        a bounded planning pass.
        """
        def planning_budget_exhausted() -> bool:
            return bool(
                deadline is not None and time.perf_counter() >= deadline
            )

        if planning_budget_exhausted():
            return None
        occupancy = np.asarray(rover_slam.occupancy)
        height, width = occupancy.shape
        offset_x, offset_y = (int(value) for value in rover_slam.origin)
        tasks = tuple(self._tasks[task_id] for task_id in task_ids)
        member_work_unit_ids = {
            unit_id for task in tasks for unit_id in task.work_unit_ids
        }
        member_cells: set[Position] = set()
        entries: dict[int, Position] = {}
        for task in tasks:
            if planning_budget_exhausted():
                return None
            component = self.registry.components.get(task.component_id)
            cells = () if component is None else component.geometry.cells
            member_cells.update(cells)
            entries[task.task_id] = task.preferred_entry
        if ordered_task_ids is None:
            ordered_task_ids = task_ids
        if connector_paths is None:
            reachable = self._known_free_reachable_mask(rover_slam)
            connector_paths = tuple(
                self._direct_reachable_path(
                    entries[first_id],
                    entries[second_id],
                    reachable,
                    rover_slam.origin,
                    deadline=deadline,
                )
                for first_id, second_id in zip(
                    ordered_task_ids,
                    ordered_task_ids[1:],
                )
            )
        if (
            frozenset(ordered_task_ids) != frozenset(task_ids)
            or len(connector_paths) != max(0, len(ordered_task_ids) - 1)
        ):
            return None
        if planning_budget_exhausted():
            return None

        # The lease authorizes service around member geometry and the
        # rover-known connectors between members.  The shared rover-to-first
        # transit leg remains ordinary transit, otherwise every concurrent
        # lease would overlap at the physical rendezvous point.
        required_cells = set(member_cells)
        for connector in connector_paths:
            if planning_budget_exhausted():
                return None
            if not connector:
                return None
            required_cells.update(connector)
        if not required_cells:
            return None
        radius = max(1, int(math.ceil(self.focused_frontier_batch_lease_margin)))
        local_required = tuple(
            (int(x) - offset_x, int(y) - offset_y)
            for x, y in required_cells
        )
        if any(
            not (0 <= local_x < width and 0 <= local_y < height)
            for local_x, local_y in local_required
        ):
            return None
        roi_left = max(0, min(x for x, _y in local_required) - radius)
        roi_right = min(
            width - 1,
            max(x for x, _y in local_required) + radius,
        )
        roi_top = max(0, min(y for _x, y in local_required) - radius)
        roi_bottom = min(
            height - 1,
            max(y for _x, y in local_required) + radius,
        )
        mask = np.zeros(
            (roi_bottom - roi_top + 1, roi_right - roi_left + 1),
            dtype=np.uint8,
        )
        for local_x, local_y in local_required:
            mask[local_y - roi_top, local_x - roi_left] = 1
        if planning_budget_exhausted():
            return None
        distance_to_required = cv2.distanceTransform(
            1 - mask,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        if planning_budget_exhausted():
            return None
        mask = np.less_equal(
            distance_to_required,
            float(radius),
        ).astype(np.uint8)
        if planning_budget_exhausted():
            return None

        protection_radius = max(
            int(self.registry.footprint.frontier_stride * 2),
            int(math.ceil(radius * 0.25)),
        )
        for unit_id, unit in self.registry.work_units.items():
            if planning_budget_exhausted():
                return None
            if (
                unit_id in member_work_unit_ids
                or unit.state not in {
                    WorkUnitState.READY,
                    WorkUnitState.CLAIMED,
                    WorkUnitState.ACTIVE,
                    WorkUnitState.BLOCKED,
                }
            ):
                continue
            center_x = unit.anchor_position[0] - offset_x - roi_left
            center_y = unit.anchor_position[1] - offset_y - roi_top
            if (
                center_x + protection_radius < 0
                or center_y + protection_radius < 0
                or center_x - protection_radius >= mask.shape[1]
                or center_y - protection_radius >= mask.shape[0]
            ):
                continue
            cv2.circle(
                mask,
                (center_x, center_y),
                protection_radius,
                0,
                -1,
            )

        for active in (
            *self._leases_by_id.values(),
            *tuple(extra_reserved_leases),
        ):
            for y, left, right in active.cell_spans:
                if planning_budget_exhausted():
                    return None
                row = y - offset_y - roi_top
                if not 0 <= row < mask.shape[0]:
                    continue
                clipped_left = max(0, left - offset_x - roi_left)
                clipped_right = min(
                    mask.shape[1] - 1,
                    right - offset_x - roi_left,
                )
                if clipped_left <= clipped_right:
                    mask[row, clipped_left:clipped_right + 1] = 0
        for x, y in extra_reserved:
            if planning_budget_exhausted():
                return None
            local_x = x - offset_x - roi_left
            local_y = y - offset_y - roi_top
            if 0 <= local_x < mask.shape[1] and 0 <= local_y < mask.shape[0]:
                mask[local_y, local_x] = 0
        if any(
            not mask[local_y - roi_top, local_x - roi_left]
            for local_x, local_y in local_required
        ):
            return None
        spans: list[tuple[int, int, int]] = []
        for mask_y in range(mask.shape[0]):
            if planning_budget_exhausted():
                return None
            xs = np.flatnonzero(mask[mask_y])
            if not len(xs):
                continue
            start = previous = int(xs[0])
            for value in xs[1:]:
                current = int(value)
                if current != previous + 1:
                    spans.append((
                        mask_y + roi_top + offset_y,
                        start + roi_left + offset_x,
                        previous + roi_left + offset_x,
                    ))
                    start = current
                previous = current
            spans.append((
                mask_y + roi_top + offset_y,
                start + roi_left + offset_x,
                previous + roi_left + offset_x,
            ))
        if not spans:
            return None
        lease_id = -1
        if directive_id >= 0:
            lease_id = self._next_lease_id
            self._next_lease_id += 1
        return SpatialLease(
            lease_id=lease_id,
            owner_drone_id=int(drone_id),
            directive_id=int(directive_id),
            issued_revision=self.registry.revision,
            cell_spans=tuple(spans),
            member_task_ids=tuple(task_ids),
        )

    @staticmethod
    def _lease_cells(lease: SpatialLease) -> set[Position]:
        return {
            (x, y)
            for y, left, right in lease.cell_spans
            for x in range(left, right + 1)
        }

    @staticmethod
    def _direct_reachable_path(
        start: Position,
        goal: Position,
        reachable: np.ndarray,
        origin: Position,
        *,
        deadline: float | None = None,
    ) -> tuple[Position, ...]:
        """Return a straight rover-known connector without graph search.

        This is deliberately only a conservative fallback for tests and for a
        highway graph that has not completed its first build.  It never tries
        to route around an obstacle, keeping endgame planning strictly bounded.
        """
        x0, y0 = (int(value) for value in start)
        x1, y1 = (int(value) for value in goal)
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1
        error = dx + dy
        points: list[Position] = []
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                return ()
            points.append((x0, y0))
            if (x0, y0) == (x1, y1):
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += step_x
            if doubled <= dx:
                error += dx
                y0 += step_y
        path = tuple(points)
        return path if FrontierTaskCoordinator._path_is_reachable(
            path, reachable, origin, deadline=deadline,
        ) else ()

    @staticmethod
    def _path_is_reachable(
        path: Iterable[Position],
        reachable: np.ndarray,
        origin: Position,
        *,
        deadline: float | None = None,
    ) -> bool:
        points = tuple((int(x), int(y)) for x, y in path)
        if not points:
            return False
        offset_x, offset_y = (int(value) for value in origin)
        height, width = reachable.shape

        def free(point: Position) -> bool:
            local_x = point[0] - offset_x
            local_y = point[1] - offset_y
            return bool(
                0 <= local_x < width
                and 0 <= local_y < height
                and reachable[local_y, local_x]
            )

        for index, point in enumerate(points):
            if (
                index % 64 == 0
                and deadline is not None
                and time.perf_counter() >= deadline
            ):
                return False
            if not free(point):
                return False
        for index, (previous, current) in enumerate(
            zip(points, points[1:]),
        ):
            if (
                index % 64 == 0
                and deadline is not None
                and time.perf_counter() >= deadline
            ):
                return False
            delta_x = current[0] - previous[0]
            delta_y = current[1] - previous[1]
            if max(abs(delta_x), abs(delta_y)) != 1:
                return False
            if delta_x and delta_y and not (
                free((previous[0] + delta_x, previous[1]))
                and free((previous[0], previous[1] + delta_y))
            ):
                return False
        return True

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
        focused_endgame: bool = False,
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
        ) -> tuple[
            int,
            int,
            int,
            float,
            float,
            tuple[tuple[int, int], ...],
        ]:
            if drone_index >= len(drones):
                return (0, 0, 0, 0.0, 0.0, ())
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
                    tail[3] + quote.route_cost * 2.0,
                    max(tail[4], quote.route_cost * 2.0),
                    ((drone_id, task_id), *tail[5]),
                ))
            if focused_endgame:
                return min(
                    candidates,
                    key=lambda item: (
                        -item[0],
                        item[4],
                        item[3],
                        -item[2],
                        -item[1],
                        item[5],
                    ),
                )
            return min(
                candidates,
                key=lambda item: (
                    -item[0],
                    -item[1],
                    -item[2],
                    item[3],
                    item[5],
                ),
            )

        _count, _depth, _affinity, _cost, _maximum, pairs = solve(0, ())
        selected = tuple(
            (drone_id, task_id, quotes[(drone_id, task_id)].route)
            for drone_id, task_id in pairs
        )
        return tuple(quotes.values()), selected

    def _focused_endgame_is_active(self) -> bool:
        """Use registry topology, never display-only floor coverage."""
        actionable = tuple(
            component
            for component in self.registry.components.values()
            if component.state == ComponentState.ACTIVE
            and any(
                self.registry.work_units[unit_id].state in {
                    WorkUnitState.READY,
                    WorkUnitState.CLAIMED,
                    WorkUnitState.ACTIVE,
                    WorkUnitState.BLOCKED,
                }
                for unit_id in component.work_unit_ids
            )
        )
        return bool(actionable) and all(
            component.exploration_mode == ExplorationMode.FOCUSED
            for component in actionable
        )

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
