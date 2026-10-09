"""Random drone exploration with A* escape and homing routes."""

from __future__ import annotations

import itertools
import logging
import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Tuple

import numpy as np
from navigation.local_slam_routes import LocalSlamRoutePlanner, bounded_local_planning

from agents.component_explorer import (
    FrontierObservationPose,
    IncidentalPocketSignature,
    IncidentalScanCandidate,
    LocalComponentNode,
    LocalDFSStack,
    ScanPlanProgress,
    incidental_scan_candidate,
    leased_local_frontiers,
    local_node_pose,
    observation_pose_on_path,
    related_local_successors,
)
from asset_config.helpers import next_cell_coords
from contracts import DroneMovementDependencies
from mapping.frontier_registry import CausalTransition, WorkUnitKind
from mapping.frontiers import (
    eight_connected_components,
    eight_neighbor_adjacency,
    known_free_frontier_mask,
)
from mapping.exploration_sectors import (
    SectorAssignment,
    SectorCheckInResult,
    SectorOutcomeReport,
    SectorSuppressionOutcome,
)
from mapping.ray_geometry import bresenham_line_points
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN
from mission.energy import EnergyReturnDecision, EnergyState
from mission.exploration_coordination import (
    BatchMember,
    BatchMemberReport,
    CoordinationReport,
    CoordinationResult,
    DFSFrame,
    DirectiveKind,
    ExplorationDirective,
    ProvisionalFrontierObservation,
    SpatialLease,
    TaskSuspension,
    WorkUnitOutcome,
)
from navigation.astar_pathfinder import (
    PATH_COMPLETE,
    PATH_PARTIAL_LIMIT,
    PATH_UNREACHABLE,
    PathResult,
)
from navigation.highway import HighwayGraphSnapshot


Position = Tuple[int, int]
CoverageCell = tuple[int, int]
CoverageEdge = tuple[CoverageCell, CoverageCell]
logger = logging.getLogger(__name__)
_GLOBAL_LAUNCH_SECTOR_TIE_RATIO = 0.25
_GLOBAL_TARGET_SWITCH_SCORE_MARGIN = 0.5
_PENDING_FRONTIER_SCAN_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True)
class _BatchMemberSnapshot:
    member: BatchMember
    disposition: str
    work_unit_outcomes: tuple[WorkUnitOutcome, ...]
    causal_successors: tuple[frozenset[Position], ...]
    visited_successor_anchors: tuple[Position, ...]
    suspension: TaskSuspension | None
    outbound_actual_path: tuple[Position, ...]
    outbound_distance: float
    service_distance: float
    service_seconds: float


@dataclass
class _CoordinationExecution:
    """Mutable execution cursor for one rover-issued directive."""

    directive: ExplorationDirective
    phase: str
    path_start_index: int
    scan: ScanPlanProgress | None = None
    dfs: LocalDFSStack | None = None
    root_component_id: int | None = None
    root_work_unit_id: int | None = None
    root_source_cells: frozenset[Position] = frozenset()
    root_scanned: bool = False
    work_unit_outcomes: list[WorkUnitOutcome] | None = None
    causal_successors: list[frozenset[Position]] | None = None
    outbound_actual_path: tuple[Position, ...] = ()
    return_actual_path: tuple[Position, ...] = ()
    return_path_source: str = "none"
    root_outbound_distance: float = 0.0
    service_distance: float = 0.0
    reposition_target: Position | None = None
    reposition_fallback_target: Position | None = None
    route_attempts: int = 0
    completed_scan_headings: list[int] | None = None
    timed_out_scan_headings: list[int] | None = None
    sensor_newly_known_cells: int = 0
    sensor_confidence_gain: float = 0.0
    local_routes: dict[int, tuple[Position, ...]] | None = None
    transit_node_id: int | None = None
    transit_path_start_index: int | None = None
    observation_node_id: int | None = None
    observation_position: Position | None = None
    observation_heading: int | None = None
    return_path_start_index: int | None = None
    pending_return_breadcrumb: tuple[Position, ...] = ()
    suspension_reason: str | None = None
    suspension_position: Position | None = None
    suspension_energy_state: EnergyState | None = None
    visited_successor_anchors: list[Position] | None = None
    pending_authoritative_nodes: list[LocalComponentNode] | None = None
    energy_checkpoint_key: tuple[Any, ...] | None = None
    reposition_attempts: int = 0
    reserved_branch_count: int = 0
    thin_successor_count: int = 0
    follow_started_at: float | None = None
    follow_last_leader_seen_at: float | None = None
    follow_last_leader_progress_at: float | None = None
    follow_last_leader_position: Position | None = None
    incidental_distance_since_sample: float = 0.0
    incidental_distance_since_selection: float = math.inf
    incidental_candidates: int = 0
    incidental_attempts: int = 0
    incidental_completions: int = 0
    incidental_timeouts: int = 0
    incidental_wait_seconds: float = 0.0
    incidental_requested_rotation: float = 0.0
    incidental_predicted_support: int = 0
    incidental_pocket_cells_closed: int = 0
    batch_pending_members: list[BatchMember] | None = None
    batch_current_member: BatchMember | None = None
    batch_completed_members: list[_BatchMemberSnapshot] | None = None
    batch_provisional_observations: list[
        ProvisionalFrontierObservation
    ] | None = None
    batch_started_at: float | None = None
    batch_completed_dfs_nodes: int = 0
    batch_current_dfs_nodes: int = 0
    batch_provisional_component_count: int = 0
    batch_detour_distance: float = 0.0
    batch_consecutive_low_gain_scans: int = 0
    batch_member_path_start_index: int | None = None
    batch_member_started_at: float | None = None
    batch_return_reason: str | None = None


@dataclass
class _IncidentalPocketAttempt:
    """Drone-local duplicate memory retained across rover directives."""

    signature: IncidentalPocketSignature
    cells: frozenset[Position]
    gateway: Position
    attempted_at: float
    revisited: bool = False


@dataclass
class _FocusedFrontierBatchProvisionalAttempt:
    """Local service memory used only to measure a later revisit."""

    directive_id: int
    observation_id: int
    cells: frozenset[Position]
    serviced_at: float
    revisited: bool = False


@dataclass(frozen=True)
class _PendingIncidentalTransitScan:
    """A side scan layered over an eligible retained transit route."""

    directive_id: int
    original_phase: str
    position: Position
    heading: int
    resume_heading: float
    candidate: IncidentalScanCandidate
    minimum_scan_sequence: int
    requested_at: float
    deadline: float
    retained_route: tuple[Position, ...]
    route_target: Position
    route_source: str
    path_status: str | None
    baseline_slam_version: int


@dataclass(frozen=True)
class _PendingFrontierScan:
    """One wall-facing sensor pose that temporarily blocks translation."""

    position: Position
    heading: int
    resume_heading: float
    frontier_target: Position
    unknown_target: Position
    reason: str
    minimum_scan_sequence: int
    baseline_geometry: tuple[Position, ...] | None
    requested_at: float
    deadline: float


@dataclass(frozen=True)
class _PendingFrontierRoute:
    """A capped A* frontier route waiting for its next segment."""

    target: Position
    recovery_reason: str


@dataclass(frozen=True)
class DroneActivitySnapshot:
    """Detached exploration phase exposed to diagnostics and UI."""

    state: str
    detail: str
    target: Position | None = None
    directive_id: int | None = None
    directive_kind: str | None = None
    task_id: int | None = None
    component_id: int | None = None
    work_unit_id: int | None = None
    dfs_depth: int = 0
    peer_id: int | None = None
    rover_id: int | None = None


@dataclass(frozen=True)
class _CoverageRecord:
    """Exponentially decaying traversal pressure for one cell or edge."""

    value: float
    updated_at: float


@dataclass(frozen=True)
class _FrontierCluster:
    """One connected local unknown boundary used for heading selection."""

    cells: tuple[Position, ...]
    size: int
    touches_wall: bool
    distance: float
    continuation_alignment: float


@dataclass(frozen=True)
class _ClusterSelection:
    """Selected component and normalized terms used to rank it."""

    cluster: _FrontierCluster | None
    support: dict[int, float]
    score: float
    size_rank: float
    proximity: float
    wall_candidate_count: int
    generic_candidate_count: int


@dataclass(frozen=True)
class _GlobalFrontierTile:
    """Frontier evidence accumulated inside one coarse SLAM cell."""

    target: Position
    size: int
    wall_cells: int

    @property
    def touches_wall(self) -> bool:
        """Return whether this cell contains a wall continuation."""
        return self.wall_cells > 0


@dataclass(frozen=True)
class _GlobalFrontierRegion:
    """One connected set of occupied coarse frontier cells."""

    tiles: tuple[_GlobalFrontierTile, ...]
    size: int
    wall_cells: int
    tile_count: int

    @property
    def touches_wall(self) -> bool:
        """Return whether any unknown boundary cell continues a known wall."""
        return self.wall_cells > 0


@dataclass(frozen=True)
class _GlobalFrontierSelection:
    """One globally scored region tile and its diagnostic terms."""

    region: _GlobalFrontierRegion | None
    position: Position | None
    score: float
    size_rank: float
    proximity: float
    eligible_region_count: int
    wall_candidate_count: int
    generic_candidate_count: int
    requester_distance: float | None
    nearest_peer_distance: float | None
    ownership_margin: float
    launch_sector_alignment: float
    ownership_contribution: float
    retained_previous: bool = False
    previous_region_overlap: float = 0.0


@dataclass(frozen=True)
class _GlobalFrontierCache:
    """Coarse whole-map regions and one stable strategic selection."""

    regions: tuple[_GlobalFrontierRegion, ...]
    target: _GlobalFrontierRegion | None
    target_position: Position | None
    target_score: float
    target_size_rank: float
    target_proximity: float
    eligible_region_count: int
    filtered_region_count: int
    wall_candidate_count: int
    generic_candidate_count: int
    requester_distance: float | None
    nearest_peer_distance: float | None
    ownership_margin: float
    launch_sector_alignment: float
    ownership_contribution: float
    target_retained: bool
    target_region_overlap: float
    slam_version: int
    built_at: float


@dataclass(frozen=True)
class _GlobalFrontierGuidance:
    """Directional evidence from the cached whole-map target."""

    support: dict[int, float]
    active: bool
    target: Position | None
    size: int
    distance: float | None
    bearing: float | None
    touches_wall: bool
    score: float
    size_rank: float
    proximity: float
    region_count: int
    eligible_region_count: int
    filtered_region_count: int
    wall_candidate_count: int
    generic_candidate_count: int
    requester_distance: float | None
    nearest_peer_distance: float | None
    ownership_margin: float
    launch_sector_alignment: float
    ownership_contribution: float
    slam_version: int


@dataclass(frozen=True)
class _HeadingBias:
    """Weighted-random evidence attached to valid movement headings."""

    weights: dict[int, float]
    wall_support: dict[int, float]
    frontier_support: dict[int, float]
    global_support: dict[int, float]
    separation_support: dict[int, float]
    coverage_cells: dict[int, CoverageCell]
    coverage_visit_pressure: dict[int, float]
    coverage_edge_pressure: dict[int, float]
    coverage_penalty_factor: dict[int, float]
    mode: str
    peer_count: int
    cluster_count: int
    eligible_cluster_count: int
    filtered_cluster_count: int
    selected_cluster_size: int
    selected_cluster_distance: float | None
    selected_cluster_touches_wall: bool
    selected_continuation_alignment: float
    selected_cluster_score: float
    selected_cluster_size_rank: float
    selected_cluster_proximity: float
    wall_candidate_count: int
    generic_candidate_count: int
    global_active: bool
    global_target: Position | None
    global_region_size: int
    global_region_distance: float | None
    global_region_bearing: float | None
    global_region_touches_wall: bool
    global_region_score: float
    global_region_size_rank: float
    global_region_proximity: float
    global_region_count: int
    global_eligible_region_count: int
    global_filtered_region_count: int
    global_wall_candidate_count: int
    global_generic_candidate_count: int
    global_requester_distance: float | None
    global_nearest_peer_distance: float | None
    global_ownership_margin: float
    global_launch_sector_alignment: float
    global_ownership_contribution: float
    global_slam_version: int


class DroneMovementController:
    """Explore locally and use A* for frontier recovery and homing."""

    _ACTIVITY_HOLD_SECONDS = 0.25
    _SHARING_SECONDS = 0.75

    def __init__(
        self,
        drone: Any,
        dependencies: DroneMovementDependencies,
    ) -> None:
        self.drone = drone
        self.dependencies = dependencies
        frontier = drone.settings.frontier
        self.border_retry_cooldown = 1.5
        self.border_retry_until: dict[Position, float] = {}
        self.frontier_stride = max(1, int(getattr(frontier, "stride", 4)))
        self.frontier_confidence_threshold = float(
            frontier.confidence_threshold
        )
        self.minimum_frontier_cluster_cells = max(
            1,
            int(getattr(frontier, "minimum_cluster_cells", 12)),
        )
        self.frontier_distance_band = max(
            1.0,
            float(getattr(frontier, "distance_band", 16.0)),
        )
        self.wall_continuation_weight = max(
            0.0,
            float(getattr(frontier, "wall_continuation_weight", 2.0)),
        )
        self.frontier_cluster_size_weight = max(
            0.0,
            float(getattr(frontier, "cluster_size_weight", 2.0)),
        )
        self.frontier_cluster_proximity_weight = max(
            0.0,
            float(getattr(frontier, "cluster_proximity_weight", 1.0)),
        )
        self.global_frontier_cell_size = max(
            1,
            int(getattr(frontier, "global_cell_size", 32)),
        )
        self.global_frontier_refresh_interval = max(
            0.0,
            float(getattr(frontier, "global_refresh_interval", 2.0)),
        )
        self.global_frontier_ownership_weight = max(
            0.0,
            float(getattr(frontier, "global_ownership_weight", 2.0)),
        )
        self.maximum_frontier_path_circuity = max(
            1.0,
            float(getattr(frontier, "maximum_path_circuity", 4.0)),
        )
        self._last_raw_frontiers: frozenset[Position] = frozenset()
        self._last_global_raw_frontiers: frozenset[Position] = frozenset()
        self._last_frontier_mask: np.ndarray | None = None
        self._frontier_component_targets: dict[
            Position,
            tuple[Position, ...],
        ] = {}
        self._suppressed_frontier_geometry: dict[
            Position,
            tuple[Position, ...] | None,
        ] = {}
        exploration = drone.settings.exploration
        self.stagnation_distance = float(
            exploration.stagnation_distance
        )
        self.stagnation_min_sensor_cells_per_px = float(
            exploration.stagnation_min_sensor_cells_per_px
        )
        self.wall_direction_bias = float(
            exploration.wall_direction_bias
        )
        self.unexplored_direction_bias = float(
            exploration.unexplored_direction_bias
        )
        self.separation_direction_bias = float(
            exploration.separation_direction_bias
        )
        self.coverage_memory_cell_size = max(
            1,
            int(exploration.coverage_memory_cell_size),
        )
        self.coverage_memory_decay_seconds = max(
            1e-9,
            float(exploration.coverage_memory_decay_seconds),
        )
        self.coverage_visit_weight = max(
            0.0,
            float(exploration.coverage_visit_weight),
        )
        self.coverage_edge_weight = max(
            0.0,
            float(exploration.coverage_edge_weight),
        )
        incidental = drone.settings.incidental_scan
        self.incidental_scan_mode = str(incidental.mode).casefold()
        self.incidental_maximum_attempts = max(
            0, int(incidental.maximum_attempts_per_directive),
        )
        self.incidental_maximum_wait = max(
            0.0, float(incidental.maximum_wait_seconds_per_directive),
        )
        self.incidental_attempt_timeout = max(
            0.0, float(incidental.attempt_timeout_seconds),
        )
        self.incidental_maximum_rotation = max(
            0.0,
            float(incidental.maximum_rotation_degrees_per_directive),
        )
        self.incidental_cooldown_ranges = max(
            0.0, float(incidental.distance_cooldown_sensor_ranges),
        )
        self.incidental_sample_spacing_ranges = max(
            0.0, float(incidental.sample_spacing_sensor_ranges),
        )
        highway = drone.settings.highway
        self.highway_mode = str(highway.mode).casefold()
        self.highway_maximum_query_ms = max(
            0.0, float(highway.maximum_query_ms),
        )
        self.highway_maximum_connector_expansions = max(
            1, int(highway.maximum_connector_expansions),
        )
        self.highway_minimum_route_sensor_ranges = max(
            0.0, float(highway.minimum_route_sensor_ranges),
        )
        self.highway_maximum_route_circuity = max(
            1.0, float(highway.maximum_route_circuity),
        )
        self._highway_snapshot: HighwayGraphSnapshot | None = None
        self._last_complete_path: tuple[Position, ...] = ()
        self._return_route_source = "highway"
        self._local_slam_planner = LocalSlamRoutePlanner(
            drone.slam_map, self.frontier_confidence_threshold, trace=self._trace,
        )
        self._last_local_route_status = "unavailable"
        coverage_started_at = self._simulation_time()
        initial_coverage_cell = self._coverage_cell(
            drone.snapshot().position
        )
        self._coverage_cell_visits: dict[
            CoverageCell,
            _CoverageRecord,
        ] = {
            initial_coverage_cell: _CoverageRecord(
                value=1.0,
                updated_at=coverage_started_at,
            )
        }
        self._coverage_edge_visits: dict[
            CoverageEdge,
            _CoverageRecord,
        ] = {}
        self._coverage_last_cell = initial_coverage_cell
        progress = drone.slam_map.progress_snapshot()
        self._stagnation_sensor_baseline = (
            progress.sensor_newly_known_cells
        )
        self._stagnation_distance_travelled = 0.0
        self._pending_frontier_scan: _PendingFrontierScan | None = None
        self._pending_incidental_scan: (
            _PendingIncidentalTransitScan | None
        ) = None
        self._incidental_attempt_memory: list[_IncidentalPocketAttempt] = []
        self._focused_frontier_batch_provisional_memory: list[
            _FocusedFrontierBatchProvisionalAttempt
        ] = []
        self._pending_frontier_route: _PendingFrontierRoute | None = None
        self._partial_route_endpoints: dict[
            Position,
            tuple[Position, ...],
        ] = {}
        self._global_frontier_cache: _GlobalFrontierCache | None = None
        self._shared_slam_changed = threading.Event()
        self._frontier_slam_version = -1
        self._sector_assignment: SectorAssignment | None = None
        self._sector_assignment_entered = False
        self._sector_ingress_failures = 0
        self._sector_check_in_required = self._sector_policy_enabled()
        self._completed_sector_id: int | None = None
        self._sector_path_start_index = 0
        self._sector_empty_confirmations = 0
        self._sector_waiting_generation: int | None = None
        self._sector_wait_started_at: float | None = None
        self._sector_wait_reason = "barrier"
        self._sector_assignment_ready: threading.Event | None = None
        self._sector_suppression_reasons: dict[int, set[str]] = {}
        self._sector_suppression_targets: dict[int, set[Position]] = {}
        self._coordination_ready: threading.Event | None = None
        self._coordination_execution: _CoordinationExecution | None = None
        self._coordination_report: CoordinationReport | None = None
        self._coordination_wait_started_at: float | None = None
        self._coordination_report_counter = 0
        self._activity_lock = threading.Lock()
        self._calculation_activity: DroneActivitySnapshot | None = None
        self._held_activity: DroneActivitySnapshot | None = None
        self._held_activity_until = 0.0
        self._sharing_activity: DroneActivitySnapshot | None = None
        self._sharing_until = 0.0

    def mark_shared_slam_changed(self) -> None:
        """Request a frontier refresh on the owning movement thread."""
        self._shared_slam_changed.set()

    def is_docked(self) -> bool:
        """Return the mission controller's physical dock state for this drone."""
        callback = self.dependencies.is_exploration_docked
        return bool(callable(callback) and callback(self.drone.id))

    def begin_peer_sharing(
        self,
        peer_id: int,
        peer_position: Position,
    ) -> None:
        """Pause translation and expose one real peer exchange to the UI."""
        self._begin_sharing(DroneActivitySnapshot(
            "Sharing",
            f"with drone {int(peer_id) + 1}",
            target=(int(peer_position[0]), int(peer_position[1])),
            peer_id=int(peer_id),
        ))

    def begin_rover_sharing(
        self,
        rover_id: int,
        rover_position: Position,
    ) -> None:
        """Pause translation and expose one real rover exchange to the UI."""
        self._begin_sharing(DroneActivitySnapshot(
            "Sharing",
            f"with rover {int(rover_id) + 1}",
            target=(int(rover_position[0]), int(rover_position[1])),
            rover_id=int(rover_id),
        ))

    def _begin_sharing(self, activity: DroneActivitySnapshot) -> None:
        """Give overlapping exchanges one bounded, non-extending pause."""
        now = self._simulation_time()
        with self._activity_lock:
            self._sharing_activity = activity
            started = now >= self._sharing_until
            if started:
                self._sharing_until = now + self._SHARING_SECONDS
        if started:
            self._trace(
                "drone_sharing_pause_started",
                duration_seconds=self._SHARING_SECONDS,
                peer_id=activity.peer_id,
                rover_id=activity.rover_id,
            )

    def _active_transient_activity(self) -> DroneActivitySnapshot | None:
        """Return a sharing/calculation state while its display window lasts."""
        now = self._simulation_time()
        with self._activity_lock:
            if (
                self._sharing_activity is not None
                and now < self._sharing_until
            ):
                return self._sharing_activity
            if self._calculation_activity is not None:
                return self._calculation_activity
            if (
                self._held_activity is not None
                and now < self._held_activity_until
            ):
                return self._held_activity
        return None

    def _sharing_remaining(self) -> float:
        """Return simulation seconds left in the current sharing dwell."""
        now = self._simulation_time()
        with self._activity_lock:
            if self._sharing_activity is None:
                return 0.0
            return max(0.0, self._sharing_until - now)

    def _begin_calculation(
        self,
        state: str,
        detail: str,
        target: Position | None = None,
    ) -> None:
        """Publish a calculation state while work runs on this worker."""
        with self._activity_lock:
            self._calculation_activity = DroneActivitySnapshot(
                state,
                detail,
                target=target,
            )

    def _end_calculation(self, state: str) -> None:
        """Keep a completed calculation briefly legible in the UI."""
        deadline = self._simulation_time() + self._ACTIVITY_HOLD_SECONDS
        with self._activity_lock:
            activity = self._calculation_activity
            if activity is None or activity.state != state:
                return
            self._calculation_activity = None
            self._held_activity = activity
            self._held_activity_until = deadline

    def activity_snapshot(self) -> DroneActivitySnapshot:
        """Describe the current policy phase without exposing mutable cursors."""
        runtime = self.drone.snapshot()
        if runtime.done:
            return DroneActivitySnapshot("Done", "mission complete")
        transient = self._active_transient_activity()
        if transient is not None:
            return transient
        if runtime.returning_home:
            return DroneActivitySnapshot(
                "Homing",
                f"home {self.drone.start_pos[0]},{self.drone.start_pos[1]}",
                target=tuple(self.drone.start_pos),
            )

        if self.is_docked():
            report = self._coordination_report
            if report is not None:
                return DroneActivitySnapshot(
                    "Docked",
                    f"at rover; report {report.report_id} pending",
                    directive_id=report.directive_id,
                    directive_kind=report.kind.value,
                )
            return DroneActivitySnapshot(
                "Docked",
                "with rover; coordinator reply pending",
            )

        pending_incidental = self._pending_incidental_scan
        if pending_incidental is not None:
            return DroneActivitySnapshot(
                "Incidental scan",
                (
                    f"heading {pending_incidental.heading} pocket "
                    f"{pending_incidental.candidate.gateway[0]},"
                    f"{pending_incidental.candidate.gateway[1]}"
                ),
                target=pending_incidental.candidate.gateway,
                directive_id=pending_incidental.directive_id,
            )

        execution = self._coordination_execution
        if execution is not None:
            directive = execution.directive
            kind = directive.kind.value
            task = directive.task
            dfs = execution.dfs
            frame = None if dfs is None else dfs.current
            work_unit_id = (
                None if frame is None else frame.node.work_unit_id
            )
            target: Position | None = None
            if execution.phase == "repositioning":
                target = execution.reposition_target
            elif execution.phase == "returning":
                target = self._rendezvous_position()
            elif frame is not None:
                target = frame.node.anchor_position
            elif directive.probe_target is not None:
                target = directive.probe_target
            elif execution.scan is not None:
                target = execution.scan.position

            states = {
                (DirectiveKind.COMPONENT_TASK, "transit"): "Task transit",
                (DirectiveKind.COMPONENT_TASK, "scanning"): "Frontier scan",
                (DirectiveKind.COMPONENT_TASK, "repositioning"): "DFS reposition",
                (DirectiveKind.COMPONENT_TASK, "returning"): "Rover return",
                (DirectiveKind.COMPONENT_BATCH, "transit"): "Batch transit",
                (DirectiveKind.COMPONENT_BATCH, "scanning"): "Batch scan",
                (DirectiveKind.COMPONENT_BATCH, "repositioning"): "Batch reposition",
                (DirectiveKind.COMPONENT_BATCH, "returning"): "Batch return",
                (DirectiveKind.COMPONENT_FOLLOW, "following"): "Branch follow",
                (DirectiveKind.COMPONENT_FOLLOW, "transit"): "Branch transit",
                (DirectiveKind.COMPONENT_FOLLOW, "scanning"): "Branch scan",
                (DirectiveKind.COMPONENT_FOLLOW, "repositioning"): "DFS reposition",
                (DirectiveKind.COMPONENT_FOLLOW, "returning"): "Rover return",
                (DirectiveKind.RADIAL_PROBE, "transit"): "Probe transit",
                (DirectiveKind.RADIAL_PROBE, "scanning"): "Probe scan",
                (DirectiveKind.RADIAL_PROBE, "returning"): "Probe return",
                (DirectiveKind.ROVER_SCAN, "scanning"): "Rover scan",
            }
            state = states.get(
                (directive.kind, execution.phase),
                execution.phase.replace("_", " ").title(),
            )
            detail_parts = [f"directive {directive.directive_id}"]
            if task is not None:
                detail_parts.extend((
                    f"task {task.task_id}",
                    f"component {task.component_id}",
                ))
            if work_unit_id is not None:
                detail_parts.append(f"unit {work_unit_id}")
            if dfs is not None:
                detail_parts.append(f"depth {len(dfs.frames)}")
            if execution.scan is not None:
                scan = execution.scan
                detail_parts.append(
                    f"scan {scan.heading_index + 1}/{len(scan.headings)}"
                )
            if target is not None:
                detail_parts.append(f"target {target[0]},{target[1]}")
            if execution.phase == "repositioning":
                detail_parts.append(
                    f"retry {execution.reposition_attempts}"
                )
            return DroneActivitySnapshot(
                state=state,
                detail=" ".join(detail_parts),
                target=target,
                directive_id=directive.directive_id,
                directive_kind=kind,
                task_id=None if task is None else task.task_id,
                component_id=None if task is None else task.component_id,
                work_unit_id=work_unit_id,
                dfs_depth=0 if dfs is None else len(dfs.frames),
            )

        report = self._coordination_report
        if report is not None:
            rover = self._rendezvous_position()
            rendezvous = (
                ""
                if rover is None
                else f" rendezvous {rover[0]},{rover[1]}"
            )
            return DroneActivitySnapshot(
                "Reporting",
                (
                    f"directive {report.directive_id}"
                    f"{rendezvous}"
                ),
                target=rover,
            )
        if self._coordination_ready is not None:
            return DroneActivitySnapshot(
                "Awaiting task",
                "at rover; coordinator reply pending",
            )
        if self._pending_frontier_scan is not None:
            pending = self._pending_frontier_scan
            return DroneActivitySnapshot(
                "Frontier scan",
                f"heading {pending.heading} target "
                f"{pending.frontier_target[0]},{pending.frontier_target[1]}",
                target=pending.frontier_target,
            )
        if self._pending_frontier_route is not None:
            pending = self._pending_frontier_route
            return DroneActivitySnapshot(
                "Task transit",
                f"frontier target {pending.target[0]},{pending.target[1]}",
                target=pending.target,
            )
        if self._component_policy_enabled():
            rover = self._rendezvous_position()
            if rover is not None and runtime.position != rover:
                return DroneActivitySnapshot(
                    "Rover check-in",
                    f"target {rover[0]},{rover[1]}",
                    target=rover,
                )
            return DroneActivitySnapshot("Checking in", "at rover")
        if runtime.explored:
            return DroneActivitySnapshot("Deployed", "local exploration")
        return DroneActivitySnapshot("Ready", "awaiting exploration")

    def _refresh_frontiers_before_mission_state(self) -> bool:
        """Apply shared or late SLAM changes before exhaustion can start home."""
        drone = self.drone
        state = drone.snapshot()
        shared_change = self._shared_slam_changed.is_set()
        if shared_change:
            self._shared_slam_changed.clear()
        if state.done or state.returning_home:
            return False

        slam_version = drone.slam_map.version
        empty_changed_map = (
            (state.explored or self._has_exploration_scope())
            and not state.frontiers
            and slam_version != self._frontier_slam_version
        )
        if not shared_change and not empty_changed_map:
            return False

        self.rebuild_frontiers(
            stride=self.frontier_stride,
            confidence_threshold=self.frontier_confidence_threshold,
        )
        self._trace(
            "drone_slam_frontiers_refreshed",
            reason=(
                "shared_slam_changed"
                if shared_change
                else "empty_frontiers_on_changed_slam"
            ),
            previous_slam_version=slam_version,
            rebuilt_slam_version=self._frontier_slam_version,
            frontier_count=len(drone.snapshot().frontiers),
        )
        return shared_change

    def move(self) -> None:
        """Advance one random step, escape route, or homing route."""
        drone = self.drone
        sharing = self._active_transient_activity()
        if sharing is not None and sharing.state == "Sharing":
            return
        if self._component_policy_enabled():
            self._move_component_policy()
            return
        state = drone.snapshot()
        if (
            self._sector_assignment_ready is not None
            and not state.done and not state.returning_home
        ):
            self._advance_sector_check_in()
            return
        self._refresh_frontiers_before_mission_state()
        if self._pending_frontier_scan is not None:
            self._advance_pending_frontier_scan()
            return
        state = drone.snapshot()
        if state.done:
            return
        if state.returning_home:
            self._pending_frontier_route = None
            if self.reach_start_point():
                drone.runtime_state.mark_done()
            return

        if self._sector_policy_enabled():
            if self._sector_check_in_required:
                self._advance_sector_check_in()
                return
            if self._sector_exhausted():
                self._start_sector_check_in(
                    reason="local_frontiers_exhausted"
                )
                return
        else:
            done, returning_home = drone.runtime_state.evaluate_mission_state()
            if done:
                return
            if returning_home:
                self._pending_frontier_route = None
                if self.reach_start_point():
                    drone.runtime_state.mark_done()
                return

        pending_route = self._pending_frontier_route
        if pending_route is not None:
            self.maybe_rebuild_frontiers()
            target_active = (
                pending_route.target in drone.snapshot().frontiers
                and self._in_exploration_scope(pending_route.target)
            )
            if target_active:
                if self.reach_border(
                    recovery_reason=pending_route.recovery_reason,
                    preferred_target=pending_route.target,
                ):
                    return
            else:
                self._clear_partial_route(pending_route.target)
            self._trace(
                "drone_partial_frontier_route_cancelled",
                target=pending_route.target,
                reason=(
                    "route_failed" if target_active else "frontier_changed"
                ),
            )
            self._pending_frontier_route = None

        if self._has_exploration_scope():
            if not drone.snapshot().frontiers:
                # Allow the second exhaustion confirmation without a random
                # step or a pointless ingress into already resolved work.
                return
            if not self._in_exploration_scope(drone.snapshot().position):
                self._advance_scoped_frontier_route()
                return

        self._trace("drone_move_start", state=self._snapshot_summary())
        if self._recover_from_stagnation():
            return
        try:
            valid_directions, border_targets, chosen_target = (
                self.find_new_node()
            )
        except AssertionError:
            self.update_borders()
            if self._has_exploration_scope():
                self._advance_scoped_frontier_route()
                return
            if self.reach_border():
                return
            if self._recover_to_assigned_sector():
                return
            self._trace(
                "drone_no_reachable_border",
                state=self._snapshot_summary(),
            )
            return

        self.explore(valid_directions, border_targets, chosen_target)

    def _move_component_policy(self) -> None:
        """Advance one rover directive while keeping sensing drone-local."""
        if self.is_docked():
            ready = self._coordination_ready
            if ready is not None and ready.is_set():
                self._advance_component_check_in()
            return
        shared_change = self._refresh_frontiers_before_mission_state()
        if shared_change and self._coordination_execution is not None:
            target_changed = self._revalidate_component_target_after_share(
                self._coordination_execution
            )
            if target_changed and self._pending_incidental_scan is not None:
                self._cancel_pending_incidental_scan(
                    "shared_slam_invalidated"
                )
        state = self.drone.snapshot()
        if state.done:
            self._cancel_pending_incidental_scan("mission_complete")
            return
        if state.returning_home:
            self._cancel_pending_incidental_scan("homing")
            if self.reach_start_point():
                self.drone.runtime_state.mark_done()
            return
        if self._coordination_execution is not None:
            self._advance_coordination_execution()
            return
        self._advance_component_check_in()

    def _revalidate_component_target_after_share(
        self,
        execution: _CoordinationExecution,
    ) -> bool:
        """Retarget stale local work from newly received peer SLAM."""
        if execution.directive.kind not in {
            DirectiveKind.COMPONENT_TASK,
            DirectiveKind.COMPONENT_BATCH,
            DirectiveKind.COMPONENT_FOLLOW,
        }:
            return False
        if execution.phase not in {"transit", "scanning", "repositioning"}:
            return False
        if execution.scan is not None and execution.scan.requested_at is not None:
            return False
        dfs = execution.dfs
        frame = None if dfs is None else dfs.current
        if dfs is None or frame is None:
            return False
        anchor = frame.node.anchor_position
        active_radius = max(2.0, float(self.frontier_stride * 2))
        if any(
            math.dist(anchor, frontier) <= active_radius
            for frontier in self._last_raw_frontiers
        ):
            return False

        local_slam = self.drone.slam_map.snapshot(point_limit=0)
        successors = related_local_successors(
            local_slam,
            frame.node.cells,
            confidence_threshold=self.frontier_confidence_threshold,
            minimum_component_cells=self.minimum_frontier_cluster_cells,
            minimum_unknown_support_cells=max(
                1,
                int(getattr(
                    self.drone.settings.frontier,
                    "minimum_unknown_support_cells",
                    64,
                )),
            ),
            lineage_radius=max(
                self.frontier_stride * 2,
                min(
                    self.global_frontier_cell_size,
                    int(math.ceil(self._sensor_range() / 2.0)),
                ),
            ),
        )
        nodes = self._ordered_branch_nodes(
            self._reachable_local_nodes(execution, successors)
        )
        if nodes:
            replacement = replace(
                nodes[0],
                local_id=frame.node.local_id,
                work_unit_id=frame.node.work_unit_id,
            )
            previous = frame.node
            frame.node = replacement
            frame.outbound_actual_path = ()
            frame.pending_children.clear()
            frame.scanned = False
            execution.scan = None
            execution.phase = "transit"
            execution.route_attempts = 0
            execution.transit_node_id = None
            execution.transit_path_start_index = None
            execution.observation_node_id = None
            execution.observation_position = None
            execution.observation_heading = None
            execution.reposition_target = None
            execution.reposition_fallback_target = None
            execution.reposition_attempts = 0
            self._trace(
                "drone_component_target_adjusted_after_share",
                directive_id=execution.directive.directive_id,
                component_id=execution.root_component_id,
                work_unit_id=replacement.work_unit_id,
                previous_anchor=previous.anchor_position,
                replacement_anchor=replacement.anchor_position,
                shared_slam_version=local_slam.version,
            )
            return True

        if frame.node.work_unit_id is not None:
            execution.work_unit_outcomes.append(WorkUnitOutcome(
                work_unit_id=frame.node.work_unit_id,
                disposition="shared_resolved",
            ))
        if execution.transit_path_start_index is not None:
            dfs.record_outbound(self._path_history_since(
                execution.transit_path_start_index
            ))
        dfs.record_scan(())
        execution.scan = None
        self._trace(
            "drone_component_target_retired_after_share",
            directive_id=execution.directive.directive_id,
            component_id=execution.root_component_id,
            work_unit_id=frame.node.work_unit_id,
            anchor=anchor,
            shared_slam_version=local_slam.version,
        )
        self._pop_component_frame(execution)
        return True

    def _component_route_invalidated_after_share(
        self,
        execution: _CoordinationExecution,
    ) -> bool:
        """Apply a received peer map and stop only if it changes local work."""
        if not self._shared_slam_changed.is_set():
            return False
        if not self._refresh_frontiers_before_mission_state():
            return False
        return self._revalidate_component_target_after_share(execution)

    def _advance_component_check_in(
        self,
        *,
        encounter_source: str | None = None,
    ) -> None:
        """Return/upload once, then consume the rover's signalled directive."""
        callback = self.dependencies.exploration_check_in
        assignment_callback = self.dependencies.exploration_assignment
        if (
            not callable(callback)
            or not callable(assignment_callback)
            or self._rendezvous_position() is None
        ):
            return
        ready = self._coordination_ready
        if ready is not None:
            contact = self.dependencies.exploration_contact
            if callable(contact) and not contact(self.drone.id):
                # The rover can leave an acknowledged rendezvous while this
                # drone awaits teammates. Rejoin before claiming any directive.
                self._coordination_ready = None
                self._coordination_wait_started_at = None
                self._trace("drone_coordination_wait_contact_lost")
            elif not ready.is_set():
                return
            else:
                result = assignment_callback(self.drone.id)
                if isinstance(result, CoordinationResult) and not result.arrived:
                    # A worker-side contact check may fail after a queued
                    # attempt. Keep the report for a fresh physical check-in.
                    self._coordination_ready = None
                    self._coordination_wait_started_at = None
                    return
                if (
                    isinstance(result, CoordinationResult)
                    and result.report_accepted
                ):
                    self._coordination_report = None
                self._apply_coordination_result(result)
                return

        current = self.drone.snapshot().position
        rover_position = self._rendezvous_position()
        if rover_position is None:
            return
        report = self._coordination_report

        def trace_encounter(source: str, result: Any) -> None:
            if report is None:
                return
            self._trace(
                "drone_report_encounter",
                report_id=report.report_id,
                directive_id=report.directive_id,
                source=source,
                position=self.drone.snapshot().position,
                remembered_endpoint=rover_position,
                queued=(
                    isinstance(result, CoordinationResult) and result.arrived
                ),
            )

        result = None
        if report is not None:
            # A moving rover may be in direct contact even while this drone's
            # remembered endpoint is elsewhere. The callback performs its
            # physical check and queues under the same lock as rover movement.
            result = callback(self.drone.id, report)
            if encounter_source is not None:
                trace_encounter(encounter_source, result)
                encounter_source = None
            elif (
                isinstance(result, CoordinationResult)
                and result.arrived
                and current != rover_position
            ):
                trace_encounter("reporting_contact", result)
            if isinstance(result, CoordinationResult) and result.arrived:
                if result.report_accepted:
                    self._coordination_report = None
                self._apply_coordination_result(result)
                return
        else:
            request_dock = self.dependencies.request_exploration_dock
            if callable(request_dock):
                result = request_dock(self.drone.id)
                if isinstance(result, CoordinationResult) and result.arrived:
                    self._apply_coordination_result(result)
                    return
        if current != rover_position:
            reached, encountered = self._return_to_coordination_point(
                rover_position,
                dock_during_check_in=(report is None),
            )
            if not reached:
                return
            if encountered and report is None:
                return
            result = callback(self.drone.id, report)
            if encountered:
                trace_encounter("reporting_path", result)
        elif result is None:
            result = callback(self.drone.id, report)
        if (
            isinstance(result, CoordinationResult)
            and result.arrived
            and (report is None or result.report_accepted)
        ):
            self._coordination_report = None
        elif (
            isinstance(result, CoordinationResult)
            and not result.arrived
            and self.drone.snapshot().position == rover_position
        ):
            # A queued physical check-in has arrived even while its report is
            # pending rover-worker acceptance. Only an actual missed contact
            # can justify abandoning this rendezvous point.
            missed = self.dependencies.rendezvous_endpoint_missed
            if callable(missed):
                next_endpoint = missed(self.drone.id, rover_position)
                if tuple(next_endpoint) != tuple(rover_position):
                    self._trace(
                        "drone_component_rendezvous_retargeted",
                        directive_id=(
                            None if report is None else report.directive_id
                        ),
                        previous_endpoint=rover_position,
                        endpoint=tuple(next_endpoint),
                        reason="rover_absent_at_confirmed_endpoint",
                    )
        self._apply_coordination_result(result)

    def _apply_coordination_result(self, result: Any) -> None:
        if not isinstance(result, CoordinationResult) or not result.arrived:
            return
        if result.highway_snapshot is not None:
            previous = self._highway_snapshot
            self._highway_snapshot = result.highway_snapshot
            self._trace(
                "drone_highway_snapshot_received",
                highway_version=result.highway_snapshot.version,
                replaced_version=(
                    None if previous is None else previous.version
                ),
                changed=(
                    previous is None
                    or previous.version != result.highway_snapshot.version
                ),
                area_count=result.highway_snapshot.area_count,
                edge_count=result.highway_snapshot.edge_count,
                known_free_cells=(
                    result.highway_snapshot.known_free_cells
                ),
                build_elapsed_ms=(
                    result.highway_snapshot.build_elapsed_ms
                ),
            )
        if result.directive is not None:
            self._install_exploration_directive(result.directive)
            return
        self._coordination_ready = result.directive_ready
        if self._coordination_wait_started_at is None:
            self._coordination_wait_started_at = self._simulation_time()
            self._trace(
                "drone_component_wait_started",
                mission_exhausted=result.mission_exhausted,
            )

    def _install_exploration_directive(
        self,
        directive: ExplorationDirective,
    ) -> None:
        self._cancel_pending_incidental_scan("directive_changed")
        waited = None
        if self._coordination_wait_started_at is not None:
            waited = max(
                0.0,
                self._simulation_time() - self._coordination_wait_started_at,
            )
        self._coordination_ready = None
        self._coordination_wait_started_at = None
        if directive.kind == DirectiveKind.HOME:
            self._trace(
                "drone_component_homing_started",
                directive_id=directive.directive_id,
                reason=directive.reason,
                waited_seconds=waited,
            )
            # HOME is issued only after the drone physically checks in.  The
            # rover is the current home, so returning to the launch pixel here
            # would add a fictitious second homing leg.
            self.drone.runtime_state.mark_done()
            return

        path_start = max(0, len(self.drone.snapshot().path_history) - 1)
        execution = _CoordinationExecution(
            directive=directive,
            phase=(
                "scanning"
                if directive.kind == DirectiveKind.ROVER_SCAN
                else "transit"
            ),
            path_start_index=path_start,
            work_unit_outcomes=[],
            causal_successors=[],
            completed_scan_headings=[],
            timed_out_scan_headings=[],
            local_routes={},
            visited_successor_anchors=[],
            pending_authoritative_nodes=[],
            reserved_branch_count=max(0, directive.reserved_branch_count),
        )
        if directive.kind == DirectiveKind.ROVER_SCAN:
            execution.scan = ScanPlanProgress(
                position=self.drone.snapshot().position,
                headings=directive.scan_headings,
                started_at=self._simulation_time(),
            )
        elif directive.kind == DirectiveKind.COMPONENT_TASK:
            if (
                directive.task is None
                or directive.claim is None
                or not directive.work_units
            ):
                return
            dfs = LocalDFSStack()
            roots = tuple(
                dfs.new_node(
                    unit.cells,
                    unit.anchor_position,
                    unit.scan_headings[0] if unit.scan_headings else 0,
                    work_unit_id=unit.work_unit_id,
                    allow_standoff=(
                        unit.kind == WorkUnitKind.FOCUSED_ANCHOR
                    ),
                )
                for unit in directive.work_units
            )
            root = roots[0]
            dfs.start(root, parent_position=self.drone.snapshot().position)
            execution.dfs = dfs
            execution.root_component_id = directive.task.component_id
            execution.root_work_unit_id = root.work_unit_id
            execution.root_source_cells = root.cells
            execution.pending_authoritative_nodes = list(roots[1:])
        elif directive.kind == DirectiveKind.COMPONENT_BATCH:
            if (
                len(directive.batch_members) < 2
                or directive.spatial_lease is None
            ):
                return
            execution.batch_pending_members = list(directive.batch_members)
            execution.batch_completed_members = []
            execution.batch_provisional_observations = []
            execution.batch_started_at = self._simulation_time()
            execution.batch_detour_distance = (
                directive.estimated_detour_cost
            )
            if not self._start_next_batch_member(execution):
                return
        elif directive.kind == DirectiveKind.COMPONENT_FOLLOW:
            if directive.task is None or not directive.work_units:
                return
            execution.phase = "following"
            execution.dfs = LocalDFSStack()
            execution.root_component_id = directive.task.component_id
            execution.root_source_cells = frozenset(
                point
                for unit in directive.work_units
                for point in unit.cells
            )
            execution.follow_started_at = self._simulation_time()
            execution.follow_last_leader_seen_at = self._simulation_time()
            execution.follow_last_leader_progress_at = self._simulation_time()
        self._coordination_execution = execution
        self._trace(
            "drone_component_directive_started",
            directive_id=directive.directive_id,
            directive_kind=directive.kind.value,
            round_id=directive.round_id,
            task_id=(None if directive.task is None else directive.task.task_id),
            component_id=(
                None if directive.task is None
                else directive.task.component_id
            ),
            work_unit_ids=(
                tuple(
                    unit_id
                    for member in directive.batch_members
                    for unit_id in member.claim.work_unit_ids
                )
                if directive.kind == DirectiveKind.COMPONENT_BATCH
                else () if directive.claim is None
                else directive.claim.work_unit_ids
            ),
            claim_token=(
                tuple(member.claim.token for member in directive.batch_members)
                if directive.kind == DirectiveKind.COMPONENT_BATCH
                else None if directive.claim is None else directive.claim.token
            ),
            scan_headings=directive.scan_headings,
            probe_target=directive.probe_target,
            waited_seconds=waited,
        )

    @bounded_local_planning("batch_member_selection")
    def _start_next_batch_member(
        self,
        execution: _CoordinationExecution,
    ) -> bool:
        """Select the next claimed member from exact drone-local routes."""
        pending = execution.batch_pending_members or []
        if not pending:
            execution.batch_current_member = None
            return False
        current = self.drone.snapshot().position
        rendezvous = self._rendezvous_position()
        best: tuple[float, tuple[int, ...], tuple[BatchMember, ...]] | None = None
        if rendezvous is not None:
            for order in itertools.permutations(pending):
                cost = self._best_local_batch_route_cost(
                    current,
                    tuple(item.task.preferred_entry for item in order),
                    rendezvous,
                    preserve_order=True,
                )
                key = (cost, tuple(item.task.task_id for item in order), order)
                if best is None or key[:2] < best[:2]:
                    best = key
        if (best is not None and math.isfinite(best[0])
                and not self._local_slam_planner.active["budget_exhausted"]):
            member = best[2][0]
            route_cost = self._local_slam_distance(
                current, member.task.preferred_entry,
            )
        else:
            # Retain the rover's admitted tour if bounded local optimization
            # cannot finish. A budget miss does not establish unreachability.
            member = pending[0] if self._local_slam_planner.active["budget_exhausted"] else min(
                pending,
                key=lambda item: (
                    self._local_slam_distance(
                        current, item.task.preferred_entry,
                    ),
                    item.task.task_id,
                ),
            )
            route_cost = self._local_slam_distance(
                current, member.task.preferred_entry,
            )
        path = self._local_slam_path(
            current, member.task.preferred_entry, allow_partial=True,
        )
        pending.remove(member)
        execution.batch_current_member = member
        execution.batch_member_path_start_index = max(
            0, len(self.drone.snapshot().path_history) - 1,
        )
        execution.batch_member_started_at = self._simulation_time()
        execution.directive = replace(
            execution.directive,
            task=member.task,
            work_units=member.work_units,
            claim=member.claim,
            outbound_route=path,
        )
        remaining_nodes = max(
            1,
            execution.directive.maximum_total_dfs_nodes
            - execution.batch_completed_dfs_nodes,
        )
        dfs = LocalDFSStack(maximum_nodes=remaining_nodes)
        roots = tuple(
            dfs.new_node(
                unit.cells,
                unit.anchor_position,
                unit.scan_headings[0] if unit.scan_headings else 0,
                work_unit_id=unit.work_unit_id,
                allow_standoff=(unit.kind == WorkUnitKind.FOCUSED_ANCHOR),
                source_task_id=member.task.task_id,
            )
            for unit in member.work_units
        )
        root = roots[0]
        dfs.start(root, parent_position=current)
        execution.dfs = dfs
        execution.batch_current_dfs_nodes = 0
        execution.root_component_id = member.task.component_id
        execution.root_work_unit_id = root.work_unit_id
        execution.root_source_cells = root.cells
        execution.root_scanned = False
        execution.work_unit_outcomes = []
        execution.causal_successors = []
        execution.visited_successor_anchors = []
        execution.pending_authoritative_nodes = list(roots[1:])
        execution.local_routes = {}
        execution.outbound_actual_path = ()
        execution.root_outbound_distance = 0.0
        execution.service_distance = 0.0
        execution.suspension_reason = None
        execution.suspension_position = None
        execution.suspension_energy_state = None
        execution.route_attempts = 0
        execution.reposition_attempts = 0
        execution.transit_node_id = None
        execution.transit_path_start_index = None
        execution.observation_node_id = None
        execution.observation_position = None
        execution.observation_heading = None
        execution.phase = "transit"
        self._trace(
            "drone_focused_frontier_batch_member_selected",
            directive_id=execution.directive.directive_id,
            lease_id=(
                None if execution.directive.spatial_lease is None
                else execution.directive.spatial_lease.lease_id
            ),
            task_id=member.task.task_id,
            component_id=member.task.component_id,
            claim_token=member.claim.token,
            exact_route_cost=route_cost,
            remaining_member_count=len(pending),
        )
        return True

    def _advance_coordination_execution(self) -> None:
        execution = self._coordination_execution
        if execution is None:
            return
        kind = execution.directive.kind
        if (
            kind == DirectiveKind.COMPONENT_BATCH
            and execution.phase != "returning"
            and execution.batch_started_at is not None
            and execution.directive.maximum_service_seconds > 0.0
            and self._simulation_time() - execution.batch_started_at
            >= execution.directive.maximum_service_seconds
        ):
            self._prepare_batch_return(execution, reason="service_time_limit")
            return
        if (
            kind != DirectiveKind.ROVER_SCAN
            and execution.phase != "returning"
            and (
                kind == DirectiveKind.COMPONENT_BATCH
                or execution.phase != "repositioning"
            )
            and self._coordination_energy_checkpoint_due(execution)
            and self._energy_requires_return(execution)
        ):
            return
        if self._pending_incidental_scan is not None:
            pending = self._pending_incidental_scan
            if (
                pending.directive_id != execution.directive.directive_id
                or pending.original_phase != execution.phase
            ):
                self._cancel_pending_incidental_scan(
                    "route_context_changed"
                )
            else:
                self._advance_pending_incidental_scan(execution)
                return
        if kind == DirectiveKind.ROVER_SCAN:
            if self._advance_coordination_scan(execution):
                self._finish_coordination_execution(execution)
            return
        if kind == DirectiveKind.RADIAL_PROBE:
            self._advance_radial_probe(execution)
            return
        if kind == DirectiveKind.COMPONENT_TASK:
            self._advance_component_task(execution)
            return
        if kind == DirectiveKind.COMPONENT_BATCH:
            self._advance_component_task(execution)
            return
        if kind == DirectiveKind.COMPONENT_FOLLOW:
            self._advance_component_follow(execution)

    def _advance_component_follow(
        self,
        execution: _CoordinationExecution,
    ) -> None:
        """Shadow a bootstrap leader, then take its first reserved branch."""
        if execution.phase != "following":
            self._advance_component_task(execution)
            return
        directive = execution.directive
        leader_id = directive.leader_drone_id
        if leader_id is None or execution.dfs is None:
            self._finish_coordination_execution(execution)
            return

        self.rebuild_frontiers(
            stride=self.frontier_stride,
            confidence_threshold=self.frontier_confidence_threshold,
        )
        local_slam = self.drone.slam_map.snapshot(point_limit=0)
        successors = related_local_successors(
            local_slam,
            execution.root_source_cells,
            confidence_threshold=self.frontier_confidence_threshold,
            minimum_component_cells=self.minimum_frontier_cluster_cells,
            minimum_unknown_support_cells=max(
                1,
                int(getattr(
                    self.drone.settings.frontier,
                    "minimum_unknown_support_cells",
                    64,
                )),
            ),
            lineage_radius=max(
                self.frontier_stride * 2,
                min(
                    self.global_frontier_cell_size,
                    int(math.ceil(self._sensor_range() / 2.0)),
                ),
            ),
        )
        nodes = self._ordered_branch_nodes(
            self._reachable_local_nodes(execution, successors)
        )
        branch_index = max(0, int(directive.follow_branch_index or 0))
        # Node zero is the leader's continuation.  A follower takes a later
        # child only after the local component actually separates.
        selected_index = branch_index + 1
        if len(nodes) > selected_index:
            branch = nodes[selected_index]
            execution.dfs.start(
                branch,
                parent_position=self.drone.snapshot().position,
            )
            execution.causal_successors.append(branch.cells)
            execution.phase = "transit"
            self._trace(
                "drone_component_branch_claimed",
                directive_id=directive.directive_id,
                leader_drone_id=leader_id,
                component_id=execution.root_component_id,
                branch_index=branch_index,
                anchor=branch.anchor_position,
            )
            return
        if nodes:
            # Carry the unbranched continuation forward so a split discovered
            # far from the bootstrap frontier still remains in lineage.
            execution.root_source_cells = nodes[0].cells

        leader_position = next((
            tuple(position)
            for drone_id, position in self.dependencies.get_drone_positions()
            if int(drone_id) == int(leader_id)
        ), None)
        now = self._simulation_time()
        if leader_position is not None:
            execution.follow_last_leader_seen_at = now
            last_position = execution.follow_last_leader_position
            if (
                last_position is None
                or math.dist(last_position, leader_position) >= 2.0
            ):
                execution.follow_last_leader_position = leader_position
                execution.follow_last_leader_progress_at = now
            last_progress = execution.follow_last_leader_progress_at
            if last_progress is not None and now - last_progress >= 15.0:
                self._trace(
                    "drone_component_branch_follow_ended",
                    directive_id=directive.directive_id,
                    leader_drone_id=leader_id,
                    component_id=execution.root_component_id,
                    reason="leader_visible_without_progress",
                )
                self._finish_coordination_execution(execution)
                return
            current = self.drone.snapshot().position
            if math.dist(current, leader_position) > max(
                2.0,
                float(self.drone.step) * 1.5,
            ):
                route = self._compute_path(current, leader_position)
                if route.path:
                    self._follow_path(
                        tuple(route.path)[:max(1, int(self.drone.step))],
                        source="component_branch_follow",
                    )
            return

        last_seen = execution.follow_last_leader_seen_at
        if last_seen is not None and now - last_seen >= 15.0:
            self._trace(
                "drone_component_branch_follow_ended",
                directive_id=directive.directive_id,
                leader_drone_id=leader_id,
                component_id=execution.root_component_id,
                reason="leader_contact_lost_without_branch",
            )
            self._finish_coordination_execution(execution)

    def _advance_radial_probe(self, execution: _CoordinationExecution) -> None:
        target = execution.directive.probe_target
        if target is None:
            self._finish_coordination_execution(execution)
            return
        if execution.phase == "transit":
            if not self._advance_directive_transit(
                execution,
                target,
                source="radial_probe_outbound",
                preferred_route=execution.directive.outbound_route,
            ):
                return
            execution.outbound_actual_path = self._path_history_since(
                execution.path_start_index
            )
            execution.scan = ScanPlanProgress(
                position=target,
                headings=execution.directive.scan_headings,
                started_at=self._simulation_time(),
            )
            execution.phase = "scanning"
            return
        if execution.phase == "scanning":
            if not self._advance_coordination_scan(execution):
                return
            execution.phase = "returning"
        if execution.phase == "returning":
            rover = self._rendezvous_position()
            if rover is None:
                return
            reached, encountered = self._return_to_coordination_point(
                rover,
                breadcrumb=execution.pending_return_breadcrumb
                or tuple(reversed(execution.outbound_actual_path)),
                execution=execution,
            )
            if not reached:
                return
            self._finish_coordination_execution(execution)
            if encountered:
                self._advance_component_check_in(
                    encounter_source="returning_path",
                )

    def _advance_component_task(self, execution: _CoordinationExecution) -> None:
        dfs = execution.dfs
        frame = None if dfs is None else dfs.current
        if execution.phase == "returning":
            rover = self._rendezvous_position()
            if rover is None:
                return
            reached, encountered = self._return_to_coordination_point(
                rover,
                breadcrumb=execution.pending_return_breadcrumb,
                execution=execution,
            )
            if not reached:
                return
            self._finish_coordination_execution(execution)
            if encountered:
                self._advance_component_check_in(
                    encounter_source="returning_path",
                )
            return
        if dfs is None or frame is None:
            self._finish_coordination_execution(execution)
            return
        if execution.phase == "repositioning":
            self._advance_component_reposition(execution)
            return
        if execution.phase == "transit":
            if execution.transit_node_id != frame.node.local_id:
                execution.transit_node_id = frame.node.local_id
                execution.transit_path_start_index = max(
                    0, len(self.drone.snapshot().path_history) - 1,
                )
                execution.observation_node_id = None
                execution.observation_position = None
                execution.observation_heading = None
            preferred = (
                execution.directive.outbound_route
                if frame.node.local_id == 0
                else (execution.local_routes or {}).get(
                    frame.node.local_id, (),
                )
            )
            if not self._advance_directive_transit(
                execution,
                frame.node.anchor_position,
                source="component_task_transit",
                preferred_route=preferred,
                observation_node=frame.node,
            ):
                return
            actual_path = self._path_history_since(
                execution.transit_path_start_index
                if execution.transit_path_start_index is not None
                else execution.path_start_index
            )
            dfs.record_outbound(actual_path)
            if frame.node.local_id == 0:
                execution.outbound_actual_path = actual_path
                execution.root_outbound_distance = (
                    self._polyline_distance(actual_path)
                )
            execution.scan = ScanPlanProgress(
                position=(
                    execution.observation_position
                    or frame.node.anchor_position
                ),
                headings=((
                    execution.observation_heading
                    if execution.observation_heading is not None
                    else frame.node.scan_heading
                ),),
                started_at=self._simulation_time(),
            )
            execution.phase = "scanning"
            execution.transit_node_id = None
            execution.transit_path_start_index = None
            return
        if execution.phase == "scanning":
            if not self._advance_coordination_scan(execution):
                return
            self._complete_component_scan(execution)

    def _complete_component_scan(
        self,
        execution: _CoordinationExecution,
    ) -> None:
        dfs = execution.dfs
        frame = None if dfs is None else dfs.current
        scan = execution.scan
        if dfs is None or frame is None or scan is None:
            self._finish_coordination_execution(execution)
            return
        if not scan.completed_headings:
            execution.scan = None
            self._prepare_execution_return(
                execution,
                reason="sensor_timeout",
            )
            return
        productive = bool(
            scan.newly_known_cells > 0
            or scan.confidence_gain > 1e-9
        )
        if execution.directive.kind == DirectiveKind.COMPONENT_BATCH:
            execution.batch_current_dfs_nodes += 1
        work_unit_id = frame.node.work_unit_id
        is_authoritative = work_unit_id is not None
        if not is_authoritative and not frame.node.provisional:
            execution.visited_successor_anchors.append(
                frame.node.anchor_position
            )
        if frame.node.provisional:
            observations = execution.batch_provisional_observations
            if observations is not None:
                observation = ProvisionalFrontierObservation(
                    observation_id=len(observations),
                    source_task_id=(
                        frame.node.source_task_id
                        if frame.node.source_task_id is not None
                        else execution.directive.task.task_id
                    ),
                    cells=frame.node.cells,
                    scan_position=scan.position,
                    scan_heading=(
                        scan.headings[0]
                        if scan.headings else frame.node.scan_heading
                    ),
                    sensor_newly_known_cells=scan.newly_known_cells,
                    sensor_confidence_gain=scan.confidence_gain,
                    local_route_distance=self._polyline_distance(
                        frame.outbound_actual_path
                    ),
                )
                observations.append(observation)
                self._focused_frontier_batch_provisional_memory.append(
                    _FocusedFrontierBatchProvisionalAttempt(
                        directive_id=execution.directive.directive_id,
                        observation_id=observation.observation_id,
                        cells=observation.cells,
                        serviced_at=self._simulation_time(),
                    )
                )
                if len(self._focused_frontier_batch_provisional_memory) > 256:
                    del self._focused_frontier_batch_provisional_memory[:-256]
        if is_authoritative:
            disposition = "sensor_gain" if productive else "zero_gain"
            execution.work_unit_outcomes.append(WorkUnitOutcome(
                work_unit_id=work_unit_id,
                disposition=disposition,
                sensor_newly_known_cells=scan.newly_known_cells,
                sensor_confidence_gain=scan.confidence_gain,
                scan_position=scan.position,
                scan_heading=(
                    scan.headings[0]
                    if scan.headings else frame.node.scan_heading
                ),
                frontier_position=frame.node.anchor_position,
                frontier_heading=frame.node.scan_heading,
            ))
            if work_unit_id == execution.root_work_unit_id:
                execution.root_scanned = True

        if (
            execution.directive.kind == DirectiveKind.COMPONENT_BATCH
            and execution.batch_completed_dfs_nodes
            + execution.batch_current_dfs_nodes
            >= execution.directive.maximum_total_dfs_nodes
        ):
            execution.scan = None
            self._prepare_batch_return(execution, reason="dfs_node_limit")
            return

        # The ordinary sensor scheduler may expose the next frontier just
        # before the directed completion is sampled.  A zero delta therefore
        # retires this anchor, but must not stop the local lineage walk when a
        # successor is already visible in the scan footprint.
        self.rebuild_frontiers(
            stride=self.frontier_stride,
            confidence_threshold=self.frontier_confidence_threshold,
        )
        local_slam = self.drone.slam_map.snapshot(point_limit=0)
        related_successors = tuple(
            cells for cells in related_local_successors(
                local_slam,
                frame.node.cells,
                confidence_threshold=self.frontier_confidence_threshold,
                minimum_component_cells=self.minimum_frontier_cluster_cells,
                minimum_unknown_support_cells=max(
                    1,
                    int(getattr(
                        self.drone.settings.frontier,
                        "minimum_unknown_support_cells",
                        64,
                    )),
                ),
                lineage_radius=max(
                    self.frontier_stride * 2,
                    min(
                        self.global_frontier_cell_size,
                        int(math.ceil(self._sensor_range() / 2.0)),
                    ),
                ),
                scan_origin=frame.node.anchor_position,
                scan_heading=frame.node.scan_heading,
                sensor_range=self._sensor_range(),
                sensor_fov_deg=self._sensor_fov(),
            )
            if cells != frame.node.cells
            and cells not in dfs.visited_geometry
        )
        if related_successors and not frame.node.provisional:
            execution.causal_successors.extend(related_successors)

        batch = execution.directive.kind == DirectiveKind.COMPONENT_BATCH
        if batch:
            low_gain = bool(
                scan.newly_known_cells
                <= execution.directive.low_gain_maximum_new_cells
                and scan.confidence_gain
                <= execution.directive.low_gain_maximum_confidence_gain
            )
            execution.batch_consecutive_low_gain_scans = (
                execution.batch_consecutive_low_gain_scans + 1
                if low_gain else 0
            )
            if execution.batch_consecutive_low_gain_scans >= (
                execution.directive.maximum_consecutive_low_gain_scans
            ):
                related_successors = ()

        nodes = self._reachable_local_nodes(
            execution,
            related_successors,
            provisional=frame.node.provisional,
        )
        leased_nodes: tuple[LocalComponentNode, ...] = ()
        if batch and self._batch_can_admit_provisional(execution):
            lease = execution.directive.spatial_lease
            dfs_reserved = tuple(
                node.cells
                for dfs_frame in dfs.frames
                for node in (
                    dfs_frame.node,
                    *dfs_frame.pending_children,
                )
            )
            excluded = tuple(dfs.visited_geometry) + dfs_reserved + tuple(
                node.cells
                for node in (execution.pending_authoritative_nodes or ())
            ) + tuple(
                member_unit.cells
                for member in execution.directive.batch_members
                for member_unit in member.work_units
            )
            leased = tuple(
                cells for cells in leased_local_frontiers(
                    local_slam,
                    allowed_cells=(
                        () if lease is None else self._spatial_lease_cells(lease)
                    ),
                    excluded_geometry=excluded,
                    confidence_threshold=self.frontier_confidence_threshold,
                    minimum_component_cells=(
                        self.minimum_frontier_cluster_cells
                    ),
                    minimum_unknown_support_cells=max(
                        1,
                        int(getattr(
                            self.drone.settings.frontier,
                            "minimum_unknown_support_cells",
                            64,
                        )),
                    ),
                )
                if cells not in related_successors
                and cells != frame.node.cells
            )
            leased_nodes = self._reachable_local_nodes(
                execution,
                leased,
                provisional=True,
            )
            remaining_slots = max(
                0,
                execution.directive.maximum_total_components
                - len(execution.directive.batch_members)
                - execution.batch_provisional_component_count,
            )
            leased_nodes = leased_nodes[:remaining_slots]
            execution.batch_provisional_component_count += len(leased_nodes)
        nodes = (*nodes, *leased_nodes)

        thin_node = None
        if not nodes and related_successors:
            thin_node = self._thin_local_successor(
                execution, related_successors,
            )
            if thin_node is not None:
                nodes = (thin_node,)
        if execution.reserved_branch_count > 0 and len(nodes) > 1:
            nodes = self._ordered_branch_nodes(nodes)
            reserved = min(
                execution.reserved_branch_count,
                len(nodes) - 1,
            )
            nodes = (*nodes[:1], *nodes[1 + reserved:])
            execution.reserved_branch_count -= reserved
            self._trace(
                "drone_component_branches_reserved_for_followers",
                directive_id=execution.directive.directive_id,
                component_id=execution.root_component_id,
                reserved_count=reserved,
                remaining_reservations=execution.reserved_branch_count,
            )
        child = dfs.record_scan(nodes)
        if child is not None and child is thin_node:
            execution.thin_successor_count += 1
            self._trace(
                "drone_component_thin_successor_followed",
                directive_id=execution.directive.directive_id,
                anchor=child.anchor_position,
                count=execution.thin_successor_count,
            )
        self._trace(
            "drone_work_unit_completed",
            directive_id=execution.directive.directive_id,
            task_id=(
                None if execution.directive.task is None
                else execution.directive.task.task_id
            ),
            component_id=execution.root_component_id,
            work_unit_id=work_unit_id,
            local_node_id=frame.node.local_id,
            productive=productive,
            sensor_newly_known_cells=scan.newly_known_cells,
            sensor_confidence_gain=scan.confidence_gain,
            successor_count=len(nodes),
            dfs_depth=len(dfs.frames),
            dfs_limit_reached=dfs.limit_reached,
        )
        execution.scan = None
        if child is not None:
            execution.phase = "transit"
            self._trace(
                "drone_dfs_pushed",
                directive_id=execution.directive.directive_id,
                local_node_id=child.local_id,
                depth=len(dfs.frames),
                pending_siblings=len(frame.pending_children),
            )
            return
        self._pop_component_frame(execution)

    @staticmethod
    def _ordered_branch_nodes(
        nodes: Iterable[LocalComponentNode],
    ) -> tuple[LocalComponentNode, ...]:
        """Give leaders and followers the same branch reservation order."""
        return tuple(sorted(
            nodes,
            key=lambda node: (
                min((point[1], point[0]) for point in node.cells),
                node.local_id,
            ),
        ))

    @bounded_local_planning("batch_successors")
    def _reachable_local_nodes(
        self,
        execution: _CoordinationExecution,
        successors: Iterable[frozenset[Position]],
        *,
        provisional: bool = False,
    ) -> tuple[LocalComponentNode, ...]:
        slam = self.drone.slam_map.snapshot(point_limit=0)
        current = self.drone.snapshot().position
        ranked: list[tuple[float, int, LocalComponentNode, tuple[Position, ...]]] = []
        dfs = execution.dfs
        if dfs is None:
            return ()
        current_frame = dfs.current
        excluded_anchors = set(dfs.visited_anchors)
        if current_frame is not None:
            excluded_anchors.add(current_frame.node.anchor_position)
        excluded_anchors.update(
            node.anchor_position
            for node in (execution.pending_authoritative_nodes or ())
        )
        accepted_detour = execution.batch_detour_distance
        for cells in successors:
            pose = local_node_pose(
                cells,
                slam,
                origin=current,
                excluded_anchors=excluded_anchors,
                minimum_anchor_spacing=self._sweep_anchor_spacing(),
                preferred_heading=(
                    None
                    if current_frame is None
                    else current_frame.node.scan_heading
                ),
            )
            if pose is None:
                continue
            anchor, heading = pose
            if execution.directive.kind == DirectiveKind.COMPONENT_BATCH:
                path = self._local_slam_path(current, anchor)
                route_status = PATH_COMPLETE if path else PATH_UNREACHABLE
            else:
                route = self._compute_path(current, anchor)
                path = tuple(route.path)
                route_status = route.status
            if route_status not in {PATH_COMPLETE, PATH_PARTIAL_LIMIT}:
                if provisional:
                    self._trace(
                        "drone_focused_frontier_batch_frontier_evaluated",
                        directive_id=execution.directive.directive_id,
                        anchor=anchor,
                        accepted=False,
                        reason="locally_unreachable",
                    )
                continue
            route_cost = self._path_distance(current, path, len(path))
            avoided = 0.0
            detour = 0.0
            if provisional:
                local_route_cost = route_cost
                if not math.isfinite(local_route_cost):
                    self._trace(
                        "drone_focused_frontier_batch_frontier_evaluated",
                        directive_id=execution.directive.directive_id,
                        anchor=anchor,
                        accepted=False,
                        reason="local_slam_unreachable",
                    )
                    continue
                economic = self._batch_local_economics(
                    execution,
                    anchor,
                    local_route_cost,
                )
                if economic is None:
                    self._trace(
                        "drone_focused_frontier_batch_frontier_evaluated",
                        directive_id=execution.directive.directive_id,
                        anchor=anchor,
                        accepted=False,
                        reason="unroutable_economic_quote",
                    )
                    continue
                avoided, detour = economic
                if avoided <= 1e-9:
                    self._trace(
                        "drone_focused_frontier_batch_frontier_evaluated",
                        directive_id=execution.directive.directive_id,
                        anchor=anchor,
                        accepted=False,
                        reason="nonpositive_avoided_round_trip",
                        avoided_round_trip=avoided,
                        detour_distance=detour,
                    )
                    continue
                if avoided + 1e-9 < execution.directive.minimum_avoided_round_trip:
                    self._trace(
                        "drone_focused_frontier_batch_frontier_evaluated",
                        directive_id=execution.directive.directive_id,
                        anchor=anchor,
                        accepted=False,
                        reason="insufficient_avoided_round_trip",
                        avoided_round_trip=avoided,
                        detour_distance=detour,
                    )
                    continue
                if accepted_detour + detour > (
                    execution.directive.maximum_detour_distance + 1e-9
                ):
                    self._trace(
                        "drone_focused_frontier_batch_frontier_evaluated",
                        directive_id=execution.directive.directive_id,
                        anchor=anchor,
                        accepted=False,
                        reason="detour_limit",
                        avoided_round_trip=avoided,
                        detour_distance=detour,
                        cumulative_detour_distance=accepted_detour,
                    )
                    continue
            node = dfs.new_node(
                cells,
                anchor,
                heading,
                allow_standoff=True,
                provisional=provisional,
                source_task_id=(
                    None if execution.directive.task is None
                    else execution.directive.task.task_id
                ),
            )
            ranked.append((
                route_cost,
                node.local_id,
                node,
                path,
            ))
            if provisional:
                accepted_detour += detour
                xs = tuple(point[0] for point in cells)
                ys = tuple(point[1] for point in cells)
                self._trace(
                    "drone_focused_frontier_batch_frontier_evaluated",
                    directive_id=execution.directive.directive_id,
                    anchor=anchor,
                    accepted=True,
                    reason="positive_avoided_round_trip",
                    avoided_round_trip=avoided,
                    detour_distance=detour,
                    cumulative_detour_distance=accepted_detour,
                    frontier_cell_count=len(cells),
                    frontier_bbox=(min(xs), min(ys), max(xs), max(ys)),
                )
        if provisional:
            execution.batch_detour_distance = accepted_detour
        ranked.sort(key=lambda item: (item[0], item[1]))
        for _cost, _local_id, node, path in ranked:
            execution.local_routes[node.local_id] = path
        return tuple(item[2] for item in ranked)

    @staticmethod
    def _spatial_lease_cells(lease: SpatialLease) -> tuple[Position, ...]:
        return tuple(
            (x, y)
            for y, left, right in lease.cell_spans
            for x in range(left, right + 1)
        )

    def _batch_can_admit_provisional(
        self,
        execution: _CoordinationExecution,
    ) -> bool:
        dfs = execution.dfs
        return bool(
            execution.directive.spatial_lease is not None
            and execution.batch_consecutive_low_gain_scans
            < execution.directive.maximum_consecutive_low_gain_scans
            and len(execution.directive.batch_members)
            + execution.batch_provisional_component_count
            < execution.directive.maximum_total_components
            and execution.batch_completed_dfs_nodes
            + execution.batch_current_dfs_nodes
            < execution.directive.maximum_total_dfs_nodes
            and execution.batch_detour_distance
            < execution.directive.maximum_detour_distance
        )

    @bounded_local_planning("batch_provisional_economics")
    def _batch_local_economics(
        self,
        execution: _CoordinationExecution,
        anchor: Position,
        route_to_anchor_cost: float,
    ) -> tuple[float, float] | None:
        rendezvous = self._rendezvous_position()
        if rendezvous is None:
            return None
        current = self.drone.snapshot().position
        remaining = tuple(
            member.task.preferred_entry
            for member in (execution.batch_pending_members or ())
        )
        baseline_cost = self._best_local_batch_route_cost(
            current,
            remaining,
            rendezvous,
        )
        later_cost = self._local_slam_distance(rendezvous, anchor)
        inserted_tail = self._best_local_batch_route_cost(
            anchor,
            remaining,
            rendezvous,
        )
        if not all(math.isfinite(value) for value in (
            baseline_cost, later_cost, inserted_tail,
        )):
            return None
        inserted = route_to_anchor_cost + inserted_tail
        detour = max(0.0, inserted - baseline_cost)
        avoided = baseline_cost + 2.0 * later_cost - inserted
        return avoided, detour

    def _best_local_batch_route_cost(
        self,
        start: Position,
        waypoints: tuple[Position, ...],
        end: Position,
        *,
        preserve_order: bool = False,
    ) -> float:
        """Return an exact small-tour cost using only drone-local SLAM."""
        orders = (waypoints,) if preserve_order else itertools.permutations(
            waypoints
        )
        best = math.inf
        for order in orders:
            points = (start, *order, end)
            cost = sum(
                self._local_slam_distance(first, second)
                for first, second in zip(points, points[1:])
            )
            best = min(best, cost)
        return best

    def _local_slam_distance(
        self,
        start: Position,
        goal: Position,
    ) -> float:
        """Shortest confidently-free distance using only this drone's SLAM."""
        route = self._local_slam_planner.route(start, goal)
        return route.cost if route.complete else math.inf

    def _local_slam_path(
        self,
        start: Position,
        goal: Position,
        *,
        allow_partial: bool = False,
    ) -> tuple[Position, ...]:
        """One exact confidently-free route using only this drone's SLAM."""
        route = self._local_slam_planner.route(start, goal)
        self._last_local_route_status = route.status
        return route.path if route.complete or (allow_partial and route.status == PATH_PARTIAL_LIMIT) else ()

    def _thin_local_successor(
        self,
        execution: _CoordinationExecution,
        successors: Iterable[frozenset[Position]],
    ) -> LocalComponentNode | None:
        """Continue a locally advancing frontier inside normal scan spacing."""
        dfs = execution.dfs
        frame = None if dfs is None else dfs.current
        if (
            dfs is None
            or frame is None
            or execution.thin_successor_count >= 8
            or dfs.created_nodes >= dfs.maximum_nodes
            or len(dfs.frames) >= dfs.maximum_depth
        ):
            return None
        slam = self.drone.slam_map.snapshot(point_limit=0)
        current = self.drone.snapshot().position
        old_support = self._local_unknown_support(slam, frame.node.cells)
        excluded = set(dfs.visited_anchors)
        excluded.add(frame.node.anchor_position)
        for cells in successors:
            if not (self._local_unknown_support(slam, cells) - old_support):
                continue
            pose = local_node_pose(
                cells,
                slam,
                origin=current,
                excluded_anchors=excluded,
                minimum_anchor_spacing=1.0,
            )
            if pose is None:
                continue
            anchor, heading = pose
            if math.dist(
                anchor, frame.node.anchor_position,
            ) + 1e-9 >= self._sweep_anchor_spacing():
                # A distant or unreachable successor is ordinary component
                # work, not a thin continuation of this scan footprint.
                continue
            if execution.directive.kind == DirectiveKind.COMPONENT_BATCH:
                path = self._local_slam_path(current, anchor)
                route_status = PATH_COMPLETE if path else PATH_UNREACHABLE
            else:
                route = self._compute_path(current, anchor)
                path = tuple(route.path)
                route_status = route.status
            if route_status not in {PATH_COMPLETE, PATH_PARTIAL_LIMIT}:
                continue
            node = dfs.new_node(
                cells,
                anchor,
                heading,
                allow_standoff=True,
                provisional=frame.node.provisional,
                source_task_id=frame.node.source_task_id,
            )
            execution.local_routes[node.local_id] = path
            return node
        return None

    def _local_unknown_support(
        self,
        slam: Any,
        cells: Iterable[Position],
    ) -> frozenset[Position]:
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        unknown = (
            (occupancy == UNKNOWN)
            | (confidence < self.frontier_confidence_threshold)
        )
        height, width = unknown.shape
        return frozenset(
            (neighbor_x, neighbor_y)
            for x, y in cells
            for neighbor_y in range(max(0, y - 1), min(height, y + 2))
            for neighbor_x in range(max(0, x - 1), min(width, x + 2))
            if unknown[neighbor_y, neighbor_x]
        )

    def _pop_component_frame(self, execution: _CoordinationExecution) -> None:
        dfs = execution.dfs
        if dfs is None:
            self._finish_coordination_execution(execution)
            return
        if len(dfs.frames) > 1:
            parent = dfs.frames[-2]
            parent.pending_children = self._rank_pending_component_nodes(
                execution,
                parent.pending_children,
            )
        backtrack, sibling = dfs.pop_completed()
        self._trace(
            "drone_dfs_popped",
            directive_id=execution.directive.directive_id,
            remaining_depth=len(dfs.frames),
            recorded_breadcrumb_points=len(backtrack),
            next_sibling_id=None if sibling is None else sibling.local_id,
        )
        if sibling is not None and dfs.current is not None:
            self._begin_component_reposition(
                execution,
                target=sibling.anchor_position,
                fallback_target=dfs.current.parent_position,
            )
            return
        self._continue_component_dfs(execution)

    def _continue_component_dfs(
        self,
        execution: _CoordinationExecution,
    ) -> None:
        dfs = execution.dfs
        frame = None if dfs is None else dfs.current
        if frame is None:
            pending = execution.pending_authoritative_nodes or []
            if dfs is not None and pending:
                pending[:] = self._rank_pending_component_nodes(
                    execution,
                    pending,
                )
            if dfs is not None and pending:
                node = pending.pop(0)
                history = self._path_history_since(
                    execution.path_start_index
                )
                dfs.start(
                    node,
                    parent_position=self.drone.snapshot().position,
                )
                self._begin_component_reposition(
                    execution,
                    target=node.anchor_position,
                    fallback_target=history[0] if history else None,
                )
                self._trace(
                    "drone_component_sweep_advanced",
                    directive_id=execution.directive.directive_id,
                    component_id=execution.root_component_id,
                    work_unit_id=node.work_unit_id,
                    anchor=node.anchor_position,
                    remaining_anchor_count=len(pending),
                )
                return
            self._prepare_completed_component_return(execution)
            return
        if not frame.scanned:
            execution.phase = "transit"
            return
        if frame.pending_children:
            frame.pending_children = self._rank_pending_component_nodes(
                execution,
                frame.pending_children,
            )
        if frame.pending_children:
            child = frame.pending_children.pop(0)
            dfs.frames.append(type(frame)(
                child,
                parent_position=frame.node.anchor_position,
            ))
            execution.phase = "transit"
            return
        self._pop_component_frame(execution)

    def _rank_pending_component_nodes(
        self,
        execution: _CoordinationExecution,
        pending: Iterable[LocalComponentNode],
    ) -> list[LocalComponentNode]:
        """Prune stale siblings, then use cheap current-position ordering."""
        nodes = tuple(pending)
        if not nodes:
            return []
        if self.drone.slam_map.version != self._frontier_slam_version:
            self.rebuild_frontiers(
                stride=self.frontier_stride,
                confidence_threshold=self.frontier_confidence_threshold,
            )
        slam = self.drone.slam_map.snapshot(point_limit=0)
        frontier_mask = self._last_frontier_mask
        current = self.drone.snapshot().position
        if frontier_mask is None:
            return sorted(
                nodes,
                key=lambda node: (
                    math.dist(current, node.anchor_position),
                    node.local_id,
                ),
            )
        dfs = execution.dfs
        visited = () if dfs is None else tuple(dfs.visited_anchors)
        radius = max(2, self.frontier_stride * 2)
        height, width = frontier_mask.shape
        near_by_id: dict[int, bool] = {}
        for node in nodes:
            x, y = node.anchor_position
            near_by_id[node.local_id] = bool(
                0 <= x < width
                and 0 <= y < height
                and np.any(frontier_mask[
                    max(0, y - radius):min(height, y + radius + 1),
                    max(0, x - radius):min(width, x + radius + 1),
                ])
            )
        stale = tuple(
            node for node in nodes if not near_by_id[node.local_id]
        )
        lineage_radius = max(
            self.frontier_stride * 2,
            min(
                self.global_frontier_cell_size,
                int(math.ceil(self._sensor_range() / 2.0)),
            ),
        )
        successors = (
            related_local_successors(
                slam,
                frozenset(point for node in stale for point in node.cells),
                confidence_threshold=self.frontier_confidence_threshold,
                minimum_component_cells=self.minimum_frontier_cluster_cells,
                minimum_unknown_support_cells=max(
                    1,
                    int(getattr(
                        self.drone.settings.frontier,
                        "minimum_unknown_support_cells",
                        64,
                    )),
                ),
                lineage_radius=lineage_radius,
            )
            if stale else ()
        )
        viable: list[LocalComponentNode] = []
        reserved_anchors = set(visited)
        for node in nodes:
            replacement = node
            if not near_by_id[node.local_id]:
                replacement = None
                for cells in successors:
                    if min(
                        (math.dist(node.anchor_position, point)
                         for point in cells),
                        default=math.inf,
                    ) > lineage_radius:
                        continue
                    pose = local_node_pose(
                        cells,
                        slam,
                        origin=current,
                        excluded_anchors=reserved_anchors,
                        minimum_anchor_spacing=1.0,
                    )
                    if pose is not None:
                        replacement = replace(
                            node,
                            cells=cells,
                            anchor_position=pose[0],
                            scan_heading=pose[1],
                        )
                        break
            if replacement is None:
                if node.work_unit_id is not None:
                    execution.work_unit_outcomes.append(WorkUnitOutcome(
                        work_unit_id=node.work_unit_id,
                        disposition="shared_resolved",
                    ))
                self._trace(
                    "drone_component_pending_node_retired",
                    directive_id=execution.directive.directive_id,
                    local_node_id=node.local_id,
                    work_unit_id=node.work_unit_id,
                )
                continue
            if replacement is not node:
                execution.local_routes.pop(node.local_id, None)
            viable.append(replacement)
            reserved_anchors.add(replacement.anchor_position)
        return sorted(
            viable,
            key=lambda node: (
                math.dist(current, node.anchor_position),
                node.local_id,
            ),
        )

    def _begin_component_reposition(
        self,
        execution: _CoordinationExecution,
        *,
        target: Position,
        fallback_target: Position | None,
    ) -> None:
        """Reposition to the next logical DFS node, not its ancestors."""
        dfs = execution.dfs
        frame = None if dfs is None else dfs.current
        execution.reposition_target = tuple(target)
        execution.reposition_fallback_target = fallback_target
        execution.reposition_attempts = 0
        execution.transit_node_id = (
            None if frame is None else frame.node.local_id
        )
        execution.observation_node_id = None
        execution.observation_position = None
        execution.observation_heading = None
        execution.transit_path_start_index = max(
            0, len(self.drone.snapshot().path_history) - 1,
        )
        execution.phase = "repositioning"
        self._trace(
            "drone_dfs_reposition_started",
            directive_id=execution.directive.directive_id,
            start=self.drone.snapshot().position,
            target=target,
            fallback_target=fallback_target,
            depth=0 if dfs is None else len(dfs.frames),
        )

    def _advance_component_reposition(
        self,
        execution: _CoordinationExecution,
    ) -> None:
        target = execution.reposition_target
        if target is None:
            self._prepare_execution_return(
                execution,
                reason="dfs_reposition_target_missing",
            )
            return
        current = self.drone.snapshot().position
        dfs = execution.dfs
        frame = None if dfs is None else dfs.current
        if (
            frame is not None
            and execution.observation_node_id == frame.node.local_id
            and execution.observation_position == current
        ):
            self._finish_component_reposition(execution)
            return
        if current == target:
            self._finish_component_reposition(execution)
            return

        interrupted_by_share = False

        def stop_for_shared_target() -> bool:
            nonlocal interrupted_by_share
            interrupted_by_share = (
                self._component_route_invalidated_after_share(execution)
            )
            return interrupted_by_share

        if execution.directive.kind == DirectiveKind.COMPONENT_BATCH:
            local_path = self._local_slam_path(current, target, allow_partial=True)
            result = PathResult(
                local_path,
                (PATH_COMPLETE if local_path and local_path[-1] == target else
                 PATH_PARTIAL_LIMIT if local_path or self._last_local_route_status == "budget_exhausted"
                 else PATH_UNREACHABLE),
                0,
                0.0 if local_path else math.inf,
            )
        else:
            result = self._compute_path(current, target)
        path = tuple(result.path)
        self._record_incidental_pocket_revisit(
            target,
            path,
            source="component_dfs_reposition_astar",
        )
        self._record_focused_frontier_batch_provisional_revisit(
            target,
            path,
            directive_id=execution.directive.directive_id,
            source="component_dfs_reposition_astar",
        )

        effective_target = target
        if (
            result.status == PATH_COMPLETE
            and frame is not None
            and frame.node.allow_standoff
        ):
            observation = self._component_observation_pose(
                frame.node,
                path,
                origin=current,
            )
            if observation is not None:
                path = observation.route_prefix
                effective_target = observation.position
                first_selection = (
                    execution.observation_node_id != frame.node.local_id
                )
                execution.observation_node_id = frame.node.local_id
                execution.observation_position = observation.position
                execution.observation_heading = observation.heading
                if first_selection:
                    self._trace(
                        "drone_component_observation_pose_selected",
                        directive_id=execution.directive.directive_id,
                        component_id=execution.root_component_id,
                        work_unit_id=frame.node.work_unit_id,
                        local_node_id=frame.node.local_id,
                        source="component_dfs_reposition_astar",
                        frontier_position=target,
                        observation_position=observation.position,
                        scan_heading=observation.heading,
                        frontier_cells=len(frame.node.cells),
                        saved_route_distance=(
                            observation.saved_route_distance
                        ),
                    )
        followed = bool(path) and self._follow_path(
            path,
            source="component_dfs_reposition_astar",
            stop_when=stop_for_shared_target,
            stop_reason="shared_slam_invalidated",
            incidental_transit=True,
            path_target=effective_target,
            path_status=result.status,
        )
        reached = (
            followed
            and self.drone.snapshot().position == effective_target
        )
        if self._pending_incidental_scan is not None:
            self._trace(
                "drone_dfs_reposition_paused_for_incidental_scan",
                directive_id=execution.directive.directive_id,
                start=current,
                target=target,
                observation_target=effective_target,
                path_status=result.status,
                path_length=len(result.path),
                pause_position=self.drone.snapshot().position,
            )
            return
        self._trace(
            "drone_dfs_reposition_path",
            directive_id=execution.directive.directive_id,
            start=current,
            target=target,
            observation_target=effective_target,
            path_status=result.status,
            path_length=len(result.path),
            reached=reached,
        )
        if interrupted_by_share:
            return
        if reached:
            self._finish_component_reposition(execution)
            return
        execution.reposition_attempts += 1
        if (
            followed
            and result.status == PATH_PARTIAL_LIMIT
            and self.drone.snapshot().position != current
            and execution.reposition_attempts < 8
        ):
            return
        if not result.path and execution.reposition_attempts < 2:
            return

        fallback_target = execution.reposition_fallback_target
        breadcrumb = self._component_reposition_breadcrumb(
            execution,
            fallback_target,
        )
        fallback_start = self.drone.snapshot().position
        fallback_followed = bool(breadcrumb) and self._follow_path(
            breadcrumb,
            source="component_dfs_breadcrumb_fallback",
            stop_when=stop_for_shared_target,
            stop_reason="shared_slam_invalidated",
        )
        fallback_reached = bool(
            fallback_followed
            and self.drone.snapshot().position == fallback_target
        )
        self._trace(
            "drone_dfs_reposition_fallback",
            directive_id=execution.directive.directive_id,
            start=fallback_start,
            target=target,
            fallback_target=fallback_target,
            breadcrumb_points=len(breadcrumb),
            reached=fallback_reached,
        )
        if interrupted_by_share:
            return
        if fallback_reached and fallback_target != target:
            # From the last physical parent, ordinary transit plans the short
            # remaining leg to the same logical target.
            self._finish_component_reposition(execution)
            return
        if fallback_reached:
            self._finish_component_reposition(execution)
            return
        dfs = execution.dfs
        frame = None if dfs is None else dfs.current
        if (
            dfs is not None
            and frame is not None
            and frame.node.work_unit_id is None
            and len(dfs.frames) > 1
        ):
            self._trace(
                "drone_component_pending_node_retired",
                directive_id=execution.directive.directive_id,
                local_node_id=frame.node.local_id,
                work_unit_id=None,
                reason="route_unreachable",
            )
            dfs.frames.pop()
            execution.reposition_target = None
            execution.reposition_fallback_target = None
            self._continue_component_dfs(execution)
            return
        self._prepare_execution_return(
            execution,
            reason=("route_planning_budget" if result.status == PATH_PARTIAL_LIMIT
                    else "dfs_reposition_unreachable"),
        )

    def _component_reposition_breadcrumb(
        self,
        execution: _CoordinationExecution,
        target: Position | None,
    ) -> tuple[Position, ...]:
        """Find an actual traversed suffix to a previously visited anchor."""
        if target is None:
            return ()
        history = self._path_history_since(execution.path_start_index)
        for index in range(len(history) - 1, -1, -1):
            if history[index] == target:
                return tuple(reversed(history[index:]))
        return ()

    @staticmethod
    def _finish_component_reposition(
        execution: _CoordinationExecution,
    ) -> None:
        execution.reposition_target = None
        execution.reposition_fallback_target = None
        execution.reposition_attempts = 0
        execution.route_attempts = 0
        execution.phase = "transit"

    def _prepare_completed_component_return(
        self,
        execution: _CoordinationExecution,
    ) -> None:
        """Return with a complete claim; only upload after physical contact."""
        if execution.directive.kind == DirectiveKind.COMPONENT_BATCH:
            self._capture_batch_member(execution, disposition="complete")
            if self._start_next_batch_member(execution):
                return
            self._prepare_batch_return(execution)
            return
        execution.outbound_actual_path = self._path_history_since(
            execution.path_start_index
        )
        total_outward_distance = self._polyline_distance(
            execution.outbound_actual_path
        )
        if execution.root_outbound_distance <= 1e-9:
            execution.root_outbound_distance = total_outward_distance
            execution.service_distance = 0.0
        else:
            execution.service_distance = max(
                0.0,
                total_outward_distance - execution.root_outbound_distance,
            )
        execution.pending_return_breadcrumb = tuple(reversed(
            execution.outbound_actual_path
        ))
        execution.return_path_start_index = max(
            0, len(self.drone.snapshot().path_history) - 1,
        )
        execution.phase = "returning"
        self._trace(
            "drone_component_return_after_logical_dfs",
            directive_id=execution.directive.directive_id,
            position=self.drone.snapshot().position,
            breadcrumb_point_count=len(execution.pending_return_breadcrumb),
        )

    def _advance_directive_transit(
        self,
        execution: _CoordinationExecution,
        target: Position,
        *,
        source: str,
        preferred_route: Iterable[Position] = (),
        observation_node: LocalComponentNode | None = None,
    ) -> bool:
        current = self.drone.snapshot().position
        if (
            observation_node is not None
            and execution.observation_node_id == observation_node.local_id
            and execution.observation_position == current
        ):
            execution.route_attempts = 0
            return True
        if current == target:
            execution.route_attempts = 0
            return True
        path = tuple(preferred_route) if execution.route_attempts == 0 else ()
        if path and path[0] != current:
            path = ()
        status = (PATH_COMPLETE if path and path[-1] == target else
                  PATH_PARTIAL_LIMIT if path else None)
        if not path:
            if execution.directive.kind == DirectiveKind.COMPONENT_BATCH:
                path = self._local_slam_path(current, target, allow_partial=True)
                status = (PATH_COMPLETE if path and path[-1] == target else
                          PATH_PARTIAL_LIMIT if path or self._last_local_route_status == "budget_exhausted"
                          else PATH_UNREACHABLE)
            else:
                result = self._compute_path(current, target)
                path = tuple(result.path)
                status = result.status
        self._record_incidental_pocket_revisit(
            target,
            path,
            source=source,
        )
        self._record_focused_frontier_batch_provisional_revisit(
            target,
            path,
            directive_id=execution.directive.directive_id,
            source=source,
        )
        execution.route_attempts += 1
        if not path:
            if execution.route_attempts >= 2:
                self._trace(
                    "drone_component_transit_failed",
                    directive_id=execution.directive.directive_id,
                    target=target,
                    source=source,
                    attempts=execution.route_attempts,
                )
                self._prepare_execution_return(
                    execution,
                    reason=("route_planning_budget" if status == PATH_PARTIAL_LIMIT
                            else "route_unreachable"),
                )
            return False
        effective_target = target
        if (
            status == PATH_COMPLETE
            and observation_node is not None
            and observation_node.allow_standoff
        ):
            observation = self._component_observation_pose(
                observation_node,
                path,
                origin=current,
            )
            if observation is not None:
                path = observation.route_prefix
                effective_target = observation.position
                first_selection = (
                    execution.observation_node_id
                    != observation_node.local_id
                )
                execution.observation_node_id = observation_node.local_id
                execution.observation_position = observation.position
                execution.observation_heading = observation.heading
                if first_selection:
                    self._trace(
                        "drone_component_observation_pose_selected",
                        directive_id=execution.directive.directive_id,
                        component_id=execution.root_component_id,
                        work_unit_id=observation_node.work_unit_id,
                        local_node_id=observation_node.local_id,
                        source=source,
                        frontier_position=target,
                        observation_position=observation.position,
                        scan_heading=observation.heading,
                        frontier_cells=len(observation_node.cells),
                        saved_route_distance=(
                            observation.saved_route_distance
                        ),
                    )
        stop_when = None
        stop_reason = None
        if execution.directive.kind in {
            DirectiveKind.COMPONENT_TASK,
            DirectiveKind.COMPONENT_BATCH,
            DirectiveKind.COMPONENT_FOLLOW,
        }:
            stop_when = lambda: self._component_route_invalidated_after_share(
                execution
            )
            stop_reason = "shared_slam_invalidated"
        followed = self._follow_path(
            path,
            source=source,
            stop_when=stop_when,
            stop_reason=stop_reason,
            incidental_transit=(
                execution.directive.kind in {
                    DirectiveKind.COMPONENT_TASK,
                    DirectiveKind.COMPONENT_BATCH,
                    DirectiveKind.COMPONENT_FOLLOW,
                }
                and execution.phase == "transit"
            ),
            path_target=effective_target,
            path_status=status,
        )
        if not followed:
            if self._pending_incidental_scan is not None:
                execution.route_attempts = max(
                    0, execution.route_attempts - 1,
                )
            return False
        if self.drone.snapshot().position == effective_target:
            execution.route_attempts = 0
            return True
        if status == PATH_PARTIAL_LIMIT and execution.route_attempts < 8:
            return False
        if execution.route_attempts >= 8:
            self._prepare_execution_return(
                execution,
                reason=("route_planning_budget"
                        if execution.directive.kind == DirectiveKind.COMPONENT_BATCH
                        and status == PATH_PARTIAL_LIMIT else "route_progress_limit"),
            )
        return False

    def _component_observation_pose(
        self,
        node: LocalComponentNode,
        path: Iterable[Position],
        *,
        origin: Position,
    ) -> FrontierObservationPose | None:
        """Trim a focused route only when local SLAM proves one safe cone."""
        return observation_pose_on_path(
            node.cells,
            self.drone.slam_map.snapshot(point_limit=0),
            path,
            origin=origin,
            sensor_range=self._sensor_range(),
            sensor_fov_deg=self._sensor_fov(),
            confidence_threshold=self.frontier_confidence_threshold,
        )

    def _maybe_sample_incidental_transit(
        self,
        *,
        edge_distance: float,
        retained_route: tuple[Position, ...],
        route_target: Position,
        route_source: str,
        path_status: str | None,
    ) -> bool:
        """Detect and optionally start one bounded side scan at a route pose."""
        execution = self._coordination_execution
        if (
            execution is None
            or self.incidental_scan_mode == "off"
            or self._pending_incidental_scan is not None
            or execution.directive.kind not in {
                DirectiveKind.COMPONENT_TASK,
                DirectiveKind.COMPONENT_BATCH,
                DirectiveKind.COMPONENT_FOLLOW,
            }
            or route_source not in {
                "component_task_transit",
                "component_dfs_reposition_astar",
            }
            or execution.phase not in {"transit", "repositioning"}
            or execution.incidental_attempts >= self.incidental_maximum_attempts
            or execution.incidental_wait_seconds
            >= self.incidental_maximum_wait - 1e-9
        ):
            return False

        travelled = max(0.0, float(edge_distance))
        execution.incidental_distance_since_sample += travelled
        execution.incidental_distance_since_selection += travelled
        sensor_range = self._sensor_range()
        cooldown = sensor_range * self.incidental_cooldown_ranges
        if execution.incidental_distance_since_selection + 1e-9 < cooldown:
            return False
        spacing = sensor_range * self.incidental_sample_spacing_ranges
        if (
            spacing > 1e-9
            and execution.incidental_distance_since_sample + 1e-9 < spacing
        ):
            return False
        execution.incidental_distance_since_sample = 0.0

        runtime = self.drone.snapshot()
        dfs = execution.dfs
        frame = None if dfs is None else dfs.current
        active_cells = () if frame is None else frame.node.cells
        detector_started = time.perf_counter()
        detection = incidental_scan_candidate(
            self.drone.slam_map.snapshot(point_limit=0),
            position=runtime.position,
            current_heading=runtime.heading_deg,
            sensor_range=sensor_range,
            sensor_fov_deg=self._sensor_fov(),
            confidence_threshold=self.frontier_confidence_threshold,
            frontier_stride=self.frontier_stride,
            active_cells=active_cells,
        )
        detector_ms = (time.perf_counter() - detector_started) * 1000.0
        self._trace(
            "drone_incidental_scan_sampled",
            directive_id=execution.directive.directive_id,
            source=route_source,
            phase=execution.phase,
            pose=runtime.position,
            heading=runtime.heading_deg,
            slam_version=self.drone.slam_map.version,
            detector_elapsed_ms=detector_ms,
            component_count=detection.component_count,
            rejection_counts=dict(detection.rejection_counts),
        )
        candidate = detection.candidate
        if candidate is None:
            return False
        execution.incidental_candidates += 1
        if not self._incidental_candidate_is_reopenable(candidate):
            return False
        if (
            execution.incidental_requested_rotation
            + candidate.heading_delta
            > self.incidental_maximum_rotation + 1e-9
        ):
            return False

        execution.incidental_attempts += 1
        execution.incidental_requested_rotation += candidate.heading_delta
        execution.incidental_predicted_support += len(candidate.cells)
        execution.incidental_distance_since_selection = 0.0
        self._remember_incidental_attempt(candidate)
        remaining_attempts = max(
            0,
            self.incidental_maximum_attempts
            - execution.incidental_attempts,
        )
        remaining_wait = max(
            0.0,
            self.incidental_maximum_wait
            - execution.incidental_wait_seconds,
        )
        remaining_rotation = max(
            0.0,
            self.incidental_maximum_rotation
            - execution.incidental_requested_rotation,
        )
        self._trace(
            "drone_incidental_scan_candidate",
            directive_id=execution.directive.directive_id,
            source=route_source,
            phase=execution.phase,
            mode=self.incidental_scan_mode,
            signature=self._incidental_signature_fields(candidate.signature),
            gateway=candidate.gateway,
            pocket_cells=len(candidate.cells),
            wall_cells=candidate.wall_cells,
            side_only_support=candidate.side_only_support,
            current_visibility=candidate.current_visibility,
            proposed_visibility=candidate.proposed_visibility,
            heading=candidate.heading,
            heading_delta=candidate.heading_delta,
            remaining_attempts=remaining_attempts,
            remaining_wait_seconds=remaining_wait,
            remaining_rotation_degrees=remaining_rotation,
        )
        if self.incidental_scan_mode != "active":
            return False

        timeout = min(self.incidental_attempt_timeout, remaining_wait)
        if timeout <= 1e-9:
            return False
        sensor = getattr(self.drone, "sensor_controller", None)
        completion = getattr(sensor, "last_completed_scan", None)
        minimum_sequence = -1 if completion is None else int(completion.sequence)
        requested_at = self._simulation_time()
        pending = _PendingIncidentalTransitScan(
            directive_id=execution.directive.directive_id,
            original_phase=execution.phase,
            position=runtime.position,
            heading=candidate.heading,
            resume_heading=float(runtime.heading_deg),
            candidate=candidate,
            minimum_scan_sequence=minimum_sequence,
            requested_at=requested_at,
            deadline=requested_at + timeout,
            retained_route=tuple(retained_route),
            route_target=(int(route_target[0]), int(route_target[1])),
            route_source=route_source,
            path_status=path_status,
            baseline_slam_version=self.drone.slam_map.version,
        )
        self._pending_incidental_scan = pending
        self.drone.runtime_state.reorient(candidate.heading)
        self._trace(
            "drone_incidental_scan_started",
            directive_id=execution.directive.directive_id,
            source=route_source,
            phase=execution.phase,
            position=runtime.position,
            heading=candidate.heading,
            minimum_scan_sequence=minimum_sequence,
            retained_route_points=len(retained_route),
            retained_route_distance=self._path_distance(
                runtime.position,
                tuple(retained_route),
                len(retained_route),
            ),
            route_target=pending.route_target,
            path_status=path_status,
            deadline=pending.deadline,
        )
        return True

    def _incidental_candidate_is_reopenable(
        self,
        candidate: IncidentalScanCandidate,
    ) -> bool:
        """Suppress a matching attempt until support or its gateway moves."""
        matching = next((
            attempt
            for attempt in reversed(self._incidental_attempt_memory)
            if (
                attempt.signature == candidate.signature
                or (
                    attempt.signature.wall_bin == candidate.signature.wall_bin
                    and bool(
                        set(attempt.signature.support_bins)
                        & set(candidate.signature.support_bins)
                    )
                )
            )
        ), None)
        if matching is None:
            return True
        new_support = len(candidate.cells - matching.cells)
        minimum_new_support = max(
            8,
            int(math.ceil(len(matching.cells) * 0.25)),
        )
        gateway_movement = math.dist(candidate.gateway, matching.gateway)
        minimum_gateway_movement = max(
            float(2 * self.frontier_stride),
            0.1 * self._sensor_range(),
        )
        return bool(
            new_support >= minimum_new_support
            or gateway_movement + 1e-9 >= minimum_gateway_movement
        )

    def _remember_incidental_attempt(
        self,
        candidate: IncidentalScanCandidate,
    ) -> None:
        self._incidental_attempt_memory.append(_IncidentalPocketAttempt(
            signature=candidate.signature,
            cells=candidate.cells,
            gateway=candidate.gateway,
            attempted_at=self._simulation_time(),
        ))
        if len(self._incidental_attempt_memory) > 256:
            del self._incidental_attempt_memory[:-256]

    def _record_incidental_pocket_revisit(
        self,
        target: Position,
        path: Iterable[Position],
        *,
        source: str,
    ) -> None:
        """Trace later ordinary DFS work that targets an attempted envelope."""
        radius = float(2 * self.frontier_stride)
        current = self.drone.snapshot().position
        route = tuple(path)
        for attempt in self._incidental_attempt_memory:
            if attempt.revisited or not any(
                math.dist(target, cell) <= radius + 1e-9
                for cell in attempt.cells
            ):
                continue
            attempt.revisited = True
            self._trace(
                "drone_incidental_pocket_revisited",
                source=source,
                target=target,
                signature=self._incidental_signature_fields(
                    attempt.signature
                ),
                gateway=attempt.gateway,
                actual_revisit_route_distance=self._path_distance(
                    current,
                    route,
                    len(route),
                ),
                attempted_at=attempt.attempted_at,
            )

    def _record_focused_frontier_batch_provisional_revisit(
        self,
        target: Position,
        path: Iterable[Position],
        *,
        directive_id: int,
        source: str,
    ) -> None:
        """Trace a later directive returning to lease-serviced geometry."""
        radius = float(2 * self.frontier_stride)
        current = self.drone.snapshot().position
        route = tuple(path)
        for attempt in self._focused_frontier_batch_provisional_memory:
            if (
                attempt.revisited
                or attempt.directive_id == int(directive_id)
                or not any(
                    math.dist(target, cell) <= radius + 1e-9
                    for cell in attempt.cells
                )
            ):
                continue
            attempt.revisited = True
            self._trace(
                "drone_focused_frontier_batch_frontier_revisited",
                source=source,
                target=target,
                source_directive_id=attempt.directive_id,
                current_directive_id=int(directive_id),
                observation_id=attempt.observation_id,
                actual_revisit_route_distance=self._path_distance(
                    current,
                    route,
                    len(route),
                ),
                serviced_at=attempt.serviced_at,
            )

    @staticmethod
    def _incidental_signature_fields(
        signature: IncidentalPocketSignature,
    ) -> dict[str, Any]:
        return {
            "support_bins": signature.support_bins,
            "gateway_bin": signature.gateway_bin,
            "wall_bin": signature.wall_bin,
        }

    def _advance_pending_incidental_scan(
        self,
        execution: _CoordinationExecution,
    ) -> None:
        """Accept only a fresh completion at the requested exact scan pose."""
        pending = self._pending_incidental_scan
        if pending is None:
            return
        sensor = getattr(self.drone, "sensor_controller", None)
        completion = getattr(sensor, "last_completed_scan", None)
        expected_pose = (
            int(pending.position[0]),
            int(pending.position[1]),
            round(float(pending.heading) % 360.0, 3),
        )
        runtime = self.drone.snapshot()
        current_pose = (
            int(runtime.position[0]),
            int(runtime.position[1]),
            round(float(runtime.heading_deg) % 360.0, 3),
        )
        if (
            completion is not None
            and completion.pose == expected_pose
            and current_pose == expected_pose
            and int(completion.sequence) > pending.minimum_scan_sequence
        ):
            self.rebuild_frontiers(
                stride=self.frontier_stride,
                confidence_threshold=self.frontier_confidence_threshold,
            )
            self._finish_pending_incidental_scan(
                execution,
                pending,
                outcome="completed",
                completion=completion,
            )
            return
        if self._simulation_time() + 1e-9 >= pending.deadline:
            self._finish_pending_incidental_scan(
                execution,
                pending,
                outcome="timed_out",
            )

    def _finish_pending_incidental_scan(
        self,
        execution: _CoordinationExecution,
        pending: _PendingIncidentalTransitScan,
        *,
        outcome: str,
        completion: Any | None = None,
        interruption_reason: str | None = None,
    ) -> None:
        now = self._simulation_time()
        wait_seconds = max(
            0.0,
            min(now, pending.deadline) - pending.requested_at,
        )
        execution.incidental_wait_seconds += wait_seconds
        closed_cells = 0
        if outcome == "completed":
            execution.incidental_completions += 1
            closed_cells = self._closed_incidental_pocket_cells(
                pending.candidate.cells
            )
            execution.incidental_pocket_cells_closed += closed_cells
        elif outcome == "timed_out":
            execution.incidental_timeouts += 1
        runtime = self.drone.snapshot()
        self._trace(
            "drone_incidental_scan_finished",
            directive_id=pending.directive_id,
            source=pending.route_source,
            phase=pending.original_phase,
            outcome=outcome,
            interruption_reason=interruption_reason,
            position=runtime.position,
            heading=runtime.heading_deg,
            requested_position=pending.position,
            requested_heading=pending.heading,
            scan_sequence=(None if completion is None else completion.sequence),
            wait_seconds=wait_seconds,
            requested_rotation=pending.candidate.heading_delta,
            newly_known_cells=(
                0 if completion is None else completion.newly_known_cells
            ),
            confidence_gain=(
                0.0 if completion is None else completion.confidence_gain
            ),
            original_pocket_cells_closed=closed_cells,
            baseline_slam_version=pending.baseline_slam_version,
            completed_slam_version=self.drone.slam_map.version,
        )
        self._pending_incidental_scan = None
        if outcome in {"completed", "timed_out"}:
            self._resume_incidental_route(execution, pending)

    def _closed_incidental_pocket_cells(
        self,
        cells: Iterable[Position],
    ) -> int:
        slam = self.drone.slam_map.snapshot(point_limit=0)
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        offset_x, offset_y = (int(value) for value in slam.origin)
        height, width = occupancy.shape
        closed = 0
        for x, y in cells:
            local_x = int(x) - offset_x
            local_y = int(y) - offset_y
            if not (0 <= local_x < width and 0 <= local_y < height):
                continue
            if (
                occupancy[local_y, local_x] != UNKNOWN
                and confidence[local_y, local_x]
                >= self.frontier_confidence_threshold
            ):
                closed += 1
        return closed

    def _resume_incidental_route(
        self,
        execution: _CoordinationExecution,
        pending: _PendingIncidentalTransitScan,
    ) -> None:
        route = pending.retained_route
        remaining_distance = self._path_distance(
            pending.position,
            route,
            len(route),
        )
        if not route:
            self._trace(
                "drone_incidental_route_resumed",
                directive_id=pending.directive_id,
                source=pending.route_source,
                disposition="retained",
                remaining_distance=0.0,
                interruption_reason=None,
            )
            return

        interrupted_by_share = False

        def stop_for_shared_target() -> bool:
            nonlocal interrupted_by_share
            interrupted_by_share = (
                self._component_route_invalidated_after_share(execution)
            )
            return interrupted_by_share

        followed = self._follow_path(
            route,
            source=pending.route_source,
            stop_when=stop_for_shared_target,
            stop_reason="shared_slam_invalidated",
            incidental_transit=True,
            path_target=pending.route_target,
            path_status=pending.path_status,
        )
        retained = bool(
            followed or self._pending_incidental_scan is not None
        )
        if (
            pending.route_source == "component_dfs_reposition_astar"
            and self._pending_incidental_scan is None
        ):
            self._trace(
                "drone_dfs_reposition_path",
                directive_id=pending.directive_id,
                start=pending.position,
                target=pending.route_target,
                observation_target=pending.route_target,
                path_status=pending.path_status,
                path_length=len(route),
                reached=(
                    followed
                    and self.drone.snapshot().position
                    == pending.route_target
                ),
                resumed_after_incidental_scan=True,
            )
        self._trace(
            "drone_incidental_route_resumed",
            directive_id=pending.directive_id,
            source=pending.route_source,
            disposition="retained" if retained else "replanned",
            remaining_distance=remaining_distance,
            interruption_reason=(
                "shared_slam_invalidated"
                if interrupted_by_share
                else None if retained else "retained_edge_invalid"
            ),
        )

    def _cancel_pending_incidental_scan(self, reason: str) -> None:
        pending = self._pending_incidental_scan
        if pending is None:
            return
        execution = self._coordination_execution
        if (
            execution is not None
            and execution.directive.directive_id == pending.directive_id
        ):
            self._finish_pending_incidental_scan(
                execution,
                pending,
                outcome="cancelled",
                interruption_reason=str(reason),
            )
        else:
            self._pending_incidental_scan = None
            self._trace(
                "drone_incidental_scan_finished",
                directive_id=pending.directive_id,
                source=pending.route_source,
                phase=pending.original_phase,
                outcome="cancelled",
                interruption_reason=str(reason),
                position=self.drone.snapshot().position,
                heading=self.drone.snapshot().heading_deg,
                requested_position=pending.position,
                requested_heading=pending.heading,
                scan_sequence=None,
                wait_seconds=max(
                    0.0,
                    min(self._simulation_time(), pending.deadline)
                    - pending.requested_at,
                ),
                requested_rotation=pending.candidate.heading_delta,
                newly_known_cells=0,
                confidence_gain=0.0,
                original_pocket_cells_closed=0,
            )
        self._trace(
            "drone_incidental_route_resumed",
            directive_id=pending.directive_id,
            source=pending.route_source,
            disposition="replanned",
            remaining_distance=self._path_distance(
                pending.position,
                pending.retained_route,
                len(pending.retained_route),
            ),
            interruption_reason=str(reason),
        )

    def _advance_coordination_scan(
        self,
        execution: _CoordinationExecution,
    ) -> bool:
        scan = execution.scan
        if scan is None:
            return True
        if scan.complete:
            return True
        heading = scan.current_heading
        if heading is None:
            return True
        sensor = getattr(self.drone, "sensor_controller", None)
        completion = getattr(sensor, "last_completed_scan", None)
        expected_pose = (
            int(scan.position[0]),
            int(scan.position[1]),
            round(float(heading) % 360.0, 3),
        )
        if scan.requested_at is None:
            if completion is not None and completion.pose == expected_pose:
                self._accept_coordination_scan_heading(
                    execution, scan, completion,
                )
                return scan.complete
            scan.minimum_scan_sequence = (
                -1 if completion is None else int(completion.sequence)
            )
            scan.requested_at = self._simulation_time()
            self.drone.runtime_state.reorient(heading)
            self._trace(
                "drone_scan_heading_requested",
                directive_id=execution.directive.directive_id,
                directive_kind=execution.directive.kind.value,
                position=scan.position,
                heading=heading,
                minimum_scan_sequence=scan.minimum_scan_sequence,
            )
            return False
        if (
            completion is not None
            and completion.pose == expected_pose
            and completion.sequence > scan.minimum_scan_sequence
        ):
            self._accept_coordination_scan_heading(
                execution, scan, completion,
            )
            return scan.complete
        now = self._simulation_time()
        if now - scan.requested_at < _PENDING_FRONTIER_SCAN_TIMEOUT_SECONDS:
            return False
        scan.timed_out_headings.append(heading)
        execution.timed_out_scan_headings.append(heading)
        scan.heading_index += 1
        scan.requested_at = None
        self._trace(
            "drone_scan_heading_timed_out",
            directive_id=execution.directive.directive_id,
            directive_kind=execution.directive.kind.value,
            position=scan.position,
            heading=heading,
            timeout_seconds=_PENDING_FRONTIER_SCAN_TIMEOUT_SECONDS,
        )
        return scan.complete

    def _accept_coordination_scan_heading(
        self,
        execution: _CoordinationExecution,
        scan: ScanPlanProgress,
        completion: Any,
    ) -> None:
        heading = scan.current_heading
        if heading is None:
            return
        newly_known = max(0, int(completion.newly_known_cells))
        confidence_gain = max(0.0, float(completion.confidence_gain))
        scan.completed_headings.append(heading)
        scan.newly_known_cells += newly_known
        scan.confidence_gain += confidence_gain
        execution.completed_scan_headings.append(heading)
        execution.sensor_newly_known_cells += newly_known
        execution.sensor_confidence_gain += confidence_gain
        scan.heading_index += 1
        scan.requested_at = None
        self._trace(
            "drone_scan_heading_completed",
            directive_id=execution.directive.directive_id,
            directive_kind=execution.directive.kind.value,
            position=scan.position,
            heading=heading,
            scan_sequence=completion.sequence,
            sensor_newly_known_cells=newly_known,
            sensor_confidence_gain=confidence_gain,
        )

    def _finish_coordination_execution(
        self,
        execution: _CoordinationExecution,
    ) -> None:
        self._cancel_pending_incidental_scan("directive_completed")
        directive = execution.directive
        if directive.kind == DirectiveKind.COMPONENT_BATCH:
            self._finish_batch_execution(execution)
            return
        claim_token = (
            None if directive.claim is None else directive.claim.token
        )
        report_id = self._next_coordination_report_id()
        causal = ()
        if execution.root_component_id is not None and execution.causal_successors:
            causal = (CausalTransition(
                predecessor_id=execution.root_component_id,
                successor_cells=tuple(dict.fromkeys(
                    execution.causal_successors
                )),
                report_id=report_id,
                visited_successor_anchors=tuple(dict.fromkeys(
                    execution.visited_successor_anchors or ()
                )),
            ),)
        suspension = self._build_task_suspension(execution)
        self._coordination_report = CoordinationReport(
            report_id=report_id,
            directive_id=directive.directive_id,
            kind=directive.kind,
            round_id=directive.round_id,
            task_id=(None if directive.task is None else directive.task.task_id),
            component_id=(
                None if directive.task is None
                else directive.task.component_id
            ),
            claim_token=claim_token,
            work_unit_outcomes=tuple(execution.work_unit_outcomes or ()),
            causal_transitions=causal,
            suspension=suspension,
            completed_scan_headings=tuple(
                execution.completed_scan_headings or ()
            ),
            timed_out_scan_headings=tuple(
                execution.timed_out_scan_headings or ()
            ),
            sensor_newly_known_cells=execution.sensor_newly_known_cells,
            sensor_confidence_gain=execution.sensor_confidence_gain,
            outbound_actual_path=execution.outbound_actual_path,
            return_actual_path=execution.return_actual_path,
            return_path_source=execution.return_path_source,
            outbound_distance=execution.root_outbound_distance,
            service_distance=execution.service_distance,
            return_distance=self._polyline_distance(
                execution.return_actual_path
            ),
        )
        self._trace(
            "drone_component_directive_completed",
            directive_id=directive.directive_id,
            directive_kind=directive.kind.value,
            report_id=report_id,
            claim_token=claim_token,
            completed_scan_headings=tuple(
                execution.completed_scan_headings or ()
            ),
            timed_out_scan_headings=tuple(
                execution.timed_out_scan_headings or ()
            ),
            sensor_newly_known_cells=execution.sensor_newly_known_cells,
            sensor_confidence_gain=execution.sensor_confidence_gain,
            outbound_distance=execution.root_outbound_distance,
            service_distance=execution.service_distance,
            return_distance=self._polyline_distance(
                execution.return_actual_path
            ),
            dfs_leaf_count=(
                0 if execution.dfs is None
                else len(execution.dfs.leaf_geometry)
            ),
            suspended=suspension is not None,
            suspension_reason=execution.suspension_reason,
            incidental_candidates=execution.incidental_candidates,
            incidental_attempts=execution.incidental_attempts,
            incidental_completions=execution.incidental_completions,
            incidental_timeouts=execution.incidental_timeouts,
            incidental_wait_seconds=execution.incidental_wait_seconds,
            incidental_requested_rotation=(
                execution.incidental_requested_rotation
            ),
            incidental_predicted_support=(
                execution.incidental_predicted_support
            ),
            incidental_pocket_cells_closed=(
                execution.incidental_pocket_cells_closed
            ),
        )
        self._coordination_execution = None

    def _finish_batch_execution(
        self,
        execution: _CoordinationExecution,
    ) -> None:
        directive = execution.directive
        report_id = self._next_coordination_report_id()
        member_reports: list[BatchMemberReport] = []
        for snapshot in execution.batch_completed_members or ():
            causal = ()
            if snapshot.causal_successors:
                causal = (CausalTransition(
                    predecessor_id=snapshot.member.task.component_id,
                    successor_cells=snapshot.causal_successors,
                    report_id=report_id,
                    visited_successor_anchors=(
                        snapshot.visited_successor_anchors
                    ),
                ),)
            member_reports.append(BatchMemberReport(
                task_id=snapshot.member.task.task_id,
                component_id=snapshot.member.task.component_id,
                component_revision=snapshot.member.task.component_revision,
                claim_token=snapshot.member.claim.token,
                disposition=snapshot.disposition,
                work_unit_outcomes=snapshot.work_unit_outcomes,
                causal_transitions=causal,
                suspension=snapshot.suspension,
            ))
        lease = directive.spatial_lease
        total_outbound = sum(
            item.outbound_distance
            for item in execution.batch_completed_members or ()
        )
        total_service = sum(
            item.service_distance
            for item in execution.batch_completed_members or ()
        )
        self._coordination_report = CoordinationReport(
            report_id=report_id,
            directive_id=directive.directive_id,
            kind=directive.kind,
            lease_id=None if lease is None else lease.lease_id,
            batch_member_reports=tuple(member_reports),
            provisional_observations=tuple(
                execution.batch_provisional_observations or ()
            ),
            completed_scan_headings=tuple(
                execution.completed_scan_headings or ()
            ),
            timed_out_scan_headings=tuple(
                execution.timed_out_scan_headings or ()
            ),
            sensor_newly_known_cells=execution.sensor_newly_known_cells,
            sensor_confidence_gain=execution.sensor_confidence_gain,
            outbound_actual_path=execution.outbound_actual_path,
            return_actual_path=execution.return_actual_path,
            return_path_source=execution.return_path_source,
            outbound_distance=total_outbound,
            service_distance=total_service,
            return_distance=self._polyline_distance(
                execution.return_actual_path
            ),
        )
        self._trace(
            "drone_component_directive_completed",
            directive_id=directive.directive_id,
            directive_kind=directive.kind.value,
            report_id=report_id,
            lease_id=None if lease is None else lease.lease_id,
            member_task_ids=tuple(
                item.task_id for item in member_reports
            ),
            claim_tokens=tuple(
                item.claim_token for item in member_reports
            ),
            member_dispositions=tuple(
                item.disposition for item in member_reports
            ),
            provisional_observation_count=len(
                execution.batch_provisional_observations or ()
            ),
            outbound_distance=total_outbound,
            service_distance=total_service,
            return_distance=self._polyline_distance(
                execution.return_actual_path
            ),
            bound_reason=execution.batch_return_reason,
        )
        self._coordination_execution = None

    def _energy_requires_return(
        self,
        execution: _CoordinationExecution,
    ) -> bool:
        callback = self.dependencies.exploration_energy_return
        if not callable(callback):
            return False
        current = self.drone.snapshot().position
        rover = self._rendezvous_position()
        if rover is None:
            return False
        route = self._compute_path(current, rover)
        route_home_cost = self._path_distance(
            current,
            tuple(route.path),
            len(route.path),
        )
        if not route.path and current != rover:
            route_home_cost = math.dist(current, rover)
        next_action_cost = 1.0
        if execution.directive.kind == DirectiveKind.COMPONENT_BATCH:
            next_action_cost = max(
                1.0,
                sum(
                    member.estimated_service_cost
                    for member in (
                        *((execution.batch_current_member,) if
                          execution.batch_current_member is not None else ()),
                        *(execution.batch_pending_members or ()),
                    )
                ),
            )
        elif execution.directive.task is not None:
            next_action_cost = max(
                1.0,
                float(execution.directive.task.estimated_effort),
            )
        decision = callback(
            self.drone.id,
            route_home_cost,
            next_action_cost,
            0.0,
        )
        if not isinstance(decision, EnergyReturnDecision):
            return False
        if not decision.must_return:
            return False
        self._trace(
            "drone_energy_return_required",
            directive_id=execution.directive.directive_id,
            directive_kind=execution.directive.kind.value,
            remaining_energy=decision.state.remaining_energy,
            route_home_cost=decision.route_home_cost,
            next_action_cost=decision.next_action_cost,
            safety_reserve=decision.safety_reserve,
        )
        self._prepare_execution_return(
            execution,
            reason="energy_reserve",
            energy_state=decision.state,
        )
        return True

    def _coordination_energy_checkpoint_due(
        self,
        execution: _CoordinationExecution,
    ) -> bool:
        """Check reserves once before each new translation or scan action."""
        if execution.phase == "transit":
            key: tuple[Any, ...] = (
                "transit",
                execution.transit_node_id,
                self.drone.snapshot().position,
            )
        elif execution.phase == "scanning":
            scan = execution.scan
            if (
                scan is None
                or scan.complete
                or scan.requested_at is not None
            ):
                return False
            key = ("scan", scan.position, scan.heading_index)
        elif (
            execution.directive.kind == DirectiveKind.COMPONENT_BATCH
            and execution.phase == "repositioning"
        ):
            key = (
                "repositioning",
                execution.reposition_target,
                self.drone.snapshot().position,
            )
        else:
            return False
        if execution.energy_checkpoint_key == key:
            return False
        execution.energy_checkpoint_key = key
        return True

    def _prepare_execution_return(
        self,
        execution: _CoordinationExecution,
        *,
        reason: str,
        energy_state: EnergyState | None = None,
    ) -> None:
        """Suspend bounded work and retain an exact breadcrumb to the rover."""
        self._cancel_pending_incidental_scan(reason)
        if execution.directive.kind == DirectiveKind.COMPONENT_BATCH:
            member_failure_reasons = {
                "route_unreachable",
                "route_progress_limit",
                "dfs_reposition_unreachable",
                "sensor_timeout",
                "route_planning_budget",
            }
            if reason in member_failure_reasons:
                execution.suspension_reason = str(reason)
                execution.suspension_position = self.drone.snapshot().position
                execution.suspension_energy_state = energy_state
                self._capture_batch_member(
                    execution,
                    disposition=(
                        "sensor_timeout"
                        if reason == "sensor_timeout"
                        else "budget_exhausted" if reason == "route_planning_budget"
                        else "unreachable"
                    ),
                )
                if self._start_next_batch_member(execution):
                    return
                self._prepare_batch_return(execution)
                return
            self._prepare_batch_return(
                execution,
                reason=reason,
                energy_state=energy_state,
            )
            return
        execution.outbound_actual_path = self._path_history_since(
            execution.path_start_index
        )
        total_outward_distance = self._polyline_distance(
            execution.outbound_actual_path
        )
        if execution.root_outbound_distance <= 1e-9:
            execution.root_outbound_distance = total_outward_distance
            execution.service_distance = 0.0
        else:
            execution.service_distance = max(
                0.0,
                total_outward_distance - execution.root_outbound_distance,
            )
        if execution.directive.kind == DirectiveKind.COMPONENT_TASK:
            execution.suspension_reason = str(reason)
            execution.suspension_position = self.drone.snapshot().position
            execution.suspension_energy_state = energy_state or EnergyState(
                remaining_energy=float(self.drone.snapshot().battery),
                capacity=100.0,
                unlimited=True,
            )
            execution.pending_return_breadcrumb = (
                self._component_return_breadcrumb(execution)
            )
        else:
            execution.pending_return_breadcrumb = tuple(reversed(
                execution.outbound_actual_path
            ))
        execution.return_path_start_index = max(
            0, len(self.drone.snapshot().path_history) - 1,
        )
        execution.phase = "returning"
        self._trace(
            "drone_component_task_suspended"
            if execution.directive.kind == DirectiveKind.COMPONENT_TASK
            else "drone_probe_aborted",
            directive_id=execution.directive.directive_id,
            reason=reason,
            position=self.drone.snapshot().position,
            breadcrumb_point_count=len(execution.pending_return_breadcrumb),
        )

    def _capture_batch_member(
        self,
        execution: _CoordinationExecution,
        *,
        disposition: str,
    ) -> None:
        member = execution.batch_current_member
        if member is None:
            return
        start_index = (
            execution.batch_member_path_start_index
            if execution.batch_member_path_start_index is not None
            else execution.path_start_index
        )
        path = self._path_history_since(start_index)
        distance = self._polyline_distance(path)
        if execution.root_outbound_distance <= 1e-9:
            execution.root_outbound_distance = distance
        execution.service_distance = max(
            0.0, distance - execution.root_outbound_distance,
        )
        suspension = self._build_task_suspension(execution)
        finished_at = self._simulation_time()
        started_at = execution.batch_member_started_at
        snapshot = _BatchMemberSnapshot(
            member=member,
            disposition=disposition,
            work_unit_outcomes=tuple(execution.work_unit_outcomes or ()),
            causal_successors=tuple(dict.fromkeys(
                execution.causal_successors or (),
            )),
            visited_successor_anchors=tuple(dict.fromkeys(
                execution.visited_successor_anchors or (),
            )),
            suspension=suspension,
            outbound_actual_path=path,
            outbound_distance=execution.root_outbound_distance,
            service_distance=execution.service_distance,
            service_seconds=max(
                0.0,
                finished_at - (finished_at if started_at is None else started_at),
            ),
        )
        if execution.batch_completed_members is None:
            execution.batch_completed_members = []
        execution.batch_completed_members.append(snapshot)
        if execution.dfs is not None:
            execution.batch_completed_dfs_nodes += (
                execution.batch_current_dfs_nodes
            )
        self._trace(
            "drone_focused_frontier_batch_member_finished",
            directive_id=execution.directive.directive_id,
            task_id=member.task.task_id,
            component_id=member.task.component_id,
            claim_token=member.claim.token,
            disposition=disposition,
            work_unit_outcome_count=len(snapshot.work_unit_outcomes),
            outbound_distance=snapshot.outbound_distance,
            service_distance=snapshot.service_distance,
            service_seconds=snapshot.service_seconds,
            suspended=suspension is not None,
        )
        execution.batch_current_member = None

    def _prepare_batch_return(
        self,
        execution: _CoordinationExecution,
        *,
        reason: str | None = None,
        energy_state: EnergyState | None = None,
    ) -> None:
        """Freeze every unfinished member and make one physical return."""
        if execution.batch_current_member is not None:
            completed_ids = {
                outcome.work_unit_id
                for outcome in execution.work_unit_outcomes or ()
            }
            member_complete = set(
                execution.batch_current_member.claim.work_unit_ids
            ) <= completed_ids
            execution.suspension_reason = None if member_complete else reason
            execution.suspension_position = self.drone.snapshot().position
            execution.suspension_energy_state = energy_state
            self._capture_batch_member(
                execution,
                disposition=(
                    "complete"
                    if reason is None or member_complete
                    else "budget_exhausted"
                ),
            )
        pending = execution.batch_pending_members or []
        if reason is not None:
            state = energy_state or EnergyState(
                remaining_energy=float(self.drone.snapshot().battery),
                capacity=100.0,
                unlimited=True,
            )
            slam_version = int(
                self.drone.slam_map.snapshot(point_limit=0).version
            )
            for member in tuple(pending):
                suspension = TaskSuspension(
                    task_id=member.task.task_id,
                    claim_token=member.claim.token,
                    drone_id=int(self.drone.id),
                    reason=str(reason),
                    dfs_stack=(),
                    remaining_work_unit_ids=member.claim.work_unit_ids,
                    position_at_suspension=self.drone.snapshot().position,
                    actual_return_path=(),
                    return_path_source="batch_not_started",
                    local_slam_version=slam_version,
                    energy_state=state,
                )
                if execution.batch_completed_members is None:
                    execution.batch_completed_members = []
                execution.batch_completed_members.append(_BatchMemberSnapshot(
                    member=member,
                    disposition="budget_exhausted",
                    work_unit_outcomes=(),
                    causal_successors=(),
                    visited_successor_anchors=(),
                    suspension=suspension,
                    outbound_actual_path=(),
                    outbound_distance=0.0,
                    service_distance=0.0,
                    service_seconds=0.0,
                ))
        pending.clear()
        execution.batch_return_reason = reason
        execution.outbound_actual_path = self._path_history_since(
            execution.path_start_index
        )
        execution.pending_return_breadcrumb = tuple(reversed(
            execution.outbound_actual_path
        ))
        execution.return_path_start_index = max(
            0, len(self.drone.snapshot().path_history) - 1,
        )
        execution.phase = "returning"
        if reason is not None:
            elapsed = (
                0.0 if execution.batch_started_at is None
                else self._simulation_time() - execution.batch_started_at
            )
            usage, limit = {
                "service_time_limit": (
                    elapsed,
                    execution.directive.maximum_service_seconds,
                ),
                "dfs_node_limit": (
                    execution.batch_completed_dfs_nodes
                    + execution.batch_current_dfs_nodes,
                    execution.directive.maximum_total_dfs_nodes,
                ),
                "energy_reserve": (None, None),
            }.get(reason, (None, None))
            self._trace(
                "drone_focused_frontier_batch_bound_reached",
                directive_id=execution.directive.directive_id,
                bound=reason,
                usage=usage,
                limit=limit,
                elapsed_seconds=elapsed,
                detour_distance=execution.batch_detour_distance,
                current_member_task_id=(
                    None if execution.batch_current_member is None
                    else execution.batch_current_member.task.task_id
                ),
                completed_member_count=len(
                    execution.batch_completed_members or ()
                ),
                suspended_member_count=sum(
                    item.suspension is not None
                    for item in execution.batch_completed_members or ()
                ),
            )

    def _component_return_breadcrumb(
        self,
        execution: _CoordinationExecution,
    ) -> tuple[Position, ...]:
        dfs = execution.dfs
        if dfs is None:
            return tuple(reversed(execution.outbound_actual_path))
        segments: list[Position] = []
        partial: tuple[Position, ...] = ()
        if execution.transit_path_start_index is not None:
            partial = self._path_history_since(
                execution.transit_path_start_index
            )
            segments.extend(reversed(partial))
        for index, frame in enumerate(reversed(dfs.frames)):
            if index == 0 and partial and not frame.outbound_actual_path:
                continue
            segments.extend(reversed(frame.outbound_actual_path))
        return tuple(segments)

    def _build_task_suspension(
        self,
        execution: _CoordinationExecution,
    ) -> TaskSuspension | None:
        directive = execution.directive
        if (
            execution.suspension_reason is None
            or directive.task is None
            or directive.claim is None
        ):
            return None
        completed = {
            outcome.work_unit_id
            for outcome in execution.work_unit_outcomes or ()
        }
        remaining = tuple(
            unit_id for unit_id in directive.claim.work_unit_ids
            if unit_id not in completed
        )
        frames = tuple(
            DFSFrame(
                task_id=directive.task.task_id,
                component_id=directive.task.component_id,
                component_revision=directive.task.component_revision,
                entry_position=frame.node.anchor_position,
                outbound_actual_path=frame.outbound_actual_path,
                pending_work_unit_ids=remaining,
                pending_child_task_ids=(),
                active_work_unit_id=(
                    execution.root_work_unit_id
                    if frame.node.local_id == 0
                    else None
                ),
            )
            for frame in (() if execution.dfs is None else execution.dfs.frames)
        )
        slam_version = int(
            self.drone.slam_map.snapshot(point_limit=0).version
        )
        return TaskSuspension(
            task_id=directive.task.task_id,
            claim_token=directive.claim.token,
            drone_id=int(self.drone.id),
            reason=execution.suspension_reason,
            dfs_stack=frames,
            remaining_work_unit_ids=remaining,
            position_at_suspension=(
                execution.suspension_position
                or self.drone.snapshot().position
            ),
            actual_return_path=execution.return_actual_path,
            return_path_source=execution.return_path_source,
            local_slam_version=slam_version,
            energy_state=(
                execution.suspension_energy_state
                or EnergyState(100.0, 100.0, unlimited=True)
            ),
        )

    def _return_to_coordination_point(
        self,
        target: Position,
        *,
        breadcrumb: Iterable[Position] = (),
        execution: _CoordinationExecution | None = None,
        dock_during_check_in: bool = False,
    ) -> tuple[bool, bool]:
        """Reach a remembered endpoint or stop at verified rover contact.

        Return ``(reached, encountered_rover)``. Reporting contact reserves a
        rover stop; an empty check-in atomically docks at the contact instead.
        """
        current = self.drone.snapshot().position
        history_start = (
            max(0, len(self.drone.snapshot().path_history) - 1)
            if execution is None
            else (
                execution.return_path_start_index
                if execution.return_path_start_index is not None
                else max(0, len(self.drone.snapshot().path_history) - 1)
            )
        )
        if execution is not None and execution.return_path_start_index is None:
            execution.return_path_start_index = history_start
        request_stop = self.dependencies.request_exploration_report_stop
        request_dock = self.dependencies.request_exploration_dock
        can_report = execution is not None or self._coordination_report is not None
        encountered = False
        initial_endpoint = self._rendezvous_position()
        retargeted = False
        dock_result: CoordinationResult | None = None

        def stop_for_coordination_contact() -> bool:
            nonlocal dock_result, encountered, retargeted
            if can_report and callable(request_stop):
                encountered = bool(request_stop(self.drone.id))
            elif dock_during_check_in and callable(request_dock):
                candidate = request_dock(self.drone.id)
                if isinstance(candidate, CoordinationResult) and candidate.arrived:
                    dock_result = candidate
                    encountered = True
            if not encountered and initial_endpoint == target:
                latest_endpoint = self._rendezvous_position()
                if latest_endpoint is not None and latest_endpoint != target:
                    retargeted = True
                    self._trace("drone_component_rendezvous_retargeted",
                                previous_endpoint=target, endpoint=latest_endpoint,
                                reason="physical_movement_evidence")
            return encountered or retargeted

        def return_stop_reason():
            return "rendezvous_updated" if retargeted else ("report" if can_report else "dock")

        def record_return(source: str) -> None:
            if execution is not None:
                execution.return_actual_path = self._path_history_since(
                    history_start
                )
                execution.return_path_source = source

        if stop_for_coordination_contact() and not retargeted:
            record_return("rover_encounter")
            if dock_result is not None:
                self._apply_coordination_result(dock_result)
            return True, True
        if retargeted:
            return False, False
        if current == target:
            return True, False
        highway_path = self._component_check_in_highway_route(
            current,
            target,
        )
        if highway_path:
            followed = self._follow_path(
                highway_path,
                source=f"component_checkin_{self._return_route_source}",
                stop_when=(
                    stop_for_coordination_contact
                    if can_report or dock_during_check_in
                    else None
                ),
                stop_reason=return_stop_reason,
            )
            if retargeted:
                return False, False
            if encountered:
                record_return("rover_encounter")
                if dock_result is not None:
                    self._apply_coordination_result(dock_result)
                return True, True
            if followed and self.drone.snapshot().position == target:
                record_return(self._return_route_source)
                return True, False
            current = self.drone.snapshot().position
        result = self._compute_path(current, target)
        if result.path:
            followed = self._follow_path(
                result.path,
                source="component_checkin_astar",
                stop_when=(
                    stop_for_coordination_contact
                    if can_report or dock_during_check_in
                    else None
                ),
                stop_reason=return_stop_reason,
            )
            if retargeted:
                return False, False
            if encountered:
                record_return("rover_encounter")
                if dock_result is not None:
                    self._apply_coordination_result(dock_result)
                return True, True
            if followed and self.drone.snapshot().position == target:
                record_return("astar")
                return True, False
            if result.status == PATH_PARTIAL_LIMIT and followed:
                return False, False
        fallback = tuple(breadcrumb)
        if fallback:
            followed = self._follow_path(
                fallback,
                source="component_checkin_breadcrumb_fallback",
                stop_when=(
                    stop_for_coordination_contact
                    if can_report or dock_during_check_in
                    else None
                ),
                stop_reason=return_stop_reason,
            )
            if retargeted:
                return False, False
            if encountered:
                record_return("rover_encounter")
                if dock_result is not None:
                    self._apply_coordination_result(dock_result)
                return True, True
            if followed and self.drone.snapshot().position == target:
                record_return("breadcrumb_fallback")
                return True, False
        return False, False

    def _component_check_in_highway_route(
        self,
        current: Position,
        target: Position,
    ) -> tuple[Position, ...]:
        """Compare physically received advice with a bounded local alternative."""
        comparison_started = time.perf_counter()
        self._return_route_source = "highway"
        if self.highway_mode == "off":
            return ()
        direct_distance = math.dist(current, target)
        minimum_distance = (
            self._sensor_range()
            * self.highway_minimum_route_sensor_ranges
        )
        snapshot = self._highway_snapshot
        local_version = self.drone.slam_map.version
        if snapshot is None or direct_distance + 1e-9 < minimum_distance:
            self._trace(
                "drone_highway_route_evaluated",
                mode=self.highway_mode,
                status=("unavailable" if snapshot is None else "short_route"),
                selected=False,
                fallback_reason=(
                    "snapshot_unavailable"
                    if snapshot is None
                    else "below_minimum_distance"
                ),
                highway_version=(
                    None if snapshot is None else snapshot.version
                ),
                local_slam_version=local_version,
                direct_distance=direct_distance,
                route_distance=None,
                route_circuity=None,
                elapsed_ms=0.0,
                expanded_nodes=0,
                graph_edges=0,
            )
            return ()
        route = snapshot.route(
            current,
            target,
            maximum_query_ms=self.highway_maximum_query_ms,
            maximum_connector_expansions=(
                self.highway_maximum_connector_expansions
            ),
        )
        route_distance = route.cost if route.complete else math.inf
        circuity = self._route_circuity(route_distance, direct_distance)
        endpoints_valid = bool(
            route.path
            and route.path[0] == tuple(current)
            and route.path[-1] == tuple(target)
        )
        locally_valid = bool(
            endpoints_valid and self._highway_path_locally_valid(route.path)
        )
        eligible = bool(
            route.complete
            and locally_valid
            and circuity <= self.highway_maximum_route_circuity + 1e-9
        )
        selected = eligible and self.highway_mode == "active"
        chosen_path = route.path if selected else ()
        shorter_local = False
        if selected:
            from navigation.return_route import local_return_alternative
            alternative = local_return_alternative(
                self.drone.slam_map.snapshot(point_limit=0), current, target,
                confidence_threshold=self.frontier_confidence_threshold,
                incumbent_cost=route_distance,
                maximum_ms=max(0.0, self.highway_maximum_query_ms -
                               (time.perf_counter() - comparison_started) * 1000),
                maximum_expansions=self.highway_maximum_connector_expansions,
                cached_path=self._last_complete_path,
            )
            shorter_local = alternative.complete
            if shorter_local:
                chosen_path = alternative.path
                selected = False
                self._return_route_source = "local"
            self._trace(
                "drone_return_route_compared", start=current, goal=target,
                highway_version=snapshot.version, local_slam_version=alternative.snapshot_version,
                highway_distance=route_distance,
                local_distance=alternative.cost if alternative.complete else None,
                comparison_status=alternative.status, selected_source=self._return_route_source,
                elapsed_ms=(time.perf_counter() - comparison_started) * 1000,
                expanded_nodes=alternative.expanded_nodes,
            )
        if not route.complete:
            fallback_reason = route.status
        elif not endpoints_valid:
            fallback_reason = "invalid_endpoints"
        elif not locally_valid:
            fallback_reason = "locally_known_occupied"
        elif circuity > self.highway_maximum_route_circuity + 1e-9:
            fallback_reason = "circuity_limit"
        elif shorter_local:
            fallback_reason = "shorter_local_route"
        elif self.highway_mode == "observe":
            fallback_reason = "observe_only"
        else:
            fallback_reason = None
        self._trace(
            "drone_highway_route_evaluated",
            mode=self.highway_mode,
            status=route.status,
            eligible=eligible,
            selected=selected,
            fallback_reason=fallback_reason,
            highway_version=snapshot.version,
            local_slam_version=local_version,
            direct_distance=direct_distance,
            route_distance=(route_distance if math.isfinite(route_distance) else None),
            route_circuity=(circuity if math.isfinite(circuity) else None),
            elapsed_ms=route.elapsed_ms,
            expanded_nodes=route.expanded_nodes,
            graph_edges=route.graph_edges,
        )
        return chosen_path

    def _highway_path_locally_valid(
        self,
        path: Iterable[Position],
    ) -> bool:
        """Reject rover advice contradicted by this drone's newer local SLAM."""
        slam = self.drone.slam_map.snapshot(point_limit=0)
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        offset_x, offset_y = (int(value) for value in slam.origin)
        height, width = occupancy.shape

        def locally_occupied(point: Position) -> bool:
            local_x = int(point[0]) - offset_x
            local_y = int(point[1]) - offset_y
            return bool(
                0 <= local_x < width
                and 0 <= local_y < height
                and occupancy[local_y, local_x] == OCCUPIED
                and confidence[local_y, local_x]
                >= self.frontier_confidence_threshold
            )

        points = tuple((int(x), int(y)) for x, y in path)
        if not points or any(locally_occupied(point) for point in points):
            return False
        for previous, current in zip(points, points[1:]):
            delta_x = current[0] - previous[0]
            delta_y = current[1] - previous[1]
            if max(abs(delta_x), abs(delta_y)) != 1:
                return False
            if delta_x and delta_y and (
                locally_occupied((previous[0] + delta_x, previous[1]))
                or locally_occupied((previous[0], previous[1] + delta_y))
            ):
                return False
        return True

    def _path_history_since(self, index: int) -> tuple[Position, ...]:
        history = self.drone.snapshot().path_history
        if not history:
            return ()
        return tuple(history[max(0, min(int(index), len(history) - 1)):])

    def _sensor_range(self) -> float:
        sensor = getattr(
            getattr(self.drone, "sensor_controller", None),
            "vision_sensor",
            None,
        )
        return float(getattr(sensor, "max_range", self.drone.radius * 4))

    def _sensor_fov(self) -> float:
        sensor = getattr(
            getattr(self.drone, "sensor_controller", None),
            "vision_sensor",
            None,
        )
        return float(getattr(sensor, "fov_deg", 60.0))

    def _sweep_anchor_spacing(self) -> float:
        """Return the lateral width of one outward-facing sensor footprint."""
        sensor = getattr(
            getattr(self.drone, "sensor_controller", None),
            "vision_sensor",
            None,
        )
        fov = self._sensor_fov()
        half_angle = min(90.0, max(0.0, fov / 2.0))
        return max(
            1.0,
            self._sensor_range() * math.sin(math.radians(half_angle)),
        )

    def _next_coordination_report_id(self) -> int:
        report_id = (int(self.drone.id) << 48) | self._coordination_report_counter
        self._coordination_report_counter += 1
        return report_id

    def _sector_policy_enabled(self) -> bool:
        """Return whether this drone was wired to the rover coordinator."""
        return (
            not self._component_policy_enabled()
            and callable(self.dependencies.sector_check_in)
        )

    def _component_policy_enabled(self) -> bool:
        """Return whether frontier-component coordination is configured."""
        return callable(self.dependencies.exploration_check_in)

    def coordination_rendezvous_pending(self) -> bool:
        """Return whether the moving rover should hold for this drone."""
        execution = self._coordination_execution
        return bool(
            self._coordination_report is not None
            or self._coordination_ready is not None
            or (
                execution is not None
                and execution.phase == "returning"
            )
        )

    def _start_sector_check_in(self, *, reason: str) -> None:
        """Retire the current assignment and begin the rover rendezvous."""
        assignment = self._sector_assignment
        snapshot = self.drone.snapshot()
        self._sector_check_in_required = True
        self._completed_sector_id = (
            None if assignment is None else assignment.sector_id
        )
        self._pending_frontier_route = None
        self._trace(
            "drone_sector_exhausted",
            reason=reason,
            sector_id=self._completed_sector_id,
            sector_generation=(
                None if assignment is None else assignment.generation
            ),
            position=snapshot.position,
            position_in_sector=(
                False
                if assignment is None
                else assignment.contains(snapshot.position)
            ),
            sector_was_entered=self._sector_assignment_entered,
            slam_version=self.drone.slam_map.version,
        )
        self._advance_sector_check_in()

    def sector_outcome_report(
        self,
        completed_sector_id: int | None,
    ) -> SectorOutcomeReport | None:
        """Return local component suppression evidence for rover arrival."""
        assignment = self._sector_assignment
        if (
            assignment is None
            or completed_sector_id != assignment.sector_id
        ):
            return None
        suppressions = tuple(
            SectorSuppressionOutcome(
                component_id=component_id,
                reasons=tuple(sorted(reasons)),
                sampled_target_count=len(
                    self._sector_suppression_targets.get(component_id, ())
                ),
            )
            for component_id, reasons in sorted(
                self._sector_suppression_reasons.items()
            )
        )
        return SectorOutcomeReport(
            sector_id=assignment.sector_id,
            generation=assignment.generation,
            suppressions=suppressions,
        )

    def _advance_sector_check_in(self) -> None:
        """Rendezvous once, then await the rover's assignment notification."""
        callback = self.dependencies.sector_check_in
        if not callable(callback) or self._rendezvous_position() is None:
            return

        drone = self.drone
        ready = self._sector_assignment_ready
        if ready is not None:
            if not ready.is_set():
                return
            assignment_callback = self.dependencies.sector_assignment
            if not callable(assignment_callback):
                return
            result = assignment_callback(drone.id)
            self._apply_sector_check_in_result(result)
            return

        current = drone.snapshot().position
        rover_position = self._rendezvous_position()
        if rover_position is None:
            return
        result = callback(drone.id, self._completed_sector_id)
        if not isinstance(result, SectorCheckInResult) or not result.arrived:
            if current != rover_position:
                path_result = self._compute_path(current, rover_position)
                followed = bool(path_result.path) and self._follow_path(
                    path_result.path,
                    source="sector_checkin_astar",
                )
                self._trace(
                    "drone_sector_check_in_path",
                    start=current,
                    target=rover_position,
                    source="astar",
                    path_length=len(path_result.path),
                    path_status=path_result.status,
                    completed=followed,
                )
                current = drone.snapshot().position
                if (
                    current != rover_position
                    and path_result.status == PATH_PARTIAL_LIMIT
                    and followed
                ):
                    return
            if current != rover_position:
                history = drone.snapshot().path_history
                outbound = history[self._sector_path_start_index:]
                breadcrumb = tuple(reversed(outbound))
                followed = bool(breadcrumb) and self._follow_path(
                    breadcrumb,
                    source="sector_checkin_breadcrumb_fallback",
                )
                self._trace(
                    "drone_sector_check_in_path",
                    start=current,
                    target=rover_position,
                    source="breadcrumb_fallback",
                    path_length=len(breadcrumb),
                    completed=followed,
                )
            result = callback(drone.id, self._completed_sector_id)

        self._apply_sector_check_in_result(result)

    def _apply_sector_check_in_result(self, result: Any) -> None:
        """Enter standby, stop, or install one asynchronously delivered sector."""
        if not isinstance(result, SectorCheckInResult) or not result.arrived:
            return
        drone = self.drone
        if result.mission_exhausted:
            self._trace_sector_wait_completed(
                outcome="mission_exhausted",
                next_generation=result.generation,
            )
            self._trace(
                "drone_sector_mission_exhausted",
                generation=result.generation,
                position=drone.snapshot().position,
            )
            drone.runtime_state.mark_done()
            return
        if result.assignment is None:
            waiting_generation = result.waiting_generation
            if self._sector_waiting_generation != waiting_generation:
                self._sector_waiting_generation = waiting_generation
                self._sector_wait_started_at = self._simulation_time()
                self._sector_wait_reason = "barrier"
                self._trace(
                    "drone_sector_waiting_for_team",
                    completed_sector_id=self._completed_sector_id,
                    generation=waiting_generation,
                )
            self._sector_assignment_ready = result.assignment_ready
            return

        assignment = result.assignment
        self._trace_sector_wait_completed(
            outcome="standby" if assignment.standby else "assigned",
            next_generation=assignment.generation,
        )
        self._sector_assignment = assignment
        self._sector_assignment_entered = assignment.contains(
            drone.snapshot().position
        )
        self._sector_ingress_failures = 0
        self._sector_check_in_required = False
        self._completed_sector_id = None
        self._sector_waiting_generation = None
        self._sector_wait_started_at = None
        self._sector_assignment_ready = None
        self._sector_suppression_reasons.clear()
        self._sector_suppression_targets.clear()
        self._sector_empty_confirmations = 0
        self._sector_path_start_index = max(
            0,
            len(drone.snapshot().path_history) - 1,
        )
        self._pending_frontier_route = None
        self._pending_frontier_scan = None
        self._global_frontier_cache = None
        self._last_raw_frontiers = frozenset()
        self._last_global_raw_frontiers = frozenset()
        self._last_frontier_mask = None
        self._frontier_component_targets.clear()
        carried_suppression_count = len(
            self._suppressed_frontier_geometry
        )
        self.border_retry_until.clear()
        self._partial_route_endpoints.clear()
        self._reset_stagnation_window()
        if assignment.standby:
            self._sector_check_in_required = True
            self._sector_assignment_ready = result.assignment_ready
            self._sector_waiting_generation = assignment.generation
            self._sector_wait_started_at = self._simulation_time()
            self._sector_wait_reason = "standby"
            drone.runtime_state.replace_frontiers(())
            self._trace(
                "drone_sector_standby", generation=assignment.generation,
                sector_id=assignment.sector_id,
                position=drone.snapshot().position,
            )
            return
        self.rebuild_frontiers(
            stride=self.frontier_stride,
            confidence_threshold=self.frontier_confidence_threshold,
        )
        self._trace(
            "drone_sector_assigned",
            sector_id=assignment.sector_id,
            generation=assignment.generation,
            seed=assignment.seed,
            gateway=assignment.gateway,
            sector_cell_count=len(assignment.cells),
            frontier_cells=assignment.frontier_cells,
            rover_slam_version=assignment.rover_slam_version,
            bootstrap=assignment.bootstrap,
            assignment_entered=self._sector_assignment_entered,
            carried_suppression_count=carried_suppression_count,
            active_suppression_count=len(
                self._suppressed_frontier_geometry
            ),
            frontier_component_count=len(assignment.frontier_components),
            frontier_component_ids=tuple(
                component.component_id
                for component in assignment.frontier_components
            ),
            estimated_effort=assignment.estimated_effort,
            effort_breakdown=assignment.effort_breakdown,
            scope_pixels=(
                None if assignment.exploration_mask is None
                else int(np.count_nonzero(assignment.exploration_mask))
            ),
        )

    def _trace_sector_wait_completed(
        self,
        *,
        outcome: str,
        next_generation: int,
    ) -> None:
        """Record one completed barrier wait without per-poll inference."""
        started_at = self._sector_wait_started_at
        if started_at is None:
            return
        self._trace(
            "drone_sector_wait_completed",
            waiting_generation=self._sector_waiting_generation,
            next_generation=int(next_generation),
            waited_seconds=max(0.0, self._simulation_time() - started_at),
            outcome=outcome,
            wait_reason=self._sector_wait_reason,
        )

    def _sector_exhausted(self) -> bool:
        """Confirm that an entered assignment has no remaining frontier."""
        assignment = self._sector_assignment
        if assignment is None:
            return False
        snapshot = self.drone.snapshot()
        position_in_sector = assignment.contains(snapshot.position)
        if position_in_sector:
            self._sector_assignment_entered = True
            if not self._has_exploration_scope():
                self._sector_ingress_failures = 0
        if not self._has_exploration_scope() and (
            not snapshot.explored or not self._sector_assignment_entered
        ):
            self._sector_empty_confirmations = 0
            return False
        if assignment.bootstrap and (
            math.dist(snapshot.position, assignment.gateway)
            < assignment.cell_size * 2.0
        ):
            self._sector_empty_confirmations = 0
            return False
        if (
            self._pending_frontier_route is not None
            or self._pending_frontier_scan is not None
            or self._frontier_slam_version != self.drone.slam_map.version
        ):
            self._sector_empty_confirmations = 0
            return False
        if snapshot.frontiers:
            self._sector_empty_confirmations = 0
            return False
        self._sector_empty_confirmations += 1
        return self._sector_empty_confirmations >= 2

    def _in_assigned_sector(self, position: Position) -> bool:
        assignment = self._sector_assignment
        return assignment is None or assignment.contains(position)

    def _has_exploration_scope(self) -> bool:
        assignment = self._sector_assignment
        return assignment is not None and assignment.exploration_mask is not None

    def _in_exploration_scope(self, position: Position) -> bool:
        assignment = self._sector_assignment
        return assignment is None or assignment.permits_exploration(position)

    def _advance_scoped_frontier_route(self) -> None:
        """Transit to permitted work, with bounded failed ingress attempts."""
        now = self._simulation_time()
        available = tuple(
            target for target in self.drone.snapshot().frontiers
            if self._in_exploration_scope(target)
            and now >= self.border_retry_until.get(target, 0.0)
        )
        if not available:
            return
        if self.reach_border(recovery_reason="assigned_neighborhood_ingress"):
            self._sector_ingress_failures = 0
            return
        self._sector_ingress_failures += 1
        if self._sector_ingress_failures >= 2:
            self._start_sector_check_in(reason="assigned_neighborhood_unreachable")

    def _exploration_window_mask(
        self, shape: tuple[int, int], origin: Position,
    ) -> np.ndarray:
        assignment = self._sector_assignment
        if assignment is None or assignment.exploration_mask is None:
            return self._sector_window_mask(shape, origin)
        mask = np.zeros(shape, dtype=bool)
        left, top = origin
        height, width = assignment.exploration_mask.shape
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(width, left + shape[1]), min(height, top + shape[0])
        if x0 < x1 and y0 < y1:
            mask[y0 - top:y1 - top, x0 - left:x1 - left] = (
                assignment.exploration_mask[y0:y1, x0:x1]
            )
        return mask

    def _permitted_random_step(self, start: Position, end: Position) -> bool:
        if not self._has_exploration_scope():
            return True
        return all(self._in_exploration_scope(point) for point in
                   bresenham_line_points(*start, *end))

    def _sector_ingress_targets(
        self,
        current: Position,
        *,
        maximum_cells: int = 8,
    ) -> tuple[Position, ...]:
        """Return nearby collision-free points inside the current assignment."""
        assignment = self._sector_assignment
        if assignment is None:
            return ()
        width = int(self.drone.game.width)
        height = int(self.drone.game.height)
        cell_size = assignment.cell_size

        def bounds(cell: CoverageCell) -> tuple[int, int, int, int]:
            cell_x, cell_y = cell
            return (
                max(0, cell_x * cell_size),
                max(0, cell_y * cell_size),
                min(width - 1, (cell_x + 1) * cell_size - 1),
                min(height - 1, (cell_y + 1) * cell_size - 1),
            )

        def distance_squared(cell: CoverageCell) -> int:
            left, top, right, bottom = bounds(cell)
            nearest_x = min(max(current[0], left), right)
            nearest_y = min(max(current[1], top), bottom)
            return (
                (nearest_x - current[0]) ** 2
                + (nearest_y - current[1]) ** 2
            )

        ordered_cells = sorted(
            assignment.cells,
            key=lambda cell: (
                distance_squared(cell),
                cell[1],
                cell[0],
            ),
        )[:max(1, int(maximum_cells))]
        targets: list[Position] = []
        seen: set[Position] = set()
        for cell in ordered_cells:
            left, top, right, bottom = bounds(cell)
            if left > right or top > bottom:
                continue
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            candidates = (
                (
                    min(max(current[0], left), right),
                    min(max(current[1], top), bottom),
                ),
                (center_x, center_y),
                ((left + center_x) // 2, center_y),
                ((right + center_x) // 2, center_y),
                (center_x, (top + center_y) // 2),
                (center_x, (bottom + center_y) // 2),
            )
            for candidate in candidates:
                if candidate in seen or not assignment.contains(candidate):
                    continue
                seen.add(candidate)
                if self.drone.runtime_state.graph_is_valid(
                    candidate,
                    candidate,
                ):
                    targets.append(candidate)
        return tuple(sorted(
            targets,
            key=lambda target: (
                math.dist(current, target),
                target[1],
                target[0],
            ),
        ))

    def _recover_to_assigned_sector(self) -> bool:
        """Route a boxed-in out-of-sector drone back into owned territory."""
        assignment = self._sector_assignment
        snapshot = self.drone.snapshot()
        current = snapshot.position
        if assignment is None or assignment.contains(current):
            return False

        targets = self._sector_ingress_targets(current)
        attempted_statuses: list[str] = []
        for target in targets[:4]:
            result = self._compute_path(current, target)
            attempted_statuses.append(result.status)
            path = tuple(result.path)
            if len(path) <= 1:
                continue
            source = "sector_ingress_astar"
            if result.status == PATH_PARTIAL_LIMIT:
                if not self._accept_partial_endpoint(
                    current,
                    target,
                    path[-1],
                ):
                    continue
                source = "sector_ingress_astar_partial"
            elif result.status != PATH_COMPLETE:
                continue
            followed = self._follow_path(path, source=source)
            position = self.drone.snapshot().position
            entered = assignment.contains(position)
            if entered:
                self._sector_assignment_entered = True
            if followed:
                self._sector_ingress_failures = 0
            self._trace(
                "drone_sector_ingress_recovery",
                start=current,
                target=target,
                source=source,
                path_status=result.status,
                path_length=len(path),
                completed=followed,
                entered_sector=entered,
                position=position,
            )
            return followed

        fallback_target = targets[0] if targets else assignment.seed
        directions, _borders, step_targets = self._direction_candidates(
            cone_center=snapshot.heading_deg,
            half_fov=180.0,
        )
        if directions:
            target_bearing = self._bearing(current, fallback_target)
            direction = min(
                directions,
                key=lambda candidate: (
                    self._angular_distance(candidate, target_bearing),
                    candidate,
                ),
            )
            self.drone.runtime_state.reorient(direction)
            step_target = step_targets[direction]
            path = bresenham_line_points(
                current[0],
                current[1],
                step_target[0],
                step_target[1],
            )
            followed = self._follow_path(
                path,
                source="sector_ingress_step",
            )
            position = self.drone.snapshot().position
            entered = assignment.contains(position)
            if entered:
                self._sector_assignment_entered = True
            if followed:
                self._sector_ingress_failures = 0
            self._trace(
                "drone_sector_ingress_recovery",
                start=current,
                target=fallback_target,
                source="full_circle_step",
                direction=direction,
                path_status="direct_step",
                path_length=len(path),
                completed=followed,
                entered_sector=entered,
                position=position,
            )
            return followed

        self._sector_ingress_failures += 1
        self._trace(
            "drone_sector_ingress_failed",
            position=current,
            candidate_count=len(targets),
            attempted_path_statuses=tuple(attempted_statuses),
            consecutive_failures=self._sector_ingress_failures,
            sector_id=assignment.sector_id,
            generation=assignment.generation,
        )
        if self._sector_ingress_failures >= 2 and snapshot.explored:
            self._trace(
                "drone_sector_ingress_abandoned",
                position=current,
                consecutive_failures=self._sector_ingress_failures,
                sector_id=assignment.sector_id,
                generation=assignment.generation,
            )
            self._start_sector_check_in(reason="sector_ingress_unreachable")
            return True
        return False

    def _sector_window_mask(
        self,
        shape: tuple[int, int],
        origin: Position,
    ) -> np.ndarray:
        """Return assigned-sector membership for a SLAM array/window."""
        assignment = self._sector_assignment
        if assignment is None:
            return np.ones(shape, dtype=bool)
        height, width = shape
        mask = np.zeros(shape, dtype=bool)
        left, top = int(origin[0]), int(origin[1])
        right = left + width
        bottom = top + height
        cell_size = assignment.cell_size
        first_x = left // cell_size
        last_x = (max(left, right - 1)) // cell_size
        first_y = top // cell_size
        last_y = (max(top, bottom - 1)) // cell_size
        for cell_y in range(first_y, last_y + 1):
            for cell_x in range(first_x, last_x + 1):
                if (cell_x, cell_y) not in assignment.cells:
                    continue
                local_left = max(0, cell_x * cell_size - left)
                local_right = min(width, (cell_x + 1) * cell_size - left)
                local_top = max(0, cell_y * cell_size - top)
                local_bottom = min(height, (cell_y + 1) * cell_size - top)
                mask[local_top:local_bottom, local_left:local_right] = True
        return mask

    def find_new_node(
        self,
    ) -> tuple[list[int], list[Position], Position]:
        """Choose a clear random heading inside the current vision cone."""
        drone = self.drone
        snapshot = drone.snapshot()
        cone_center = float(snapshot.heading_deg) % 360.0
        vision_sensor = getattr(
            getattr(drone, "sensor_controller", None),
            "vision_sensor",
            None,
        )
        vision_fov = float(getattr(vision_sensor, "fov_deg", 60.0))
        half_fov = max(0.0, min(180.0, vision_fov / 2.0))
        valid_directions, border_targets, step_targets = (
            self._direction_candidates(
                cone_center=cone_center,
                half_fov=half_fov,
            )
        )

        valid_directions = [
            direction for direction in valid_directions
            if self._permitted_random_step(snapshot.position, step_targets[direction])
        ]
        assert valid_directions
        bias = self._exploration_heading_bias(
            valid_directions,
            step_targets,
            vision_fov=vision_fov,
        )
        weighted_chooser = getattr(
            drone.exploration_policy,
            "choose_weighted_direction",
            None,
        )
        if callable(weighted_chooser):
            chosen_direction = weighted_chooser(bias.weights)
        else:
            chosen_direction = drone.exploration_policy.choose_direction(
                valid_directions
            )
        if chosen_direction not in valid_directions:
            raise ValueError("exploration policy selected an invalid direction")
        drone.runtime_state.set_direction(chosen_direction)
        chosen_target = step_targets[chosen_direction]
        self._trace(
            "drone_random_direction_selected",
            direction=chosen_direction,
            target=chosen_target,
            valid_direction_count=len(valid_directions),
            border_count=len(border_targets),
            vision_cone_center=cone_center,
            vision_fov_deg=vision_fov,
            selection_mode=bias.mode,
            selected_weight=bias.weights[chosen_direction],
            selected_wall_support=bias.wall_support[chosen_direction],
            selected_frontier_support=(
                bias.frontier_support[chosen_direction]
            ),
            selected_global_frontier_support=(
                bias.global_support[chosen_direction]
            ),
            selected_separation_support=(
                bias.separation_support[chosen_direction]
            ),
            selected_coverage_cell=(
                bias.coverage_cells[chosen_direction]
            ),
            selected_coverage_visit_pressure=(
                bias.coverage_visit_pressure[chosen_direction]
            ),
            selected_coverage_edge_pressure=(
                bias.coverage_edge_pressure[chosen_direction]
            ),
            selected_coverage_penalty_factor=(
                bias.coverage_penalty_factor[chosen_direction]
            ),
            minimum_coverage_penalty_factor=min(
                bias.coverage_penalty_factor.values()
            ),
            maximum_coverage_visit_pressure=max(
                bias.coverage_visit_pressure.values()
            ),
            maximum_coverage_edge_pressure=max(
                bias.coverage_edge_pressure.values()
            ),
            coverage_known_cell_count=len(self._coverage_cell_visits),
            coverage_known_edge_count=len(self._coverage_edge_visits),
            maximum_wall_support=max(bias.wall_support.values()),
            maximum_frontier_support=max(bias.frontier_support.values()),
            maximum_global_frontier_support=max(
                bias.global_support.values()
            ),
            peer_count=bias.peer_count,
            frontier_cluster_count=bias.cluster_count,
            eligible_frontier_cluster_count=bias.eligible_cluster_count,
            filtered_frontier_cluster_count=bias.filtered_cluster_count,
            selected_frontier_cluster_size=bias.selected_cluster_size,
            selected_frontier_cluster_distance=(
                bias.selected_cluster_distance
            ),
            selected_frontier_touches_wall=(
                bias.selected_cluster_touches_wall
            ),
            selected_wall_continuation_alignment=(
                bias.selected_continuation_alignment
            ),
            selected_frontier_cluster_score=bias.selected_cluster_score,
            selected_frontier_cluster_size_rank=(
                bias.selected_cluster_size_rank
            ),
            selected_frontier_cluster_proximity=(
                bias.selected_cluster_proximity
            ),
            wall_frontier_candidate_count=bias.wall_candidate_count,
            generic_frontier_candidate_count=bias.generic_candidate_count,
            minimum_frontier_cluster_cells=(
                self.minimum_frontier_cluster_cells
            ),
            global_frontier_active=bias.global_active,
            global_frontier_target=bias.global_target,
            global_frontier_region_size=bias.global_region_size,
            global_frontier_region_distance=bias.global_region_distance,
            global_frontier_region_bearing=bias.global_region_bearing,
            global_frontier_touches_wall=(
                bias.global_region_touches_wall
            ),
            global_frontier_region_score=bias.global_region_score,
            global_frontier_region_size_rank=(
                bias.global_region_size_rank
            ),
            global_frontier_region_proximity=(
                bias.global_region_proximity
            ),
            global_frontier_region_count=bias.global_region_count,
            global_frontier_eligible_region_count=(
                bias.global_eligible_region_count
            ),
            global_frontier_filtered_region_count=(
                bias.global_filtered_region_count
            ),
            global_wall_frontier_candidate_count=(
                bias.global_wall_candidate_count
            ),
            global_generic_frontier_candidate_count=(
                bias.global_generic_candidate_count
            ),
            global_frontier_requester_distance=(
                bias.global_requester_distance
            ),
            global_frontier_nearest_peer_distance=(
                bias.global_nearest_peer_distance
            ),
            global_frontier_ownership_margin=(
                bias.global_ownership_margin
            ),
            global_frontier_launch_sector_alignment=(
                bias.global_launch_sector_alignment
            ),
            global_frontier_ownership_contribution=(
                bias.global_ownership_contribution
            ),
            global_frontier_slam_version=bias.global_slam_version,
        )
        return valid_directions, border_targets, chosen_target

    def _exploration_heading_bias(
        self,
        valid_directions: Iterable[int],
        step_targets: dict[int, Position],
        *,
        vision_fov: float,
    ) -> _HeadingBias:
        """Score wall tips, generic unknown borders, and team separation."""
        directions = tuple(int(direction) for direction in valid_directions)
        drone = self.drone
        current = drone.snapshot().position
        vision_sensor = getattr(
            getattr(drone, "sensor_controller", None),
            "vision_sensor",
            None,
        )
        sensor_range = float(getattr(
            vision_sensor,
            "max_range",
            drone.radius * 4,
        ))
        margin = int(math.ceil(sensor_range + drone.step + 2.0))
        slam = drone.slam_map.snapshot_window((
            current[0] - margin,
            current[1] - margin,
            current[0] + margin + 1,
            current[1] + margin + 1,
        ))
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        known = confidence >= self.frontier_confidence_threshold
        known_free = known & (occupancy == FREE)
        known_occupied = known & (occupancy == OCCUPIED)
        unknown = (~known) | (occupancy == UNKNOWN)
        frontier_unknown = unknown & self._neighbor_adjacency(known_free)
        frontier_unknown &= self._exploration_window_mask(
            frontier_unknown.shape,
            slam.origin,
        )
        wall_unknown = (
            frontier_unknown & self._neighbor_adjacency(known_occupied)
        )

        clusters = self._frontier_clusters(
            frontier_unknown,
            wall_unknown,
            slam.origin,
            current=current,
            heading=float(drone.snapshot().heading_deg),
        )
        eligible_clusters = tuple(
            cluster
            for cluster in clusters
            if cluster.size >= self.minimum_frontier_cluster_cells
        )
        selection = self._select_frontier_cluster(
            eligible_clusters,
            directions,
            step_targets,
            sensor_range=sensor_range,
            half_fov=max(0.0, min(180.0, vision_fov / 2.0)),
        )
        selected_cluster = selection.cluster
        frontier_support = dict(selection.support)
        wall_support = {
            direction: (
                selection.support[direction]
                if selected_cluster is not None
                and selected_cluster.touches_wall
                else 0.0
            )
            for direction in directions
        }
        separation_support, peer_count = self._separation_scores(
            directions,
            sensor_range=sensor_range,
        )
        global_guidance = self._global_frontier_guidance(
            directions,
            current=current,
            heading=float(drone.snapshot().heading_deg),
            sensor_range=sensor_range,
        )
        normalized_wall = self._normalize_scores(wall_support)
        normalized_frontier = self._normalize_scores(frontier_support)
        wall_tracking = (
            selected_cluster is not None
            and selected_cluster.touches_wall
        )
        frontier_tracking = selected_cluster is not None
        assignment = self._sector_assignment
        sector_ingress = bool(
            assignment is not None and not assignment.contains(current)
        )
        sector_support = (
            self._directional_progress_scores(
                directions,
                self._bearing(current, assignment.seed),
            )
            if sector_ingress and assignment is not None
            else {direction: 0.0 for direction in directions}
        )
        (
            coverage_cells,
            coverage_visit_pressure,
            coverage_edge_pressure,
            coverage_penalty_factor,
        ) = self._coverage_heading_penalties(
            directions,
            current=current,
            apply_penalty=not sector_ingress,
        )
        if sector_ingress:
            mode = "sector_ingress"
        elif global_guidance.active and global_guidance.touches_wall:
            mode = "global_wall_tracking"
        elif global_guidance.active:
            mode = "global_unexplored_region"
        elif wall_tracking:
            mode = "wall_tracking"
        elif frontier_tracking:
            mode = "unexplored_region"
        else:
            mode = "distributed_random"

        weights: dict[int, float] = {}
        for direction in directions:
            local_information_bias = (
                self.wall_direction_bias * normalized_wall[direction]
                if wall_tracking
                else self.unexplored_direction_bias
                * normalized_frontier[direction]
            )
            information_bias = local_information_bias
            if global_guidance.active:
                global_weight = (
                    self.wall_direction_bias
                    if global_guidance.touches_wall
                    else self.unexplored_direction_bias
                )
                # A whole-map target is strategic.  Nearby exact geometry
                # remains useful for collision-safe wall following, but does
                # not get to pull the drone back into a small local hole.
                information_bias = (
                    0.35 * local_information_bias
                    + global_weight * global_guidance.support[direction]
                )
            if sector_ingress:
                information_bias += (
                    2.0
                    * self.unexplored_direction_bias
                    * sector_support[direction]
                )
            weight = max(
                1e-6,
                1.0
                + information_bias
                + self.separation_direction_bias
                * separation_support[direction],
            )
            weight *= coverage_penalty_factor[direction]
            if (
                assignment is not None
                and assignment.contains(current)
                and not assignment.contains(step_targets[direction])
            ):
                weight *= 0.05
            weights[direction] = max(1e-6, weight)
        return _HeadingBias(
            weights=weights,
            wall_support=wall_support,
            frontier_support=frontier_support,
            global_support=global_guidance.support,
            separation_support=separation_support,
            coverage_cells=coverage_cells,
            coverage_visit_pressure=coverage_visit_pressure,
            coverage_edge_pressure=coverage_edge_pressure,
            coverage_penalty_factor=coverage_penalty_factor,
            mode=mode,
            peer_count=peer_count,
            cluster_count=len(clusters),
            eligible_cluster_count=len(eligible_clusters),
            filtered_cluster_count=(
                len(clusters) - len(eligible_clusters)
            ),
            selected_cluster_size=(
                selected_cluster.size if selected_cluster is not None else 0
            ),
            selected_cluster_distance=(
                selected_cluster.distance
                if selected_cluster is not None
                else None
            ),
            selected_cluster_touches_wall=(
                selected_cluster.touches_wall
                if selected_cluster is not None
                else False
            ),
            selected_continuation_alignment=(
                selected_cluster.continuation_alignment
                if selected_cluster is not None
                else 0.0
            ),
            selected_cluster_score=selection.score,
            selected_cluster_size_rank=selection.size_rank,
            selected_cluster_proximity=selection.proximity,
            wall_candidate_count=selection.wall_candidate_count,
            generic_candidate_count=selection.generic_candidate_count,
            global_active=global_guidance.active,
            global_target=global_guidance.target,
            global_region_size=global_guidance.size,
            global_region_distance=global_guidance.distance,
            global_region_bearing=global_guidance.bearing,
            global_region_touches_wall=global_guidance.touches_wall,
            global_region_score=global_guidance.score,
            global_region_size_rank=global_guidance.size_rank,
            global_region_proximity=global_guidance.proximity,
            global_region_count=global_guidance.region_count,
            global_eligible_region_count=(
                global_guidance.eligible_region_count
            ),
            global_filtered_region_count=(
                global_guidance.filtered_region_count
            ),
            global_wall_candidate_count=(
                global_guidance.wall_candidate_count
            ),
            global_generic_candidate_count=(
                global_guidance.generic_candidate_count
            ),
            global_requester_distance=(
                global_guidance.requester_distance
            ),
            global_nearest_peer_distance=(
                global_guidance.nearest_peer_distance
            ),
            global_ownership_margin=global_guidance.ownership_margin,
            global_launch_sector_alignment=(
                global_guidance.launch_sector_alignment
            ),
            global_ownership_contribution=(
                global_guidance.ownership_contribution
            ),
            global_slam_version=global_guidance.slam_version,
        )

    def _coverage_cell(self, position: Position) -> CoverageCell:
        """Return the coarse coverage-memory cell for a map position."""
        return (
            int(position[0]) // self.coverage_memory_cell_size,
            int(position[1]) // self.coverage_memory_cell_size,
        )

    @staticmethod
    def _coverage_edge(
        first: CoverageCell,
        second: CoverageCell,
    ) -> CoverageEdge:
        """Return one direction-independent coarse traversal edge."""
        return (first, second) if first <= second else (second, first)

    def _coverage_record_value(
        self,
        record: _CoverageRecord | None,
        now: float,
    ) -> float:
        """Return an exponentially decayed visit pressure."""
        if record is None:
            return 0.0
        elapsed = max(0.0, float(now) - record.updated_at)
        return record.value * math.exp(
            -elapsed / self.coverage_memory_decay_seconds
        )

    def _increment_coverage_record(
        self,
        records: dict[Any, _CoverageRecord],
        key: Any,
        now: float,
    ) -> None:
        """Add one visit after lazily decaying the previous pressure."""
        records[key] = _CoverageRecord(
            value=self._coverage_record_value(records.get(key), now) + 1.0,
            updated_at=float(now),
        )

    def _coverage_heading_penalties(
        self,
        directions: Iterable[int],
        *,
        current: Position,
        apply_penalty: bool,
    ) -> tuple[
        dict[int, CoverageCell],
        dict[int, float],
        dict[int, float],
        dict[int, float],
    ]:
        """Score projected cells and repeated edges for candidate headings."""
        now = self._simulation_time()
        current_cell = self._coverage_cell(current)
        width = max(1, int(self.drone.game.width))
        height = max(1, int(self.drone.game.height))
        cells: dict[int, CoverageCell] = {}
        visit_pressure: dict[int, float] = {}
        edge_pressure: dict[int, float] = {}
        penalty_factor: dict[int, float] = {}
        for raw_direction in directions:
            direction = int(raw_direction)
            projected = next_cell_coords(
                *current,
                self.coverage_memory_cell_size,
                direction,
            )
            projected = (
                min(max(int(projected[0]), 0), width - 1),
                min(max(int(projected[1]), 0), height - 1),
            )
            cell = self._coverage_cell(projected)
            edge = self._coverage_edge(current_cell, cell)
            if cell == current_cell:
                # A coarse cell does not distinguish headings within itself;
                # penalizing it would arbitrarily overpower other evidence
                # near map edges and cell centers.
                visits = 0.0
                traversals = 0.0
            else:
                visits = self._coverage_record_value(
                    self._coverage_cell_visits.get(cell),
                    now,
                )
                traversals = self._coverage_record_value(
                    self._coverage_edge_visits.get(edge),
                    now,
                )
            pressure = (
                self.coverage_visit_weight * math.log1p(visits)
                + self.coverage_edge_weight * math.log1p(traversals)
            )
            cells[direction] = cell
            visit_pressure[direction] = visits
            edge_pressure[direction] = traversals
            penalty_factor[direction] = (
                1.0 if not apply_penalty else 1.0 / (1.0 + pressure)
            )
        return cells, visit_pressure, edge_pressure, penalty_factor

    def _record_coverage_transition(
        self,
        previous: Position,
        current: Position,
        now: float,
    ) -> tuple[int, int, int, int]:
        """Record one coarse cell crossing and return trace counters."""
        previous_cell = self._coverage_cell(previous)
        current_cell = self._coverage_cell(current)
        if self._coverage_last_cell != previous_cell:
            self._increment_coverage_record(
                self._coverage_cell_visits,
                previous_cell,
                now,
            )
            self._coverage_last_cell = previous_cell
        if current_cell == previous_cell:
            return 0, 0, 0, 0

        edge = self._coverage_edge(previous_cell, current_cell)
        revisited = current_cell in self._coverage_cell_visits
        repeated_edge = edge in self._coverage_edge_visits
        self._increment_coverage_record(
            self._coverage_cell_visits,
            current_cell,
            now,
        )
        self._increment_coverage_record(
            self._coverage_edge_visits,
            edge,
            now,
        )
        self._coverage_last_cell = current_cell
        return (
            1,
            0 if revisited else 1,
            1 if revisited else 0,
            1 if repeated_edge else 0,
        )

    def _global_frontier_guidance(
        self,
        directions: tuple[int, ...],
        *,
        current: Position,
        heading: float,
        sensor_range: float,
    ) -> _GlobalFrontierGuidance:
        """Return cheap per-step bearings from a periodically rebuilt cache."""
        cache = self._ensure_global_frontier_cache(
            current=current,
            heading=heading,
        )
        zero_support = {direction: 0.0 for direction in directions}
        region = cache.target
        target_position = cache.target_position
        if region is None or target_position is None:
            return _GlobalFrontierGuidance(
                support=zero_support,
                active=False,
                target=None,
                size=0,
                distance=None,
                bearing=None,
                touches_wall=False,
                score=0.0,
                size_rank=0.0,
                proximity=0.0,
                region_count=len(cache.regions),
                eligible_region_count=cache.eligible_region_count,
                filtered_region_count=cache.filtered_region_count,
                wall_candidate_count=cache.wall_candidate_count,
                generic_candidate_count=cache.generic_candidate_count,
                requester_distance=cache.requester_distance,
                nearest_peer_distance=cache.nearest_peer_distance,
                ownership_margin=cache.ownership_margin,
                launch_sector_alignment=cache.launch_sector_alignment,
                ownership_contribution=cache.ownership_contribution,
                slam_version=cache.slam_version,
            )

        distance = math.dist(current, target_position)
        bearing = self._bearing(current, target_position)
        local_window_radius = sensor_range + self.drone.step + 2.0
        active = distance > local_window_radius
        support = (
            self._directional_progress_scores(directions, bearing)
            if active
            else zero_support
        )
        return _GlobalFrontierGuidance(
            support=support,
            active=active,
            target=target_position,
            size=region.size,
            distance=distance,
            bearing=bearing,
            touches_wall=region.touches_wall,
            score=cache.target_score,
            size_rank=cache.target_size_rank,
            proximity=cache.target_proximity,
            region_count=len(cache.regions),
            eligible_region_count=cache.eligible_region_count,
            filtered_region_count=cache.filtered_region_count,
            wall_candidate_count=cache.wall_candidate_count,
            generic_candidate_count=cache.generic_candidate_count,
            requester_distance=cache.requester_distance,
            nearest_peer_distance=cache.nearest_peer_distance,
            ownership_margin=cache.ownership_margin,
            launch_sector_alignment=cache.launch_sector_alignment,
            ownership_contribution=cache.ownership_contribution,
            slam_version=cache.slam_version,
        )

    def _ensure_global_frontier_cache(
        self,
        *,
        current: Position,
        heading: float,
        slam: Any | None = None,
    ) -> _GlobalFrontierCache:
        """Rebuild coarse whole-map regions only after the refresh cadence."""
        now = self._simulation_time()
        cache = self._global_frontier_cache
        current_version = self.drone.slam_map.version
        if cache is not None and (
            cache.slam_version == current_version
            or now - cache.built_at < self.global_frontier_refresh_interval
        ):
            return cache

        started = time.perf_counter()
        if slam is None:
            slam = self.drone.slam_map.snapshot(point_limit=0)
        regions = self._coarse_global_frontier_regions(slam)
        selection = self._select_global_frontier_region(
            regions,
            current=current,
            heading=heading,
            preferred_region=(cache.target if cache is not None else None),
            preferred_position=(
                cache.target_position if cache is not None else None
            ),
        )
        cache = _GlobalFrontierCache(
            regions=regions,
            target=selection.region,
            target_position=selection.position,
            target_score=selection.score,
            target_size_rank=selection.size_rank,
            target_proximity=selection.proximity,
            eligible_region_count=selection.eligible_region_count,
            filtered_region_count=(
                len(regions) - selection.eligible_region_count
            ),
            wall_candidate_count=selection.wall_candidate_count,
            generic_candidate_count=selection.generic_candidate_count,
            requester_distance=selection.requester_distance,
            nearest_peer_distance=selection.nearest_peer_distance,
            ownership_margin=selection.ownership_margin,
            launch_sector_alignment=(
                selection.launch_sector_alignment
            ),
            ownership_contribution=selection.ownership_contribution,
            target_retained=selection.retained_previous,
            target_region_overlap=selection.previous_region_overlap,
            slam_version=int(slam.version),
            built_at=now,
        )
        self._global_frontier_cache = cache
        self._trace(
            "drone_global_frontiers_rebuilt",
            slam_version=cache.slam_version,
            coarse_cell_size=self.global_frontier_cell_size,
            region_count=len(regions),
            eligible_region_count=selection.eligible_region_count,
            filtered_region_count=cache.filtered_region_count,
            wall_candidate_count=selection.wall_candidate_count,
            generic_candidate_count=selection.generic_candidate_count,
            selected_target=selection.position,
            selected_region_size=(
                selection.region.size
                if selection.region is not None
                else 0
            ),
            selected_region_touches_wall=(
                selection.region.touches_wall
                if selection.region is not None
                else False
            ),
            selected_score=selection.score,
            selected_size_rank=selection.size_rank,
            selected_proximity=selection.proximity,
            selected_requester_distance=selection.requester_distance,
            selected_nearest_peer_distance=(
                selection.nearest_peer_distance
            ),
            selected_ownership_margin=selection.ownership_margin,
            selected_launch_sector_alignment=(
                selection.launch_sector_alignment
            ),
            selected_ownership_contribution=(
                selection.ownership_contribution
            ),
            selected_target_retained=selection.retained_previous,
            selected_target_region_overlap=(
                selection.previous_region_overlap
            ),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return cache

    def _coarse_global_frontier_regions(
        self,
        slam: Any,
    ) -> tuple[_GlobalFrontierRegion, ...]:
        """Aggregate full-SLAM frontier pixels before connected components."""
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        known = confidence >= self.frontier_confidence_threshold
        known_free = known & (occupancy == FREE)
        known_occupied = known & (occupancy == OCCUPIED)
        unknown = (~known) | (occupancy == UNKNOWN)
        frontier_unknown = unknown & self._neighbor_adjacency(known_free)
        frontier_unknown &= self._exploration_window_mask(
            frontier_unknown.shape,
            (0, 0),
        )
        rows, columns = np.nonzero(frontier_unknown)
        if len(rows) == 0:
            return ()

        wall_unknown = (
            frontier_unknown & self._neighbor_adjacency(known_occupied)
        )
        cell_size = self.global_frontier_cell_size
        height, width = occupancy.shape
        tile_columns = max(1, math.ceil(width / cell_size))
        tile_rows = max(1, math.ceil(height / cell_size))
        tile_x = columns // cell_size
        tile_y = rows // cell_size
        flat_indices = tile_y * tile_columns + tile_x
        tile_total = tile_rows * tile_columns
        counts = np.bincount(flat_indices, minlength=tile_total)
        wall_counts = np.bincount(
            flat_indices,
            weights=wall_unknown[rows, columns].astype(np.float64),
            minlength=tile_total,
        )
        x_sums = np.bincount(
            flat_indices,
            weights=columns.astype(np.float64),
            minlength=tile_total,
        )
        y_sums = np.bincount(
            flat_indices,
            weights=rows.astype(np.float64),
            minlength=tile_total,
        )

        def tile_target(index: int) -> Position:
            center_x = x_sums[index] / counts[index]
            center_y = y_sums[index] / counts[index]
            if not self._has_exploration_scope():
                return (int(round(center_x)), int(round(center_y)))
            members = np.flatnonzero(flat_indices == index)
            distances = ((columns[members] - center_x) ** 2
                         + (rows[members] - center_y) ** 2)
            nearest = members[int(np.argmin(distances))]
            return (int(columns[nearest]), int(rows[nearest]))

        remaining = {int(index) for index in np.flatnonzero(counts)}
        regions: list[_GlobalFrontierRegion] = []
        for seed in sorted(remaining):
            if seed not in remaining:
                continue
            remaining.remove(seed)
            stack = [seed]
            members: list[int] = []
            while stack:
                flat_index = stack.pop()
                members.append(flat_index)
                member_y, member_x = divmod(flat_index, tile_columns)
                for offset_y in (-1, 0, 1):
                    for offset_x in (-1, 0, 1):
                        if offset_x == 0 and offset_y == 0:
                            continue
                        neighbor_x = member_x + offset_x
                        neighbor_y = member_y + offset_y
                        if not (
                            0 <= neighbor_x < tile_columns
                            and 0 <= neighbor_y < tile_rows
                        ):
                            continue
                        neighbor = neighbor_y * tile_columns + neighbor_x
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            stack.append(neighbor)

            member_indices = np.asarray(members, dtype=np.int64)
            size = int(np.sum(counts[member_indices]))
            tiles = tuple(sorted(
                (
                    _GlobalFrontierTile(
                        target=tile_target(index),
                        size=int(counts[index]),
                        wall_cells=int(round(float(wall_counts[index]))),
                    )
                    for index in members
                ),
                key=lambda tile: tile.target,
            ))
            regions.append(_GlobalFrontierRegion(
                tiles=tiles,
                size=size,
                wall_cells=int(round(float(np.sum(
                    wall_counts[member_indices]
                )))),
                tile_count=len(members),
            ))
        return tuple(sorted(
            regions,
            key=lambda region: (region.tiles[0].target, -region.size),
        ))

    def _global_region_key(
        self,
        region: _GlobalFrontierRegion,
    ) -> frozenset[Position]:
        """Return stable coarse-cell membership for one global region."""
        cell_size = self.global_frontier_cell_size
        return frozenset(
            (tile.target[0] // cell_size, tile.target[1] // cell_size)
            for tile in region.tiles
        )

    def _select_global_frontier_region(
        self,
        regions: tuple[_GlobalFrontierRegion, ...],
        *,
        current: Position,
        heading: float,
        preferred_region: _GlobalFrontierRegion | None = None,
        preferred_position: Position | None = None,
    ) -> _GlobalFrontierSelection:
        """Score regions while retaining a still-competitive prior target."""
        eligible = tuple(
            region
            for region in regions
            if region.size >= self.minimum_frontier_cluster_cells
        )
        wall_regions = tuple(
            region for region in eligible if region.touches_wall
        )
        generic_regions = tuple(
            region for region in eligible if not region.touches_wall
        )
        tier = wall_regions or generic_regions
        if not tier:
            return _GlobalFrontierSelection(
                region=None,
                position=None,
                score=0.0,
                size_rank=0.0,
                proximity=0.0,
                eligible_region_count=len(eligible),
                wall_candidate_count=0,
                generic_candidate_count=0,
                requester_distance=None,
                nearest_peer_distance=None,
                ownership_margin=0.0,
                launch_sector_alignment=0.0,
                ownership_contribution=0.0,
            )

        sizes = sorted({region.size for region in tier})
        size_ranks = (
            {sizes[0]: 1.0}
            if len(sizes) == 1
            else {
                size: index / float(len(sizes) - 1)
                for index, size in enumerate(sizes)
            }
        )
        candidate_regions: list[_GlobalFrontierRegion] = []
        candidate_positions: list[Position] = []
        candidate_alignments: list[float] = []
        candidate_distances: list[float] = []
        candidate_bearings: list[float] = []
        candidate_proximities: list[float] = []
        candidate_size_ranks: list[float] = []
        current_array = np.asarray(current, dtype=np.float64)
        normalized_heading = float(heading) % 360.0
        preferred_region_key = (
            self._global_region_key(preferred_region)
            if preferred_region is not None
            else frozenset()
        )
        preferred_cell = (
            (
                preferred_position[0] // self.global_frontier_cell_size,
                preferred_position[1] // self.global_frontier_cell_size,
            )
            if preferred_position is not None
            else None
        )
        for region in tier:
            representative_tiles = (
                tuple(tile for tile in region.tiles if tile.touches_wall)
                if region.touches_wall
                else region.tiles
            )
            targets = np.asarray(
                [tile.target for tile in representative_tiles],
                dtype=np.float64,
            )
            deltas = targets - current_array
            distances = np.hypot(deltas[:, 0], deltas[:, 1])
            bearings = (
                np.degrees(np.arctan2(deltas[:, 0], -deltas[:, 1]))
                % 360.0
            )
            bearings = np.where(distances <= 1e-12, 0.0, bearings)
            angular_distances = np.abs(
                (bearings - normalized_heading + 180.0) % 360.0
                - 180.0
            )
            alignments = np.maximum(
                0.0,
                1.0 - angular_distances / 180.0,
            )
            tile_sizes = np.fromiter(
                (tile.size for tile in representative_tiles),
                dtype=np.int64,
                count=len(representative_tiles),
            )
            retained_tile_indices: list[int] = []
            if (
                preferred_cell is not None
                and preferred_region_key & self._global_region_key(region)
            ):
                retained_tile_indices = [
                    index
                    for index, tile in enumerate(representative_tiles)
                    if (
                        tile.target[0] // self.global_frontier_cell_size,
                        tile.target[1] // self.global_frontier_cell_size,
                    ) == preferred_cell
                ]
            if retained_tile_indices:
                retained_targets = targets[retained_tile_indices]
                preferred_array = np.asarray(
                    preferred_position,
                    dtype=np.float64,
                )
                retained_deltas = retained_targets - preferred_array
                retained_distances = np.hypot(
                    retained_deltas[:, 0],
                    retained_deltas[:, 1],
                )
                retained_order = np.lexsort((
                    retained_targets[:, 1],
                    retained_targets[:, 0],
                    retained_distances,
                ))
                selected_index = retained_tile_indices[
                    int(retained_order[0])
                ]
            else:
                if region.touches_wall:
                    ordering = np.lexsort((
                        targets[:, 1],
                        targets[:, 0],
                        -tile_sizes,
                        distances,
                        -alignments,
                    ))
                else:
                    ordering = np.lexsort((
                        targets[:, 1],
                        targets[:, 0],
                        -tile_sizes,
                        distances,
                    ))
                selected_index = int(ordering[0])
            distance = float(distances[selected_index])
            candidate_regions.append(region)
            candidate_positions.append(
                representative_tiles[selected_index].target
            )
            candidate_alignments.append(float(alignments[selected_index]))
            candidate_distances.append(distance)
            candidate_bearings.append(float(bearings[selected_index]))
            candidate_proximities.append(
                1.0 / (1.0 + distance / self.frontier_distance_band)
            )
            candidate_size_ranks.append(size_ranks[region.size])

        target_array = np.asarray(
            candidate_positions,
            dtype=np.float64,
        )
        requester_distances = np.asarray(
            candidate_distances,
            dtype=np.float64,
        )
        proximities = np.asarray(
            candidate_proximities,
            dtype=np.float64,
        )
        region_size_ranks = np.asarray(
            candidate_size_ranks,
            dtype=np.float64,
        )
        scores = (
            self.frontier_cluster_size_weight * region_size_ranks
            + self.frontier_cluster_proximity_weight * proximities
        )
        if wall_regions:
            scores += self.wall_continuation_weight * np.asarray(
                candidate_alignments,
                dtype=np.float64,
            )

        peer_positions = tuple(
            position
            for drone_id, position in self.dependencies.get_drone_positions()
            if int(drone_id) != int(self.drone.id)
        )
        if peer_positions:
            peer_array = np.asarray(peer_positions, dtype=np.float64)
            peer_deltas = (
                target_array[:, np.newaxis, :]
                - peer_array[np.newaxis, :, :]
            )
            nearest_peer_distances = np.min(
                np.hypot(peer_deltas[:, :, 0], peer_deltas[:, :, 1]),
                axis=1,
            )
            map_height, map_width = self.drone.slam_map.shape
            map_diagonal = max(
                1.0,
                math.hypot(max(0, map_width - 1), max(0, map_height - 1)),
            )
            ownership_margins = np.clip(
                (nearest_peer_distances - requester_distances)
                / map_diagonal,
                -1.0,
                1.0,
            )
            requester_peer_deltas = peer_array - current_array
            nearest_requester_peer = float(np.min(np.hypot(
                requester_peer_deltas[:, 0],
                requester_peer_deltas[:, 1],
            )))
            overlap_radius = max(1.0, float(self.drone.step))
            position_overlap = max(
                0.0,
                1.0 - nearest_requester_peer / overlap_radius,
            )
            launch_radius = max(
                overlap_radius,
                float(self.global_frontier_cell_size),
            )
            launch_area_strength = max(
                0.0,
                1.0
                - math.dist(current, self.drone.start_pos) / launch_radius,
            )
            launch_overlap = position_overlap * launch_area_strength
            drone_count = max(
                1,
                int(self.drone.settings.mission_config.num_drones),
            )
            sector_heading = 360.0 * float(self.drone.id) / drone_count
            sector_distances = np.abs(
                (
                    np.asarray(candidate_bearings) - sector_heading + 180.0
                ) % 360.0
                - 180.0
            )
            launch_sector_alignments = (
                1.0 + np.cos(np.deg2rad(sector_distances))
            ) / 2.0
            ownership_contributions = (
                self.global_frontier_ownership_weight
                * (
                    ownership_margins
                    + _GLOBAL_LAUNCH_SECTOR_TIE_RATIO
                    * launch_overlap
                    * launch_sector_alignments
                )
            )
            scores += ownership_contributions
        else:
            nearest_peer_distances = np.full(len(tier), np.nan)
            ownership_margins = np.zeros(len(tier), dtype=np.float64)
            launch_sector_alignments = np.zeros(
                len(tier),
                dtype=np.float64,
            )
            ownership_contributions = np.zeros(
                len(tier),
                dtype=np.float64,
            )

        region_sizes = np.fromiter(
            (region.size for region in candidate_regions),
            dtype=np.int64,
            count=len(candidate_regions),
        )
        selected_index = int(np.lexsort((
            target_array[:, 1],
            target_array[:, 0],
            requester_distances,
            -region_sizes,
            -scores,
        ))[0])
        retained_previous = False
        previous_region_overlap = 0.0
        if preferred_region is not None:
            preferred_key = self._global_region_key(preferred_region)
            matches: list[tuple[float, int]] = []
            for index, region in enumerate(candidate_regions):
                candidate_key = self._global_region_key(region)
                intersection = len(preferred_key & candidate_key)
                if intersection == 0:
                    continue
                union = len(preferred_key | candidate_key)
                overlap = intersection / max(1, union)
                matches.append((overlap, index))
            if matches:
                previous_region_overlap, previous_index = max(
                    matches,
                    key=lambda item: (
                        item[0],
                        float(scores[item[1]]),
                        -float(requester_distances[item[1]]),
                    ),
                )
                if (
                    float(scores[previous_index])
                    >= float(scores[selected_index])
                    - _GLOBAL_TARGET_SWITCH_SCORE_MARGIN
                ):
                    selected_index = previous_index
                    retained_previous = True
        nearest_peer_distance = (
            float(nearest_peer_distances[selected_index])
            if peer_positions
            else None
        )
        return _GlobalFrontierSelection(
            region=candidate_regions[selected_index],
            position=candidate_positions[selected_index],
            score=float(scores[selected_index]),
            size_rank=float(region_size_ranks[selected_index]),
            proximity=float(proximities[selected_index]),
            eligible_region_count=len(eligible),
            wall_candidate_count=len(wall_regions),
            generic_candidate_count=len(generic_regions),
            requester_distance=float(requester_distances[selected_index]),
            nearest_peer_distance=nearest_peer_distance,
            ownership_margin=float(ownership_margins[selected_index]),
            launch_sector_alignment=float(
                launch_sector_alignments[selected_index]
            ),
            ownership_contribution=float(
                ownership_contributions[selected_index]
            ),
            retained_previous=retained_previous,
            previous_region_overlap=previous_region_overlap,
        )

    @classmethod
    def _directional_progress_scores(
        cls,
        directions: Iterable[int],
        target_bearing: float,
    ) -> dict[int, float]:
        """Normalize which valid headings turn most toward a remote target."""
        closeness = {
            int(direction): 180.0 - cls._angular_distance(
                direction,
                target_bearing,
            )
            for direction in directions
        }
        if not closeness:
            return {}
        minimum = min(closeness.values())
        maximum = max(closeness.values())
        if maximum - minimum <= 1e-9:
            return {direction: 1.0 for direction in closeness}
        return {
            direction: (value - minimum) / (maximum - minimum)
            for direction, value in closeness.items()
        }

    @staticmethod
    def _bearing(origin: Position, target: Position) -> float:
        """Return simulator heading from origin to target."""
        return math.degrees(math.atan2(
            target[0] - origin[0],
            -(target[1] - origin[1]),
        )) % 360.0

    def _frontier_clusters(
        self,
        frontier_mask: np.ndarray,
        wall_mask: np.ndarray,
        window_origin: Position,
        *,
        current: Position,
        heading: float,
    ) -> tuple[_FrontierCluster, ...]:
        """Extract deterministic eight-connected local boundary components."""
        rows, columns = np.nonzero(frontier_mask)
        remaining = {
            (int(column), int(row))
            for row, column in zip(rows, columns)
        }
        clusters: list[_FrontierCluster] = []
        for seed in sorted(remaining):
            if seed not in remaining:
                continue
            remaining.remove(seed)
            stack = [seed]
            local_cells: list[Position] = []
            touches_wall = False
            while stack:
                local_x, local_y = stack.pop()
                local_cells.append((local_x, local_y))
                touches_wall = touches_wall or bool(
                    wall_mask[local_y, local_x]
                )
                for offset_y in (-1, 0, 1):
                    for offset_x in (-1, 0, 1):
                        if offset_x == 0 and offset_y == 0:
                            continue
                        neighbor = (
                            local_x + offset_x,
                            local_y + offset_y,
                        )
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            stack.append(neighbor)

            global_cells = tuple(sorted(
                (
                    local_x + int(window_origin[0]),
                    local_y + int(window_origin[1]),
                )
                for local_x, local_y in local_cells
            ))
            distance = min(
                math.dist(current, cell) for cell in global_cells
            )
            centroid_x = sum(cell[0] for cell in global_cells) / len(
                global_cells
            )
            centroid_y = sum(cell[1] for cell in global_cells) / len(
                global_cells
            )
            bearing = math.degrees(math.atan2(
                centroid_x - current[0],
                -(centroid_y - current[1]),
            )) % 360.0
            angular_delta = self._angular_distance(bearing, heading)
            alignment = max(0.0, 1.0 - angular_delta / 180.0)
            clusters.append(_FrontierCluster(
                cells=global_cells,
                size=len(global_cells),
                touches_wall=touches_wall,
                distance=distance,
                continuation_alignment=alignment,
            ))
        return tuple(clusters)

    def _select_frontier_cluster(
        self,
        clusters: Iterable[_FrontierCluster],
        directions: tuple[int, ...],
        step_targets: dict[int, Position],
        *,
        sensor_range: float,
        half_fov: float,
    ) -> _ClusterSelection:
        """Apply strict wall/generic tiers and weighted within-tier scoring."""
        candidates = tuple(clusters)
        wall_clusters = tuple(
            cluster for cluster in candidates if cluster.touches_wall
        )
        generic_clusters = tuple(
            cluster for cluster in candidates if not cluster.touches_wall
        )

        wall_candidates = self._actionable_cluster_support(
            wall_clusters,
            directions,
            step_targets,
            sensor_range=sensor_range,
            half_fov=half_fov,
        )
        tier = wall_candidates
        if not tier:
            tier = self._actionable_cluster_support(
                generic_clusters,
                directions,
                step_targets,
                sensor_range=sensor_range,
                half_fov=half_fov,
            )
        if not tier:
            return _ClusterSelection(
                cluster=None,
                support={direction: 0.0 for direction in directions},
                score=0.0,
                size_rank=0.0,
                proximity=0.0,
                wall_candidate_count=len(wall_clusters),
                generic_candidate_count=len(generic_clusters),
            )

        size_ranks = self._cluster_size_ranks(
            candidate[0] for candidate in tier
        )
        scored: list[
            tuple[float, float, float, _FrontierCluster, dict[int, float]]
        ] = []
        for cluster, support in tier:
            size_rank = size_ranks[cluster.size]
            proximity = 1.0 / (
                1.0 + cluster.distance / self.frontier_distance_band
            )
            score = (
                self.frontier_cluster_size_weight * size_rank
                + self.frontier_cluster_proximity_weight * proximity
            )
            if cluster.touches_wall:
                score += (
                    self.wall_continuation_weight
                    * cluster.continuation_alignment
                )
            scored.append((score, size_rank, proximity, cluster, support))

        score, size_rank, proximity, cluster, support = min(
            scored,
            key=lambda item: (
                -item[0],
                -item[3].size,
                item[3].distance,
                item[3].cells[0],
            ),
        )
        return _ClusterSelection(
            cluster=cluster,
            support=support,
            score=score,
            size_rank=size_rank,
            proximity=proximity,
            wall_candidate_count=len(wall_clusters),
            generic_candidate_count=len(generic_clusters),
        )

    def _actionable_cluster_support(
        self,
        clusters: Iterable[_FrontierCluster],
        directions: tuple[int, ...],
        step_targets: dict[int, Position],
        *,
        sensor_range: float,
        half_fov: float,
    ) -> tuple[tuple[_FrontierCluster, dict[int, float]], ...]:
        """Attach heading support to components visible from a candidate step."""
        actionable = []
        for cluster in clusters:
            support = self._cone_point_support_scores(
                directions,
                step_targets,
                cluster.cells,
                sensor_range=sensor_range,
                half_fov=half_fov,
            )
            if max(support.values(), default=0.0) > 0.0:
                actionable.append((cluster, support))
        return tuple(actionable)

    @staticmethod
    def _cluster_size_ranks(
        clusters: Iterable[_FrontierCluster],
    ) -> dict[int, float]:
        """Return normalized ordinal ranks with the largest size at one."""
        sizes = sorted({cluster.size for cluster in clusters})
        if len(sizes) == 1:
            return {sizes[0]: 1.0}
        denominator = float(len(sizes) - 1)
        return {
            size: index / denominator
            for index, size in enumerate(sizes)
        }

    @staticmethod
    def _neighbor_adjacency(mask: np.ndarray) -> np.ndarray:
        """Return cells adjacent to at least one true eight-neighbor."""
        return eight_neighbor_adjacency(mask)

    def _cone_point_support_scores(
        self,
        directions: Iterable[int],
        step_targets: dict[int, Position],
        support_points: Iterable[Position],
        *,
        sensor_range: float,
        half_fov: float,
    ) -> dict[int, float]:
        """Estimate boundary evidence visible after each candidate step."""
        points = np.asarray(tuple(support_points), dtype=np.float32)
        if points.size == 0:
            return {int(direction): 0.0 for direction in directions}
        point_x = points[:, 0]
        point_y = points[:, 1]
        scores: dict[int, float] = {}
        current = self.drone.snapshot().position
        for direction in directions:
            target = step_targets[direction]
            lookahead_x = (float(current[0]) + float(target[0])) / 2.0
            lookahead_y = (float(current[1]) + float(target[1])) / 2.0
            delta_x = point_x - lookahead_x
            delta_y = point_y - lookahead_y
            distance = np.hypot(delta_x, delta_y)
            angles = np.degrees(np.arctan2(delta_x, -delta_y))
            angle_delta = (
                angles - float(direction) + 180.0
            ) % 360.0 - 180.0
            visible = (
                (distance > 0.0)
                & (distance <= sensor_range)
                & (np.abs(angle_delta) <= half_fov)
            )
            if not np.any(visible):
                scores[direction] = 0.0
                continue
            scores[direction] = float(np.sum(
                1.0 - 0.5 * distance[visible] / max(sensor_range, 1.0),
                dtype=np.float64,
            ))
        return scores

    def _separation_scores(
        self,
        directions: Iterable[int],
        *,
        sensor_range: float,
    ) -> tuple[dict[int, float], int]:
        """Prefer headings away from nearby peers and initial launch overlap."""
        drone = self.drone
        current = drone.snapshot().position
        positions = tuple(self.dependencies.get_drone_positions())
        peer_positions = tuple(
            position
            for drone_id, position in positions
            if int(drone_id) != int(drone.id)
        )
        separation_radius = max(sensor_range * 2.0, drone.step * 8.0)
        vector_x = 0.0
        vector_y = 0.0
        for peer_position in peer_positions:
            delta_x = float(current[0] - peer_position[0])
            delta_y = float(current[1] - peer_position[1])
            distance = math.hypot(delta_x, delta_y)
            if distance <= 1e-9 or distance >= separation_radius:
                continue
            strength = (1.0 - distance / separation_radius) ** 2
            vector_x += delta_x / distance * strength
            vector_y += delta_y / distance * strength

        drone_count = int(drone.settings.mission_config.num_drones)
        if drone_count > 1:
            launch_distance = math.dist(current, drone.start_pos)
            launch_strength = max(
                0.0,
                1.0 - launch_distance / separation_radius,
            )
            sector_heading = 360.0 * float(drone.id) / drone_count
            sector_radians = math.radians(sector_heading)
            vector_x += math.sin(sector_radians) * launch_strength
            vector_y -= math.cos(sector_radians) * launch_strength

        magnitude = math.hypot(vector_x, vector_y)
        if magnitude <= 1e-9:
            scores = {int(direction): 0.0 for direction in directions}
            return scores, len(peer_positions)
        preferred_heading = math.degrees(
            math.atan2(vector_x, -vector_y)
        ) % 360.0
        scores = {
            int(direction): (
                1.0
                + math.cos(math.radians(
                    self._angular_distance(direction, preferred_heading)
                ))
            ) / 2.0
            for direction in directions
        }
        return scores, len(peer_positions)

    @staticmethod
    def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
        """Scale nonnegative heading evidence into the unit interval."""
        maximum = max(scores.values(), default=0.0)
        if maximum <= 0.0:
            return {direction: 0.0 for direction in scores}
        return {
            direction: max(0.0, value) / maximum
            for direction, value in scores.items()
        }

    def _direction_candidates(
        self,
        *,
        cone_center: float,
        half_fov: float,
    ) -> tuple[list[int], list[Position], dict[int, Position]]:
        """Return collision-free headings and their look-ahead coordinates."""
        drone = self.drone
        current = drone.snapshot().position
        valid_directions: list[int] = []
        border_targets: list[Position] = []
        step_targets: dict[int, Position] = {}

        for direction in range(360):
            if self._angular_distance(direction, cone_center) > half_fov:
                continue
            border = next_cell_coords(
                *current,
                drone.radius + 1,
                direction,
            )
            step_target = next_cell_coords(
                *current,
                drone.step,
                direction,
            )
            if not drone.runtime_state.graph_is_valid(current, border):
                continue
            if not drone.runtime_state.graph_is_valid(current, step_target):
                continue
            valid_directions.append(direction)
            border_targets.append(border)
            step_targets[direction] = step_target
        return valid_directions, border_targets, step_targets

    @staticmethod
    def _angular_distance(first: float, second: float) -> float:
        """Return the shortest absolute distance between two headings."""
        return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)

    def _recover_from_stagnation(self) -> bool:
        """Redirect a low-information random walk toward local unknown space."""
        distance = self._stagnation_distance_travelled
        if distance < self.stagnation_distance:
            return False

        progress = self.drone.slam_map.progress_snapshot()
        sensor_cells = max(
            0,
            progress.sensor_newly_known_cells
            - self._stagnation_sensor_baseline,
        )
        sensor_cells_per_px = sensor_cells / max(distance, 1e-9)
        stagnant = (
            sensor_cells_per_px
            < self.stagnation_min_sensor_cells_per_px
        )
        self._trace(
            "drone_stagnation_window",
            travelled_distance=distance,
            sensor_newly_known_cells=sensor_cells,
            sensor_cells_per_px=sensor_cells_per_px,
            minimum_sensor_cells_per_px=(
                self.stagnation_min_sensor_cells_per_px
            ),
            stagnant=stagnant,
            slam_version=progress.version,
        )
        self._reset_stagnation_window(progress)
        if not stagnant:
            return False

        self.rebuild_frontiers(
            stride=self.frontier_stride,
            confidence_threshold=self.frontier_confidence_threshold,
        )
        frontiers = self.drone.snapshot().frontiers
        self._trace(
            "drone_stagnation_detected",
            travelled_distance=distance,
            sensor_newly_known_cells=sensor_cells,
            sensor_cells_per_px=sensor_cells_per_px,
            frontier_count=len(frontiers),
        )
        if self._start_frontier_scan(
            frontiers,
            reason="local_unknown_frontier",
        ):
            return True
        if self.reach_border(
            avoid_recent_trail=True,
            recovery_reason="stagnation",
        ):
            return True

        self._trace(
            "drone_stagnation_unresolved",
            frontier_count=len(frontiers),
            state=self._snapshot_summary(),
        )
        return False

    def _reset_stagnation_window(self, progress: Any | None = None) -> None:
        """Start a fresh sensor-local productivity window."""
        current = progress or self.drone.slam_map.progress_snapshot()
        self._stagnation_sensor_baseline = (
            current.sensor_newly_known_cells
        )
        self._stagnation_distance_travelled = 0.0

    def _start_frontier_scan(
        self,
        frontiers: Iterable[Position],
        *,
        reason: str,
    ) -> bool:
        """Face one visible unknown boundary and wait for its sensor scan."""
        drone = self.drone
        snapshot = drone.snapshot()
        slam = drone.slam_map.snapshot(point_limit=0)
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        unknown = (
            (occupancy == UNKNOWN)
            | (confidence < self.frontier_confidence_threshold)
        )
        if self._has_exploration_scope():
            unknown &= self._exploration_window_mask(unknown.shape, (0, 0))
        vision_sensor = getattr(
            getattr(drone, "sensor_controller", None),
            "vision_sensor",
            None,
        )
        sensor_range = float(getattr(
            vision_sensor,
            "max_range",
            drone.radius * 4,
        ))
        current_x, current_y = snapshot.position
        height, width = unknown.shape
        supported: list[
            tuple[float, Position, Position, int, float, int]
        ] = []

        for target in frontiers:
            if not self._in_exploration_scope(target):
                continue
            target_x, target_y = target
            if not (0 <= target_x < width and 0 <= target_y < height):
                continue
            distance = math.dist(snapshot.position, target)
            if distance > sensor_range:
                continue
            if (
                target != snapshot.position
                and not drone.runtime_state.graph_is_valid(
                    snapshot.position,
                    target,
                )
            ):
                continue

            unknown_neighbors: list[Position] = []
            for neighbor_y in range(
                max(0, target_y - 1),
                min(height, target_y + 2),
            ):
                for neighbor_x in range(
                    max(0, target_x - 1),
                    min(width, target_x + 2),
                ):
                    if unknown[neighbor_y, neighbor_x]:
                        unknown_neighbors.append((neighbor_x, neighbor_y))
            if not unknown_neighbors:
                continue

            distance_weight = 1.0 - 0.5 * min(
                1.0,
                distance / max(sensor_range, 1.0),
            )
            weight = len(unknown_neighbors) * distance_weight
            for unknown_target in unknown_neighbors:
                delta_x = unknown_target[0] - current_x
                delta_y = unknown_target[1] - current_y
                if delta_x == 0 and delta_y == 0:
                    continue
                angle = math.degrees(math.atan2(delta_x, -delta_y)) % 360.0
                supported.append((
                    weight,
                    target,
                    unknown_target,
                    len(unknown_neighbors),
                    distance,
                    int(round(angle)) % 360,
                ))

        if not supported:
            return False

        best_score = max(item[0] for item in supported)
        if best_score <= 0.0:
            return False
        strongest = [
            item for item in supported
            if math.isclose(item[0], best_score, rel_tol=1e-9)
        ]
        nearest_distance = min(item[4] for item in strongest)
        strongest = [
            item for item in strongest
            if math.isclose(item[4], nearest_distance, abs_tol=1e-9)
        ]
        best_directions = sorted({item[5] for item in strongest})
        chosen_direction = drone.exploration_policy.choose_direction(
            best_directions
        )
        selected = max(
            (item for item in strongest if item[5] == chosen_direction),
            key=lambda item: (item[3], item[1], item[2]),
        )
        geometry = None
        if selected[1] in self._last_raw_frontiers:
            geometry = self._frontier_geometry_signature(
                selected[1],
                self._last_global_raw_frontiers,
            )
        sensor = getattr(drone, "sensor_controller", None)
        last_scan = getattr(sensor, "last_completed_scan", None)
        progress = drone.slam_map.progress_snapshot()
        expected_pose = (
            int(snapshot.position[0]),
            int(snapshot.position[1]),
            round(float(chosen_direction) % 360.0, 3),
        )
        reuse_completed_scan = bool(
            last_scan is not None
            and last_scan.pose == expected_pose
        )
        published_sequence = (
            max(0, int(last_scan.sequence))
            if last_scan is not None
            else 0
        )
        minimum_sequence = published_sequence
        if reuse_completed_scan:
            minimum_sequence -= 1
        requested_at = self._simulation_time()

        self._pending_frontier_scan = _PendingFrontierScan(
            position=snapshot.position,
            heading=chosen_direction,
            resume_heading=float(snapshot.heading_deg),
            frontier_target=selected[1],
            unknown_target=selected[2],
            reason=reason,
            minimum_scan_sequence=minimum_sequence,
            baseline_geometry=geometry,
            requested_at=requested_at,
            deadline=(
                requested_at + _PENDING_FRONTIER_SCAN_TIMEOUT_SECONDS
            ),
        )
        drone.runtime_state.reorient(chosen_direction)
        self._trace(
            "drone_stagnation_scan_started",
            position=snapshot.position,
            incoming_heading=snapshot.heading_deg,
            direction=chosen_direction,
            frontier_target=selected[1],
            unknown_target=selected[2],
            frontier_distance=selected[4],
            unknown_neighbor_count=selected[3],
            unknown_support_score=best_score,
            candidate_heading_count=len(best_directions),
            minimum_scan_sequence=minimum_sequence,
            published_scan_sequence=published_sequence,
            slam_completed_scan_sequence=(
                progress.completed_scan_sequence
            ),
            reused_completed_scan=reuse_completed_scan,
            timeout_seconds=_PENDING_FRONTIER_SCAN_TIMEOUT_SECONDS,
            slam_version=slam.version,
            reason=reason,
        )
        return True

    def _advance_pending_frontier_scan(self) -> None:
        """Finish a requested scan, then restore a movement-safe heading."""
        pending = self._pending_frontier_scan
        if pending is None:
            return
        sensor = getattr(self.drone, "sensor_controller", None)
        completion = getattr(sensor, "last_completed_scan", None)
        expected_pose = (
            pending.position[0],
            pending.position[1],
            round(float(pending.heading) % 360.0, 3),
        )
        if (
            completion is None
            or completion.sequence <= pending.minimum_scan_sequence
            or completion.pose != expected_pose
        ):
            now = self._simulation_time()
            if now < pending.deadline:
                return
            progress = self.drone.slam_map.progress_snapshot()
            self._pending_frontier_scan = None
            self._trace(
                "drone_stagnation_scan_timed_out",
                position=pending.position,
                direction=pending.heading,
                resume_heading=pending.resume_heading,
                frontier_target=pending.frontier_target,
                unknown_target=pending.unknown_target,
                reason=pending.reason,
                expected_pose=expected_pose,
                minimum_scan_sequence=pending.minimum_scan_sequence,
                published_scan_sequence=(
                    None if completion is None else completion.sequence
                ),
                published_scan_pose=(
                    None if completion is None else completion.pose
                ),
                slam_completed_scan_sequence=(
                    progress.completed_scan_sequence
                ),
                waited_seconds=max(0.0, now - pending.requested_at),
                timeout_seconds=(
                    _PENDING_FRONTIER_SCAN_TIMEOUT_SECONDS
                ),
            )
            self._restore_movement_heading_after_scan(pending)
            return

        sensor_cells = max(0, int(completion.newly_known_cells))
        confidence_gain = max(0.0, float(completion.confidence_gain))
        productive = sensor_cells > 0 or confidence_gain > 1e-9
        self.rebuild_frontiers(
            stride=self.frontier_stride,
            confidence_threshold=self.frontier_confidence_threshold,
            global_heading=pending.resume_heading,
        )
        current_geometry = None
        if pending.frontier_target in self._last_raw_frontiers:
            current_geometry = self._frontier_geometry_signature(
                pending.frontier_target,
                self._last_global_raw_frontiers,
            )

        suppressed = False
        disposition = "sensor_gain"
        if not productive:
            if current_geometry is None:
                disposition = "frontier_resolved"
            elif (
                pending.baseline_geometry is not None
                and current_geometry != pending.baseline_geometry
            ):
                disposition = "geometry_changed"
            else:
                self._suppress_frontier_target(
                    pending.frontier_target,
                    reason="zero_gain_directed_scan",
                    whole_component=True,
                )
                suppressed = True
                disposition = "unchanged_geometry_suppressed"

        self._pending_frontier_scan = None
        self._trace(
            "drone_stagnation_scan_completed",
            position=pending.position,
            direction=pending.heading,
            frontier_target=pending.frontier_target,
            unknown_target=pending.unknown_target,
            reason=pending.reason,
            completed_scan_sequence=completion.sequence,
            sensor_newly_known_cells=sensor_cells,
            sensor_confidence_gain=confidence_gain,
            productive=productive,
            frontier_suppressed=suppressed,
            disposition=disposition,
            slam_version=self.drone.slam_map.version,
        )
        self._restore_movement_heading_after_scan(pending)

    def _restore_movement_heading_after_scan(
        self,
        pending: _PendingFrontierScan,
    ) -> bool:
        """Restore the closest safe heading to the pre-scan travel heading."""
        drone = self.drone
        snapshot = drone.snapshot()
        valid_directions, _border_targets, _step_targets = (
            self._direction_candidates(
                cone_center=snapshot.heading_deg,
                half_fov=180.0,
            )
        )
        if not valid_directions:
            self._trace(
                "drone_stagnation_scan_no_safe_exit",
                position=snapshot.position,
                scan_direction=pending.heading,
                resume_heading=pending.resume_heading,
                frontier_target=pending.frontier_target,
            )
            self._reset_stagnation_window()
            return False

        resume_heading = float(pending.resume_heading) % 360.0
        chosen_direction = min(
            valid_directions,
            key=lambda direction: (
                self._angular_distance(direction, resume_heading),
                int(direction),
            ),
        )
        resume_delta = self._angular_distance(
            chosen_direction,
            resume_heading,
        )
        drone.runtime_state.reorient(chosen_direction)
        self._trace(
            "drone_stagnation_scan_exit_reoriented",
            position=snapshot.position,
            incoming_heading=snapshot.heading_deg,
            direction=chosen_direction,
            scan_direction=pending.heading,
            resume_heading=resume_heading,
            resume_heading_delta=resume_delta,
            exact_resume=resume_delta <= 1e-9,
            frontier_target=pending.frontier_target,
            valid_direction_count=len(valid_directions),
        )
        self._reset_stagnation_window()
        return True

    def explore(
        self,
        valid_directions: list[int],
        border_targets: list[Position],
        chosen_target: Position,
    ) -> bool:
        """Walk one collision-checked straight step without invoking A*."""
        drone = self.drone
        snapshot = drone.snapshot()
        assigned_borders = tuple(
            target for target in border_targets
            if self._in_exploration_scope(target)
        )
        drone.runtime_state.begin_exploration(
            snapshot.direction,
            assigned_borders,
        )
        if not self._permitted_random_step(snapshot.position, chosen_target):
            return False
        path = bresenham_line_points(
            snapshot.position[0],
            snapshot.position[1],
            chosen_target[0],
            chosen_target[1],
        )
        followed = self._follow_path(path, source="random_step")
        self._trace(
            "drone_random_step",
            direction=snapshot.direction,
            target=chosen_target,
            valid_direction_count=len(valid_directions),
            completed=followed,
        )
        return followed

    def reach_border(
        self,
        *,
        avoid_recent_trail: bool = False,
        recovery_reason: str = "boxed_in",
        preferred_target: Position | None = None,
    ) -> bool:
        """Use A* to reach the nearest viable SLAM frontier."""
        drone = self.drone
        snapshot = drone.snapshot()
        frontiers = sorted(
            (target for target in snapshot.frontiers
             if self._in_exploration_scope(target)),
            key=lambda target: self._distance_from(snapshot.position, target),
        )
        if preferred_target is not None:
            frontiers = [
                target for target in frontiers
                if target == preferred_target
            ]
        if not frontiers:
            return False

        recent_trail = (
            self._recent_trail_positions()
            if avoid_recent_trail else ()
        )
        if avoid_recent_trail:
            eligible = tuple(
                target
                for target in frontiers
                if target not in self._suppressed_frontier_geometry
                and not self._target_near_recent_trail(
                    target,
                    recent_trail,
                )
            )
            self._trace(
                "drone_stagnation_frontier_filter",
                frontier_count=len(frontiers),
                eligible_frontier_count=len(eligible),
                recent_trail_point_count=len(recent_trail),
                recent_trail_clearance=self._recent_trail_clearance(),
            )
            frontiers = list(eligible)
            if not frontiers:
                return False

        now = self._simulation_time()
        for target in frontiers:
            if target in self._suppressed_frontier_geometry:
                continue
            current = drone.snapshot().position
            if target == current:
                self._trace_frontier_arrival(
                    target,
                    recovery_reason=recovery_reason,
                    route_distance=0.0,
                    direct_distance=0.0,
                    route_circuity=1.0,
                )
                if (
                    (recovery_reason == "stagnation" or self._has_exploration_scope())
                    and self._start_frontier_scan(
                        (target,),
                        reason="frontier_arrival_unknown",
                    )
                ):
                    return True
                self._suppress_reached_border(target)
                self._reorient_after_border(target)
                return True
            if now < self.border_retry_until.get(target, 0.0):
                continue

            result = self._compute_path(current, target)
            path = result.path
            route_distance = self._path_distance(
                current,
                tuple(path),
                len(path),
            )
            direct_distance = math.dist(current, target)
            segment_direct_distance = math.dist(
                current,
                path[-1] if path else current,
            )
            segment_circuity = self._route_circuity(
                route_distance,
                segment_direct_distance,
            )
            route_circuity = (
                self._route_circuity(route_distance, direct_distance)
                if result.status == PATH_COMPLETE
                else None
            )
            self._trace(
                "drone_border_path",
                start=current,
                target=target,
                path_length=len(path),
                path_status=result.status,
                path_iterations=result.iterations,
                path_remaining_distance=result.remaining_distance,
                segment_endpoint=path[-1] if path else None,
                recovery_reason=recovery_reason,
                route_distance=route_distance,
                direct_distance=direct_distance,
                route_circuity=route_circuity,
                segment_direct_distance=segment_direct_distance,
                segment_circuity=segment_circuity,
                maximum_route_circuity=(
                    self.maximum_frontier_path_circuity
                ),
            )
            if recovery_reason == "stagnation":
                self._trace(
                    "drone_stagnation_frontier_path",
                    start=current,
                    target=target,
                    path_length=len(path),
                    path_status=result.status,
                )
            if len(path) <= 1:
                self.border_retry_until[target] = (
                    now + self.border_retry_cooldown
                )
                continue
            if result.status == PATH_PARTIAL_LIMIT:
                if not self._accept_partial_endpoint(current, target, path[-1]):
                    self.border_retry_until[target] = (
                        now + self.border_retry_cooldown
                    )
                    continue
                self._pending_frontier_route = _PendingFrontierRoute(
                    target=target,
                    recovery_reason=recovery_reason,
                )
                return self._follow_path(
                    path,
                    source="border_astar_partial",
                )
            if result.status != PATH_COMPLETE:
                self.border_retry_until[target] = (
                    now + self.border_retry_cooldown
                )
                continue
            if (
                route_circuity is not None
                and route_circuity > self.maximum_frontier_path_circuity
            ):
                self._trace(
                    "drone_frontier_route_rejected",
                    start=current,
                    target=target,
                    route_distance=route_distance,
                    direct_distance=direct_distance,
                    route_circuity=route_circuity,
                    maximum_route_circuity=(
                        self.maximum_frontier_path_circuity
                    ),
                    recovery_reason=recovery_reason,
                    reason="excessive_path_circuity",
                )
                self._suppress_frontier_target(
                    target,
                    reason="excessive_path_circuity",
                    whole_component=True,
                )
                continue
            self._clear_partial_route(target)
            self._pending_frontier_route = None
            if not self._follow_path(path, source="border_astar"):
                return False

            self._trace_frontier_arrival(
                target,
                recovery_reason=recovery_reason,
                route_distance=route_distance,
                direct_distance=direct_distance,
                route_circuity=(
                    1.0 if route_circuity is None else route_circuity
                ),
            )

            if (
                (recovery_reason == "stagnation" or self._has_exploration_scope())
                and self._start_frontier_scan(
                    (target,),
                    reason="frontier_arrival_unknown",
                )
            ):
                return True
            self._suppress_reached_border(target)
            self._reorient_after_border(target)
            return True
        return False

    def _recent_trail_positions(self) -> tuple[Position, ...]:
        """Return the recent breadcrumb suffix covering one gain window."""
        history = self.drone.snapshot().path_history
        if not history:
            return ()
        recent = [history[-1]]
        accumulated = 0.0
        for point in reversed(history[:-1]):
            accumulated += math.dist(point, recent[-1])
            recent.append(point)
            if accumulated >= self.stagnation_distance:
                break
        return tuple(recent)

    def _recent_trail_clearance(self) -> float:
        """Return the minimum separation for a stagnation A* target."""
        return float(max(self.drone.step * 2, self.frontier_stride * 2))

    def _target_near_recent_trail(
        self,
        target: Position,
        recent_trail: Iterable[Position],
    ) -> bool:
        """Return whether a target would send recovery back onto recent path."""
        clearance = self._recent_trail_clearance()
        return any(
            math.dist(target, point) <= clearance
            for point in recent_trail
        )

    def _suppress_reached_border(self, target: Position) -> None:
        """Retire a reached target until its local geometry changes."""
        self._suppress_frontier_target(
            target,
            reason="reached_unchanged_local_geometry",
        )

    def _suppress_frontier_target(
        self,
        target: Position,
        *,
        reason: str,
        whole_component: bool = False,
    ) -> None:
        """Retire one target, or its component, until geometry changes."""
        targets = (target,)
        if whole_component:
            targets = self._frontier_component_targets.get(target, ())
            if not targets and self._last_frontier_mask is not None:
                targets = self._sampled_frontier_component(
                    self._last_frontier_mask,
                    target,
                    self._last_raw_frontiers,
                )
                for candidate in targets:
                    self._frontier_component_targets[candidate] = targets
            if not targets:
                targets = (target,)
        for candidate in targets:
            geometry = None
            if candidate in self._last_raw_frontiers:
                geometry = self._frontier_geometry_signature(
                    candidate,
                    self._last_global_raw_frontiers,
                )
            self._suppressed_frontier_geometry[candidate] = geometry
            self.drone.runtime_state.remove_frontier(candidate)
            self.border_retry_until.pop(candidate, None)
            self._clear_partial_route(candidate)
        assigned_component_id = self._assigned_frontier_component_id(targets)
        if assigned_component_id is not None:
            self._sector_suppression_reasons.setdefault(
                assigned_component_id,
                set(),
            ).add(str(reason))
            self._sector_suppression_targets.setdefault(
                assigned_component_id,
                set(),
            ).update(targets)
        if (
            self._pending_frontier_route is not None
            and self._pending_frontier_route.target in targets
        ):
            self._pending_frontier_route = None
        target_geometry = self._suppressed_frontier_geometry.get(target)
        self._trace(
            "drone_border_target_suppressed",
            target=target,
            reason=reason,
            local_geometry_point_count=(
                0 if target_geometry is None else len(target_geometry)
            ),
            suppressed_target_count=len(targets),
            suppressed_target_sample=targets[:12],
            assigned_component_id=assigned_component_id,
            slam_version=self.drone.slam_map.version,
        )

    def _assigned_frontier_component_id(
        self,
        targets: Iterable[Position],
    ) -> int | None:
        """Match sampled local targets to the current rover component ID."""
        assignment = self._sector_assignment
        if assignment is None or not assignment.frontier_components:
            return None
        target_set = frozenset(targets)
        if not target_set:
            return None
        ranked: list[tuple[int, float, int]] = []
        for component in assignment.frontier_components:
            overlap = len(target_set & component.cells)
            nearest = min(
                (
                    (target[0] - point[0]) ** 2
                    + (target[1] - point[1]) ** 2
                    for target in target_set
                    for point in component.cells
                ),
                default=float("inf"),
            )
            ranked.append((overlap, -float(nearest), component.component_id))
        overlap, negative_distance, component_id = max(ranked)
        maximum_distance = float(self.frontier_stride * 2) ** 2
        if overlap <= 0 and -negative_distance > maximum_distance:
            return None
        return int(component_id)

    def _reorient_after_border(self, target: Position) -> bool:
        """Turn toward a usable exit after A* reaches an escape border."""
        drone = self.drone
        snapshot = drone.snapshot()
        valid_directions, _border_targets, step_targets = (
            self._direction_candidates(
                cone_center=snapshot.heading_deg,
                half_fov=180.0,
            )
        )
        if not valid_directions:
            self._trace(
                "drone_recovery_no_outgoing_heading",
                target=target,
                position=snapshot.position,
                incoming_heading=snapshot.heading_deg,
            )
            return False

        chosen_direction = drone.exploration_policy.choose_direction(
            valid_directions
        )
        self._apply_reorientation(
            chosen_direction,
            step_targets,
            event="drone_recovery_reoriented",
            valid_direction_count=len(valid_directions),
            target=target,
        )
        return True

    def _apply_reorientation(
        self,
        chosen_direction: int,
        step_targets: dict[int, Position],
        *,
        event: str,
        valid_direction_count: int,
        **fields: Any,
    ) -> None:
        """Rotate in place, retain one exit border, and reset gain tracking."""
        drone = self.drone
        snapshot = drone.snapshot()
        drone.runtime_state.reorient(chosen_direction)
        chosen_border = next_cell_coords(
            *snapshot.position,
            drone.radius + 1,
            chosen_direction,
        )
        if self._in_exploration_scope(chosen_border):
            drone.runtime_state.merge_frontiers((chosen_border,))
        self._trace(
            event,
            position=snapshot.position,
            incoming_heading=snapshot.heading_deg,
            direction=chosen_direction,
            step_target=step_targets[chosen_direction],
            border_target=chosen_border,
            valid_direction_count=valid_direction_count,
            **fields,
        )
        self._reset_stagnation_window()

    def reach_start_point(self) -> bool:
        """Use A* to return to the drone's starting position."""
        drone = self.drone
        current = drone.snapshot().position
        if current == drone.start_pos:
            return True
        result = self._compute_path(current, drone.start_pos)
        path = result.path
        self._trace(
            "drone_homing_path",
            start=current,
            target=drone.start_pos,
            path_length=len(path),
            path_status=result.status,
            path_iterations=result.iterations,
            path_remaining_distance=result.remaining_distance,
            segment_endpoint=path[-1] if path else None,
        )
        if not path:
            return False
        if result.status == PATH_PARTIAL_LIMIT:
            if not self._accept_partial_endpoint(
                current,
                drone.start_pos,
                path[-1],
            ):
                return False
            self._follow_path(path, source="home_astar_partial")
            return False
        if result.status != PATH_COMPLETE:
            return False
        self._clear_partial_route(drone.start_pos)
        return self._follow_path(
            path,
            source="home_astar",
        ) and drone.snapshot().position == drone.start_pos

    def update_borders(self) -> None:
        """Refresh SLAM frontier targets when the cooldown permits."""
        self.maybe_rebuild_frontiers()

    def maybe_rebuild_frontiers(self) -> bool:
        """Rebuild frontiers at most once per configured cooldown."""
        now = self._simulation_time()
        if not self.drone.runtime_state.reserve_frontier_rebuild(now):
            return False
        self.rebuild_frontiers(
            stride=self.frontier_stride,
            confidence_threshold=self.frontier_confidence_threshold,
        )
        return True

    def rebuild_frontiers(
        self,
        *,
        stride: int = 4,
        confidence_threshold: float = 0.6,
        global_heading: float | None = None,
    ) -> None:
        """Recalculate local frontiers while publishing that UI state."""
        self._begin_calculation(
            "Recalculating",
            "refreshing local frontiers",
        )
        try:
            self._rebuild_frontiers(
                stride=stride,
                confidence_threshold=confidence_threshold,
                global_heading=global_heading,
            )
        finally:
            self._end_calculation("Recalculating")

    def _rebuild_frontiers(
        self,
        *,
        stride: int = 4,
        confidence_threshold: float = 0.6,
        global_heading: float | None = None,
    ) -> None:
        """Extract known-free cells bordering unknown local SLAM cells."""
        slam = self.drone.slam_map.snapshot(point_limit=0)
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        frontier_mask = known_free_frontier_mask(
            occupancy,
            confidence,
            confidence_threshold,
        )
        sampling_stride = max(1, int(stride))
        sampled = frontier_mask[::sampling_stride, ::sampling_stride]
        rows, columns = np.where(sampled)
        all_raw_frontiers = tuple(
            (
                int(column * sampling_stride),
                int(row * sampling_stride),
            )
            for row, column in zip(rows, columns)
        )
        eligible_mask = frontier_mask & self._exploration_window_mask(
            frontier_mask.shape, (0, 0),
        )
        fallback_targets: set[Position] = set()
        if self._has_exploration_scope():
            sampled_targets = set(all_raw_frontiers)
            for component in eight_connected_components(eligible_mask):
                if not sampled_targets.intersection(component):
                    fallback_targets.add(component[0])
            # Keep off-stride suppression geometry alive even when its old
            # scope is absent from this assignment.
            fallback_targets.update(
                point for point in self._suppressed_frontier_geometry
                if 0 <= point[1] < frontier_mask.shape[0]
                and 0 <= point[0] < frontier_mask.shape[1]
                and frontier_mask[point[1], point[0]]
            )
            all_raw_frontiers = tuple(sorted(
                sampled_targets | fallback_targets, key=lambda p: (p[1], p[0]),
            ))
        raw_frontiers = tuple(
            target
            for target in all_raw_frontiers
            if self._in_exploration_scope((
                target[0],
                target[1],
            ))
        )
        all_raw_frontier_set = frozenset(all_raw_frontiers)
        raw_frontier_set = frozenset(raw_frontiers)
        self._last_frontier_mask = frontier_mask
        self._frontier_component_targets.clear()
        reactivated: list[Position] = []
        for target, stored_geometry in tuple(
            self._suppressed_frontier_geometry.items()
        ):
            if target not in all_raw_frontier_set:
                self._suppressed_frontier_geometry.pop(target, None)
                continue
            if target not in raw_frontier_set:
                continue
            current_geometry = self._frontier_geometry_signature(
                target,
                all_raw_frontier_set,
            )
            if stored_geometry is None:
                self._suppressed_frontier_geometry[target] = current_geometry
            elif current_geometry != stored_geometry:
                self._suppressed_frontier_geometry.pop(target, None)
                reactivated.append(target)

        suppressed = tuple(
            target
            for target in raw_frontiers
            if target in self._suppressed_frontier_geometry
        )
        frontiers = tuple(
            target
            for target in raw_frontiers
            if target not in self._suppressed_frontier_geometry
        )
        self._last_raw_frontiers = raw_frontier_set
        self._last_global_raw_frontiers = all_raw_frontier_set
        self._frontier_slam_version = int(slam.version)
        self.drone.runtime_state.replace_frontiers(frontiers)
        self.border_retry_until = {
            target: retry_until
            for target, retry_until in self.border_retry_until.items()
            if target in raw_frontier_set
        }
        self._trace(
            "drone_frontiers_rebuilt",
            frontier_count=len(frontiers),
            frontier_sample=frontiers[:12],
            raw_frontier_count=len(raw_frontiers),
            global_raw_frontier_count=len(all_raw_frontiers),
            scope_excluded_frontier_count=sum(
                self._in_assigned_sector(target)
                and not self._in_exploration_scope(target)
                for target in all_raw_frontiers
            ),
            scope_frontier_pixels=int(np.count_nonzero(eligible_mask)),
            fallback_frontier_count=len(fallback_targets),
            suppressed_frontier_count=len(suppressed),
            suppressed_frontier_sample=suppressed[:12],
            reactivated_frontier_count=len(reactivated),
            reactivated_frontier_sample=tuple(reactivated[:12]),
            slam_version=slam.version,
        )
        if self._global_frontier_cache is None:
            state = self.drone.snapshot()
            self._ensure_global_frontier_cache(
                current=state.position,
                heading=(
                    float(state.heading_deg)
                    if global_heading is None
                    else float(global_heading)
                ),
                slam=slam,
            )

    @staticmethod
    def _sampled_frontier_component(
        frontier_mask: np.ndarray,
        target: Position,
        sampled_frontiers: frozenset[Position],
    ) -> tuple[Position, ...]:
        """Return sampled targets in the target's full-resolution component."""
        height, width = frontier_mask.shape
        target_x, target_y = target
        if (
            not sampled_frontiers
            or not (0 <= target_x < width and 0 <= target_y < height)
            or not bool(frontier_mask[target_y, target_x])
        ):
            return ()

        pending = [target]
        visited = {target}
        while pending:
            x, y = pending.pop()
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    if offset_x == 0 and offset_y == 0:
                        continue
                    neighbor = (x + offset_x, y + offset_y)
                    if (
                        0 <= neighbor[0] < width
                        and 0 <= neighbor[1] < height
                        and neighbor not in visited
                        and bool(frontier_mask[neighbor[1], neighbor[0]])
                    ):
                        visited.add(neighbor)
                        pending.append(neighbor)
        return tuple(sorted(
            visited & sampled_frontiers,
            key=lambda point: (point[1], point[0]),
        ))

    def _frontier_geometry_signature(
        self,
        target: Position,
        frontiers: Iterable[Position],
    ) -> tuple[Position, ...]:
        """Describe sampled frontier geometry local to one reached target."""
        local_radius = max(
            float(self.drone.radius) * 2.0,
            float(self.frontier_stride) * 2.0,
        )
        radius_squared = local_radius * local_radius
        target_x, target_y = target
        offsets = (
            (point[0] - target_x, point[1] - target_y)
            for point in frontiers
            if (
                (point[0] - target_x) ** 2
                + (point[1] - target_y) ** 2
            ) <= radius_squared
        )
        return tuple(sorted(offsets))

    def mission_completed(self) -> bool:
        """Return whether the drone has completed exploration and homing."""
        if self._component_policy_enabled() or self._sector_policy_enabled():
            done = self.drone.snapshot().done
        else:
            done, _returning_home = (
                self.drone.runtime_state.evaluate_mission_state()
            )
        if done:
            logger.info("Drone %s has completed the mission", self.drone.id)
        return done

    def get_distance(self, target: Position) -> float:
        """Return the current border-priority distance for compatibility."""
        return self._distance_from(self.drone.snapshot().position, target)

    def _distance_from(self, position: Position, target: Position) -> float:
        distance = math.dist(position, target)
        if distance <= self.drone.radius:
            return float(self.drone.game.width)
        return distance

    def _compute_path(self, start: Position, goal: Position) -> PathResult:
        """Ask for a complete physical route or one capped route segment."""
        started = time.perf_counter()
        path_result: PathResult | None = None
        self._begin_calculation(
            "Pathfinding",
            f"target {goal[0]},{goal[1]}",
            target=goal,
        )
        try:
            segment_planner = self.dependencies.compute_path_segment
            if callable(segment_planner):
                result = segment_planner(start, goal)
                if isinstance(result, PathResult):
                    path_result = result

            if path_result is None:
                path = tuple(self.dependencies.compute_path(start, goal))
                status = (
                    PATH_COMPLETE
                    if path and path[-1] == goal
                    else PATH_UNREACHABLE
                )
                remaining = (
                    0.0
                    if status == PATH_COMPLETE
                    else math.dist(path[-1] if path else start, goal)
                )
                path_result = PathResult(path, status, 0, remaining)
            if path_result.status == PATH_COMPLETE:
                self._last_complete_path = tuple(path_result.path)
            return path_result
        finally:
            self._trace(
                "drone_path_request_completed",
                start=start,
                goal=goal,
                status=(
                    "error" if path_result is None else path_result.status
                ),
                iterations=(
                    0 if path_result is None else path_result.iterations
                ),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                path_point_count=(
                    0 if path_result is None else len(path_result.path)
                ),
                endpoint=(
                    None
                    if path_result is None or not path_result.path
                    else path_result.path[-1]
                ),
                remaining_distance=(
                    None
                    if path_result is None
                    else path_result.remaining_distance
                ),
            )
            self._end_calculation("Pathfinding")

    def _accept_partial_endpoint(
        self,
        start: Position,
        goal: Position,
        endpoint: Position,
    ) -> bool:
        """Accept a capped segment only when it advances to a fresh endpoint."""
        advances = math.dist(endpoint, goal) < math.dist(start, goal) - 1e-6
        recent = self._partial_route_endpoints.get(goal, ())
        accepted = advances and endpoint not in recent
        self._trace(
            "drone_astar_partial_segment",
            start=start,
            goal=goal,
            endpoint=endpoint,
            accepted=accepted,
            advances=advances,
            repeated_endpoint=endpoint in recent,
            remaining_distance=math.dist(endpoint, goal),
        )
        if accepted:
            self._partial_route_endpoints[goal] = (*recent[-7:], endpoint)
        return accepted

    def _clear_partial_route(self, goal: Position) -> None:
        """Forget loop protection after completing or retiring a route."""
        self._partial_route_endpoints.pop(goal, None)

    @staticmethod
    def _route_circuity(
        route_distance: float,
        direct_distance: float,
    ) -> float:
        """Return routed distance divided by direct displacement."""
        if direct_distance <= 1e-9:
            return 1.0 if route_distance <= 1e-9 else math.inf
        return max(1.0, float(route_distance) / float(direct_distance))

    def _trace_frontier_arrival(
        self,
        target: Position,
        *,
        recovery_reason: str,
        route_distance: float,
        direct_distance: float,
        route_circuity: float,
    ) -> None:
        """Record a physically reached frontier for reversal diagnostics."""
        assignment = self._sector_assignment
        geometry = self._frontier_geometry_signature(
            target,
            self._last_global_raw_frontiers,
        )
        self._trace(
            "drone_frontier_reached",
            target=target,
            position=self.drone.snapshot().position,
            recovery_reason=recovery_reason,
            route_distance=route_distance,
            direct_distance=direct_distance,
            route_circuity=route_circuity,
            sector_id=(None if assignment is None else assignment.sector_id),
            generation=(
                None if assignment is None else assignment.generation
            ),
            local_geometry_point_count=len(geometry),
            local_geometry_sample=geometry[:12],
            slam_version=self.drone.slam_map.version,
        )

    def _simulation_time(self) -> float:
        return float(self.dependencies.simulation_time())

    def _rendezvous_position(self) -> Position | None:
        """Return only the endpoint carried in this drone's local protocol."""
        callback = self.dependencies.get_check_in_position
        if not callable(callback):
            return None
        position = callback()
        if position is None:
            return None
        return int(position[0]), int(position[1])

    def _follow_path(
        self,
        path: Iterable[Position],
        *,
        source: str = "path",
        stop_when: Callable[[], bool] | None = None,
        stop_reason: str | Callable[[], str] | None = None,
        incidental_transit: bool = False,
        path_target: Position | None = None,
        path_status: str | None = None,
    ) -> bool:
        """Walk a path while recording the breadcrumb history used by rendering."""
        points = tuple((int(point[0]), int(point[1])) for point in path)
        started = self._simulation_time()
        start = self.drone.snapshot().position
        end = start
        moved_points = 0
        completed = True
        stopped_for_contact = False
        coverage_cell_entries = 0
        coverage_new_cell_entries = 0
        coverage_revisit_entries = 0
        coverage_repeated_edge_entries = 0
        for point_index, node in enumerate(points):
            if node == self.drone.snapshot().position:
                continue
            if not self.dependencies.pause_checkpoint():
                completed = False
                break
            previous = self.drone.snapshot().position
            if not self.drone.runtime_state.graph_is_valid(previous, node):
                completed = False
                break
            self.drone.runtime_state.move_to(node)
            end = node
            moved_points += 1
            (
                cell_entries,
                new_cell_entries,
                revisit_entries,
                repeated_edge_entries,
            ) = self._record_coverage_transition(
                previous,
                node,
                self._simulation_time(),
            )
            coverage_cell_entries += cell_entries
            coverage_new_cell_entries += new_cell_entries
            coverage_revisit_entries += revisit_entries
            coverage_repeated_edge_entries += repeated_edge_entries
            contact_checkpoint = self.dependencies.physical_contact_checkpoint
            if callable(contact_checkpoint):
                contact_checkpoint(self.drone.id)
            if stop_when is not None and stop_when():
                # A contact callback either serialized rover arrival or
                # invalidated this route from newly exchanged peer SLAM.
                stopped_for_contact = True
                completed = False
                if stop_reason == "shared_slam_invalidated":
                    execution = self._coordination_execution
                    directive = None if execution is None else execution.directive
                    self._trace(
                        "drone_route_interrupted_after_share",
                        source=source,
                        position=end,
                        slam_version=self.drone.slam_map.version,
                        directive_id=(
                            None if directive is None else directive.directive_id
                        ),
                        directive_kind=(
                            None if directive is None else directive.kind.value
                        ),
                        next_phase=(
                            None if execution is None else execution.phase
                        ),
                    )
                break
            sharing_wait = self._sharing_remaining()
            if sharing_wait > 0.0 and not self.dependencies.wait_simulation_delay(
                sharing_wait
            ):
                completed = False
                break
            if not self.dependencies.wait_simulation_delay(
                self.drone.delay / self.drone.speed_factor
            ):
                completed = False
                break
            if (
                incidental_transit
                and self._maybe_sample_incidental_transit(
                    edge_distance=math.dist(previous, node),
                    retained_route=points[point_index + 1:],
                    route_target=(path_target or points[-1]),
                    route_source=source,
                    path_status=path_status,
                )
            ):
                completed = False
                break

        travelled_distance = self._path_distance(
            start,
            points,
            moved_points,
        )
        self._stagnation_distance_travelled += travelled_distance
        actual_stop_reason = stop_reason() if callable(stop_reason) else stop_reason
        self._trace(
            "drone_motion",
            source=source,
            completed=completed,
            stopped_for_report=(
                stopped_for_contact and actual_stop_reason in {None, "report"}
            ),
            stop_reason=actual_stop_reason if stopped_for_contact else None,
            start=start,
            end=end,
            point_count=moved_points,
            travelled_distance=travelled_distance,
            started_sim_time=started,
            ended_sim_time=self._simulation_time(),
            coverage_cell_entries=coverage_cell_entries,
            coverage_new_cell_entries=coverage_new_cell_entries,
            coverage_revisit_entries=coverage_revisit_entries,
            coverage_repeated_edge_entries=(
                coverage_repeated_edge_entries
            ),
            coverage_known_cell_count=len(self._coverage_cell_visits),
            coverage_known_edge_count=len(self._coverage_edge_visits),
        )
        return completed

    @staticmethod
    def _path_distance(
        start: Position,
        points: tuple[Position, ...],
        moved_points: int,
    ) -> float:
        if moved_points <= 0:
            return 0.0
        moved: list[Position] = [start]
        for point in points:
            if point == moved[-1]:
                continue
            moved.append(point)
            if len(moved) - 1 >= moved_points:
                break
        return sum(
            math.dist(previous, current)
            for previous, current in zip(moved, moved[1:])
        )

    @staticmethod
    def _polyline_distance(path: Iterable[Position]) -> float:
        """Return exact travelled length for an already ordered path."""
        points = tuple(path)
        return sum(
            math.dist(previous, current)
            for previous, current in zip(points, points[1:])
        )

    def _trace(self, event: str, **fields: Any) -> None:
        trace = getattr(self.dependencies, "runtime_trace", None)
        if trace is not None:
            trace.record(
                event,
                sim_time=self._simulation_time(),
                drone_id=self.drone.id,
                **fields,
            )

    def _snapshot_summary(self) -> dict[str, Any]:
        snapshot = self.drone.snapshot()
        return {
            "position": snapshot.position,
            "direction": snapshot.direction,
            "frontier_count": len(snapshot.frontiers),
            "returning_home": snapshot.returning_home,
            "done": snapshot.done,
        }
