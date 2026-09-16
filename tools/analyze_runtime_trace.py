"""Summarize Cave Game runtime JSONL traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Hashable, Iterable, Mapping, Sequence


LEGACY_MCTS_BUDGET_MS = 40.0
DECISION_EVENTS = {"drone_decision", "drone_post_rebuild_decision"}
LOCAL_MCTS_EVENTS = {
    "drone_local_mcts_decision",
    "local_mcts_search",
    "mcts_search",
}
FORBIDDEN_NODE_ROLES = frozenset({"travelled", "traveled", "known_free"})
ALLOWED_GOAL_CHANGE_REASONS = frozenset({
    "selected",
    "reached",
    "invalidated",
    "route_edge_retired",
    "belief_corridor_invalid",
    "stalled",
    "reversal",
    "reservation_lost",
    "scan_complete",
    "zero_gain",
    "recovery_complete",
    "home",
    "collision",
    "goal_retired",
    "retired",
    "no_actionable_frontier",
})


@dataclass(frozen=True)
class WaypointDensityMetrics:
    """Observed waypoint topology and spatial-density measurements."""

    node_count: int
    edge_count: int
    edge_mutation_count: int
    unique_connection_count: int
    source_counts: Mapping[str, int]
    occupied_spatial_cells: int
    average_nodes_per_occupied_cell: float | None
    maximum_nodes_in_cell: int
    median_nearest_neighbor_px: float | None
    nodes_with_neighbor_within_8_px: int
    neighbor_within_8_px_rate: float | None
    nodes_with_neighbor_within_16_px: int
    neighbor_within_16_px_rate: float | None
    degree_two_nodes: int
    degree_two_rate: float | None
    reported_node_count: int | None
    reported_edge_count: int | None
    role_counts: Mapping[str, int]
    forbidden_node_role_count: int


@dataclass(frozen=True)
class DroneTargetRetentionMetrics:
    """Route-segment follow-ups and target changes for one drone."""

    completed_segments: int
    segment_followups: int
    retained_after_segment: int
    reranked_frontier_after_segment: int
    switched_after_segment: int
    local_step_after_segment: int
    rotate_after_segment: int
    unclassified_followups: int
    segments_without_followup: int
    retention_rate: float | None
    coordinate_switch_proxy_rate: float | None
    goal_identity_followups: int
    route_abandonments: int
    route_abandonment_rate: float | None
    route_cursor_followups: int
    route_cursor_continuations: int
    route_cursor_resets_or_regressions: int
    route_cursor_continuation_rate: float | None
    decision_target_transitions: int
    decision_target_switches: int


@dataclass(frozen=True)
class TargetRetentionMetrics:
    """Aggregate persistent-target and route-abandonment measurements."""

    completed_segments: int
    segment_followups: int
    retained_after_segment: int
    reranked_frontier_after_segment: int
    switched_after_segment: int
    local_step_after_segment: int
    rotate_after_segment: int
    unclassified_followups: int
    segments_without_followup: int
    retention_rate: float | None
    coordinate_switch_proxy_rate: float | None
    goal_identity_followups: int
    route_abandonments: int
    route_abandonment_rate: float | None
    route_cursor_followups: int
    route_cursor_continuations: int
    route_cursor_resets_or_regressions: int
    route_cursor_continuation_rate: float | None
    decision_target_transitions: int
    decision_target_switches: int
    by_drone: Mapping[int, DroneTargetRetentionMetrics]


@dataclass(frozen=True)
class FrontierFallbackMetrics:
    """Legacy frontier fallback reuse and regeneration measurements."""

    fallback_count: int
    zero_reward_fallbacks: int
    unique_targets: int
    repeated_target_selections: int
    regenerated_after_reach: int
    regenerated_drone_targets: tuple[tuple[int, Hashable], ...]


@dataclass(frozen=True)
class DroneABAReversalMetrics:
    """A-B-A revisit measurements for one drone's reached goals."""

    arrivals: int
    reversal_opportunities: int
    reversals: int
    reversal_rate: float | None


@dataclass(frozen=True)
class ABAReversalMetrics:
    """Aggregate A-B-A revisit measurements."""

    arrivals: int
    reversal_opportunities: int
    reversals: int
    reversal_rate: float | None
    normalized_window_start_s: float | None
    by_drone: Mapping[int, DroneABAReversalMetrics]


@dataclass(frozen=True)
class MCTSTimingMetrics:
    """MCTS iteration coverage and hard-budget timing measurements."""

    search_count: int
    timing_sample_count: int
    missing_timing_count: int
    iteration_sample_count: int
    missing_iteration_count: int
    at_most_one_iteration_count: int
    at_most_one_iteration_rate: float | None
    budget_ms: float
    budget_source: str
    over_budget_count: int
    over_budget_rate: float | None
    total_elapsed_ms: float
    mean_elapsed_ms: float | None
    median_elapsed_ms: float | None
    p95_elapsed_ms: float | None
    p99_elapsed_ms: float | None
    maximum_elapsed_ms: float | None
    zero_reward_frontier_fallbacks: int
    root_coverage_sample_count: int
    missing_root_coverage_count: int
    complete_root_coverage_count: int
    incomplete_root_coverage_count: int
    unsafe_incomplete_search_count: int


@dataclass(frozen=True)
class RouteCacheMetrics:
    """Waypoint route health, timing, reuse, and segment execution metrics."""

    route_calls: int
    successful_routes: int
    failed_routes: int
    status_counts: Mapping[str, int]
    unique_drone_target_pairs: int
    timing_sample_count: int
    total_route_time_ms: float
    mean_route_time_ms: float | None
    median_route_time_ms: float | None
    p95_route_time_ms: float | None
    maximum_route_time_ms: float | None
    cache_hits: int
    cache_misses: int
    cache_unknown: int
    cache_hit_rate: float | None
    segment_calls: int
    successful_segments: int
    astar_attempts: int
    astar_selected_paths: int
    stored_polyline_selected_paths: int
    repeated_route_calls: int
    repeated_cache_hits: int
    repeated_cache_misses: int
    repeated_cache_unknown: int
    repeated_cache_hit_rate: float | None
    persistent_edge_astar_calls: int
    connector_astar_calls: int
    unchanged_valid_intent_replans: int
    replan_reason_counts: Mapping[str, int]


@dataclass(frozen=True)
class TraceSchemaValidationMetrics:
    """Replacement-schema coverage and legacy dual-write validation."""

    replacement_event_count: int
    missing_field_count: int
    missing_fields: tuple[str, ...]
    stable_id_conflict_count: int
    legacy_field_event_count: int
    legacy_trace_fields_present: bool
    valid: bool


@dataclass(frozen=True)
class AcceptanceMetrics:
    """Every trace-measurable Phase-7 acceptance signal in one record."""

    forbidden_node_role_count: int
    average_nodes_per_occupied_cell: float | None
    neighbor_within_8_px_rate: float | None
    degree_two_rate: float | None
    maximum_nodes_in_cell: int
    repeated_route_cache_hit_rate: float | None
    route_lookup_p95_ms: float | None
    goal_changes: int
    unexplained_goal_changes: int
    second_reversal_triggers: int
    second_reversal_recoveries: int
    second_reversal_retirements: int
    watchdog_threshold_triggers: int
    watchdog_transition_delays: int
    mcts_p99_elapsed_ms: float | None
    mcts_maximum_elapsed_ms: float | None
    incomplete_root_coverage_count: int
    unsafe_incomplete_search_count: int
    persistent_edge_astar_calls: int
    connector_astar_calls: int
    unchanged_valid_intent_replans: int
    newly_known_cells_per_travelled_px: float | None
    confidence_gain_per_travelled_px: float | None
    planner_cave_map_field_count: int
    replacement_schema_valid: bool
    replacement_schema_missing_field_count: int
    legacy_trace_fields_present: bool


@dataclass(frozen=True)
class DroneInformationEfficiencyMetrics:
    """Information gain and actual travel for one drone."""

    travelled_distance_px: float
    distance_source: str
    motion_samples: int
    completed_scans: int
    gain_samples: int
    scans_missing_gain: int
    gain_telemetry_complete: bool
    newly_known_samples: int
    scans_missing_newly_known: int
    newly_known_telemetry_complete: bool
    confidence_gain_samples: int
    scans_missing_confidence_gain: int
    confidence_gain_telemetry_complete: bool
    newly_known_cells: int
    confidence_gain: float
    newly_known_cells_per_travelled_px: float | None
    confidence_gain_per_travelled_px: float | None


@dataclass(frozen=True)
class InformationEfficiencyMetrics:
    """Aggregate information gain per travelled pixel."""

    travelled_distance_px: float
    distance_source: str
    motion_samples: int
    completed_scans: int
    gain_samples: int
    scans_missing_gain: int
    gain_telemetry_complete: bool
    newly_known_samples: int
    scans_missing_newly_known: int
    newly_known_telemetry_complete: bool
    confidence_gain_samples: int
    scans_missing_confidence_gain: int
    confidence_gain_telemetry_complete: bool
    newly_known_cells: int
    confidence_gain: float
    newly_known_cells_per_travelled_px: float | None
    confidence_gain_per_travelled_px: float | None
    by_drone: Mapping[int, DroneInformationEfficiencyMetrics]


@dataclass(frozen=True)
class RuntimeTraceMetrics:
    """Structured characterization result for one runtime trace."""

    event_count: int
    waypoint_density: WaypointDensityMetrics
    target_retention: TargetRetentionMetrics
    frontier_fallbacks: FrontierFallbackMetrics
    aba_reversals: ABAReversalMetrics
    mcts: MCTSTimingMetrics
    routes: RouteCacheMetrics
    information_efficiency: InformationEfficiencyMetrics
    schema_validation: TraceSchemaValidationMetrics
    acceptance: AcceptanceMetrics


@dataclass
class _NodeRecord:
    position: tuple[float, float]
    source: str
    roles: frozenset[str] = frozenset()


@dataclass
class _EdgeRecord:
    start: Hashable | None
    end: Hashable | None


@dataclass(frozen=True)
class _RouteCursorState:
    """Comparable persistent-route identity and execution cursor."""

    route_identity: Hashable
    edge_cursor: int
    polyline_cursor: int


def _finite_float(value: Any) -> float | None:
    """Return a finite float or ``None`` for absent/invalid telemetry."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    """Return an integer or ``None`` for absent/invalid telemetry."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _boolean(value: Any) -> bool | None:
    """Return a strict boolean without treating non-empty strings as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _normalized_label(value: Any) -> str:
    """Normalize enum-like trace labels into stable lower-case values."""
    normalized = str(value).strip().lower()
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]
    return normalized.replace("-", "_").replace(" ", "_")


def _role_values(payload: Mapping[str, Any]) -> frozenset[str]:
    """Read plural replacement roles while accepting the singular alias."""
    value = payload.get("roles")
    if value is None and "role" in payload:
        value = payload.get("role")
    if value is None:
        return frozenset()
    if isinstance(value, Mapping):
        values: Sequence[Any] = tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        values = (value,)
    return frozenset(
        label
        for item in values
        if (label := _normalized_label(item))
    )


def _position(value: Any) -> tuple[float, float] | None:
    """Normalize an ``[x, y]``-like value for comparisons and geometry."""
    if isinstance(value, Mapping):
        value = value.get("position") or value.get("waypoint")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    x = _finite_float(value[0])
    y = _finite_float(value[1])
    if x is None or y is None:
        return None
    return (x, y)


def _ratio(numerator: int, denominator: int) -> float | None:
    """Return a ratio while preserving the distinction from no samples."""
    if denominator <= 0:
        return None
    return numerator / denominator


def _nearest_rank_percentile(
    values: Sequence[float],
    percentile: float,
) -> float | None:
    """Return a deterministic nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _hashable(value: Any) -> Hashable | None:
    """Normalize JSON values into stable identities."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _hashable(item)) for key, item in value.items())
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_hashable(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return str(value)
    return value


def _nested_mapping(event: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Read a nested object without spreading type checks across collectors."""
    value = event.get(key)
    return value if isinstance(value, Mapping) else {}


def _field(
    event: Mapping[str, Any],
    names: Sequence[str],
    *nested_names: str,
) -> Any:
    """Return the first present top-level or nested field."""
    for name in names:
        if name in event:
            return event[name]
    for nested_name in nested_names:
        nested = _nested_mapping(event, nested_name)
        for name in names:
            if name in nested:
                return nested[name]
    return None


def _target_identity(event: Mapping[str, Any]) -> Hashable | None:
    """Extract a stable goal identity from legacy or replacement telemetry."""
    payloads = [
        event,
        _nested_mapping(event, "decision"),
        _nested_mapping(event, "navigation_intent"),
        _nested_mapping(event, "intent"),
        _nested_mapping(event, "movement_outcome"),
        _nested_mapping(event, "outcome"),
        _nested_mapping(event, "route"),
    ]
    for payload in payloads:
        for name in ("goal_cluster_id", "frontier_cluster_id", "cluster_id"):
            if payload.get(name) is not None:
                return ("cluster", _hashable(payload[name]))
    for payload in payloads:
        for name in ("goal", "goal_target", "route_target", "target"):
            if payload.get(name) is not None:
                return ("target", _hashable(payload[name]))
    return None


def _stable_goal_identity(event: Mapping[str, Any]) -> Hashable | None:
    """Return a persistent goal ID, never a legacy local coordinate proxy."""
    payloads = [
        event,
        _nested_mapping(event, "decision"),
        _nested_mapping(event, "navigation_intent"),
        _nested_mapping(event, "intent"),
        _nested_mapping(event, "movement_outcome"),
        _nested_mapping(event, "outcome"),
        _nested_mapping(event, "route"),
    ]
    for payload in payloads:
        for name in (
            "goal_cluster_id",
            "frontier_cluster_id",
            "cluster_id",
            "navigation_goal_id",
        ):
            if payload.get(name) is not None:
                return ("cluster", _hashable(payload[name]))
    return None


def _route_cursor_state(
    event: Mapping[str, Any],
) -> _RouteCursorState | None:
    """Extract a replacement-schema route identity and cursor."""
    payloads = [
        _nested_mapping(event, "navigation_intent"),
        _nested_mapping(event, "intent"),
        event,
    ]
    for payload in payloads:
        route_identity: Hashable | None = None
        for name in ("route_id", "active_route_id"):
            if payload.get(name) is not None:
                route_identity = ("route", _hashable(payload[name]))
                break
        if route_identity is None:
            for name in ("route_edge_ids", "edge_ids"):
                if payload.get(name) is not None:
                    route_identity = ("edges", _hashable(payload[name]))
                    break
        edge_cursor = _integer(
            _field(
                payload,
                ("edge_cursor", "route_edge_cursor", "active_edge_cursor"),
            )
        )
        if route_identity is None or edge_cursor is None or edge_cursor < 0:
            continue
        polyline_cursor = _integer(
            _field(
                payload,
                ("polyline_cursor", "edge_polyline_cursor", "point_cursor"),
            )
        )
        if polyline_cursor is None:
            polyline_cursor = 0
        if polyline_cursor < 0:
            continue
        return _RouteCursorState(
            route_identity=route_identity,
            edge_cursor=edge_cursor,
            polyline_cursor=polyline_cursor,
        )
    return None


def _decision_kind(event: Mapping[str, Any]) -> str:
    """Extract the normalized decision kind."""
    decision = _nested_mapping(event, "decision")
    value = decision.get("kind", event.get("decision_kind", ""))
    return str(value).lower()


def _node_key(payload: Mapping[str, Any]) -> Hashable | None:
    """Extract a stable node key, falling back to legacy coordinate identity."""
    for name in ("waypoint_id", "node_id", "id"):
        if payload.get(name) is not None:
            return ("id", _hashable(payload[name]))
    point = _position(payload.get("waypoint") or payload.get("position"))
    if point is None:
        return None
    return ("position", point)


def _endpoint_key(payload: Mapping[str, Any], prefix: str) -> Hashable | None:
    """Extract an edge endpoint in ID-based or coordinate-based form."""
    for name in (
        f"{prefix}_node_id",
        f"{prefix}_waypoint_id",
        f"{prefix}_id",
    ):
        if payload.get(name) is not None:
            return ("id", _hashable(payload[name]))
    point = _position(payload.get(prefix))
    if point is None:
        return None
    return ("position", point)


def analyze_waypoint_density(
    events: Sequence[Mapping[str, Any]],
    *,
    spatial_cell_size: int = 32,
) -> WaypointDensityMetrics:
    """Measure active graph density from legacy events or ID-based records."""
    if spatial_cell_size <= 0:
        raise ValueError("spatial_cell_size must be positive")

    nodes: dict[Hashable, _NodeRecord] = {}
    position_keys: dict[tuple[float, float], Hashable] = {}
    edges: dict[Hashable, _EdgeRecord] = {}
    reported_nodes: int | None = None
    reported_edges: int | None = None
    anonymous_edge = 0
    edge_mutations = 0
    replacement_edge_ids = {
        _hashable(identifier)
        for event in events
        if str(event.get("event", "")) == "waypoint_graph_delta"
        for collection in ("added_edges", "updated_edges")
        for payload in event.get(collection, ())
        if isinstance(payload, Mapping)
        if (identifier := _field(payload, ("edge_id", "id"))) is not None
    }

    node_add_names = {"waypoint_added", "waypoint_node_added"}
    node_remove_names = {"waypoint_removed", "waypoint_node_retired"}
    edge_add_names = {"waypoint_edge_added", "waypoint_edge_created"}
    edge_remove_names = {"waypoint_edge_removed", "waypoint_edge_retired"}

    def add_node(payload: Mapping[str, Any]) -> None:
        key = _node_key(payload)
        point = _position(payload.get("waypoint") or payload.get("position"))
        if key is None or point is None:
            return
        existing_position_key = position_keys.get(point)
        if key[0] == "position" and existing_position_key is not None:
            # A legacy alias for an ID-based record must update that record,
            # not create a second active node at the same coordinate.
            key = existing_position_key
        elif (
            key[0] == "id"
            and existing_position_key is not None
            and existing_position_key[0] == "position"
        ):
            legacy = nodes.pop(existing_position_key)
            nodes[key] = legacy
        source_value = payload.get("source", payload.get("role", "unknown"))
        if isinstance(source_value, Sequence) and not isinstance(
            source_value,
            (str, bytes),
        ):
            source = "+".join(sorted(str(item) for item in source_value))
        else:
            source = str(source_value)
        roles = _role_values(payload)
        previous = nodes.get(key)
        if previous is not None:
            if not roles:
                roles = previous.roles
            if source == "unknown":
                source = previous.source
        nodes[key] = _NodeRecord(
            position=point,
            source=source,
            roles=roles,
        )
        position_keys[point] = key

    def add_edge(
        payload: Mapping[str, Any],
        *,
        replacement_record: bool = False,
    ) -> None:
        nonlocal anonymous_edge, edge_mutations
        identifier = _field(payload, ("edge_id", "id"))
        start = _endpoint_key(payload, "start")
        end = _endpoint_key(payload, "end")
        if start is not None and start[0] == "position":
            start = position_keys.get(start[1], start)
        if end is not None and end[0] == "position":
            end = position_keys.get(end[1], end)
        if identifier is not None:
            key: Hashable = ("id", _hashable(identifier))
        elif start is not None and end is not None:
            # The legacy graph stores one active edge per unordered endpoint
            # pair and evidence source. Repeated events for that key are
            # shorter replacements, not additional active edges.
            key = (
                "legacy",
                frozenset((start, end)),
                str(payload.get("source", "unknown")),
            )
        else:
            anonymous_edge += 1
            key = ("anonymous", anonymous_edge)
        is_dual_write_alias = (
            not replacement_record
            and identifier is not None
            and _hashable(identifier) in replacement_edge_ids
        )
        if not is_dual_write_alias:
            edge_mutations += 1
        previous = edges.get(key)
        if previous is not None:
            if start not in nodes and previous.start in nodes:
                start = previous.start
            if end not in nodes and previous.end in nodes:
                end = previous.end
        edges[key] = _EdgeRecord(
            start=start,
            end=end,
        )

    def remove_edge(payload: Mapping[str, Any]) -> None:
        identifier = _field(payload, ("edge_id", "id"))
        if identifier is not None:
            edges.pop(("id", _hashable(identifier)), None)
            return
        start = _endpoint_key(payload, "start")
        end = _endpoint_key(payload, "end")
        if start is None or end is None:
            return
        edges.pop(
            (
                "legacy",
                frozenset((start, end)),
                str(payload.get("source", "unknown")),
            ),
            None,
        )

    def remove_node(key: Hashable) -> None:
        record = nodes.pop(key, None)
        if record is not None and position_keys.get(record.position) == key:
            position_keys.pop(record.position, None)

    for event in events:
        event_name = str(event.get("event", ""))
        node_count = _integer(event.get("node_count", event.get("graph_nodes")))
        edge_count = _integer(event.get("edge_count", event.get("graph_edges")))
        if node_count is not None:
            reported_nodes = node_count
        if edge_count is not None:
            reported_edges = edge_count

        if event_name in node_add_names:
            add_node(event)
        elif event_name in node_remove_names:
            key = _node_key(event)
            if key is not None:
                remove_node(key)
        elif event_name in edge_add_names:
            add_edge(event)
        elif event_name in edge_remove_names:
            remove_edge(event)

        if event_name != "waypoint_graph_delta":
            continue
        for payload in event.get("added_nodes", ()):
            if isinstance(payload, Mapping):
                add_node(payload)
        for payload in event.get("updated_nodes", ()):
            if isinstance(payload, Mapping):
                add_node(payload)
        for identifier in event.get("retired_node_ids", ()):
            remove_node(("id", _hashable(identifier)))
        for payload in event.get("added_edges", ()):
            if isinstance(payload, Mapping):
                add_edge(payload, replacement_record=True)
        for payload in event.get("updated_edges", ()):
            if isinstance(payload, Mapping):
                add_edge(payload, replacement_record=True)
        for identifier in event.get("retired_edge_ids", ()):
            edges.pop(("id", _hashable(identifier)), None)

    positions = [record.position for record in nodes.values()]
    nearest: list[float] = []
    for index, point in enumerate(positions):
        distances = (
            math.dist(point, other)
            for other_index, other in enumerate(positions)
            if other_index != index
        )
        nearest_distance = min(distances, default=math.inf)
        if math.isfinite(nearest_distance):
            nearest.append(nearest_distance)

    cells: Counter[tuple[int, int]] = Counter(
        (
            math.floor(point[0] / spatial_cell_size),
            math.floor(point[1] / spatial_cell_size),
        )
        for point in positions
    )
    neighbors: dict[Hashable, set[Hashable]] = defaultdict(set)
    connections: set[frozenset[Hashable]] = set()
    for edge in edges.values():
        if (
            edge.start in nodes
            and edge.end in nodes
            and edge.start != edge.end
        ):
            neighbors[edge.start].add(edge.end)
            neighbors[edge.end].add(edge.start)
            connections.add(frozenset((edge.start, edge.end)))

    # Characterization radii are exclusive, matching the merge candidate
    # convention used by the original density audit.
    within_8 = sum(distance < 8.0 for distance in nearest)
    within_16 = sum(distance < 16.0 for distance in nearest)
    degree_two = sum(len(neighbors[key]) == 2 for key in nodes)
    node_count = len(nodes)
    occupied_cells = len(cells)
    role_counts = Counter(
        role
        for node in nodes.values()
        for role in node.roles
    )
    forbidden_node_role_count = sum(
        bool(node.roles & FORBIDDEN_NODE_ROLES)
        for node in nodes.values()
    )
    return WaypointDensityMetrics(
        node_count=node_count,
        edge_count=len(edges),
        edge_mutation_count=edge_mutations,
        unique_connection_count=len(connections),
        source_counts=dict(Counter(node.source for node in nodes.values())),
        occupied_spatial_cells=occupied_cells,
        average_nodes_per_occupied_cell=(
            node_count / occupied_cells if occupied_cells else None
        ),
        maximum_nodes_in_cell=max(cells.values(), default=0),
        median_nearest_neighbor_px=(
            statistics.median(nearest) if nearest else None
        ),
        nodes_with_neighbor_within_8_px=within_8,
        neighbor_within_8_px_rate=_ratio(within_8, node_count),
        nodes_with_neighbor_within_16_px=within_16,
        neighbor_within_16_px_rate=_ratio(within_16, node_count),
        degree_two_nodes=degree_two,
        degree_two_rate=_ratio(degree_two, node_count),
        reported_node_count=reported_nodes,
        reported_edge_count=reported_edges,
        role_counts=dict(role_counts),
        forbidden_node_role_count=forbidden_node_role_count,
    )


def analyze_target_retention(
    events: Sequence[Mapping[str, Any]],
) -> TargetRetentionMetrics:
    """Measure target persistence after route-segment execution.

    Legacy STEP/ROTATE targets are local primitives, so a coordinate change is
    reported only as a proxy.  True route abandonment is calculated solely
    when both events carry a stable navigation-goal identity.
    """
    counters: dict[int, Counter[str]] = defaultdict(Counter)
    pending_segment: dict[
        int,
        tuple[
            Hashable | None,
            Hashable | None,
            _RouteCursorState | None,
        ],
    ] = {}
    candidate_decision: dict[int, Mapping[str, Any]] = {}
    previous_decision_target: dict[int, Hashable] = {}

    def classify_followup(
        drone_id: int,
        decision: Mapping[str, Any],
    ) -> None:
        segment_target, segment_goal, segment_cursor = pending_segment.pop(
            drone_id
        )
        target = _target_identity(decision)
        goal = _stable_goal_identity(decision)
        route_cursor = _route_cursor_state(decision)
        kind = _decision_kind(decision)
        count = counters[drone_id]
        count["segment_followups"] += 1

        if kind == "frontier" and target == segment_target:
            count["retained_after_segment"] += 1
        elif kind == "frontier":
            count["reranked_frontier_after_segment"] += 1
            count["switched_after_segment"] += 1
        elif kind == "step":
            count["local_step_after_segment"] += 1
            count["switched_after_segment"] += 1
        elif kind == "rotate":
            count["rotate_after_segment"] += 1
            count["switched_after_segment"] += 1
        else:
            count["unclassified_followups"] += 1
            if target != segment_target:
                count["switched_after_segment"] += 1

        if segment_goal is not None and goal is not None:
            count["goal_identity_followups"] += 1
            if segment_goal != goal:
                count["route_abandonments"] += 1
        if segment_cursor is not None and route_cursor is not None:
            count["route_cursor_followups"] += 1
            cursor_continues = (
                segment_cursor.route_identity == route_cursor.route_identity
                and (
                    route_cursor.edge_cursor,
                    route_cursor.polyline_cursor,
                )
                >= (
                    segment_cursor.edge_cursor,
                    segment_cursor.polyline_cursor,
                )
            )
            if cursor_continues:
                count["route_cursor_continuations"] += 1
            else:
                count["route_cursor_resets_or_regressions"] += 1

    for event in events:
        drone_id = _integer(event.get("drone_id"))
        if drone_id is None:
            continue
        event_name = str(event.get("event", ""))
        is_segment_complete = event_name in {
            "drone_waypoint_segment_complete",
            "drone_route_segment_complete",
            "navigation_route_segment_complete",
        }
        if is_segment_complete:
            if drone_id in pending_segment:
                decision = candidate_decision.get(drone_id)
                if decision is None:
                    counters[drone_id]["segments_without_followup"] += 1
                    pending_segment.pop(drone_id)
                else:
                    classify_followup(drone_id, decision)
            counters[drone_id]["completed_segments"] += 1
            pending_segment[drone_id] = (
                _target_identity(event),
                _stable_goal_identity(event),
                _route_cursor_state(event),
            )
            candidate_decision.pop(drone_id, None)
            continue

        if event_name == "drone_action_result":
            decision = candidate_decision.pop(drone_id, None)
            if drone_id in pending_segment and decision is not None:
                classify_followup(drone_id, decision)
            continue

        if event_name not in DECISION_EVENTS:
            continue

        target = _target_identity(event)
        previous = previous_decision_target.get(drone_id)
        if target is not None:
            if previous is not None:
                counters[drone_id]["decision_target_transitions"] += 1
                if target != previous:
                    counters[drone_id]["decision_target_switches"] += 1
            previous_decision_target[drone_id] = target

        # A post-rebuild decision supersedes an exhausted initial decision.
        candidate_decision[drone_id] = event

    for drone_id in tuple(pending_segment):
        decision = candidate_decision.get(drone_id)
        if decision is None:
            counters[drone_id]["segments_without_followup"] += 1
            pending_segment.pop(drone_id)
        else:
            classify_followup(drone_id, decision)

    by_drone: dict[int, DroneTargetRetentionMetrics] = {}
    for drone_id, count in sorted(counters.items()):
        followups = count["segment_followups"]
        by_drone[drone_id] = DroneTargetRetentionMetrics(
            completed_segments=count["completed_segments"],
            segment_followups=followups,
            retained_after_segment=count["retained_after_segment"],
            reranked_frontier_after_segment=(
                count["reranked_frontier_after_segment"]
            ),
            switched_after_segment=count["switched_after_segment"],
            local_step_after_segment=count["local_step_after_segment"],
            rotate_after_segment=count["rotate_after_segment"],
            unclassified_followups=count["unclassified_followups"],
            segments_without_followup=count["segments_without_followup"],
            retention_rate=_ratio(count["retained_after_segment"], followups),
            coordinate_switch_proxy_rate=_ratio(
                count["switched_after_segment"],
                followups,
            ),
            goal_identity_followups=count["goal_identity_followups"],
            route_abandonments=count["route_abandonments"],
            route_abandonment_rate=_ratio(
                count["route_abandonments"],
                count["goal_identity_followups"],
            ),
            route_cursor_followups=count["route_cursor_followups"],
            route_cursor_continuations=(
                count["route_cursor_continuations"]
            ),
            route_cursor_resets_or_regressions=(
                count["route_cursor_resets_or_regressions"]
            ),
            route_cursor_continuation_rate=_ratio(
                count["route_cursor_continuations"],
                count["route_cursor_followups"],
            ),
            decision_target_transitions=count["decision_target_transitions"],
            decision_target_switches=count["decision_target_switches"],
        )

    total: Counter[str] = Counter()
    for count in counters.values():
        total.update(count)
    followups = total["segment_followups"]
    return TargetRetentionMetrics(
        completed_segments=total["completed_segments"],
        segment_followups=followups,
        retained_after_segment=total["retained_after_segment"],
        reranked_frontier_after_segment=(
            total["reranked_frontier_after_segment"]
        ),
        switched_after_segment=total["switched_after_segment"],
        local_step_after_segment=total["local_step_after_segment"],
        rotate_after_segment=total["rotate_after_segment"],
        unclassified_followups=total["unclassified_followups"],
        segments_without_followup=total["segments_without_followup"],
        retention_rate=_ratio(total["retained_after_segment"], followups),
        coordinate_switch_proxy_rate=_ratio(
            total["switched_after_segment"],
            followups,
        ),
        goal_identity_followups=total["goal_identity_followups"],
        route_abandonments=total["route_abandonments"],
        route_abandonment_rate=_ratio(
            total["route_abandonments"],
            total["goal_identity_followups"],
        ),
        route_cursor_followups=total["route_cursor_followups"],
        route_cursor_continuations=total["route_cursor_continuations"],
        route_cursor_resets_or_regressions=(
            total["route_cursor_resets_or_regressions"]
        ),
        route_cursor_continuation_rate=_ratio(
            total["route_cursor_continuations"],
            total["route_cursor_followups"],
        ),
        decision_target_transitions=total["decision_target_transitions"],
        decision_target_switches=total["decision_target_switches"],
        by_drone=by_drone,
    )


def analyze_frontier_fallbacks(
    events: Sequence[Mapping[str, Any]],
) -> FrontierFallbackMetrics:
    """Measure repeated and regenerated legacy FRONTIER fallbacks."""
    reached: dict[int, set[Hashable]] = defaultdict(set)
    selections: Counter[tuple[int, Hashable]] = Counter()
    regenerated: set[tuple[int, Hashable]] = set()
    fallback_count = 0
    zero_reward = 0
    regenerated_count = 0

    for event in events:
        drone_id = _integer(event.get("drone_id"))
        if drone_id is None:
            continue
        event_name = str(event.get("event", ""))
        if event_name in {
            "drone_frontier_reached",
            "drone_frontier_cluster_reached",
            "navigation_goal_reached",
        }:
            target = _target_identity(event)
            if target is not None:
                reached[drone_id].add(target)
            continue
        if event_name not in DECISION_EVENTS or _decision_kind(event) != "frontier":
            continue

        fallback_count += 1
        target = _target_identity(event)
        if target is not None:
            key = (drone_id, target)
            selections[key] += 1
            if target in reached[drone_id]:
                regenerated_count += 1
                regenerated.add(key)

        mcts = _nested_mapping(event, "mcts")
        reward = _finite_float(mcts.get("selected_reward"))
        if reward == 0.0:
            zero_reward += 1

    regenerated_targets = tuple(sorted(regenerated, key=repr))
    return FrontierFallbackMetrics(
        fallback_count=fallback_count,
        zero_reward_fallbacks=zero_reward,
        unique_targets=len(selections),
        repeated_target_selections=sum(
            max(0, count - 1) for count in selections.values()
        ),
        regenerated_after_reach=regenerated_count,
        regenerated_drone_targets=regenerated_targets,
    )


def analyze_aba_reversals(
    events: Sequence[Mapping[str, Any]],
    *,
    normalized_window_start_s: float | None = None,
) -> ABAReversalMetrics:
    """Measure A-B-A patterns in physically reached goals.

    ``normalized_window_start_s`` is relative to the first finite simulation
    timestamp.  The arrival history resets at the cutoff, so an A-B-A triplet
    never reaches backward across the requested window boundary.
    """
    arrivals: dict[int, list[Hashable]] = defaultdict(list)
    observed_drone_ids = {
        drone_id
        for event in events
        if (drone_id := _integer(event.get("drone_id"))) is not None
    }
    arrival_names = {
        "drone_frontier_reached",
        "drone_frontier_cluster_reached",
        "navigation_goal_reached",
    }
    timestamps = [
        value
        for event in events
        if (value := _finite_float(event.get("sim_time"))) is not None
    ]
    trace_start = min(timestamps, default=0.0)
    cutoff = (
        trace_start + normalized_window_start_s
        if normalized_window_start_s is not None
        else None
    )
    for event in events:
        if str(event.get("event", "")) not in arrival_names:
            continue
        sim_time = _finite_float(event.get("sim_time"))
        if cutoff is not None and (sim_time is None or sim_time < cutoff):
            continue
        drone_id = _integer(event.get("drone_id"))
        if drone_id is None:
            continue
        target = _target_identity(event)
        if target is None:
            state = _nested_mapping(event, "state")
            point = _position(state.get("position"))
            target = ("target", point) if point is not None else None
        if target is not None:
            arrivals[drone_id].append(target)

    by_drone: dict[int, DroneABAReversalMetrics] = {}
    total_arrivals = 0
    total_opportunities = 0
    total_reversals = 0
    for drone_id in sorted(observed_drone_ids):
        goals = arrivals[drone_id]
        opportunities = max(0, len(goals) - 2)
        reversals = sum(
            first == third and first != second
            for first, second, third in zip(goals, goals[1:], goals[2:])
        )
        by_drone[drone_id] = DroneABAReversalMetrics(
            arrivals=len(goals),
            reversal_opportunities=opportunities,
            reversals=reversals,
            reversal_rate=_ratio(reversals, opportunities),
        )
        total_arrivals += len(goals)
        total_opportunities += opportunities
        total_reversals += reversals
    return ABAReversalMetrics(
        arrivals=total_arrivals,
        reversal_opportunities=total_opportunities,
        reversals=total_reversals,
        reversal_rate=_ratio(total_reversals, total_opportunities),
        normalized_window_start_s=normalized_window_start_s,
        by_drone=by_drone,
    )


def analyze_mcts_timing(
    events: Sequence[Mapping[str, Any]],
    *,
    legacy_budget_ms: float = LEGACY_MCTS_BUDGET_MS,
) -> MCTSTimingMetrics:
    """Measure MCTS work and elapsed time against the configured budget."""
    if not math.isfinite(legacy_budget_ms) or legacy_budget_ms <= 0.0:
        raise ValueError("legacy_budget_ms must be a positive finite value")

    budget_ms = legacy_budget_ms
    budget_source = "legacy_default"
    for event in events:
        if str(event.get("event", "")) != "mission_constructed":
            continue
        configured = _finite_float(
            _field(
                event,
                (
                    "mcts_decision_time_budget_ms",
                    "mcts_time_budget_ms",
                    "planning_budget_ms",
                ),
                "mcts",
            )
        )
        if configured is not None and configured > 0.0:
            budget_ms = configured
            budget_source = "trace"

    searches: list[Mapping[str, Any]] = []
    seen_search_ids: set[Hashable] = set()
    for event in events:
        event_name = str(event.get("event", ""))
        nested = _nested_mapping(event, "mcts")
        schema_version = _integer(event.get("schema_version")) or 0
        performed = _boolean(nested.get("performed"))
        search: Mapping[str, Any] | None = None
        if event_name == "drone_local_mcts_decision" and nested:
            if performed is not False:
                search = nested
        elif event_name in {"mcts_search", "local_mcts_search"}:
            candidate = nested or event
            if _boolean(candidate.get("performed")) is not False:
                search = candidate
        elif event_name in DECISION_EVENTS and nested:
            # Schema-v3 strategic decisions deliberately carry an MCTS
            # compatibility object with performed=false. Legacy decisions did
            # not distinguish the object from an actual search.
            if performed is True or (schema_version < 3 and performed is None):
                search = nested
        if search is None:
            continue
        search_id = _hashable(
            _field(search, ("search_id", "mcts_search_id"))
        )
        if search_id is not None:
            if search_id in seen_search_ids:
                continue
            seen_search_ids.add(search_id)
        searches.append(search)

    elapsed_values: list[float] = []
    iteration_values: list[int] = []
    over_budget = 0
    zero_reward_frontiers = 0
    root_coverage_values: list[bool] = []
    unsafe_incomplete = 0
    for search in searches:
        elapsed = _finite_float(
            _field(search, ("elapsed_ms", "decision_elapsed_ms"))
        )
        if elapsed is not None:
            elapsed_values.append(elapsed)
            search_budget = _finite_float(
                _field(search, ("budget_ms", "time_budget_ms"))
            )
            effective_budget = (
                search_budget
                if search_budget is not None and search_budget > 0.0
                else budget_ms
            )
            if elapsed > effective_budget:
                over_budget += 1
        iterations = _integer(search.get("iterations"))
        if iterations is not None:
            iteration_values.append(iterations)
        coverage = _boolean(search.get("root_coverage_complete"))
        if coverage is not None:
            root_coverage_values.append(coverage)
            if not coverage and iterations is not None and iterations > 0:
                unsafe_incomplete += 1
        selected_kind = str(search.get("selected_kind", "")).lower()
        selected_reward = _finite_float(search.get("selected_reward"))
        if selected_kind == "frontier" and selected_reward == 0.0:
            zero_reward_frontiers += 1

    at_most_one = sum(value <= 1 for value in iteration_values)
    search_count = len(searches)
    return MCTSTimingMetrics(
        search_count=search_count,
        timing_sample_count=len(elapsed_values),
        missing_timing_count=search_count - len(elapsed_values),
        iteration_sample_count=len(iteration_values),
        missing_iteration_count=search_count - len(iteration_values),
        at_most_one_iteration_count=at_most_one,
        at_most_one_iteration_rate=_ratio(
            at_most_one,
            len(iteration_values),
        ),
        budget_ms=budget_ms,
        budget_source=budget_source,
        over_budget_count=over_budget,
        over_budget_rate=_ratio(over_budget, len(elapsed_values)),
        total_elapsed_ms=sum(elapsed_values),
        mean_elapsed_ms=(
            statistics.fmean(elapsed_values) if elapsed_values else None
        ),
        median_elapsed_ms=(
            statistics.median(elapsed_values) if elapsed_values else None
        ),
        p95_elapsed_ms=_nearest_rank_percentile(elapsed_values, 0.95),
        p99_elapsed_ms=_nearest_rank_percentile(elapsed_values, 0.99),
        maximum_elapsed_ms=max(elapsed_values, default=None),
        zero_reward_frontier_fallbacks=zero_reward_frontiers,
        root_coverage_sample_count=len(root_coverage_values),
        missing_root_coverage_count=(
            search_count - len(root_coverage_values)
        ),
        complete_root_coverage_count=sum(root_coverage_values),
        incomplete_root_coverage_count=sum(
            not value for value in root_coverage_values
        ),
        unsafe_incomplete_search_count=unsafe_incomplete,
    )


def _route_cache_hit(event: Mapping[str, Any]) -> bool | None:
    """Read cache telemetry while distinguishing legacy unknown values."""
    value = _field(event, ("cache_hit",), "route")
    if isinstance(value, bool):
        return value
    status = _field(event, ("cache_status",), "route")
    if status is None:
        return None
    normalized = str(status).lower()
    if normalized in {"hit", "cache_hit", "reused"}:
        return True
    if normalized in {"miss", "cache_miss", "rebuilt"}:
        return False
    return None


def analyze_route_cache(
    events: Sequence[Mapping[str, Any]],
) -> RouteCacheMetrics:
    """Measure route outcomes, repeated requests, timing, and cache use."""
    status_counts: Counter[str] = Counter()
    pairs: set[tuple[int | None, Hashable]] = set()
    elapsed_values: list[float] = []
    cache_hits = 0
    cache_misses = 0
    cache_unknown = 0
    route_calls = 0
    successful_routes = 0

    segment_calls = 0
    successful_segments = 0
    astar_attempts = 0
    astar_selected = 0
    stored_selected = 0
    repeated_route_calls = 0
    repeated_cache_hits = 0
    repeated_cache_misses = 0
    repeated_cache_unknown = 0
    seen_route_keys: set[Hashable] = set()
    persistent_edge_astar_calls = 0
    connector_astar_calls = 0
    unchanged_valid_intent_replans = 0
    replan_reasons: Counter[str] = Counter()

    for event in events:
        event_name = str(event.get("event", ""))
        if event_name in {
            "drone_waypoint_route",
            "drone_strategic_route",
            "drone_route_lookup",
            "drone_homing_route",
        }:
            route_calls += 1
            route_payload = _nested_mapping(event, "route")
            status = str(event.get("status", route_payload.get("status", "unknown")))
            status_counts[status] += 1
            found = _boolean(event.get("found", route_payload.get("found")))
            successful = found if found is not None else status == "ok"
            successful_routes += int(successful)
            target = _target_identity(event)
            if target is not None:
                pairs.add((_integer(event.get("drone_id")), target))
            elapsed = _finite_float(
                _field(
                    event,
                    (
                        "route_lookup_elapsed_ms",
                        "lookup_elapsed_ms",
                        "route_elapsed_ms",
                        "elapsed_ms",
                    ),
                    "route",
                )
            )
            if elapsed is not None:
                elapsed_values.append(elapsed)
            cache_hit = _route_cache_hit(event)
            if cache_hit is True:
                cache_hits += 1
            elif cache_hit is False:
                cache_misses += 1
            else:
                cache_unknown += 1

            cache_eligible = _boolean(
                _field(event, ("cache_eligible",), "route")
            )
            explicit_key = _field(
                event,
                ("cache_key", "route_cache_key"),
                "route",
            )
            node_ids = _field(event, ("node_ids", "route_node_ids"), "route")
            goal_node = None
            if (
                isinstance(node_ids, Sequence)
                and not isinstance(node_ids, (str, bytes))
                and node_ids
            ):
                goal_node = node_ids[-1]
            route_key = _hashable(explicit_key)
            if route_key is None and target is not None:
                route_key = _hashable((
                    _integer(event.get("drone_id")),
                    target,
                    _field(event, ("topology_revision",), "route"),
                    _field(
                        event,
                        ("requester_knowledge_revision", "knowledge_revision"),
                        "route",
                    ),
                    goal_node,
                ))
            if (
                successful
                and route_key is not None
                and cache_eligible is not False
            ):
                if route_key in seen_route_keys:
                    repeated_route_calls += 1
                    if cache_hit is True:
                        repeated_cache_hits += 1
                    elif cache_hit is False:
                        repeated_cache_misses += 1
                    else:
                        repeated_cache_unknown += 1
                else:
                    seen_route_keys.add(route_key)

            replanned = _boolean(event.get("route_replanned"))
            active_valid = _boolean(event.get("active_intent_valid"))
            reason = _normalized_label(event.get("replan_reason", ""))
            if reason:
                replan_reasons[reason] += 1
            if (
                (replanned is True and active_valid is True)
                or reason in {"unchanged_valid_intent", "unchanged_valid"}
                or _boolean(event.get("unchanged_valid_intent_replan")) is True
            ):
                unchanged_valid_intent_replans += 1
            continue

        if event_name not in {
            "drone_waypoint_segment_path",
            "drone_route_segment_path",
        }:
            continue
        segment_calls += 1
        source = str(event.get("path_source", "unknown")).lower()
        path_len = _integer(event.get("path_len"))
        successful_segments += int(
            source not in {"failed", "unknown", "none"}
            and (path_len is None or path_len > 1)
        )
        if "astar_path_len" in event or source == "astar":
            astar_attempts += 1
        if source == "astar":
            astar_selected += 1
        if source in {
            "stored_polyline",
            "trusted_route_fallback",
            "trusted_path",
            "persistent_route",
        }:
            stored_selected += 1
        persistent_calls = _integer(
            _field(
                event,
                ("persistent_edge_astar_calls",),
                "execution",
            )
        )
        connector_calls = _integer(
            _field(event, ("connector_astar_calls",), "execution")
        )
        if persistent_calls is not None and persistent_calls >= 0:
            persistent_edge_astar_calls += persistent_calls
        elif source in {"persistent_route", "stored_polyline"}:
            astar_len = _integer(event.get("astar_path_len"))
            if astar_len is not None and astar_len > 0:
                persistent_edge_astar_calls += 1
        if connector_calls is not None and connector_calls >= 0:
            connector_astar_calls += connector_calls

    known_cache_samples = cache_hits + cache_misses
    known_repeated_samples = repeated_cache_hits + repeated_cache_misses
    return RouteCacheMetrics(
        route_calls=route_calls,
        successful_routes=successful_routes,
        failed_routes=route_calls - successful_routes,
        status_counts=dict(status_counts),
        unique_drone_target_pairs=len(pairs),
        timing_sample_count=len(elapsed_values),
        total_route_time_ms=sum(elapsed_values),
        mean_route_time_ms=(
            statistics.fmean(elapsed_values) if elapsed_values else None
        ),
        median_route_time_ms=(
            statistics.median(elapsed_values) if elapsed_values else None
        ),
        p95_route_time_ms=_nearest_rank_percentile(elapsed_values, 0.95),
        maximum_route_time_ms=max(elapsed_values, default=None),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_unknown=cache_unknown,
        cache_hit_rate=_ratio(cache_hits, known_cache_samples),
        segment_calls=segment_calls,
        successful_segments=successful_segments,
        astar_attempts=astar_attempts,
        astar_selected_paths=astar_selected,
        stored_polyline_selected_paths=stored_selected,
        repeated_route_calls=repeated_route_calls,
        repeated_cache_hits=repeated_cache_hits,
        repeated_cache_misses=repeated_cache_misses,
        repeated_cache_unknown=repeated_cache_unknown,
        repeated_cache_hit_rate=_ratio(
            repeated_cache_hits,
            known_repeated_samples,
        ),
        persistent_edge_astar_calls=persistent_edge_astar_calls,
        connector_astar_calls=connector_astar_calls,
        unchanged_valid_intent_replans=unchanged_valid_intent_replans,
        replan_reason_counts=dict(replan_reasons),
    )


def analyze_information_efficiency(
    events: Sequence[Mapping[str, Any]],
) -> InformationEfficiencyMetrics:
    """Measure exact sensor signals per exact traced movement distance.

    Legacy traces do not contain either signal.  They intentionally report an
    unavailable (``None``) efficiency instead of silently treating missing
    telemetry as zero gain or estimating curved travel from endpoint deltas.
    Newly-known cell counts and confidence deltas remain separate because they
    have different units; no arbitrary combined score is produced.
    """
    counters: dict[int, Counter[str]] = defaultdict(Counter)
    distances: dict[int, float] = defaultdict(float)
    confidence: dict[int, float] = defaultdict(float)

    for event in events:
        drone_id = _integer(event.get("drone_id"))
        if drone_id is None:
            continue
        event_name = str(event.get("event", ""))
        if event_name == "drone_motion":
            distance = _finite_float(event.get("travelled_distance"))
            if distance is not None and distance >= 0.0:
                distances[drone_id] += distance
                counters[drone_id]["motion_samples"] += 1
            continue
        if event_name != "sensor_scan":
            continue

        counters[drone_id]["completed_scans"] += 1
        newly_known = _integer(event.get("newly_known_cells"))
        if newly_known is not None and newly_known < 0:
            newly_known = None
        confidence_gain = _finite_float(event.get("confidence_gain"))
        if confidence_gain is not None and confidence_gain < 0.0:
            confidence_gain = None

        if newly_known is None:
            counters[drone_id]["scans_missing_newly_known"] += 1
        else:
            counters[drone_id]["newly_known_samples"] += 1
            counters[drone_id]["newly_known_cells"] += newly_known
        if confidence_gain is None:
            counters[drone_id]["scans_missing_confidence_gain"] += 1
        else:
            counters[drone_id]["confidence_gain_samples"] += 1
            confidence[drone_id] += confidence_gain
        if newly_known is None or confidence_gain is None:
            counters[drone_id]["scans_missing_gain"] += 1
        else:
            counters[drone_id]["gain_samples"] += 1

    drone_ids = sorted(set(counters) | set(distances) | set(confidence))
    by_drone: dict[int, DroneInformationEfficiencyMetrics] = {}
    for drone_id in drone_ids:
        count = counters[drone_id]
        distance = distances[drone_id]
        completed_scans = count["completed_scans"]
        complete_gain = (
            completed_scans > 0 and count["scans_missing_gain"] == 0
        )
        complete_newly_known = (
            completed_scans > 0
            and count["scans_missing_newly_known"] == 0
        )
        complete_confidence = (
            completed_scans > 0
            and count["scans_missing_confidence_gain"] == 0
        )
        newly_known = count["newly_known_cells"]
        confidence_gain = confidence[drone_id]
        by_drone[drone_id] = DroneInformationEfficiencyMetrics(
            travelled_distance_px=distance,
            distance_source=(
                "drone_motion" if count["motion_samples"] else "unavailable"
            ),
            motion_samples=count["motion_samples"],
            completed_scans=completed_scans,
            gain_samples=count["gain_samples"],
            scans_missing_gain=count["scans_missing_gain"],
            gain_telemetry_complete=complete_gain,
            newly_known_samples=count["newly_known_samples"],
            scans_missing_newly_known=(
                count["scans_missing_newly_known"]
            ),
            newly_known_telemetry_complete=complete_newly_known,
            confidence_gain_samples=count["confidence_gain_samples"],
            scans_missing_confidence_gain=(
                count["scans_missing_confidence_gain"]
            ),
            confidence_gain_telemetry_complete=complete_confidence,
            newly_known_cells=newly_known,
            confidence_gain=confidence_gain,
            newly_known_cells_per_travelled_px=(
                newly_known / distance
                if complete_newly_known and distance > 0.0
                else None
            ),
            confidence_gain_per_travelled_px=(
                confidence_gain / distance
                if complete_confidence and distance > 0.0
                else None
            ),
        )

    total_distance = sum(distances.values())
    total_count: Counter[str] = Counter()
    for count in counters.values():
        total_count.update(count)
    total_newly_known = total_count["newly_known_cells"]
    total_confidence = sum(confidence.values())
    completed_scans = total_count["completed_scans"]
    complete_gain = (
        completed_scans > 0 and total_count["scans_missing_gain"] == 0
    )
    complete_newly_known = (
        completed_scans > 0
        and total_count["scans_missing_newly_known"] == 0
    )
    complete_confidence = (
        completed_scans > 0
        and total_count["scans_missing_confidence_gain"] == 0
    )
    return InformationEfficiencyMetrics(
        travelled_distance_px=total_distance,
        distance_source=(
            "drone_motion" if total_count["motion_samples"] else "unavailable"
        ),
        motion_samples=total_count["motion_samples"],
        completed_scans=completed_scans,
        gain_samples=total_count["gain_samples"],
        scans_missing_gain=total_count["scans_missing_gain"],
        gain_telemetry_complete=complete_gain,
        newly_known_samples=total_count["newly_known_samples"],
        scans_missing_newly_known=(
            total_count["scans_missing_newly_known"]
        ),
        newly_known_telemetry_complete=complete_newly_known,
        confidence_gain_samples=total_count["confidence_gain_samples"],
        scans_missing_confidence_gain=(
            total_count["scans_missing_confidence_gain"]
        ),
        confidence_gain_telemetry_complete=complete_confidence,
        newly_known_cells=total_newly_known,
        confidence_gain=total_confidence,
        newly_known_cells_per_travelled_px=(
            total_newly_known / total_distance
            if complete_newly_known and total_distance > 0.0
            else None
        ),
        confidence_gain_per_travelled_px=(
            total_confidence / total_distance
            if complete_confidence and total_distance > 0.0
            else None
        ),
        by_drone=by_drone,
    )


def _mapping_path_present(
    payload: Mapping[str, Any],
    path: Sequence[str],
) -> bool:
    """Return whether every key in a nested mapping path is present."""
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False
        current = current[key]
    return True


def _schema_validation(
    events: Sequence[Mapping[str, Any]],
) -> TraceSchemaValidationMetrics:
    """Validate required schema-v3 fields without rejecting legacy traces."""
    requirements: Mapping[str, tuple[tuple[str, ...], ...]] = {
        "drone_waypoint_route": (
            ("drone_id",),
            ("target",),
            ("status",),
            ("route_elapsed_ms",),
            ("route",),
            ("route", "route_id"),
            ("route", "cache_hit"),
            ("route", "topology_revision"),
            ("route", "requester_knowledge_revision"),
            ("route", "node_ids"),
            ("route", "edge_ids"),
            ("route", "total_cost"),
            ("route", "remaining_cost"),
            ("replan_reason",),
            ("route_replanned",),
            ("active_intent_valid",),
        ),
        "drone_strategic_route": (
            ("drone_id",),
            ("route",),
            ("route", "route_id"),
            ("route", "cache_hit"),
            ("route", "topology_revision"),
            ("route", "requester_knowledge_revision"),
            ("route", "node_ids"),
            ("route", "edge_ids"),
            ("route", "remaining_cost"),
            ("replan_reason",),
            ("route_replanned",),
            ("active_intent_valid",),
        ),
        "drone_navigation_transition": (
            ("drone_id",),
            ("mode_transition",),
            ("mode_transition", "from_mode"),
            ("mode_transition", "to_mode"),
            ("mode_transition", "reason"),
            ("intent",),
        ),
        "drone_waypoint_segment_path": (
            ("path_source",),
            ("persistent_edge_astar_calls",),
            ("connector_astar_calls",),
        ),
        "drone_route_segment_path": (
            ("path_source",),
            ("persistent_edge_astar_calls",),
            ("connector_astar_calls",),
        ),
        "drone_local_mcts_decision": (
            ("drone_id",),
            ("mcts",),
            ("mcts", "performed"),
            ("mcts", "elapsed_ms"),
            ("mcts", "iterations"),
            ("mcts", "root_coverage_complete"),
            ("mcts", "budget_ms"),
        ),
        "drone_local_policy_decision": (
            ("drone_id",),
            ("decision",),
            ("decision", "exploration_mode"),
            ("decision", "local_primitive"),
        ),
        "drone_watchdog": (
            ("drone_id",),
            ("triggered_reason",),
            ("watchdog",),
            ("watchdog", "last_progress_time"),
            ("watchdog", "distance_without_progress"),
            ("watchdog", "recent_visits"),
            ("watchdog", "reversal_count"),
            ("watchdog", "revisit_ratio"),
        ),
        "drone_motion": (
            ("drone_id",),
            ("travelled_distance",),
        ),
        "sensor_scan": (
            ("drone_id",),
            ("newly_known_cells",),
            ("confidence_gain",),
        ),
    }
    legacy_markers = {
        "waypoint_added": ("waypoint", "source"),
        "waypoint_edge_added": ("start", "end"),
        "drone_waypoint_route": ("target", "status", "route_elapsed_ms"),
        "drone_waypoint_segment_path": ("path_source", "astar_path_len"),
    }
    missing: list[str] = []
    replacement_events = 0
    legacy_events = 0
    stable_conflicts = 0
    stable_records: dict[tuple[str, Hashable], Hashable] = {}

    def remember(kind: str, identifier: Any, signature: Any) -> None:
        nonlocal stable_conflicts
        key_value = _hashable(identifier)
        if key_value is None:
            return
        key = (kind, key_value)
        normalized_signature = _hashable(signature)
        previous = stable_records.get(key)
        if previous is not None and previous != normalized_signature:
            stable_conflicts += 1
        else:
            stable_records[key] = normalized_signature

    for index, event in enumerate(events):
        event_name = str(event.get("event", ""))
        marker_fields = legacy_markers.get(event_name)
        if marker_fields and all(field in event for field in marker_fields):
            legacy_events += 1

        if (_integer(event.get("schema_version")) or 0) < 3:
            continue
        if event_name == "waypoint_graph_delta":
            replacement_events += 1
            for field in (
                "topology_revision",
                "added_nodes",
                "added_edges",
                "retired_node_ids",
                "retired_edge_ids",
                "edge_replacements",
            ):
                if field not in event:
                    missing.append(f"{index}:{event_name}.{field}")
            for node_index, node in enumerate(event.get("added_nodes", ())):
                if not isinstance(node, Mapping):
                    missing.append(
                        f"{index}:{event_name}.added_nodes[{node_index}]"
                    )
                    continue
                for field in ("node_id", "position", "roles"):
                    if field not in node:
                        missing.append(
                            f"{index}:{event_name}.added_nodes[{node_index}].{field}"
                        )
                remember("node", node.get("node_id"), node.get("position"))
            for edge_index, edge in enumerate(event.get("added_edges", ())):
                if not isinstance(edge, Mapping):
                    missing.append(
                        f"{index}:{event_name}.added_edges[{edge_index}]"
                    )
                    continue
                for field in ("edge_id", "start_node_id", "end_node_id"):
                    if field not in edge:
                        missing.append(
                            f"{index}:{event_name}.added_edges[{edge_index}].{field}"
                        )
                remember(
                    "edge",
                    edge.get("edge_id"),
                    (edge.get("start_node_id"), edge.get("end_node_id")),
                )
            continue

        event_requirements = requirements.get(event_name)
        if event_requirements is None:
            continue
        replacement_events += 1
        for path in event_requirements:
            if not _mapping_path_present(event, path):
                missing.append(f"{index}:{event_name}.{'.'.join(path)}")
        if event_name == "drone_navigation_transition":
            transition = _nested_mapping(event, "mode_transition")
            current_intent = _nested_mapping(event, "intent")
            terminal = transition.get("to_mode") is None
            identity_payload = (
                _nested_mapping(event, "previous_intent")
                if terminal and not current_intent
                else current_intent
            )
            for field in ("intent_id", "mode", "route_id"):
                if field not in identity_payload:
                    source = "previous_intent" if terminal else "intent"
                    missing.append(
                        f"{index}:{event_name}.{source}.{field}"
                    )
        route = _nested_mapping(event, "route")
        if route:
            remember(
                "route",
                route.get("route_id"),
                (
                    route.get("topology_revision"),
                    route.get("requester_knowledge_revision"),
                    route.get("edge_ids"),
                ),
            )

    valid = replacement_events > 0 and not missing and stable_conflicts == 0
    return TraceSchemaValidationMetrics(
        replacement_event_count=replacement_events,
        missing_field_count=len(missing),
        missing_fields=tuple(missing),
        stable_id_conflict_count=stable_conflicts,
        legacy_field_event_count=legacy_events,
        legacy_trace_fields_present=legacy_events > 0,
        valid=valid,
    )


def _planner_cave_map_field_count(
    events: Sequence[Mapping[str, Any]],
) -> int:
    """Count recursively exposed cave-map fields in planner-facing events."""
    planner_tokens = (
        "decision",
        "navigation",
        "intent",
        "route",
        "mcts",
        "watchdog",
        "frontier",
        "waypoint",
        "planner",
    )

    def nested_planner_payload(payload: Mapping[str, Any]) -> bool:
        return any(
            any(token in _normalized_label(key) for token in planner_tokens)
            for key in payload
        )

    def count(value: Any) -> int:
        if isinstance(value, Mapping):
            return sum(
                int(_normalized_label(key) == "cave_map") + count(item)
                for key, item in value.items()
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return sum(count(item) for item in value)
        return 0

    total = 0
    for event in events:
        name = _normalized_label(event.get("event", ""))
        if any(token in name for token in planner_tokens) or nested_planner_payload(event):
            total += count(event)
    return total


def _navigation_acceptance_counts(
    events: Sequence[Mapping[str, Any]],
) -> Mapping[str, int]:
    """Correlate stable goal changes and watchdog responses in event order."""
    counts: Counter[str] = Counter()
    previous_goals: dict[int, Hashable | None] = {}
    goal_observed: set[int] = set()
    previous_reversals: dict[int, int] = defaultdict(int)
    pending_watchdogs: dict[int, tuple[bool, str]] = {}
    planning_tick_events = DECISION_EVENTS | {
        "drone_local_mcts_decision",
        "drone_local_policy_decision",
        "drone_navigation_tick",
        "drone_intent_result",
    }
    retirement_events = {
        "drone_frontier_cluster_retired",
        "frontier_cluster_retired",
        "navigation_goal_retired",
    }

    for event in events:
        event_name = str(event.get("event", ""))
        drone_id = _integer(event.get("drone_id"))
        if drone_id is None:
            continue

        if event_name == "drone_watchdog":
            watchdog = _nested_mapping(event, "watchdog")
            reversal_count = _integer(watchdog.get("reversal_count"))
            revisit_ratio = _finite_float(watchdog.get("revisit_ratio"))
            distance = _finite_float(watchdog.get("distance_without_progress"))
            reason_value = event.get("triggered_reason")
            reason = "" if reason_value is None else _normalized_label(reason_value)
            previous = previous_reversals[drone_id]
            second_reversal = (
                reversal_count is not None
                and reversal_count >= 2
                and previous < 2
            ) or (
                reason == "reversal"
                and reversal_count is None
                and drone_id not in pending_watchdogs
            )
            if reversal_count is not None:
                previous_reversals[drone_id] = reversal_count
            threshold = bool(
                reason not in {"", "none"}
                or second_reversal
                or (revisit_ratio is not None and revisit_ratio >= 0.60)
                or (distance is not None and distance >= 64.0)
            )
            if threshold and drone_id not in pending_watchdogs:
                counts["watchdog_threshold_triggers"] += 1
                if second_reversal:
                    counts["second_reversal_triggers"] += 1
                pending_watchdogs[drone_id] = (second_reversal, reason)
            continue

        if event_name == "drone_navigation_transition":
            intent = _nested_mapping(event, "intent") or _nested_mapping(
                event, "navigation_intent"
            )
            goal_present = any(
                name in intent
                for name in (
                    "goal_cluster_id",
                    "frontier_cluster_id",
                    "cluster_id",
                )
            )
            if goal_present:
                goal = None
                for name in (
                    "goal_cluster_id",
                    "frontier_cluster_id",
                    "cluster_id",
                ):
                    if name in intent:
                        goal = _hashable(intent.get(name))
                        break
                previous_goal = previous_goals.get(drone_id)
                if drone_id not in goal_observed or previous_goal != goal:
                    counts["goal_changes"] += 1
                    transition = _nested_mapping(event, "mode_transition")
                    reason_value = event.get("replan_reason")
                    if reason_value is None:
                        reason_value = transition.get(
                            "reason",
                            event.get("transition_reason"),
                        )
                    reason = (
                        "" if reason_value is None else _normalized_label(reason_value)
                    )
                    selected_over_active_goal = (
                        reason == "selected"
                        and drone_id in goal_observed
                        and previous_goal is not None
                    )
                    if (
                        reason not in ALLOWED_GOAL_CHANGE_REASONS
                        or selected_over_active_goal
                    ):
                        counts["unexplained_goal_changes"] += 1
                previous_goals[drone_id] = goal
                goal_observed.add(drone_id)
            elif "intent" in event or "navigation_intent" in event:
                # A canonical transition with no current intent explicitly
                # ends the active-goal epoch.  Without this reset, the next
                # valid SELECTED transition looks like it replaced a still-
                # active goal and is reported as unexplained.
                previous_goals.pop(drone_id, None)
                goal_observed.discard(drone_id)

            pending = pending_watchdogs.pop(drone_id, None)
            if pending is not None:
                transition = _nested_mapping(event, "mode_transition")
                from_mode = transition.get("from_mode")
                to_mode = transition.get("to_mode")
                changed = (
                    to_mode is not None
                    and _normalized_label(to_mode) != _normalized_label(from_mode)
                )
                second_reversal, _reason = pending
                if second_reversal and _normalized_label(to_mode) == "recovery":
                    counts["second_reversal_recoveries"] += 1
                elif second_reversal and goal_present and goal is None:
                    counts["second_reversal_retirements"] += 1
                if not changed:
                    counts["watchdog_transition_delays"] += 1
            continue

        if event_name in retirement_events and drone_id in pending_watchdogs:
            second_reversal, reason = pending_watchdogs[drone_id]
            if second_reversal:
                counts["second_reversal_retirements"] += 1
                # Retirement satisfies the reversal-specific gate, but the
                # general watchdog gate still requires a mode transition by
                # the next planning tick.
                pending_watchdogs[drone_id] = (False, reason)
            continue

        if event_name in planning_tick_events and drone_id in pending_watchdogs:
            pending_watchdogs.pop(drone_id)
            counts["watchdog_transition_delays"] += 1

    counts["watchdog_transition_delays"] += len(pending_watchdogs)
    return counts


def _build_acceptance_metrics(
    events: Sequence[Mapping[str, Any]],
    *,
    density: WaypointDensityMetrics,
    reversals: ABAReversalMetrics,
    mcts: MCTSTimingMetrics,
    routes: RouteCacheMetrics,
    efficiency: InformationEfficiencyMetrics,
    schema: TraceSchemaValidationMetrics,
) -> AcceptanceMetrics:
    navigation = _navigation_acceptance_counts(events)
    return AcceptanceMetrics(
        forbidden_node_role_count=density.forbidden_node_role_count,
        average_nodes_per_occupied_cell=density.average_nodes_per_occupied_cell,
        neighbor_within_8_px_rate=density.neighbor_within_8_px_rate,
        degree_two_rate=density.degree_two_rate,
        maximum_nodes_in_cell=density.maximum_nodes_in_cell,
        repeated_route_cache_hit_rate=routes.repeated_cache_hit_rate,
        route_lookup_p95_ms=routes.p95_route_time_ms,
        goal_changes=navigation.get("goal_changes", 0),
        unexplained_goal_changes=navigation.get("unexplained_goal_changes", 0),
        second_reversal_triggers=navigation.get("second_reversal_triggers", 0),
        second_reversal_recoveries=navigation.get(
            "second_reversal_recoveries", 0
        ),
        second_reversal_retirements=navigation.get(
            "second_reversal_retirements", 0
        ),
        watchdog_threshold_triggers=navigation.get(
            "watchdog_threshold_triggers", 0
        ),
        watchdog_transition_delays=navigation.get(
            "watchdog_transition_delays", 0
        ),
        mcts_p99_elapsed_ms=mcts.p99_elapsed_ms,
        mcts_maximum_elapsed_ms=mcts.maximum_elapsed_ms,
        incomplete_root_coverage_count=mcts.incomplete_root_coverage_count,
        unsafe_incomplete_search_count=mcts.unsafe_incomplete_search_count,
        persistent_edge_astar_calls=routes.persistent_edge_astar_calls,
        connector_astar_calls=routes.connector_astar_calls,
        unchanged_valid_intent_replans=(
            routes.unchanged_valid_intent_replans
        ),
        newly_known_cells_per_travelled_px=(
            efficiency.newly_known_cells_per_travelled_px
        ),
        confidence_gain_per_travelled_px=(
            efficiency.confidence_gain_per_travelled_px
        ),
        planner_cave_map_field_count=_planner_cave_map_field_count(events),
        replacement_schema_valid=schema.valid,
        replacement_schema_missing_field_count=schema.missing_field_count,
        legacy_trace_fields_present=schema.legacy_trace_fields_present,
    )


def analyze_trace(
    events: Iterable[dict[str, Any]],
    *,
    spatial_cell_size: int = 32,
    legacy_mcts_budget_ms: float = LEGACY_MCTS_BUDGET_MS,
    reversal_window_start_s: float | None = None,
) -> RuntimeTraceMetrics:
    """Build all structured characterization metrics in one replay."""
    materialized = tuple(events)
    sequences = [_integer(event.get("sequence")) for event in materialized]
    if materialized and all(value is not None for value in sequences):
        # Schema v2 guarantees an in-lock sequence.  Legacy and partially
        # upgraded traces deliberately retain JSONL/file order, which is the
        # only reliable tiebreaker when perf/simulation timestamps coincide.
        materialized = tuple(
            event
            for _sequence, _index, event in sorted(
                zip(sequences, range(len(materialized)), materialized),
                key=lambda item: (item[0], item[1]),
            )
        )
    density = analyze_waypoint_density(
        materialized,
        spatial_cell_size=spatial_cell_size,
    )
    retention = analyze_target_retention(materialized)
    fallbacks = analyze_frontier_fallbacks(materialized)
    reversals = analyze_aba_reversals(
        materialized,
        normalized_window_start_s=reversal_window_start_s,
    )
    mcts = analyze_mcts_timing(
        materialized,
        legacy_budget_ms=legacy_mcts_budget_ms,
    )
    routes = analyze_route_cache(materialized)
    efficiency = analyze_information_efficiency(materialized)
    schema = _schema_validation(materialized)
    acceptance = _build_acceptance_metrics(
        materialized,
        density=density,
        reversals=reversals,
        mcts=mcts,
        routes=routes,
        efficiency=efficiency,
        schema=schema,
    )
    return RuntimeTraceMetrics(
        event_count=len(materialized),
        waypoint_density=density,
        target_retention=retention,
        frontier_fallbacks=fallbacks,
        aba_reversals=reversals,
        mcts=mcts,
        routes=routes,
        information_efficiency=efficiency,
        schema_validation=schema,
        acceptance=acceptance,
    )


def load_events(path: Path) -> Iterable[dict[str, Any]]:
    """Yield parsed events from one JSONL trace."""
    with path.open("r", encoding="utf-8") as trace_file:
        for line in trace_file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def latest_trace(log_dir: Path) -> Path:
    """Return the newest mission trace in a log directory."""
    traces = sorted(log_dir.glob("mission_trace_*.jsonl"))
    if not traces:
        raise FileNotFoundError(f"No mission traces found in {log_dir}")
    return traces[-1]


def _format_optional(value: float | None, suffix: str = "") -> str:
    """Format an optional scalar without turning missing telemetry into zero."""
    return "N/A" if value is None else f"{value:.2f}{suffix}"


def _format_rate(value: float | None) -> str:
    """Format an optional ratio as a percentage."""
    return "N/A" if value is None else f"{value * 100.0:.1f}%"


def format_characterization(metrics: RuntimeTraceMetrics) -> list[str]:
    """Format structured Phase-0 metrics for the command-line report."""
    density = metrics.waypoint_density
    retention = metrics.target_retention
    fallbacks = metrics.frontier_fallbacks
    reversals = metrics.aba_reversals
    mcts = metrics.mcts
    routes = metrics.routes
    efficiency = metrics.information_efficiency

    lines = ["", "Characterization:"]
    lines.append(
        "  waypoint topology: "
        f"nodes={density.node_count} "
        f"active_edges={density.edge_count} "
        f"edge_mutations={density.edge_mutation_count} "
        f"connections={density.unique_connection_count} "
        f"cells={density.occupied_spatial_cells} "
        f"max_cell={density.maximum_nodes_in_cell}"
    )
    lines.append(
        "  waypoint density: "
        f"median_nn={_format_optional(density.median_nearest_neighbor_px, 'px')} "
        f"neighbor_lt_8={_format_rate(density.neighbor_within_8_px_rate)} "
        f"neighbor_lt_16={_format_rate(density.neighbor_within_16_px_rate)} "
        f"degree_two={_format_rate(density.degree_two_rate)}"
    )
    lines.append(
        "  segment follow-up: "
        f"samples={retention.segment_followups} "
        f"retained={_format_rate(retention.retention_rate)} "
        f"coordinate_switch_proxy={_format_rate(retention.coordinate_switch_proxy_rate)} "
        f"true_route_abandonment={_format_rate(retention.route_abandonment_rate)} "
        f"route_cursor_continuation={_format_rate(retention.route_cursor_continuation_rate)}"
    )
    lines.append(
        "  frontier fallback: "
        f"count={fallbacks.fallback_count} "
        f"zero_reward={fallbacks.zero_reward_fallbacks} "
        f"regenerated_after_reach={fallbacks.regenerated_after_reach}"
    )
    lines.append(
        "  A-B-A reversals: "
        f"window_start={_format_optional(reversals.normalized_window_start_s, 's')} "
        f"aggregate={reversals.reversals}/{reversals.reversal_opportunities} "
        f"({_format_rate(reversals.reversal_rate)})"
    )
    for drone_id, drone_metrics in reversals.by_drone.items():
        lines.append(
            f"    d{drone_id}: "
            f"{drone_metrics.reversals}/"
            f"{drone_metrics.reversal_opportunities} "
            f"({_format_rate(drone_metrics.reversal_rate)})"
        )
    if mcts.search_count or mcts.timing_sample_count:
        lines.append(
            "  legacy MCTS: "
            f"searches={mcts.search_count} "
            f"iterations_le_1={mcts.at_most_one_iteration_count} "
            f"({_format_rate(mcts.at_most_one_iteration_rate)}) "
            f"median={_format_optional(mcts.median_elapsed_ms, 'ms')} "
            f"mean={_format_optional(mcts.mean_elapsed_ms, 'ms')} "
            f"p99={_format_optional(mcts.p99_elapsed_ms, 'ms')} "
            f"max={_format_optional(mcts.maximum_elapsed_ms, 'ms')} "
            f"incomplete_roots={mcts.incomplete_root_coverage_count} "
            f"over_budget={mcts.over_budget_count}/"
            f"{mcts.timing_sample_count} "
            f"budget={mcts.budget_ms:.2f}ms[{mcts.budget_source}]"
        )
    lines.append(
        "  waypoint routes: "
        f"calls={routes.route_calls} "
        f"ok={routes.successful_routes} "
        f"failed={routes.failed_routes} "
        f"unique_drone_targets={routes.unique_drone_target_pairs} "
        f"total={routes.total_route_time_ms:.2f}ms "
        f"p95={_format_optional(routes.p95_route_time_ms, 'ms')} "
        f"cache_hit_rate={_format_rate(routes.cache_hit_rate)} "
        f"repeated_cache_hit_rate={_format_rate(routes.repeated_cache_hit_rate)}"
    )
    distance = (
        "N/A"
        if efficiency.distance_source == "unavailable"
        else f"{efficiency.travelled_distance_px:.2f}px"
    )
    lines.append(
        "  sensor efficiency: "
        f"distance={distance} "
        f"new_cells={efficiency.newly_known_cells} "
        "new_cells_per_px="
        f"{_format_optional(efficiency.newly_known_cells_per_travelled_px)} "
        "confidence_per_px="
        f"{_format_optional(efficiency.confidence_gain_per_travelled_px)} "
        f"gain_coverage={efficiency.gain_samples}/"
        f"{efficiency.completed_scans}"
    )
    acceptance = metrics.acceptance
    schema = metrics.schema_validation
    precise_cells = (
        "N/A"
        if acceptance.newly_known_cells_per_travelled_px is None
        else f"{acceptance.newly_known_cells_per_travelled_px:.6f}"
    )
    precise_confidence = (
        "N/A"
        if acceptance.confidence_gain_per_travelled_px is None
        else f"{acceptance.confidence_gain_per_travelled_px:.6f}"
    )
    lines.extend([
        "",
        "Final acceptance:",
        (
            "  strategic graph gates: "
            f"forbidden_roles={acceptance.forbidden_node_role_count} "
            "avg_nodes_per_32="
            f"{_format_optional(acceptance.average_nodes_per_occupied_cell)} "
            "neighbor_lt_8="
            f"{_format_rate(acceptance.neighbor_within_8_px_rate)} "
            f"degree_two={_format_rate(acceptance.degree_two_rate)} "
            f"max_cell={acceptance.maximum_nodes_in_cell}"
        ),
        (
            "  goal/watchdog gates: "
            f"goal_changes={acceptance.goal_changes} "
            f"unexplained={acceptance.unexplained_goal_changes} "
            "aba_reversal="
            f"{_format_rate(reversals.reversal_rate)} "
            "second_reversal="
            f"{acceptance.second_reversal_recoveries + acceptance.second_reversal_retirements}/"
            f"{acceptance.second_reversal_triggers} "
            f"watchdog_delays={acceptance.watchdog_transition_delays}"
        ),
        (
            "  route execution gates: "
            "repeated_cache_hit="
            f"{_format_rate(acceptance.repeated_route_cache_hit_rate)} "
            f"lookup_p95={_format_optional(acceptance.route_lookup_p95_ms, 'ms')} "
            f"persistent_astar={acceptance.persistent_edge_astar_calls} "
            f"connector_astar={acceptance.connector_astar_calls} "
            "unchanged_valid_replans="
            f"{acceptance.unchanged_valid_intent_replans}"
        ),
        (
            "  belief/schema gates: "
            f"new_cells_per_px={precise_cells} "
            f"confidence_per_px={precise_confidence} "
            f"planner_cave_map_fields={acceptance.planner_cave_map_field_count} "
            f"replacement_valid={acceptance.replacement_schema_valid} "
            "replacement_missing="
            f"{acceptance.replacement_schema_missing_field_count} "
            f"stable_id_conflicts={schema.stable_id_conflict_count} "
            f"legacy_fields={acceptance.legacy_trace_fields_present}"
        ),
    ])
    if mcts.search_count or mcts.timing_sample_count:
        lines.append(
            "  legacy MCTS gates: "
            f"p99={_format_optional(acceptance.mcts_p99_elapsed_ms, 'ms')} "
            f"max={_format_optional(acceptance.mcts_maximum_elapsed_ms, 'ms')} "
            f"incomplete_roots={acceptance.incomplete_root_coverage_count} "
            f"unsafe_incomplete={acceptance.unsafe_incomplete_search_count}"
        )
    return lines


def _trace_event_time(event: Mapping[str, Any]) -> float | None:
    """Return the trace's monotonic event time from either clock field."""
    sim_time = _finite_float(event.get("sim_time"))
    if sim_time is not None:
        return sim_time
    return _finite_float(event.get("perf_time"))


def _exploration_completion_event_names(
    events: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Return terminal events appropriate to the trace's declared policy."""
    component_quiescence = any(
        event.get("event") == "mission_constructed"
        and str(event.get("exploration_completion", "")).startswith(
            "physical_team_quiescence"
        )
        for event in events
    )
    names = {"exploration_complete_presented"}
    if not component_quiescence:
        names.update({
            "team_wall_mapping_tolerance_reached",
            "team_wall_mapping_complete",
        })
    return frozenset(names)


def _sector_wait_intervals(
    events: Sequence[Mapping[str, Any]],
) -> list[tuple[int, str, float, float]]:
    """Separate standby from barrier waits, including waits open at shutdown."""
    times = [t for event in events if (t := _trace_event_time(event)) is not None]
    if not times:
        return []
    completion_events = _exploration_completion_event_names(events)
    terminals = [
        t for event in events
        if event.get("event") in completion_events
        or event.get("event") in {"mission_shutdown_started", "trace_closed"}
        if (t := _trace_event_time(event)) is not None
    ]
    end = min(terminals) if terminals else max(times)
    pending: dict[int, tuple[str, float]] = {}
    intervals: list[tuple[int, str, float, float]] = []
    for event in events:
        timestamp = _trace_event_time(event)
        drone_id = _integer(event.get("drone_id"))
        if timestamp is None or drone_id is None:
            continue
        name = event.get("event")
        if name in {"drone_sector_waiting_for_team", "drone_sector_standby"}:
            reason = "standby" if name == "drone_sector_standby" else "barrier"
            previous = pending.get(drone_id)
            if previous is not None:
                intervals.append((drone_id, previous[0], previous[1], min(end, timestamp)))
            pending[drone_id] = (reason, timestamp)
        elif name == "drone_sector_wait_completed":
            duration = _finite_float(event.get("waited_seconds"))
            previous = pending.pop(drone_id, None)
            if previous is None and duration is not None:
                previous = (str(event.get("wait_reason", "barrier")), timestamp - duration)
            if previous is not None:
                intervals.append((drone_id, previous[0], previous[1], min(end, timestamp)))
    intervals.extend((drone_id, reason, start, end)
                     for drone_id, (reason, start) in pending.items())
    return [item for item in intervals if item[3] >= item[2]]


def _sector_epoch_summary_lines(
    events: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Summarize assigned work, yield, travel, and termination per epoch."""
    assignments: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    publications = {
        generation: event for event in events
        if event.get("event") == "rover_sector_epoch_published"
        and (generation := _integer(event.get("generation"))) is not None
    }
    for event in events:
        if event.get("event") != "drone_sector_assigned":
            continue
        generation = _integer(event.get("generation"))
        if generation is not None and generation not in publications:
            assignments[generation].append(event)
    for generation, event in publications.items():
        assignments[generation] = [
            dict(item, sim_time=_trace_event_time(event))
            for item in event.get("assignments", ())
        ]
    if not assignments:
        return []

    starts = {
        generation: min(
            timestamp
            for event in generation_events
            if (timestamp := _trace_event_time(event)) is not None
        )
        for generation, generation_events in assignments.items()
        if any(_trace_event_time(event) is not None for event in generation_events)
    }
    if not starts:
        return []
    completion_events = _exploration_completion_event_names(events)
    terminal_times = [
        timestamp
        for event in events
        if (
            event.get("event") in completion_events
            or event.get("event") in {
                "mission_shutdown_started",
                "trace_closed",
            }
        )
        and (timestamp := _trace_event_time(event)) is not None
    ]
    trace_end = min(terminal_times) if terminal_times else max(
        timestamp for event in events
        if (timestamp := _trace_event_time(event)) is not None
    )
    wall_observations = [
        event
        for event in events
        if event.get("event") in {
            "frame_summary",
            "team_wall_mapping_tolerance_reached",
            "team_wall_mapping_complete",
            "exploration_complete_presented",
        }
        and _trace_event_time(event) is not None
    ]

    def wall_pixels(event: Mapping[str, Any]) -> int | None:
        direct = _integer(event.get("mapped_wall_pixels"))
        if direct is not None:
            return direct
        return _integer(
            _nested_mapping(event, "wall_mapping").get(
                "mapped_wall_pixels"
            )
        )

    lines = ["", "Sector epoch yield:"]
    wait_intervals = _sector_wait_intervals(events)
    ordered = sorted(starts)
    for index, generation in enumerate(ordered):
        start = starts[generation]
        end = (
            starts[ordered[index + 1]]
            if index + 1 < len(ordered)
            else trace_end
        )
        if end < start:
            end = start
        interval = [
            event
            for event in events
            if (
                (timestamp := _trace_event_time(event)) is not None
                and start <= timestamp < end
            )
        ]
        work_by_drone = {
            int(event.get("drone_id", -1)): int(
                event.get("frontier_cells", 0) or 0
            )
            for event in assignments[generation]
        }
        standby_ids = {
            int(event["drone_id"]) for event in assignments[generation]
            if event.get("standby", False)
        }
        active_ids = set(work_by_drone) - standby_ids
        waits = {
            reason: sum(
                max(0.0, min(end, wait_end) - max(start, wait_start))
                for _drone_id, wait_reason, wait_start, wait_end in wait_intervals
                if wait_reason == reason
            ) for reason in ("barrier", "standby")
        }
        standby_distance = sum(
            float(event.get("travelled_distance", 0.0) or 0.0)
            for event in interval if event.get("event") == "drone_motion"
            and _integer(event.get("drone_id")) in standby_ids
        )
        scope_values = [event.get("scope_pixels") for event in assignments[generation]
                        if not event.get("standby", False)]
        scope_pixels = (
            str(sum(int(value) for value in scope_values))
            if scope_values and all(value is not None for value in scope_values)
            else "N/A"
        )
        estimated_effort = sum(float(event.get("estimated_effort", 0.0) or 0.0)
                               for event in assignments[generation])
        effort_terms = "N/A"
        if all("effort_breakdown" in event for event in assignments[generation]):
            terms = ("frontier_scan", "approach", "setup", "dispersion", "unknown_area")
            totals = {
                term: sum(
                    float(_nested_mapping(event, "effort_breakdown").get(term, 0.0) or 0.0)
                    for event in assignments[generation]
                ) for term in terms
            }
            effort_terms = ",".join(f"{term}:{total:.2f}" for term, total in totals.items())
        excluded = sum(int(event.get("scope_excluded_frontier_count", 0) or 0)
                       for event in interval if event.get("event") == "drone_frontiers_rebuilt")
        sensor_gain = sum(
            int(event.get("newly_known_cells", 0) or 0)
            for event in interval
            if event.get("event") == "sensor_scan"
        )
        travelled = sum(
            float(event.get("travelled_distance", 0.0) or 0.0)
            for event in interval
            if event.get("event") == "drone_motion"
        )
        wall_start = max(
            (
                event
                for event in wall_observations
                if (
                    (timestamp := _trace_event_time(event)) is not None
                    and timestamp <= start
                )
            ),
            key=lambda event: (
                _trace_event_time(event)
                if _trace_event_time(event) is not None
                else -math.inf
            ),
            default=None,
        )
        wall_end = max(
            (
                event
                for event in wall_observations
                if (
                    (timestamp := _trace_event_time(event)) is not None
                    and timestamp <= end
                )
            ),
            key=lambda event: (
                _trace_event_time(event)
                if _trace_event_time(event) is not None
                else -math.inf
            ),
            default=None,
        )
        wall_delta: int | None = None
        if wall_start is not None and wall_end is not None:
            start_pixels = wall_pixels(wall_start)
            end_pixels = wall_pixels(wall_end)
            if start_pixels is not None and end_pixels is not None:
                wall_delta = end_pixels - start_pixels
        work = ",".join(
            f"d{drone_id}:{frontiers}"
            for drone_id, frontiers in sorted(work_by_drone.items())
        )
        efficiency = sensor_gain / travelled if travelled > 0.0 else None
        lines.append(
            f"  generation {generation}: duration={end - start:.2f}s "
            f"assigned_frontiers={sum(work_by_drone.values())} "
            f"work=[{work}] sensor_gain={sensor_gain} "
            f"distance={travelled:.2f}px "
            "gain_per_px="
            f"{('N/A' if efficiency is None else f'{efficiency:.4f}')} "
            f"wall_gain={('N/A' if wall_delta is None else wall_delta)} "
            "arrivals="
            f"{sum(event.get('event') == 'drone_frontier_reached' for event in interval)} "
            "route_rejections="
            f"{sum(event.get('event') == 'drone_frontier_route_rejected' for event in interval)} "
            "suppressions="
            f"{sum(event.get('event') == 'drone_border_target_suppressed' for event in interval)} "
            "sector_exhaustions="
            f"{sum(event.get('event') == 'drone_sector_exhausted' for event in interval)} "
            f"active={sorted(active_ids)} standby={sorted(standby_ids)} "
            f"scope_pixels={scope_pixels} estimated_effort={estimated_effort:.2f} "
            f"effort_terms=[{effort_terms}] "
            f"excluded_frontier_samples={excluded} "
            f"barrier_wait={waits['barrier']:.2f}s standby_time={waits['standby']:.2f}s "
            f"standby_distance={standby_distance:.2f}px"
        )
    return lines


def _component_exploration_summary_lines(
    events: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Summarize discovery rounds, lineage, claims, and quiescence."""
    component_event_names = {
        "rover_discovery_round_started",
        "rover_discovery_round_completed",
        "rover_frontier_registry_reconciled",
        "rover_frontier_lineage_changed",
        "rover_task_claimed",
        "rover_task_completed",
        "rover_branch_follow_assigned",
        "rover_branch_follow_completed",
        "drone_component_branch_claimed",
        "drone_component_branches_reserved_for_followers",
        "drone_component_branch_follow_ended",
        "drone_component_task_suspended",
        "drone_energy_return_required",
        "rover_exploration_quiescence_evaluated",
        "drone_component_target_adjusted_after_share",
        "drone_component_target_retired_after_share",
        "drone_route_interrupted_after_share",
        "drone_component_observation_pose_selected",
        "drone_component_sweep_advanced",
        "drone_dfs_backtrack_replanned",
        "drone_dfs_popped",
        "drone_dfs_reposition_started",
        "drone_dfs_reposition_path",
        "drone_dfs_reposition_fallback",
        "drone_component_check_in_queued",
        "rover_component_check_in_processed",
        "drone_docked",
        "drone_undocked",
        "rover_frontier_target_acquired",
        "rover_frontier_target_departure_authorized",
        "rover_frontier_staging_reached",
        "rover_frontier_target_invalidated",
        "rover_route_planned",
        "rover_focused_endgame_changed",
        "rover_focused_endgame_assignment",
        "rover_focused_endgame_staging_selected",
        "rover_focused_endgame_staging_held",
        "rover_route_negative_cached",
        "rover_route_negative_cache_hit",
        "rover_frontier_route_retry_suppressed",
        "rover_rendezvous_endpoint_proposed",
        "drone_rover_rendezvous_ack_exchanged",
        "drone_rendezvous_message_relayed",
        "drone_rendezvous_endpoint_fallback",
        "drone_component_rendezvous_retargeted",
        "rover_rendezvous_endpoint_reached",
        "exploration_complete_presented",
    }
    relevant = [
        event for event in events
        if event.get("event") in component_event_names
    ]
    if not relevant:
        return []

    round_starts = [
        event for event in relevant
        if event.get("event") == "rover_discovery_round_started"
    ]
    round_completions = [
        event for event in relevant
        if event.get("event") == "rover_discovery_round_completed"
    ]
    round_kinds = Counter(
        str(event.get("round_kind", "unknown")) for event in round_starts
    )
    lineages = Counter(
        str(event.get("transition_kind", "unknown"))
        for event in relevant
        if event.get("event") == "rover_frontier_lineage_changed"
    )
    claims = [
        event for event in relevant
        if event.get("event") == "rover_task_claimed"
    ]
    reports = [
        event for event in relevant
        if event.get("event") == "rover_task_completed"
    ]
    follower_assignments = [
        event for event in relevant
        if event.get("event") == "rover_branch_follow_assigned"
    ]
    follower_reports = [
        event for event in relevant
        if event.get("event") == "rover_branch_follow_completed"
    ]
    branch_claims = [
        event for event in relevant
        if event.get("event") == "drone_component_branch_claimed"
    ]
    suspension_reasons = Counter(
        str(event.get("suspension_reason", "unknown"))
        for event in reports if event.get("suspended")
    )
    outcomes: Counter[str] = Counter()
    for event in reports:
        for outcome in event.get("work_unit_outcomes", ()) or ():
            if isinstance(outcome, Mapping):
                outcomes[str(outcome.get("disposition", "unknown"))] += 1

    reconciliations = [
        event for event in relevant
        if event.get("event") == "rover_frontier_registry_reconciled"
    ]
    latest_registry = max(
        reconciliations,
        key=lambda event: _integer(event.get("revision")) or -1,
        default=None,
    )
    quiescence = [
        event for event in relevant
        if event.get("event") == "rover_exploration_quiescence_evaluated"
    ]
    homing_reasons = Counter(
        str(event.get("reason", "unknown")) for event in quiescence
    )
    waits: defaultdict[int, float] = defaultdict(float)
    for event in events:
        if event.get("event") != "drone_component_directive_started":
            continue
        drone_id = _integer(event.get("drone_id"))
        waited = _finite_float(event.get("waited_seconds"))
        if drone_id is not None and waited is not None and waited >= 0.0:
            waits[drone_id] += waited

    local_adaptation = Counter(
        str(event.get("event"))
        for event in relevant
        if event.get("event") in {
            "drone_component_target_adjusted_after_share",
            "drone_component_target_retired_after_share",
            "drone_component_sweep_advanced",
            "drone_dfs_backtrack_replanned",
        }
    )
    route_interruptions = [
        event for event in relevant
        if event.get("event") == "drone_route_interrupted_after_share"
    ]
    route_interruption_sources = Counter(
        str(event.get("source", "unknown"))
        for event in route_interruptions
    )
    observation_poses = [
        event for event in relevant
        if event.get("event") == "drone_component_observation_pose_selected"
    ]
    observation_sources = Counter(
        str(event.get("source", "unknown"))
        for event in observation_poses
    )
    observation_savings = [
        value for event in observation_poses
        if (
            value := _finite_float(event.get("saved_route_distance"))
        ) is not None and value >= 0.0
    ]
    dfs_navigation = Counter(
        str(event.get("event"))
        for event in relevant
        if event.get("event") in {
            "drone_dfs_popped",
            "drone_dfs_reposition_started",
            "drone_dfs_reposition_path",
            "drone_dfs_reposition_fallback",
        }
    )
    dfs_motion: defaultdict[str, float] = defaultdict(float)
    for event in events:
        if event.get("event") != "drone_motion":
            continue
        source = str(event.get("source", ""))
        if source not in {
            "component_dfs_reposition_astar",
            "component_dfs_breadcrumb_fallback",
            "component_dfs_backtrack",
            "component_dfs_backtrack_replan",
        }:
            continue
        distance = _finite_float(event.get("travelled_distance"))
        if distance is not None and distance >= 0.0:
            dfs_motion[source] += distance
    queued_check_ins = [
        event for event in relevant
        if event.get("event") == "drone_component_check_in_queued"
    ]
    processed_check_ins = [
        event for event in relevant
        if event.get("event") == "rover_component_check_in_processed"
    ]
    queue_wait_ms = [
        value for event in processed_check_ins
        if (value := _finite_float(event.get("queue_wait_ms"))) is not None
    ]
    processing_ms = [
        value for event in processed_check_ins
        if (value := _finite_float(event.get("processing_ms"))) is not None
    ]
    docked = [
        event for event in relevant
        if event.get("event") == "drone_docked"
    ]
    undocked = [
        event for event in relevant
        if event.get("event") == "drone_undocked"
    ]
    docking_sources = Counter(
        str(event.get("source", "unknown")) for event in docked
    )
    release_reasons = Counter(
        str(event.get("reason", "unknown")) for event in undocked
    )
    carried_distance = sum(
        value for event in undocked
        if (value := _finite_float(event.get("carried_distance"))) is not None
        and value >= 0.0
    )
    carried_steps = sum(
        max(0, _integer(event.get("carried_steps")) or 0)
        for event in undocked
    )
    docked_seconds = sum(
        value for event in undocked
        if (value := _finite_float(event.get("docked_seconds"))) is not None
        and value >= 0.0
    )
    learned_endpoint_epochs = {
        parsed
        for event in undocked
        for epoch in (event.get("learned_endpoint_epochs", ()) or ())
        if (parsed := _integer(epoch)) is not None
    }
    rover_staging = Counter(
        str(event.get("event"))
        for event in relevant
        if event.get("event") in {
            "rover_frontier_target_acquired",
            "rover_frontier_target_departure_authorized",
            "rover_frontier_staging_reached",
            "rover_frontier_target_invalidated",
            "rover_route_planned",
        }
    )
    focused_endgame = Counter(
        str(event.get("event"))
        for event in relevant
        if event.get("event") in {
            "rover_focused_endgame_assignment",
            "rover_focused_endgame_staging_selected",
            "rover_focused_endgame_staging_held",
            "rover_route_negative_cached",
            "rover_route_negative_cache_hit",
            "rover_frontier_route_retry_suppressed",
        }
    )
    focused_transitions = [
        event for event in relevant
        if event.get("event") == "rover_focused_endgame_changed"
    ]
    focused_activations = sum(
        _boolean(event.get("active")) is True
        for event in focused_transitions
    )
    focused_deactivations = sum(
        _boolean(event.get("active")) is False
        for event in focused_transitions
    )
    focused_assignments = [
        event for event in relevant
        if event.get("event") == "rover_focused_endgame_assignment"
    ]
    estimated_outbound = sum(
        value for event in focused_assignments
        if (
            value := _finite_float(event.get("estimated_outbound_cost"))
        ) is not None and value >= 0.0
    )
    estimated_round_trip = sum(
        value for event in focused_assignments
        if (
            value := _finite_float(event.get("estimated_round_trip_cost"))
        ) is not None and value >= 0.0
    )
    report_distances = {
        field: sum(
            value for event in reports
            if (value := _finite_float(event.get(field))) is not None
            and value >= 0.0
        )
        for field in (
            "outbound_distance",
            "service_distance",
            "return_distance",
        )
    }
    rendezvous = Counter(
        str(event.get("event"))
        for event in relevant
        if event.get("event") in {
            "rover_rendezvous_endpoint_proposed",
            "drone_rover_rendezvous_ack_exchanged",
            "drone_rendezvous_message_relayed",
            "drone_rendezvous_endpoint_fallback",
            "rover_rendezvous_endpoint_reached",
        }
    )

    lines = ["", "Component exploration:"]
    lines.append(
        "  discovery rounds: "
        f"started={len(round_starts)} completed={len(round_completions)} "
        f"rover_scan={round_kinds['rover_scan']} "
        f"radial_probe={round_kinds['radial_probe']}"
    )
    if latest_registry is not None:
        modes = latest_registry.get("exploration_modes", {}) or {}
        lines.append(
            "  latest registry: "
            f"revision={latest_registry.get('revision', 'unknown')} "
            f"active_components={latest_registry.get('active_component_count', 0)} "
            f"ready_units={latest_registry.get('ready_work_unit_count', 0)} "
            f"dormant={latest_registry.get('dormant_component_count', 0)} "
            f"resolved={latest_registry.get('resolved_component_count', 0)} "
            f"modes={dict(sorted(modes.items())) if isinstance(modes, Mapping) else {}}"
        )
    lines.append(
        "  tasks: "
        f"claims={len(claims)} reports={len(reports)} "
        f"outcomes={dict(sorted(outcomes.items()))} "
        f"suspensions={dict(sorted(suspension_reasons.items()))}"
    )
    if follower_assignments or follower_reports or branch_claims:
        lines.append(
            "  bootstrap followers: "
            f"assigned={len(follower_assignments)} "
            f"branches_claimed={len(branch_claims)} "
            f"reports={len(follower_reports)}"
        )
    lines.append(
        "  lineage: " + str(dict(sorted(lineages.items())))
    )
    if local_adaptation:
        lines.append(
            "  local autonomy: "
            "shared_adjustments="
            f"{local_adaptation['drone_component_target_adjusted_after_share']} "
            "shared_retirements="
            f"{local_adaptation['drone_component_target_retired_after_share']} "
            "sweep_advances="
            f"{local_adaptation['drone_component_sweep_advanced']} "
            "backtrack_replans="
            f"{local_adaptation['drone_dfs_backtrack_replanned']}"
        )
    if route_interruptions:
        lines.append(
            "  mid-route peer sharing: "
            f"interruptions={len(route_interruptions)} "
            f"sources={dict(sorted(route_interruption_sources.items()))}"
        )
    if observation_poses:
        lines.append(
            "  focused observation poses: "
            f"selected={len(observation_poses)} "
            "shortened="
            f"{sum(value > 1e-9 for value in observation_savings)} "
            f"route_suffix_avoided={sum(observation_savings):.1f}px "
            f"sources={dict(sorted(observation_sources.items()))}"
        )
    if dfs_navigation or dfs_motion:
        lines.append(
            "  DFS navigation: "
            f"logical_pops={dfs_navigation['drone_dfs_popped']} "
            f"repositions={dfs_navigation['drone_dfs_reposition_started']} "
            f"astar_px={dfs_motion['component_dfs_reposition_astar']:.1f} "
            f"breadcrumb_px={dfs_motion['component_dfs_breadcrumb_fallback']:.1f} "
            "legacy_retrace_px="
            f"{dfs_motion['component_dfs_backtrack'] + dfs_motion['component_dfs_backtrack_replan']:.1f}"
        )
    if queued_check_ins or processed_check_ins:
        lines.append(
            "  async rover check-ins: "
            f"queued={len(queued_check_ins)} processed={len(processed_check_ins)} "
            "queue_wait_avg="
            f"{_format_optional(statistics.mean(queue_wait_ms) if queue_wait_ms else None, 'ms')} "
            "queue_wait_max="
            f"{_format_optional(max(queue_wait_ms, default=None), 'ms')} "
            "processing_avg="
            f"{_format_optional(statistics.mean(processing_ms) if processing_ms else None, 'ms')} "
            "processing_max="
            f"{_format_optional(max(processing_ms, default=None), 'ms')}"
        )
    if docked or undocked:
        lines.append(
            "  docking: "
            f"acquired={len(docked)} released={len(undocked)} "
            f"active_at_trace_end={max(0, len(docked) - len(undocked))} "
            "intercepts="
            f"{sum(count for source, count in docking_sources.items() if 'intercept' in source)} "
            f"carried={carried_distance:.1f}px steps={carried_steps} "
            f"docked={docked_seconds:.2f}s "
            f"learned_epochs={sorted(learned_endpoint_epochs)} "
            f"sources={dict(sorted(docking_sources.items()))} "
            f"releases={dict(sorted(release_reasons.items()))}"
        )
    if rover_staging:
        lines.append(
            "  moving rovers: "
            f"targets={rover_staging['rover_frontier_target_acquired']} "
            "departures_authorized="
            f"{rover_staging['rover_frontier_target_departure_authorized']} "
            f"reached={rover_staging['rover_frontier_staging_reached']} "
            f"invalidated={rover_staging['rover_frontier_target_invalidated']} "
            f"routes={rover_staging['rover_route_planned']}"
        )
    if focused_transitions or focused_endgame:
        lines.append(
            "  focused endgame economics: "
            f"activations={focused_activations} "
            f"deactivations={focused_deactivations} "
            "assignments="
            f"{focused_endgame['rover_focused_endgame_assignment']} "
            f"estimated_outbound={estimated_outbound:.1f}px "
            f"estimated_round_trip={estimated_round_trip:.1f}px "
            "staging_selected="
            f"{focused_endgame['rover_focused_endgame_staging_selected']} "
            "staging_held="
            f"{focused_endgame['rover_focused_endgame_staging_held']} "
            "route_cached="
            f"{focused_endgame['rover_route_negative_cached']} "
            f"cache_hits={focused_endgame['rover_route_negative_cache_hit']} "
            "retry_suppressed="
            f"{focused_endgame['rover_frontier_route_retry_suppressed']}"
        )
    if reports and any(report_distances.values()):
        lines.append(
            "  reported sortie distance: "
            f"outbound={report_distances['outbound_distance']:.1f}px "
            f"service={report_distances['service_distance']:.1f}px "
            f"return={report_distances['return_distance']:.1f}px"
        )
    if rendezvous:
        lines.append(
            "  rendezvous protocol: "
            f"proposals={rendezvous['rover_rendezvous_endpoint_proposed']} "
            f"direct_acks={rendezvous['drone_rover_rendezvous_ack_exchanged']} "
            f"relays={rendezvous['drone_rendezvous_message_relayed']} "
            "fallbacks="
            f"{rendezvous['drone_rendezvous_endpoint_fallback']} "
            f"reached={rendezvous['rover_rendezvous_endpoint_reached']}"
        )
        fallback_events = [
            event for event in relevant
            if event.get("event") == "drone_rendezvous_endpoint_fallback"
        ]
        queued_checkins = [
            event for event in relevant
            if event.get("event") == "drone_component_check_in_queued"
        ]
        fallback_sources = Counter(
            str(event.get("target_source", "unrecorded"))
            for event in fallback_events
        )
        queued_contact_fallbacks = sum(
            1 for fallback in fallback_events
            if any(
                queued.get("drone_id") == fallback.get("drone_id")
                and (fallback_time := _trace_event_time(fallback)) is not None
                and (queued_time := _trace_event_time(queued)) is not None
                and abs(fallback_time - queued_time) <= 0.02
                for queued in queued_checkins
            )
        )
        if fallback_events:
            lines.append(
                "  rendezvous fallbacks: "
                f"sources={dict(sorted(fallback_sources.items()))} "
                f"same_tick_queued_checkins={queued_contact_fallbacks}"
            )
    if waits:
        lines.append(
            "  completed wait by drone: "
            + ", ".join(
                f"{drone_id}={duration:.2f}s"
                for drone_id, duration in sorted(waits.items())
            )
        )
    if quiescence:
        lines.append(
            "  quiescence: "
            f"events={len(quiescence)} reasons={dict(sorted(homing_reasons.items()))} "
            f"max_ready_tasks={max(int(event.get('ready_task_count', 0) or 0) for event in quiescence)} "
            f"max_live_claims={max(int(event.get('live_claim_count', 0) or 0) for event in quiescence)}"
        )
    return lines


def _endgame_cost_lines(
    events: Sequence[Mapping[str, Any]],
    completion_trigger: Mapping[str, Any] | None,
) -> list[str]:
    """Compare display coverage with observed travel and rover knowledge.

    This is offline telemetry. The mission-wide floor ratio is not a lawful
    coordinator input because distant drones update it without rover contact.
    """
    frames = sorted(
        (
            (event_time, event, ratio)
            for event in events
            if event.get("event") == "frame_summary"
            if (event_time := _trace_event_time(event)) is not None
            if (ratio := _finite_float(event.get("floor_exploration_ratio")))
            is not None
        ),
        key=lambda item: item[0],
    )
    if not frames:
        return []
    completion_time = (
        None if completion_trigger is None
        else _trace_event_time(completion_trigger)
    )
    if completion_time is not None:
        frames = [frame for frame in frames if frame[0] <= completion_time]
    if not frames:
        return []
    final_frame_time, _end_frame, end_ratio = frames[-1]
    end_time = (
        final_frame_time if completion_time is None else completion_time
    )
    if completion_trigger is not None:
        end_ratio = (
            _finite_float(completion_trigger.get("floor_exploration_ratio"))
            or end_ratio
        )
    mission_start = next(
        (
            time for event in events
            if event.get("event") == "mission_run_started"
            if (time := _trace_event_time(event)) is not None
        ),
        frames[0][0],
    )
    checkins = sorted(
        (
            (time, event)
            for event in events
            if event.get("event") == "drone_component_check_in"
            if (time := _trace_event_time(event)) is not None
            and time <= end_time
        ),
        key=lambda item: item[0],
    )

    def rover_position(frame: Mapping[str, Any]) -> tuple[float, float] | None:
        states = frame.get("rover_states") or ()
        if not states:
            return None
        position = states[0].get("position") or ()
        if len(position) != 2:
            return None
        x, y = (_finite_float(value) for value in position)
        return None if x is None or y is None else (x, y)

    lines: list[str] = []
    for threshold in (0.97, 0.98, 0.985):
        crossing = next(
            (index for index, frame in enumerate(frames)
             if frame[2] >= threshold),
            None,
        )
        if crossing is None:
            continue
        start_time, _start_frame, start_ratio = frames[crossing]
        rover_distance = 0.0
        for earlier, later in zip(frames[crossing:], frames[crossing + 1:]):
            old_position = rover_position(earlier[1])
            new_position = rover_position(later[1])
            if old_position is not None and new_position is not None:
                rover_distance += math.dist(old_position, new_position)

        drone_distance = 0.0
        sensor_slam_gain = 0
        counts: Counter[str] = Counter()
        for event in events:
            name = event.get("event")
            time = _trace_event_time(event)
            if name == "drone_motion":
                distance = _finite_float(event.get("travelled_distance"))
                began = _finite_float(event.get("started_sim_time"))
                ended = _finite_float(event.get("ended_sim_time"))
                if ended is None:
                    ended = time
                if distance is None or distance < 0.0 or ended is None:
                    continue
                if began is None or began >= ended:
                    if start_time <= ended <= end_time:
                        drone_distance += distance
                else:
                    overlap = max(
                        0.0,
                        min(ended, end_time) - max(began, start_time),
                    )
                    drone_distance += distance * overlap / (ended - began)
            elif time is not None and start_time <= time <= end_time:
                if name == "sensor_scan":
                    sensor_slam_gain += max(
                        0, _integer(event.get("newly_known_cells")) or 0
                    )
                elif name in {
                    "rover_task_claimed",
                    "rover_task_completed",
                    "rover_rendezvous_endpoint_proposed",
                }:
                    counts[str(name)] += 1

        earlier_checkin = next(
            (event for time, event in reversed(checkins)
             if time <= start_time),
            None,
        )
        latest_checkin = (
            checkins[-1][1]
            if checkins and checkins[-1][0] > start_time else None
        )

        def known_delta(field: str) -> str:
            if earlier_checkin is None or latest_checkin is None:
                return "N/A"
            earlier = _integer(earlier_checkin.get(field))
            latest = _integer(latest_checkin.get(field))
            if earlier is None or latest is None:
                return "N/A"
            return str(max(0, latest - earlier))

        if not lines:
            lines.extend([
                "",
                "Endgame cost (offline floor; sampled rover distance/knowledge):",
            ])
        lines.append(
            f"  from {start_ratio * 100.0:.2f}% at "
            f"t={start_time - mission_start:.1f}s: "
            f"floor_gain={max(0.0, end_ratio - start_ratio) * 100.0:.3f}pp "
            f"elapsed={end_time - start_time:.1f}s "
            f"drone={drone_distance:.0f}px "
            f"rover_sampled={rover_distance:.0f}px "
            f"sensor_slam_gain={sensor_slam_gain} "
            "rover_known_slam_delta="
            f"{known_delta('rover_slam_newly_known_cells')} "
            "rover_known_terrain_delta="
            f"{known_delta('rover_terrain_known_floor_cells')} "
            f"claims={counts['rover_task_claimed']} "
            f"reports={counts['rover_task_completed']} "
            f"proposals={counts['rover_rendezvous_endpoint_proposed']}"
        )
    return lines


def summarize(
    events: Iterable[dict[str, Any]],
    *,
    reversal_window_start_s: float | None = None,
) -> list[str]:
    """Build a compact text summary of drone decision and path events."""
    materialized = tuple(events)
    completion_event_names = _exploration_completion_event_names(
        materialized
    )
    metrics = analyze_trace(
        materialized,
        reversal_window_start_s=reversal_window_start_s,
    )
    event_counts: Counter[str] = Counter()
    per_drone_counts: dict[int, Counter[str]] = defaultdict(Counter)
    last_by_drone: dict[int, deque[dict[str, Any]]] = defaultdict(
        lambda: deque(maxlen=12)
    )
    last_decision: dict[int, dict[str, Any]] = {}
    waypoint_route_statuses: dict[int, Counter[str]] = defaultdict(Counter)
    waypoint_bridge_statuses: dict[int, Counter[str]] = defaultdict(Counter)
    waypoint_gateway_statuses: dict[int, Counter[str]] = defaultdict(Counter)
    waypoint_segment_sources: dict[int, Counter[str]] = defaultdict(Counter)
    waypoint_route_time_total: dict[int, float] = defaultdict(float)
    waypoint_route_time_max: dict[int, float] = defaultdict(float)
    stagnation_scan_dispositions: dict[int, Counter[str]] = defaultdict(
        Counter
    )
    stagnation_scan_sensor_cells: dict[int, int] = defaultdict(int)
    stagnation_scan_confidence_gain: dict[int, float] = defaultdict(float)
    scan_exit_resume_deltas: dict[int, list[float]] = defaultdict(list)
    scan_exit_exact_resumes: Counter[int] = Counter()
    sensor_stage_timings: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    heading_selection_modes: dict[int, Counter[str]] = defaultdict(Counter)
    heading_cluster_sizes: dict[int, list[int]] = defaultdict(list)
    heading_cluster_scores: dict[int, list[tuple[float, float, float]]] = (
        defaultdict(list)
    )
    heading_cluster_totals: dict[int, Counter[str]] = defaultdict(Counter)
    global_frontier_sizes: dict[int, list[int]] = defaultdict(list)
    global_frontier_distances: dict[int, list[float]] = defaultdict(list)
    global_frontier_ownership: dict[
        int,
        list[tuple[float, float, float, float, float]],
    ] = defaultdict(list)
    global_frontier_cache_ms: dict[int, list[float]] = defaultdict(list)
    global_frontier_cache_totals: dict[int, Counter[str]] = defaultdict(
        Counter
    )
    astar_path_statuses: dict[int, Counter[str]] = defaultdict(Counter)
    frontier_route_circuities: dict[int, list[float]] = defaultdict(list)
    sector_wait_durations: dict[int, list[float]] = defaultdict(list)
    sector_standby_durations: dict[int, list[float]] = defaultdict(list)
    for drone_id, reason, start, end in _sector_wait_intervals(materialized):
        destination = sector_standby_durations if reason == "standby" else sector_wait_durations
        destination[drone_id].append(end - start)
    coverage_heading_terms: dict[
        int,
        list[tuple[float, float, float]],
    ] = defaultdict(list)
    coverage_motion_totals: dict[int, Counter[str]] = defaultdict(Counter)
    coverage_known_counts: dict[int, tuple[int, int]] = {}
    partial_segment_outcomes: dict[int, Counter[str]] = defaultdict(Counter)
    waypoint_graph_size: dict[int, tuple[int, int]] = {}
    last_frame: dict[str, Any] | None = None
    completion_trigger: dict[str, Any] | None = None
    sector_frontier_filters: list[dict[str, Any]] = []
    sector_frontier_outcomes: list[dict[str, Any]] = []
    sector_workload_balances: list[dict[str, Any]] = []
    sharing_protocol: dict[str, Counter[str]] = defaultdict(Counter)
    trace_path = "-"
    trace_times = [
        timestamp
        for event in materialized
        if (timestamp := _trace_event_time(event)) is not None
    ]
    trace_start_time = min(trace_times, default=0.0)

    for event in materialized:
        event_name = str(event.get("event", "unknown"))
        event_counts[event_name] += 1
        if event_name == "trace_started":
            trace_path = str(event.get("path", "-"))
        if event_name == "frame_summary":
            last_frame = event
        if event_name in completion_event_names:
            completion_trigger = event
        if event_name == "rover_sector_frontiers_filtered":
            sector_frontier_filters.append(event)
        if event_name == "rover_sector_frontier_outcomes":
            sector_frontier_outcomes.append(event)
        if event_name == "rover_sector_workload_balanced":
            sector_workload_balances.append(event)
        if event_name in {
            "drone_rover_check_in",
            "drone_rover_departure_share",
            "drone_rover_proximity_share",
            "drone_sharing_pair",
            "drone_sharing_suppressed",
        }:
            sharing_protocol[event_name][
                str(event.get("reason", "unknown"))
            ] += 1

        drone_id = event.get("drone_id")
        if drone_id is None:
            continue
        drone_id = int(drone_id)
        per_drone_counts[drone_id][event_name] += 1
        if event_name == "sensor_scan":
            for field in (
                "vision_elapsed_ms",
                "slam_elapsed_ms",
                "terrain_elapsed_ms",
                "sensor_elapsed_ms",
            ):
                elapsed = _finite_float(event.get(field))
                if elapsed is not None and elapsed >= 0.0:
                    sensor_stage_timings[drone_id][field].append(elapsed)
        if event_name == "drone_stagnation_scan_completed":
            stagnation_scan_dispositions[drone_id][
                str(event.get("disposition", "unknown"))
            ] += 1
            stagnation_scan_sensor_cells[drone_id] += int(
                event.get("sensor_newly_known_cells", 0) or 0
            )
            stagnation_scan_confidence_gain[drone_id] += float(
                event.get("sensor_confidence_gain", 0.0) or 0.0
            )
        if event_name == "drone_stagnation_scan_exit_reoriented":
            resume_delta = _finite_float(
                event.get("resume_heading_delta")
            )
            if resume_delta is not None and resume_delta >= 0.0:
                scan_exit_resume_deltas[drone_id].append(resume_delta)
                if bool(event.get("exact_resume", False)):
                    scan_exit_exact_resumes[drone_id] += 1
        if event_name == "drone_random_direction_selected":
            heading_selection_modes[drone_id][
                str(event.get("selection_mode", "legacy_uniform"))
            ] += 1
            coverage_factor = _finite_float(
                event.get("selected_coverage_penalty_factor")
            )
            if coverage_factor is not None:
                coverage_heading_terms[drone_id].append((
                    float(event.get(
                        "selected_coverage_visit_pressure", 0.0,
                    ) or 0.0),
                    float(event.get(
                        "selected_coverage_edge_pressure", 0.0,
                    ) or 0.0),
                    coverage_factor,
                ))
            selected_size = int(
                event.get("selected_frontier_cluster_size", 0) or 0
            )
            if selected_size > 0:
                heading_cluster_sizes[drone_id].append(selected_size)
            if (
                selected_size > 0
                and "selected_frontier_cluster_score" in event
            ):
                heading_cluster_scores[drone_id].append((
                    float(event.get("selected_frontier_cluster_score", 0.0)),
                    float(event.get(
                        "selected_frontier_cluster_size_rank", 0.0,
                    )),
                    float(event.get(
                        "selected_frontier_cluster_proximity", 0.0,
                    )),
                ))
            heading_cluster_totals[drone_id]["observed"] += int(
                event.get("frontier_cluster_count", 0) or 0
            )
            heading_cluster_totals[drone_id]["eligible"] += int(
                event.get("eligible_frontier_cluster_count", 0) or 0
            )
            heading_cluster_totals[drone_id]["filtered"] += int(
                event.get("filtered_frontier_cluster_count", 0) or 0
            )
            heading_cluster_totals[drone_id]["wall_candidates"] += int(
                event.get("wall_frontier_candidate_count", 0) or 0
            )
            heading_cluster_totals[drone_id]["generic_candidates"] += int(
                event.get("generic_frontier_candidate_count", 0) or 0
            )
            if bool(event.get("global_frontier_active", False)):
                global_frontier_sizes[drone_id].append(int(
                    event.get("global_frontier_region_size", 0) or 0
                ))
                distance = event.get("global_frontier_region_distance")
                if distance is not None:
                    global_frontier_distances[drone_id].append(
                        float(distance)
                    )
                if "global_frontier_ownership_margin" in event:
                    global_frontier_ownership[drone_id].append((
                        float(event.get(
                            "global_frontier_ownership_margin", 0.0,
                        ) or 0.0),
                        float(event.get(
                            "global_frontier_launch_sector_alignment", 0.0,
                        ) or 0.0),
                        float(event.get(
                            "global_frontier_ownership_contribution", 0.0,
                        ) or 0.0),
                        float(event.get(
                            "global_frontier_requester_distance", 0.0,
                        ) or 0.0),
                        float(event.get(
                            "global_frontier_nearest_peer_distance", 0.0,
                        ) or 0.0),
                    ))
        if event_name == "drone_motion" and (
            "coverage_cell_entries" in event
        ):
            totals = coverage_motion_totals[drone_id]
            for field in (
                "coverage_cell_entries",
                "coverage_new_cell_entries",
                "coverage_revisit_entries",
                "coverage_repeated_edge_entries",
            ):
                totals[field] += int(event.get(field, 0) or 0)
            coverage_known_counts[drone_id] = (
                int(event.get("coverage_known_cell_count", 0) or 0),
                int(event.get("coverage_known_edge_count", 0) or 0),
            )
        if event_name == "drone_global_frontiers_rebuilt":
            global_frontier_cache_ms[drone_id].append(float(
                event.get("elapsed_ms", 0.0) or 0.0
            ))
            global_frontier_cache_totals[drone_id]["regions"] += int(
                event.get("region_count", 0) or 0
            )
            global_frontier_cache_totals[drone_id]["eligible"] += int(
                event.get("eligible_region_count", 0) or 0
            )
            global_frontier_cache_totals[drone_id]["filtered"] += int(
                event.get("filtered_region_count", 0) or 0
            )
            if "selected_target_retained" in event:
                global_frontier_cache_totals[drone_id][
                    "retention_samples"
                ] += 1
                global_frontier_cache_totals[drone_id]["retained"] += int(
                    bool(event.get("selected_target_retained", False))
                )
            global_frontier_cache_totals[drone_id]["forced"] += int(
                bool(event.get("forced", False))
            )
            global_frontier_cache_totals[drone_id]["suppressed"] += int(
                event.get("suppressed_region_count", 0) or 0
            )
        if event_name in {
            "drone_border_path",
            "drone_global_frontier_path",
            "drone_homing_path",
            "drone_sector_ingress_recovery",
        }:
            route_kind = {
                "drone_border_path": "border",
                "drone_global_frontier_path": "global",
                "drone_homing_path": "home",
                "drone_sector_ingress_recovery": "sector_ingress",
            }[event_name]
            astar_path_statuses[drone_id][
                f"{route_kind}:{event.get('path_status', 'legacy')}"
            ] += 1
            if event_name == "drone_border_path":
                circuity = _finite_float(event.get("route_circuity"))
                if circuity is not None and circuity >= 1.0:
                    frontier_route_circuities[drone_id].append(circuity)
        if event_name == "drone_astar_partial_segment":
            partial_segment_outcomes[drone_id][
                "accepted" if event.get("accepted") else "rejected"
            ] += 1
        if event_name == "drone_sector_wait_completed" and _trace_event_time(event) is None:
            waited = _finite_float(event.get("waited_seconds"))
            if waited is not None and waited >= 0.0:
                destination = (sector_standby_durations
                               if event.get("wait_reason") == "standby"
                               else sector_wait_durations)
                destination[drone_id].append(waited)
        last_by_drone[drone_id].append(event)
        if event_name == "drone_waypoint_route":
            waypoint_route_statuses[drone_id][
                str(event.get("status", "unknown"))
            ] += 1
            bridge_status = event.get("bridge_status")
            if bridge_status is not None:
                waypoint_bridge_statuses[drone_id][
                    str(bridge_status)
                ] += 1
            waypoint_gateway_statuses[drone_id][
                str(event.get("gateway_status", "unknown"))
            ] += 1
            route_elapsed_ms = float(event.get("route_elapsed_ms", 0.0))
            if math.isfinite(route_elapsed_ms):
                waypoint_route_time_total[drone_id] += route_elapsed_ms
                waypoint_route_time_max[drone_id] = max(
                    waypoint_route_time_max[drone_id],
                    route_elapsed_ms,
                )
            waypoint_graph_size[drone_id] = (
                int(event.get("graph_nodes", 0)),
                int(event.get("graph_edges", 0)),
            )
        if event_name == "drone_waypoint_segment_path":
            waypoint_segment_sources[drone_id][
                str(event.get("path_source", "unknown"))
            ] += 1
        if event_name in {"drone_decision", "drone_post_rebuild_decision"}:
            last_decision[drone_id] = event

    lines = [f"Trace: {trace_path}", ""]
    lines.append("Top events:")
    for name, count in event_counts.most_common(12):
        lines.append(f"  {name}: {count}")
    lines.extend(format_characterization(metrics))
    lines.extend(_component_exploration_summary_lines(materialized))
    lines.extend(_sector_epoch_summary_lines(materialized))
    lines.extend(_endgame_cost_lines(materialized, completion_trigger))

    mission_started = next((
        _trace_event_time(event)
        for event in materialized
        if event.get("event") == "mission_run_started"
    ), None)
    mission_ended = next((
        _trace_event_time(event)
        for event in reversed(materialized)
        if event.get("event") in {
            "mission_shutdown_complete",
            "trace_closed",
        }
    ), None)
    completion_time = (
        None
        if completion_trigger is None
        else _trace_event_time(completion_trigger)
    )
    if mission_started is not None:
        lines.extend(["", "Mission timing:"])
        if completion_time is not None:
            lines.append(
                "  completion="
                f"{max(0.0, completion_time - mission_started):.2f}s"
            )
        if mission_ended is not None:
            lines.append(
                "  shutdown_complete="
                f"{max(0.0, mission_ended - mission_started):.2f}s"
            )

    if completion_trigger is not None:
        if completion_trigger.get("event") == "exploration_complete_presented":
            ratio = _finite_float(
                completion_trigger.get("floor_exploration_ratio")
            )
            coverage = (
                "N/A" if ratio is None else f"{ratio * 100.0:.2f}%"
            )
            description = (
                "Completion trigger: exploration_complete_presented "
                f"floor={coverage}"
            )
        else:
            missing = completion_trigger.get("missing_wall_pixels", 0)
            description = (
                "Completion trigger: "
                f"{completion_trigger.get('event')} "
                f"mapped={completion_trigger.get('mapped_wall_pixels')}"
                f"/{completion_trigger.get('total_wall_pixels')} "
                f"missing={missing}"
            )
        lines.extend(["", description])

    if sharing_protocol:
        lines.extend(["", "Sharing protocol:"])
        labels = {
            "drone_rover_check_in": "rover arrivals",
            "drone_rover_departure_share": "rover departures",
            "drone_rover_proximity_share": "periodic rover shares",
            "drone_sharing_pair": "drone pairs",
            "drone_sharing_suppressed": "rover-area pair suppressions",
        }
        for event_name, label in labels.items():
            outcomes = sharing_protocol.get(event_name)
            if not outcomes:
                continue
            details = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(outcomes.items())
            )
            lines.append(
                f"  {label}: total={sum(outcomes.values())} {details}"
            )

    if sector_frontier_filters:
        lines.extend(["", "Rover frontier significance:"])
        for event in sector_frontier_filters:
            lines.append(
                "  generation "
                f"{event.get('generation')}: "
                f"raw={event.get('raw_frontier_pixels', 0)}px/"
                f"{event.get('raw_component_count', 0)} components, "
                "significant="
                f"{event.get('significant_frontier_pixels', 0)}px/"
                f"{event.get('significant_component_count', 0)} components, "
                f"rescued={event.get('unknown_supported_component_count', 0)}, "
                "rescue_candidates="
                f"{event.get('unknown_supported_candidate_count', 0)}, "
                "duplicate_gateways="
                f"{event.get('redundant_unknown_supported_component_count', 0)}, "
                "unknown_basins="
                f"{event.get('significant_unknown_basin_count', 0)}/"
                f"{event.get('frontier_unknown_basin_count', 0)}, "
                "border_unknown_basins="
                f"{event.get('border_connected_unknown_basin_count', 0)}, "
                "border_rescues_rejected="
                f"{event.get('border_connected_unknown_supported_candidate_count', 0)}, "
                f"discarded={event.get('discarded_frontier_pixels', 0)}px/"
                f"{event.get('discarded_component_count', 0)} components, "
                f"thresholds={event.get('minimum_component_cells', 0)}px-or-"
                f"{event.get('minimum_unknown_support_cells', 0)}unknown, "
                f"mission_exhausted={event.get('mission_exhausted', False)}"
            )

    if sector_frontier_outcomes:
        lines.extend(["", "Rover frontier outcome memory:"])
        for event in sector_frontier_outcomes:
            lines.append(
                "  generation "
                f"{event.get('generation')}: "
                "previous_generation="
                f"{event.get('previous_generation')}, "
                "occupied_gain="
                f"{event.get('confident_occupied_gain')}, "
                "evaluated="
                f"{event.get('evaluated_component_count', 0)}, "
                "productive="
                f"{event.get('productive_component_count', 0)}, "
                "unchanged="
                f"{event.get('locally_unchanged_component_count', 0)}, "
                "reported_zero_gain="
                f"{event.get('reported_zero_gain_component_count', 0)}, "
                "resolved="
                f"{event.get('resolved_component_count', 0)}, "
                "remembered="
                f"{event.get('remembered_component_count', 0)}, "
                "suppressed="
                f"{event.get('suppressed_frontier_pixels', 0)}px/"
                f"{event.get('suppressed_component_count', 0)} components, "
                "remaining="
                f"{event.get('remaining_frontier_pixels', 0)}px/"
                f"{event.get('remaining_component_count', 0)} components, "
                f"mission_exhausted={event.get('mission_exhausted', False)}"
            )
            for component in event.get("component_outcomes", ()):
                lines.append(
                    "    component "
                    f"{component.get('component_id')}: "
                    f"{component.get('disposition')}, "
                    "size="
                    f"{component.get('previous_size', 0)}->"
                    f"{component.get('current_size', 0)}, "
                    f"iou={component.get('overlap_iou', 0.0):.3f}, "
                    "local_known_gain="
                    f"{component.get('local_confident_cell_gain', 0)}, "
                    "local_occupied_gain="
                    f"{component.get('local_confident_occupied_gain', 0)}, "
                    "local_confidence_gain="
                    f"{component.get('local_confidence_gain', 0.0):.3f}, "
                    "reports="
                    f"{component.get('suppression_reasons', [])}"
                )

    if sector_workload_balances:
        lines.extend(["", "Rover sector workload balance:"])
        for event in sector_workload_balances:
            initial_effort = [
                round(float(value), 2)
                for value in event.get("initial_estimated_efforts", ())
            ]
            balanced_effort = [
                round(float(value), 2)
                for value in event.get("balanced_estimated_efforts", ())
            ]
            lines.append(
                "  generation "
                f"{event.get('generation')}: "
                f"initial={event.get('initial_frontier_workloads', [])} "
                f"balanced={event.get('balanced_frontier_workloads', [])} "
                "spread="
                f"{event.get('initial_workload_spread', 0)}->"
                f"{event.get('balanced_workload_spread', 0)} "
                f"estimated_effort={initial_effort}->{balanced_effort} "
                "effort_spread="
                f"{event.get('initial_estimated_effort_spread', 0.0):.2f}->"
                f"{event.get('balanced_estimated_effort_spread', 0.0):.2f} "
                f"moved_cells={event.get('moved_coarse_cell_count', 0)}"
            )

    if last_frame is not None:
        lines.extend(
            [
                "",
                (
                    "Last frame: "
                    "t="
                    f"{max(0.0, (_trace_event_time(last_frame) or trace_start_time) - trace_start_time):.2f}s, "
                    f"fps={last_frame.get('fps', 0):.1f}, "
                    f"dirty_maps={last_frame.get('dirty_maps', 0)}"
                ),
            ]
        )
        for state in last_frame.get("drone_states", []):
            lines.append(
                "  "
                f"d{state.get('id')}: pos={state.get('position')} "
                f"frontiers={state.get('frontiers')} "
                f"home={state.get('returning_home')} "
                f"done={state.get('done')} "
                f"activity={state.get('activity')} "
                f"target={state.get('activity_target')} "
                f"dfs_depth={state.get('dfs_depth')} "
                f"slam={state.get('slam_version')}"
            )
        for state in last_frame.get("rover_states", []):
            lines.append(
                "  "
                f"r{state.get('id')}: pos={state.get('position')} "
                f"target={state.get('target')} "
                f"status={state.get('status')} "
                f"path_remaining={state.get('path_remaining')} "
                f"slam={state.get('slam_version')}"
            )

    for drone_id in sorted(per_drone_counts):
        lines.extend(["", f"Drone {drone_id}:"])
        counts = per_drone_counts[drone_id]
        interesting = (
            "drone_random_direction_selected",
            "drone_global_frontiers_rebuilt",
            "drone_local_frontier_exhaustion_deferred",
            "drone_global_frontier_exhaustion_confirmed",
            "drone_global_frontier_path",
            "drone_global_frontier_region_suppressed",
            "drone_slam_frontiers_refreshed",
            "drone_random_step",
            "drone_border_path",
            "drone_frontier_reached",
            "drone_frontier_route_rejected",
            "drone_homing_path",
            "drone_sector_exhausted",
            "drone_sector_check_in_path",
            "drone_sector_waiting_for_team",
            "drone_sector_wait_completed",
            "drone_sector_ingress_recovery",
            "drone_sector_ingress_failed",
            "drone_sector_ingress_abandoned",
            "drone_astar_partial_segment",
            "drone_partial_frontier_route_cancelled",
            "drone_border_target_suppressed",
            "drone_recovery_reoriented",
            "drone_recovery_no_outgoing_heading",
            "drone_no_reachable_border",
            "drone_stagnation_window",
            "drone_stagnation_detected",
            "drone_stagnation_reoriented",
            "drone_stagnation_scan_started",
            "drone_stagnation_scan_completed",
            "drone_stagnation_scan_timed_out",
            "drone_stagnation_scan_exit_reoriented",
            "drone_stagnation_scan_no_safe_exit",
            "drone_stagnation_frontier_filter",
            "drone_stagnation_frontier_path",
            "drone_stagnation_arrival_reoriented",
            "drone_stagnation_unresolved",
            "drone_decision",
            "drone_post_rebuild_decision",
            "drone_policy_exhausted",
            "drone_frontier_path",
            "drone_frontier_direct_path_failed",
            "drone_frontier_direct_path_skipped",
            "drone_waypoint_route",
            "drone_waypoint_bridge",
            "drone_waypoint_segment_path",
            "drone_waypoint_segment_complete",
            "drone_frontier_targets_exhausted",
            "frontier_continuation_retained",
            "frontier_continuation_suppressed",
            "frontier_wall_tracking_advanced",
            "drone_policy_path_invalid",
            "drone_start_homing_after_exhaustion",
            "sensor_scan",
            "sensor_pose_static_skip",
        )
        for name in interesting:
            if counts[name]:
                lines.append(f"  {name}: {counts[name]}")
        sensor_timings = sensor_stage_timings[drone_id]
        total_timings = sensor_timings["sensor_elapsed_ms"]
        if total_timings:
            def average_timing(field: str) -> float:
                values = sensor_timings[field]
                return statistics.mean(values) if values else 0.0

            lines.append(
                "  sensor scan timings: "
                f"samples={len(total_timings)} "
                f"avg_total={statistics.mean(total_timings):.2f}ms "
                "avg_vision="
                f"{average_timing('vision_elapsed_ms'):.2f}ms "
                f"avg_slam={average_timing('slam_elapsed_ms'):.2f}ms "
                "avg_terrain="
                f"{average_timing('terrain_elapsed_ms'):.2f}ms"
            )
        if stagnation_scan_dispositions[drone_id]:
            outcomes = ", ".join(
                f"{disposition}={count}"
                for disposition, count in stagnation_scan_dispositions[
                    drone_id
                ].most_common()
            )
            lines.append(
                "  directed scan outcomes: "
                f"{outcomes}; "
                f"sensor_cells={stagnation_scan_sensor_cells[drone_id]} "
                "confidence_gain="
                f"{stagnation_scan_confidence_gain[drone_id]:.2f}"
            )
        resume_deltas = scan_exit_resume_deltas[drone_id]
        if resume_deltas:
            exact_resumes = scan_exit_exact_resumes[drone_id]
            lines.append(
                "  directed scan exits: "
                f"exact_resume={exact_resumes}/{len(resume_deltas)} "
                f"fallback={len(resume_deltas) - exact_resumes} "
                f"avg_delta={statistics.mean(resume_deltas):.1f}deg "
                f"max_delta={max(resume_deltas):.1f}deg"
            )
        if heading_selection_modes[drone_id]:
            modes = ", ".join(
                f"{mode}={count}"
                for mode, count in heading_selection_modes[
                    drone_id
                ].most_common()
            )
            lines.append(f"  heading selection modes: {modes}")
        coverage_terms = coverage_heading_terms[drone_id]
        coverage_totals = coverage_motion_totals[drone_id]
        if coverage_terms or coverage_totals:
            entries = coverage_totals["coverage_cell_entries"]
            revisits = coverage_totals["coverage_revisit_entries"]
            repeated_edges = coverage_totals[
                "coverage_repeated_edge_entries"
            ]
            known_cells, known_edges = coverage_known_counts.get(
                drone_id,
                (0, 0),
            )
            factors = [term[2] for term in coverage_terms]
            average_visit_pressure = (
                statistics.mean(term[0] for term in coverage_terms)
                if coverage_terms
                else 0.0
            )
            average_edge_pressure = (
                statistics.mean(term[1] for term in coverage_terms)
                if coverage_terms
                else 0.0
            )
            lines.append(
                "  coverage memory: "
                f"decisions={len(coverage_terms)} "
                "penalized="
                f"{sum(factor < 1.0 - 1e-9 for factor in factors)} "
                "avg_factor="
                f"{(statistics.mean(factors) if factors else 1.0):.3f} "
                "min_factor="
                f"{(min(factors) if factors else 1.0):.3f} "
                "avg_visit_pressure="
                f"{average_visit_pressure:.2f} "
                "avg_edge_pressure="
                f"{average_edge_pressure:.2f} "
                f"cell_entries={entries} "
                f"new={coverage_totals['coverage_new_cell_entries']} "
                f"revisits={revisits}/"
                f"{entries} ({(revisits / entries if entries else 0.0):.1%}) "
                f"repeated_edges={repeated_edges}/"
                f"{entries} ({(repeated_edges / entries if entries else 0.0):.1%}) "
                f"known={known_cells}c/{known_edges}e"
            )
        cluster_sizes = heading_cluster_sizes[drone_id]
        if cluster_sizes or heading_cluster_totals[drone_id]["observed"]:
            totals = heading_cluster_totals[drone_id]
            size_summary = (
                f"selected={len(cluster_sizes)} "
                f"avg_size={sum(cluster_sizes) / len(cluster_sizes):.1f} "
                f"max_size={max(cluster_sizes)}"
                if cluster_sizes
                else "selected=0"
            )
            lines.append(
                "  heading frontier clusters: "
                f"{size_summary}; observed={totals['observed']} "
                f"eligible={totals['eligible']} filtered={totals['filtered']} "
                f"wall_candidates={totals['wall_candidates']} "
                f"generic_candidates={totals['generic_candidates']}"
            )
            score_terms = heading_cluster_scores[drone_id]
            if score_terms:
                lines.append(
                    "  selected cluster score terms: "
                    f"avg_score={sum(v[0] for v in score_terms) / len(score_terms):.2f} "
                    f"avg_size_rank={sum(v[1] for v in score_terms) / len(score_terms):.2f} "
                    f"avg_proximity={sum(v[2] for v in score_terms) / len(score_terms):.2f}"
                )
        global_sizes = global_frontier_sizes[drone_id]
        if global_sizes:
            global_distances = global_frontier_distances[drone_id]
            lines.append(
                "  active global frontier guidance: "
                f"decisions={len(global_sizes)} "
                f"avg_size={sum(global_sizes) / len(global_sizes):.1f} "
                f"max_size={max(global_sizes)} "
                f"avg_distance={sum(global_distances) / len(global_distances):.1f}px"
            )
            ownership_terms = global_frontier_ownership[drone_id]
            if ownership_terms:
                term_count = len(ownership_terms)
                owned_ratio = (
                    sum(term[0] > 0.0 for term in ownership_terms)
                    / term_count
                )
                averages = tuple(
                    sum(term[index] for term in ownership_terms) / term_count
                    for index in range(5)
                )
                lines.append(
                    "  active global target ownership: "
                    f"owned={owned_ratio:.1%} "
                    f"avg_margin={averages[0]:.3f} "
                    f"avg_sector_alignment={averages[1]:.3f} "
                    f"avg_contribution={averages[2]:.3f} "
                    f"avg_requester_distance={averages[3]:.1f}px "
                    f"avg_nearest_peer_distance={averages[4]:.1f}px"
                )
        cache_times = global_frontier_cache_ms[drone_id]
        if cache_times:
            totals = global_frontier_cache_totals[drone_id]
            retention_samples = totals["retention_samples"]
            retained = (
                f"{totals['retained']}/{retention_samples}"
                if retention_samples
                else "N/A"
            )
            lines.append(
                "  global frontier cache: "
                f"rebuilds={len(cache_times)} "
                f"avg_ms={sum(cache_times) / len(cache_times):.2f} "
                f"max_ms={max(cache_times):.2f} "
                f"avg_regions={totals['regions'] / len(cache_times):.1f} "
                f"avg_eligible={totals['eligible'] / len(cache_times):.1f} "
                f"avg_filtered={totals['filtered'] / len(cache_times):.1f} "
                f"retained={retained} "
                f"forced={totals['forced']} "
                "avg_suppressed="
                f"{totals['suppressed'] / len(cache_times):.1f}"
            )
        if astar_path_statuses[drone_id]:
            statuses = ", ".join(
                f"{status}={count}"
                for status, count in astar_path_statuses[
                    drone_id
                ].most_common()
            )
            lines.append(f"  A* path statuses: {statuses}")
        route_circuities = frontier_route_circuities[drone_id]
        if route_circuities:
            lines.append(
                "  frontier route circuity: "
                f"samples={len(route_circuities)} "
                f"avg={statistics.mean(route_circuities):.2f}x "
                f"max={max(route_circuities):.2f}x "
                f"over_2x={sum(value > 2.0 for value in route_circuities)} "
                f"over_4x={sum(value > 4.0 for value in route_circuities)}"
            )
        if partial_segment_outcomes[drone_id]:
            outcomes = ", ".join(
                f"{outcome}={count}"
                for outcome, count in partial_segment_outcomes[
                    drone_id
                ].most_common()
            )
            lines.append(f"  A* partial segments: {outcomes}")
        waits = sector_wait_durations[drone_id]
        standby = sector_standby_durations[drone_id]
        if standby:
            lines.append(
                f"  sector standby: count={len(standby)} total={sum(standby):.2f}s "
                f"max={max(standby):.2f}s"
            )
        if waits:
            lines.append(
                "  sector barrier waits: "
                f"count={len(waits)} total={sum(waits):.2f}s "
                f"avg={statistics.mean(waits):.2f}s "
                f"max={max(waits):.2f}s"
            )
        if waypoint_route_statuses[drone_id]:
            statuses = ", ".join(
                f"{status}={count}"
                for status, count in waypoint_route_statuses[drone_id].most_common()
            )
            lines.append(f"  waypoint route statuses: {statuses}")
        if waypoint_gateway_statuses[drone_id]:
            statuses = ", ".join(
                f"{status}={count}"
                for status, count in waypoint_gateway_statuses[
                    drone_id
                ].most_common()
            )
            lines.append(f"  waypoint gateway statuses: {statuses}")
        if waypoint_bridge_statuses[drone_id]:
            statuses = ", ".join(
                f"{status}={count}"
                for status, count in waypoint_bridge_statuses[
                    drone_id
                ].most_common()
            )
            lines.append(f"  waypoint bridge statuses: {statuses}")
        route_count = sum(waypoint_route_statuses[drone_id].values())
        if route_count:
            graph_nodes, graph_edges = waypoint_graph_size.get(
                drone_id,
                (0, 0),
            )
            lines.append(
                "  waypoint route timing: "
                f"avg={waypoint_route_time_total[drone_id] / route_count:.2f}ms "
                f"max={waypoint_route_time_max[drone_id]:.2f}ms "
                f"graph={graph_nodes}n/{graph_edges}e"
            )
        if waypoint_segment_sources[drone_id]:
            sources = ", ".join(
                f"{source}={count}"
                for source, count in waypoint_segment_sources[drone_id].most_common()
            )
            lines.append(f"  waypoint segment paths: {sources}")

        decision = last_decision.get(drone_id)
        if decision is not None:
            summary = decision.get("decision", {})
            lines.append(
                "  last decision: "
                f"{summary.get('kind')} "
                f"mode={summary.get('exploration_mode')} "
                f"target={summary.get('target')} "
                f"dir={summary.get('direction')} "
                f"path={summary.get('planned_path_len')} "
                f"frontiers={summary.get('frontier_count')} "
                f"primitive={summary.get('local_primitive')}"
            )

        lines.append("  last events:")
        for event in last_by_drone[drone_id]:
            event_time = _trace_event_time(event)
            lines.append(
                "    "
                f"{max(0.0, (event_time or trace_start_time) - trace_start_time):7.2f}s "
                f"{event.get('event')}"
            )

    return lines


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trace",
        nargs="?",
        help="Trace JSONL path. Defaults to newest logs/mission_trace_*.jsonl.",
    )
    parser.add_argument(
        "--reversal-window-start",
        type=float,
        default=None,
        help="Measure A-B-A arrivals after this trace-relative time in seconds.",
    )
    args = parser.parse_args()

    path = Path(args.trace) if args.trace else latest_trace(Path("logs"))
    print(
        "\n".join(
            summarize(
                load_events(path),
                reversal_window_start_s=args.reversal_window_start,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
