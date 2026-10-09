"""Summarize Cave Game runtime JSONL traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import sys
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

# Support both `python tools/analyze_runtime_trace.py` and package imports.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep the analyzer's existing metric imports available to offline callers.
from tools.trace_metrics import (
    ABAReversalMetrics as ABAReversalMetrics,
    ALLOWED_GOAL_CHANGE_REASONS as ALLOWED_GOAL_CHANGE_REASONS,
    AcceptanceMetrics as AcceptanceMetrics,
    DECISION_EVENTS as DECISION_EVENTS,
    DroneABAReversalMetrics as DroneABAReversalMetrics,
    DroneInformationEfficiencyMetrics as DroneInformationEfficiencyMetrics,
    DroneTargetRetentionMetrics as DroneTargetRetentionMetrics,
    FORBIDDEN_NODE_ROLES as FORBIDDEN_NODE_ROLES,
    FrontierFallbackMetrics as FrontierFallbackMetrics,
    InformationEfficiencyMetrics as InformationEfficiencyMetrics,
    LEGACY_MCTS_BUDGET_MS as LEGACY_MCTS_BUDGET_MS,
    LOCAL_MCTS_EVENTS as LOCAL_MCTS_EVENTS,
    MCTSTimingMetrics as MCTSTimingMetrics,
    RouteCacheMetrics as RouteCacheMetrics,
    RuntimeTraceMetrics as RuntimeTraceMetrics,
    TargetRetentionMetrics as TargetRetentionMetrics,
    TraceSchemaValidationMetrics as TraceSchemaValidationMetrics,
    WaypointDensityMetrics as WaypointDensityMetrics,
    _boolean,
    _finite_float,
    _format_optional,
    _integer,
    _nearest_rank_percentile,
    _nested_mapping,
    analyze_aba_reversals as analyze_aba_reversals,
    analyze_frontier_fallbacks as analyze_frontier_fallbacks,
    analyze_information_efficiency as analyze_information_efficiency,
    analyze_mcts_timing as analyze_mcts_timing,
    analyze_route_cache as analyze_route_cache,
    analyze_target_retention as analyze_target_retention,
    analyze_trace as analyze_trace,
    analyze_waypoint_density as analyze_waypoint_density,
    format_characterization as format_characterization,
)


def load_events(path: Path) -> Iterable[dict[str, Any]]:
    """Yield parsed events, normalizing legacy batch telemetry names."""
    with path.open("r", encoding="utf-8") as trace_file:
        for line in trace_file:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_name = event.get("event")
            if isinstance(event_name, str) and "_endgame_batch_" in event_name:
                event["event"] = event_name.replace(
                    "_endgame_batch_",
                    "_focused_frontier_batch_",
                )
            if (
                "focused_frontier_batch_mode" not in event
                and "endgame_batch_mode" in event
            ):
                event["focused_frontier_batch_mode"] = event[
                    "endgame_batch_mode"
                ]
            yield event


def latest_trace(log_dir: Path) -> Path:
    """Return the newest mission trace in a log directory."""
    traces = sorted(log_dir.glob("mission_trace_*.jsonl"))
    if not traces:
        raise FileNotFoundError(f"No mission traces found in {log_dir}")
    return traces[-1]


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
        "rover_focused_frontier_batch_evaluated",
        "rover_focused_frontier_batch_issued",
        "drone_focused_frontier_batch_member_selected",
        "drone_focused_frontier_batch_member_finished",
        "drone_focused_frontier_batch_frontier_evaluated",
        "drone_focused_frontier_batch_frontier_revisited",
        "drone_focused_frontier_batch_bound_reached",
        "rover_focused_frontier_batch_report_accepted",
        "rover_focused_frontier_batch_report_rejected",
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
    batch_evaluations = [
        event for event in relevant
        if event.get("event") == "rover_focused_frontier_batch_evaluated"
    ]
    batch_issues = [
        event for event in relevant
        if event.get("event") == "rover_focused_frontier_batch_issued"
    ]
    batch_member_finishes = [
        event for event in relevant
        if event.get("event") == "drone_focused_frontier_batch_member_finished"
    ]
    batch_frontier_evaluations = [
        event for event in relevant
        if event.get("event") == "drone_focused_frontier_batch_frontier_evaluated"
    ]
    batch_frontier_revisits = [
        event for event in relevant
        if event.get("event") == "drone_focused_frontier_batch_frontier_revisited"
    ]
    batch_bounds = Counter(
        str(event.get("bound", "unknown"))
        for event in relevant
        if event.get("event") == "drone_focused_frontier_batch_bound_reached"
    )
    batch_reports_accepted = [
        event for event in relevant
        if event.get("event") == "rover_focused_frontier_batch_report_accepted"
    ]
    batch_reports_rejected = [
        event for event in relevant
        if event.get("event") == "rover_focused_frontier_batch_report_rejected"
    ]
    batch_report_replays = sum(
        _boolean(event.get("replayed")) is True
        for event in batch_reports_accepted
    )
    batch_statuses: Counter[str] = Counter()
    batch_dispositions: Counter[str] = Counter()
    for event in batch_reports_accepted:
        if _boolean(event.get("replayed")) is True:
            continue
        for item in event.get("member_statuses", ()) or ():
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                batch_statuses[str(item[1])] += 1
        batch_dispositions.update(
            str(value)
            for value in event.get("member_dispositions", ()) or ()
        )
    batch_evaluation_reasons = Counter(
        str(event.get("reason", "unknown")) for event in batch_evaluations
    )
    batch_observe_opportunities = sum(
        str(event.get("mode", "")).casefold() == "observe"
        and _boolean(event.get("accepted")) is True
        for event in batch_evaluations
    )
    batch_frontier_decisions = Counter(
        "accepted" if _boolean(event.get("accepted")) is True
        else str(event.get("reason", "rejected"))
        for event in batch_frontier_evaluations
    )
    issued_lease_ids = {
        lease_id for event in batch_issues
        if (lease_id := _integer(event.get("lease_id"))) is not None
    }
    released_lease_ids = {
        lease_id for event in batch_reports_accepted
        if _boolean(event.get("replayed")) is not True
        and (lease_id := _integer(event.get("lease_id"))) is not None
    }
    batch_directive_ids = {
        directive_id for event in batch_issues
        if (directive_id := _integer(event.get("directive_id"))) is not None
    }
    batch_patch_c_overlap = sum(
        event.get("event") == "drone_incidental_scan_started"
        and _integer(event.get("directive_id")) in batch_directive_ids
        for event in events
    )
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
    if batch_evaluations or batch_issues or batch_reports_accepted:
        batch_sizes = tuple(
            len(event.get("member_task_ids", ()) or ())
            for event in batch_issues
        )

        def batch_sum(field: str, source: Iterable[Mapping[str, Any]]) -> float:
            return sum(
                _finite_float(event.get(field)) or 0.0
                for event in source
            )

        nonreplayed_reports = tuple(
            event for event in batch_reports_accepted
            if _boolean(event.get("replayed")) is not True
        )
        provisional_count = sum(
            max(0, _integer(event.get("provisional_observation_count")) or 0)
            for event in nonreplayed_reports
        )
        provisional_gain = sum(
            max(0, _integer(event.get("provisional_newly_known_cells")) or 0)
            for event in nonreplayed_reports
        )
        lines.append(
            "  focused frontier batching: "
            f"evaluated={len(batch_evaluations)} "
            f"observe_opportunities={batch_observe_opportunities} "
            f"issued={len(batch_issues)} sizes={batch_sizes} "
            f"separate={batch_sum('estimated_separate_cost', batch_issues):.1f}px "
            f"combined={batch_sum('estimated_combined_cost', batch_issues):.1f}px "
            f"avoided={batch_sum('estimated_avoided_round_trip', batch_issues):.1f}px "
            f"detour={batch_sum('estimated_detour_cost', batch_issues):.1f}px "
            f"member_outbound={batch_sum('outbound_distance', batch_member_finishes):.1f}px "
            f"member_service={batch_sum('service_distance', batch_member_finishes):.1f}px "
            f"return={batch_sum('return_distance', nonreplayed_reports):.1f}px "
            f"service={batch_sum('service_seconds', batch_member_finishes):.2f}s "
            f"provisional={provisional_count}/{provisional_gain}cells "
            f"revisits={len(batch_frontier_revisits)}/"
            f"{batch_sum('actual_revisit_route_distance', batch_frontier_revisits):.1f}px "
            f"reports={len(nonreplayed_reports)} "
            f"rejected={len(batch_reports_rejected)} "
            f"replays={batch_report_replays} "
            f"member_statuses={dict(sorted(batch_statuses.items()))} "
            f"member_dispositions={dict(sorted(batch_dispositions.items()))} "
            f"bounds={dict(sorted(batch_bounds.items()))} "
            "frontiers="
            f"{dict(sorted(batch_frontier_decisions.items()))} "
            "evaluation_reasons="
            f"{dict(sorted(batch_evaluation_reasons.items()))} "
            "claims_open="
            f"{max(0, sum(batch_sizes) - sum(batch_statuses.values()))} "
            f"leases_open={len(issued_lease_ids - released_lease_ids)} "
            f"patch_c_overlap={batch_patch_c_overlap}"
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


def _highway_summary_lines(
    events: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Summarize graph work, drone use, and frontier-batch planning cost."""
    builds = tuple(
        event for event in events
        if event.get("event") == "rover_highway_build_completed"
    )
    snapshots = tuple(
        event for event in events
        if event.get("event") == "drone_highway_snapshot_received"
    )
    routes = tuple(
        event for event in events
        if event.get("event") == "drone_highway_route_evaluated"
    )
    comparisons = tuple(event for event in events
                        if event.get("event") == "drone_return_route_compared")
    local_planning = tuple(event for event in events
                           if event.get("event") == "drone_local_route_planning_completed")
    path_requests = tuple(
        event for event in events
        if event.get("event") == "drone_path_request_completed"
    )
    planning = tuple(
        event for event in events
        if event.get("event") == "rover_focused_frontier_batch_planning_completed"
    )
    if not (builds or snapshots or routes or comparisons or path_requests or planning or local_planning):
        return []

    def timings(source: Iterable[Mapping[str, Any]]) -> list[float]:
        return [
            value for event in source
            if (value := _finite_float(event.get("elapsed_ms"))) is not None
            and value >= 0.0
        ]

    lines = ["", "Highway routing:"]
    if builds:
        elapsed = timings(builds)
        latest = builds[-1]
        lines.append(
            "  graph builds: "
            f"attempts={len(builds)} "
            f"statuses={dict(sorted(Counter(str(event.get('status', 'unknown')) for event in builds).items()))} "
            f"total={sum(elapsed):.2f}ms "
            f"mean={_format_optional(statistics.mean(elapsed) if elapsed else None, 'ms')} "
            f"p95={_format_optional(_nearest_rank_percentile(elapsed, 0.95), 'ms')} "
            f"max={_format_optional(max(elapsed, default=None), 'ms')} "
            f"published_version={latest.get('published_version')} "
            f"areas={latest.get('area_count', 0)} "
            f"edges={latest.get('edge_count', 0)} "
            "retained_previous="
            f"{sum(_boolean(event.get('retained_previous')) is True for event in builds)}"
        )
        if latest.get("topology") == "corridor_backbone":
            lines.append(
                "  corridor backbone: "
                f"nodes={latest.get('area_count', 0)} "
                f"components={latest.get('component_count', 0)} "
                f"pruned={latest.get('pruned_branches', 0)} "
                f"capillaries={latest.get('capillary_branches', 0)} "
                f"access={latest.get('measured_access_distance')}/"
                f"{latest.get('maximum_access_distance')}px"
            )
        background = [event for event in builds if event.get("build_kind") in {"regional", "full"}]
        if background:
            completed = [event for event in background if event.get("status") == "complete"]
            lines.append(
                "  background updates: "
                f"jobs={len(background)} kinds={dict(sorted(Counter(str(event.get('build_kind')) for event in background).items()))} "
                f"superseded={sum(event.get('status') == 'superseded' for event in background)} "
                f"rebuilt_regions={sum(int(event.get('rebuilt_regions', 0) or 0) for event in completed)} "
                f"reused_regions={sum(int(event.get('reused_regions', 0) or 0) for event in completed)}"
            )
    if snapshots:
        lines.append(
            "  physical publications: "
            f"received={len(snapshots)} "
            f"changed={sum(_boolean(event.get('changed')) is True for event in snapshots)} "
            f"drones={len({_integer(event.get('drone_id')) for event in snapshots})}"
        )
    if routes:
        elapsed = timings(routes)
        finite_costs = [
            value for event in routes
            if (value := _finite_float(event.get("route_distance"))) is not None
            and value >= 0.0
        ]
        finite_circuity = [
            value for event in routes
            if (value := _finite_float(event.get("route_circuity"))) is not None
            and value >= 0.0
        ]
        lines.append(
            "  drone return routes: "
            f"evaluated={len(routes)} "
            f"eligible={sum(_boolean(event.get('eligible')) is True for event in routes)} "
            f"selected={sum(_boolean(event.get('selected')) is True for event in routes)} "
            f"statuses={dict(sorted(Counter(str(event.get('status', 'unknown')) for event in routes).items()))} "
            f"fallbacks={dict(sorted(Counter(str(event.get('fallback_reason')) for event in routes if event.get('fallback_reason') is not None).items()))} "
            f"query_total={sum(elapsed):.2f}ms "
            f"query_max={_format_optional(max(elapsed, default=None), 'ms')} "
            f"route_mean={_format_optional(statistics.mean(finite_costs) if finite_costs else None, 'px')} "
            f"circuity_mean={_format_optional(statistics.mean(finite_circuity) if finite_circuity else None)}"
        )
    if comparisons:
        elapsed = timings(comparisons)
        savings = sum(
            max(0, (_finite_float(event.get("highway_distance")) or 0) -
                (_finite_float(event.get("local_distance")) or 0))
            for event in comparisons if event.get("selected_source") == "local"
        )
        lines.append(
            "  local return comparisons: "
            f"count={len(comparisons)} "
            f"selected={dict(sorted(Counter(str(event.get('selected_source')) for event in comparisons).items()))} "
            f"statuses={dict(sorted(Counter(str(event.get('comparison_status')) for event in comparisons).items()))} "
            f"planned_savings={savings:.2f}px "
            f"total={sum(elapsed):.2f}ms max={_format_optional(max(elapsed, default=None), 'ms')}"
        )
    promotions = [event for event in events if event.get("event") == "drone_rendezvous_target_promoted"]
    retargets = [event for event in events if event.get("event") == "drone_component_rendezvous_retargeted"
                and event.get("reason") == "physical_movement_evidence"]
    if promotions or retargets:
        lines.append(
            "  contact-carried return updates: "
            f"promotions={len(promotions)} "
            f"evidence={dict(sorted(Counter(str(event.get('evidence')) for event in promotions).items()))} "
            f"interrupted_routes={len(retargets)}"
        )
    if local_planning:
        elapsed = timings(local_planning)
        lines.append(
            "  drone-local route planning: "
            f"passes={len(local_planning)} "
            f"statuses={dict(sorted(Counter(str(event.get('status', 'unknown')) for event in local_planning).items()))} "
            f"max={_format_optional(max(elapsed, default=None), 'ms')} "
            f"route_queries={sum(int(event.get('route_queries', 0) or 0) for event in local_planning)} "
            f"cache_hits={sum(int(event.get('route_cache_hits', 0) or 0) for event in local_planning)}"
        )
    if path_requests:
        elapsed = timings(path_requests)
        lines.append(
            "  A* requests: "
            f"count={len(path_requests)} "
            f"statuses={dict(sorted(Counter(str(event.get('status', 'unknown')) for event in path_requests).items()))} "
            f"total={sum(elapsed):.2f}ms "
            f"mean={_format_optional(statistics.mean(elapsed) if elapsed else None, 'ms')} "
            f"p95={_format_optional(_nearest_rank_percentile(elapsed, 0.95), 'ms')} "
            f"max={_format_optional(max(elapsed, default=None), 'ms')}"
        )
    if planning:
        elapsed = timings(planning)
        lines.append(
            "  focused frontier batch planning: "
            f"passes={len(planning)} "
            f"statuses={dict(sorted(Counter(str(event.get('status', 'unknown')) for event in planning).items()))} "
            f"total={sum(elapsed):.2f}ms "
            f"mean={_format_optional(statistics.mean(elapsed) if elapsed else None, 'ms')} "
            f"max={_format_optional(max(elapsed, default=None), 'ms')} "
            f"route_queries={sum(max(0, _integer(event.get('route_queries')) or 0) for event in planning)} "
            f"cache_hits={sum(max(0, _integer(event.get('route_cache_hits')) or 0) for event in planning)} "
            f"candidates={sum(max(0, _integer(event.get('candidate_count')) or 0) for event in planning)} "
            f"batches={sum(max(0, _integer(event.get('planned_batch_count')) or 0) for event in planning)}"
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
    incidental_outcomes: dict[int, Counter[str]] = defaultdict(Counter)
    incidental_route_resumes: dict[int, Counter[str]] = defaultdict(Counter)
    incidental_wait_seconds: dict[int, float] = defaultdict(float)
    incidental_rotation_degrees: dict[int, float] = defaultdict(float)
    incidental_newly_known: dict[int, int] = defaultdict(int)
    incidental_pocket_cells_closed: dict[int, int] = defaultdict(int)
    incidental_predicted_support: dict[int, int] = defaultdict(int)
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
        if event_name == "drone_incidental_scan_finished":
            incidental_outcomes[drone_id][
                str(event.get("outcome", "unknown"))
            ] += 1
            incidental_wait_seconds[drone_id] += float(
                event.get("wait_seconds", 0.0) or 0.0
            )
            incidental_rotation_degrees[drone_id] += float(
                event.get("requested_rotation", 0.0) or 0.0
            )
            incidental_newly_known[drone_id] += int(
                event.get("newly_known_cells", 0) or 0
            )
            incidental_pocket_cells_closed[drone_id] += int(
                event.get("original_pocket_cells_closed", 0) or 0
            )
        if event_name == "drone_incidental_scan_candidate":
            incidental_predicted_support[drone_id] += int(
                event.get("pocket_cells", 0) or 0
            )
        if event_name == "drone_incidental_route_resumed":
            incidental_route_resumes[drone_id][
                str(event.get("disposition", "unknown"))
            ] += 1
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
    lines.extend(_highway_summary_lines(materialized))
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
            "drone_incidental_scan_sampled",
            "drone_incidental_scan_candidate",
            "drone_incidental_scan_started",
            "drone_incidental_scan_finished",
            "drone_incidental_route_resumed",
            "drone_incidental_pocket_revisited",
        )
        for name in interesting:
            if counts[name]:
                lines.append(f"  {name}: {counts[name]}")
        if (
            incidental_outcomes[drone_id]
            or counts["drone_incidental_scan_candidate"]
        ):
            outcomes = incidental_outcomes[drone_id]
            resumes = incidental_route_resumes[drone_id]
            lines.append(
                "  incidental_scan_summary: "
                f"completed={outcomes['completed']} "
                f"timed_out={outcomes['timed_out']} "
                f"cancelled={outcomes['cancelled']} "
                f"wait={incidental_wait_seconds[drone_id]:.3f}s "
                f"rotation={incidental_rotation_degrees[drone_id]:.1f}deg "
                f"newly_known={incidental_newly_known[drone_id]} "
                f"predicted_support={incidental_predicted_support[drone_id]} "
                "pocket_cells_closed="
                f"{incidental_pocket_cells_closed[drone_id]} "
                f"retained={resumes['retained']} "
                f"replanned={resumes['replanned']}"
            )
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
