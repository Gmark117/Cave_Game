"""Summarize Cave Game runtime JSONL traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
from typing import Any, Iterable


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


def summarize(events: Iterable[dict[str, Any]]) -> list[str]:
    """Build a compact text summary of drone decision and path events."""
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
    waypoint_graph_size: dict[int, tuple[int, int]] = {}
    last_frame: dict[str, Any] | None = None
    trace_path = "-"

    for event in events:
        event_name = str(event.get("event", "unknown"))
        event_counts[event_name] += 1
        if event_name == "trace_started":
            trace_path = str(event.get("path", "-"))
        if event_name == "frame_summary":
            last_frame = event

        drone_id = event.get("drone_id")
        if drone_id is None:
            continue
        drone_id = int(drone_id)
        per_drone_counts[drone_id][event_name] += 1
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

    if last_frame is not None:
        lines.extend(
            [
                "",
                (
                    "Last frame: "
                    f"t={last_frame.get('sim_time', 0):.2f}s, "
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
                f"slam={state.get('slam_version')}"
            )

    for drone_id in sorted(per_drone_counts):
        lines.extend(["", f"Drone {drone_id}:"])
        counts = per_drone_counts[drone_id]
        interesting = (
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
            "drone_policy_path_invalid",
            "drone_start_homing_after_exhaustion",
            "sensor_scan",
            "sensor_pose_static_skip",
        )
        for name in interesting:
            if counts[name]:
                lines.append(f"  {name}: {counts[name]}")
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
            mcts = decision.get("mcts") or {}
            lines.append(
                "  last decision: "
                f"{summary.get('kind')} "
                f"target={summary.get('target')} "
                f"dir={summary.get('direction')} "
                f"path={summary.get('planned_path_len')} "
                f"frontiers={summary.get('frontier_count')} "
                f"mcts={mcts.get('selected_kind')} "
                f"reward={mcts.get('selected_reward')}"
            )

        lines.append("  last events:")
        for event in last_by_drone[drone_id]:
            lines.append(
                "    "
                f"{event.get('sim_time', 0):7.2f}s "
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
    args = parser.parse_args()

    path = Path(args.trace) if args.trace else latest_trace(Path("logs"))
    print("\n".join(summarize(load_events(path))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
