"""Summarize Cave Game runtime JSONL traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
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
            "drone_frontier_targets_exhausted",
            "drone_policy_path_invalid",
            "drone_start_homing_after_exhaustion",
            "sensor_scan",
            "sensor_pose_static_skip",
        )
        for name in interesting:
            if counts[name]:
                lines.append(f"  {name}: {counts[name]}")

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
