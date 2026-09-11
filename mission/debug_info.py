"""Control-panel debug information for a running mission."""

from typing import Iterable, List, Optional

from agents.drone_runtime_state import DroneSnapshot
from contracts import MissionDebugDependencies


class MissionDebugInfo:
    """Build small runtime status lines for the control center."""

    def __init__(self, dependencies: MissionDebugDependencies) -> None:
        """Store callbacks used to build debug lines on demand."""
        self.dependencies = dependencies

    def build_debug_lines(
        self,
        drone_snapshots: Optional[Iterable[DroneSnapshot]] = None,
    ) -> List[str]:
        """Build live agent-state lines for the debug tab."""
        dependencies = self.dependencies
        lines: List[str] = []
        for drone in dependencies.get_drones():
            activity_method = getattr(
                getattr(drone, "movement_controller", None),
                "activity_snapshot",
                None,
            )
            if not callable(activity_method):
                continue
            activity = activity_method()
            lines.append(
                f"D{int(drone.id)} {activity.state}: {activity.detail}"
            )
        for rover in dependencies.get_rovers():
            snapshot_method = getattr(rover, "snapshot", None)
            snapshot = snapshot_method() if callable(snapshot_method) else rover
            target = getattr(snapshot, "target", None)
            target_text = (
                "none" if target is None else f"{target[0]},{target[1]}"
            )
            if snapshot.status == "Rendezvous":
                reason = "holding for drone rendezvous"
            elif snapshot.status == "Ready" and target is None:
                reason = "no reachable component target"
            elif snapshot.status == "Staging":
                reason = "staged near frontier"
            elif snapshot.status == "Advancing":
                reason = "following component target"
            else:
                reason = "updating navigation"
            lines.append(
                f"R{int(rover.id)} {snapshot.status}: "
                f"pos {snapshot.position[0]},{snapshot.position[1]} "
                f"target {target_text}; {reason}"
            )
        return lines

    def build_system_lines(
        self,
        drone_snapshots: Optional[Iterable[DroneSnapshot]] = None,
    ) -> List[str]:
        """Build mapping, tracing, and timing lines for the system tab."""
        dependencies = self.dependencies
        if drone_snapshots is None:
            drone_snapshots = (
                drone.snapshot() for drone in dependencies.get_drones()
            )
        snapshots = tuple(drone_snapshots)
        now = dependencies.simulation_time()
        frontier_count = sum(
            len(snapshot.frontiers) for snapshot in snapshots
        )
        selected_id = dependencies.presentation.selected_drone_heatmap_id
        selected_rover_id = getattr(
            dependencies.presentation,
            "selected_rover_heatmap_id",
            None,
        )
        if selected_rover_id is not None:
            selected_label = f"rover {selected_rover_id}"
        elif selected_id is not None:
            selected_label = f"drone {selected_id}"
        else:
            selected_label = "all/none selected"

        cooldown_remaining = 0.0
        if snapshots:
            cooldown_remaining = min(
                max(
                    0.0,
                    snapshot.frontier_rebuild_cooldown
                    - (now - snapshot.last_frontier_rebuild),
                )
                for snapshot in snapshots
            )

        lines = [
            f"SLAM view: {selected_label}",
            f"Dirty maps: {dependencies.dirty_map_count()}",
            f"Frontiers: {frontier_count}",
            f"Frontier cooldown: {cooldown_remaining:.2f}s",
        ]
        trace = dependencies.runtime_trace
        if trace is not None and getattr(trace, "enabled", False):
            lines.append(f"Trace: {trace.path}")

        profiler = dependencies.frame_profiler
        if profiler is not None:
            timing = profiler.snapshot()
            if timing.sample_count > 0:
                stages = timing.stages_ms
                lines.extend(
                    [
                        (
                            f"Frame rate: {timing.fps:.1f} FPS "
                            f"({timing.frame_ms:.1f} ms)"
                        ),
                        (
                            f"Frame work/wait: {timing.work_ms:.1f} / "
                            f"{timing.wait_ms:.1f} ms"
                        ),
                        (
                            "Stages ms: "
                            f"share {stages.get('sharing', 0.0):.1f}, "
                            f"sense {stages.get('sensors', 0.0):.1f}, "
                            f"render {stages.get('render', 0.0):.1f}"
                        ),
                    ]
                )

        return lines

    def build_lines(
        self,
        drone_snapshots: Optional[Iterable[DroneSnapshot]] = None,
    ) -> List[str]:
        """Build the legacy combined stream for compatibility."""
        snapshots = (
            None if drone_snapshots is None else tuple(drone_snapshots)
        )
        return (
            self.build_system_lines(snapshots)
            + self.build_debug_lines(snapshots)
        )
