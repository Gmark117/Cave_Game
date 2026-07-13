"""Control-panel debug information for a running mission."""

from typing import Iterable, List, Optional

from agents.drone_runtime_state import DroneSnapshot
from contracts import MissionDebugDependencies


class MissionDebugInfo:
    """Build small runtime status lines for the control center."""

    def __init__(self, dependencies: MissionDebugDependencies) -> None:
        """Store callbacks used to build debug lines on demand."""
        self.dependencies = dependencies

    def build_lines(
        self,
        drone_snapshots: Optional[Iterable[DroneSnapshot]] = None,
    ) -> List[str]:
        """Build runtime debug lines for the control panel."""
        dependencies = self.dependencies
        if drone_snapshots is None:
            drone_snapshots = (
                drone.snapshot() for drone in dependencies.get_drones()
            )
        snapshots = tuple(drone_snapshots)
        now = dependencies.simulation_time()
        dirty_maps = dependencies.dirty_map_count()
        frontier_count = sum(
            len(snapshot.frontiers) for snapshot in snapshots
        )
        selected_id = dependencies.presentation.selected_drone_heatmap_id
        selected_label = (
            "all/none selected" if selected_id is None else f"drone {selected_id}"
        )

        cooldown_remaining = 0.0
        if snapshots:
            cooldown_remaining = min(
                max(
                    0.0,
                    snapshot.frontier_rebuild_cooldown
                    - (
                        now
                        - snapshot.last_frontier_rebuild
                    ),
                )
                for snapshot in snapshots
            )

        lines = [
            f"SLAM view: {selected_label}",
            f"Dirty maps: {dirty_maps}",
            f"Frontiers: {frontier_count}",
            f"Frontier cooldown: {cooldown_remaining:.2f}s",
        ]
        mcts_line = self._mcts_debug_line(selected_id)
        if mcts_line is not None:
            lines.append(mcts_line)
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

    def _mcts_debug_line(self, selected_id: int | None) -> str | None:
        """Return a concise line for the selected drone's MCTS search."""
        try:
            drones = tuple(self.dependencies.get_drones())
        except (TypeError, AttributeError):
            return None
        if not drones:
            return None

        indexed_drones = list(enumerate(drones))
        if selected_id is not None and 0 <= selected_id < len(drones):
            indexed_drones = [(selected_id, drones[selected_id])]

        for index, drone in indexed_drones:
            policy = getattr(drone, "exploration_policy", None)
            diagnostics = getattr(policy, "last_search_diagnostics", None)
            if diagnostics is None:
                continue

            config = getattr(policy, "config", None)
            max_iterations = getattr(
                config,
                "iterations",
                diagnostics.iterations,
            )
            selected_kind = diagnostics.selected_kind or "none"
            direction_text = (
                "-"
                if diagnostics.selected_direction is None
                else f"{diagnostics.selected_direction}deg"
            )
            target = diagnostics.selected_target
            target_text = (
                "-"
                if target is None
                else f"{target[0]},{target[1]}"
            )
            return (
                f"MCTS d{index}: {selected_kind} {direction_text} "
                f"-> {target_text}, {diagnostics.iterations}/"
                f"{max_iterations} it, {diagnostics.generated_nodes} nodes, "
                f"{diagnostics.elapsed_ms:.0f} ms"
            )

        return None
