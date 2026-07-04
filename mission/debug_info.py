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
