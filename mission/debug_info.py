"""Control-panel debug information for a running mission."""

import time
from typing import Any, List


class MissionDebugInfo:
    """Build small runtime status lines for the control center."""

    def __init__(self, control: Any) -> None:
        self.control = control

    def build_lines(self) -> List[str]:
        """Build runtime debug lines for the control panel."""
        control = self.control
        now = time.perf_counter()
        dirty_maps = sum(
            1
            for drone in control.drones
            if getattr(drone, "slam_map", None) is not None
            and drone.slam_map.dirty
        )
        frontier_count = sum(
            len(getattr(drone, "border", ())) for drone in control.drones
        )
        selected_id = control.presentation.selected_drone_heatmap_id
        selected_label = (
            "all/none selected" if selected_id is None else f"drone {selected_id}"
        )

        cooldown_remaining = 0.0
        if control.drones:
            cooldown_remaining = min(
                max(
                    0.0,
                    drone.frontier_rebuild_cooldown
                    - (now - drone.last_frontier_rebuild),
                )
                for drone in control.drones
            )

        lines = [
            f"SLAM view: {selected_label}",
            f"Dirty maps: {dirty_maps}",
            f"Frontiers: {frontier_count}",
            f"Frontier cooldown: {cooldown_remaining:.2f}s",
        ]

        profiler = getattr(control, "frame_profiler", None)
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
