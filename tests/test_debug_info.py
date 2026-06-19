import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mission.frame_timing import FrameProfiler
from mission.debug_info import MissionDebugInfo


class MissionDebugInfoTests(unittest.TestCase):
    def test_build_lines_summarizes_current_mapping_state(self) -> None:
        drones = [
            SimpleNamespace(
                slam_map=SimpleNamespace(dirty=True),
                border=[(1, 1), (2, 2)],
                frontier_rebuild_cooldown=1.5,
                last_frontier_rebuild=9.5,
            ),
            SimpleNamespace(
                slam_map=SimpleNamespace(dirty=False),
                border=[(3, 3)],
                frontier_rebuild_cooldown=2.0,
                last_frontier_rebuild=9.0,
            ),
        ]
        control = SimpleNamespace(
            drones=drones,
            presentation=SimpleNamespace(selected_drone_heatmap_id=1),
        )

        with patch("mission.debug_info.time.perf_counter", return_value=10.0):
            lines = MissionDebugInfo(control).build_lines()

        self.assertEqual(
            lines,
            [
                "SLAM view: drone 1",
                "Dirty maps: 1",
                "Frontiers: 3",
                "Frontier cooldown: 1.00s",
            ],
        )

    def test_build_lines_includes_smoothed_frame_performance(self) -> None:
        profiler = FrameProfiler()
        profiler.record(
            frame_seconds=0.1,
            wait_seconds=0.04,
            stages={
                "sharing": 0.01,
                "sensors": 0.02,
                "render": 0.03,
            },
        )
        control = SimpleNamespace(
            drones=[],
            presentation=SimpleNamespace(selected_drone_heatmap_id=None),
            frame_profiler=profiler,
        )

        lines = MissionDebugInfo(control).build_lines()

        self.assertEqual(
            lines[-3:],
            [
                "Frame rate: 10.0 FPS (100.0 ms)",
                "Frame work/wait: 60.0 / 40.0 ms",
                "Stages ms: share 10.0, sense 20.0, render 30.0",
            ],
        )


if __name__ == "__main__":
    unittest.main()
