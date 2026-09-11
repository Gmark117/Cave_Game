import unittest
from types import SimpleNamespace
from mission.frame_timing import FrameProfiler
from mission.debug_info import MissionDebugInfo
from contracts import MissionDebugDependencies


class MissionDebugInfoTests(unittest.TestCase):
    def test_build_lines_summarizes_current_mapping_state(self) -> None:
        snapshots = [
            SimpleNamespace(
                frontiers=((1, 1), (2, 2)),
                frontier_rebuild_cooldown=1.5,
                last_frontier_rebuild=9.5,
            ),
            SimpleNamespace(
                frontiers=((3, 3),),
                frontier_rebuild_cooldown=2.0,
                last_frontier_rebuild=9.0,
            ),
        ]
        dependencies = MissionDebugDependencies(
            get_drones=lambda: [object(), object()],
            presentation=SimpleNamespace(selected_drone_heatmap_id=1),
            dirty_map_count=lambda: 1,
            simulation_time=lambda: 10.0,
        )

        lines = MissionDebugInfo(dependencies).build_system_lines(snapshots)

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
        dependencies = MissionDebugDependencies(
            get_drones=lambda: [],
            presentation=SimpleNamespace(selected_drone_heatmap_id=None),
            dirty_map_count=lambda: 0,
            simulation_time=lambda: 10.0,
            frame_profiler=profiler,
        )

        lines = MissionDebugInfo(dependencies).build_system_lines()

        self.assertEqual(
            lines[-3:],
            [
                "Frame rate: 10.0 FPS (100.0 ms)",
                "Frame work/wait: 60.0 / 40.0 ms",
                "Stages ms: share 10.0, sense 20.0, render 30.0",
            ],
        )

    def test_build_lines_exposes_drone_phase_and_rover_hold_reason(self) -> None:
        drone = SimpleNamespace(
            id=2,
            movement_controller=SimpleNamespace(
                activity_snapshot=lambda: SimpleNamespace(
                    state="DFS backtrack",
                    detail="directive 5 component 4 depth 15 target 1115,403",
                ),
            ),
        )
        rover = SimpleNamespace(
            id=0,
            snapshot=lambda: SimpleNamespace(
                position=(428, 436),
                target=None,
                status="Rendezvous",
            ),
        )
        dependencies = MissionDebugDependencies(
            get_drones=lambda: [drone],
            get_rovers=lambda: [rover],
            presentation=SimpleNamespace(
                selected_drone_heatmap_id=None,
                selected_rover_heatmap_id=0,
            ),
            dirty_map_count=lambda: 0,
            simulation_time=lambda: 10.0,
        )
        snapshots = [SimpleNamespace(
            frontiers=(),
            frontier_rebuild_cooldown=1.0,
            last_frontier_rebuild=10.0,
        )]

        info = MissionDebugInfo(dependencies)
        lines = info.build_debug_lines(snapshots)

        self.assertIn(
            "D2 DFS backtrack: directive 5 component 4 depth 15 "
            "target 1115,403",
            lines,
        )
        self.assertIn(
            "R0 Rendezvous: pos 428,436 target none; "
            "holding for drone rendezvous",
            lines,
        )
        self.assertNotIn("SLAM view: rover 0", lines)
        self.assertIn(
            "SLAM view: rover 0",
            info.build_system_lines(snapshots),
        )

if __name__ == "__main__":
    unittest.main()
