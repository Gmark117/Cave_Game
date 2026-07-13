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

        lines = MissionDebugInfo(dependencies).build_lines(snapshots)

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

        lines = MissionDebugInfo(dependencies).build_lines()

        self.assertEqual(
            lines[-3:],
            [
                "Frame rate: 10.0 FPS (100.0 ms)",
                "Frame work/wait: 60.0 / 40.0 ms",
                "Stages ms: share 10.0, sense 20.0, render 30.0",
            ],
        )

    def test_build_lines_includes_selected_drone_mcts_diagnostics(self) -> None:
        diagnostics = SimpleNamespace(
            selected_kind="translate",
            selected_direction=22,
            selected_target=(10, 12),
            iterations=5,
            generated_nodes=6,
            elapsed_ms=42.5,
        )
        drone = SimpleNamespace(
            exploration_policy=SimpleNamespace(
                last_search_diagnostics=diagnostics,
                config=SimpleNamespace(iterations=256),
            )
        )
        dependencies = MissionDebugDependencies(
            get_drones=lambda: [drone],
            presentation=SimpleNamespace(selected_drone_heatmap_id=0),
            dirty_map_count=lambda: 0,
            simulation_time=lambda: 10.0,
        )

        lines = MissionDebugInfo(dependencies).build_lines(
            [
                SimpleNamespace(
                    frontiers=(),
                    frontier_rebuild_cooldown=0.25,
                    last_frontier_rebuild=10.0,
                )
            ]
        )

        self.assertIn(
            "MCTS d0: translate 22deg -> 10,12, 5/256 it, 6 nodes, 42 ms",
            lines,
        )

    def test_mcts_diagnostics_allow_directionless_frontier_fallback(self) -> None:
        diagnostics = SimpleNamespace(
            selected_kind="frontier",
            selected_direction=None,
            selected_target=(10, 12),
            iterations=5,
            generated_nodes=6,
            elapsed_ms=42.5,
        )
        drone = SimpleNamespace(
            exploration_policy=SimpleNamespace(
                last_search_diagnostics=diagnostics,
                config=SimpleNamespace(iterations=256),
            )
        )
        dependencies = MissionDebugDependencies(
            get_drones=lambda: [drone],
            presentation=SimpleNamespace(selected_drone_heatmap_id=0),
            dirty_map_count=lambda: 0,
            simulation_time=lambda: 10.0,
        )

        lines = MissionDebugInfo(dependencies).build_lines(
            [
                SimpleNamespace(
                    frontiers=(),
                    frontier_rebuild_cooldown=0.25,
                    last_frontier_rebuild=10.0,
                )
            ]
        )

        self.assertIn(
            "MCTS d0: frontier - -> 10,12, 5/256 it, 6 nodes, 42 ms",
            lines,
        )


if __name__ == "__main__":
    unittest.main()
