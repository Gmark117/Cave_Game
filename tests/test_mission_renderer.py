import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from agents.drone_runtime_state import DroneSnapshot
from contracts import MissionRendererDependencies
from rendering.mission_renderer import MissionRenderer


class RecordingWindow:
    def __init__(self, events) -> None:
        self.events = events

    def fill(self, color) -> None:
        self.events.append("clear")


class RecordingAgentRenderer:
    def __init__(self, prefix: str, events) -> None:
        self.prefix = prefix
        self.events = events
        self.path_snapshot = None
        self.vision_snapshot = None
        self.icon_snapshot = None

    def draw_path(self, snapshot=None) -> None:
        self.path_snapshot = snapshot
        self.events.append(f"{self.prefix}_path")

    def draw_vision_overlay(self, snapshot) -> None:
        self.vision_snapshot = snapshot
        self.events.append(f"{self.prefix}_vision")

    def draw_icon(self, snapshot=None) -> None:
        self.icon_snapshot = snapshot
        self.events.append(f"{self.prefix}_icon")


class MissionRendererTests(unittest.TestCase):
    @staticmethod
    def make_dependencies(control) -> MissionRendererDependencies:
        return MissionRendererDependencies(
            get_window=lambda: control.game.window,
            slam_view=getattr(control, "slam_view", SimpleNamespace(draw=lambda: None)),
            debug_info=getattr(
                control,
                "debug_info",
                SimpleNamespace(build_lines=lambda snapshots: []),
            ),
            get_control_center=lambda: getattr(control, "control_center", None),
            get_drones=lambda: getattr(control, "drones", []),
            get_rovers=lambda: getattr(control, "rovers", []),
            presentation=getattr(
                control,
                "presentation",
                SimpleNamespace(
                    show_terrain_heatmap=False,
                    selected_drone_heatmap_id=None,
                    show_full_map=False,
                ),
            ),
            is_paused=lambda: getattr(control, "is_paused", False),
            is_music_enabled=lambda: getattr(control, "music_enabled", True),
            waypoint_renderer=getattr(control, "waypoint_renderer", None),
        )

    def test_draw_uses_stable_scene_layer_order(self) -> None:
        events = []
        drone_snapshot = DroneSnapshot(
            position=(2, 3),
            direction=0,
            direction_history=(),
            path_history=((2, 3),),
            frontiers=(),
            returning_home=False,
            done=False,
            explored=True,
            heading_deg=0.0,
            ray_points=(),
            battery=100,
            show_path=True,
            show_vision=True,
            frontier_rebuild_cooldown=0.25,
            last_frontier_rebuild=0.0,
        )
        drone_renderer = RecordingAgentRenderer("drone", events)
        drone = SimpleNamespace(
            id=0,
            color=(1, 2, 3),
            snapshot=Mock(return_value=drone_snapshot),
            renderer=drone_renderer,
        )
        rover = SimpleNamespace(
            id=0,
            color=(4, 5, 6),
            battery=2400,
            status="Ready",
            renderer=RecordingAgentRenderer("rover", events),
        )
        slam_view = SimpleNamespace(
            draw=lambda: events.append("slam"),
        )
        waypoint_renderer = SimpleNamespace(
            draw=lambda window: events.append("waypoints"),
        )
        build_debug_lines = Mock(
            side_effect=lambda snapshots: events.append("debug") or ["line"],
        )
        debug_info = SimpleNamespace(
            build_lines=build_debug_lines,
        )
        draw_control_center = Mock(
            side_effect=lambda *args: events.append("control_center"),
        )
        control_center = SimpleNamespace(
            draw_control_center=draw_control_center,
        )
        control = SimpleNamespace(
            game=SimpleNamespace(window=RecordingWindow(events)),
            slam_view=slam_view,
            waypoint_renderer=waypoint_renderer,
            debug_info=debug_info,
            control_center=control_center,
            drones=[drone],
            rovers=[rover],
            presentation=SimpleNamespace(
                show_terrain_heatmap=False,
                selected_drone_heatmap_id=None,
                show_full_map=False,
            ),
            is_paused=True,
            music_enabled=False,
        )
        renderer = MissionRenderer(self.make_dependencies(control))

        renderer.draw()

        self.assertEqual(
            events,
            [
                "clear",
                "slam",
                "waypoints",
                "drone_path",
                "rover_path",
                "drone_vision",
                "drone_icon",
                "rover_icon",
                "debug",
                "control_center",
            ],
        )
        control_center_args = draw_control_center.call_args.args
        self.assertIsNot(control_center_args[0][0], drone)
        self.assertIsNot(control_center_args[1][0], rover)
        drone.snapshot.assert_called_once_with()
        self.assertIs(drone_renderer.path_snapshot, drone_snapshot)
        self.assertIs(drone_renderer.vision_snapshot, drone_snapshot)
        self.assertIs(drone_renderer.icon_snapshot, drone_snapshot)
        self.assertIs(
            build_debug_lines.call_args.args[0][0],
            drone_snapshot,
        )
        self.assertTrue(control_center_args[5])
        self.assertFalse(control_center_args[6])
        self.assertFalse(control_center_args[7])

    def test_draw_skips_black_clear_when_static_background_draws(self) -> None:
        events = []
        slam_view = SimpleNamespace(
            draw_static_background=lambda: events.append("background") or True,
            draw=lambda: events.append("slam"),
        )
        draw_control_center = Mock(
            side_effect=lambda *args: events.append("control_center"),
        )
        control = SimpleNamespace(
            game=SimpleNamespace(window=RecordingWindow(events)),
            slam_view=slam_view,
            control_center=SimpleNamespace(
                draw_control_center=draw_control_center,
            ),
            drones=[],
            rovers=[],
            presentation=SimpleNamespace(
                show_terrain_heatmap=False,
                selected_drone_heatmap_id=None,
                show_full_map=True,
            ),
        )

        MissionRenderer(self.make_dependencies(control)).draw()

        self.assertEqual(
            events,
            ["background", "slam", "control_center"],
        )

    def test_selected_drone_view_hides_shared_waypoint_overlay(self) -> None:
        waypoint_renderer = SimpleNamespace(draw=Mock())
        control = SimpleNamespace(
            game=SimpleNamespace(window=RecordingWindow([])),
            slam_view=SimpleNamespace(draw=lambda: None),
            waypoint_renderer=waypoint_renderer,
            control_center=SimpleNamespace(
                draw_control_center=Mock(),
            ),
            drones=[],
            rovers=[],
            presentation=SimpleNamespace(
                show_terrain_heatmap=False,
                selected_drone_heatmap_id=0,
                show_full_map=False,
            ),
        )

        MissionRenderer(self.make_dependencies(control)).draw()

        waypoint_renderer.draw.assert_not_called()


if __name__ == "__main__":
    unittest.main()
