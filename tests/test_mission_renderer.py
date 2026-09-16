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
        self.sharing_pair = None

    def draw_path(self, snapshot=None) -> None:
        self.path_snapshot = snapshot
        self.events.append(f"{self.prefix}_path")

    def draw_vision_overlay(self, snapshot) -> None:
        self.vision_snapshot = snapshot
        self.events.append(f"{self.prefix}_vision")

    def draw_icon(self, snapshot=None) -> None:
        self.icon_snapshot = snapshot
        self.events.append(f"{self.prefix}_icon")

    def draw_sharing_cue(
        self,
        snapshot,
        target_position,
        target_radius=None,
    ) -> None:
        self.sharing_pair = (snapshot, target_position, target_radius)
        self.events.append(f"{self.prefix}_sharing")


class MissionRendererTests(unittest.TestCase):
    @staticmethod
    def make_dependencies(control) -> MissionRendererDependencies:
        return MissionRendererDependencies(
            get_window=lambda: control.game.window,
            slam_view=getattr(control, "slam_view", SimpleNamespace(draw=lambda: None)),
            debug_info=getattr(
                control,
                "debug_info",
                SimpleNamespace(
                    build_debug_lines=lambda snapshots: [],
                    build_system_lines=lambda snapshots: [],
                ),
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
                    selected_rover_heatmap_id=None,
                    show_full_map=False,
                ),
            ),
            is_paused=lambda: getattr(control, "is_paused", False),
            is_music_enabled=lambda: getattr(control, "music_enabled", True),
            sector_renderer=getattr(control, "sector_renderer", None),
            get_sector_snapshot=lambda: getattr(
                control,
                "sector_snapshot",
                None,
            ),
            is_exploration_complete=lambda: getattr(
                control,
                "exploration_complete",
                False,
            ),
            get_docked_drone_ids=lambda: frozenset(getattr(
                control,
                "docked_drone_ids",
                (),
            )),
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
        build_debug_lines = Mock(
            side_effect=lambda snapshots: events.append("debug") or ["line"],
        )
        build_system_lines = Mock(
            side_effect=lambda snapshots: events.append("system")
            or ["system line"],
        )
        debug_info = SimpleNamespace(
            build_debug_lines=build_debug_lines,
            build_system_lines=build_system_lines,
        )
        draw_control_center = Mock(
            side_effect=lambda *args, **kwargs: events.append("control_center"),
        )
        control_center = SimpleNamespace(
            draw_control_center=draw_control_center,
        )
        control = SimpleNamespace(
            game=SimpleNamespace(window=RecordingWindow(events)),
            slam_view=slam_view,
            debug_info=debug_info,
            control_center=control_center,
            drones=[drone],
            rovers=[rover],
            presentation=SimpleNamespace(
                show_terrain_heatmap=False,
                selected_drone_heatmap_id=None,
                selected_rover_heatmap_id=None,
                show_full_map=False,
            ),
            is_paused=True,
            music_enabled=False,
            exploration_complete=True,
            sector_snapshot=object(),
        )
        control.sector_renderer = SimpleNamespace(
            draw=lambda *args: events.append("sectors"),
        )
        renderer = MissionRenderer(self.make_dependencies(control))

        renderer.draw()

        self.assertEqual(
            events,
            [
                "clear",
                "slam",
                "sectors",
                "drone_path",
                "rover_path",
                "drone_vision",
                "drone_icon",
                "rover_icon",
                "debug",
                "system",
                "control_center",
            ],
        )
        control_center_values = draw_control_center.call_args.kwargs
        self.assertIsNot(control_center_values["drone_statuses"][0], drone)
        self.assertIsNot(control_center_values["rover_statuses"][0], rover)
        drone.snapshot.assert_called_once_with()
        self.assertIs(drone_renderer.path_snapshot, drone_snapshot)
        self.assertIs(drone_renderer.vision_snapshot, drone_snapshot)
        self.assertIs(drone_renderer.icon_snapshot, drone_snapshot)
        self.assertIs(
            build_debug_lines.call_args.args[0][0],
            drone_snapshot,
        )
        self.assertIs(
            build_system_lines.call_args.args[0][0],
            drone_snapshot,
        )
        self.assertEqual(control_center_values["debug_lines"], ["line"])
        self.assertEqual(
            control_center_values["system_lines"],
            ["system line"],
        )
        self.assertTrue(control_center_values["is_paused"])
        self.assertFalse(control_center_values["music_enabled"])
        self.assertFalse(control_center_values["show_full_map"])
        self.assertTrue(control_center_values["exploration_complete"])

    def test_draw_omits_vision_overlay_for_docked_drone(self) -> None:
        events = []
        snapshot = DroneSnapshot(
            position=(2, 3),
            direction=0,
            direction_history=(),
            path_history=((2, 3),),
            frontiers=(),
            returning_home=False,
            done=False,
            explored=True,
            heading_deg=0.0,
            ray_points=((2, 1),),
            battery=100,
            show_path=True,
            show_vision=True,
            frontier_rebuild_cooldown=0.25,
            last_frontier_rebuild=0.0,
        )
        drone_renderer = RecordingAgentRenderer("drone", events)
        drone = SimpleNamespace(
            id=2,
            color=(1, 2, 3),
            snapshot=Mock(return_value=snapshot),
            renderer=drone_renderer,
        )
        control = SimpleNamespace(
            game=SimpleNamespace(window=RecordingWindow(events)),
            slam_view=SimpleNamespace(draw=lambda: events.append("slam")),
            debug_info=SimpleNamespace(
                build_debug_lines=lambda _snapshots: [],
                build_system_lines=lambda _snapshots: [],
            ),
            control_center=SimpleNamespace(
                draw_control_center=lambda **_kwargs: None,
            ),
            drones=[drone],
            rovers=[],
            presentation=SimpleNamespace(
                show_terrain_heatmap=False,
                selected_drone_heatmap_id=None,
                selected_rover_heatmap_id=None,
                show_full_map=False,
            ),
            docked_drone_ids={2},
        )

        MissionRenderer(self.make_dependencies(control)).draw()

        self.assertNotIn("drone_vision", events)
        self.assertIn("drone_path", events)
        self.assertIn("drone_icon", events)

    def test_draw_skips_black_clear_when_static_background_draws(self) -> None:
        events = []
        slam_view = SimpleNamespace(
            draw_static_background=lambda: events.append("background") or True,
            draw=lambda: events.append("slam"),
        )
        draw_control_center = Mock(
            side_effect=lambda *args, **kwargs: events.append("control_center"),
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
                selected_rover_heatmap_id=None,
                show_full_map=True,
            ),
        )

        MissionRenderer(self.make_dependencies(control)).draw()

        self.assertEqual(
            events,
            ["background", "slam", "control_center"],
        )

    def test_draw_renders_one_cue_for_an_active_sharing_pair(self) -> None:
        events = []

        def snapshot(position):
            return DroneSnapshot(
                position=position,
                direction=0,
                direction_history=(),
                path_history=(position,),
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

        first_snapshot = snapshot((2, 3))
        second_snapshot = snapshot((6, 3))
        first_renderer = RecordingAgentRenderer("first", events)
        second_renderer = RecordingAgentRenderer("second", events)
        first = SimpleNamespace(
            id=0,
            color=(1, 2, 3),
            snapshot=Mock(return_value=first_snapshot),
            renderer=first_renderer,
            movement_controller=SimpleNamespace(
                activity_snapshot=Mock(return_value=SimpleNamespace(
                    state="Sharing",
                    detail="with drone 2",
                    target=(6, 3),
                    peer_id=1,
                )),
            ),
        )
        second = SimpleNamespace(
            id=1,
            color=(4, 5, 6),
            snapshot=Mock(return_value=second_snapshot),
            renderer=second_renderer,
            movement_controller=SimpleNamespace(
                activity_snapshot=Mock(return_value=SimpleNamespace(
                    state="Sharing",
                    detail="with drone 1",
                    target=(2, 3),
                    peer_id=0,
                )),
            ),
        )
        control = SimpleNamespace(
            game=SimpleNamespace(window=RecordingWindow(events)),
            slam_view=SimpleNamespace(draw=lambda: events.append("slam")),
            debug_info=SimpleNamespace(
                build_debug_lines=lambda _snapshots: [],
                build_system_lines=lambda _snapshots: [],
            ),
            control_center=SimpleNamespace(
                draw_control_center=lambda **_kwargs: events.append(
                    "control_center"
                ),
            ),
            drones=[first, second],
            rovers=[],
            presentation=SimpleNamespace(
                show_terrain_heatmap=False,
                selected_drone_heatmap_id=None,
                selected_rover_heatmap_id=None,
                show_full_map=False,
            ),
        )

        MissionRenderer(self.make_dependencies(control)).draw()

        self.assertEqual(
            [event for event in events if event.endswith("_sharing")],
            ["first_sharing"],
        )
        self.assertEqual(
            first_renderer.sharing_pair,
            (first_snapshot, second_snapshot.position, None),
        )

    def test_draw_renders_cue_for_active_drone_rover_sharing(self) -> None:
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
            movement_controller=SimpleNamespace(
                activity_snapshot=Mock(return_value=SimpleNamespace(
                    state="Sharing",
                    detail="with rover 1",
                    target=(6, 3),
                    peer_id=None,
                    rover_id=0,
                )),
            ),
        )
        rover_snapshot = SimpleNamespace(
            position=(6, 3),
            target=None,
            status="Ready",
            battery=2400,
        )
        rover = SimpleNamespace(
            id=0,
            color=(4, 5, 6),
            pos=(6, 3),
            battery=2400,
            status="Ready",
            snapshot=Mock(return_value=rover_snapshot),
            renderer=RecordingAgentRenderer("rover", events),
        )
        control = SimpleNamespace(
            game=SimpleNamespace(window=RecordingWindow(events)),
            slam_view=SimpleNamespace(draw=lambda: events.append("slam")),
            debug_info=SimpleNamespace(
                build_debug_lines=lambda _snapshots: [],
                build_system_lines=lambda _snapshots: [],
            ),
            control_center=SimpleNamespace(
                draw_control_center=lambda **_kwargs: None,
            ),
            drones=[drone],
            rovers=[rover],
            presentation=SimpleNamespace(
                show_terrain_heatmap=False,
                selected_drone_heatmap_id=None,
                selected_rover_heatmap_id=None,
                show_full_map=False,
            ),
        )

        MissionRenderer(self.make_dependencies(control)).draw()

        self.assertEqual(
            drone_renderer.sharing_pair,
            (drone_snapshot, (6, 3), None),
        )

if __name__ == "__main__":
    unittest.main()
