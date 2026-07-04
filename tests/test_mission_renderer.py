import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from agents.drone_runtime_state import DroneSnapshot
from asset_config.rendering import Colors
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
                ),
            ),
            is_paused=lambda: getattr(control, "is_paused", False),
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
            debug_info=debug_info,
            control_center=control_center,
            drones=[drone],
            rovers=[rover],
            presentation=SimpleNamespace(
                show_terrain_heatmap=False,
                selected_drone_heatmap_id=None,
            ),
        )
        renderer = MissionRenderer(self.make_dependencies(control))
        renderer.draw_stop_button = Mock(
            side_effect=lambda: events.append("stop"),
        )
        renderer.draw_restart_button = Mock(
            side_effect=lambda: events.append("restart"),
        )
        renderer.draw_pause_button = Mock(
            side_effect=lambda: events.append("pause"),
        )

        renderer.draw()

        self.assertEqual(
            events,
            [
                "clear",
                "slam",
                "drone_path",
                "rover_path",
                "drone_vision",
                "drone_icon",
                "rover_icon",
                "debug",
                "control_center",
                "stop",
                "restart",
                "pause",
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

    def test_mission_button_rects_and_pixels_are_renderer_owned(self) -> None:
        window = pygame.Surface((170, 70), pygame.SRCALPHA)
        control = SimpleNamespace(
            game=SimpleNamespace(window=window),
            is_paused=False,
        )
        renderer = MissionRenderer(self.make_dependencies(control))

        renderer.draw_stop_button()
        renderer.draw_restart_button()
        renderer.draw_pause_button()

        self.assertEqual(renderer.stop_button_rect, pygame.Rect(10, 10, 40, 40))
        self.assertEqual(
            renderer.restart_button_rect,
            pygame.Rect(58, 10, 40, 40),
        )
        self.assertEqual(
            renderer.pause_button_rect,
            pygame.Rect(106, 10, 40, 40),
        )
        self.assertFalse(
            renderer.stop_button_rect.colliderect(
                renderer.restart_button_rect,
            )
        )
        self.assertFalse(
            renderer.restart_button_rect.colliderect(
                renderer.pause_button_rect,
            )
        )
        self.assertEqual(window.get_at((15, 15))[:3], (255, 0, 0))
        self.assertEqual(
            window.get_at((63, 15))[:3],
            Colors.BLUE.value,
        )
        self.assertEqual(
            window.get_at((111, 15))[:3],
            Colors.OCHRE.value,
        )
        self.assertEqual(
            window.get_at(renderer.stop_button_rect.center)[:3],
            (255, 255, 255),
        )
        self.assertEqual(window.get_at((20, 20))[:3], (255, 255, 255))
        self.assertEqual(window.get_at((22, 22))[:3], (255, 0, 0))
        restart_pixels = pygame.surfarray.array3d(
            window.subsurface(renderer.restart_button_rect)
        )
        white_restart_pixels = np.all(
            restart_pixels == np.array((255, 255, 255)),
            axis=2,
        )
        self.assertGreater(int(np.count_nonzero(white_restart_pixels)), 40)
        self.assertEqual(
            window.get_at(renderer.pause_button_rect.center)[:3],
            Colors.OCHRE.value,
        )
        self.assertGreater(
            int(np.count_nonzero(pygame.surfarray.array_alpha(window))),
            0,
        )

    def test_pause_button_switches_to_play_symbol_when_paused(self) -> None:
        window = pygame.Surface((170, 70), pygame.SRCALPHA)
        control = SimpleNamespace(
            game=SimpleNamespace(window=window),
            is_paused=True,
        )
        renderer = MissionRenderer(self.make_dependencies(control))

        renderer.draw_pause_button()

        self.assertEqual(window.get_at((111, 15))[:3], Colors.GREEN.value)
        self.assertEqual(
            window.get_at(renderer.pause_button_rect.center)[:3],
            (255, 255, 255),
        )


if __name__ == "__main__":
    unittest.main()
