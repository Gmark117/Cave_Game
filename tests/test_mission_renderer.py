import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

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

    def draw_path(self) -> None:
        self.events.append(f"{self.prefix}_path")

    def draw_vision_overlay(self) -> None:
        self.events.append(f"{self.prefix}_vision")

    def draw_icon(self) -> None:
        self.events.append(f"{self.prefix}_icon")


class MissionRendererTests(unittest.TestCase):
    def test_draw_uses_stable_scene_layer_order(self) -> None:
        events = []
        drone = SimpleNamespace(
            renderer=RecordingAgentRenderer("drone", events),
        )
        rover = SimpleNamespace(
            renderer=RecordingAgentRenderer("rover", events),
        )
        slam_view = SimpleNamespace(
            draw=lambda: events.append("slam"),
        )
        debug_info = SimpleNamespace(
            build_lines=lambda: events.append("debug") or ["line"],
        )
        control_center = SimpleNamespace(
            draw_control_center=lambda *args: events.append("control_center"),
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
        renderer = MissionRenderer(control)
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

    def test_mission_button_rects_and_pixels_are_renderer_owned(self) -> None:
        window = pygame.Surface((170, 70), pygame.SRCALPHA)
        control = SimpleNamespace(
            game=SimpleNamespace(window=window),
            is_paused=False,
        )
        renderer = MissionRenderer(control)

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
        self.assertEqual(window.get_at((63, 15))[:3], (230, 190, 100))
        self.assertEqual(window.get_at((111, 15))[:3], (0, 0, 153))
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
            (0, 0, 153),
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
        renderer = MissionRenderer(control)

        renderer.draw_pause_button()

        self.assertEqual(
            window.get_at(renderer.pause_button_rect.center)[:3],
            (255, 255, 255),
        )


if __name__ == "__main__":
    unittest.main()
