import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from ControlCenter import (
    ControlCenter,
    ControlCenterRenderState,
    ControlCenterTabState,
)
from asset_config.rendering import Colors


class ControlCenterTests(unittest.TestCase):
    def test_tab_state_registers_selects_and_clears_hit_areas(self) -> None:
        state = ControlCenterTabState()
        state.register("debug", pygame.Rect(10, 10, 20, 20))

        self.assertEqual(state.hit_test((15, 15)), "debug")
        self.assertEqual(state.active_tab, "debug")
        self.assertIsNone(state.hit_test((0, 0)))
        state.clear()
        self.assertEqual(state.rects, {})

    def test_timer_and_percentage_colors_are_deterministic(self) -> None:
        center = object.__new__(ControlCenter)
        center.tic = None
        center.paused_at = None
        center.paused_duration = 0.0
        self.assertEqual(center.format_timer(), "00:00")

        center.tic = 10.0
        with patch("ControlCenter.time.perf_counter", return_value=75.0):
            self.assertEqual(center.format_timer(), "01:05")

        self.assertEqual(center.percent_color(10), Colors.RED.value)
        self.assertEqual(center.percent_color(50), Colors.YELLOW.value)
        self.assertEqual(center.percent_color(90), Colors.GREEN.value)

    def test_timer_excludes_paused_time(self) -> None:
        center = object.__new__(ControlCenter)
        center.tic = 10.0
        center.paused_at = None
        center.paused_duration = 0.0

        with patch(
            "ControlCenter.time.perf_counter",
            side_effect=[40.0, 100.0],
        ):
            center.pause_timer()
            self.assertEqual(center.format_timer(), "00:30")
            center.resume_timer()

        with patch("ControlCenter.time.perf_counter", return_value=120.0):
            self.assertEqual(center.format_timer(), "00:50")

    def test_handle_click_returns_tokens_and_toggles_agent_overlays(self) -> None:
        center = object.__new__(ControlCenter)
        center._render_state = ControlCenterRenderState()
        center.heatmap_toggle_rect = pygame.Rect(0, 0, 10, 10)
        center._tab_state = ControlCenterTabState()
        center._tab_state.register("rovers", pygame.Rect(20, 0, 10, 10))
        center.drone_toggle_rects = {
            (0, "path"): pygame.Rect(40, 0, 10, 10),
            (0, "vision"): pygame.Rect(60, 0, 10, 10),
            (0, "terrain"): pygame.Rect(80, 0, 10, 10),
        }
        drone = SimpleNamespace(toggle_path=Mock(), toggle_vision=Mock())

        self.assertEqual(
            center.handle_click((5, 5), [drone]),
            ("terrain_heatmap", None),
        )
        self.assertEqual(
            center.handle_click((25, 5), [drone]),
            ("control_tab", None),
        )
        self.assertEqual(center.active_tab, "rovers")
        self.assertEqual(
            center.handle_click((45, 5), [drone]),
            ("drone_overlay", 0),
        )
        drone.toggle_path.assert_called_once_with()
        self.assertEqual(
            center.handle_click((65, 5), [drone]),
            ("drone_overlay", 0),
        )
        drone.toggle_vision.assert_called_once_with()
        self.assertEqual(
            center.handle_click((85, 5), [drone]),
            ("drone_heatmap", 0),
        )


if __name__ == "__main__":
    unittest.main()
