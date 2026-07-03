import os
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from ui.control_center.controller import ControlCenterController, ControlHitMap


class ControlCenterControllerTests(unittest.TestCase):
    def test_timer_excludes_paused_time(self) -> None:
        controller = ControlCenterController()

        with patch(
            "ui.control_center.controller.time.perf_counter",
            side_effect=[10.0, 40.0, 100.0, 120.0],
        ):
            controller.start_timer()
            controller.pause_timer()
            self.assertEqual(controller.format_timer(), "00:30")
            controller.resume_timer()
            self.assertEqual(controller.format_timer(), "00:50")

    def test_hit_map_returns_existing_actions_and_selects_tabs(self) -> None:
        controller = ControlCenterController()
        hit_map = ControlHitMap(
            heatmap_toggle=(0, 0, 10, 10),
            tabs=(("rovers", (20, 0, 10, 10)),),
            drone_toggles=(
                (0, "path", (40, 0, 10, 10)),
                (0, "vision", (60, 0, 10, 10)),
                (0, "terrain", (80, 0, 10, 10)),
            ),
        )

        self.assertEqual(
            controller.handle_click((5, 5), hit_map),
            ("terrain_heatmap", None),
        )
        self.assertEqual(
            controller.handle_click((25, 5), hit_map),
            ("control_tab", None),
        )
        self.assertEqual(controller.active_tab, "rovers")
        self.assertEqual(
            controller.handle_click((45, 5), hit_map),
            ("drone_path", 0),
        )
        self.assertEqual(
            controller.handle_click((65, 5), hit_map),
            ("drone_vision", 0),
        )
        self.assertEqual(
            controller.handle_click((85, 5), hit_map),
            ("drone_heatmap", 0),
        )

    def test_hit_map_is_detached_from_source_rect_values(self) -> None:
        source = [0, 0, 10, 10]
        hit_map = ControlHitMap(heatmap_toggle=source)
        source[0] = 99

        self.assertEqual(hit_map.heatmap_toggle, (0, 0, 10, 10))


if __name__ == "__main__":
    unittest.main()
