import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from ControlCenterRenderer import ControlCenterRenderer
from rendering.control_center_view_model import (
    ControlCenterViewModel,
    DroneStatusView,
    RoverStatusView,
)


class ControlCenterRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.HIDDEN)

    def test_render_consumes_one_view_model_and_returns_hit_map(self) -> None:
        window = pygame.Surface((1920, 1080), pygame.SRCALPHA)
        game = SimpleNamespace(window=window)
        with patch.object(
            ControlCenterRenderer,
            "_load_tab_sprites",
        ):
            renderer = ControlCenterRenderer(game)
        view = ControlCenterViewModel(
            elapsed_time="00:12",
            explored_percent=40,
            active_tab="drones",
            drone_statuses=(
                DroneStatusView(
                    id=0,
                    name="Blinky",
                    color=(255, 0, 0),
                    battery=100,
                    status="Deployed",
                    show_path=True,
                    show_vision=True,
                ),
            ),
            rover_statuses=(),
            show_terrain_heatmap=False,
            selected_drone_heatmap_id=None,
            debug_lines=(),
        )

        hit_map = renderer.render(view)

        self.assertIsNotNone(hit_map.heatmap_toggle)
        self.assertEqual(len(hit_map.tabs), 4)
        self.assertEqual(
            {(drone_id, action) for drone_id, action, _ in hit_map.drone_toggles},
            {(0, "path"), (0, "vision"), (0, "terrain")},
        )
        self.assertFalse(hasattr(renderer, "control_center"))

    def test_all_tabs_render_from_frame_data_without_facade_callbacks(
        self,
    ) -> None:
        window = pygame.Surface((1920, 1080), pygame.SRCALPHA)
        game = SimpleNamespace(window=window)
        with patch.object(
            ControlCenterRenderer,
            "_load_tab_sprites",
        ):
            renderer = ControlCenterRenderer(game)

        for active_tab in ("drones", "rovers", "debug", "system"):
            with self.subTest(active_tab=active_tab):
                hit_map = renderer.render(
                    ControlCenterViewModel(
                        elapsed_time="00:12",
                        explored_percent=40,
                        active_tab=active_tab,
                        drone_statuses=(
                            DroneStatusView(
                                id=0,
                                name="Blinky",
                                color=(255, 0, 0),
                                battery=100,
                                status="Deployed",
                                show_path=True,
                                show_vision=True,
                            ),
                        ),
                        rover_statuses=(
                            RoverStatusView(
                                id=0,
                                name="Huey",
                                color=(220, 0, 0),
                                battery=2400,
                                status="Ready",
                            ),
                        ),
                        show_terrain_heatmap=False,
                        selected_drone_heatmap_id=None,
                        debug_lines=("State: ready",),
                    )
                )

                self.assertEqual(len(hit_map.tabs), 4)


if __name__ == "__main__":
    unittest.main()
