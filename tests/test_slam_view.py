import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from SlamMap import SlamMap
from mapping.terrain_knowledge import TerrainKnowledge
from rendering.slam_view import SlamViewService


def make_drone(shape=(3, 3)):
    terrain_knowledge = TerrainKnowledge(np.zeros(shape, dtype=np.uint8))
    return SimpleNamespace(
        slam_lock=threading.Lock(),
        slam_map=SlamMap(*shape),
        terrain_knowledge=terrain_knowledge,
        known_roughness=terrain_knowledge.roughness,
        terrain_confidence=terrain_knowledge.confidence,
    )


class SlamViewServiceTests(unittest.TestCase):
    def make_control(self):
        terrain_knowledge = TerrainKnowledge(
            np.zeros((3, 3), dtype=np.uint8)
        )
        return SimpleNamespace(
            drones=[],
            floor_mask=np.ones((3, 3), dtype=bool),
            settings=SimpleNamespace(
                slam_render_point_tail=10,
                slam_render_interval=0.1,
            ),
            presentation=SimpleNamespace(
                terrain_heatmap_dirty=True,
                selected_drone_heatmap_id=None,
                show_terrain_heatmap=False,
            ),
            slam_renderer=SimpleNamespace(
                surface=pygame.Surface((3, 3), pygame.SRCALPHA),
                render=Mock(),
            ),
            game=SimpleNamespace(
                window=pygame.Surface((3, 3), pygame.SRCALPHA),
            ),
            terrain_knowledge=terrain_knowledge,
            known_roughness=terrain_knowledge.roughness,
            terrain_confidence=terrain_knowledge.confidence,
        )

    def test_refresh_without_drones_clears_cached_surface(self) -> None:
        control = self.make_control()
        control.slam_renderer.surface.fill((255, 0, 0, 255))

        SlamViewService(control).refresh()

        self.assertEqual(control.slam_renderer.surface.get_at((0, 0)).a, 0)
        self.assertFalse(control.presentation.terrain_heatmap_dirty)

    def test_combined_view_uses_highest_confidence_cell(self) -> None:
        control = self.make_control()
        low = make_drone()
        high = make_drone()
        low.slam_map.occupancy[1, 1] = 0
        low.slam_map.confidence[1, 1] = 0.4
        high.slam_map.occupancy[1, 1] = 1
        high.slam_map.confidence[1, 1] = 0.9
        control.drones = [low, high]

        SlamViewService(control).refresh()

        args = control.slam_renderer.render.call_args.args
        self.assertEqual(int(args[0][1, 1]), 1)
        self.assertAlmostEqual(float(args[1][1, 1]), 0.9)
        self.assertFalse(low.slam_map.dirty)
        self.assertFalse(high.slam_map.dirty)

    def test_selected_heatmap_uses_drone_local_terrain(self) -> None:
        control = self.make_control()
        drone = make_drone()
        drone.known_roughness[1, 1] = 0.7
        drone.terrain_confidence[1, 1] = 1.0
        control.drones = [drone]
        control.presentation.selected_drone_heatmap_id = 0
        control.presentation.show_terrain_heatmap = True

        SlamViewService(control).refresh()

        kwargs = control.slam_renderer.render.call_args.kwargs
        np.testing.assert_array_equal(
            kwargs["roughness"],
            drone.known_roughness,
        )
        np.testing.assert_array_equal(
            kwargs["roughness_conf"],
            drone.terrain_confidence,
        )

    def test_draw_refreshes_dirty_map_then_blits_cached_surface(self) -> None:
        control = self.make_control()
        drone = make_drone()
        control.drones = [drone]
        control.game.window = SimpleNamespace(blit=Mock())
        service = SlamViewService(control)
        service.refresh = Mock()

        service.draw()

        service.refresh.assert_called_once_with()
        control.game.window.blit.assert_called_once_with(
            control.slam_renderer.surface,
            (0, 0),
        )

    def test_draw_throttles_dirty_surface_rebuilds_but_keeps_blitting(self) -> None:
        control = self.make_control()
        control.settings.slam_render_interval = 0.5
        drone = make_drone()
        control.drones = [drone]
        control.game.window = SimpleNamespace(blit=Mock())
        service = SlamViewService(control)

        with unittest.mock.patch(
            "rendering.slam_view.time.perf_counter",
            side_effect=[10.0, 10.0, 10.1, 10.6, 10.6],
        ):
            service.draw()
            drone.slam_map.dirty = True
            service.draw()
            self.assertTrue(drone.slam_map.dirty)
            service.draw()

        self.assertEqual(control.slam_renderer.render.call_count, 2)
        self.assertFalse(drone.slam_map.dirty)
        self.assertEqual(control.game.window.blit.call_count, 3)


if __name__ == "__main__":
    unittest.main()
