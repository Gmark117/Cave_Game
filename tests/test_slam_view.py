import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from config.simulation_config import RenderingConfig, SimulationConfig
from mapping.slam_map import UNKNOWN, SlamMap, SlamSnapshot
from mapping.terrain_knowledge import TerrainKnowledge
from contracts import SlamViewDependencies
from rendering.slam_view import SlamViewService


def make_drone(shape=(3, 3)):
    terrain_knowledge = TerrainKnowledge(np.zeros(shape, dtype=np.uint8))
    return SimpleNamespace(
        slam_map=SlamMap(*shape),
        terrain_knowledge=terrain_knowledge,
    )


def seed_slam(
    drone,
    x: int,
    y: int,
    occupancy_value: int,
    confidence_value: float,
) -> None:
    occupancy = np.full(drone.slam_map.shape, UNKNOWN, dtype=np.int8)
    confidence = np.zeros(drone.slam_map.shape, dtype=np.float32)
    occupancy[y, x] = occupancy_value
    confidence[y, x] = confidence_value
    drone.slam_map.merge_from(
        SlamSnapshot(occupancy, confidence)
    )


class SlamViewServiceTests(unittest.TestCase):
    def make_control(self):
        terrain_knowledge = TerrainKnowledge(
            np.zeros((3, 3), dtype=np.uint8)
        )
        control = SimpleNamespace(
            drones=[],
            floor_mask=np.ones((3, 3), dtype=bool),
            settings=SimulationConfig(
                rendering=RenderingConfig(
                    point_tail=10,
                    refresh_interval=0.1,
                )
            ),
            presentation=SimpleNamespace(
                terrain_heatmap_dirty=True,
                selected_drone_heatmap_id=None,
                show_terrain_heatmap=False,
                show_full_map=False,
            ),
            slam_renderer=SimpleNamespace(
                surface=pygame.Surface((3, 3), pygame.SRCALPHA),
                render=Mock(),
                full_map_underlay=Mock(
                    return_value=pygame.Surface((3, 3)),
                ),
            ),
            game=SimpleNamespace(
                window=pygame.Surface((3, 3), pygame.SRCALPHA),
            ),
            terrain_knowledge=terrain_knowledge,
        )
        control.dependencies = SlamViewDependencies(
            rendering=control.settings.rendering,
            terrain_knowledge=control.terrain_knowledge,
            presentation=control.presentation,
            slam_renderer=control.slam_renderer,
            get_drones=lambda: control.drones,
            get_window=lambda: control.game.window,
        )
        return control

    def test_refresh_without_drones_clears_cached_surface(self) -> None:
        control = self.make_control()
        control.slam_renderer.surface.fill((255, 0, 0, 255))

        SlamViewService(control.dependencies).refresh()

        self.assertEqual(control.slam_renderer.surface.get_at((0, 0)).a, 0)
        self.assertFalse(control.presentation.terrain_heatmap_dirty)

    def test_combined_view_uses_highest_confidence_cell(self) -> None:
        control = self.make_control()
        low = make_drone()
        high = make_drone()
        seed_slam(low, 1, 1, 0, 0.4)
        seed_slam(high, 1, 1, 1, 0.9)
        control.drones = [low, high]
        service = SlamViewService(control.dependencies)

        service.refresh()

        args = control.slam_renderer.render.call_args.args
        self.assertEqual(int(args[0][1, 1]), 1)
        self.assertAlmostEqual(float(args[1][1, 1]), 0.9)
        self.assertEqual(service.dirty_map_count(), 0)

    def test_draw_blits_full_map_underlay_before_slam_surface(self) -> None:
        control = self.make_control()
        control.presentation.show_full_map = True
        drone = make_drone()
        control.drones = [drone]
        control.game.window = SimpleNamespace(blit=Mock())
        service = SlamViewService(control.dependencies)
        service.refresh = Mock()

        service.draw_static_background()
        service.draw()

        control.slam_renderer.full_map_underlay.assert_called_once_with(
            control.terrain_knowledge.floor_mask,
        )
        self.assertEqual(control.game.window.blit.call_count, 2)
        self.assertIs(
            control.game.window.blit.call_args_list[0].args[0],
            control.slam_renderer.full_map_underlay.return_value,
        )
        self.assertIs(
            control.game.window.blit.call_args_list[1].args[0],
            control.slam_renderer.surface,
        )

    def test_full_map_underlay_can_be_disabled(self) -> None:
        control = self.make_control()
        control.presentation.show_full_map = False
        control.drones = [make_drone()]
        control.game.window = SimpleNamespace(blit=Mock())

        drew_background = SlamViewService(
            control.dependencies,
        ).draw_static_background()

        self.assertFalse(drew_background)
        control.slam_renderer.full_map_underlay.assert_not_called()
        control.game.window.blit.assert_not_called()

    def test_selected_heatmap_uses_drone_local_terrain(self) -> None:
        control = self.make_control()
        drone = make_drone()
        drone.terrain_knowledge.roughness[1, 1] = 0.7
        drone.terrain_knowledge.confidence[1, 1] = 1.0
        control.drones = [drone]
        control.presentation.selected_drone_heatmap_id = 0
        control.presentation.show_terrain_heatmap = True

        SlamViewService(control.dependencies).refresh()

        kwargs = control.slam_renderer.render.call_args.kwargs
        np.testing.assert_array_equal(
            kwargs["roughness"],
            drone.terrain_knowledge.roughness,
        )
        np.testing.assert_array_equal(
            kwargs["roughness_conf"],
            drone.terrain_knowledge.confidence,
        )

    def test_draw_refreshes_dirty_map_then_blits_cached_surface(self) -> None:
        control = self.make_control()
        drone = make_drone()
        control.drones = [drone]
        control.presentation.show_full_map = False
        control.game.window = SimpleNamespace(blit=Mock())
        service = SlamViewService(control.dependencies)
        service.refresh = Mock()

        service.draw()

        service.refresh.assert_called_once_with()
        control.game.window.blit.assert_called_once_with(
            control.slam_renderer.surface,
            (0, 0),
        )

    def test_draw_throttles_dirty_surface_rebuilds_but_keeps_blitting(self) -> None:
        control = self.make_control()
        control.presentation.show_full_map = False
        control.settings = replace(
            control.settings,
            rendering=replace(
                control.settings.rendering,
                refresh_interval=0.5,
            ),
        )
        control.dependencies = replace(
            control.dependencies,
            rendering=control.settings.rendering,
        )
        drone = make_drone()
        control.drones = [drone]
        control.game.window = SimpleNamespace(blit=Mock())
        service = SlamViewService(control.dependencies)

        with unittest.mock.patch(
            "rendering.slam_view.time.perf_counter",
            side_effect=[10.0, 10.0, 10.1, 10.6, 10.6],
        ):
            service.draw()
            seed_slam(drone, 1, 1, 1, 0.9)
            service.draw()
            self.assertEqual(service.dirty_map_count(), 1)
            service.draw()

        self.assertEqual(control.slam_renderer.render.call_count, 2)
        self.assertEqual(service.dirty_map_count(), 0)
        self.assertEqual(control.game.window.blit.call_count, 3)

    def test_update_during_render_remains_pending_for_next_refresh(self) -> None:
        control = self.make_control()
        drone = make_drone()
        control.drones = [drone]
        service = SlamViewService(control.dependencies)

        def update_while_rendering(*args, **kwargs) -> None:
            seed_slam(drone, 1, 1, 1, 0.9)

        control.slam_renderer.render.side_effect = update_while_rendering
        service.refresh()

        self.assertEqual(service.dirty_map_count(), 1)


if __name__ == "__main__":
    unittest.main()
