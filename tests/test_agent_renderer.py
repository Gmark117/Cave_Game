import os
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from Drone import Drone
from Rover import Rover
from rendering.agent_renderer import DroneRenderer, RoverRenderer


class RenderControl:
    delay = 1 / 15
    terrain_roughness = np.full((64, 64), 0.4, dtype=np.float32)

    def record_terrain_scan(self, samples) -> None:
        pass


class AgentRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        settings = SimpleNamespace(
            map_dim="SMALL",
            slam_scan_interval=0.0,
            slam_scan_rays=5,
            slam_point_cloud_max_points=50,
            frontier_rebuild_cooldown=0.25,
            frontier_stride=4,
            frontier_confidence_threshold=0.6,
        )
        self.window = pygame.Surface((64, 64), pygame.SRCALPHA)
        self.game = SimpleNamespace(
            sim_settings=settings,
            window=self.window,
            width=64,
            height=64,
        )
        self.control = RenderControl()
        self.cave = np.zeros((64, 64), dtype=np.uint8)

    def test_drone_renderer_owns_surfaces_and_draws_cached_state(self) -> None:
        icon = pygame.Surface((4, 4), pygame.SRCALPHA)
        icon.fill((255, 255, 255, 255))
        drone = Drone(
            self.game,
            self.control,
            0,
            (32, 32),
            (255, 0, 0),
            icon,
            self.cave,
        )
        drone.graph.add_node((40, 32))
        drone.ray_points = [(40, 20), (24, 20)]

        self.assertIsInstance(drone.renderer, DroneRenderer)
        self.assertIs(drone.floor_surf, drone.renderer.path_surface)
        self.assertIs(drone.vision_overlay, drone.renderer.vision_surface)

        drone.renderer.draw_path()
        drone.renderer.draw_vision_overlay()
        drone.renderer.draw_icon()

        self.assertGreater(
            int(np.count_nonzero(pygame.surfarray.array_alpha(drone.floor_surf))),
            0,
        )
        self.assertGreater(
            int(
                np.count_nonzero(
                    pygame.surfarray.array_alpha(drone.vision_overlay)
                )
            ),
            0,
        )

    def test_rover_renderer_owns_path_surface(self) -> None:
        icon = pygame.Surface((4, 4), pygame.SRCALPHA)
        icon.fill((255, 255, 255, 255))
        rover = Rover(
            self.game,
            self.control,
            0,
            (32, 32),
            (0, 255, 0),
            icon,
            self.cave,
        )
        rover.graph.add_node((36, 32))

        self.assertIsInstance(rover.renderer, RoverRenderer)
        self.assertIs(rover.floor_surf, rover.renderer.path_surface)

        rover.renderer.draw_path()
        rover.renderer.draw_icon()

        self.assertGreater(
            int(np.count_nonzero(pygame.surfarray.array_alpha(rover.floor_surf))),
            0,
        )


if __name__ == "__main__":
    unittest.main()

