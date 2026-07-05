import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from agents.drone import Drone
from agents.rover import Rover
from config.simulation_config import MissionConfig, SimulationConfig, SlamConfig
from rendering.agent_renderer import DroneRenderer, RoverRenderer


class RenderControl:
    delay = 1 / 15
    terrain_roughness = np.full((64, 64), 0.4, dtype=np.float32)
    terrain_fusion = SimpleNamespace(record_scan=lambda samples: None)
    rover_targets = SimpleNamespace(acquire=Mock(), release=Mock())

    def compute_path(self, start, goal):
        return []

    def compute_rover_path(self, start, goal):
        return []

    def simulation_time(self) -> float:
        return 1.0

    def pause_checkpoint(self) -> bool:
        return True

    def wait_simulation_delay(self, duration: float) -> bool:
        return True


class AgentRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        settings = SimulationConfig(
            mission_config=MissionConfig(map_dim="SMALL"),
            slam=SlamConfig(
                scan_interval=0.0,
                scan_rays=5,
                point_cloud_max_points=50,
            ),
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
        drone.runtime_state.move_to((40, 32))
        drone.runtime_state.set_ray_points([(40, 20), (24, 20)])
        snapshot = drone.snapshot()

        self.assertIsInstance(drone.renderer, DroneRenderer)

        drone.renderer.draw_path(snapshot)
        drone.renderer.draw_vision_overlay(snapshot)
        drone.renderer.draw_icon(snapshot)

        self.assertGreater(
            int(
                np.count_nonzero(
                    pygame.surfarray.array_alpha(
                        drone.renderer.path_surface
                    )
                )
            ),
            0,
        )
        self.assertGreater(
            int(
                np.count_nonzero(
                    pygame.surfarray.array_alpha(
                        drone.renderer.vision_surface
                    )
                )
            ),
            0,
        )

    def test_drone_path_draws_only_new_segments(self) -> None:
        icon = pygame.Surface((4, 4), pygame.SRCALPHA)
        drone = Drone(
            self.game,
            self.control,
            0,
            (32, 32),
            (255, 0, 0),
            icon,
            self.cave,
        )
        drone.runtime_state.move_to((36, 32))
        first_snapshot = drone.snapshot()

        with patch(
            "rendering.agent_renderer.pygame.draw.line",
            return_value=pygame.Rect(0, 0, 1, 1),
        ) as draw_line:
            drone.renderer.draw_path(first_snapshot)
            drone.renderer.draw_path(first_snapshot)

            drone.runtime_state.move_to((40, 32))
            drone.renderer.draw_path(drone.snapshot())

        self.assertEqual(draw_line.call_count, 2)

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

        rover.renderer.draw_path()
        rover.renderer.draw_icon()

        self.assertGreater(
            int(
                np.count_nonzero(
                    pygame.surfarray.array_alpha(
                        rover.renderer.path_surface
                    )
                )
            ),
            0,
        )

    def test_rover_path_draws_only_new_segments(self) -> None:
        icon = pygame.Surface((4, 4), pygame.SRCALPHA)
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

        with patch(
            "rendering.agent_renderer.pygame.draw.line",
            return_value=pygame.Rect(0, 0, 1, 1),
        ) as draw_line:
            rover.renderer.draw_path()
            rover.renderer.draw_path()

            rover.graph.add_node((40, 32))
            rover.renderer.draw_path()

        self.assertEqual(draw_line.call_count, 2)


if __name__ == "__main__":
    unittest.main()
