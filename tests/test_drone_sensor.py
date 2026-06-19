import os
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from Drone import Drone
from SlamMap import OCCUPIED, SlamMap


class RecordingControl:
    delay = 1 / 15
    terrain_roughness = np.full((64, 64), 0.4, dtype=np.float32)

    def __init__(self) -> None:
        self.samples = []

    def record_terrain_scan(self, samples) -> None:
        self.samples.extend(samples)


class DroneSensorTests(unittest.TestCase):
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
        game = SimpleNamespace(
            sim_settings=settings,
            window=self.window,
            width=64,
            height=64,
        )
        self.control = RecordingControl()
        cave = np.zeros((64, 64), dtype=np.uint8)
        icon = pygame.Surface((4, 4), pygame.SRCALPHA)
        self.drone = Drone(
            game,
            self.control,
            0,
            (32, 32),
            (255, 0, 0),
            icon,
            cave,
        )

    def test_sensor_update_mutates_maps_but_overlay_draw_does_not(self) -> None:
        self.drone.show_vision = False
        self.drone.update_sensors()

        self.assertTrue(self.drone.ray_points)
        self.assertGreater(
            int(np.count_nonzero(self.drone.slam_map.confidence)),
            0,
        )
        self.assertTrue(self.control.samples)

        occupancy_before = self.drone.slam_map.occupancy.copy()
        confidence_before = self.drone.slam_map.confidence.copy()
        roughness_before = self.drone.known_roughness.copy()

        self.drone.show_vision = True
        self.drone.draw_vision_overlay()

        np.testing.assert_array_equal(
            occupancy_before,
            self.drone.slam_map.occupancy,
        )
        np.testing.assert_array_equal(
            confidence_before,
            self.drone.slam_map.confidence,
        )
        np.testing.assert_array_equal(
            roughness_before,
            self.drone.known_roughness,
        )

    def test_terrain_merge_is_confidence_weighted_and_ignores_walls(self) -> None:
        self.drone.terrain_knowledge.floor_mask[0, 1] = False
        self.drone.known_roughness[0, 0] = 0.2
        self.drone.terrain_confidence[0, 0] = 0.5
        source_roughness = np.full((64, 64), -1.0, dtype=np.float32)
        source_confidence = np.zeros((64, 64), dtype=np.float32)
        source_roughness[0, 0] = 0.8
        source_confidence[0, 0] = 0.5
        source_roughness[0, 1] = 1.0
        source_confidence[0, 1] = 1.0

        self.drone.merge_terrain_data(
            source_roughness,
            source_confidence,
        )

        self.assertAlmostEqual(float(self.drone.known_roughness[0, 0]), 0.5)
        self.assertAlmostEqual(float(self.drone.terrain_confidence[0, 0]), 1.0)
        self.assertEqual(float(self.drone.terrain_confidence[0, 1]), 0.0)

    def test_slam_merge_uses_other_map_when_confidence_is_higher(self) -> None:
        other = SlamMap(64, 64)
        other.occupancy[2, 2] = OCCUPIED
        other.confidence[2, 2] = 0.9

        self.drone.merge_slam_map(other)

        self.assertEqual(int(self.drone.slam_map.occupancy[2, 2]), OCCUPIED)
        self.assertAlmostEqual(
            float(self.drone.slam_map.confidence[2, 2]),
            0.9,
        )

    def test_terrain_compatibility_properties_expose_owned_model(self) -> None:
        self.assertIs(
            self.drone.known_roughness,
            self.drone.terrain_knowledge.roughness,
        )
        self.assertIs(
            self.drone.terrain_confidence,
            self.drone.terrain_knowledge.confidence,
        )
        self.assertIs(
            self.drone.terrain_lock,
            self.drone.terrain_knowledge.lock,
        )


if __name__ == "__main__":
    unittest.main()
