import os
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from Drone import Drone
from SimulationConfig import MissionConfig, SimulationConfig, SlamConfig
from SlamMap import OCCUPIED, UNKNOWN, SlamSnapshot
from mapping.terrain_knowledge import TerrainSnapshot


class RecordingControl:
    delay = 1 / 15
    terrain_roughness = np.full((64, 64), 0.4, dtype=np.float32)

    def __init__(self) -> None:
        self.samples = []
        self.terrain_fusion = SimpleNamespace(
            record_scan=lambda samples: self.samples.extend(samples),
        )

    def compute_path(self, start, goal):
        return []

    def simulation_time(self) -> float:
        return 1.0

    def pause_checkpoint(self) -> bool:
        return True

    def wait_simulation_delay(self, duration: float) -> bool:
        return True


class DroneSensorTests(unittest.TestCase):
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
        self.drone.runtime_state.set_overlay_visibility(
            show_path=True,
            show_vision=False,
        )
        self.drone.update_sensors()
        snapshot = self.drone.slam_map.snapshot()
        runtime_snapshot = self.drone.snapshot()

        self.assertTrue(runtime_snapshot.ray_points)
        self.assertGreater(
            int(np.count_nonzero(snapshot.confidence)),
            0,
        )
        self.assertTrue(self.control.samples)

        occupancy_before = snapshot.occupancy
        confidence_before = snapshot.confidence
        roughness_before = self.drone.terrain_knowledge.roughness.copy()

        self.drone.runtime_state.set_overlay_visibility(
            show_path=True,
            show_vision=True,
        )
        self.drone.renderer.draw_vision_overlay(self.drone.snapshot())

        after_draw = self.drone.slam_map.snapshot()
        np.testing.assert_array_equal(
            occupancy_before,
            after_draw.occupancy,
        )
        np.testing.assert_array_equal(
            confidence_before,
            after_draw.confidence,
        )
        np.testing.assert_array_equal(
            roughness_before,
            self.drone.terrain_knowledge.roughness,
        )

    def test_terrain_merge_is_confidence_weighted_and_ignores_walls(self) -> None:
        self.drone.terrain_knowledge.floor_mask[0, 1] = False
        self.drone.terrain_knowledge.roughness[0, 0] = 0.2
        self.drone.terrain_knowledge.confidence[0, 0] = 0.5
        source_roughness = np.full((64, 64), -1.0, dtype=np.float32)
        source_confidence = np.zeros((64, 64), dtype=np.float32)
        source_roughness[0, 0] = 0.8
        source_confidence[0, 0] = 0.5
        source_roughness[0, 1] = 1.0
        source_confidence[0, 1] = 1.0

        self.drone.terrain_knowledge.merge_from(
            TerrainSnapshot(source_roughness, source_confidence),
        )

        self.assertAlmostEqual(
            float(self.drone.terrain_knowledge.roughness[0, 0]),
            0.5,
        )
        self.assertAlmostEqual(
            float(self.drone.terrain_knowledge.confidence[0, 0]),
            1.0,
        )
        self.assertEqual(
            float(self.drone.terrain_knowledge.confidence[0, 1]),
            0.0,
        )

    def test_slam_merge_uses_other_map_when_confidence_is_higher(self) -> None:
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[2, 2] = OCCUPIED
        confidence[2, 2] = 0.9

        self.drone.slam_map.merge_from(
            SlamSnapshot(occupancy, confidence)
        )
        snapshot = self.drone.slam_map.snapshot()

        self.assertEqual(int(snapshot.occupancy[2, 2]), OCCUPIED)
        self.assertAlmostEqual(
            float(snapshot.confidence[2, 2]),
            0.9,
        )

    def test_removed_terrain_aliases_do_not_shadow_owned_model(self) -> None:
        self.assertFalse(hasattr(self.drone, "known_roughness"))
        self.assertFalse(hasattr(self.drone, "terrain_confidence"))
        self.assertFalse(hasattr(self.drone, "terrain_lock"))


if __name__ == "__main__":
    unittest.main()
