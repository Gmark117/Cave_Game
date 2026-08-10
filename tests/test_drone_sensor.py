import os
import unittest
import math
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from agents.drone import Drone
from config.simulation_config import MissionConfig, SimulationConfig, SlamConfig
from mapping.drone_sensor import LIDAR_RANGE_RADIUS_MULTIPLIER
from mapping.localization import PoseEstimate
from mapping.slam_map import OCCUPIED, UNKNOWN, SlamSnapshot
from mapping.terrain_knowledge import TerrainSnapshot
from mapping.vision_sensor import VisionScan
from navigation.navigation_intent import MovementMode, NavigationIntent


class RecordingControl:
    delay = 1 / 15
    terrain_roughness = np.full((64, 64), 0.4, dtype=np.float32)

    def __init__(self) -> None:
        self.samples = []
        self.runtime_trace = RecordingTrace()
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


class RecordingTrace:
    def __init__(self) -> None:
        self.events = []

    def record(self, event, **fields) -> None:
        self.events.append((event, fields))


class DroneSensorTests(unittest.TestCase):
    def setUp(self) -> None:
        settings = SimulationConfig(
            mission_config=MissionConfig(map_dim="LARGE"),
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

    def test_sensor_scan_trace_reports_progress_delta_and_cumulative_gain(
        self,
    ) -> None:
        self.drone.update_sensors()

        event = next(
            fields
            for name, fields in self.control.runtime_trace.events
            if name == "sensor_scan"
        )
        progress = self.drone.slam_map.progress_snapshot()

        self.assertEqual(event["completed_scan_sequence"], 1)
        self.assertGreater(event["newly_known_cells"], 0)
        self.assertGreater(event["confidence_gain"], 0.0)
        self.assertGreater(event["visible_cell_count"], event["ray_count"])
        self.assertEqual(
            event["visible_cell_count"],
            event["visible_free_cells"] + event["visible_occupied_cells"],
        )
        self.assertEqual(
            event["cumulative_sensor_newly_known_cells"],
            progress.sensor_newly_known_cells,
        )
        self.assertAlmostEqual(
            event["cumulative_sensor_confidence_gain"],
            progress.sensor_confidence_gain,
        )

    def test_zero_gain_sensor_scan_is_traced_as_completed(self) -> None:
        self.drone.sensor_controller.vision_sensor.scan_cone = (
            lambda origin, heading: VisionScan((), (), ())
        )

        self.drone.update_sensors()

        event = next(
            fields
            for name, fields in self.control.runtime_trace.events
            if name == "sensor_scan"
        )
        progress = self.drone.slam_map.progress_snapshot()

        self.assertFalse(event["slam_updated"])
        self.assertEqual(event["completed_scan_sequence"], 1)
        self.assertEqual(event["newly_known_cells"], 0)
        self.assertEqual(event["confidence_gain"], 0.0)
        self.assertEqual(progress.completed_scan_sequence, 1)

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

    def test_live_sensor_range_is_capped_by_drone_radius_multiplier(self) -> None:
        sensor = self.drone.sensor_controller.vision_sensor

        self.assertEqual(
            sensor.max_range,
            self.drone.radius * LIDAR_RANGE_RADIUS_MULTIPLIER,
        )
        self.assertLess(sensor.max_range, int(math.hypot(64, 64)))

    def test_sensor_update_uses_localizer_pose_as_slam_origin(self) -> None:
        class FakeLocalizer:
            def estimate(self, runtime_snapshot, timestamp):
                self.runtime_snapshot = runtime_snapshot
                self.timestamp = timestamp
                return PoseEstimate(
                    position=(40, 32),
                    heading_deg=90.0,
                    confidence=0.5,
                    source="fake",
                    timestamp=timestamp,
                )

        fake_localizer = FakeLocalizer()
        self.drone.localizer = fake_localizer
        recorded = {}
        original_update = self.drone.slam_map.update_from_observations

        def record_slam_origin(origin, *, free_cells, occupied_cells):
            recorded["slam_origin"] = origin
            return original_update(
                origin,
                free_cells=free_cells,
                occupied_cells=occupied_cells,
            )

        def record_terrain_origin(origin, ray_hits, step=2):
            recorded["terrain_origin"] = origin
            recorded["terrain_step"] = step
            return []

        self.drone.slam_map.update_from_observations = record_slam_origin
        self.drone.sensor_controller.roughness_sampler.sample_from_rays = (
            record_terrain_origin
        )

        self.drone.update_sensors()

        self.assertEqual(recorded["slam_origin"], (40, 32))
        self.assertEqual(recorded["terrain_origin"], (40, 32))
        self.assertEqual(recorded["terrain_step"], 2)
        self.assertEqual(self.drone.snapshot().position, (32, 32))
        self.assertEqual(
            self.drone.sensor_controller.latest_pose_estimate.source,
            "fake",
        )
        self.assertEqual(fake_localizer.runtime_snapshot.position, (32, 32))

    def test_sensor_update_skips_unchanged_pose(self) -> None:
        sensor = self.drone.sensor_controller.vision_sensor
        original_cast = sensor.cast_cone
        calls = []

        def record_cast(origin, heading):
            calls.append((origin, heading))
            return original_cast(origin, heading)

        sensor.cast_cone = record_cast

        self.drone.update_sensors()
        first_version = self.drone.slam_map.version
        self.drone.update_sensors()

        self.assertEqual(len(calls), 1)
        self.assertEqual(self.drone.slam_map.version, first_version)

        self.drone.runtime_state.begin_exploration(90)
        self.drone.update_sensors()

        self.assertEqual(len(calls), 2)

    def test_scan_intent_admits_one_unchanged_pose_sensor_sequence(self) -> None:
        self.drone.update_sensors()
        progress = self.drone.slam_map.progress_snapshot()
        self.drone.runtime_state.set_navigation_intent(NavigationIntent(
            mode=MovementMode.SCAN,
            target=self.drone.snapshot().position,
            scan_sequence=progress.completed_scan_sequence,
        ))

        self.drone.update_sensors()
        admitted = self.drone.slam_map.progress_snapshot()
        self.drone.update_sensors()
        suppressed = self.drone.slam_map.progress_snapshot()

        self.assertEqual(
            admitted.completed_scan_sequence,
            progress.completed_scan_sequence + 1,
        )
        self.assertEqual(
            suppressed.completed_scan_sequence,
            admitted.completed_scan_sequence,
        )

    def test_pending_local_scan_admits_one_unchanged_pose_sequence(self) -> None:
        self.drone.update_sensors()
        progress = self.drone.slam_map.progress_snapshot()
        self.drone.runtime_state.set_navigation_intent(NavigationIntent(
            mode=MovementMode.TRAVEL,
            target=(40, 32),
            scan_sequence=progress.completed_scan_sequence,
            local_scan_pending=True,
        ))

        self.drone.update_sensors()
        admitted = self.drone.slam_map.progress_snapshot()
        self.drone.update_sensors()
        suppressed = self.drone.slam_map.progress_snapshot()

        self.assertEqual(
            admitted.completed_scan_sequence,
            progress.completed_scan_sequence + 1,
        )
        self.assertEqual(
            suppressed.completed_scan_sequence,
            admitted.completed_scan_sequence,
        )


if __name__ == "__main__":
    unittest.main()
