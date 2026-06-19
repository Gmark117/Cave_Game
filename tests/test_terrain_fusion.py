import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from mapping.terrain_fusion import TerrainFusionService, fuse_terrain_samples
from mapping.terrain_knowledge import TerrainKnowledge


class TerrainFusionTests(unittest.TestCase):
    def test_fuses_floor_samples_and_ignores_walls(self) -> None:
        roughness = np.full((2, 2), -1.0, dtype=np.float32)
        confidence = np.zeros((2, 2), dtype=np.float32)
        cave = np.array([[0, 1], [0, 0]], dtype=np.uint8)

        updated = fuse_terrain_samples(
            roughness,
            confidence,
            cave,
            [
                (0, 0, 0.8, 0.5),
                (0, 0, 0.4, 0.5),
                (1, 0, 1.0, 1.0),
            ],
        )

        self.assertTrue(updated)
        self.assertAlmostEqual(float(roughness[0, 0]), 0.6)
        self.assertEqual(float(confidence[0, 0]), 1.0)
        self.assertEqual(float(confidence[0, 1]), 0.0)

    def test_service_updates_progress_and_marks_heatmap_dirty(self) -> None:
        cave = np.zeros((2, 2), dtype=np.uint8)
        control = SimpleNamespace(
            terrain_knowledge=TerrainKnowledge(cave),
            map_matrix=cave,
            last_explored_update=0.0,
            explored_update_interval=0.5,
            floor_cells=4,
            control_center=SimpleNamespace(explored_percent=0),
            presentation=SimpleNamespace(terrain_heatmap_dirty=False),
        )

        with patch(
            "mapping.terrain_fusion.pygame.time.get_ticks",
            return_value=1000,
        ):
            TerrainFusionService(control).record_scan(
                [(0, 0, 0.5, 0.5), (1, 0, 0.8, 0.5)]
            )

        self.assertEqual(control.control_center.explored_percent, 50)
        self.assertEqual(control.last_explored_update, 1.0)
        self.assertTrue(control.presentation.terrain_heatmap_dirty)

    def test_telemetry_fusion_does_not_mutate_agent_local_knowledge(self) -> None:
        cave = np.zeros((2, 2), dtype=np.uint8)
        drone_knowledge = TerrainKnowledge(cave)
        rover_knowledge = TerrainKnowledge(cave)
        control = SimpleNamespace(
            terrain_knowledge=TerrainKnowledge(cave),
            last_explored_update=0.0,
            explored_update_interval=0.5,
            control_center=SimpleNamespace(explored_percent=0),
            presentation=SimpleNamespace(terrain_heatmap_dirty=False),
        )

        with patch(
            "mapping.terrain_fusion.pygame.time.get_ticks",
            return_value=1000,
        ):
            TerrainFusionService(control).record_scan(
                [(0, 0, 0.8, 0.75)]
            )

        self.assertAlmostEqual(
            float(control.terrain_knowledge.confidence[0, 0]),
            0.75,
        )
        self.assertEqual(float(drone_knowledge.confidence[0, 0]), 0.0)
        self.assertEqual(float(rover_knowledge.confidence[0, 0]), 0.0)
        self.assertEqual(float(drone_knowledge.roughness[0, 0]), -1.0)
        self.assertEqual(float(rover_knowledge.roughness[0, 0]), -1.0)


if __name__ == "__main__":
    unittest.main()
