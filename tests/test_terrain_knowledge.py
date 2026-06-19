import unittest

import numpy as np

from mapping.terrain_knowledge import TerrainKnowledge, TerrainSnapshot


class TerrainKnowledgeTests(unittest.TestCase):
    def test_initializes_unknown_arrays_and_validates_shapes(self) -> None:
        cave = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        knowledge = TerrainKnowledge(cave)

        self.assertEqual(knowledge.roughness.shape, cave.shape)
        self.assertTrue(np.all(knowledge.roughness == -1.0))
        self.assertTrue(np.all(knowledge.confidence == 0.0))
        self.assertEqual(knowledge.floor_cells, 3)

        with self.assertRaises(ValueError):
            TerrainKnowledge(cave, roughness=np.zeros((3, 3)))

    def test_records_samples_and_rejects_walls_and_out_of_bounds(self) -> None:
        cave = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        knowledge = TerrainKnowledge(cave)

        updated = knowledge.record_samples(
            [
                (0, 0, 0.8, 0.5),
                (0, 0, 0.4, 0.5),
                (1, 0, 1.0, 1.0),
                (9, 9, 1.0, 1.0),
            ]
        )

        self.assertTrue(updated)
        self.assertAlmostEqual(float(knowledge.roughness[0, 0]), 0.6)
        self.assertAlmostEqual(float(knowledge.confidence[0, 0]), 1.0)
        self.assertEqual(float(knowledge.confidence[0, 1]), 0.0)

    def test_snapshot_is_detached_from_live_state(self) -> None:
        knowledge = TerrainKnowledge(np.zeros((2, 2), dtype=np.uint8))
        knowledge.record_samples([(0, 0, 0.5, 0.5)])

        snapshot = knowledge.snapshot()
        snapshot.roughness[0, 0] = 1.0
        snapshot.confidence[0, 0] = 0.0

        self.assertAlmostEqual(float(knowledge.roughness[0, 0]), 0.5)
        self.assertAlmostEqual(float(knowledge.confidence[0, 0]), 0.5)

    def test_merge_is_weighted_and_supports_smaller_source_extent(self) -> None:
        target = TerrainKnowledge(np.zeros((3, 3), dtype=np.uint8))
        target.record_samples([(0, 0, 0.2, 0.5)])
        source = TerrainSnapshot(
            roughness=np.array([[0.8, 0.6]], dtype=np.float32),
            confidence=np.array([[0.5, 1.0]], dtype=np.float32),
        )

        updated = target.merge_from(source)

        self.assertTrue(updated)
        self.assertAlmostEqual(float(target.roughness[0, 0]), 0.5)
        self.assertAlmostEqual(float(target.confidence[0, 0]), 1.0)
        self.assertAlmostEqual(float(target.roughness[0, 1]), 0.6)
        self.assertEqual(float(target.confidence[2, 2]), 0.0)

    def test_known_mask_and_explored_ratio_cover_floor_only(self) -> None:
        cave = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        knowledge = TerrainKnowledge(cave)
        knowledge.record_samples([(0, 0, 0.2, 0.5), (1, 1, 0.4, 0.8)])

        known = knowledge.known_mask()

        self.assertTrue(known[0, 0])
        self.assertFalse(known[0, 1])
        self.assertTrue(known[1, 1])
        self.assertAlmostEqual(knowledge.explored_ratio(), 2 / 3)

    def test_operations_are_safe_while_owner_lock_is_already_held(self) -> None:
        knowledge = TerrainKnowledge(np.zeros((2, 2), dtype=np.uint8))

        with knowledge.lock:
            knowledge.record_samples([(0, 0, 0.4, 0.5)])
            snapshot = knowledge.snapshot()

        self.assertAlmostEqual(float(snapshot.roughness[0, 0]), 0.4)
        self.assertAlmostEqual(float(snapshot.confidence[0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
