import unittest
from types import SimpleNamespace

import numpy as np

from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from mapping.wall_mapping import exposed_wall_mask, wall_mapping_snapshot


class WallMappingTests(unittest.TestCase):
    def test_target_contains_only_wall_pixels_exposed_to_cave_floor(self) -> None:
        cave = np.ones((7, 7), dtype=np.uint8)
        cave[1:6, 1:6] = 0
        cave[3, 3] = 1

        target = exposed_wall_mask(cave)

        self.assertTrue(target[0, 3])
        self.assertTrue(target[3, 3])
        self.assertFalse(target[6, 6])
        self.assertFalse(target[2, 2])

    def test_progress_combines_occupied_slam_without_using_terrain(self) -> None:
        cave = np.ones((5, 5), dtype=np.uint8)
        cave[1:4, 1:4] = 0
        target = exposed_wall_mask(cave)
        targets = tuple((int(x), int(y)) for y, x in zip(*np.where(target)))
        split = len(targets) // 2

        def drone_with(points: tuple[tuple[int, int], ...]):
            occupancy = np.full(cave.shape, UNKNOWN, dtype=np.int8)
            confidence = np.zeros(cave.shape, dtype=np.float32)
            for x, y in points:
                occupancy[y, x] = OCCUPIED
                confidence[y, x] = 0.9
            # Confident free evidence and arbitrary roughness are irrelevant to
            # wall completion.
            occupancy[2, 2] = FREE
            confidence[2, 2] = 1.0
            return SimpleNamespace(
                slam_map=SimpleNamespace(
                    snapshot=lambda point_limit=0: SlamSnapshot(
                        occupancy, confidence, version=4
                    )
                ),
                terrain_knowledge=SimpleNamespace(
                    explored_ratio=lambda: 1.0,
                ),
            )

        progress = wall_mapping_snapshot(
            cave,
            (drone_with(targets[:split]), drone_with(targets[split:])),
            confidence_threshold=0.6,
        )

        self.assertEqual(progress.mapped_wall_pixels, progress.total_wall_pixels)
        self.assertEqual(progress.ratio, 1.0)
        self.assertTrue(progress.complete)


if __name__ == "__main__":
    unittest.main()
