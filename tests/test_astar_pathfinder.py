import unittest
from multiprocessing import shared_memory

import numpy as np

import AStarPathfinder


class AStarPathfinderTests(unittest.TestCase):
    def test_shared_memory_astar_finds_wall_avoiding_path(self) -> None:
        cave = np.zeros((5, 5), dtype=np.uint8)
        cave[2, :4] = 1
        cave[2, 2] = 0
        shm = shared_memory.SharedMemory(create=True, size=cave.nbytes)
        try:
            shared_map = np.ndarray(cave.shape, dtype=np.uint8, buffer=shm.buf)
            shared_map[:] = cave

            path = AStarPathfinder.compute_path(
                shm.name,
                cave.shape,
                (0, 0),
                (4, 4),
            )
        finally:
            shm.close()
            shm.unlink()

        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))
        self.assertTrue(all(cave[y, x] == 0 for x, y in path))

    def test_shared_memory_astar_rejects_blocked_or_out_of_bounds_goal(self) -> None:
        cave = np.zeros((2, 2), dtype=np.uint8)
        cave[1, 1] = 1
        shm = shared_memory.SharedMemory(create=True, size=cave.nbytes)
        try:
            shared_map = np.ndarray(cave.shape, dtype=np.uint8, buffer=shm.buf)
            shared_map[:] = cave

            blocked = AStarPathfinder.compute_path(
                shm.name,
                cave.shape,
                (0, 0),
                (1, 1),
            )
            outside = AStarPathfinder.compute_path(
                shm.name,
                cave.shape,
                (0, 0),
                (3, 3),
            )
        finally:
            shm.close()
            shm.unlink()

        self.assertEqual(blocked, [])
        self.assertEqual(outside, [])

    def test_weighted_astar_avoids_rough_direct_route(self) -> None:
        cave = np.zeros((5, 5), dtype=np.uint8)
        roughness = np.zeros((5, 5), dtype=np.float32)
        confidence = np.ones((5, 5), dtype=np.float32)
        roughness[2, 1:4] = 1.0

        path = AStarPathfinder.compute_weighted_path(
            cave,
            roughness,
            confidence,
            (0, 2),
            (4, 2),
        )

        self.assertEqual(path[0], (0, 2))
        self.assertEqual(path[-1], (4, 2))
        self.assertNotIn((2, 2), path)


if __name__ == "__main__":
    unittest.main()
