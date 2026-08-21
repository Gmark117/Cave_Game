import unittest
from multiprocessing import shared_memory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from navigation import astar_pathfinder


class AStarPathfinderTests(unittest.TestCase):
    def _compute_shared_path(self, cave, start, goal):
        shm = shared_memory.SharedMemory(create=True, size=cave.nbytes)
        try:
            shared_map = np.ndarray(cave.shape, dtype=np.uint8, buffer=shm.buf)
            shared_map[:] = cave

            return astar_pathfinder.compute_path(
                shm.name,
                cave.shape,
                start,
                goal,
            )
        finally:
            shm.close()
            shm.unlink()

    def _compute_shared_result(self, cave, start, goal, max_iters):
        shm = shared_memory.SharedMemory(create=True, size=cave.nbytes)
        try:
            shared_map = np.ndarray(cave.shape, dtype=np.uint8, buffer=shm.buf)
            shared_map[:] = cave
            return astar_pathfinder.compute_path_segment(
                shm.name,
                cave.shape,
                start,
                goal,
                max_iters=max_iters,
            )
        finally:
            shm.close()
            shm.unlink()

    def test_shared_memory_astar_finds_wall_avoiding_path(self) -> None:
        cave = np.zeros((5, 5), dtype=np.uint8)
        cave[2, :4] = 1
        cave[2, 2] = 0

        path = self._compute_shared_path(cave, (0, 0), (4, 4))

        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))
        self.assertTrue(all(cave[y, x] == 0 for x, y in path))

    def test_capped_astar_returns_a_goal_directed_partial_segment(self) -> None:
        cave = np.zeros((20, 20), dtype=np.uint8)

        result = self._compute_shared_result(
            cave,
            (0, 0),
            (19, 19),
            max_iters=3,
        )

        self.assertEqual(result.status, astar_pathfinder.PATH_PARTIAL_LIMIT)
        self.assertEqual(result.path[0], (0, 0))
        self.assertNotEqual(result.path[-1], (19, 19))
        self.assertLess(
            result.remaining_distance,
            astar_pathfinder.DIAG_COST * 19,
        )
        self.assertEqual(result.iterations, 3)

    def test_legacy_astar_api_does_not_expose_a_partial_as_complete(self) -> None:
        cave = np.zeros((20, 20), dtype=np.uint8)
        shm = shared_memory.SharedMemory(create=True, size=cave.nbytes)
        try:
            shared_map = np.ndarray(cave.shape, dtype=np.uint8, buffer=shm.buf)
            shared_map[:] = cave
            path = astar_pathfinder.compute_path(
                shm.name,
                cave.shape,
                (0, 0),
                (19, 19),
                max_iters=3,
            )
        finally:
            shm.close()
            shm.unlink()

        self.assertEqual(path, [])

    def test_shared_memory_astar_rejects_blocked_or_out_of_bounds_goal(self) -> None:
        cave = np.zeros((2, 2), dtype=np.uint8)
        cave[1, 1] = 1
        shm = shared_memory.SharedMemory(create=True, size=cave.nbytes)
        try:
            shared_map = np.ndarray(cave.shape, dtype=np.uint8, buffer=shm.buf)
            shared_map[:] = cave

            blocked = astar_pathfinder.compute_path(
                shm.name,
                cave.shape,
                (0, 0),
                (1, 1),
            )
            outside = astar_pathfinder.compute_path(
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

    def test_structured_astar_distinguishes_invalid_and_unreachable(self) -> None:
        blocked_goal = np.zeros((3, 3), dtype=np.uint8)
        blocked_goal[2, 2] = 1
        disconnected = np.zeros((3, 3), dtype=np.uint8)
        disconnected[1, :] = 1

        invalid = self._compute_shared_result(
            blocked_goal,
            (0, 0),
            (2, 2),
            max_iters=100,
        )
        unreachable = self._compute_shared_result(
            disconnected,
            (0, 0),
            (2, 2),
            max_iters=100,
        )

        self.assertEqual(
            invalid.status,
            astar_pathfinder.PATH_INVALID_ENDPOINT,
        )
        self.assertEqual(
            unreachable.status,
            astar_pathfinder.PATH_UNREACHABLE,
        )

    def test_shared_memory_astar_applies_diagonal_corner_rule(self) -> None:
        tight_corner = np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=np.uint8,
        )
        one_open_side = np.array(
            [
                [0, 0],
                [1, 0],
            ],
            dtype=np.uint8,
        )

        self.assertEqual(
            self._compute_shared_path(tight_corner, (0, 0), (1, 1)),
            [],
        )
        self.assertEqual(
            self._compute_shared_path(one_open_side, (0, 0), (1, 1)),
            [(0, 0), (1, 1)],
        )

    def test_shared_memory_astar_closes_worker_attachment(self) -> None:
        cave = np.zeros((2, 2), dtype=np.uint8)
        shared_buffer = bytearray(cave.nbytes)
        shared_map = np.ndarray(cave.shape, dtype=np.uint8, buffer=shared_buffer)
        shared_map[:] = cave
        worker_shm = SimpleNamespace(buf=shared_buffer, close=Mock())

        with patch(
            "navigation.astar_pathfinder.shared_memory.SharedMemory",
            return_value=worker_shm,
        ):
            path = astar_pathfinder.compute_path(
                "mission-map",
                cave.shape,
                (0, 0),
                (1, 1),
            )

        self.assertEqual(path, [(0, 0), (1, 1)])
        worker_shm.close.assert_called_once_with()

    def test_weighted_astar_matches_unweighted_when_costs_are_neutral(self) -> None:
        cave = np.zeros((5, 5), dtype=np.uint8)
        cave[2, :4] = 1
        cave[2, 2] = 0
        roughness = np.zeros((5, 5), dtype=np.float32)
        confidence = np.ones((5, 5), dtype=np.float32)

        shared_path = self._compute_shared_path(cave, (0, 0), (4, 4))
        weighted_path = astar_pathfinder.compute_weighted_path(
            cave,
            roughness,
            confidence,
            (0, 0),
            (4, 4),
            roughness_weight=0.0,
            unknown_penalty=0.0,
            low_confidence_penalty=0.0,
        )

        self.assertEqual(weighted_path, shared_path)

    def test_weighted_astar_avoids_rough_direct_route(self) -> None:
        cave = np.zeros((5, 5), dtype=np.uint8)
        roughness = np.zeros((5, 5), dtype=np.float32)
        confidence = np.ones((5, 5), dtype=np.float32)
        roughness[2, 1:4] = 1.0

        path = astar_pathfinder.compute_weighted_path(
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
