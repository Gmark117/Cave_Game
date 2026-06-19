import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import AStarPathfinder
from navigation.pathfinding import PathfindingService


class PathfindingServiceTests(unittest.TestCase):
    def test_construction_does_not_allocate_runtime_resources(self) -> None:
        cave_map = np.zeros((4, 5), dtype=np.uint8)

        with patch(
            "navigation.pathfinding.shared_memory.SharedMemory"
        ) as shared_memory:
            with patch(
                "navigation.pathfinding.ProcessPoolExecutor"
            ) as process_pool:
                service = PathfindingService(cave_map, agent_count=3)

        shared_memory.assert_not_called()
        process_pool.assert_not_called()
        self.assertFalse(service.ready)
        self.assertEqual(service.compute_path((0, 0), (1, 1)), [])

    def test_start_submit_and_shutdown_own_runtime_resources(self) -> None:
        cave_map = np.array(
            [
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
        map_shm = SimpleNamespace(
            name="mission-map",
            buf=bytearray(cave_map.nbytes),
            close=Mock(),
            unlink=Mock(),
        )
        future = SimpleNamespace(result=Mock(return_value=[(0, 0), (1, 1)]))
        pool = SimpleNamespace(
            submit=Mock(return_value=future),
            shutdown=Mock(),
        )
        service = PathfindingService(cave_map, agent_count=3)

        with patch(
            "navigation.pathfinding.shared_memory.SharedMemory",
            return_value=map_shm,
        ) as shared_memory:
            with patch(
                "navigation.pathfinding.ProcessPoolExecutor",
                return_value=pool,
            ) as process_pool:
                with patch(
                    "navigation.pathfinding.os.cpu_count",
                    return_value=8,
                ):
                    service.start()
                    service.start()

        shared_memory.assert_called_once_with(
            create=True,
            size=cave_map.nbytes,
        )
        process_pool.assert_called_once_with(max_workers=3)
        np.testing.assert_array_equal(
            np.ndarray(
                cave_map.shape,
                dtype=np.uint8,
                buffer=map_shm.buf,
            ),
            cave_map,
        )
        self.assertTrue(service.ready)

        semaphore = Mock()
        service.pool_sem = semaphore
        result = service.compute_path((0, 0), (1, 1))

        self.assertEqual(result, [(0, 0), (1, 1)])
        pool.submit.assert_called_once_with(
            AStarPathfinder.compute_path,
            "mission-map",
            cave_map.shape,
            (0, 0),
            (1, 1),
        )
        semaphore.acquire.assert_called_once_with()
        semaphore.release.assert_called_once_with()

        service.shutdown()
        service.shutdown()

        pool.shutdown.assert_called_once_with(wait=True)
        map_shm.close.assert_called_once_with()
        map_shm.unlink.assert_called_once_with()
        self.assertFalse(service.ready)

    def test_shared_memory_failure_disables_worker_pool(self) -> None:
        service = PathfindingService(
            np.zeros((2, 2), dtype=np.uint8),
            agent_count=2,
        )

        with self.assertLogs("navigation.pathfinding", level="WARNING"):
            with patch(
                "navigation.pathfinding.shared_memory.SharedMemory",
                side_effect=OSError("unavailable"),
            ):
                with patch(
                    "navigation.pathfinding.ProcessPoolExecutor"
                ) as process_pool:
                    service.start()

        process_pool.assert_not_called()
        self.assertFalse(service.ready)
        self.assertEqual(service.compute_path((0, 0), (1, 1)), [])

    def test_weighted_path_delegates_with_service_cave_map(self) -> None:
        cave_map = np.zeros((3, 3), dtype=np.uint8)
        roughness = np.full((3, 3), 0.25, dtype=np.float32)
        confidence = np.ones((3, 3), dtype=np.float32)
        service = PathfindingService(cave_map, agent_count=1)

        with patch(
            "navigation.pathfinding.AStarPathfinder.compute_weighted_path",
            return_value=[(0, 0), (1, 1)],
        ) as compute_weighted_path:
            result = service.compute_weighted_path(
                roughness,
                confidence,
                (0, 0),
                (1, 1),
            )

        self.assertEqual(result, [(0, 0), (1, 1)])
        args = compute_weighted_path.call_args.args
        np.testing.assert_array_equal(args[0], cave_map)
        self.assertIs(args[1], roughness)
        self.assertIs(args[2], confidence)
        self.assertEqual(args[3:], ((0, 0), (1, 1)))


if __name__ == "__main__":
    unittest.main()
