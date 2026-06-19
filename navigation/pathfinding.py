"""Pathfinding resource lifecycle and mission-facing planning API."""

import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from typing import List, Optional, Tuple

import numpy as np

import AStarPathfinder


logger = logging.getLogger(__name__)


class PathfindingService:
    """Own drone A* workers and rover terrain-aware path planning."""

    def __init__(self, cave_map: np.ndarray, agent_count: int) -> None:
        self.cave_map = np.asarray(cave_map, dtype=np.uint8)
        self.agent_count = max(1, int(agent_count))
        self.map_shm: Optional[shared_memory.SharedMemory] = None
        self.map_shape: Optional[Tuple[int, int]] = None
        self.pool: Optional[ProcessPoolExecutor] = None
        self.pool_sem: Optional[threading.Semaphore] = None

    @property
    def ready(self) -> bool:
        """Return whether drone path requests can be submitted."""
        return (
            self.map_shm is not None
            and self.map_shape is not None
            and self.pool is not None
            and self.pool_sem is not None
        )

    def start(self) -> None:
        """Create the shared cave map and bounded A* worker pool."""
        if self.ready:
            return

        cave_map = np.ascontiguousarray(self.cave_map, dtype=np.uint8)
        map_shm: Optional[shared_memory.SharedMemory] = None
        try:
            map_shm = shared_memory.SharedMemory(
                create=True,
                size=cave_map.nbytes,
            )
            shared_map = np.ndarray(
                cave_map.shape,
                dtype=cave_map.dtype,
                buffer=map_shm.buf,
            )
            shared_map[:] = cave_map
        except (OSError, ValueError, BufferError) as exc:
            if map_shm is not None:
                self._close_shared_memory(map_shm)
            logger.warning(
                "Shared memory setup failed; process-pool pathfinding disabled: %s",
                exc,
            )
            return

        self.map_shm = map_shm
        self.map_shape = cave_map.shape

        cpu_count = os.cpu_count() or 1
        available_workers = max(1, cpu_count - 1)
        max_workers = min(self.agent_count, available_workers)
        try:
            self.pool = ProcessPoolExecutor(max_workers=max_workers)
        except (RuntimeError, ValueError, OSError):
            self.shutdown()
            raise
        self.pool_sem = threading.Semaphore(max_workers)

    def compute_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Compute a drone path using the shared-memory worker pool."""
        if not self.ready:
            return []

        map_shm = self.map_shm
        map_shape = self.map_shape
        pool = self.pool
        pool_sem = self.pool_sem
        if (
            map_shm is None
            or map_shape is None
            or pool is None
            or pool_sem is None
        ):
            return []

        acquired = False
        try:
            pool_sem.acquire()
            acquired = True
            future = pool.submit(
                AStarPathfinder.compute_path,
                map_shm.name,
                map_shape,
                start,
                goal,
            )
            return future.result()
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning(
                "Pathfinding pool request failed for %s -> %s: %s",
                start,
                goal,
                exc,
            )
            return []
        finally:
            if acquired:
                pool_sem.release()

    def compute_weighted_path(
        self,
        roughness_map: np.ndarray,
        confidence_map: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Compute a rover path using terrain roughness and confidence."""
        return AStarPathfinder.compute_weighted_path(
            self.cave_map,
            roughness_map,
            confidence_map,
            start,
            goal,
        )

    def shutdown(self) -> None:
        """Release worker and shared-memory resources if allocated."""
        if self.pool is not None:
            try:
                self.pool.shutdown(wait=True)
            except (RuntimeError, ValueError, OSError):
                pass
            finally:
                self.pool = None
                self.pool_sem = None

        if self.map_shm is not None:
            try:
                self._close_shared_memory(self.map_shm)
            finally:
                self.map_shm = None
                self.map_shape = None

    @staticmethod
    def _close_shared_memory(
        map_shm: shared_memory.SharedMemory,
    ) -> None:
        """Close and unlink one shared-memory allocation."""
        try:
            map_shm.close()
        except (BufferError, OSError):
            pass
        try:
            map_shm.unlink()
        except FileNotFoundError:
            pass
        except (BufferError, OSError):
            pass
