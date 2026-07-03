"""Multiprocessing runner for cave-carving worms."""

from dataclasses import dataclass
import os
from typing import Callable, Optional

import numpy as np

from MapGenHelpers import (
    make_derangement,
    monitor_worms,
    safe_shm_close,
    safe_shm_create,
    start_worms,
)


@dataclass(frozen=True)
class WormRunResult:
    """Result copied out of shared-memory worm carving."""

    bin_map: np.ndarray
    completed_workers: int
    worker_crashed: bool


class WormProcessRunner:
    """Own shared-memory allocation, workers, monitoring, and cleanup."""

    def run(
        self,
        initial_map: np.ndarray,
        requested_workers: int,
        worm_x: list[int],
        worm_y: list[int],
        worm_inputs: tuple[int, int, int],
        seed: int,
        rng: np.random.Generator,
        worker_finished: Optional[Callable[[], None]] = None,
    ) -> WormRunResult:
        """Carve `initial_map` with worker processes and return a copied map."""

        num_cpus = os.cpu_count() or 1
        worker_count = min(requested_workers, num_cpus)
        completed_workers = 0
        shm = None
        try:
            init_map = initial_map.astype(np.uint8)
            shm, shm_arr = safe_shm_create(init_map)
            targets = make_derangement(worker_count, rng)
            proc_list = start_worms(
                shm.name,
                worker_count,
                worm_x,
                worm_y,
                worm_inputs,
                seed,
                targets,
                init_map.shape[0],
                init_map.shape[1],
            )

            def _update_finished() -> None:
                nonlocal completed_workers
                completed_workers += 1
                if worker_finished is not None:
                    worker_finished()

            worker_crashed = monitor_worms(proc_list, _update_finished)
            if worker_crashed:
                print("MapGenerator: one or more worms crashed during generation")
            return WormRunResult(
                bin_map=np.array(shm_arr, dtype=np.uint8),
                completed_workers=completed_workers,
                worker_crashed=worker_crashed,
            )
        finally:
            try:
                safe_shm_close(shm)
            except (AttributeError, OSError):
                pass
