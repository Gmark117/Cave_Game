"""Pure cave-generation orchestration."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from asset_config.mapgen import MapGen, WormInputs
from generation.cave_post_processor import CavePostProcessor
from generation.terrain_roughness_generator import TerrainRoughnessGenerator
from generation.worm_process_runner import WormProcessRunner


@dataclass(frozen=True)
class CaveGenerationProgress:
    """Application callbacks for user-visible generation progress."""

    on_digging: Optional[Callable[[], None]] = None
    on_post_processing: Optional[Callable[[], None]] = None


@dataclass(frozen=True)
class CaveGenerationResult:
    """Generated cave data consumed by mission setup."""

    bin_map: np.ndarray
    terrain_roughness: np.ndarray
    worm_x: list[int]
    worm_y: list[int]
    worm_inputs: tuple[int, int, int]
    completed_workers: int
    worker_crashed: bool


def build_worm_starts(
    width: int,
    height: int,
    rng: np.random.Generator,
    count: int = MapGen.DEFAULT_NUM_PROCESSES,
) -> tuple[list[int], list[int]]:
    """Build deterministic worm starts around stable cave anchors."""

    anchor_x = [
        width / 4,
        3 * width / 4,
        3 * width / 4,
        width / 4,
        width / 2,
    ]
    anchor_y = [
        height / 4,
        height / 4,
        3 * height / 4,
        3 * height / 4,
        height / 2,
    ]
    random_count = max(0, count - len(anchor_x))
    random_x = [
        int(
            rng.integers(
                MapGen.BORDER_THICKNESS,
                width - MapGen.BORDER_THICKNESS + 1,
            )
        )
        for _ in range(random_count)
    ]
    random_y = [
        int(
            rng.integers(
                MapGen.BORDER_THICKNESS,
                height - MapGen.BORDER_THICKNESS + 1,
            )
        )
        for _ in range(random_count)
    ]

    worm_x = list(map(int, (anchor_x + random_x)[:count]))
    worm_y = list(map(int, (anchor_y + random_y)[:count]))
    return worm_x, worm_y


class CaveGenerator:
    """Generate a binary cave map and matching terrain roughness."""

    def __init__(
        self,
        width: int,
        height: int,
        seed: int,
        map_dim: str,
        num_processes: int = MapGen.DEFAULT_NUM_PROCESSES,
        runner: Optional[WormProcessRunner] = None,
        post_processor: Optional[CavePostProcessor] = None,
        roughness_generator: Optional[TerrainRoughnessGenerator] = None,
    ) -> None:
        """Store generation parameters and injectable collaborators."""
        self.width = width
        self.height = height
        self.seed = seed
        self.map_dim = map_dim
        self.num_processes = num_processes
        self.runner = runner or WormProcessRunner()
        self.post_processor = post_processor or CavePostProcessor()
        self.roughness_generator = (
            roughness_generator or TerrainRoughnessGenerator()
        )

    def generate(
        self,
        progress: Optional[CaveGenerationProgress] = None,
    ) -> CaveGenerationResult:
        """Run worm carving, post-processing, and roughness synthesis."""

        rng = np.random.default_rng(self.seed)
        worm_inputs = tuple(WormInputs[self.map_dim].value)
        # One RNG seeded here drives starts, worker target pairing, and terrain
        # roughness so a mission seed remains reproducible.
        worm_x, worm_y = build_worm_starts(
            self.width,
            self.height,
            rng,
            self.num_processes,
        )
        initial_map = np.ones([self.height, self.width])

        if progress is not None and progress.on_digging is not None:
            progress.on_digging()

        run_result = self.runner.run(
            initial_map,
            self.num_processes,
            worm_x,
            worm_y,
            worm_inputs,
            self.seed,
            rng,
        )

        if progress is not None and progress.on_post_processing is not None:
            progress.on_post_processing()

        bin_map = self.post_processor.process(
            run_result.bin_map,
            self.width,
            self.height,
            self.seed,
            worm_inputs,
        )
        terrain_roughness = self.roughness_generator.generate(bin_map, rng)
        return CaveGenerationResult(
            bin_map=bin_map,
            terrain_roughness=terrain_roughness,
            worm_x=worm_x,
            worm_y=worm_y,
            worm_inputs=worm_inputs,
            completed_workers=run_result.completed_workers,
            worker_crashed=run_result.worker_crashed,
        )
