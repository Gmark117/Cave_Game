import os
import unittest

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from asset_config.mapgen import MapGen, WormInputs
from generation.cave_generator import (
    CaveGenerationProgress,
    CaveGenerator,
    build_worm_starts,
)
from generation.worm_process_runner import WormRunResult


class FakeRunner:
    def __init__(self, bin_map: np.ndarray) -> None:
        self.bin_map = bin_map
        self.calls = []

    def run(self, *args):
        self.calls.append(args)
        return WormRunResult(
            bin_map=self.bin_map,
            completed_workers=2,
            worker_crashed=False,
        )


class FakePostProcessor:
    def __init__(self, bin_map: np.ndarray) -> None:
        self.bin_map = bin_map
        self.calls = []

    def process(self, *args):
        self.calls.append(args)
        return self.bin_map


class FakeRoughnessGenerator:
    def __init__(self, roughness: np.ndarray) -> None:
        self.roughness = roughness
        self.calls = []

    def generate(self, *args):
        self.calls.append(args)
        return self.roughness


class CaveGeneratorTests(unittest.TestCase):
    def test_worm_starts_are_deterministic_and_within_bounds(self) -> None:
        first_rng = np.random.default_rng(5)
        second_rng = np.random.default_rng(5)

        first_x, first_y = build_worm_starts(200, 180, first_rng)
        second_x, second_y = build_worm_starts(200, 180, second_rng)

        self.assertEqual(first_x, second_x)
        self.assertEqual(first_y, second_y)
        self.assertEqual(len(first_x), MapGen.DEFAULT_NUM_PROCESSES)
        self.assertTrue(all(0 <= x < 200 for x in first_x))
        self.assertTrue(all(0 <= y < 180 for y in first_y))

    def test_generate_coordinates_services_and_progress(self) -> None:
        raw_map = np.zeros((3, 4), dtype=np.uint8)
        processed_map = np.ones((3, 4), dtype=np.uint8)
        roughness = np.full((3, 4), 0.5, dtype=np.float32)
        runner = FakeRunner(raw_map)
        post_processor = FakePostProcessor(processed_map)
        roughness_generator = FakeRoughnessGenerator(roughness)
        events = []
        generator = CaveGenerator(
            width=4,
            height=3,
            seed=11,
            map_dim="SMALL",
            num_processes=2,
            runner=runner,
            post_processor=post_processor,
            roughness_generator=roughness_generator,
        )

        result = generator.generate(
            CaveGenerationProgress(
                on_digging=lambda: events.append("digging"),
                on_post_processing=lambda: events.append("post"),
            )
        )

        self.assertEqual(events, ["digging", "post"])
        self.assertIs(result.bin_map, processed_map)
        self.assertIs(result.terrain_roughness, roughness)
        self.assertEqual(result.completed_workers, 2)
        self.assertFalse(result.worker_crashed)
        runner_args = runner.calls[0]
        self.assertEqual(runner_args[1], 2)
        self.assertEqual(runner_args[4], tuple(WormInputs.SMALL.value))
        self.assertEqual(runner_args[5], 11)
        post_args = post_processor.calls[0]
        self.assertIs(post_args[0], raw_map)
        self.assertEqual(post_args[1:4], (4, 3, 11))
        self.assertEqual(post_args[4], tuple(WormInputs.SMALL.value))
        self.assertIs(roughness_generator.calls[0][0], processed_map)


if __name__ == "__main__":
    unittest.main()
