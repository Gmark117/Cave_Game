import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import MapGenerator as map_generator_module
from MapGenerator import MapGenerator, _image_path_from_key
from asset_config.media import Images


class MapGeneratorTests(unittest.TestCase):
    def test_image_key_resolution_is_case_insensitive_and_validated(self) -> None:
        self.assertEqual(
            _image_path_from_key("cave_floor"),
            Images.CAVE_FLOOR.value,
        )
        with self.assertRaisesRegex(KeyError, "Unknown image key"):
            _image_path_from_key("missing")

    def test_set_starts_is_deterministic_and_within_bounds(self) -> None:
        first = object.__new__(MapGenerator)
        first.width = 200
        first.height = 180
        first.rng = np.random.default_rng(5)
        second = object.__new__(MapGenerator)
        second.width = 200
        second.height = 180
        second.rng = np.random.default_rng(5)

        first.set_starts()
        second.set_starts()

        self.assertEqual(first.worm_x, second.worm_x)
        self.assertEqual(first.worm_y, second.worm_y)
        self.assertEqual(len(first.worm_x), MapGenerator.NUM_PROCESSES)
        self.assertTrue(all(0 <= x < first.width for x in first.worm_x))
        self.assertTrue(all(0 <= y < first.height for y in first.worm_y))

    def test_terrain_roughness_is_floor_only_bounded_and_reproducible(self) -> None:
        cave = np.ones((8, 8), dtype=np.uint8)
        cave[1:7, 1:7] = 0

        first = object.__new__(MapGenerator)
        first.width = 8
        first.height = 8
        first.bin_map = cave
        first.rng = np.random.default_rng(9)
        second = object.__new__(MapGenerator)
        second.width = 8
        second.height = 8
        second.bin_map = cave.copy()
        second.rng = np.random.default_rng(9)

        first.build_terrain_roughness()
        second.build_terrain_roughness()

        np.testing.assert_allclose(
            first.terrain_roughness,
            second.terrain_roughness,
        )
        self.assertTrue(np.all(first.terrain_roughness[cave == 1] == 0.0))
        self.assertGreaterEqual(float(first.terrain_roughness.min()), 0.0)
        self.assertLessEqual(float(first.terrain_roughness.max()), 1.0)

    def test_dig_map_delegates_workers_copies_result_and_cleans_up(self) -> None:
        generator = object.__new__(MapGenerator)
        generator.bin_map = np.ones((3, 4), dtype=np.float32)
        generator.height = 3
        generator.width = 4
        generator.worm_x = [1, 2]
        generator.worm_y = [1, 2]
        generator.worm_inputs = (3, 4, 5)
        generator.settings = SimpleNamespace(seed=7)
        generator.rng = np.random.default_rng(7)
        generator.proc_counter = 0
        generator.game = SimpleNamespace(
            menu=SimpleNamespace(blit_loading=Mock()),
        )
        shm = SimpleNamespace(name="test-map")
        shared_result = np.zeros((3, 4), dtype=np.uint8)

        def monitor(_processes, callback):
            callback()
            callback()
            return False

        with patch("MapGenerator.os.cpu_count", return_value=2):
            with patch(
                "MapGenerator.safe_shm_create",
                return_value=(shm, shared_result),
            ):
                with patch(
                    "MapGenerator.start_worms",
                    return_value=["p0", "p1"],
                ) as start_worms:
                    with patch(
                        "MapGenerator.monitor_worms",
                        side_effect=monitor,
                    ):
                        with patch("MapGenerator.safe_shm_close") as close:
                            generator.dig_map(4)

        self.assertEqual(generator.proc_counter, 2)
        np.testing.assert_array_equal(generator.bin_map, shared_result)
        start_worms.assert_called_once()
        close.assert_called_once_with(shm)

    def test_save_map_writes_image_and_matrix_under_project_assets(self) -> None:
        generator = object.__new__(MapGenerator)
        generator.bin_map = np.array([[1, 0], [0, 1]], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_module = Path(temp_dir) / "MapGenerator.py"
            with patch.object(map_generator_module, "__file__", str(fake_module)):
                generator.save_map()

            map_dir = Path(temp_dir) / "Assets" / "Map"
            self.assertTrue((map_dir / "map.png").exists())
            self.assertTrue((map_dir / "map_matrix.txt").exists())
            saved_matrix = np.loadtxt(map_dir / "map_matrix.txt")
            np.testing.assert_array_equal(saved_matrix, generator.bin_map)


if __name__ == "__main__":
    unittest.main()
