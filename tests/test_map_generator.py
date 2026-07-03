import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from MapGenerator import MapGenerator, _image_path_from_key
from SimulationConfig import MissionConfig, SimulationConfig
from asset_config.gameplay import Display
from asset_config.media import Images
from generation import CaveGenerationResult


class MapGeneratorTests(unittest.TestCase):
    def test_image_key_resolution_is_case_insensitive_and_validated(self) -> None:
        self.assertEqual(
            _image_path_from_key("cave_floor"),
            Images.CAVE_FLOOR.value,
        )
        with self.assertRaisesRegex(KeyError, "Unknown image key"):
            _image_path_from_key("missing")

    def test_facade_generates_writes_and_exposes_mission_inputs(self) -> None:
        bin_map = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        roughness = np.array([[0.0, 0.2], [0.4, 0.0]], dtype=np.float32)
        result = CaveGenerationResult(
            bin_map=bin_map,
            terrain_roughness=roughness,
            worm_x=[1, 2],
            worm_y=[3, 4],
            worm_inputs=(5, 6, 7),
            completed_workers=2,
            worker_crashed=False,
        )
        cave_generator = Mock()
        cave_generator.generate.return_value = result
        game = SimpleNamespace(menu=SimpleNamespace(blit_loading=Mock()))
        settings = SimulationConfig(
            mission_config=MissionConfig(seed=7, map_dim="SMALL")
        )

        with patch("MapGenerator.CaveGenerator", return_value=cave_generator) as generator_cls:
            with patch("MapGenerator.MapArtifactWriter") as writer_cls:
                cartographer = MapGenerator(game, settings)

        generator_cls.assert_called_once_with(
            Display.FULL_W - Display.LEGEND_WIDTH,
            Display.FULL_H,
            7,
            "SMALL",
            MapGenerator.NUM_PROCESSES,
        )
        cave_generator.generate.assert_called_once()
        progress = cave_generator.generate.call_args.args[0]
        progress.on_digging()
        progress.on_post_processing()
        self.assertEqual(
            game.menu.blit_loading.call_args_list,
            [
                unittest.mock.call(["Digging..."]),
                unittest.mock.call(["Breeding bats..."]),
            ],
        )
        writer = writer_cls.return_value
        self.assertIs(writer.write.call_args.args[0], bin_map)
        self.assertIs(cartographer.bin_map, bin_map)
        self.assertIs(cartographer.terrain_roughness, roughness)
        self.assertEqual(cartographer.worm_x, [1, 2])
        self.assertEqual(cartographer.worm_y, [3, 4])
        self.assertEqual(cartographer.worm_inputs, (5, 6, 7))
        self.assertEqual(cartographer.proc_counter, 2)
        self.assertFalse(cartographer.worker_crashed)

    def test_digging_progress_ignores_missing_loading_overlay(self) -> None:
        cartographer = object.__new__(MapGenerator)
        cartographer.game = SimpleNamespace()

        cartographer._show_digging_progress()


if __name__ == "__main__":
    unittest.main()
