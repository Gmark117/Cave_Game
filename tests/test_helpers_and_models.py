import unittest
from dataclasses import FrozenInstanceError
import json
import tempfile
from pathlib import Path

from config.simulation_config import SimulationConfig, SlamConfig, TraceConfig
from mission.runtime_trace import RuntimeTraceLogger
from mapping.poi import POI
from asset_config.gameplay import GameOptions
from asset_config.helpers import next_cell_coords, wall_hit
from asset_config.media import Audio, Images
from asset_config.rendering import Fonts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HelperAndModelTests(unittest.TestCase):
    def test_next_cell_coords_uses_game_heading_convention(self) -> None:
        self.assertEqual(next_cell_coords(10, 10, 5, 0), (10, 5))
        self.assertEqual(next_cell_coords(10, 10, 5, 90), (15, 10))
        self.assertEqual(next_cell_coords(10, 10, 5, 180), (10, 15))
        self.assertEqual(next_cell_coords(10, 10, 5, 270), (5, 10))

    def test_wall_hit_reads_xy_positions_from_yx_maps(self) -> None:
        cave = [
            [0, 1],
            [0, 0],
        ]

        self.assertTrue(wall_hit(cave, (1, 0)))
        self.assertFalse(wall_hit(cave, (0, 1)))

    def test_nested_simulation_config_is_immutable_and_validated(self) -> None:
        settings = SimulationConfig(slam=SlamConfig(scan_rays=24))

        self.assertEqual(settings.slam.scan_rays, 24)
        self.assertFalse(settings.trace.enabled)
        with self.assertRaises(FrozenInstanceError):
            settings.slam.scan_rays = 12
        with self.assertRaises(ValueError):
            SlamConfig(scan_rays=0)
        with self.assertRaises(ValueError):
            TraceConfig(directory="")

    def test_runtime_trace_writes_jsonl_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = RuntimeTraceLogger(
                Path(temp_dir),
                TraceConfig(enabled=True, directory="logs"),
            )
            trace.record("example", value=3, position=(1, 2))
            path = trace.path
            trace.close()

            self.assertIsNotNone(path)
            lines = Path(path).read_text(encoding="utf-8").splitlines()

        events = [json.loads(line) for line in lines]

        self.assertEqual(events[0]["event"], "trace_started")
        self.assertEqual(events[1]["event"], "example")
        self.assertEqual(events[1]["position"], [1, 2])
        self.assertEqual(events[-1]["event"], "trace_closed")

    def test_configuration_resources_and_options_are_available(self) -> None:
        self.assertEqual(GameOptions.MAP_SIZE, ["Small", "Medium", "Large"])
        for resource in (
            Audio.AMBIENT.value,
            Audio.BUTTON.value,
            Images.DRONE.value,
            Images.ROVER.value,
            Fonts.SMALL.value,
            Fonts.BIG.value,
        ):
            self.assertTrue(resource.exists(), resource)

    def test_dependency_manifest_lists_runtime_dependencies(self) -> None:
        requirements = {
            line.strip()
            for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }

        self.assertGreaterEqual(
            requirements,
            {"pygame", "numpy", "opencv-python"},
        )

    def test_gitignore_excludes_runtime_outputs(self) -> None:
        patterns = (PROJECT_ROOT / ".gitignore").read_text()

        for pattern in (
            "__pycache__/",
            "GameConfig/options.local.ini",
            "GameConfig/simulation.local.ini",
            "Assets/Map/*.png",
            "Assets/Map/*.txt",
            "logs/",
        ):
            self.assertIn(pattern, patterns)

    def test_poi_identity_is_based_on_id(self) -> None:
        first = POI("poi-1", "chamber", (2, 3))
        duplicate = POI("poi-1", "formation", (8, 9))
        other = POI("poi-2", "chamber", (2, 3))

        self.assertEqual(first, duplicate)
        self.assertNotEqual(first, other)
        self.assertEqual({first, duplicate, other}, {first, other})


if __name__ == "__main__":
    unittest.main()
