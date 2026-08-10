import concurrent.futures
import unittest
from dataclasses import FrozenInstanceError
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from config.simulation_config import (
    SimulationConfig,
    SlamConfig,
    TraceConfig,
    WaypointConfig,
)
from mission.runtime_trace import RuntimeTraceLogger
from mapping.poi import POI
from asset_config.gameplay import GameOptions
from asset_config.helpers import next_cell_coords, wall_hit
from asset_config.media import Audio, Images
from asset_config.rendering import Fonts
from tools.analyze_runtime_trace import summarize


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
        with self.assertRaises(ValueError):
            WaypointConfig(spatial_hash_cell=4, merge_radius=4.0)

    def test_runtime_trace_writes_jsonl_events(self) -> None:
        self.assertEqual(RuntimeTraceLogger.SCHEMA_VERSION, 3)
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
        self.assertEqual(
            [event["sequence"] for event in events],
            list(range(len(events))),
        )
        self.assertTrue(
            all(
                event["schema_version"]
                == RuntimeTraceLogger.SCHEMA_VERSION
                for event in events
            )
        )

    def test_runtime_trace_sequence_matches_concurrent_file_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = RuntimeTraceLogger(
                Path(temp_dir),
                TraceConfig(enabled=True, directory="logs"),
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=4
            ) as executor:
                list(
                    executor.map(
                        lambda value: trace.record(
                            "concurrent",
                            value=value,
                        ),
                        range(40),
                    )
                )
            path = trace.path
            trace.close()
            self.assertIsNotNone(path)
            events = [
                json.loads(line)
                for line in Path(path).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(
            [event["sequence"] for event in events],
            list(range(len(events))),
        )
        self.assertEqual(
            sum(event["event"] == "concurrent" for event in events),
            40,
        )

    def test_runtime_trace_uses_distinct_files_within_the_same_instant(
        self,
    ) -> None:
        fixed_now = datetime(2026, 7, 15, 12, 0, 0, 123456)
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "mission.runtime_trace.datetime"
        ) as mocked_datetime:
            mocked_datetime.now.return_value = fixed_now
            first = RuntimeTraceLogger(
                Path(temp_dir),
                TraceConfig(enabled=True, directory="logs"),
            )
            second = RuntimeTraceLogger(
                Path(temp_dir),
                TraceConfig(enabled=True, directory="logs"),
            )
            first_path = first.path
            second_path = second.path
            first.close()
            second.close()

            self.assertIsNotNone(first_path)
            self.assertIsNotNone(second_path)
            self.assertNotEqual(first_path, second_path)
            for path in (first_path, second_path):
                events = [
                    json.loads(line)
                    for line in Path(path).read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertEqual(
                    [event["sequence"] for event in events],
                    list(range(len(events))),
                )

    def test_runtime_trace_reserves_schema_and_ordering_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = RuntimeTraceLogger(
                Path(temp_dir),
                TraceConfig(enabled=True, directory="logs"),
            )
            trace.record(
                "example",
                schema_version=-1,
                sequence=999,
                wall_time=-1.0,
                perf_time=-1.0,
            )
            path = trace.path
            trace.close()
            self.assertIsNotNone(path)
            event = json.loads(
                Path(path).read_text(encoding="utf-8").splitlines()[1]
            )

        self.assertEqual(event["schema_version"], RuntimeTraceLogger.SCHEMA_VERSION)
        self.assertEqual(event["sequence"], 1)
        self.assertGreater(event["wall_time"], 0.0)
        self.assertGreater(event["perf_time"], 0.0)

    def test_trace_analyzer_summarizes_waypoint_health(self) -> None:
        lines = summarize(
            [
                {"event": "trace_started", "path": "example.jsonl"},
                {
                    "event": "drone_waypoint_route",
                    "drone_id": 0,
                    "sim_time": 1.0,
                    "status": "ok",
                    "bridge_status": "ok",
                    "gateway_status": "ok",
                    "route_elapsed_ms": 0.5,
                    "graph_nodes": 33,
                    "graph_edges": 32,
                },
                {
                    "event": "drone_waypoint_segment_path",
                    "drone_id": 0,
                    "sim_time": 1.1,
                    "path_source": "astar",
                },
            ]
        )
        summary = "\n".join(lines)

        self.assertIn("waypoint route statuses: ok=1", summary)
        self.assertIn("waypoint bridge statuses: ok=1", summary)
        self.assertIn("waypoint gateway statuses: ok=1", summary)
        self.assertIn("avg=0.50ms max=0.50ms graph=33n/32e", summary)
        self.assertIn("waypoint segment paths: astar=1", summary)
        self.assertIn("Characterization:", summary)
        self.assertIn("waypoint routes: calls=1 ok=1 failed=0", summary)

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
