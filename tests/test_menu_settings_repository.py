import configparser
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

from ui.menu.settings_repository import AudioSettings, MenuSettingsRepository
from config.simulation_config import SimulationConfig


class MenuSettingsRepositoryTests(unittest.TestCase):
    def make_repository(self):
        temporary_directory = tempfile.TemporaryDirectory()
        root = Path(temporary_directory.name)
        (root / "GameConfig").mkdir()
        return temporary_directory, MenuSettingsRepository(root)

    def test_audio_options_round_trip_with_existing_schema(self) -> None:
        temporary_directory, repository = self.make_repository()
        self.addCleanup(temporary_directory.cleanup)

        repository.save_audio(AudioSettings(60, "off", "on"))
        loaded = repository.load_audio()

        self.assertEqual(loaded, AudioSettings(60, "off", "on"))
        config = configparser.ConfigParser()
        config.read(repository.options_path)
        self.assertTrue(config.has_section("Options"))

    def test_audio_defaults_load_before_local_overrides(self) -> None:
        temporary_directory, repository = self.make_repository()
        self.addCleanup(temporary_directory.cleanup)
        repository.options_default_path.write_text(
            "[Options]\n"
            "volume = 20\n"
            "music = off\n"
            "button = off\n"
        )

        self.assertEqual(
            repository.load_audio(),
            AudioSettings(20, "off", "off"),
        )

        repository.save_audio(AudioSettings(80, "on", "on"))

        self.assertEqual(
            repository.load_audio(),
            AudioSettings(80, "on", "on"),
        )
        self.assertTrue(repository.options_path.exists())
        self.assertEqual(
            repository.options_default_path.read_text(),
            "[Options]\n"
            "volume = 20\n"
            "music = off\n"
            "button = off\n",
        )

    def test_simulation_round_trip_uses_nested_sections(self) -> None:
        temporary_directory, repository = self.make_repository()
        self.addCleanup(temporary_directory.cleanup)
        defaults = SimulationConfig()
        source = replace(
            defaults,
            mission_config=replace(
                defaults.mission_config,
                objective=1,
                map_dim="LARGE",
                seed=444,
                num_drones=6,
            ),
            slam=replace(defaults.slam, scan_interval=0.5),
            sharing=replace(defaults.sharing, pair_cooldown=2.5),
            frontier=replace(
                defaults.frontier,
                confidence_threshold=0.75,
            ),
            waypoints=replace(defaults.waypoints, spatial_hash_cell=24),
            exploration=replace(defaults.exploration, iterations=32),
            rendering=replace(defaults.rendering, refresh_interval=0.2),
            trace=replace(defaults.trace, enabled=True),
        )

        repository.save_simulation(source)
        loaded = repository.load_simulation(defaults)

        self.assertEqual(loaded, source)
        config = configparser.ConfigParser()
        config.read(repository.simulation_path)
        self.assertEqual(
            set(config.sections()),
            {
                "MISSION",
                "SLAM",
                "SHARING",
                "FRONTIER",
                "WAYPOINTS",
                "EXPLORATION",
                "RENDERING",
                "TRACE",
            },
        )

    def test_simulation_default_loads_before_local_override(self) -> None:
        temporary_directory, repository = self.make_repository()
        self.addCleanup(temporary_directory.cleanup)
        repository.simulation_default_path.write_text(
            "[MISSION]\n"
            "objective = Exploration\n"
            "map_dimension = Small\n"
            "seed = 5\n"
            "drones = 3\n"
        )
        defaults = SimulationConfig()

        loaded_default = repository.load_simulation(defaults)
        self.assertEqual(loaded_default.mission_config.seed, 5)

        local = replace(
            defaults,
            mission_config=replace(
                defaults.mission_config,
                seed=99,
            ),
        )
        repository.save_simulation(local)

        loaded_local = repository.load_simulation(defaults)
        self.assertEqual(loaded_local.mission_config.seed, 99)
        self.assertTrue(repository.simulation_path.exists())

    def test_missing_simulation_files_return_no_loaded_settings(self) -> None:
        temporary_directory, repository = self.make_repository()
        self.addCleanup(temporary_directory.cleanup)

        self.assertIsNone(repository.load_simulation(SimulationConfig()))

    def test_invalid_section_restores_only_that_section_defaults(self) -> None:
        temporary_directory, repository = self.make_repository()
        self.addCleanup(temporary_directory.cleanup)
        repository.simulation_path.write_text(
            "[MISSION]\n"
            "objective = Exploration\n"
            "map_dimension = Small\n"
            "seed = 5\n"
            "drones = 3\n"
            "\n"
            "[SLAM]\n"
            "scan_interval = invalid\n"
            "scan_rays = 12\n"
            "\n"
            "[SHARING]\n"
            "pair_cooldown = 2.0\n"
        )
        defaults = SimulationConfig()

        loaded = repository.load_simulation(defaults)

        self.assertEqual(loaded.slam, defaults.slam)
        self.assertEqual(loaded.sharing.pair_cooldown, 2.0)

    def test_phase6_typed_navigation_schema_contains_only_live_controls(self) -> None:
        defaults = SimulationConfig()

        self.assertEqual(
            {field.name for field in fields(defaults.frontier)},
            {
                "confidence_threshold",
                "minimum_unknown_support",
                "continuation_min_distance",
                "continuation_scan_headings",
                "rebuild_cooldown",
                "cluster_match_distance",
                "missing_refresh_limit",
                "gateway_min_separation",
            },
        )
        self.assertEqual(
            {field.name for field in fields(defaults.waypoints)},
            {
                "enabled",
                "spatial_hash_cell",
                "merge_radius",
                "connector_distance",
                "gateway_connector_distance",
                "route_cache_capacity",
                "connector_limit",
                "turn_threshold_degrees",
                "minimum_turn_leg",
                "chokepoint_narrow_clearance",
                "chokepoint_shoulder_clearance",
                "chokepoint_shoulder_length",
                "recovery_anchor_interval",
            },
        )
        self.assertEqual(
            {field.name for field in fields(defaults.exploration)},
            {
                "policy",
                "iterations",
                "horizon",
                "planning_rays",
                "uct_exploration",
                "discount",
                "decision_time_budget_ms",
            },
        )

    def test_phase6_navigation_defaults_match_the_locked_plan(self) -> None:
        defaults = SimulationConfig()

        self.assertEqual(defaults.waypoints.spatial_hash_cell, 32)
        self.assertEqual(defaults.waypoints.merge_radius, 8.0)
        self.assertEqual(defaults.waypoints.connector_distance, 64.0)
        self.assertEqual(
            defaults.waypoints.gateway_connector_distance,
            192.0,
        )
        self.assertEqual(defaults.waypoints.route_cache_capacity, 64)
        self.assertEqual(defaults.frontier.cluster_match_distance, 32.0)
        self.assertEqual(defaults.frontier.minimum_unknown_support, 4)
        self.assertEqual(defaults.frontier.continuation_min_distance, 12.0)
        self.assertEqual(defaults.frontier.continuation_scan_headings, 3)
        self.assertEqual(defaults.frontier.missing_refresh_limit, 3)
        self.assertEqual(defaults.frontier.gateway_min_separation, 64.0)
        self.assertEqual(defaults.exploration.decision_time_budget_ms, 40.0)

    def test_legacy_navigation_keys_are_readable_but_ignored(self) -> None:
        temporary_directory, repository = self.make_repository()
        self.addCleanup(temporary_directory.cleanup)
        repository.simulation_path.write_text(
            "[FRONTIER]\n"
            "stride = no-longer-a-number\n"
            "confidence_threshold = 0.75\n"
            "continuation_min_distance = 15.0\n"
            "continuation_scan_headings = 2\n"
            "\n"
            "[WAYPOINTS]\n"
            "enabled = true\n"
            "spacing = no-longer-a-number\n"
            "direct_path_limit = no-longer-a-number\n"
            "merge_radius = 6.0\n"
            "connector_distance = 48.0\n"
            "\n"
            "[EXPLORATION]\n"
            "iterations = 17\n"
            "branching_factor = no-longer-a-number\n"
            "frontier_cluster_limit = no-longer-a-number\n"
            "rollout_temperature = no-longer-a-number\n"
        )

        loaded = repository.load_simulation(SimulationConfig())

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.frontier.confidence_threshold, 0.75)
        self.assertEqual(loaded.frontier.continuation_min_distance, 15.0)
        self.assertEqual(loaded.frontier.continuation_scan_headings, 2)
        self.assertEqual(loaded.waypoints.merge_radius, 6.0)
        self.assertEqual(loaded.waypoints.connector_distance, 48.0)
        self.assertEqual(loaded.waypoints.spatial_hash_cell, 32)
        self.assertEqual(loaded.waypoints.gateway_connector_distance, 192.0)
        self.assertEqual(loaded.waypoints.route_cache_capacity, 64)
        self.assertEqual(loaded.exploration.iterations, 17)

    def test_next_save_replaces_all_legacy_navigation_keys(self) -> None:
        temporary_directory, repository = self.make_repository()
        self.addCleanup(temporary_directory.cleanup)

        repository.save_simulation(SimulationConfig())

        config = configparser.ConfigParser()
        config.read(repository.simulation_path)
        self.assertEqual(
            set(config["FRONTIER"]),
            {
                "confidence_threshold",
                "minimum_unknown_support",
                "continuation_min_distance",
                "continuation_scan_headings",
                "rebuild_cooldown",
                "cluster_match_distance",
                "missing_refresh_limit",
                "gateway_min_separation",
            },
        )
        self.assertEqual(
            set(config["WAYPOINTS"]),
            {
                "enabled",
                "spatial_hash_cell",
                "merge_radius",
                "connector_distance",
                "gateway_connector_distance",
                "route_cache_capacity",
                "connector_limit",
                "turn_threshold_degrees",
                "minimum_turn_leg",
                "chokepoint_narrow_clearance",
                "chokepoint_shoulder_clearance",
                "chokepoint_shoulder_length",
                "recovery_anchor_interval",
            },
        )
        self.assertEqual(
            set(config["EXPLORATION"]),
            {
                "policy",
                "iterations",
                "horizon",
                "planning_rays",
                "uct_exploration",
                "discount",
                "decision_time_budget_ms",
            },
        )
        self.assertEqual(config.getint("WAYPOINTS", "spatial_hash_cell"), 32)
        self.assertEqual(config.getfloat("WAYPOINTS", "merge_radius"), 8.0)
        self.assertEqual(
            config.getfloat("WAYPOINTS", "gateway_connector_distance"),
            192.0,
        )
        self.assertEqual(config.getint("WAYPOINTS", "route_cache_capacity"), 64)


if __name__ == "__main__":
    unittest.main()
