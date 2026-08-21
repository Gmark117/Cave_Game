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
                minimum_cluster_cells=9,
                distance_band=20.0,
                wall_continuation_weight=2.5,
                cluster_size_weight=3.0,
                cluster_proximity_weight=0.5,
                global_cell_size=24,
                global_refresh_interval=3.0,
            ),
            exploration=replace(
                defaults.exploration,
                policy="random",
                stagnation_distance=144.0,
                stagnation_min_sensor_cells_per_px=0.75,
                wall_direction_bias=5.0,
                unexplored_direction_bias=2.5,
                separation_direction_bias=1.25,
            ),
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

    def test_navigation_schema_contains_only_random_escape_controls(self) -> None:
        defaults = SimulationConfig()

        self.assertEqual(
            {field.name for field in fields(defaults.frontier)},
            {
                "confidence_threshold",
                "stride",
                "rebuild_cooldown",
                "minimum_cluster_cells",
                "distance_band",
                "wall_continuation_weight",
                "cluster_size_weight",
                "cluster_proximity_weight",
                "global_cell_size",
                "global_refresh_interval",
            },
        )
        self.assertEqual(
            {field.name for field in fields(defaults.exploration)},
            {
                "policy",
                "stagnation_distance",
                "stagnation_min_sensor_cells_per_px",
                "wall_direction_bias",
                "unexplored_direction_bias",
                "separation_direction_bias",
            },
        )

    def test_random_navigation_defaults(self) -> None:
        defaults = SimulationConfig()

        self.assertEqual(defaults.frontier.stride, 4)
        self.assertEqual(defaults.frontier.confidence_threshold, 0.6)
        self.assertEqual(defaults.frontier.minimum_cluster_cells, 12)
        self.assertEqual(defaults.frontier.distance_band, 16.0)
        self.assertEqual(defaults.frontier.wall_continuation_weight, 2.0)
        self.assertEqual(defaults.frontier.cluster_size_weight, 2.0)
        self.assertEqual(defaults.frontier.cluster_proximity_weight, 1.0)
        self.assertEqual(defaults.frontier.global_cell_size, 32)
        self.assertEqual(defaults.frontier.global_refresh_interval, 2.0)
        self.assertEqual(defaults.exploration.policy, "random")
        self.assertEqual(defaults.exploration.stagnation_distance, 120.0)
        self.assertEqual(
            defaults.exploration.stagnation_min_sensor_cells_per_px,
            0.5,
        )
        self.assertEqual(defaults.exploration.wall_direction_bias, 4.0)
        self.assertEqual(defaults.exploration.unexplored_direction_bias, 2.0)
        self.assertEqual(defaults.exploration.separation_direction_bias, 1.5)

    def test_legacy_navigation_keys_are_readable_but_ignored(self) -> None:
        temporary_directory, repository = self.make_repository()
        self.addCleanup(temporary_directory.cleanup)
        repository.simulation_path.write_text(
            "[FRONTIER]\n"
            "stride = 2\n"
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
            "policy = mcts\n"
            "iterations = 17\n"
            "branching_factor = no-longer-a-number\n"
            "frontier_cluster_limit = no-longer-a-number\n"
            "rollout_temperature = no-longer-a-number\n"
        )

        loaded = repository.load_simulation(SimulationConfig())

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.frontier.confidence_threshold, 0.75)
        self.assertEqual(loaded.frontier.stride, 2)
        self.assertFalse(hasattr(loaded, "waypoints"))
        self.assertEqual(loaded.exploration.policy, "random")

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
                "stride",
                "rebuild_cooldown",
                "minimum_cluster_cells",
                "distance_band",
                "wall_continuation_weight",
                "cluster_size_weight",
                "cluster_proximity_weight",
                "global_cell_size",
                "global_refresh_interval",
            },
        )
        self.assertEqual(
            set(config["EXPLORATION"]),
            {
                "policy",
                "stagnation_distance",
                "stagnation_min_sensor_cells_per_px",
                "wall_direction_bias",
                "unexplored_direction_bias",
                "separation_direction_bias",
            },
        )
        self.assertFalse(config.has_section("WAYPOINTS"))
        self.assertEqual(config.get("EXPLORATION", "policy"), "random")


if __name__ == "__main__":
    unittest.main()
