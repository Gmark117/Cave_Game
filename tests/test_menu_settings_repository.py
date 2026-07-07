import configparser
import tempfile
import unittest
from dataclasses import replace
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
            frontier=replace(defaults.frontier, stride=2),
            rendering=replace(defaults.rendering, refresh_interval=0.2),
        )

        repository.save_simulation(source)
        loaded = repository.load_simulation(defaults)

        self.assertEqual(loaded, source)
        config = configparser.ConfigParser()
        config.read(repository.simulation_path)
        self.assertEqual(
            set(config.sections()),
            {"MISSION", "SLAM", "SHARING", "FRONTIER", "RENDERING"},
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


if __name__ == "__main__":
    unittest.main()
