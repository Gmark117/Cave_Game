import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from ui.menu.facade import Menu
from ui.menu.models import SelectorItem, TextInputItem
from ui.menu.settings_repository import MenuSettingsRepository
from config.simulation_config import SimulationConfig


def make_simulation_items(
    mission: int = 0,
    map_size: int = 1,
    seed: str = "19",
    drones: int = 0,
):
    return [
        SimpleNamespace(selectable=False),
        SelectorItem("Objective", (0, 0), options=("Exploration", "Rescue"), value=mission),
        SelectorItem("Size", (0, 0), options=("Small", "Medium", "Large"), value=map_size),
        TextInputItem("Seed", (0, 0), text=seed),
        SelectorItem("Drones", (0, 0), options=(3, 4, 5, 6, 7, 8), value=drones),
    ]


class MenuFacadeTests(unittest.TestCase):
    def make_menu(self, root: Path | None = None) -> Menu:
        menu = object.__new__(Menu)
        menu.game = SimpleNamespace(start_mission=Mock())
        menu.simulation = make_simulation_items()
        menu.config = SimulationConfig()
        if root is not None:
            menu.settings_repository = MenuSettingsRepository(root)
        return menu

    def test_build_sim_settings_normalizes_menu_values(self) -> None:
        menu = self.make_menu()
        menu.simulation = make_simulation_items(
            mission=1,
            map_size=2,
            seed="123",
            drones=3,
        )
        menu.config = replace(
            menu.config,
            slam=replace(menu.config.slam, scan_rays=24),
        )

        settings = menu.build_sim_settings()

        self.assertEqual(settings.mission_config.objective, 1)
        self.assertEqual(settings.mission_config.map_dim, "LARGE")
        self.assertEqual(settings.mission_config.seed, 123)
        self.assertEqual(settings.mission_config.num_drones, 6)
        self.assertEqual(settings.slam.scan_rays, 24)

    def test_simulation_settings_round_trip_through_ini(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "GameConfig").mkdir()
            source = self.make_menu(root)
            source.simulation = make_simulation_items(
                mission=1,
                map_size=2,
                seed="444",
                drones=2,
            )
            source.config = replace(
                source.config,
                slam=replace(source.config.slam, scan_interval=0.5),
                rendering=replace(
                    source.config.rendering,
                    refresh_interval=0.2,
                ),
                sharing=replace(
                    source.config.sharing,
                    rover_interval=0.75,
                ),
                frontier=replace(
                    source.config.frontier,
                    confidence_threshold=0.75,
                ),
                exploration=replace(
                    source.config.exploration,
                    iterations=64,
                ),
            )
            source.save_simulation_settings()

            target = self.make_menu(root)
            target.simulation = make_simulation_items(
                mission=0,
                map_size=0,
                seed="",
                drones=0,
            )
            target.load_simulation_settings()

        self.assertEqual(target.simulation[1].value, 1)
        self.assertEqual(target.simulation[2].value, 2)
        self.assertEqual(target.simulation[3].text, "444")
        self.assertEqual(target.simulation[4].value, 2)
        self.assertEqual(target.config.slam.scan_interval, 0.5)
        self.assertEqual(target.config.rendering.refresh_interval, 0.2)
        self.assertEqual(target.config.sharing.rover_interval, 0.75)
        self.assertEqual(target.config.frontier.confidence_threshold, 0.75)
        self.assertEqual(target.config.exploration.iterations, 64)

    def test_default_seed_is_owned_only_by_text_item(self) -> None:
        menu = self.make_menu()
        menu.set_default_seed()

        self.assertTrue(menu.simulation[3].text)
        self.assertFalse(hasattr(menu, "seed_input"))

    def test_start_mission_requires_seed_and_preserves_call_order(self) -> None:
        menu = self.make_menu()
        menu.show_menu = True
        menu.save_simulation_settings = Mock()

        menu.start_mission()

        menu.save_simulation_settings.assert_called_once_with()
        menu.game.start_mission.assert_called_once_with()
        self.assertFalse(menu.show_menu)

    def test_toggle_music_updates_selector_mixer_and_local_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            menu = self.make_menu(root)
            menu.volume = 60
            menu.sound_on_off = "on"
            menu.button_on_off = "on"
            menu.audio = SimpleNamespace(set_music=Mock())
            menu.options = [
                SimpleNamespace(selectable=False),
                SimpleNamespace(value=menu.volume),
                SelectorItem(
                    "Music",
                    (0, 0),
                    options=("on", "off"),
                    value=0,
                ),
                SelectorItem(
                    "Button",
                    (0, 0),
                    options=("on", "off"),
                    value=0,
                ),
            ]

            enabled = menu.toggle_music()
            loaded = menu.settings_repository.load_audio()

        self.assertFalse(enabled)
        self.assertEqual(menu.sound_on_off, "off")
        self.assertEqual(menu.options[2].value, 1)
        menu.audio.set_music.assert_called_once_with(False)
        self.assertEqual(loaded.volume, 60)
        self.assertEqual(loaded.music, "off")
        self.assertEqual(loaded.button, "on")


if __name__ == "__main__":
    unittest.main()
