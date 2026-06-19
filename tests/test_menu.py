import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from Menu import Menu, MenuItem, MenuItemType
from SimSettings import SimSettings


def make_simulation_items(
    mission: int = 0,
    map_size: int = 1,
    seed: str = "19",
    drones: int = 0,
):
    return [
        SimpleNamespace(),
        SimpleNamespace(value=mission),
        SimpleNamespace(value=map_size),
        SimpleNamespace(text_input=seed),
        SimpleNamespace(value=drones),
    ]


class MenuItemTests(unittest.TestCase):
    def test_selector_changes_without_wrapping(self) -> None:
        game = SimpleNamespace(
            LEFT_KEY=False,
            RIGHT_KEY=True,
            menu=SimpleNamespace(_play_button=Mock()),
        )
        item = MenuItem(
            game,
            "Size",
            (0, 0),
            MenuItemType.SELECTOR,
            value=0,
            options=["S", "M"],
        )

        self.assertTrue(item.handle_input(game))
        self.assertEqual(item.value, 1)
        self.assertFalse(item.handle_input(game))
        self.assertEqual(item.value, 1)

    def test_slider_clamps_to_declared_range(self) -> None:
        game = SimpleNamespace(
            LEFT_KEY=True,
            RIGHT_KEY=False,
            menu=SimpleNamespace(_play_button=Mock()),
        )
        item = MenuItem(
            game,
            "Volume",
            (0, 0),
            MenuItemType.SLIDER,
            value=0,
            options=[0, 100, 20],
        )

        item.handle_input(game)
        self.assertEqual(item.value, 0)
        game.LEFT_KEY = False
        game.RIGHT_KEY = True
        item.handle_input(game)
        self.assertEqual(item.value, 20)


class MenuSettingsTests(unittest.TestCase):
    def make_menu(self) -> Menu:
        menu = object.__new__(Menu)
        menu.simulation = make_simulation_items()
        defaults = SimSettings()
        menu.slam_defaults = defaults
        menu.slam_scan_interval = defaults.slam_scan_interval
        menu.slam_scan_rays = defaults.slam_scan_rays
        menu.slam_point_cloud_max_points = defaults.slam_point_cloud_max_points
        menu.slam_render_point_tail = defaults.slam_render_point_tail
        menu.slam_render_interval = defaults.slam_render_interval
        menu.rover_share_interval = defaults.rover_share_interval
        menu.frontier_stride = defaults.frontier_stride
        menu.frontier_confidence_threshold = (
            defaults.frontier_confidence_threshold
        )
        menu.frontier_rebuild_cooldown = defaults.frontier_rebuild_cooldown
        return menu

    def test_build_sim_settings_normalizes_menu_values(self) -> None:
        menu = self.make_menu()
        menu.simulation = make_simulation_items(
            mission=1,
            map_size=2,
            seed="123",
            drones=3,
        )
        menu.slam_scan_rays = 24

        settings = menu.build_sim_settings()

        self.assertEqual(settings.mission, 1)
        self.assertEqual(settings.map_dim, "LARGE")
        self.assertEqual(settings.seed, 123)
        self.assertEqual(settings.num_drones, 6)
        self.assertEqual(settings.slam_scan_rays, 24)

    def test_navigation_skips_non_selectable_items(self) -> None:
        menu = self.make_menu()
        menu.current_menu = [
            SimpleNamespace(selectable=False),
            SimpleNamespace(selectable=True),
            SimpleNamespace(selectable=False),
            SimpleNamespace(selectable=True),
        ]
        menu.current_index = 1

        self.assertEqual(menu._get_next_selectable("down"), 3)
        self.assertEqual(menu._get_next_selectable("up"), 3)

    def test_simulation_settings_round_trip_through_ini(self) -> None:
        source = self.make_menu()
        source.simulation = make_simulation_items(
            mission=1,
            map_size=2,
            seed="444",
            drones=2,
        )
        source.slam_scan_interval = 0.5
        source.slam_render_interval = 0.2
        source.rover_share_interval = 0.75
        source.frontier_stride = 2

        target = self.make_menu()
        target.simulation = make_simulation_items(
            mission=0,
            map_size=0,
            seed="",
            drones=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "GameConfig"
            config_dir.mkdir()
            with patch("Menu.GAME_DIR", Path(temp_dir)):
                source.save_symSettings()
                target.load_symSettings()

        self.assertEqual(target.simulation[1].value, 1)
        self.assertEqual(target.simulation[2].value, 2)
        self.assertEqual(target.simulation[3].text_input, "444")
        self.assertEqual(target.simulation[4].value, 2)
        self.assertEqual(target.slam_scan_interval, 0.5)
        self.assertEqual(target.slam_render_interval, 0.2)
        self.assertEqual(target.rover_share_interval, 0.75)
        self.assertEqual(target.frontier_stride, 2)


if __name__ == "__main__":
    unittest.main()
