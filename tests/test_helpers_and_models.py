import unittest

from POI import POI
from SimSettings import SimSettings
from asset_config.gameplay import GameOptions
from asset_config.helpers import next_cell_coords, wall_hit
from asset_config.media import Audio, Images
from asset_config.rendering import Fonts


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

    def test_sim_settings_preserve_explicit_runtime_values(self) -> None:
        settings = SimSettings(
            mission=1,
            map_dim="LARGE",
            seed=42,
            num_drones=6,
            slam_scan_rays=24,
            frontier_stride=2,
        )

        self.assertEqual(settings.mission, 1)
        self.assertEqual(settings.map_dim, "LARGE")
        self.assertEqual(settings.seed, 42)
        self.assertEqual(settings.num_drones, 6)
        self.assertEqual(settings.slam_scan_rays, 24)
        self.assertEqual(settings.frontier_stride, 2)

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

    def test_poi_identity_is_based_on_id(self) -> None:
        first = POI("poi-1", "chamber", (2, 3))
        duplicate = POI("poi-1", "formation", (8, 9))
        other = POI("poi-2", "chamber", (2, 3))

        self.assertEqual(first, duplicate)
        self.assertNotEqual(first, other)
        self.assertEqual({first, duplicate, other}, {first, other})


if __name__ == "__main__":
    unittest.main()
