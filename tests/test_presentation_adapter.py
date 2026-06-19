import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from PresentationAdapter import PresentationAdapter


def make_drones(count: int = 3):
    return [
        SimpleNamespace(
            show_path=True,
            show_vision=True,
        )
        for _ in range(count)
    ]


class PresentationAdapterTests(unittest.TestCase):
    def test_global_heatmap_hides_and_restores_all_agent_overlays(self) -> None:
        adapter = PresentationAdapter(10, 10)
        drones = make_drones()
        control_center = SimpleNamespace(
            handle_click=Mock(return_value=("terrain_heatmap", None)),
        )

        adapter.handle_click((1, 1), control_center, drones)

        self.assertTrue(adapter.show_terrain_heatmap)
        self.assertTrue(adapter.terrain_heatmap_dirty)
        self.assertTrue(all(not drone.show_path for drone in drones))
        self.assertTrue(all(not drone.show_vision for drone in drones))

        adapter.handle_click((1, 1), control_center, drones)
        self.assertTrue(all(drone.show_path for drone in drones))
        self.assertTrue(all(drone.show_vision for drone in drones))

    def test_selected_drone_limits_visible_overlays(self) -> None:
        adapter = PresentationAdapter(10, 10)
        drones = make_drones()
        control_center = SimpleNamespace(
            handle_click=Mock(return_value=("drone_heatmap", 1)),
        )

        adapter.handle_click((1, 1), control_center, drones)

        self.assertEqual(adapter.selected_drone_heatmap_id, 1)
        self.assertEqual(
            [drone.show_vision for drone in drones],
            [False, True, False],
        )
        self.assertEqual(
            [drone.show_path for drone in drones],
            [False, True, False],
        )

        control_center.handle_click.return_value = ("terrain_heatmap", None)
        adapter.handle_click((1, 1), control_center, drones)
        self.assertTrue(adapter.show_terrain_heatmap)
        self.assertEqual(
            [drone.show_vision for drone in drones],
            [False, True, False],
        )
        self.assertEqual(
            [drone.show_path for drone in drones],
            [False, False, False],
        )

    def test_unhandled_click_leaves_state_unchanged(self) -> None:
        adapter = PresentationAdapter(10, 10)
        drones = make_drones()
        control_center = SimpleNamespace(handle_click=Mock(return_value=None))

        adapter.handle_click((9, 9), control_center, drones)

        self.assertFalse(adapter.show_terrain_heatmap)
        self.assertIsNone(adapter.selected_drone_heatmap_id)
        self.assertTrue(all(drone.show_path for drone in drones))


if __name__ == "__main__":
    unittest.main()
