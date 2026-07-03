import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from mission.presentation_adapter import PresentationAdapter


def make_drones(count: int = 3):
    drones = []
    for _ in range(count):
        drone = SimpleNamespace(
            show_path=True,
            show_vision=True,
        )
        drone.toggle_path = Mock(
            side_effect=lambda drone=drone: setattr(
                drone,
                "show_path",
                not drone.show_path,
            )
        )
        drone.toggle_vision = Mock(
            side_effect=lambda drone=drone: setattr(
                drone,
                "show_vision",
                not drone.show_vision,
            )
        )
        drone.set_overlay_visibility = Mock(
            side_effect=lambda *,
            show_path,
            show_vision,
            drone=drone: (
                setattr(drone, "show_path", show_path),
                setattr(drone, "show_vision", show_vision),
            )
        )
        drones.append(drone)
    return drones


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

    def test_reset_owns_default_heatmap_and_overlay_state(self) -> None:
        adapter = PresentationAdapter(10, 10)
        drones = make_drones()
        adapter.show_terrain_heatmap = True
        adapter.selected_drone_heatmap_id = 2
        adapter.terrain_heatmap_dirty = False
        for drone in drones:
            drone.show_path = False
            drone.show_vision = False

        adapter.reset(drones)

        self.assertFalse(adapter.show_terrain_heatmap)
        self.assertIsNone(adapter.selected_drone_heatmap_id)
        self.assertTrue(adapter.terrain_heatmap_dirty)
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

    def test_path_and_vision_actions_are_applied_by_adapter(self) -> None:
        adapter = PresentationAdapter(10, 10)
        drones = make_drones()
        control_center = SimpleNamespace(
            handle_click=Mock(side_effect=[
                ("drone_path", 1),
                ("drone_vision", 1),
            ]),
        )

        adapter.handle_click((1, 1), control_center, drones)
        adapter.handle_click((1, 1), control_center, drones)

        drones[1].toggle_path.assert_called_once_with()
        drones[1].toggle_vision.assert_called_once_with()
        self.assertFalse(drones[1].show_path)
        self.assertFalse(drones[1].show_vision)

    def test_invalid_drone_action_leaves_state_unchanged(self) -> None:
        adapter = PresentationAdapter(10, 10)
        drones = make_drones()
        control_center = SimpleNamespace(
            handle_click=Mock(return_value=("drone_heatmap", 99)),
        )

        adapter.handle_click((1, 1), control_center, drones)

        self.assertIsNone(adapter.selected_drone_heatmap_id)
        self.assertTrue(all(drone.show_path for drone in drones))
        self.assertTrue(all(drone.show_vision for drone in drones))

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
