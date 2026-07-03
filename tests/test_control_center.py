import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ui.control_center.controller import ControlHitMap
from ui.control_center.facade import ControlCenter
from asset_config.rendering import Colors


class ControlCenterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.center = object.__new__(ControlCenter)
        self.center.game = SimpleNamespace()
        self.center._controller = Mock(
            active_tab="drones",
            explored_percent=40,
        )
        self.center._controller.format_timer.return_value = "00:12"
        self.center._renderer = Mock()
        self.center._renderer.render.return_value = ControlHitMap(
            heatmap_toggle=(1, 2, 3, 4),
        )
        self.center._renderer.percent_color.side_effect = (
            lambda value, maximum=100: (
                Colors.RED.value
                if value < maximum * 0.2
                else (
                    Colors.YELLOW.value
                    if value < maximum * 0.8
                    else Colors.GREEN.value
                )
            )
        )
        self.center._hit_map = ControlHitMap()
        self.center._num_drones = 0
        self.center._num_rovers = 0

    def test_draw_builds_one_frame_model_and_retains_hit_map(self) -> None:
        drones = (
            SimpleNamespace(id=0),
            SimpleNamespace(id=1),
            SimpleNamespace(id=2),
        )
        rovers = (
            SimpleNamespace(id=0),
            SimpleNamespace(id=1),
        )

        self.center.draw_control_center(
            drones,
            rovers,
            show_terrain_heatmap=False,
            selected_drone_heatmap_id=1,
            debug_lines=["line"],
        )

        view = self.center._renderer.render.call_args.args[0]
        self.assertEqual(view.elapsed_time, "00:12")
        self.assertEqual(view.explored_percent, 40)
        self.assertEqual(view.active_tab, "drones")
        self.assertIs(view.drone_statuses, drones)
        self.assertIs(view.rover_statuses, rovers)
        self.assertEqual(view.debug_lines, ("line",))
        self.assertEqual(self.center.num_drones, 3)
        self.assertEqual(self.center.num_rovers, 2)
        self.assertEqual(
            self.center._hit_map.heatmap_toggle,
            (1, 2, 3, 4),
        )

    def test_public_state_and_actions_delegate_to_controller(self) -> None:
        self.center.start_timer()
        self.center.pause_timer()
        self.center.resume_timer()
        self.center.set_explored_percent(55)
        self.center._controller.handle_click.return_value = (
            "drone_path",
            0,
        )

        action = self.center.handle_click((10, 20))

        self.center._controller.start_timer.assert_called_once_with()
        self.center._controller.pause_timer.assert_called_once_with()
        self.center._controller.resume_timer.assert_called_once_with()
        self.center._controller.set_explored_percent.assert_called_once_with(
            55
        )
        self.center._controller.handle_click.assert_called_once_with(
            (10, 20),
            self.center._hit_map,
        )
        self.assertEqual(action, ("drone_path", 0))

    def test_percentage_color_public_helper_is_preserved(self) -> None:
        self.assertEqual(
            self.center.percent_color(10),
            Colors.RED.value,
        )
        self.assertEqual(
            self.center.percent_color(50),
            Colors.YELLOW.value,
        )
        self.assertEqual(
            self.center.percent_color(90),
            Colors.GREEN.value,
        )


if __name__ == "__main__":
    unittest.main()
