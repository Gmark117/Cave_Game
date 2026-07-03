import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np

from agents.drone_runtime_state import DroneRuntimeState
from rendering.control_center_view_model import (
    ControlCenterViewModel,
    build_drone_status_views,
    build_rover_status_views,
)


class ControlCenterViewModelTests(unittest.TestCase):
    def test_frame_view_model_is_immutable_and_detached(self) -> None:
        debug_lines = ["line"]
        view = ControlCenterViewModel(
            elapsed_time="01:02",
            explored_percent=25,
            active_tab="debug",
            drone_statuses=(),
            rover_statuses=(),
            show_terrain_heatmap=False,
            selected_drone_heatmap_id=None,
            debug_lines=debug_lines,
        )
        debug_lines.append("later")

        self.assertEqual(view.debug_lines, ("line",))
        with self.assertRaises(FrozenInstanceError):
            view.active_tab = "system"

    def test_drone_views_are_detached_and_reflect_live_state(self) -> None:
        drone = SimpleNamespace(
            id=0,
            color=(1, 2, 3),
        )
        state = DroneRuntimeState(
            start_position=(0, 0),
            cave=np.zeros((2, 2), dtype=np.uint8),
            direction=0,
            frontier_rebuild_cooldown=0.25,
        )

        initial = build_drone_status_views([drone], [state.snapshot()])

        state.set_battery(72)
        state.begin_exploration(0, [])
        state.toggle_path()
        deployed = build_drone_status_views([drone], [state.snapshot()])

        state.set_returning_home()
        homing = build_drone_status_views([drone], [state.snapshot()])

        state.mark_done()
        completed = build_drone_status_views([drone], [state.snapshot()])

        self.assertEqual(initial[0].name, "Blinky")
        self.assertEqual(initial[0].battery, 100)
        self.assertEqual(initial[0].status, "Ready")
        self.assertTrue(initial[0].show_path)
        with self.assertRaises(FrozenInstanceError):
            initial[0].battery = 0
        self.assertEqual(deployed[0].battery, 72)
        self.assertEqual(deployed[0].status, "Deployed")
        self.assertFalse(deployed[0].show_path)
        self.assertEqual(homing[0].status, "Homing")
        self.assertEqual(completed[0].status, "Done")

    def test_rover_views_are_detached_and_reflect_live_status(self) -> None:
        rover = SimpleNamespace(
            id=0,
            color=(4, 5, 6),
            battery=2400,
            status="Ready",
        )

        initial = build_rover_status_views([rover])

        rover.battery = 1800
        rover.status = "Updating"
        updated = build_rover_status_views([rover])

        self.assertEqual(initial[0].name, "Huey")
        self.assertEqual(initial[0].battery, 2400)
        self.assertEqual(initial[0].status, "Ready")
        self.assertEqual(updated[0].battery, 1800)
        self.assertEqual(updated[0].status, "Updating")


if __name__ == "__main__":
    unittest.main()
