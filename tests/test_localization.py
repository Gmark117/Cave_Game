import unittest

import numpy as np

from agents.drone_runtime_state import DroneRuntimeState
from mapping.localization import PerfectPoseLocalizer, PoseEstimate


class LocalizationTests(unittest.TestCase):
    def test_pose_estimate_normalizes_values_and_validates_confidence(self) -> None:
        pose = PoseEstimate(
            position=(2.7, 3.2),
            heading_deg=90,
            confidence=0.75,
            source="test",
            timestamp=12,
        )

        self.assertEqual(pose.position, (2, 3))
        self.assertEqual(pose.heading_deg, 90.0)
        self.assertEqual(pose.confidence, 0.75)
        self.assertEqual(pose.timestamp, 12.0)

        with self.assertRaises(ValueError):
            PoseEstimate(
                position=(0, 0),
                heading_deg=0.0,
                confidence=1.2,
                source="bad",
                timestamp=0.0,
            )

    def test_perfect_pose_localizer_uses_detached_runtime_snapshot(self) -> None:
        state = DroneRuntimeState(
            start_position=(1, 2),
            cave=np.zeros((8, 8), dtype=np.uint8),
            direction=0,
            frontier_rebuild_cooldown=0.25,
        )
        state.move_to((4, 2))
        snapshot = state.snapshot()

        estimate = PerfectPoseLocalizer().estimate(snapshot, timestamp=12.5)

        self.assertEqual(estimate.position, (4, 2))
        self.assertEqual(estimate.heading_deg, 90.0)
        self.assertEqual(estimate.confidence, 1.0)
        self.assertEqual(estimate.source, "perfect-runtime")
        self.assertEqual(estimate.timestamp, 12.5)
        self.assertEqual(state.snapshot().position, (4, 2))


if __name__ == "__main__":
    unittest.main()
