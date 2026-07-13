import unittest
from unittest.mock import patch

import numpy as np

from agents.drone_runtime_state import DroneSnapshot
from agents.exploration_policy import (
    ExplorationContext,
    ExplorationDecisionKind,
    FrontierExplorationPolicy,
)
from asset_config.helpers import next_cell_coords
from mapping.localization import PoseEstimate
from mapping.slam_map import FREE, UNKNOWN, SlamSnapshot
from mapping.terrain_knowledge import TerrainSnapshot


def make_runtime_snapshot(
    *,
    position=(16, 16),
    frontiers=(),
    battery=100,
    returning_home=False,
) -> DroneSnapshot:
    return DroneSnapshot(
        position=position,
        direction=0,
        direction_history=(),
        path_history=(position,),
        frontiers=frontiers,
        returning_home=returning_home,
        done=False,
        explored=True,
        heading_deg=0.0,
        ray_points=(),
        battery=battery,
        show_path=True,
        show_vision=True,
        frontier_rebuild_cooldown=0.25,
        last_frontier_rebuild=0.0,
    )


def make_context(
    *,
    runtime_snapshot=None,
    slam_snapshot=None,
    terrain_snapshot=None,
) -> ExplorationContext:
    runtime = runtime_snapshot or make_runtime_snapshot()
    cave = np.zeros((32, 32), dtype=np.uint8)
    return ExplorationContext(
        pose_estimate=PoseEstimate(
            position=runtime.position,
            heading_deg=runtime.heading_deg,
            confidence=1.0,
            source="test",
            timestamp=1.0,
        ),
        runtime_snapshot=runtime,
        cave_map=cave,
        start_position=(16, 16),
        step=10,
        radius=5,
        map_width=32,
        frontier_stride=4,
        frontier_confidence_threshold=0.6,
        battery=runtime.battery,
        slam_snapshot=slam_snapshot,
        terrain_snapshot=terrain_snapshot,
    )


class ExplorationPolicyTests(unittest.TestCase):
    def test_decide_returns_homing_when_runtime_is_returning_home(self) -> None:
        runtime = make_runtime_snapshot(returning_home=True)
        context = make_context(runtime_snapshot=runtime)

        decision = FrontierExplorationPolicy().decide(
            context,
            lambda current, candidate: True,
        )

        self.assertEqual(decision.kind, ExplorationDecisionKind.HOMING)
        self.assertEqual(decision.target, context.start_position)

    def test_decide_returns_frontier_when_local_step_is_exhausted(self) -> None:
        runtime = make_runtime_snapshot(
            frontiers=((18, 16), (26, 16)),
        )
        context = make_context(runtime_snapshot=runtime)

        decision = FrontierExplorationPolicy().decide(
            context,
            lambda current, candidate: False,
        )

        self.assertEqual(decision.kind, ExplorationDecisionKind.FRONTIER)
        self.assertEqual(decision.target, (26, 16))
        self.assertEqual(decision.frontier_targets, ((26, 16), (18, 16)))

    def test_decide_returns_exhausted_when_no_step_or_frontier_exists(self) -> None:
        context = make_context()

        decision = FrontierExplorationPolicy().decide(
            context,
            lambda current, candidate: False,
        )

        self.assertEqual(decision.kind, ExplorationDecisionKind.EXHAUSTED)
        self.assertIsNone(decision.target)

    def test_choose_next_step_keeps_integer_degree_candidates(self) -> None:
        context = make_context()
        calls = []

        def is_valid(current, candidate) -> bool:
            calls.append((current, candidate))
            return True

        with patch("agents.exploration_policy.rand.choice", return_value=90):
            decision = FrontierExplorationPolicy().choose_next_step(
                context,
                is_valid,
            )

        self.assertEqual(decision.kind, ExplorationDecisionKind.STEP)
        self.assertEqual(decision.direction, 90)
        self.assertEqual(
            decision.target,
            next_cell_coords(16, 16, 10, 90),
        )
        self.assertEqual(decision.valid_directions, tuple(range(360)))
        self.assertEqual(len(decision.frontier_targets), 360)
        self.assertEqual(len(calls), 361)

    def test_choose_next_step_reports_exhausted_when_no_direction_is_valid(self) -> None:
        context = make_context()

        decision = FrontierExplorationPolicy().choose_next_step(
            context,
            lambda current, candidate: False,
        )

        self.assertEqual(decision.kind, ExplorationDecisionKind.EXHAUSTED)
        self.assertIsNone(decision.target)

    def test_prioritize_frontiers_deprioritizes_already_visible_targets(self) -> None:
        runtime = make_runtime_snapshot(
            position=(10, 10),
            frontiers=((12, 10), (20, 10)),
        )
        context = make_context(runtime_snapshot=runtime)

        frontiers = FrontierExplorationPolicy().prioritize_frontiers(context)

        self.assertEqual(frontiers, ((20, 10), (12, 10)))

    def test_prioritize_frontiers_prefers_large_unexplored_cluster(
        self,
    ) -> None:
        occupancy = np.full((32, 32), FREE, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        occupancy[10, 11] = UNKNOWN
        confidence[10, 11] = 0.0
        occupancy[8:12, 24:28] = UNKNOWN
        confidence[8:12, 24:28] = 0.0
        runtime = make_runtime_snapshot(
            position=(0, 10),
            frontiers=((10, 10), (23, 10)),
        )
        context = make_context(
            runtime_snapshot=runtime,
            slam_snapshot=SlamSnapshot(occupancy, confidence),
            terrain_snapshot=TerrainSnapshot(
                np.full((32, 32), -1.0, dtype=np.float32),
                np.zeros((32, 32), dtype=np.float32),
            ),
        )

        frontiers = FrontierExplorationPolicy().prioritize_frontiers(context)

        self.assertEqual(frontiers[0], (23, 10))

    def test_prioritize_frontiers_prefers_low_confidence_over_small_pocket(
        self,
    ) -> None:
        occupancy = np.full((32, 32), FREE, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        occupancy[10, 11] = UNKNOWN
        confidence[10, 11] = 0.0
        confidence[9:12, 23:26] = 0.2
        runtime = make_runtime_snapshot(
            position=(0, 10),
            frontiers=((10, 10), (24, 10)),
        )
        context = make_context(
            runtime_snapshot=runtime,
            slam_snapshot=SlamSnapshot(occupancy, confidence),
            terrain_snapshot=TerrainSnapshot(
                np.full((32, 32), -1.0, dtype=np.float32),
                np.zeros((32, 32), dtype=np.float32),
            ),
        )

        frontiers = FrontierExplorationPolicy().prioritize_frontiers(context)

        self.assertEqual(frontiers[0], (24, 10))

    def test_extract_frontiers_uses_local_slam_and_terrain_snapshots(self) -> None:
        occupancy = np.full((32, 32), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((32, 32), dtype=np.float32)
        occupancy[20, 20] = FREE
        confidence[20, 20] = 1.0
        terrain_confidence = np.zeros((32, 32), dtype=np.float32)
        context = make_context(
            slam_snapshot=SlamSnapshot(occupancy, confidence),
            terrain_snapshot=TerrainSnapshot(
                np.full((32, 32), -1.0, dtype=np.float32),
                terrain_confidence,
            ),
        )

        frontiers = FrontierExplorationPolicy().extract_frontiers(
            context,
            stride=1,
            confidence_threshold=0.6,
        )

        self.assertEqual(frontiers, ((20, 20),))

    def test_extract_frontiers_can_use_local_terrain_confidence(self) -> None:
        occupancy = np.full((32, 32), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((32, 32), dtype=np.float32)
        occupancy[20, 20] = FREE
        confidence[20, 20] = 0.1
        terrain_confidence = np.zeros((32, 32), dtype=np.float32)
        terrain_confidence[20, 20] = 1.0
        context = make_context(
            slam_snapshot=SlamSnapshot(occupancy, confidence),
            terrain_snapshot=TerrainSnapshot(
                np.full((32, 32), -1.0, dtype=np.float32),
                terrain_confidence,
            ),
        )

        frontiers = FrontierExplorationPolicy().extract_frontiers(
            context,
            stride=1,
            confidence_threshold=0.6,
        )

        self.assertEqual(frontiers, ((20, 20),))


if __name__ == "__main__":
    unittest.main()
