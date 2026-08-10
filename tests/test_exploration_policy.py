import unittest
from dataclasses import replace

import numpy as np

from agents.drone_runtime_state import DroneSnapshot
from agents.exploration_policy import (
    ExplorationContext,
    ExplorationDecisionKind,
    FrontierExplorationPolicy,
)
from mapping.localization import PoseEstimate
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from navigation.navigation_intent import MovementMode, NavigationIntent


def make_runtime_snapshot(
    *,
    position=(16, 16),
    frontiers=(),
    cluster_ids=(),
    returning_home=False,
    intent=None,
) -> DroneSnapshot:
    return DroneSnapshot(
        position=position,
        direction=0,
        direction_history=(),
        path_history=(position,),
        frontiers=tuple(frontiers),
        returning_home=returning_home,
        done=False,
        explored=True,
        heading_deg=0.0,
        ray_points=(),
        battery=100,
        show_path=True,
        show_vision=True,
        frontier_rebuild_cooldown=0.25,
        last_frontier_rebuild=0.0,
        frontier_cluster_ids=tuple(cluster_ids),
        movement_mode=(
            MovementMode.TRAVEL if intent is None else intent.mode
        ),
        navigation_intent=intent,
    )


def make_context(
    *,
    runtime_snapshot=None,
    occupancy=None,
    confidence=None,
    version=4,
) -> ExplorationContext:
    runtime = runtime_snapshot or make_runtime_snapshot()
    if occupancy is None:
        occupancy = np.full((32, 32), UNKNOWN, dtype=np.int8)
    if confidence is None:
        confidence = np.zeros((32, 32), dtype=np.float32)
    return ExplorationContext(
        pose_estimate=PoseEstimate(
            position=runtime.position,
            heading_deg=runtime.heading_deg,
            confidence=1.0,
            source="test",
            timestamp=1.0,
        ),
        runtime_snapshot=runtime,
        start_position=(16, 16),
        step=10,
        radius=5,
        frontier_confidence_threshold=0.6,
        slam_snapshot=SlamSnapshot(
            occupancy,
            confidence,
            version=version,
        ),
    )


def travel_intent(*, mode=MovementMode.TRAVEL) -> NavigationIntent:
    return NavigationIntent(
        intent_id=41,
        route_id=17,
        mode=mode,
        cluster_id=77,
        gateway_id=12,
        assignment_token=5,
        target=(30, 16),
        topology_revision=9,
        requester_knowledge_revision=4,
        route_node_ids=(1, 2),
        route_edge_ids=(31,),
        route_paths=(((16, 16), (30, 16)),),
        route_sources=("travelled",),
        route_segment_edge_ids=(31,),
        remaining_route_cost=14.0,
        selection_slam_version=4,
    )


class ExplorationPolicyTests(unittest.TestCase):
    def test_frontier_local_prefix_uses_canonical_oriented_raster_ties(self) -> None:
        prefix = FrontierExplorationPolicy._intent_prefix(
            (0, 0),
            (((0, 0), (1, 2)),),
            edge_cursor=0,
            polyline_cursor=0,
            budget=4.0,
        )

        self.assertEqual(prefix, ((0, 0), (1, 1), (1, 2)))

    def test_global_selection_uses_stable_cluster_ids(self) -> None:
        runtime = make_runtime_snapshot(
            frontiers=((25, 16), (8, 16), (30, 16)),
            cluster_ids=(9, 2, 14),
        )

        decision = FrontierExplorationPolicy().decide(
            make_context(runtime_snapshot=runtime)
        )

        self.assertEqual(decision.kind, ExplorationDecisionKind.FRONTIER)
        self.assertEqual(decision.cluster_id, 2)
        self.assertEqual(decision.target, (8, 16))
        self.assertEqual(decision.frontier_cluster_ids, (2, 9, 14))
        self.assertEqual(
            decision.frontier_targets,
            ((8, 16), (25, 16), (30, 16)),
        )

    def test_raw_or_misaligned_frontier_coordinates_are_not_a_fallback(self) -> None:
        policy = FrontierExplorationPolicy()
        raw_only = make_runtime_snapshot(frontiers=((25, 16),))
        misaligned = make_runtime_snapshot(
            frontiers=((25, 16), (8, 16)),
            cluster_ids=(9,),
        )

        self.assertEqual(
            policy.decide(make_context(runtime_snapshot=raw_only)).kind,
            ExplorationDecisionKind.EXHAUSTED,
        )
        self.assertEqual(
            policy.decide(make_context(runtime_snapshot=misaligned)).kind,
            ExplorationDecisionKind.EXHAUSTED,
        )

    def test_global_selection_is_belief_only(self) -> None:
        runtime = make_runtime_snapshot(
            frontiers=((25, 16), (8, 16)),
            cluster_ids=(9, 2),
        )
        unknown = np.full((32, 32), UNKNOWN, dtype=np.int8)
        occupied = np.full((32, 32), OCCUPIED, dtype=np.int8)
        low = np.zeros((32, 32), dtype=np.float32)
        high = np.ones((32, 32), dtype=np.float32)
        baseline = make_context(
            runtime_snapshot=runtime,
            occupancy=unknown,
            confidence=low,
        )
        changed_belief = make_context(
            runtime_snapshot=runtime,
            occupancy=occupied,
            confidence=high,
            version=99,
        )

        self.assertEqual(
            FrontierExplorationPolicy().decide(changed_belief),
            FrontierExplorationPolicy().decide(baseline),
        )

    def test_returning_home_preempts_cluster_selection(self) -> None:
        runtime = make_runtime_snapshot(
            frontiers=((25, 16),),
            cluster_ids=(9,),
            returning_home=True,
        )

        decision = FrontierExplorationPolicy().decide(
            make_context(runtime_snapshot=runtime)
        )

        self.assertEqual(decision.kind, ExplorationDecisionKind.HOMING)
        self.assertEqual(decision.target, (16, 16))

    def test_local_travel_follows_active_route_without_goal_reselection(self) -> None:
        intent = travel_intent()
        context = make_context(
            runtime_snapshot=make_runtime_snapshot(
                frontiers=((30, 16),),
                cluster_ids=(77,),
                intent=intent,
            )
        )
        policy = FrontierExplorationPolicy()

        first = policy.decide(context)
        second = policy.decide_local(context)

        self.assertEqual(second, first)
        self.assertEqual(first.kind, ExplorationDecisionKind.STEP)
        self.assertEqual(first.cluster_id, 77)
        self.assertEqual(first.target, (26, 16))
        self.assertEqual(first.direction, 90)
        self.assertEqual(first.local_primitive, "follow_edge")
        self.assertEqual(first.planned_path[-1], first.target)

    def test_local_scan_and_recovery_map_to_bounded_primitives(self) -> None:
        policy = FrontierExplorationPolicy()
        scan_intent = replace(
            travel_intent(mode=MovementMode.SCAN),
            route_paths=(),
            route_sources=(),
            route_segment_edge_ids=(),
            remaining_route_cost=0.0,
            scan_heading_cursor=2,
        )
        recovery_intent = replace(
            travel_intent(mode=MovementMode.RECOVERY),
            cluster_id=None,
            gateway_id=None,
            assignment_token=None,
            target=(6, 16),
            route_node_ids=(),
            route_edge_ids=(),
            route_paths=(((16, 16), (6, 16)),),
            route_segment_edge_ids=(None,),
            remaining_route_cost=10.0,
        )

        scan = policy.decide_local(make_context(
            runtime_snapshot=make_runtime_snapshot(intent=scan_intent)
        ))
        recovery = policy.decide_local(make_context(
            runtime_snapshot=make_runtime_snapshot(intent=recovery_intent)
        ))

        self.assertEqual(scan.kind, ExplorationDecisionKind.ROTATE)
        self.assertEqual(scan.direction, 180)
        self.assertEqual(scan.local_primitive, "rotate_scan")
        self.assertEqual(recovery.kind, ExplorationDecisionKind.STEP)
        self.assertEqual(recovery.target, (6, 16))
        self.assertEqual(recovery.direction, 270)
        self.assertEqual(recovery.local_primitive, "recovery")


if __name__ == "__main__":
    unittest.main()
