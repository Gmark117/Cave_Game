import unittest
from dataclasses import replace
from unittest.mock import Mock

import numpy as np

from agents.drone_runtime_state import DroneSnapshot
from agents.exploration_policy import ExplorationContext, ExplorationDecisionKind
from agents.local_mcts_controller import (
    LocalMctsDecision,
    LocalMctsDiagnostics,
    LocalPrimitive,
    LocalRootDiagnostic,
)
from agents.mcts_exploration_policy import MctsExplorationPolicy
from config.simulation_config import ExplorationConfig
from mapping.localization import PoseEstimate
from mapping.slam_map import FREE, UNKNOWN, SlamSnapshot
from navigation.navigation_intent import MovementMode, NavigationIntent


def make_config(**overrides) -> ExplorationConfig:
    values = {
        "iterations": 12,
        "horizon": 4,
        "planning_rays": 5,
        "uct_exploration": 1.414,
        "discount": 0.95,
        "decision_time_budget_ms": 40.0,
    }
    values.update(overrides)
    return ExplorationConfig(**values)


def make_runtime_snapshot(
    *,
    position=(20, 20),
    heading=0.0,
    frontiers=(),
    cluster_ids=(),
    returning_home=False,
    intent=None,
) -> DroneSnapshot:
    return DroneSnapshot(
        position=position,
        direction=int(heading),
        direction_history=(),
        path_history=((20, 22), position),
        frontiers=tuple(frontiers),
        returning_home=returning_home,
        done=False,
        explored=True,
        heading_deg=heading,
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


def make_context(*, runtime=None, slam=True, version=3) -> ExplorationContext:
    runtime = runtime or make_runtime_snapshot()
    snapshot = None
    if slam:
        snapshot = SlamSnapshot(
            np.full((40, 40), FREE, dtype=np.int8),
            np.ones((40, 40), dtype=np.float32),
            version=version,
        )
    return ExplorationContext(
        pose_estimate=PoseEstimate(
            position=runtime.position,
            heading_deg=runtime.heading_deg,
            confidence=1.0,
            source="test",
            timestamp=1.0,
        ),
        runtime_snapshot=runtime,
        start_position=(20, 20),
        step=5,
        radius=5,
        frontier_confidence_threshold=0.6,
        slam_snapshot=snapshot,
    )


def make_intent(*, mode=MovementMode.TRAVEL) -> NavigationIntent:
    return NavigationIntent(
        intent_id=8,
        route_id=13,
        mode=mode,
        cluster_id=17,
        gateway_id=9,
        assignment_token=4,
        target=(20, 10),
        topology_revision=2,
        requester_knowledge_revision=3,
        route_node_ids=(1, 2),
        route_edge_ids=(8,),
        route_paths=(((20, 20), (20, 10)),),
        route_sources=("travelled",),
        route_segment_edge_ids=(8,),
        remaining_route_cost=10.0,
        selection_slam_version=3,
    )


def local_decision(primitive=LocalPrimitive.DEVIATE_LEFT) -> LocalMctsDecision:
    root = LocalRootDiagnostic(
        primitive=primitive,
        target=(19, 15),
        heading_deg=345.0,
        visits=3,
        mean_reward=1.25,
        initial_reward=1.0,
    )
    diagnostics = LocalMctsDiagnostics(
        iterations=7,
        root_visits=(root,),
        selected_primitive=primitive,
        selected_reward=1.25,
        generated_nodes=11,
        root_coverage_complete=True,
        overrun_stage=None,
        fallback_primitive=None,
        elapsed_ms=4.5,
        budget_ms=40.0,
        search_budget_ms=36.0,
        reserved_budget_ms=4.0,
        window_bounds=(0, 0, 40, 40),
        preprocessing_cells=1600,
        deadline_checks=(("root", 5),),
        slam_version=3,
    )
    return LocalMctsDecision(
        primitive=primitive,
        target=(19, 15),
        heading_deg=345.0,
        path=((20, 20), (19, 15)),
        diagnostics=diagnostics,
    )


class MctsExplorationPolicyTests(unittest.TestCase):
    def test_global_selection_remains_deterministic_and_does_not_run_mcts(self) -> None:
        runtime = make_runtime_snapshot(
            frontiers=((30, 20), (10, 20)),
            cluster_ids=(9, 2),
        )
        policy = MctsExplorationPolicy(make_config(), seed=7)
        policy._local_controller.decide = Mock(
            side_effect=AssertionError("global selection must not run local MCTS")
        )

        decision = policy.decide(make_context(runtime=runtime, version=19))

        self.assertEqual(decision.kind, ExplorationDecisionKind.FRONTIER)
        self.assertEqual(decision.cluster_id, 2)
        self.assertEqual(decision.target, (10, 20))
        self.assertFalse(policy.last_search_diagnostics.performed)
        self.assertEqual(policy.last_search_diagnostics.slam_version, 19)

    def test_local_adapter_forwards_only_detached_belief_and_active_intent(self) -> None:
        intent = make_intent()
        runtime = make_runtime_snapshot(
            frontiers=((20, 10),),
            cluster_ids=(17,),
            intent=intent,
        )
        context = make_context(runtime=runtime)
        policy = MctsExplorationPolicy(make_config(), seed=7)
        policy._local_controller.decide = Mock(return_value=local_decision())

        decision = policy.decide(context)

        request = policy._local_controller.decide.call_args.args[0]
        self.assertIs(request.intent, intent)
        self.assertIs(request.slam_snapshot, context.slam_snapshot)
        self.assertEqual(request.position, (20, 20))
        self.assertEqual(request.step, 5)
        self.assertEqual(decision.kind, ExplorationDecisionKind.STEP)
        self.assertEqual(decision.cluster_id, 17)
        self.assertEqual(decision.local_primitive, "deviate_left")
        self.assertEqual(decision.planned_path, ((20, 20), (19, 15)))

    def test_local_adapter_exposes_complete_search_diagnostics(self) -> None:
        intent = make_intent()
        runtime = make_runtime_snapshot(intent=intent)
        policy = MctsExplorationPolicy(make_config(), seed=7)
        policy._local_controller.decide = Mock(return_value=local_decision())

        policy.decide_local(make_context(runtime=runtime))
        diagnostics = policy.last_search_diagnostics

        self.assertTrue(diagnostics.performed)
        self.assertEqual(diagnostics.iterations, 7)
        self.assertTrue(diagnostics.root_coverage_complete)
        self.assertEqual(diagnostics.selected_kind, "deviate_left")
        self.assertEqual(diagnostics.selected_direction, 345)
        self.assertEqual(diagnostics.slam_version, 3)
        self.assertEqual(diagnostics.root_visits[0].kind, "deviate_left")

    def test_rotate_scan_maps_to_rotate_decision(self) -> None:
        intent = make_intent(mode=MovementMode.SCAN)
        runtime = make_runtime_snapshot(intent=intent, heading=30.0)
        policy = MctsExplorationPolicy(make_config(), seed=7)
        policy._local_controller.decide = Mock()

        decision = policy.decide_local(make_context(runtime=runtime))

        self.assertEqual(decision.kind, ExplorationDecisionKind.ROTATE)
        self.assertEqual(decision.direction, 90)
        self.assertEqual(decision.target, (20, 20))
        self.assertEqual(decision.planned_path, ((20, 20),))
        self.assertEqual(decision.local_primitive, "rotate_scan")
        policy._local_controller.decide.assert_not_called()
        self.assertFalse(policy.last_search_diagnostics.performed)
        self.assertEqual(
            policy.last_search_diagnostics.selected_kind,
            "rotate_scan",
        )

    def test_single_choice_recovery_uses_the_stored_intent_without_search(self) -> None:
        intent = make_intent(mode=MovementMode.RECOVERY)
        runtime = make_runtime_snapshot(intent=intent, heading=30.0)
        policy = MctsExplorationPolicy(make_config(), seed=7)
        policy._local_controller.decide = Mock()

        decision = policy.decide_local(make_context(runtime=runtime))

        self.assertEqual(decision.kind, ExplorationDecisionKind.STEP)
        self.assertEqual(decision.cluster_id, intent.cluster_id)
        self.assertEqual(decision.local_primitive, "recovery")
        policy._local_controller.decide.assert_not_called()
        self.assertFalse(policy.last_search_diagnostics.performed)
        self.assertEqual(
            policy.last_search_diagnostics.selected_kind,
            "recovery",
        )

    def test_missing_intent_or_belief_skips_local_search(self) -> None:
        policy = MctsExplorationPolicy(make_config(), seed=7)
        policy._local_controller.decide = Mock()
        intent_runtime = make_runtime_snapshot(intent=make_intent())

        without_intent = policy.decide_local(make_context())
        without_belief = policy.decide_local(
            make_context(runtime=intent_runtime, slam=False)
        )

        self.assertEqual(without_intent.kind, ExplorationDecisionKind.EXHAUSTED)
        self.assertEqual(without_belief.kind, ExplorationDecisionKind.EXHAUSTED)
        policy._local_controller.decide.assert_not_called()
        self.assertFalse(policy.last_search_diagnostics.performed)

    def test_window_provider_can_supply_belief_without_whole_snapshot(self) -> None:
        intent_runtime = make_runtime_snapshot(intent=make_intent())
        policy = MctsExplorationPolicy(make_config(), seed=7)
        policy._local_controller.decide = Mock(return_value=local_decision())
        provider = Mock(return_value=SlamSnapshot(
            np.full((8, 8), UNKNOWN, dtype=np.int8),
            np.zeros((8, 8), dtype=np.float32),
            version=23,
        ))

        decision = policy.decide_local(
            make_context(runtime=intent_runtime, slam=False),
            slam_snapshot_provider=provider,
            slam_shape=(40, 40),
            slam_version_hint=23,
        )

        request = policy._local_controller.decide.call_args.args[0]
        self.assertIsNone(request.slam_snapshot)
        self.assertIs(request.slam_snapshot_provider, provider)
        self.assertEqual(request.slam_shape, (40, 40))
        self.assertEqual(request.slam_version_hint, 23)
        self.assertEqual(decision.kind, ExplorationDecisionKind.STEP)


if __name__ == "__main__":
    unittest.main()
