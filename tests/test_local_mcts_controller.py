import unittest
from dataclasses import replace

import numpy as np

from agents.local_mcts_controller import (
    LOCAL_PLANNING_RAY_CAP,
    LocalMctsController,
    LocalMctsRequest,
    LocalPrimitive,
    LocalRewardComponents,
    _DeadlineGuard,
)
from config.simulation_config import ExplorationConfig
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from navigation.navigation_intent import MovementMode, NavigationIntent


class IncrementingClock:
    """Deterministic clock that charges every cooperative deadline check."""

    def __init__(self, increment: float) -> None:
        self.increment = float(increment)
        self.now = -self.increment
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        self.now += self.increment
        return self.now


class SwitchClock:
    """Clock whose deadline state can be changed by an instrumented search."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make_config(**overrides) -> ExplorationConfig:
    values = {
        "iterations": 8,
        "horizon": 3,
        "planning_rays": 3,
        "decision_time_budget_ms": 0.0,
    }
    values.update(overrides)
    return ExplorationConfig(**values)


def make_request(
    *,
    shape=(96, 96),
    position=(48, 48),
    heading=0.0,
    mode=MovementMode.TRAVEL,
    occupancy=None,
    confidence=None,
    route_paths=None,
    edge_cursor=0,
    polyline_cursor=0,
    recent_visits=(),
    previous_primitive=None,
    stalled=False,
) -> LocalMctsRequest:
    if occupancy is None:
        occupancy = np.full(shape, FREE, dtype=np.int8)
    if confidence is None:
        confidence = np.ones(shape, dtype=np.float32)
    if route_paths is None:
        route_paths = (((position[0], position[1]), (position[0], position[1] - 24)),)
    intent = NavigationIntent(
        mode=mode,
        cluster_id=77,
        gateway_id=12,
        assignment_token=5,
        target=route_paths[-1][-1],
        topology_revision=9,
        requester_knowledge_revision=4,
        route_node_ids=(1, 2),
        route_edge_ids=(31,),
        route_paths=tuple(route_paths),
        route_sources=("travelled",) * len(route_paths),
        route_segment_edge_ids=(31,) * len(route_paths),
        edge_cursor=edge_cursor,
        polyline_cursor=polyline_cursor,
        remaining_route_cost=24.0,
        selection_slam_version=4,
    )
    return LocalMctsRequest(
        position=position,
        heading_deg=heading,
        step=6,
        radius=4,
        slam_snapshot=SlamSnapshot(
            occupancy,
            confidence,
            version=4,
        ),
        intent=intent,
        recent_visits=tuple(recent_visits),
        previous_primitive=previous_primitive,
        stalled=stalled,
        confidence_threshold=0.6,
    )


class LocalMctsControllerTests(unittest.TestCase):
    def test_preprocessing_is_inside_budget_and_uses_safe_fallback(self) -> None:
        clock = IncrementingClock(0.010)
        route = tuple((48, 48 - index) for index in range(30))
        controller = LocalMctsController(
            make_config(decision_time_budget_ms=40.0),
            seed=4,
            clock=clock,
        )

        decision = controller.decide(make_request(route_paths=(route,)))

        self.assertEqual(decision.primitive, LocalPrimitive.FOLLOW_EDGE)
        self.assertEqual(decision.diagnostics.iterations, 0)
        self.assertFalse(decision.diagnostics.root_coverage_complete)
        self.assertEqual(decision.diagnostics.overrun_stage, "preprocessing")
        self.assertGreaterEqual(clock.calls, 4)

    def test_busy_slam_window_uses_immediate_safe_fallback(self) -> None:
        controller = LocalMctsController(
            make_config(decision_time_budget_ms=40.0),
            seed=4,
            clock=lambda: 0.0,
        )
        request = replace(
            make_request(),
            slam_snapshot=None,
            slam_shape=(64, 64),
            slam_snapshot_provider=lambda _bounds: None,
        )

        decision = controller.decide(request)

        self.assertEqual(decision.primitive, LocalPrimitive.FOLLOW_EDGE)
        self.assertFalse(decision.diagnostics.root_coverage_complete)
        self.assertEqual(
            decision.diagnostics.overrun_stage,
            "preprocessing_lock",
        )
        self.assertEqual(decision.diagnostics.preprocessing_cells, 0)

    def test_complete_search_evaluates_every_bounded_root_before_uct(self) -> None:
        controller = LocalMctsController(
            make_config(iterations=5),
            seed=10,
            clock=lambda: 0.0,
        )

        decision = controller.decide(make_request())
        roots = decision.diagnostics.root_visits

        self.assertEqual(
            {root.primitive for root in roots},
            {
                LocalPrimitive.FOLLOW_EDGE,
                LocalPrimitive.DEVIATE_LEFT,
                LocalPrimitive.DEVIATE_RIGHT,
                LocalPrimitive.ROTATE_SCAN,
                LocalPrimitive.RECOVERY,
            },
        )
        self.assertTrue(decision.diagnostics.root_coverage_complete)
        self.assertTrue(all(root.visits >= 1 for root in roots))
        self.assertEqual(decision.diagnostics.iterations, 5)

    def test_single_root_scan_stops_after_complete_root_evaluation(self) -> None:
        controller = LocalMctsController(
            make_config(iterations=100, decision_time_budget_ms=40.0),
            seed=10,
            clock=lambda: 0.0,
        )

        decision = controller.decide(make_request(mode=MovementMode.SCAN))

        self.assertTrue(decision.diagnostics.root_coverage_complete)
        self.assertEqual(decision.diagnostics.iterations, 0)
        self.assertEqual(len(decision.diagnostics.root_visits), 1)
        self.assertEqual(
            decision.diagnostics.root_visits[0].primitive,
            LocalPrimitive.ROTATE_SCAN,
        )
        self.assertEqual(decision.diagnostics.root_visits[0].visits, 1)

    def test_single_root_recovery_stops_after_complete_root_evaluation(self) -> None:
        controller = LocalMctsController(
            make_config(iterations=100, decision_time_budget_ms=40.0),
            seed=10,
            clock=lambda: 0.0,
        )

        decision = controller.decide(
            make_request(mode=MovementMode.RECOVERY)
        )

        self.assertTrue(decision.diagnostics.root_coverage_complete)
        self.assertEqual(decision.diagnostics.iterations, 0)
        self.assertEqual(len(decision.diagnostics.root_visits), 1)
        self.assertEqual(
            decision.diagnostics.root_visits[0].primitive,
            LocalPrimitive.RECOVERY,
        )
        self.assertEqual(decision.diagnostics.root_visits[0].visits, 1)

    def test_budget_reserves_thirty_percent_for_scheduler_and_diagnostics(self) -> None:
        controller = LocalMctsController(
            make_config(iterations=1, decision_time_budget_ms=40.0),
            seed=10,
            clock=lambda: 0.0,
        )

        diagnostics = controller.decide(make_request()).diagnostics

        self.assertEqual(diagnostics.budget_ms, 40.0)
        self.assertEqual(diagnostics.search_budget_ms, 28.0)
        self.assertEqual(diagnostics.reserved_budget_ms, 12.0)

    def test_root_deviations_are_one_bounded_step_at_plus_minus_fifteen(self) -> None:
        controller = LocalMctsController(
            make_config(iterations=1),
            seed=10,
            clock=lambda: 0.0,
        )

        roots = controller.decide(make_request(heading=0.0)).diagnostics.root_visits
        by_primitive = {root.primitive: root for root in roots}
        left = by_primitive[LocalPrimitive.DEVIATE_LEFT]
        right = by_primitive[LocalPrimitive.DEVIATE_RIGHT]

        self.assertAlmostEqual(left.heading_deg, 345.0)
        self.assertAlmostEqual(right.heading_deg, 15.0)
        self.assertLessEqual(np.hypot(left.target[0] - 48, left.target[1] - 48), 6.0)
        self.assertLessEqual(np.hypot(right.target[0] - 48, right.target[1] - 48), 6.0)

    def test_deadline_during_ray_loop_forces_no_search_iteration(self) -> None:
        clock = IncrementingClock(0.00035)
        controller = LocalMctsController(
            make_config(
                iterations=100,
                planning_rays=5,
                decision_time_budget_ms=40.0,
            ),
            seed=3,
            clock=clock,
        )

        decision = controller.decide(make_request(shape=(192, 192), position=(96, 96)))

        self.assertEqual(decision.diagnostics.iterations, 0)
        self.assertFalse(decision.diagnostics.root_coverage_complete)
        self.assertEqual(decision.diagnostics.overrun_stage, "ray_cell")
        self.assertEqual(len(decision.diagnostics.root_visits), 5)
        self.assertTrue(any(
            root.visits == 0 for root in decision.diagnostics.root_visits
        ))

    def test_partial_uct_rollout_is_not_backpropagated_after_deadline(self) -> None:
        clock = SwitchClock()

        class PartialRolloutController(LocalMctsController):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.inside_uct = False

            def _select_uct_arm(self, arms):
                self.inside_uct = True
                return super()._select_uct_arm(arms)

            def _evaluate_action(
                self,
                request,
                window,
                state,
                action,
                guard,
            ):
                if self.inside_uct and state.depth > 0:
                    clock.now = 1.0
                return super()._evaluate_action(
                    request,
                    window,
                    state,
                    action,
                    guard,
                )

        controller = PartialRolloutController(
            make_config(
                iterations=1,
                horizon=2,
                decision_time_budget_ms=40.0,
            ),
            seed=3,
            clock=clock,
        )

        decision = controller.decide(make_request())

        self.assertTrue(decision.diagnostics.root_coverage_complete)
        self.assertEqual(decision.diagnostics.overrun_stage, "expansion")
        self.assertEqual(decision.diagnostics.iterations, 0)
        self.assertIsNone(decision.diagnostics.fallback_primitive)
        self.assertTrue(
            all(root.visits == 1 for root in decision.diagnostics.root_visits)
        )

    def test_unknown_cell_produces_gain_and_occludes_cells_behind_it(self) -> None:
        occupancy = np.full((32, 32), FREE, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        occupancy[14, 16] = UNKNOWN
        confidence[14, 16] = 0.0
        occupancy[13, 16] = UNKNOWN
        confidence[13, 16] = 0.0
        controller = LocalMctsController(
            make_config(planning_rays=1),
            seed=1,
            clock=lambda: 0.0,
        )
        request = make_request(
            shape=(32, 32),
            position=(16, 16),
            occupancy=occupancy,
            confidence=confidence,
        )

        gain_cells = controller.predict_gain_cells(
            request,
            position=(16, 16),
            heading_deg=0.0,
        )

        self.assertEqual(gain_cells, ((16, 14),))
        self.assertNotIn((16, 13), gain_cells)

    def test_confident_occupied_cell_stops_ray_without_gain(self) -> None:
        occupancy = np.full((32, 32), FREE, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        occupancy[14, 16] = OCCUPIED
        occupancy[13, 16] = UNKNOWN
        confidence[13, 16] = 0.0
        controller = LocalMctsController(
            make_config(planning_rays=1),
            seed=1,
            clock=lambda: 0.0,
        )
        request = make_request(
            shape=(32, 32),
            position=(16, 16),
            occupancy=occupancy,
            confidence=confidence,
        )

        gain_cells = controller.predict_gain_cells(
            request,
            position=(16, 16),
            heading_deg=0.0,
        )

        self.assertEqual(gain_cells, ())

    def test_low_confidence_free_or_occupied_cell_gains_and_occludes(self) -> None:
        for occupancy_value in (FREE, OCCUPIED):
            with self.subTest(occupancy=occupancy_value):
                occupancy = np.full((32, 32), FREE, dtype=np.int8)
                confidence = np.ones((32, 32), dtype=np.float32)
                occupancy[14, 16] = occupancy_value
                confidence[14, 16] = 0.2
                occupancy[13, 16] = UNKNOWN
                confidence[13, 16] = 0.0
                controller = LocalMctsController(
                    make_config(planning_rays=1),
                    seed=1,
                    clock=lambda: 0.0,
                )
                request = make_request(
                    shape=(32, 32),
                    position=(16, 16),
                    occupancy=occupancy,
                    confidence=confidence,
                )

                gain_cells = controller.predict_gain_cells(
                    request,
                    position=(16, 16),
                    heading_deg=0.0,
                )

                self.assertEqual(gain_cells, ((16, 14),))
                self.assertNotIn((16, 13), gain_cells)

    def test_reward_uses_the_locked_normalized_weighting(self) -> None:
        components = LocalRewardComponents(
            route_progress=0.8,
            information_gain=0.7,
            revisit=0.6,
            oscillation=0.5,
            target_switch=0.4,
            turn=0.3,
            time_energy=0.2,
            collision_risk=0.1,
        )

        score = components.score

        self.assertAlmostEqual(
            score,
            2 * 0.8
            + 3 * 0.7
            - 1.5 * 0.6
            - 2 * 0.5
            - 2.5 * 0.4
            - 0.25 * 0.3
            - 0.25 * 0.2
            - 4 * 0.1,
        )
        for name in ("route_progress", "information_gain"):
            improved = replace(components, **{name: 1.0})
            self.assertGreater(improved.score, score)
        for name in (
            "revisit",
            "oscillation",
            "target_switch",
            "turn",
            "time_energy",
            "collision_risk",
        ):
            penalized = replace(components, **{name: 1.0})
            self.assertLess(penalized.score, score)

    def test_oscillation_requires_an_actual_opposite_previous_primitive(self) -> None:
        controller = LocalMctsController(
            make_config(iterations=1),
            seed=5,
            clock=lambda: 0.0,
        )
        request = make_request()
        state = controller.build_search_state(request)
        guard = _DeadlineGuard(lambda: 0.0, None)
        window = controller._build_local_window(request, guard)
        actions = controller._build_actions(request, window, state)

        for action in actions:
            with self.subTest(primitive=action.primitive):
                components, _ = controller._evaluate_action(
                    request,
                    window,
                    state,
                    action,
                    guard,
                )
                self.assertEqual(components.oscillation, 0.0)
                if action.primitive in {
                    LocalPrimitive.DEVIATE_LEFT,
                    LocalPrimitive.DEVIATE_RIGHT,
                }:
                    self.assertEqual(components.route_progress, 0.0)

        state_after_left = replace(
            state,
            previous_primitive=LocalPrimitive.DEVIATE_LEFT,
        )
        by_primitive = {
            action.primitive: action
            for action in controller._build_actions(
                request,
                window,
                state_after_left,
            )
        }
        right_components, _ = controller._evaluate_action(
            request,
            window,
            state_after_left,
            by_primitive[LocalPrimitive.DEVIATE_RIGHT],
            guard,
        )
        left_components, _ = controller._evaluate_action(
            request,
            window,
            state_after_left,
            by_primitive[LocalPrimitive.DEVIATE_LEFT],
            guard,
        )

        self.assertEqual(right_components.oscillation, 1.0)
        self.assertEqual(left_components.oscillation, 0.0)

    def test_root_gain_cells_propagate_and_cannot_pay_twice(self) -> None:
        class GainPropagationController(LocalMctsController):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.current_root = None
                self.root_gains = {}
                self.root_transition_count = 0
                self.repeated_gains = set()

            def _evaluate_action(
                self,
                request,
                window,
                state,
                action,
                guard,
            ):
                components, gains = super()._evaluate_action(
                    request,
                    window,
                    state,
                    action,
                    guard,
                )
                if state.depth == 0:
                    self.root_gains[action.primitive] = gains
                elif self.current_root is not None:
                    self.repeated_gains.update(
                        gains & self.root_gains.get(self.current_root, frozenset())
                    )
                return components, gains

            def _transition_state(
                self,
                state,
                action,
                reward,
                *,
                gain_cells=(),
            ):
                if state.depth == 0:
                    self.current_root = action.primitive
                    self.root_transition_count += 1
                    expected = self.root_gains.get(action.primitive, frozenset())
                    if frozenset(gain_cells) != expected:
                        raise AssertionError("root gain cells were not propagated")
                return super()._transition_state(
                    state,
                    action,
                    reward,
                    gain_cells=gain_cells,
                )

            def _choose_simulation_action(self, actions, rng):
                return next(
                    action
                    for action in actions
                    if action.primitive == self.current_root
                )

        occupancy = np.full((96, 96), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((96, 96), dtype=np.float32)
        controller = GainPropagationController(
            make_config(iterations=1, horizon=2, planning_rays=1),
            seed=9,
            clock=lambda: 0.0,
        )

        decision = controller.decide(make_request(
            occupancy=occupancy,
            confidence=confidence,
        ))

        self.assertTrue(decision.diagnostics.root_coverage_complete)
        self.assertEqual(
            controller.root_transition_count,
            len(decision.diagnostics.root_visits) + 1,
        )
        self.assertTrue(controller.root_gains[LocalPrimitive.RECOVERY])
        self.assertEqual(controller.repeated_gains, set())

    def test_full_root_rollouts_finish_before_first_uct_selection(self) -> None:
        class CoverageSpyController(LocalMctsController):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.first_uct_snapshot = None

            def _evaluate_action(
                self,
                request,
                window,
                state,
                action,
                guard,
            ):
                route_progress = 0.25 if state.depth == 0 else 0.5
                return LocalRewardComponents(
                    route_progress=route_progress,
                ), frozenset()

            def _select_uct_arm(self, arms):
                if self.first_uct_snapshot is None:
                    self.first_uct_snapshot = tuple(
                        (
                            arm.visits,
                            arm.initial_reward,
                            arm.immediate_reward,
                        )
                        for arm in arms
                    )
                return super()._select_uct_arm(arms)

        controller = CoverageSpyController(
            make_config(iterations=1, horizon=2, discount=0.5),
            seed=2,
            clock=lambda: 0.0,
        )

        decision = controller.decide(make_request())

        self.assertTrue(decision.diagnostics.root_coverage_complete)
        self.assertIsNotNone(controller.first_uct_snapshot)
        for visits, initial_reward, immediate_reward in controller.first_uct_snapshot:
            self.assertEqual(visits, 1)
            self.assertGreater(initial_reward, immediate_reward)
        self.assertEqual(decision.diagnostics.iterations, 1)
        self.assertEqual(
            sum(root.visits for root in decision.diagnostics.root_visits),
            len(decision.diagnostics.root_visits) + 1,
        )

    def test_search_state_carries_goal_cursor_visits_primitive_and_stall(self) -> None:
        request = make_request(
            edge_cursor=2,
            polyline_cursor=7,
            recent_visits=(31, 44, 31),
            previous_primitive=LocalPrimitive.DEVIATE_LEFT,
            stalled=True,
        )
        controller = LocalMctsController(make_config(), seed=1)

        state = controller.build_search_state(request)

        self.assertEqual(state.goal_id, 77)
        self.assertEqual(state.route_edge_cursor, 2)
        self.assertEqual(state.polyline_cursor, 7)
        self.assertEqual(state.recent_visits, (31, 44, 31))
        self.assertEqual(state.previous_primitive, LocalPrimitive.DEVIATE_LEFT)
        self.assertTrue(state.stalled)

    def test_curved_suffix_and_connector_cursor_progress_are_monotonic(self) -> None:
        exact_suffix, capped = LocalMctsController._raster_suffix_after_cursor(
            ((0, 0), (2, 3)),
            1,
            (1, 1),
            _DeadlineGuard(lambda: 0.0, None),
        )
        self.assertFalse(capped)
        self.assertEqual(exact_suffix, ((1, 1), (1, 2), (2, 3)))
        self.assertNotIn((2, 2), exact_suffix)

        boundary_request = replace(
            make_request(
                shape=(32, 32),
                position=(10, 10),
                route_paths=(
                    ((10, 10), (12, 10)),
                    ((12, 10), (15, 10)),
                ),
            ),
            step=2,
        )
        boundary_controller = LocalMctsController(
            make_config(iterations=1),
            seed=8,
            clock=lambda: 0.0,
        )
        boundary_state = boundary_controller.build_search_state(boundary_request)
        boundary_window = boundary_controller._build_local_window(
            boundary_request,
            _DeadlineGuard(lambda: 0.0, None),
        )
        boundary_follow = next(
            action
            for action in boundary_controller._build_actions(
                boundary_request,
                boundary_window,
                boundary_state,
            )
            if action.primitive == LocalPrimitive.FOLLOW_EDGE
        )
        self.assertEqual(
            boundary_follow.path,
            ((10, 10), (11, 10), (12, 10)),
        )
        self.assertEqual(boundary_follow.result_edge_cursor, 1)
        self.assertEqual(boundary_follow.result_polyline_cursor, 0)

        position = (20, 17)
        first_segment = ((20, 20), (20, 16), (22, 16))
        connector = ((22, 16), (25, 16))
        request = make_request(
            shape=(48, 48),
            position=position,
            route_paths=(first_segment, connector),
            polyline_cursor=3,
        )
        request = replace(
            request,
            intent=replace(
                request.intent,
                target=(25, 16),
                route_edge_ids=(31,),
                route_sources=("travelled", "belief_connector"),
                route_segment_edge_ids=(31, None),
                remaining_route_cost=6.0,
            ),
        )
        controller = LocalMctsController(
            make_config(iterations=1, horizon=2),
            seed=8,
            clock=lambda: 0.0,
        )
        state = controller.build_search_state(request)
        guard = _DeadlineGuard(lambda: 0.0, None)
        window = controller._build_local_window(request, guard)
        follow = next(
            action
            for action in controller._build_actions(request, window, state)
            if action.primitive == LocalPrimitive.FOLLOW_EDGE
        )

        self.assertEqual(
            follow.path,
            (
                (20, 17),
                (20, 16),
                (21, 16),
                (22, 16),
                (23, 16),
                (24, 16),
                (25, 16),
            ),
        )
        self.assertNotIn((20, 18), follow.path)
        self.assertEqual(follow.result_edge_cursor, 2)
        self.assertEqual(follow.result_polyline_cursor, 0)

        advanced = controller._transition_state(state, follow, reward=1.0)

        self.assertEqual(advanced.position, (25, 16))
        self.assertEqual(advanced.route_edge_cursor, 2)
        self.assertEqual(advanced.polyline_cursor, 0)
        self.assertEqual(advanced.local_route_cursor, 6)
        self.assertEqual(advanced.remaining_route_cost, 0.0)

    def test_local_window_stays_bounded_on_a_large_map(self) -> None:
        controller = LocalMctsController(
            make_config(iterations=1),
            seed=8,
            clock=lambda: 0.0,
        )

        decision = controller.decide(
            make_request(shape=(1200, 1200), position=(600, 600))
        )
        left, top, right, bottom = decision.diagnostics.window_bounds

        self.assertLessEqual(right - left, 193)
        self.assertLessEqual(bottom - top, 193)
        self.assertEqual(
            decision.diagnostics.preprocessing_cells,
            (right - left) * (bottom - top),
        )
        self.assertLess(decision.diagnostics.preprocessing_cells, 1200 * 1200)

    def test_extreme_configured_ray_count_is_capped_locally(self) -> None:
        controller = LocalMctsController(
            make_config(
                iterations=1,
                horizon=1,
                planning_rays=1_000_000,
            ),
            seed=8,
            clock=lambda: 0.0,
        )

        decision = controller.decide(make_request())
        checks = dict(decision.diagnostics.deadline_checks)
        maximum_range = max(4 * 4, 6 * 2, 1)
        maximum_ray_checks = (
            len(decision.diagnostics.root_visits)
            * LOCAL_PLANNING_RAY_CAP
            * (maximum_range + 1)
        )

        self.assertEqual(
            controller._planning_ray_count(),
            LOCAL_PLANNING_RAY_CAP,
        )
        self.assertTrue(decision.diagnostics.root_coverage_complete)
        self.assertLessEqual(checks.get("ray_cell", 0), maximum_ray_checks)

    def test_remote_belief_changes_do_not_change_local_search(self) -> None:
        occupancy = np.full((600, 600), FREE, dtype=np.int8)
        confidence = np.ones((600, 600), dtype=np.float32)
        changed_occupancy = occupancy.copy()
        changed_confidence = confidence.copy()
        changed_occupancy[:100, :100] = UNKNOWN
        changed_confidence[:100, :100] = 0.0
        config = make_config(iterations=6)

        first = LocalMctsController(
            config,
            seed=12,
            clock=lambda: 0.0,
        ).decide(make_request(
            shape=(600, 600),
            position=(300, 300),
            occupancy=occupancy,
            confidence=confidence,
        ))
        second_request = make_request(
            shape=(600, 600),
            position=(300, 300),
            occupancy=changed_occupancy,
            confidence=changed_confidence,
        )
        second_request = replace(
            second_request,
            slam_snapshot=replace(second_request.slam_snapshot, version=99),
        )
        second = LocalMctsController(
            config,
            seed=12,
            clock=lambda: 0.0,
        ).decide(second_request)

        self.assertEqual(second.primitive, first.primitive)
        self.assertEqual(second.target, first.target)
        self.assertEqual(second.heading_deg, first.heading_deg)
        self.assertEqual(second.path, first.path)
        self.assertEqual(
            replace(
                second.diagnostics,
                slam_version=first.diagnostics.slam_version,
            ),
            first.diagnostics,
        )

    def test_deadline_fallback_is_mode_specific_and_deterministic(self) -> None:
        expected = {
            MovementMode.TRAVEL: LocalPrimitive.FOLLOW_EDGE,
            MovementMode.SCAN: LocalPrimitive.ROTATE_SCAN,
            MovementMode.RECOVERY: LocalPrimitive.RECOVERY,
        }
        for mode, primitive in expected.items():
            with self.subTest(mode=mode):
                decision = LocalMctsController(
                    make_config(decision_time_budget_ms=40.0),
                    seed=3,
                    clock=IncrementingClock(0.040),
                ).decide(make_request(mode=mode))

                self.assertEqual(decision.primitive, primitive)
                self.assertEqual(decision.diagnostics.iterations, 0)
                self.assertEqual(
                    decision.diagnostics.fallback_primitive,
                    primitive,
                )

    def test_fixed_seed_and_request_produce_identical_decision_and_diagnostics(self) -> None:
        request = make_request(
            previous_primitive=LocalPrimitive.DEVIATE_RIGHT,
            recent_visits=((1, 1), (2, 1), (1, 1)),
        )
        first = LocalMctsController(
            make_config(iterations=12),
            seed=29,
            clock=lambda: 0.0,
        ).decide(request)
        second = LocalMctsController(
            make_config(iterations=12),
            seed=29,
            clock=lambda: 0.0,
        ).decide(request)

        self.assertEqual(second, first)


if __name__ == "__main__":
    unittest.main()
