import random
import unittest
from dataclasses import replace

import numpy as np

from agents.drone_runtime_state import DroneSnapshot
from agents.exploration_policy import ExplorationContext, ExplorationDecisionKind
from agents.mcts_exploration_policy import (
    MctsExplorationPolicy,
    MctsNode,
    SearchAction,
    SearchState,
    TRANSLATE,
)
from config.simulation_config import ExplorationConfig, SimulationConfig
from mapping.localization import PoseEstimate
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from mapping.terrain_knowledge import TerrainSnapshot


def make_config(**overrides) -> ExplorationConfig:
    values = {
        "iterations": 32,
        "horizon": 4,
        "branching_factor": 6,
        "frontier_cluster_limit": 4,
        "planning_rays": 5,
        "uct_exploration": 1.414,
        "discount": 0.95,
        "rollout_temperature": 0.35,
    }
    values.update(overrides)
    return ExplorationConfig(**values)


def make_runtime_snapshot(
    *,
    position=(20, 20),
    heading=0.0,
    returning_home=False,
) -> DroneSnapshot:
    return DroneSnapshot(
        position=position,
        direction=int(heading),
        direction_history=(),
        path_history=(position,),
        frontiers=(),
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
    )


def make_context(
    *,
    occupancy=None,
    confidence=None,
    position=(20, 20),
    heading=0.0,
    step=5,
    radius=5,
    version=3,
    frontiers=(),
) -> ExplorationContext:
    if occupancy is None:
        occupancy = np.full((40, 40), UNKNOWN, dtype=np.int8)
    if confidence is None:
        confidence = np.zeros_like(occupancy, dtype=np.float32)
    runtime = make_runtime_snapshot(position=position, heading=heading)
    runtime = replace(runtime, frontiers=tuple(frontiers))
    terrain_confidence = np.zeros_like(confidence, dtype=np.float32)
    return ExplorationContext(
        pose_estimate=PoseEstimate(
            position=position,
            heading_deg=heading,
            confidence=1.0,
            source="test",
            timestamp=1.0,
        ),
        runtime_snapshot=runtime,
        cave_map=np.zeros_like(occupancy, dtype=np.uint8),
        start_position=position,
        step=step,
        radius=radius,
        map_width=occupancy.shape[1],
        frontier_stride=1,
        frontier_confidence_threshold=0.6,
        battery=runtime.battery,
        slam_snapshot=SlamSnapshot(occupancy, confidence, version=version),
        terrain_snapshot=TerrainSnapshot(
            np.full_like(confidence, -1.0, dtype=np.float32),
            terrain_confidence,
        ),
    )


def mark_free(
    occupancy: np.ndarray,
    confidence: np.ndarray,
    cells,
    *,
    value: float = 1.0,
) -> None:
    for x, y in cells:
        occupancy[y, x] = FREE
        confidence[y, x] = value


def known_square(
    *,
    shape=(40, 40),
    x_range=range(15, 26),
    y_range=range(15, 26),
) -> tuple[np.ndarray, np.ndarray]:
    occupancy = np.full(shape, UNKNOWN, dtype=np.int8)
    confidence = np.zeros(shape, dtype=np.float32)
    mark_free(
        occupancy,
        confidence,
        ((x, y) for x in x_range for y in y_range),
    )
    return occupancy, confidence


class MctsExplorationPolicyTests(unittest.TestCase):
    def test_exploration_config_defaults_and_validation(self) -> None:
        settings = SimulationConfig()

        self.assertEqual(settings.exploration.policy, "mcts")
        self.assertEqual(settings.exploration.iterations, 256)
        self.assertEqual(settings.exploration.horizon, 4)

        with self.assertRaises(ValueError):
            ExplorationConfig(policy="random")
        with self.assertRaises(ValueError):
            ExplorationConfig(iterations=0)

    def test_deterministic_decision_for_identical_seed_and_snapshot(self) -> None:
        occupancy, confidence = known_square()
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(20, 20),
            heading=90.0,
        )
        config = make_config(
            iterations=48,
            branching_factor=8,
            decision_time_budget_ms=0.0,
        )

        policy_a = MctsExplorationPolicy(config, seed=123)
        policy_b = MctsExplorationPolicy(config, seed=123)
        decision_a = policy_a.decide(
            context,
            lambda current, candidate: (_ for _ in ()).throw(
                AssertionError("validator must not be used during MCTS")
            ),
        )
        decision_b = policy_b.decide(
            context,
            lambda current, candidate: True,
        )

        self.assertEqual(decision_a.kind, decision_b.kind)
        self.assertEqual(decision_a.target, decision_b.target)
        self.assertEqual(decision_a.direction, decision_b.direction)
        self.assertEqual(decision_a.planned_path, decision_b.planned_path)

    def test_translation_requires_known_free_confident_cells(self) -> None:
        occupancy = np.full((16, 16), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((16, 16), dtype=np.float32)
        policy = MctsExplorationPolicy(make_config(), seed=1)
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(4, 4),
            step=3,
        )

        self.assertEqual(
            policy._known_free_path(context, (4, 4), (7, 4)),
            (),
        )

        mark_free(
            occupancy,
            confidence,
            [(4, 4), (5, 4), (6, 4), (7, 4)],
            value=0.5,
        )
        self.assertEqual(
            policy._known_free_path(context, (4, 4), (7, 4)),
            (),
        )

        mark_free(
            occupancy,
            confidence,
            [(4, 4), (5, 4), (6, 4), (7, 4)],
            value=1.0,
        )
        occupancy[4, 6] = OCCUPIED
        self.assertEqual(
            policy._known_free_path(context, (4, 4), (7, 4)),
            (),
        )

        occupancy[4, 6] = FREE
        self.assertEqual(
            policy._known_free_path(context, (4, 4), (-1, 4)),
            (),
        )
        self.assertEqual(
            policy._known_free_path(context, (4, 4), (7, 4)),
            ((5, 4), (6, 4), (7, 4)),
        )

    def test_frontier_clustering_augments_root_actions(self) -> None:
        occupancy = np.full((30, 30), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((30, 30), dtype=np.float32)
        mark_free(occupancy, confidence, [(26, 20), (27, 20), (5, 5)])
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(20, 20),
        )
        policy = MctsExplorationPolicy(
            make_config(frontier_cluster_limit=2, branching_factor=4),
            seed=1,
        )

        frontiers = policy._frontier_cluster_centroids(context, (20, 20))
        directions = policy._candidate_directions(
            SearchState((20, 20), 0, 0, frozenset()),
            random.Random(5),
            root_frontiers=frontiers,
        )

        self.assertEqual(len(frontiers), 2)
        self.assertIn(frontiers[0], ((26, 20), (27, 20)))
        self.assertEqual(
            directions[0],
            policy._direction_to((20, 20), frontiers[0]),
        )

    def test_budgeted_frontier_clustering_prefers_nearest_not_leftmost(
        self,
    ) -> None:
        occupancy = np.full((40, 40), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((40, 40), dtype=np.float32)
        mark_free(occupancy, confidence, [(5, 20), (30, 20)])
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(28, 20),
            heading=90.0,
            radius=1,
        )
        policy = MctsExplorationPolicy(
            make_config(frontier_cluster_limit=1),
            seed=1,
        )

        frontiers = policy._frontier_cluster_centroids(
            context,
            (28, 20),
            deadline=0.0,
        )

        self.assertEqual(frontiers, ((30, 20),))

    def test_rotation_selected_when_translation_blocked_but_scan_has_gain(self) -> None:
        occupancy = np.full((32, 32), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((32, 32), dtype=np.float32)
        mark_free(occupancy, confidence, [(16, 16)])
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(16, 16),
            heading=0.0,
        )
        policy = MctsExplorationPolicy(
            make_config(iterations=24, branching_factor=4, planning_rays=3),
            seed=7,
        )

        decision = policy.decide(context, lambda current, candidate: True)

        self.assertEqual(decision.kind, ExplorationDecisionKind.ROTATE)
        self.assertEqual(decision.target, (16, 16))
        self.assertIsNotNone(decision.direction)
        self.assertGreater(
            policy.last_search_diagnostics.selected_reward,
            0.0,
        )
        self.assertNotEqual(decision.direction, context.pose_estimate.heading_deg)

    def test_frontier_fallback_when_local_gain_is_exhausted(self) -> None:
        occupancy = np.full((32, 32), FREE, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(16, 16),
            heading=0.0,
            frontiers=((24, 16),),
        )
        policy = MctsExplorationPolicy(
            make_config(iterations=12, branching_factor=4),
            seed=3,
        )

        decision = policy.decide(context, lambda current, candidate: True)

        self.assertEqual(decision.kind, ExplorationDecisionKind.FRONTIER)
        self.assertEqual(decision.target, (24, 16))
        self.assertEqual(decision.frontier_targets, ((24, 16),))
        self.assertEqual(
            policy.last_search_diagnostics.selected_kind,
            "frontier",
        )

    def test_frontier_fallback_deprioritizes_too_close_root_frontiers(
        self,
    ) -> None:
        occupancy = np.full((40, 40), FREE, dtype=np.int8)
        confidence = np.ones((40, 40), dtype=np.float32)
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(10, 10),
            radius=5,
            frontiers=((30, 10),),
        )
        policy = MctsExplorationPolicy(
            make_config(frontier_cluster_limit=1),
            seed=3,
        )

        decision = policy._fallback_frontier_decision(
            context,
            root_frontiers=((12, 10),),
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.target, (30, 10))

    def test_frontier_fallback_limits_large_target_lists(self) -> None:
        occupancy = np.full((80, 80), FREE, dtype=np.int8)
        confidence = np.ones((80, 80), dtype=np.float32)
        frontiers = tuple(
            (x, y)
            for y in range(5, 75, 5)
            for x in range(5, 75, 5)
        )
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(40, 40),
            radius=10,
            frontiers=frontiers,
        )
        policy = MctsExplorationPolicy(
            make_config(frontier_cluster_limit=3),
            seed=3,
        )

        decision = policy._fallback_frontier_decision(
            context,
            root_frontiers=(),
        )

        self.assertIsNotNone(decision)
        self.assertLessEqual(len(decision.frontier_targets), 64)

    def test_frontier_fallback_keeps_remote_large_frontier_candidate(
        self,
    ) -> None:
        occupancy = np.full((140, 140), FREE, dtype=np.int8)
        confidence = np.ones((140, 140), dtype=np.float32)
        occupancy[95:115, 95:115] = UNKNOWN
        confidence[95:115, 95:115] = 0.0
        local_frontiers = tuple(
            (x, y)
            for y in range(12, 32)
            for x in range(12, 32)
        )
        remote_frontier = (94, 105)
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(20, 20),
            radius=10,
            frontiers=(*local_frontiers, remote_frontier),
        )
        policy = MctsExplorationPolicy(
            make_config(frontier_cluster_limit=3),
            seed=3,
        )

        decision = policy._fallback_frontier_decision(
            context,
            root_frontiers=(),
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.target, remote_frontier)
        self.assertIn(remote_frontier, decision.frontier_targets)
        self.assertLessEqual(len(decision.frontier_targets), 64)

    def test_registered_large_frontier_bypasses_fallback_candidate_cap(
        self,
    ) -> None:
        occupancy = np.full((140, 140), FREE, dtype=np.int8)
        confidence = np.ones((140, 140), dtype=np.float32)
        occupancy[95:115, 95:115] = UNKNOWN
        confidence[95:115, 95:115] = 0.0
        local_frontiers = tuple(
            (x, y)
            for y in range(12, 32)
            for x in range(12, 32)
        )
        remote_frontier = (94, 105)
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(20, 20),
            radius=10,
            frontiers=(*local_frontiers, remote_frontier),
        )
        policy = MctsExplorationPolicy(
            make_config(frontier_cluster_limit=3),
            seed=3,
        )
        policy.update_priority_frontier_registry(
            context,
            context.runtime_snapshot.frontiers,
        )
        policy._fallback_frontier_candidates = (
            lambda context, frontiers, root_frontiers: list(
                local_frontiers[:64]
            )
        )

        decision = policy._fallback_frontier_decision(
            context,
            root_frontiers=(),
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.target, remote_frontier)
        self.assertIn(remote_frontier, decision.frontier_targets)

    def test_root_frontier_hints_prefer_large_unexplored_over_local_pocket(
        self,
    ) -> None:
        occupancy = np.full((140, 140), FREE, dtype=np.int8)
        confidence = np.ones((140, 140), dtype=np.float32)
        occupancy[22, 22] = UNKNOWN
        confidence[22, 22] = 0.0
        occupancy[95:115, 95:115] = UNKNOWN
        confidence[95:115, 95:115] = 0.0
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(20, 20),
            radius=10,
        )
        policy = MctsExplorationPolicy(
            make_config(frontier_cluster_limit=1),
            seed=3,
        )

        frontiers = policy._frontier_cluster_centroids(context, (20, 20))

        self.assertEqual(len(frontiers), 1)
        self.assertGreaterEqual(frontiers[0][0], 94)
        self.assertGreaterEqual(frontiers[0][1], 94)

    def test_information_gain_dedupes_and_stops_at_known_walls(self) -> None:
        occupancy = np.full((24, 24), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((24, 24), dtype=np.float32)
        mark_free(occupancy, confidence, [(10, 10)])
        occupancy[8, 10] = OCCUPIED
        confidence[8, 10] = 1.0
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(10, 10),
        )
        policy = MctsExplorationPolicy(
            make_config(planning_rays=1),
            seed=1,
        )

        visible = policy._visible_unknown_cells(
            context,
            (10, 10),
            0,
            frozenset(),
        )
        repeated = policy._visible_unknown_cells(
            context,
            (10, 10),
            0,
            visible,
        )

        self.assertIn((10, 9), visible)
        self.assertNotIn((10, 8), visible)
        self.assertNotIn((10, 7), visible)
        self.assertEqual(repeated, frozenset())

    def test_information_gain_hierarchy_demotes_small_unknown_pockets(
        self,
    ) -> None:
        occupancy = np.full((40, 40), FREE, dtype=np.int8)
        confidence = np.ones((40, 40), dtype=np.float32)
        occupancy[5, 5] = UNKNOWN
        confidence[5, 5] = 0.0
        confidence[10, 10] = 0.0
        occupancy[20:25, 20:25] = UNKNOWN
        confidence[20:25, 20:25] = 0.0
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(15, 15),
            radius=5,
        )
        policy = MctsExplorationPolicy(make_config(), seed=1)
        grid = policy._build_grid(context)
        cache = policy._build_cache()

        small_unknown = policy._information_gain(
            context,
            grid,
            cache,
            frozenset({(5, 5)}),
        )
        low_confidence = policy._information_gain(
            context,
            grid,
            cache,
            frozenset({(10, 10)}),
        )
        large_unknown = policy._information_gain(
            context,
            grid,
            cache,
            frozenset({(22, 22)}),
        )

        self.assertGreater(large_unknown, low_confidence)
        self.assertGreater(low_confidence, small_unknown)
        self.assertGreater(small_unknown, 0.0)

    def test_mcts_records_iterations_expansion_and_backpropagation(self) -> None:
        occupancy, confidence = known_square()
        context = make_context(occupancy=occupancy, confidence=confidence)
        policy = MctsExplorationPolicy(
            make_config(iterations=20, horizon=3, branching_factor=5),
            seed=22,
        )

        policy.decide(context, lambda current, candidate: True)
        diagnostics = policy.last_search_diagnostics

        self.assertEqual(diagnostics.iterations, 20)
        self.assertEqual(diagnostics.slam_version, 3)
        self.assertGreater(diagnostics.generated_nodes, 1)
        self.assertEqual(
            sum(root.visits for root in diagnostics.root_visits),
            20,
        )

    def test_uct_discounted_rollout_and_backpropagation_helpers(self) -> None:
        policy = MctsExplorationPolicy(
            make_config(
                discount=0.5,
                horizon=2,
                uct_exploration=0.0,
            ),
            seed=5,
        )
        root_state = SearchState((0, 0), 0, 0, frozenset())
        root = MctsNode(
            root_state,
            parent=None,
            incoming_action=None,
            untried_actions=(),
        )
        low = MctsNode(
            root_state,
            parent=root,
            incoming_action=SearchAction(
                TRANSLATE,
                0,
                (0, 1),
                ((0, 1),),
                frozenset(),
                1.0,
                0,
            ),
            untried_actions=(),
        )
        high = MctsNode(
            root_state,
            parent=root,
            incoming_action=SearchAction(
                TRANSLATE,
                90,
                (1, 0),
                ((1, 0),),
                frozenset(),
                5.0,
                1,
            ),
            untried_actions=(),
        )
        root.children = [low, high]
        root.visits = 2
        low.visits = high.visits = 1
        low.accumulated_reward = 1.0
        high.accumulated_reward = 5.0

        self.assertIs(policy._select_uct_child(root), high)

        policy._backpropagate(high, 12.0)
        self.assertEqual(high.visits, 2)
        self.assertEqual(root.visits, 3)
        self.assertEqual(high.accumulated_reward, 17.0)

        action_one = SearchAction(
            TRANSLATE,
            0,
            (0, 1),
            ((0, 1),),
            frozenset(),
            10.0,
            0,
        )
        action_two = SearchAction(
            TRANSLATE,
            0,
            (0, 2),
            ((0, 2),),
            frozenset(),
            4.0,
            0,
        )

        def fake_generate(
            context,
            grid,
            cache,
            state,
            rng,
            *,
            root_frontiers,
            deadline=None,
        ):
            _ = deadline
            if state.depth == 0:
                return (action_one,)
            if state.depth == 1:
                return (action_two,)
            return ()

        policy._generate_actions = fake_generate
        context = make_context()
        reward = policy._rollout(
            context,
            policy._build_grid(context),
            policy._build_cache(),
            SearchState((0, 0), 0, 0, frozenset()),
            random.Random(1),
        )

        self.assertEqual(reward, 12.0)

    def test_rollout_stops_when_cooperative_deadline_has_elapsed(self) -> None:
        policy = MctsExplorationPolicy(make_config(), seed=1)
        context = make_context()

        reward = policy._rollout(
            context,
            policy._build_grid(context),
            policy._build_cache(),
            SearchState((20, 20), 0, 0, frozenset()),
            random.Random(1),
            deadline=0.0,
        )

        self.assertEqual(reward, 0.0)

    def test_gain_biased_rollout_prefers_informative_actions(self) -> None:
        policy = MctsExplorationPolicy(
            make_config(rollout_temperature=0.25),
            seed=9,
        )
        low = SearchAction(
            TRANSLATE,
            0,
            (0, 1),
            ((0, 1),),
            frozenset(),
            1.0,
            0,
        )
        high = SearchAction(
            TRANSLATE,
            90,
            (1, 0),
            ((1, 0),),
            frozenset(),
            8.0,
            1,
        )
        rng = random.Random(4)

        selections = [
            policy._choose_rollout_action((low, high), rng)
            for _ in range(200)
        ]

        self.assertGreater(selections.count(high), selections.count(low))

    def test_seeded_multi_drone_policy_smoke_exposes_diagnostics(self) -> None:
        occupancy, confidence = known_square(shape=(48, 48))
        context = make_context(
            occupancy=occupancy,
            confidence=confidence,
            position=(20, 20),
            version=11,
        )
        config = make_config(
            iterations=12,
            branching_factor=5,
            decision_time_budget_ms=0.0,
        )

        for drone_id in range(3):
            policy = MctsExplorationPolicy(
                config,
                seed=19 + drone_id * 9_973,
            )
            decision = policy.decide(context, lambda current, candidate: True)
            diagnostics = policy.last_search_diagnostics

            self.assertIn(
                decision.kind,
                {
                    ExplorationDecisionKind.STEP,
                    ExplorationDecisionKind.ROTATE,
                    ExplorationDecisionKind.EXHAUSTED,
                },
            )
            self.assertEqual(diagnostics.iterations, 12)
            self.assertEqual(diagnostics.slam_version, 11)


if __name__ == "__main__":
    unittest.main()
