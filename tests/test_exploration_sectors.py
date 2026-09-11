import unittest

import numpy as np

from mapping.exploration_sectors import (
    ExplorationSectorCoordinator,
    SectorOutcomeReport,
    SectorSuppressionOutcome,
    _TrackedFrontierComponent,
    _ScopedFrontierWork,
)
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot


def unknown_slam(shape: tuple[int, int], version: int = 0) -> SlamSnapshot:
    return SlamSnapshot(
        occupancy=np.full(shape, UNKNOWN, dtype=np.int8),
        confidence=np.zeros(shape, dtype=np.float32),
        version=version,
    )


class ExplorationSectorCoordinatorTests(unittest.TestCase):
    def _single_gateway_epoch(self):
        bootstrap = self._bootstrap()
        occupancy = np.full(self.shape, OCCUPIED, dtype=np.int8)
        confidence = np.ones(self.shape, dtype=np.float32)
        occupancy[40:50, 42:52] = UNKNOWN
        confidence[40:50, 42:52] = 0.0
        occupancy[41, 41] = FREE
        slam = SlamSnapshot(occupancy, confidence, version=10)
        for item in bootstrap:
            result = self.coordinator.check_in(item.owner_drone_id, item.sector_id, slam)
        return slam, result

    def test_zero_work_stands_by_and_only_active_return_opens_barrier(self) -> None:
        slam, published = self._single_gateway_epoch()
        assignments = tuple(self.coordinator.claim_assignment(i).assignment for i in range(3))
        self.assertEqual(len(published.published_assignments), 3)
        self.assertEqual([item.standby for item in assignments], [False, True, True])
        self.assertEqual(self.coordinator.snapshot().waiting_drone_ids, frozenset({1, 2}))
        for item in assignments[1:]:
            self.assertEqual(item.cells, frozenset())
            self.assertEqual(item.frontier_components, ())
            self.assertEqual(item.estimated_effort, 0.0)
            self.assertFalse(item.permits_exploration((41, 41)))
        active = assignments[0]
        self.assertTrue(active.permits_exploration((41, 41)))
        self.assertFalse(active.permits_exploration((120, 80)))
        self.assertTrue(active.contains((120, 80)))
        self.assertFalse(active.exploration_mask.flags.writeable)
        result = self.coordinator.check_in(0, active.sector_id, slam)
        self.assertTrue(result.mission_exhausted)
        self.assertEqual(self.coordinator.generation, 2)
        self.assertTrue(all(self.coordinator.claim_assignment(i).mission_exhausted for i in range(3)))

    def test_standby_can_reactivate_without_checkin_and_late_claim_gets_latest_epoch(self) -> None:
        for consume_standby_first in (False, True):
            with self.subTest(consume_standby_first=consume_standby_first):
                self.setUp()
                slam, _published = self._single_gateway_epoch()
                active = self.coordinator.claim_assignment(0).assignment
                ready = self.coordinator._assignment_ready[1]
                if consume_standby_first:
                    standby = self.coordinator.claim_assignment(1)
                    self.assertTrue(standby.assignment.standby)
                    self.assertFalse(ready.is_set())
                    self.assertIsNone(self.coordinator.claim_assignment(1).assignment)
                occupancy, confidence = slam.occupancy.copy(), slam.confidence.copy()
                occupancy[49, 51] = OCCUPIED  # Local gain retains the first component.
                confidence[49, 51] = 1.0
                occupancy[66:76, 96:106] = UNKNOWN
                confidence[66:76, 96:106] = 0.0
                occupancy[65, 95] = FREE
                changed = SlamSnapshot(occupancy, confidence, version=11)
                result = self.coordinator.check_in(0, active.sector_id, changed)
                self.assertFalse(result.mission_exhausted)
                self.assertTrue(ready.is_set())
                reactivated = self.coordinator.claim_assignment(1)
                self.assertEqual(reactivated.assignment.generation, 2)
                self.assertFalse(reactivated.assignment.standby)
                self.assertFalse(ready.is_set())
                self.assertEqual(self.coordinator.claim_assignment(2).assignment.generation, 2)
                self.assertEqual(self.coordinator.generation, 2)
                self.coordinator.stop()
                self.assertTrue(all(event.is_set() for event in self.coordinator._assignment_ready.values()))

    def test_scope_effort_deduplicates_and_normalizes_unknown_area(self) -> None:
        shape = (32, 64)
        coordinator = ExplorationSectorCoordinator(shape, 2, (0, 0), cell_size=32)
        components = tuple(_TrackedFrontierComponent(i, frozenset({point}), 0)
                           for i, point in enumerate(((15, 15), (17, 15))))
        unknown = np.zeros(shape, dtype=bool)
        unknown[15:25, 20:40] = True
        scope = np.ones(shape, dtype=bool)
        work = _ScopedFrontierWork(
            components, (scope, scope), unknown, 32,
            np.ones((1, 2)), np.array([[0.0, 1.0]]), coordinator._aggregate_mask,
        )
        full = work.estimate({(0, 0), (1, 0)})
        clipped = work.estimate({(0, 0)})
        self.assertEqual(full.unknown_area, 200 / (32 * 32))
        self.assertEqual(clipped.unknown_area, 120 / (32 * 32))
        self.assertEqual(full.setup, 2.0)
        self.assertGreater(full.total, clipped.total)
        self.assertEqual(work.estimate({(1, 0)}).total, 0.0)

    def test_coarse_approach_accounts_for_terrain_and_return_trip(self) -> None:
        costs = np.ones((3, 4))
        plain = self.coordinator._coarse_approach_distances(costs)
        costs[1, 1] = 5.0
        costly = self.coordinator._coarse_approach_distances(costs)
        self.assertGreater(costly[1, 1], plain[1, 1])
        component = _TrackedFrontierComponent(1, frozenset({(32, 32)}), 0)
        work = _ScopedFrontierWork(
            (component,), (np.ones(self.shape, dtype=bool),),
            np.zeros(self.shape, dtype=bool), 32, costs, costly,
            self.coordinator._aggregate_mask,
        )
        self.assertEqual(work.estimate({(1, 1)}).approach, 2 * costly[1, 1])

    def test_rebalance_cannot_reduce_effort_by_dropping_basin_coverage(self) -> None:
        shape = (32, 96)
        coordinator = ExplorationSectorCoordinator(shape, 2, (0, 0), cell_size=32)
        components = (
            _TrackedFrontierComponent(0, frozenset({(15, 15)}), 0),
            _TrackedFrontierComponent(1, frozenset({(80, 15)}), 0),
        )
        unknown = np.zeros(shape, dtype=bool)
        unknown[:, 32:64] = True
        first = np.zeros(shape, dtype=bool)
        first[:, :64] = True
        second = np.zeros(shape, dtype=bool)
        second[:, 64:] = True
        work = _ScopedFrontierWork(
            components, (first, second), unknown, 32,
            np.ones((1, 3)), np.zeros((1, 3)), coordinator._aggregate_mask,
        )
        owners = np.array([[0, 0, 1]])
        counts = np.array([[1, 0, 1]])
        balanced, diagnostics = coordinator._rebalance_frontier_work(
            owners, counts, ((0, 0), (2, 0)),
            estimated_effort=np.array([[1.0, 32.0, 1.0]]),
            estimate_assignment=work.estimate,
            preserves_transfer=work.preserves_transfer,
        )
        np.testing.assert_array_equal(balanced, owners)
        self.assertEqual(diagnostics.moved_coarse_cell_count, 0)
        # Another owner of the same component may receive the basin cell.
        shared = _ScopedFrontierWork(
            (_TrackedFrontierComponent(0, frozenset({(15, 15), (80, 15)}), 0),),
            (np.ones(shape, dtype=bool),), unknown, 32,
            np.ones((1, 3)), np.zeros((1, 3)), coordinator._aggregate_mask,
        )
        self.assertTrue(shared.preserves_transfer({(0, 0), (1, 0)}, {(2, 0)}, (1, 0)))

    def test_migrated_gateway_gets_new_identity_despite_old_component_suppression(self) -> None:
        slam, _published = self._single_gateway_epoch()
        active = self.coordinator.claim_assignment(0).assignment
        old_id = active.frontier_components[0].component_id
        occupancy, confidence = slam.occupancy.copy(), slam.confidence.copy()
        occupancy[40:43, 42] = OCCUPIED
        confidence[40:43, 42] = 1.0
        occupancy[50, 51] = FREE
        moved = SlamSnapshot(occupancy, confidence, version=11)
        result = self.coordinator.check_in(0, active.sector_id, moved, SectorOutcomeReport(
            active.sector_id, active.generation,
            (SectorSuppressionOutcome(old_id, ("zero_gain_directed_scan",), 1),),
        ))
        self.assertFalse(result.mission_exhausted)
        next_assignment = self.coordinator.claim_assignment(0).assignment
        self.assertNotEqual(next_assignment.frontier_components[0].component_id, old_id)
        self.assertTrue(next_assignment.permits_exploration((51, 50)))

    def setUp(self) -> None:
        self.shape = (96, 128)
        self.coordinator = ExplorationSectorCoordinator(
            self.shape,
            3,
            (64, 48),
            cell_size=32,
            confidence_threshold=0.6,
        )

    def _bootstrap(self):
        slam = unknown_slam(self.shape)
        first = self.coordinator.check_in(0, None, slam)
        second = self.coordinator.check_in(1, None, slam)
        third = self.coordinator.check_in(2, None, slam)
        results = (
            self.coordinator.claim_assignment(0),
            self.coordinator.claim_assignment(1),
            self.coordinator.claim_assignment(2),
        )
        self.assertTrue(first.waiting_for_team)
        self.assertTrue(second.waiting_for_team)
        self.assertTrue(third.waiting_for_team)
        self.assertTrue(all(result.arrived for result in results))
        return tuple(result.assignment for result in results)

    def test_bootstrap_waits_for_team_and_builds_disjoint_coverage(self) -> None:
        assignments = self._bootstrap()

        self.assertTrue(all(assignment is not None for assignment in assignments))
        self.assertTrue(all(assignment.bootstrap for assignment in assignments))
        cell_sets = [assignment.cells for assignment in assignments]
        self.assertFalse(cell_sets[0] & cell_sets[1])
        self.assertFalse(cell_sets[0] & cell_sets[2])
        self.assertFalse(cell_sets[1] & cell_sets[2])
        self.assertEqual(
            set().union(*cell_sets),
            {(x, y) for y in range(3) for x in range(4)},
        )

        snapshot = self.coordinator.snapshot()
        self.assertEqual(snapshot.generation, 0)
        self.assertEqual(snapshot.assignments, assignments)
        self.assertEqual(snapshot.waiting_drone_ids, frozenset())

    def test_snapshot_keeps_epoch_visible_while_drone_waits(self) -> None:
        assignments = self._bootstrap()
        slam = unknown_slam(self.shape, version=2)

        self.coordinator.check_in(0, assignments[0].sector_id, slam)
        snapshot = self.coordinator.snapshot()

        self.assertEqual(snapshot.assignments, assignments)
        self.assertEqual(snapshot.waiting_drone_ids, frozenset({0}))

    def test_next_epoch_uses_rover_frontiers_and_remains_contiguous(self) -> None:
        bootstrap = self._bootstrap()
        occupancy = np.full(self.shape, UNKNOWN, dtype=np.int8)
        confidence = np.zeros(self.shape, dtype=np.float32)
        occupancy[16:80, 16:112] = FREE
        confidence[16:80, 16:112] = 1.0
        rover_slam = SlamSnapshot(occupancy, confidence, version=41)

        self.coordinator.check_in(0, bootstrap[0].sector_id, rover_slam)
        self.coordinator.check_in(1, bootstrap[1].sector_id, rover_slam)
        last = self.coordinator.check_in(
            2,
            bootstrap[2].sector_id,
            rover_slam,
        )
        self.assertTrue(last.waiting_for_team)
        workload = last.workload_diagnostics
        self.assertIsNotNone(workload)
        assert workload is not None
        self.assertEqual(len(workload.initial_estimated_efforts), 3)
        self.assertEqual(len(workload.balanced_estimated_efforts), 3)
        assignments = (
            self.coordinator.claim_assignment(0).assignment,
            self.coordinator.claim_assignment(1).assignment,
            self.coordinator.claim_assignment(2).assignment,
        )

        self.assertEqual({item.generation for item in assignments}, {1})
        self.assertTrue(all(not item.bootstrap for item in assignments))
        self.assertGreater(sum(item.frontier_cells for item in assignments), 0)
        self.assertGreater(sum(item.estimated_effort for item in assignments), 0)
        self.assertTrue(any(
            item.frontier_components
            for item in assignments
        ))
        self.assertEqual(
            sum(len(item.cells) for item in assignments),
            12,
        )
        self.assertEqual(
            len(set().union(*(item.cells for item in assignments))),
            12,
        )
        for assignment in assignments:
            self.assertTrue(self._is_connected(assignment.cells))

    def test_no_rover_frontier_ends_the_sector_mission(self) -> None:
        bootstrap = self._bootstrap()
        known = SlamSnapshot(
            occupancy=np.full(self.shape, FREE, dtype=np.int8),
            confidence=np.ones(self.shape, dtype=np.float32),
            version=50,
        )

        self.coordinator.check_in(0, bootstrap[0].sector_id, known)
        self.coordinator.check_in(1, bootstrap[1].sector_id, known)
        result = self.coordinator.check_in(2, bootstrap[2].sector_id, known)

        self.assertTrue(result.mission_exhausted)
        self.assertIsNone(result.assignment)
        self.assertTrue(
            self.coordinator.check_in(0, bootstrap[0].sector_id, known)
            .mission_exhausted
        )

    def test_rebalances_weighted_boundary_cells_without_disconnect(self) -> None:
        owners = np.asarray([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 2, 2],
        ], dtype=np.int16)
        frontier_counts = np.zeros_like(owners, dtype=np.int32)
        frontier_counts[:, 1] = 10

        balanced, diagnostics = (
            self.coordinator._rebalance_frontier_work(
                owners,
                frontier_counts,
                ((0, 1), (3, 0), (3, 2)),
            )
        )

        self.assertEqual(
            diagnostics.initial_frontier_workloads,
            (30, 0, 0),
        )
        self.assertEqual(
            diagnostics.balanced_frontier_workloads,
            (10, 10, 10),
        )
        self.assertEqual(diagnostics.moved_coarse_cell_count, 2)
        for owner in range(3):
            cells = {
                (int(x), int(y))
                for y, x in zip(*np.where(balanced == owner))
            }
            self.assertTrue(self._is_connected(cells))

    def test_tiny_residual_frontier_does_not_start_an_epoch(self) -> None:
        bootstrap = self._bootstrap()
        occupancy = np.full(self.shape, FREE, dtype=np.int8)
        confidence = np.ones(self.shape, dtype=np.float32)
        occupancy[48, 64] = UNKNOWN
        confidence[48, 64] = 0.0
        rover_slam = SlamSnapshot(occupancy, confidence, version=51)

        self.coordinator.check_in(
            0, bootstrap[0].sector_id, rover_slam
        )
        self.coordinator.check_in(
            1, bootstrap[1].sector_id, rover_slam
        )
        result = self.coordinator.check_in(
            2, bootstrap[2].sector_id, rover_slam
        )

        self.assertTrue(result.mission_exhausted)
        self.assertIsNotNone(result.frontier_diagnostics)
        diagnostics = result.frontier_diagnostics
        assert diagnostics is not None
        self.assertEqual(diagnostics.raw_frontier_pixels, 8)
        self.assertEqual(diagnostics.significant_frontier_pixels, 0)
        self.assertEqual(diagnostics.discarded_component_count, 1)

    def test_unchanged_zero_gain_frontiers_are_not_reassigned(self) -> None:
        bootstrap = self._bootstrap()
        occupancy = np.full(self.shape, UNKNOWN, dtype=np.int8)
        confidence = np.zeros(self.shape, dtype=np.float32)
        occupancy[16:80, 16:112] = FREE
        confidence[16:80, 16:112] = 1.0
        rover_slam = SlamSnapshot(occupancy, confidence, version=41)

        for drone_id, assignment in enumerate(bootstrap):
            result = self.coordinator.check_in(
                drone_id,
                assignment.sector_id,
                rover_slam,
            )
        self.assertFalse(result.mission_exhausted)
        generation_one = tuple(
            self.coordinator.claim_assignment(drone_id).assignment
            for drone_id in range(3)
        )
        self.assertTrue(all(generation_one))

        for drone_id, assignment in enumerate(generation_one):
            assert assignment is not None
            result = self.coordinator.check_in(
                drone_id,
                assignment.sector_id,
                rover_slam,
            )

        self.assertTrue(result.mission_exhausted)
        self.assertIsNotNone(result.outcome_diagnostics)
        outcome = result.outcome_diagnostics
        assert outcome is not None
        self.assertEqual(outcome.confident_occupied_gain, 0)
        self.assertGreater(outcome.suppressed_component_count, 0)
        self.assertEqual(outcome.remaining_component_count, 0)

    def test_far_away_global_gain_does_not_save_unchanged_component(
        self,
    ) -> None:
        coordinator = ExplorationSectorCoordinator(
            (256, 256),
            3,
            (128, 128),
            cell_size=32,
            confidence_threshold=0.6,
        )
        initial = unknown_slam((256, 256))
        for drone_id in range(3):
            coordinator.check_in(drone_id, None, initial)
        bootstrap = tuple(
            coordinator.claim_assignment(drone_id).assignment
            for drone_id in range(3)
        )

        occupancy = np.full((256, 256), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((256, 256), dtype=np.float32)
        occupancy[96:160, 96:160] = FREE
        confidence[96:160, 96:160] = 1.0
        base = SlamSnapshot(occupancy, confidence, version=10)
        for drone_id, assignment in enumerate(bootstrap):
            assert assignment is not None
            coordinator.check_in(drone_id, assignment.sector_id, base)
        generation_one = tuple(
            coordinator.claim_assignment(drone_id).assignment
            for drone_id in range(3)
        )

        changed_occupancy = occupancy.copy()
        changed_confidence = confidence.copy()
        changed_occupancy[0:8, 0:8] = OCCUPIED
        changed_confidence[0:8, 0:8] = 1.0
        far_gain = SlamSnapshot(
            changed_occupancy,
            changed_confidence,
            version=11,
        )
        for drone_id, assignment in enumerate(generation_one):
            assert assignment is not None
            result = coordinator.check_in(
                drone_id,
                assignment.sector_id,
                far_gain,
            )

        self.assertTrue(result.mission_exhausted)
        outcome = result.outcome_diagnostics
        assert outcome is not None
        self.assertEqual(outcome.confident_occupied_gain, 64)
        self.assertEqual(outcome.locally_unchanged_component_count, 1)
        self.assertEqual(outcome.productive_component_count, 0)
        self.assertEqual(outcome.remaining_component_count, 0)

    def test_local_component_gain_keeps_unchanged_geometry(self) -> None:
        coordinator = ExplorationSectorCoordinator(
            (256, 256),
            3,
            (128, 128),
            cell_size=32,
            confidence_threshold=0.6,
        )
        initial = unknown_slam((256, 256))
        for drone_id in range(3):
            coordinator.check_in(drone_id, None, initial)
        bootstrap = tuple(
            coordinator.claim_assignment(drone_id).assignment
            for drone_id in range(3)
        )
        occupancy = np.full((256, 256), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((256, 256), dtype=np.float32)
        occupancy[96:160, 96:160] = FREE
        confidence[96:160, 96:160] = 1.0
        base = SlamSnapshot(occupancy, confidence, version=10)
        for drone_id, assignment in enumerate(bootstrap):
            assert assignment is not None
            coordinator.check_in(drone_id, assignment.sector_id, base)
        generation_one = tuple(
            coordinator.claim_assignment(drone_id).assignment
            for drone_id in range(3)
        )

        changed_occupancy = occupancy.copy()
        changed_confidence = confidence.copy()
        changed_occupancy[128, 128] = OCCUPIED
        changed_confidence[128, 128] = 1.0
        local_gain = SlamSnapshot(
            changed_occupancy,
            changed_confidence,
            version=11,
        )
        for drone_id, assignment in enumerate(generation_one):
            assert assignment is not None
            result = coordinator.check_in(
                drone_id,
                assignment.sector_id,
                local_gain,
            )

        self.assertFalse(result.mission_exhausted)
        outcome = result.outcome_diagnostics
        assert outcome is not None
        self.assertEqual(outcome.productive_component_count, 1)
        self.assertEqual(outcome.locally_unchanged_component_count, 0)
        self.assertEqual(outcome.remaining_component_count, 1)
        generation_two = tuple(
            coordinator.claim_assignment(drone_id).assignment
            for drone_id in range(3)
        )
        first_ids = {
            component.component_id
            for assignment in generation_one
            if assignment is not None
            for component in assignment.frontier_components
        }
        second_ids = {
            component.component_id
            for assignment in generation_two
            if assignment is not None
            for component in assignment.frontier_components
        }
        self.assertEqual(second_ids, first_ids)

    def test_reported_zero_gain_retires_component_despite_local_gain(
        self,
    ) -> None:
        coordinator = ExplorationSectorCoordinator(
            (256, 256),
            3,
            (128, 128),
            cell_size=32,
            confidence_threshold=0.6,
        )
        initial = unknown_slam((256, 256))
        for drone_id in range(3):
            coordinator.check_in(drone_id, None, initial)
        bootstrap = tuple(
            coordinator.claim_assignment(drone_id).assignment
            for drone_id in range(3)
        )
        occupancy = np.full((256, 256), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((256, 256), dtype=np.float32)
        occupancy[96:160, 96:160] = FREE
        confidence[96:160, 96:160] = 1.0
        base = SlamSnapshot(occupancy, confidence, version=10)
        for drone_id, assignment in enumerate(bootstrap):
            assert assignment is not None
            coordinator.check_in(drone_id, assignment.sector_id, base)
        generation_one = tuple(
            coordinator.claim_assignment(drone_id).assignment
            for drone_id in range(3)
        )
        reporter = next(
            assignment
            for assignment in generation_one
            if assignment is not None and assignment.frontier_components
        )
        component_id = reporter.frontier_components[0].component_id

        changed_occupancy = occupancy.copy()
        changed_confidence = confidence.copy()
        changed_occupancy[128, 128] = OCCUPIED
        changed_confidence[128, 128] = 1.0
        local_gain = SlamSnapshot(
            changed_occupancy,
            changed_confidence,
            version=11,
        )
        report = SectorOutcomeReport(
            sector_id=reporter.sector_id,
            generation=reporter.generation,
            suppressions=(SectorSuppressionOutcome(
                component_id=component_id,
                reasons=("zero_gain_directed_scan",),
                sampled_target_count=4,
            ),),
        )
        for drone_id, assignment in enumerate(generation_one):
            assert assignment is not None
            result = coordinator.check_in(
                drone_id,
                assignment.sector_id,
                local_gain,
                report if assignment is reporter else None,
            )

        self.assertTrue(result.mission_exhausted)
        outcome = result.outcome_diagnostics
        assert outcome is not None
        self.assertEqual(outcome.reported_zero_gain_component_count, 1)
        self.assertEqual(outcome.remaining_component_count, 0)
        self.assertIn(
            "zero_gain_directed_scan",
            outcome.components[0].suppression_reasons,
        )

    def test_estimated_effort_includes_component_approach_distance(
        self,
    ) -> None:
        near = _TrackedFrontierComponent(
            component_id=1,
            cells=frozenset({(64, 48)}),
            baseline_confident_occupied_cells=0,
        )
        far = _TrackedFrontierComponent(
            component_id=2,
            cells=frozenset({(0, 0)}),
            baseline_confident_occupied_cells=0,
        )
        mask = np.zeros(self.shape, dtype=bool)
        mask[48, 64] = True
        mask[0, 0] = True
        frontier_counts = self.coordinator._aggregate_mask(mask)
        traversal = np.ones_like(frontier_counts, dtype=np.float64)

        _effort, component_efforts = (
            self.coordinator._estimated_frontier_effort(
                (near, far),
                frontier_counts,
                traversal,
            )
        )

        self.assertGreater(component_efforts[2], component_efforts[1])

    def test_component_effort_normalizes_unknown_area_to_coarse_cells(
        self,
    ) -> None:
        component = _TrackedFrontierComponent(
            component_id=1,
            cells=frozenset({(15, 15)}),
            baseline_confident_occupied_cells=0,
        )
        frontier_counts = np.zeros((1, 1), dtype=np.int32)
        frontier_counts[0, 0] = 1

        _effort, component_efforts = (
            self.coordinator._estimated_frontier_effort(
                (component,),
                frontier_counts,
                np.ones((1, 1), dtype=np.float64),
                approach_distances=np.zeros((1, 1), dtype=np.float64),
                unknown_support=(32 * 32,),
            )
        )

        # One full 32x32 unknown tile contributes one relative work unit.
        self.assertEqual(component_efforts[1], 3.0)

    @staticmethod
    def _is_connected(cells) -> bool:
        remaining = set(cells)
        if not remaining:
            return True
        stack = [remaining.pop()]
        while stack:
            x, y = stack.pop()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        return not remaining


if __name__ == "__main__":
    unittest.main()
