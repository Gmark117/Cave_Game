import unittest

import numpy as np

from mapping.exploration_sectors import (
    ExplorationSectorCoordinator,
    SectorOutcomeReport,
    SectorSuppressionOutcome,
    _TrackedFrontierComponent,
)
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot


def unknown_slam(shape: tuple[int, int], version: int = 0) -> SlamSnapshot:
    return SlamSnapshot(
        occupancy=np.full(shape, UNKNOWN, dtype=np.int8),
        confidence=np.zeros(shape, dtype=np.float32),
        version=version,
    )


class ExplorationSectorCoordinatorTests(unittest.TestCase):
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
