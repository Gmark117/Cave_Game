import math
import threading
import unittest

import numpy as np

from mapping.frontier_registry import WorkUnitState
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from mission.energy import (
    EnergyRequirement,
    EnergyState,
    ReserveEnergyPolicy,
    UnlimitedEnergyPolicy,
)
from mission.exploration_coordination import (
    CoordinationReport,
    DirectiveKind,
    ExplorationPhase,
    ExplorationTask,
    FrontierTaskCoordinator,
    TaskState,
    TaskSuspension,
    WorkUnitOutcome,
)
def slam_with_line(*, version: int = 1) -> SlamSnapshot:
    occupancy = np.full((32, 32), UNKNOWN, dtype=np.int8)
    confidence = np.zeros((32, 32), dtype=np.float32)
    occupancy[8, 4:18] = FREE
    confidence[8, 4:18] = 1.0
    occupancy[8:17, 16] = FREE
    confidence[8:17, 16] = 1.0
    return SlamSnapshot(occupancy, confidence, version=version)


def known_free_slam(*, version: int = 1) -> SlamSnapshot:
    return SlamSnapshot(
        np.full((32, 32), FREE, dtype=np.int8),
        np.ones((32, 32), dtype=np.float32),
        version=version,
    )


def split_line_slam(*, version: int = 2, y: int = 8) -> SlamSnapshot:
    occupancy = np.full((32, 32), UNKNOWN, dtype=np.int8)
    confidence = np.zeros((32, 32), dtype=np.float32)
    occupancy[y, 4:9] = FREE
    occupancy[y, 13:18] = FREE
    occupancy[y, 9:13] = OCCUPIED
    confidence[y, 4:18] = 1.0
    occupancy[y:17, 16] = FREE
    confidence[y:17, 16] = 1.0
    return SlamSnapshot(occupancy, confidence, version=version)


class EnergyPolicyTests(unittest.TestCase):
    def test_unlimited_energy_accepts_and_never_forces_return(self) -> None:
        policy = UnlimitedEnergyPolicy()
        state = policy.state()
        requirement = EnergyRequirement(1000.0, 1000.0, 1000.0, 1000.0)
        self.assertTrue(policy.can_accept(state, requirement))
        self.assertFalse(policy.must_return(
            state,
            route_home_cost=1000.0,
            next_action_cost=1000.0,
            safety_reserve=1000.0,
        ))

    def test_finite_energy_preserves_home_and_reserve(self) -> None:
        policy = ReserveEnergyPolicy()
        state = EnergyState(20.0, 100.0)
        self.assertTrue(policy.can_accept(
            state,
            EnergyRequirement(4.0, 5.0, 6.0, 5.0),
        ))
        self.assertFalse(policy.can_accept(
            state,
            EnergyRequirement(5.0, 5.0, 6.0, 5.0),
        ))
        self.assertTrue(policy.must_return(
            state,
            route_home_cost=11.0,
            next_action_cost=5.0,
            safety_reserve=5.0,
        ))


class FrontierTaskCoordinatorTests(unittest.TestCase):
    def coordinator(self, *, drones: int = 3) -> FrontierTaskCoordinator:
        return FrontierTaskCoordinator(
            (32, 32),
            drones,
            (16, 16),
            sensor_range=4.0,
            sensor_fov_deg=60.0,
            minimum_component_cells=1,
            minimum_unknown_support_cells=1,
            frontier_stride=1,
            global_cell_size=8,
        )

    def _start_initial_round(self, coordinator, slam):
        for drone_id in range(coordinator.drone_count):
            coordinator.check_in(drone_id, slam)
        return tuple(
            coordinator.claim_directive(drone_id).directive
            for drone_id in range(coordinator.drone_count)
        )

    def _finish_round(self, coordinator, directives, slam, report_start=0):
        for offset, directive in enumerate(directives):
            coordinator.check_in(
                offset,
                slam,
                report=CoordinationReport(
                    report_id=report_start + offset,
                    directive_id=directive.directive_id,
                    kind=directive.kind,
                    completed_scan_headings=directive.scan_headings,
                ),
            )

    def test_nonblocking_snapshot_uses_last_publication_while_locked(self):
        coordinator = self.coordinator(drones=1)
        published = coordinator.snapshot()
        acquired = threading.Event()
        release = threading.Event()

        def hold_coordinator_lock():
            with coordinator._lock:
                acquired.set()
                release.wait(timeout=1.0)

        holder = threading.Thread(target=hold_coordinator_lock)
        holder.start()
        self.assertTrue(acquired.wait(timeout=1.0))
        try:
            self.assertIs(coordinator.snapshot(blocking=False), published)
        finally:
            release.set()
            holder.join(timeout=1.0)
        self.assertFalse(holder.is_alive())

    def test_initial_scan_distributes_exact_full_circle(self) -> None:
        coordinator = self.coordinator()
        directives = self._start_initial_round(coordinator, slam_with_line())

        self.assertTrue(all(
            item.kind == DirectiveKind.ROVER_SCAN for item in directives
        ))
        headings = sorted(
            heading for item in directives for heading in item.scan_headings
        )
        self.assertEqual(headings, [0, 60, 120, 180, 240, 300])
        self.assertTrue(all(len(item.scan_headings) == 2 for item in directives))

    def test_component_claim_bundles_all_sweep_anchors(self) -> None:
        coordinator = self.coordinator(drones=2)
        slam = slam_with_line()
        scans = self._start_initial_round(coordinator, slam)
        self._finish_round(coordinator, scans, slam)
        directives = tuple(
            coordinator.claim_directive(drone_id).directive
            for drone_id in range(2)
        )
        component_tasks = tuple(
            item for item in directives
            if item is not None and item.kind == DirectiveKind.COMPONENT_TASK
        )
        self.assertEqual(len(component_tasks), 1)
        task = component_tasks[0]
        self.assertGreater(len(task.claim.work_unit_ids), 1)
        self.assertEqual(
            task.claim.work_unit_ids,
            task.task.work_unit_ids,
        )
        snapshot = coordinator.snapshot()
        self.assertEqual(len(snapshot.claims), 1)
        followers = tuple(
            item for item in directives
            if item is not None and item.kind == DirectiveKind.COMPONENT_FOLLOW
        )
        self.assertEqual(len(followers), 1)
        self.assertEqual(followers[0].leader_drone_id, task.claim.owner_drone_id)
        self.assertEqual(task.reserved_branch_count, 1)

    def test_follower_reservation_blocks_child_reassignment_until_report(self) -> None:
        coordinator = self.coordinator(drones=2)
        initial = slam_with_line()
        scans = self._start_initial_round(coordinator, initial)
        self._finish_round(coordinator, scans, initial)
        directives = {
            drone_id: coordinator.claim_directive(drone_id).directive
            for drone_id in range(2)
        }
        leader_id, leader = next(
            item for item in directives.items()
            if item[1].kind == DirectiveKind.COMPONENT_TASK
        )
        follower_id, follower = next(
            item for item in directives.items()
            if item[1].kind == DirectiveKind.COMPONENT_FOLLOW
        )

        coordinator.check_in(
            leader_id,
            split_line_slam(),
            report=CoordinationReport(
                report_id=80,
                directive_id=leader.directive_id,
                kind=leader.kind,
                task_id=leader.task.task_id,
                component_id=leader.task.component_id,
                claim_token=leader.claim.token,
                work_unit_outcomes=tuple(
                    WorkUnitOutcome(unit_id, "sensor_gain")
                    for unit_id in leader.claim.work_unit_ids
                ),
            ),
        )

        self.assertIsNone(coordinator.claim_directive(leader_id).directive)
        self.assertFalse(coordinator.snapshot().mission_exhausted)

        coordinator.check_in(
            follower_id,
            known_free_slam(version=3),
            report=CoordinationReport(
                report_id=81,
                directive_id=follower.directive_id,
                kind=follower.kind,
                task_id=follower.task.task_id,
                component_id=follower.task.component_id,
            ),
        )

        self.assertTrue(coordinator.snapshot().mission_exhausted)
        self.assertEqual(
            coordinator.claim_directive(leader_id).directive.kind,
            DirectiveKind.HOME,
        )
        self.assertEqual(
            coordinator.claim_directive(follower_id).directive.kind,
            DirectiveKind.HOME,
        )

    def test_component_phase_does_not_start_endgame_probes(self) -> None:
        coordinator = self.coordinator(drones=1)
        slam = slam_with_line()
        scans = self._start_initial_round(coordinator, slam)
        self._finish_round(coordinator, scans, slam)
        task = coordinator.claim_directive(0).directive
        self.assertEqual(task.kind, DirectiveKind.COMPONENT_TASK)
        coordinator.check_in(
            0,
            known_free_slam(version=2),
            report=CoordinationReport(
                report_id=90,
                directive_id=task.directive_id,
                kind=task.kind,
                task_id=task.task.task_id,
                component_id=task.task.component_id,
                claim_token=task.claim.token,
                work_unit_outcomes=tuple(
                    WorkUnitOutcome(unit_id, "sensor_gain")
                    for unit_id in task.claim.work_unit_ids
                ),
            ),
        )

        directive = coordinator.claim_directive(0).directive
        self.assertEqual(directive.kind, DirectiveKind.HOME)
        self.assertEqual(coordinator.snapshot().phase, ExplorationPhase.COMPLETE)

    def test_no_frontier_starts_reachable_radial_probe(self) -> None:
        coordinator = self.coordinator(drones=2)
        slam = known_free_slam()
        scans = self._start_initial_round(coordinator, slam)
        self._finish_round(coordinator, scans, slam)
        directives = tuple(
            coordinator.claim_directive(drone_id).directive
            for drone_id in range(2)
        )
        probes = tuple(
            item for item in directives
            if item is not None and item.kind == DirectiveKind.RADIAL_PROBE
        )
        self.assertEqual(len(probes), 2)
        self.assertTrue(all(len(item.scan_headings) == 6 for item in probes))
        self.assertGreaterEqual(
            math.dist(probes[0].probe_target, probes[1].probe_target),
            coordinator.sensor_range * 0.5,
        )

    def test_radial_reports_complete_one_round_before_next_ring(self) -> None:
        coordinator = self.coordinator(drones=1)
        slam = known_free_slam()
        scans = self._start_initial_round(coordinator, slam)
        self._finish_round(coordinator, scans, slam)
        first = coordinator.claim_directive(0).directive
        self.assertEqual(first.kind, DirectiveKind.RADIAL_PROBE)

        coordinator.check_in(
            0,
            slam,
            report=CoordinationReport(
                report_id=30,
                directive_id=first.directive_id,
                kind=first.kind,
                round_id=first.round_id,
                completed_scan_headings=first.scan_headings,
            ),
        )
        second = coordinator.claim_directive(0).directive

        self.assertEqual(second.kind, DirectiveKind.RADIAL_PROBE)
        self.assertGreater(second.round_id, first.round_id)
        self.assertGreater(
            math.dist(coordinator.rover_position, second.probe_target),
            math.dist(coordinator.rover_position, first.probe_target),
        )

    def test_exhausted_bootstrap_probes_issue_home(self) -> None:
        coordinator = self.coordinator(drones=1)
        coordinator._probe_exhausted.add(0)
        slam = known_free_slam()
        scans = self._start_initial_round(coordinator, slam)
        self._finish_round(coordinator, scans, slam)

        directive = coordinator.claim_directive(0).directive

        self.assertEqual(directive.kind, DirectiveKind.HOME)
        self.assertTrue(coordinator.snapshot().mission_exhausted)

    def test_unreachable_component_is_not_mistaken_for_completion(self) -> None:
        coordinator = FrontierTaskCoordinator(
            (32, 32),
            1,
            (16, 16),
            sensor_range=4.0,
            sensor_fov_deg=60.0,
            minimum_component_cells=1,
            minimum_unknown_support_cells=1,
            frontier_stride=1,
            global_cell_size=8,
        )
        occupancy = np.full((32, 32), OCCUPIED, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        occupancy[6:11, 2:20] = UNKNOWN
        confidence[6:11, 2:20] = 0.0
        occupancy[8, 4:18] = FREE
        confidence[8, 4:18] = 1.0
        occupancy[16, 16] = FREE
        confidence[16, 16] = 1.0
        slam = SlamSnapshot(occupancy, confidence, version=1)
        scans = self._start_initial_round(coordinator, slam)
        self._finish_round(coordinator, scans, slam)

        directive = coordinator.claim_directive(0).directive

        self.assertIsNone(directive)
        self.assertFalse(coordinator.snapshot().mission_exhausted)
        self.assertEqual(
            coordinator.snapshot().phase,
            ExplorationPhase.COMPONENT_EXPLORATION,
        )

    def test_stale_claim_token_does_not_release_active_work(self) -> None:
        coordinator = self.coordinator(drones=1)
        slam = slam_with_line()
        scans = self._start_initial_round(coordinator, slam)
        self._finish_round(coordinator, scans, slam)
        directive = coordinator.claim_directive(0).directive

        rejected = coordinator.check_in(
            0,
            slam,
            report=CoordinationReport(
                report_id=10,
                directive_id=directive.directive_id,
                kind=DirectiveKind.COMPONENT_TASK,
                claim_token=directive.claim.token + 1,
            ),
        )

        self.assertFalse(rejected.report_accepted)
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.claims, (directive.claim,))
        self.assertEqual(
            next(item for item in snapshot.tasks
                 if item.task_id == directive.task.task_id).state,
            TaskState.ACTIVE,
        )

    def test_duplicate_report_keeps_already_published_directive_signalled(self) -> None:
        coordinator = self.coordinator(drones=1)
        slam = slam_with_line()
        scans = self._start_initial_round(coordinator, slam)
        directive = scans[0]
        report = CoordinationReport(
            report_id=12,
            directive_id=directive.directive_id,
            kind=directive.kind,
            round_id=directive.round_id,
            completed_scan_headings=directive.scan_headings,
        )
        first = coordinator.check_in(0, slam, report=report)
        second = coordinator.check_in(0, slam, report=report)

        self.assertTrue(first.report_accepted)
        self.assertTrue(second.report_accepted)
        self.assertTrue(second.directive_ready.is_set())
        self.assertEqual(
            coordinator.claim_directive(0).directive.kind,
            DirectiveKind.COMPONENT_TASK,
        )

    def test_suspended_claim_is_released_and_reassigned_with_new_token(self) -> None:
        coordinator = self.coordinator(drones=1)
        slam = slam_with_line()
        scans = self._start_initial_round(coordinator, slam)
        self._finish_round(coordinator, scans, slam)
        first = coordinator.claim_directive(0).directive
        suspension = TaskSuspension(
            task_id=first.task.task_id,
            claim_token=first.claim.token,
            drone_id=0,
            reason="energy_reserve",
            dfs_stack=(),
            remaining_work_unit_ids=first.claim.work_unit_ids,
            position_at_suspension=(16, 16),
            actual_return_path=((18, 16), (16, 16)),
            return_path_source="astar",
            local_slam_version=slam.version,
            energy_state=EnergyState(10.0, 100.0),
        )

        accepted = coordinator.check_in(
            0,
            slam,
            report=CoordinationReport(
                report_id=11,
                directive_id=first.directive_id,
                kind=DirectiveKind.COMPONENT_TASK,
                claim_token=first.claim.token,
                suspension=suspension,
            ),
        )
        second = coordinator.claim_directive(0).directive

        self.assertTrue(accepted.report_accepted)
        self.assertEqual(second.kind, DirectiveKind.COMPONENT_TASK)
        self.assertEqual(second.task.task_id, first.task.task_id)
        self.assertNotEqual(second.claim.token, first.claim.token)

    def test_split_children_release_after_component_claim_reports(self) -> None:
        coordinator = self.coordinator(drones=2)
        initial_slam = slam_with_line()
        scans = self._start_initial_round(coordinator, initial_slam)
        self._finish_round(coordinator, scans, initial_slam)
        directives = tuple(
            coordinator.claim_directive(drone_id).directive
            for drone_id in range(2)
        )
        first = next(
            directive for directive in directives
            if directive.kind == DirectiveKind.COMPONENT_TASK
        )
        split_slam = split_line_slam()

        coordinator.check_in(
            0,
            split_slam,
            report=CoordinationReport(
                report_id=20,
                directive_id=first.directive_id,
                kind=DirectiveKind.COMPONENT_TASK,
                claim_token=first.claim.token,
            ),
        )
        released_snapshot = coordinator.snapshot()
        child_ids = {
            child_id
            for component in released_snapshot.components
            if component.component_id == first.task.component_id
            for child_id in component.child_ids
        }
        child_units = [
            unit for unit in released_snapshot.work_units
            if unit.component_id in child_ids
        ]
        self.assertTrue(child_units)
        self.assertFalse(any(
            unit.state == WorkUnitState.BLOCKED for unit in child_units
        ))
        self.assertTrue(all(
            task.parent_task_id == first.task.task_id
            for task in released_snapshot.tasks
            if task.component_id in child_ids
        ))

    def test_assignment_prefers_deeper_continuation_before_route_cost(self) -> None:
        coordinator = self.coordinator(drones=1)
        coordinator._last_reported_task_by_drone[0] = 40
        shallow = ExplorationTask(
            task_id=41,
            component_id=1,
            component_revision=0,
            work_unit_ids=(1,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(17, 16),
            estimated_effort=1.0,
        )
        child = ExplorationTask(
            task_id=42,
            component_id=2,
            component_revision=0,
            work_unit_ids=(2,),
            parent_task_id=40,
            depth=1,
            preferred_entry=(25, 16),
            estimated_effort=1.0,
        )

        _quotes, selected = coordinator._assign_tasks(
            (0,),
            (shallow, child),
        )

        self.assertEqual(selected[0][1], child.task_id)

    def test_assignment_delegates_exact_routes_to_drone_workers(self) -> None:
        coordinator = self.coordinator(drones=3)
        tasks = tuple(
            ExplorationTask(
                task_id=task_id,
                component_id=task_id,
                component_revision=0,
                work_unit_ids=(task_id,),
                parent_task_id=None,
                depth=0,
                preferred_entry=(20 + task_id, 16),
                estimated_effort=1.0,
            )
            for task_id in range(2)
        )

        quotes, selected = coordinator._assign_tasks((0, 1, 2), tasks)

        self.assertEqual(len(quotes), len(tasks) * 3)
        self.assertEqual(len(selected), len(tasks))
        self.assertTrue(all(not quote.route for quote in quotes))
        self.assertTrue(all(not route for _drone, _task, route in selected))

    def test_assignment_uses_rover_connected_region_eligibility(self) -> None:
        coordinator = self.coordinator(drones=1)
        tasks = tuple(
            ExplorationTask(
                task_id=task_id,
                component_id=task_id,
                component_revision=0,
                work_unit_ids=(task_id,),
                parent_task_id=None,
                depth=0,
                preferred_entry=(17 + task_id, 16),
                estimated_effort=1.0,
            )
            for task_id in range(3)
        )

        quotes, selected = coordinator._assign_tasks(
            (0,),
            tasks,
            reachable_task_ids=frozenset({1}),
        )

        self.assertEqual(tuple(quote.task_id for quote in quotes), (1,))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][1], 1)

    def test_route_rejection_is_scoped_to_drone_and_component_revision(self) -> None:
        coordinator = self.coordinator(drones=2)
        task = ExplorationTask(
            task_id=7,
            component_id=4,
            component_revision=3,
            work_unit_ids=(9,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(20, 16),
            estimated_effort=1.0,
        )
        coordinator._route_rejections[(0, task.task_id)] = 3

        quotes, _selected = coordinator._assign_tasks((0, 1), (task,))

        self.assertEqual(
            tuple((quote.drone_id, quote.task_id) for quote in quotes),
            ((1, task.task_id),),
        )
        task.component_revision = 4
        refreshed, _selected = coordinator._assign_tasks((0,), (task,))
        self.assertEqual(len(refreshed), 1)


if __name__ == "__main__":
    unittest.main()
