import math
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from mapping.frontier_registry import (
    ComponentWorkUnit,
    ComponentState,
    ExplorationMode,
    FrontierComponentRecord,
    FrontierGeometry,
    WorkUnitKind,
    WorkUnitState,
)
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from mission.energy import (
    EnergyRequirement,
    EnergyState,
    ReserveEnergyPolicy,
    UnlimitedEnergyPolicy,
)
from mission.exploration_coordination import (
    BatchMemberReport,
    CoordinationReport,
    DirectiveKind,
    ExplorationDirective,
    ExplorationPhase,
    ExplorationTask,
    FrontierTaskCoordinator,
    TaskState,
    TaskSuspension,
    WorkUnitOutcome,
)
from navigation.highway import build_highway_graph


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
    def coordinator(
        self,
        *,
        drones: int = 3,
        **kwargs,
    ) -> FrontierTaskCoordinator:
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
            **kwargs,
        )

    @staticmethod
    def _install_focused_tasks(
        coordinator: FrontierTaskCoordinator,
        entries: tuple[tuple[int, int], ...],
    ) -> None:
        coordinator.registry.revision = 1
        for task_id, entry in enumerate(entries):
            geometry = FrontierGeometry(
                cells=frozenset({entry}),
                bounding_box=(*entry, *entry),
                centroid=(float(entry[0]), float(entry[1])),
                wall_contact_cells=0,
                wall_contact_ratio=0.0,
                longest_wall_run_ratio=0.0,
                unknown_support_cells=1,
                slam_version=1,
            )
            coordinator.registry.components[task_id] = FrontierComponentRecord(
                component_id=task_id,
                state=ComponentState.ACTIVE,
                geometry=geometry,
                geometry_revision=0,
                parent_ids=(),
                child_ids=(),
                first_seen_revision=1,
                last_seen_revision=1,
                missing_reconciliations=0,
                dormant_reason=None,
                exploration_mode=ExplorationMode.FOCUSED,
                work_unit_ids=(task_id,),
            )
            coordinator.registry.work_units[task_id] = ComponentWorkUnit(
                work_unit_id=task_id,
                component_id=task_id,
                component_revision=0,
                kind=WorkUnitKind.FOCUSED_ANCHOR,
                cells=frozenset({entry}),
                anchor_position=entry,
                scan_headings=(90,),
                estimated_effort=1.0,
                state=WorkUnitState.READY,
            )
            coordinator._tasks[task_id] = ExplorationTask(
                task_id=task_id,
                component_id=task_id,
                component_revision=0,
                work_unit_ids=(task_id,),
                parent_task_id=None,
                depth=0,
                preferred_entry=entry,
                estimated_effort=1.0,
            )
            coordinator._task_by_unit[task_id] = task_id
            coordinator._component_task_ids[task_id] = [task_id]
        coordinator.registry._next_component_id = len(entries)
        coordinator.registry._next_work_unit_id = len(entries)
        coordinator._next_task_id = len(entries)
        coordinator._initial_scan_complete = True
        coordinator._phase = ExplorationPhase.COMPONENT_EXPLORATION
        coordinator._bootstrap_followers_issued = True
        coordinator._waiting.update(range(coordinator.drone_count))

    @staticmethod
    def _complete_batch_report(directive, *, report_id: int):
        return CoordinationReport(
            report_id=report_id,
            directive_id=directive.directive_id,
            kind=DirectiveKind.COMPONENT_BATCH,
            lease_id=directive.spatial_lease.lease_id,
            batch_member_reports=tuple(
                BatchMemberReport(
                    task_id=member.task.task_id,
                    component_id=member.task.component_id,
                    component_revision=member.task.component_revision,
                    claim_token=member.claim.token,
                    disposition="complete",
                    work_unit_outcomes=tuple(
                        WorkUnitOutcome(unit_id, "sensor_gain")
                        for unit_id in member.claim.work_unit_ids
                    ),
                )
                for member in directive.batch_members
            ),
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

    def test_first_component_dispatch_consumes_bootstrap_follow_window(self) -> None:
        coordinator = self.coordinator(drones=1)
        slam = slam_with_line()
        scans = self._start_initial_round(coordinator, slam)
        self._finish_round(coordinator, scans, slam)

        first = coordinator.claim_directive(0).directive
        self.assertIsNotNone(first)
        self.assertEqual(first.kind, DirectiveKind.COMPONENT_TASK)
        self.assertTrue(coordinator._bootstrap_followers_issued)

    def test_lost_waiting_contact_blocks_quiescence_until_recheck_in(self) -> None:
        coordinator = self.coordinator(drones=2)
        slam = known_free_slam()
        coordinator._initial_scan_complete = True
        coordinator._phase = ExplorationPhase.COMPONENT_EXPLORATION
        coordinator._waiting.update((0, 1))
        coordinator.waiting_contact_lost(0)

        self.assertFalse(coordinator._team_is_quiescent())
        self.assertFalse(coordinator.snapshot().mission_exhausted)
        coordinator.check_in(0, slam)
        self.assertTrue(coordinator.snapshot().mission_exhausted)

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

    def test_focused_endgame_prefers_round_trip_before_continuation(self) -> None:
        coordinator = self.coordinator(drones=1)
        coordinator._last_reported_task_by_drone[0] = 40
        near = ExplorationTask(
            task_id=41,
            component_id=1,
            component_revision=0,
            work_unit_ids=(1,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(17, 16),
            estimated_effort=1.0,
        )
        far_continuation = ExplorationTask(
            task_id=42,
            component_id=2,
            component_revision=0,
            work_unit_ids=(2,),
            parent_task_id=40,
            depth=3,
            preferred_entry=(30, 16),
            estimated_effort=1.0,
        )

        _quotes, selected = coordinator._assign_tasks(
            (0,),
            (near, far_continuation),
            focused_endgame=True,
        )

        self.assertEqual(selected[0][1], near.task_id)

    def test_focused_endgame_requires_every_actionable_component(self) -> None:
        coordinator = self.coordinator(drones=1)
        coordinator.registry.work_units = {
            1: SimpleNamespace(state=WorkUnitState.READY),
            2: SimpleNamespace(state=WorkUnitState.READY),
        }
        focused = SimpleNamespace(
            state=ComponentState.ACTIVE,
            work_unit_ids=(1,),
            exploration_mode=ExplorationMode.FOCUSED,
        )
        sweep = SimpleNamespace(
            state=ComponentState.ACTIVE,
            work_unit_ids=(2,),
            exploration_mode=ExplorationMode.SWEEP,
        )
        coordinator.registry.components = {1: focused, 2: sweep}

        self.assertFalse(coordinator._focused_endgame_is_active())
        sweep.exploration_mode = ExplorationMode.FOCUSED
        self.assertTrue(coordinator._focused_endgame_is_active())
        coordinator.registry.work_units[1].state = WorkUnitState.VISITED
        coordinator.registry.work_units[2].state = WorkUnitState.VISITED
        self.assertFalse(coordinator._focused_endgame_is_active())

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

    def test_active_focused_frontier_batch_seeds_every_idle_drone_before_attachment(
        self,
    ) -> None:
        coordinator = self.coordinator(
            drones=2,
            focused_frontier_batch_mode="active",
            focused_frontier_batch_maximum_detour_sensor_ranges=100.0,
            focused_frontier_batch_minimum_avoided_round_trip_sensor_ranges=0.0,
        )
        self._install_focused_tasks(
            coordinator,
            ((20, 16), (24, 16), (28, 16)),
        )

        coordinator._schedule(known_free_slam())
        directives = tuple(
            coordinator.claim_directive(drone_id).directive
            for drone_id in range(2)
        )

        self.assertTrue(all(item is not None for item in directives))
        batch = next(
            item for item in directives
            if item.kind == DirectiveKind.COMPONENT_BATCH
        )
        single = next(
            item for item in directives
            if item.kind == DirectiveKind.COMPONENT_TASK
        )
        self.assertEqual(len(batch.batch_members), 2)
        self.assertIsNone(batch.task)
        self.assertIsNone(batch.claim)
        self.assertEqual(len(coordinator.snapshot().claims), 3)
        self.assertEqual(
            len({
                member.claim.token for member in batch.batch_members
            } | {single.claim.token}),
            3,
        )
        self.assertEqual(
            set(batch.spatial_lease.member_task_ids),
            {member.task.task_id for member in batch.batch_members},
        )

    def test_observe_focused_frontier_batch_is_assignment_and_claim_pure(self) -> None:
        off = self.coordinator(drones=2, focused_frontier_batch_mode="off")
        observe = self.coordinator(
            drones=2,
            focused_frontier_batch_mode="observe",
            focused_frontier_batch_maximum_detour_sensor_ranges=100.0,
            focused_frontier_batch_minimum_avoided_round_trip_sensor_ranges=0.0,
        )
        entries = ((20, 16), (24, 16), (28, 16))
        self._install_focused_tasks(off, entries)
        self._install_focused_tasks(observe, entries)

        off._schedule(known_free_slam())
        observe._schedule(known_free_slam())

        self.assertEqual(
            tuple(
                (drone_id, directive.kind, directive.task.task_id)
                for drone_id, directive in sorted(off._pending_directives.items())
            ),
            tuple(
                (drone_id, directive.kind, directive.task.task_id)
                for drone_id, directive
                in sorted(observe._pending_directives.items())
            ),
        )
        self.assertEqual(off.snapshot().claims, observe.snapshot().claims)
        self.assertFalse(observe.snapshot().spatial_leases)
        self.assertTrue(observe._take_batch_evaluations())

    def test_focused_frontier_batch_planning_budget_falls_back_to_singular_claim(
        self,
    ) -> None:
        coordinator = self.coordinator(
            drones=1,
            focused_frontier_batch_mode="active",
            focused_frontier_batch_maximum_detour_sensor_ranges=100.0,
            focused_frontier_batch_minimum_avoided_round_trip_sensor_ranges=0.0,
            focused_frontier_batch_maximum_planning_ms=1e-9,
        )
        self._install_focused_tasks(coordinator, ((20, 16), (24, 16)))

        coordinator._schedule(known_free_slam())
        directive = coordinator.claim_directive(0).directive
        summary = coordinator._last_batch_planning_summary

        self.assertEqual(directive.kind, DirectiveKind.COMPONENT_TASK)
        self.assertEqual(len(coordinator.snapshot().claims), 1)
        self.assertFalse(coordinator.snapshot().spatial_leases)
        self.assertIsNotNone(summary)
        self.assertEqual(summary.status, "planning_budget")
        self.assertEqual(summary.planned_batch_count, 0)

    def test_focused_frontier_batch_uses_highway_for_obstacle_connector(self) -> None:
        coordinator = self.coordinator(
            drones=1,
            focused_frontier_batch_mode="active",
            focused_frontier_batch_maximum_detour_sensor_ranges=100.0,
            focused_frontier_batch_minimum_avoided_round_trip_sensor_ranges=0.0,
        )
        self._install_focused_tasks(coordinator, ((18, 16), (24, 16)))
        occupancy = np.full((32, 32), FREE, dtype=np.int8)
        confidence = np.ones((32, 32), dtype=np.float32)
        occupancy[10:23, 20] = OCCUPIED
        slam = SlamSnapshot(occupancy, confidence, version=7)
        build = build_highway_graph(
            slam,
            confidence_threshold=0.5,
            macro_cell_size=8,
            maximum_build_ms=250.0,
            maximum_connector_expansions=4096,
        )
        self.assertIsNotNone(build.snapshot)
        coordinator.update_highway_snapshot(build.snapshot)

        coordinator._schedule(slam)
        directive = coordinator.claim_directive(0).directive
        summary = coordinator._last_batch_planning_summary

        self.assertEqual(directive.kind, DirectiveKind.COMPONENT_BATCH)
        self.assertEqual(len(directive.batch_members), 2)
        self.assertIsNotNone(directive.spatial_lease)
        self.assertEqual(summary.highway_version, 7)
        self.assertLessEqual(summary.route_queries, 128)

    def test_focused_frontier_batch_large_map_planning_meets_wall_budget(self) -> None:
        budget_ms = 250.0
        coordinator = FrontierTaskCoordinator(
            (1010, 1615),
            3,
            (807, 505),
            sensor_range=160.0,
            sensor_fov_deg=60.0,
            minimum_component_cells=1,
            minimum_unknown_support_cells=1,
            frontier_stride=4,
            global_cell_size=32,
            focused_frontier_batch_mode="active",
            focused_frontier_batch_maximum_detour_sensor_ranges=100.0,
            focused_frontier_batch_minimum_avoided_round_trip_sensor_ranges=0.0,
            focused_frontier_batch_maximum_planning_ms=budget_ms,
        )
        self._install_focused_tasks(
            coordinator,
            tuple(
                (560 + (index % 4) * 90, 340 + (index // 4) * 90)
                for index in range(12)
            ),
        )
        slam = SlamSnapshot(
            np.full((1010, 1615), FREE, dtype=np.int8),
            np.ones((1010, 1615), dtype=np.float32),
            version=1,
        )

        started = time.perf_counter()
        coordinator._schedule(slam)
        scheduling_ms = (time.perf_counter() - started) * 1000.0
        directives = tuple(
            coordinator.claim_directive(drone_id).directive
            for drone_id in range(3)
        )
        summary = coordinator._last_batch_planning_summary

        batches = tuple(
            directive for directive in directives
            if directive.kind == DirectiveKind.COMPONENT_BATCH
        )
        self.assertTrue(batches)
        self.assertTrue(all(
            directive.spatial_lease is not None for directive in batches
        ))
        self.assertEqual(summary.status, "complete")
        self.assertEqual(summary.planned_batch_count, len(batches))
        self.assertLessEqual(summary.elapsed_ms, budget_ms)
        self.assertLessEqual(summary.route_queries, 128)
        # Scheduling includes the full-map reachability pass in addition to
        # the planning interval, so leave only a small non-planner allowance.
        self.assertLessEqual(scheduling_ms, budget_ms + 50.0)

    def test_spatial_leases_clip_to_disjoint_raster_cells(self) -> None:
        coordinator = self.coordinator(
            drones=2,
            focused_frontier_batch_mode="active",
        )
        self._install_focused_tasks(
            coordinator,
            ((18, 16), (20, 16), (26, 16), (28, 16)),
        )
        slam = known_free_slam()
        first = coordinator._build_spatial_lease(
            0, 40, (0, 1), slam,
        )
        self.assertIsNotNone(first)
        coordinator._leases_by_id[first.lease_id] = first
        second = coordinator._build_spatial_lease(
            1, 41, (2, 3), slam,
        )

        self.assertIsNotNone(second)
        self.assertFalse(
            coordinator._lease_cells(first)
            & coordinator._lease_cells(second)
        )

    def test_malformed_batch_report_has_zero_partial_mutation(self) -> None:
        coordinator = self.coordinator(
            drones=1,
            focused_frontier_batch_mode="active",
            focused_frontier_batch_maximum_detour_sensor_ranges=100.0,
            focused_frontier_batch_minimum_avoided_round_trip_sensor_ranges=0.0,
        )
        self._install_focused_tasks(
            coordinator,
            ((20, 16), (24, 16)),
        )
        coordinator._schedule(known_free_slam())
        directive = coordinator.claim_directive(0).directive
        valid = self._complete_batch_report(directive, report_id=100)
        malformed_members = list(valid.batch_member_reports)
        malformed_members[1] = replace(
            malformed_members[1],
            claim_token=malformed_members[1].claim_token + 100,
        )

        rejected = coordinator.check_in(
            0,
            known_free_slam(version=2),
            report=replace(
                valid,
                batch_member_reports=tuple(malformed_members),
            ),
        )

        self.assertFalse(rejected.report_accepted)
        self.assertEqual(len(coordinator.snapshot().claims), 2)
        self.assertEqual(len(coordinator.snapshot().spatial_leases), 1)
        self.assertTrue(all(
            coordinator.registry.work_units[unit_id].state
            == WorkUnitState.ACTIVE
            for member in directive.batch_members
            for unit_id in member.claim.work_unit_ids
        ))
        self.assertNotIn(100, coordinator._accepted_report_ids)

    def test_batch_report_accepts_stale_peer_and_replay_once(self) -> None:
        coordinator = self.coordinator(
            drones=1,
            focused_frontier_batch_mode="active",
            focused_frontier_batch_maximum_detour_sensor_ranges=100.0,
            focused_frontier_batch_minimum_avoided_round_trip_sensor_ranges=0.0,
        )
        self._install_focused_tasks(
            coordinator,
            ((20, 16), (24, 16)),
        )
        coordinator._schedule(known_free_slam())
        directive = coordinator.claim_directive(0).directive
        stale = directive.batch_members[0]
        stale_unit = coordinator.registry.work_units[
            stale.claim.work_unit_ids[0]
        ]
        stale_unit.state = WorkUnitState.VISITED
        stale_unit.terminal_reason = "lineage_closed_elsewhere"
        report = self._complete_batch_report(directive, report_id=101)

        accepted = coordinator.check_in(
            0,
            known_free_slam(version=2),
            report=report,
        )
        revision = coordinator.registry.revision
        newer_active = ExplorationDirective(
            directive_id=999,
            kind=DirectiveKind.HOME,
            reason="newer_directive",
        )
        coordinator._active_directives[0] = newer_active
        replayed = coordinator.check_in(
            0,
            known_free_slam(version=2),
            report=report,
        )

        self.assertTrue(accepted.report_accepted)
        self.assertIn(
            (stale.task.task_id, "stale"),
            accepted.batch_member_statuses,
        )
        self.assertIn("applicable", dict(accepted.batch_member_statuses).values())
        self.assertFalse(coordinator.snapshot().claims)
        self.assertFalse(coordinator.snapshot().spatial_leases)
        self.assertEqual(stale_unit.terminal_reason, "lineage_closed_elsewhere")
        self.assertTrue(replayed.report_accepted)
        self.assertTrue(replayed.report_replayed)
        self.assertIs(coordinator._active_directives[0], newer_active)
        self.assertEqual(coordinator.registry.revision, revision)


if __name__ == "__main__":
    unittest.main()
