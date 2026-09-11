import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from agents.drone_movement import _CoordinationExecution
from agents.component_explorer import (
    LocalComponentNode,
    LocalDFSStack,
    local_node_pose,
    related_local_successors,
)
from agents.drone import Drone
from config.simulation_config import MissionConfig, SimulationConfig, SlamConfig
from mapping.drone_sensor import SensorScanCompletion
from mapping.frontier_registry import (
    ComponentWorkUnit,
    WorkUnitKind,
    WorkUnitState,
)
from mapping.terrain_knowledge import TerrainKnowledge
from mapping.slam_map import FREE, UNKNOWN, SlamSnapshot
from mission.exploration_coordination import (
    ClaimLease,
    CoordinationReport,
    CoordinationResult,
    DirectiveKind,
    ExplorationDirective,
    ExplorationTask,
    TaskState,
)
from mission.energy import EnergyReturnDecision, EnergyState
from navigation.astar_pathfinder import PATH_COMPLETE, PathResult


class ComponentControl:
    delay = 1 / 15
    terrain_roughness = np.zeros((64, 64), dtype=np.float32)

    def __init__(self) -> None:
        self.now = 1.0
        self.ready = threading.Event()
        self.directive = None
        self.reports = []
        self.energy_calls = 0
        self.force_energy_return_after = None
        self.path_result = None
        self.terrain_knowledge = TerrainKnowledge(
            np.zeros((64, 64), dtype=np.uint8)
        )
        self.terrain_fusion = SimpleNamespace(record_scan=lambda samples: None)

    def compute_path(self, start, goal):
        return [tuple(start), tuple(goal)]

    def compute_path_segment(self, start, goal):
        if self.path_result is not None:
            return self.path_result
        return PathResult(
            (tuple(start), tuple(goal)),
            PATH_COMPLETE,
            1,
            0.0,
        )

    def simulation_time(self):
        return self.now

    @staticmethod
    def pause_checkpoint():
        return True

    @staticmethod
    def wait_simulation_delay(_duration):
        return True

    @staticmethod
    def get_check_in_position():
        return (16, 16)

    def exploration_check_in(self, _drone_id, report):
        if report is not None:
            self.reports.append(report)
        return CoordinationResult(
            arrived=True,
            report_accepted=True,
            waiting=True,
            directive_ready=self.ready,
        )

    def exploration_assignment(self, _drone_id):
        self.ready.clear()
        return CoordinationResult(
            arrived=True,
            directive=self.directive,
        )

    def exploration_energy_return(
        self,
        _drone_id,
        route_home_cost,
        next_action_cost,
        safety_reserve,
    ):
        self.energy_calls += 1
        required = (
            self.force_energy_return_after is not None
            and self.energy_calls >= self.force_energy_return_after
        )
        return EnergyReturnDecision(
            state=EnergyState(5.0, 100.0, unlimited=not required),
            route_home_cost=route_home_cost,
            next_action_cost=next_action_cost,
            safety_reserve=safety_reserve,
            must_return=required,
        )


class LocalDFSStackTests(unittest.TestCase):
    def test_children_are_depth_first_and_backtrack_exact_outbound_path(self):
        stack = LocalDFSStack()
        root = stack.new_node(frozenset({(1, 1)}), (1, 1), 0)
        first = stack.new_node(frozenset({(2, 1)}), (2, 1), 90)
        second = stack.new_node(frozenset({(1, 2)}), (1, 2), 180)
        stack.start(root, parent_position=(0, 1))
        stack.record_outbound(((0, 1), (1, 1)))

        self.assertEqual(stack.record_scan((first, second)), first)
        stack.record_outbound(((1, 1), (2, 1)))
        self.assertIsNone(stack.record_scan(()))
        backtrack, sibling = stack.pop_completed()

        self.assertEqual(backtrack, ((2, 1), (1, 1)))
        self.assertEqual(sibling, second)
        self.assertEqual(stack.current.node, second)

    def test_node_limit_keeps_bounded_prefix_instead_of_dropping_all(self):
        stack = LocalDFSStack(maximum_nodes=2)
        root = stack.new_node(frozenset({(1, 1)}), (1, 1), 0)
        first = stack.new_node(frozenset({(2, 1)}), (2, 1), 90)
        second = stack.new_node(frozenset({(3, 1)}), (3, 1), 90)
        stack.start(root, parent_position=(0, 1))

        self.assertEqual(stack.record_scan((first, second)), first)
        self.assertTrue(stack.limit_reached)
        self.assertEqual(stack.created_nodes, 2)

    def test_local_pose_skips_visited_footprint_and_advances_outward(self):
        slam = SlamSnapshot(
            np.full((64, 64), UNKNOWN, dtype=np.int8),
            np.zeros((64, 64), dtype=np.float32),
            version=1,
        )
        cells = frozenset((x, 20) for x in range(10, 51))

        pose = local_node_pose(
            cells,
            slam,
            origin=(20, 20),
            excluded_anchors=((20, 20),),
            minimum_anchor_spacing=10.0,
            preferred_heading=90.0,
        )

        self.assertIsNotNone(pose)
        self.assertEqual(pose[0], (50, 20))
        self.assertIsNone(local_node_pose(
            cells,
            slam,
            origin=(20, 20),
            excluded_anchors=((20, 20),),
            minimum_anchor_spacing=100.0,
            preferred_heading=90.0,
        ))

    def test_scan_footprint_continues_to_frontier_beyond_lineage_radius(self):
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[32, 10] = FREE
        occupancy[32, 30] = FREE
        confidence[32, 10] = 1.0
        confidence[32, 30] = 1.0
        slam = SlamSnapshot(occupancy, confidence, version=1)
        source = frozenset({(10, 32)})

        lineage_only = related_local_successors(
            slam,
            source,
            confidence_threshold=0.5,
            minimum_component_cells=1,
            minimum_unknown_support_cells=1,
            lineage_radius=2,
        )
        scan_visible = related_local_successors(
            slam,
            source,
            confidence_threshold=0.5,
            minimum_component_cells=1,
            minimum_unknown_support_cells=1,
            lineage_radius=2,
            scan_origin=(10, 32),
            scan_heading=90.0,
            sensor_range=25.0,
            sensor_fov_deg=30.0,
        )

        self.assertNotIn(frozenset({(30, 32)}), lineage_only)
        self.assertIn(frozenset({(30, 32)}), scan_visible)


class ComponentMovementTests(unittest.TestCase):
    def setUp(self) -> None:
        pygame.init()
        settings = SimulationConfig(
            mission_config=MissionConfig(map_dim="LARGE", seed=19),
            slam=SlamConfig(
                scan_interval=0.0,
                scan_rays=5,
                point_cloud_max_points=50,
            ),
        )
        game = SimpleNamespace(
            sim_settings=settings,
            window=pygame.Surface((64, 64), pygame.SRCALPHA),
            width=64,
            height=64,
        )
        self.control = ComponentControl()
        self.drone = Drone(
            game,
            self.control,
            0,
            (16, 16),
            (255, 0, 0),
            pygame.Surface((4, 4), pygame.SRCALPHA),
            np.zeros((64, 64), dtype=np.uint8),
        )

    def _deliver(self, directive):
        self.control.directive = directive
        self.drone.move()
        self.control.ready.set()
        self.drone.move()

    def test_final_home_directive_completes_at_rover_without_launch_leg(self) -> None:
        self._deliver(ExplorationDirective(
            directive_id=6,
            kind=DirectiveKind.HOME,
            reason="component_work_exhausted",
        ))

        self.assertTrue(self.drone.snapshot().done)
        self.assertEqual(self.drone.snapshot().position, (16, 16))

    def test_reporting_falls_forward_only_after_confirmed_endpoint_is_empty(
        self,
    ) -> None:
        endpoint = [(16, 16)]
        arrival = Mock(return_value=CoordinationResult(arrived=False))
        controller = self.drone.movement_controller

        def missed(_drone_id, observed):
            self.assertEqual(observed, (16, 16))
            endpoint[0] = (30, 30)
            return endpoint[0]

        controller.dependencies = controller.dependencies.__class__(
            **{
                **controller.dependencies.__dict__,
                "get_check_in_position": lambda: endpoint[0],
                "rendezvous_endpoint_missed": missed,
                "exploration_check_in": arrival,
            }
        )
        controller._coordination_report = CoordinationReport(
            report_id=10,
            directive_id=7,
            kind=DirectiveKind.COMPONENT_TASK,
        )

        self.drone.move()

        arrival.assert_called_once()
        activity = controller.activity_snapshot()
        self.assertEqual(activity.state, "Reporting")
        self.assertEqual(activity.target, (30, 30))
        self.assertIn("rendezvous 30,30", activity.detail)

    def test_component_follower_claims_first_separated_branch(self) -> None:
        unit = ComponentWorkUnit(
            work_unit_id=5,
            component_id=2,
            component_revision=0,
            kind=WorkUnitKind.FOCUSED_ANCHOR,
            cells=frozenset({(16, 16)}),
            anchor_position=(16, 16),
            scan_headings=(90,),
            estimated_effort=1.0,
            state=WorkUnitState.CLAIMED,
        )
        task = ExplorationTask(
            task_id=3,
            component_id=2,
            component_revision=0,
            work_unit_ids=(5,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(16, 16),
            estimated_effort=1.0,
            state=TaskState.CLAIMED,
        )
        self._deliver(ExplorationDirective(
            directive_id=7,
            kind=DirectiveKind.COMPONENT_FOLLOW,
            task=task,
            work_units=(unit,),
            leader_drone_id=1,
            follow_branch_index=0,
        ))
        continuation = LocalComponentNode(
            0, frozenset({(20, 16)}), (20, 16), 90,
        )
        branch = LocalComponentNode(
            1, frozenset({(20, 20)}), (20, 20), 180,
        )

        with patch.object(
            self.drone.movement_controller,
            "_reachable_local_nodes",
            return_value=(branch, continuation),
        ):
            self.drone.move()

        execution = self.drone.movement_controller._coordination_execution
        self.assertEqual(execution.phase, "transit")
        self.assertEqual(execution.dfs.current.node.cells, branch.cells)
        self.assertEqual(execution.causal_successors, [branch.cells])

    def test_rover_scan_uses_sensor_completion_then_uploads_report(self) -> None:
        directive = ExplorationDirective(
            directive_id=7,
            kind=DirectiveKind.ROVER_SCAN,
            round_id=2,
            scan_headings=(0,),
        )
        self._deliver(directive)

        self.drone.move()
        self.drone.sensor_controller._last_completed_scan = SensorScanCompletion(
            pose=(16, 16, 0.0),
            sequence=1,
            newly_known_cells=4,
            confidence_gain=2.5,
        )
        self.drone.move()
        self.drone.move()

        self.assertEqual(len(self.control.reports), 1)
        report = self.control.reports[0]
        self.assertEqual(report.kind, DirectiveKind.ROVER_SCAN)
        self.assertEqual(report.completed_scan_headings, (0,))
        self.assertEqual(report.sensor_newly_known_cells, 4)

    def test_zero_gain_component_scan_retires_only_claimed_work_unit(self) -> None:
        unit = ComponentWorkUnit(
            work_unit_id=11,
            component_id=3,
            component_revision=0,
            kind=WorkUnitKind.SWEEP_ANCHOR,
            cells=frozenset({(16, 16), (17, 16)}),
            anchor_position=(16, 16),
            scan_headings=(90,),
            estimated_effort=2.0,
            state=WorkUnitState.CLAIMED,
        )
        task = ExplorationTask(
            task_id=5,
            component_id=3,
            component_revision=0,
            work_unit_ids=(11,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(16, 16),
            estimated_effort=2.0,
            state=TaskState.CLAIMED,
        )
        claim = ClaimLease(5, (11,), 0, 19, 1)
        directive = ExplorationDirective(
            directive_id=8,
            kind=DirectiveKind.COMPONENT_TASK,
            task=task,
            work_units=(unit,),
            claim=claim,
        )
        self._deliver(directive)

        self.drone.move()
        self.drone.sensor_controller._last_completed_scan = SensorScanCompletion(
            pose=(16, 16, 90.0),
            sequence=1,
            newly_known_cells=0,
            confidence_gain=0.0,
        )
        self.drone.move()
        self.drone.move()
        self.drone.move()

        self.assertEqual(len(self.control.reports), 1)
        report = self.control.reports[0]
        self.assertEqual(report.claim_token, 19)
        self.assertEqual(len(report.work_unit_outcomes), 1)
        self.assertEqual(report.work_unit_outcomes[0].work_unit_id, 11)
        self.assertEqual(report.work_unit_outcomes[0].disposition, "zero_gain")
        self.assertEqual(
            self.drone.movement_controller._suppressed_frontier_geometry,
            {},
        )

    def test_zero_gain_anchor_still_follows_visible_local_successor(self) -> None:
        unit = ComponentWorkUnit(
            work_unit_id=12,
            component_id=3,
            component_revision=0,
            kind=WorkUnitKind.FOCUSED_ANCHOR,
            cells=frozenset({(16, 16)}),
            anchor_position=(16, 16),
            scan_headings=(90,),
            estimated_effort=1.0,
            state=WorkUnitState.CLAIMED,
        )
        task = ExplorationTask(
            task_id=6,
            component_id=3,
            component_revision=0,
            work_unit_ids=(12,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(16, 16),
            estimated_effort=1.0,
            state=TaskState.CLAIMED,
        )
        self._deliver(ExplorationDirective(
            directive_id=14,
            kind=DirectiveKind.COMPONENT_TASK,
            task=task,
            work_units=(unit,),
            claim=ClaimLease(6, (12,), 0, 43, 1),
        ))

        self.drone.move()
        self.drone.sensor_controller._last_completed_scan = (
            SensorScanCompletion(
                pose=(16, 16, 90.0),
                sequence=1,
                newly_known_cells=0,
                confidence_gain=0.0,
            )
        )
        with patch(
            "agents.drone_movement.related_local_successors",
            return_value=(frozenset({(28, 16)}),),
        ), patch.object(
            self.drone.movement_controller,
            "_sweep_anchor_spacing",
            return_value=8.0,
        ):
            self.drone.move()

        execution = self.drone.movement_controller._coordination_execution
        self.assertIsNotNone(execution)
        self.assertEqual(len(execution.dfs.frames), 2)
        self.assertFalse(self.control.reports)

    def test_energy_checkpoint_suspends_and_returns_over_recorded_path(self) -> None:
        unit = ComponentWorkUnit(
            work_unit_id=21,
            component_id=4,
            component_revision=0,
            kind=WorkUnitKind.FOCUSED_ANCHOR,
            cells=frozenset({(20, 16)}),
            anchor_position=(20, 16),
            scan_headings=(90,),
            estimated_effort=1.0,
            state=WorkUnitState.CLAIMED,
        )
        task = ExplorationTask(
            task_id=6,
            component_id=4,
            component_revision=0,
            work_unit_ids=(21,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(20, 16),
            estimated_effort=1.0,
            state=TaskState.CLAIMED,
        )
        directive = ExplorationDirective(
            directive_id=9,
            kind=DirectiveKind.COMPONENT_TASK,
            outbound_route=((16, 16), (20, 16)),
            task=task,
            work_units=(unit,),
            claim=ClaimLease(6, (21,), 0, 23, 1),
        )
        self.control.force_energy_return_after = 2
        self._deliver(directive)

        self.drone.move()
        self.assertEqual(self.drone.snapshot().position, (20, 16))
        self.drone.move()
        self.drone.move()
        self.drone.move()

        self.assertEqual(self.drone.snapshot().position, (16, 16))
        self.assertEqual(len(self.control.reports), 1)
        report = self.control.reports[0]
        self.assertIsNotNone(report.suspension)
        self.assertEqual(report.suspension.reason, "energy_reserve")
        self.assertEqual(report.suspension.position_at_suspension, (20, 16))
        self.assertEqual(report.outbound_actual_path[0], (16, 16))
        self.assertEqual(report.outbound_actual_path[-1], (20, 16))
        self.assertEqual(report.return_actual_path[-1], (16, 16))

    def test_component_scan_timeout_suspends_instead_of_recording_zero_gain(self) -> None:
        unit = ComponentWorkUnit(
            work_unit_id=31,
            component_id=5,
            component_revision=0,
            kind=WorkUnitKind.FOCUSED_ANCHOR,
            cells=frozenset({(16, 16)}),
            anchor_position=(16, 16),
            scan_headings=(180,),
            estimated_effort=1.0,
            state=WorkUnitState.CLAIMED,
        )
        task = ExplorationTask(
            task_id=7,
            component_id=5,
            component_revision=0,
            work_unit_ids=(31,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(16, 16),
            estimated_effort=1.0,
            state=TaskState.CLAIMED,
        )
        self._deliver(ExplorationDirective(
            directive_id=10,
            kind=DirectiveKind.COMPONENT_TASK,
            task=task,
            work_units=(unit,),
            claim=ClaimLease(7, (31,), 0, 29, 1),
        ))

        self.drone.move()
        self.drone.move()
        self.control.now = 5.0
        self.drone.move()
        self.drone.move()
        self.drone.move()

        report = self.control.reports[0]
        self.assertEqual(report.work_unit_outcomes, ())
        self.assertIsNotNone(report.suspension)
        self.assertEqual(report.suspension.reason, "sensor_timeout")

    def test_component_sweep_services_all_claimed_anchors_before_return(self) -> None:
        units = tuple(
            ComponentWorkUnit(
                work_unit_id=unit_id,
                component_id=8,
                component_revision=0,
                kind=WorkUnitKind.SWEEP_ANCHOR,
                cells=frozenset({anchor}),
                anchor_position=anchor,
                scan_headings=(90,),
                estimated_effort=1.0,
                state=WorkUnitState.CLAIMED,
            )
            for unit_id, anchor in ((41, (16, 16)), (42, (28, 16)))
        )
        task = ExplorationTask(
            task_id=9,
            component_id=8,
            component_revision=0,
            work_unit_ids=(41, 42),
            parent_task_id=None,
            depth=0,
            preferred_entry=(16, 16),
            estimated_effort=2.0,
            state=TaskState.CLAIMED,
        )
        self._deliver(ExplorationDirective(
            directive_id=11,
            kind=DirectiveKind.COMPONENT_TASK,
            task=task,
            work_units=units,
            claim=ClaimLease(9, (41, 42), 0, 31, 1),
        ))

        sequence = 0
        for _ in range(30):
            execution = self.drone.movement_controller._coordination_execution
            if (
                execution is not None
                and execution.scan is not None
                and execution.scan.requested_at is not None
            ):
                sequence += 1
                heading = execution.scan.current_heading
                position = execution.scan.position
                self.drone.sensor_controller._last_completed_scan = (
                    SensorScanCompletion(
                        pose=(position[0], position[1], float(heading)),
                        sequence=sequence,
                        newly_known_cells=0,
                        confidence_gain=0.0,
                    )
                )
            self.drone.move()
            if self.control.reports:
                break

        self.assertEqual(len(self.control.reports), 1)
        self.assertEqual(
            tuple(
                outcome.work_unit_id
                for outcome in self.control.reports[0].work_unit_outcomes
            ),
            (41, 42),
        )

    def test_shared_slam_can_retire_stale_component_target_locally(self) -> None:
        unit = ComponentWorkUnit(
            work_unit_id=51,
            component_id=9,
            component_revision=0,
            kind=WorkUnitKind.FOCUSED_ANCHOR,
            cells=frozenset({(24, 16)}),
            anchor_position=(24, 16),
            scan_headings=(90,),
            estimated_effort=1.0,
            state=WorkUnitState.CLAIMED,
        )
        task = ExplorationTask(
            task_id=10,
            component_id=9,
            component_revision=0,
            work_unit_ids=(51,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(24, 16),
            estimated_effort=1.0,
            state=TaskState.CLAIMED,
        )
        self._deliver(ExplorationDirective(
            directive_id=12,
            kind=DirectiveKind.COMPONENT_TASK,
            task=task,
            work_units=(unit,),
            claim=ClaimLease(10, (51,), 0, 37, 1),
        ))

        self.drone.movement_controller.mark_shared_slam_changed()
        for _ in range(6):
            self.drone.move()
            if self.control.reports:
                break

        self.assertEqual(len(self.control.reports), 1)
        outcomes = self.control.reports[0].work_unit_outcomes
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].work_unit_id, 51)
        self.assertEqual(outcomes[0].disposition, "shared_resolved")

    def test_drone_rejects_unreachable_delegated_route_at_rover(self) -> None:
        unit = ComponentWorkUnit(
            work_unit_id=61,
            component_id=10,
            component_revision=0,
            kind=WorkUnitKind.FOCUSED_ANCHOR,
            cells=frozenset({(24, 16)}),
            anchor_position=(24, 16),
            scan_headings=(90,),
            estimated_effort=1.0,
            state=WorkUnitState.CLAIMED,
        )
        task = ExplorationTask(
            task_id=11,
            component_id=10,
            component_revision=0,
            work_unit_ids=(61,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(24, 16),
            estimated_effort=1.0,
            state=TaskState.CLAIMED,
        )
        self.control.path_result = PathResult(
            (),
            "unreachable",
            1,
            8.0,
        )
        self._deliver(ExplorationDirective(
            directive_id=13,
            kind=DirectiveKind.COMPONENT_TASK,
            task=task,
            work_units=(unit,),
            claim=ClaimLease(11, (61,), 0, 41, 1),
        ))

        for _ in range(8):
            self.drone.move()
            if self.control.reports:
                break

        self.assertEqual(self.drone.snapshot().position, (16, 16))
        self.assertEqual(len(self.control.reports), 1)
        self.assertEqual(
            self.control.reports[0].suspension.reason,
            "route_unreachable",
        )

    def test_failed_dfs_breadcrumb_replans_instead_of_retrying_forever(self) -> None:
        controller = self.drone.movement_controller
        dfs = LocalDFSStack()
        node = dfs.new_node(frozenset({(20, 16)}), (24, 16), 90)
        dfs.start(node, parent_position=(20, 16))
        execution = _CoordinationExecution(
            directive=ExplorationDirective(
                directive_id=14,
                kind=DirectiveKind.COMPONENT_TASK,
            ),
            phase="backtracking",
            path_start_index=0,
            dfs=dfs,
            pending_backtrack=((16, 16), (20, 16)),
            work_unit_outcomes=[],
            causal_successors=[],
            completed_scan_headings=[],
            timed_out_scan_headings=[],
            local_routes={},
            visited_successor_anchors=[],
            pending_authoritative_nodes=[],
        )
        controller._coordination_execution = execution
        activity = controller.activity_snapshot()
        self.assertEqual(activity.state, "DFS backtrack")
        self.assertEqual(activity.target, (20, 16))
        self.assertEqual(activity.dfs_depth, 1)

        calls = []

        def follow(path, *, source="path"):
            calls.append(source)
            if source == "component_dfs_backtrack":
                return False
            self.drone.runtime_state.move_to(tuple(path)[-1])
            return True

        with patch.object(controller, "_follow_path", side_effect=follow):
            with patch.object(
                controller,
                "_compute_path",
                return_value=PathResult(
                    ((16, 16), (20, 16)),
                    PATH_COMPLETE,
                    1,
                    0.0,
                ),
            ):
                controller._advance_component_task(execution)

        self.assertEqual(
            calls,
            [
                "component_dfs_backtrack",
                "component_dfs_backtrack_replan",
            ],
        )
        self.assertEqual(self.drone.snapshot().position, (20, 16))
        self.assertEqual(execution.pending_backtrack, ())
        self.assertEqual(execution.backtrack_attempts, 0)
        self.assertEqual(execution.phase, "transit")
        controller._coordination_execution = None


if __name__ == "__main__":
    unittest.main()
