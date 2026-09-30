import os
import math
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from agents.drone_movement import _CoordinationExecution
from agents.component_explorer import (
    IncidentalScanCandidate,
    IncidentalScanDetection,
    IncidentalPocketSignature,
    LocalComponentNode,
    LocalDFSStack,
    incidental_scan_candidate,
    leased_local_frontiers,
    local_node_pose,
    observation_pose_on_path,
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
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from mission.exploration_coordination import (
    BatchMember,
    ClaimLease,
    CoordinationReport,
    CoordinationResult,
    DirectiveKind,
    ExplorationDirective,
    ExplorationTask,
    SpatialLease,
    TaskState,
    WorkUnitOutcome,
)
from mission.energy import EnergyReturnDecision, EnergyState
from navigation.astar_pathfinder import (
    PATH_COMPLETE,
    PATH_UNREACHABLE,
    PathResult,
)
from navigation.highway import build_highway_graph


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
    @staticmethod
    def _side_pocket_slam(*, wall=True, open_window=False, occluded=False):
        occupancy = np.full((48, 48), FREE, dtype=np.int8)
        confidence = np.ones((48, 48), dtype=np.float32)
        occupancy[19:22, 27:30] = UNKNOWN
        confidence[19:22, 27:30] = 0.0
        if wall:
            occupancy[18:23, 30] = OCCUPIED
        if open_window:
            occupancy[20, 30:33] = UNKNOWN
            confidence[20, 30:33] = 0.0
        if occluded:
            occupancy[18:23, 24] = OCCUPIED
        return SlamSnapshot(occupancy, confidence, version=7)

    def test_incidental_detector_accepts_enclosed_wall_adjacent_side_pocket(self):
        first = incidental_scan_candidate(
            self._side_pocket_slam(),
            position=(20, 20),
            current_heading=0.0,
            sensor_range=12.0,
            sensor_fov_deg=60.0,
            confidence_threshold=0.6,
            frontier_stride=4,
        )
        second = incidental_scan_candidate(
            self._side_pocket_slam(),
            position=(20, 20),
            current_heading=0.0,
            sensor_range=12.0,
            sensor_fov_deg=60.0,
            confidence_threshold=0.6,
            frontier_stride=4,
        )

        self.assertIsNotNone(first.candidate)
        assert first.candidate is not None and second.candidate is not None
        self.assertEqual(first.candidate.heading, 90)
        self.assertEqual(len(first.candidate.cells), 9)
        self.assertEqual(first.candidate.signature, second.candidate.signature)
        self.assertGreater(first.candidate.side_only_support, 0)

    def test_incidental_detector_rejects_open_no_wall_occluded_visible_rear_and_active(self):
        cases = (
            (self._side_pocket_slam(open_window=True), 0.0, (), "window_boundary"),
            (self._side_pocket_slam(wall=False), 0.0, (), "no_wall"),
            (self._side_pocket_slam(occluded=True), 0.0, (), "occluded"),
            (self._side_pocket_slam(), 90.0, (), "already_visible"),
            (self._side_pocket_slam(), 270.0, (), "rear_facing"),
            (self._side_pocket_slam(), 0.0, ((27, 20),), "active_target"),
        )
        for slam, heading, active, rejection in cases:
            with self.subTest(rejection=rejection):
                result = incidental_scan_candidate(
                    slam,
                    position=(20, 20),
                    current_heading=heading,
                    sensor_range=12.0,
                    sensor_fov_deg=60.0,
                    confidence_threshold=0.6,
                    frontier_stride=4,
                    active_cells=active,
                )
                self.assertIsNone(result.candidate)
                self.assertGreater(dict(result.rejection_counts)[rejection], 0)

    def test_incidental_detector_rejects_pocket_that_cannot_fit_one_cone(self):
        occupancy = np.full((48, 48), FREE, dtype=np.int8)
        confidence = np.ones((48, 48), dtype=np.float32)
        occupancy[14:27, 27] = UNKNOWN
        confidence[14:27, 27] = 0.0
        occupancy[14:27, 28] = OCCUPIED
        result = incidental_scan_candidate(
            SlamSnapshot(occupancy, confidence, version=1),
            position=(20, 20),
            current_heading=0.0,
            sensor_range=12.0,
            sensor_fov_deg=60.0,
            confidence_threshold=0.6,
            frontier_stride=4,
        )

        self.assertIsNone(result.candidate)
        self.assertGreater(
            dict(result.rejection_counts)["oversized_cone"], 0
        )

    def test_observation_pose_trims_known_clear_route_suffix(self):
        occupancy = np.full((24, 24), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((24, 24), dtype=np.float32)
        occupancy[8:13, 1:11] = FREE
        confidence[8:13, 1:11] = 1.0
        cells = frozenset({(10, 9), (10, 10), (10, 11)})
        path = tuple((x, 10) for x in range(1, 11))

        pose = observation_pose_on_path(
            cells,
            SlamSnapshot(occupancy, confidence, version=1),
            path,
            origin=(1, 10),
            sensor_range=8.0,
            sensor_fov_deg=60.0,
            confidence_threshold=0.5,
        )

        self.assertIsNotNone(pose)
        self.assertEqual(pose.position, (4, 10))
        self.assertEqual(pose.heading, 90)
        self.assertEqual(pose.route_prefix[-1], pose.position)
        self.assertEqual(pose.saved_route_distance, 6.0)

    def test_observation_pose_requires_known_clear_line_of_sight(self):
        occupancy = np.full((24, 24), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((24, 24), dtype=np.float32)
        occupancy[8:13, 1:11] = FREE
        confidence[8:13, 1:11] = 1.0
        occupancy[10, 6] = OCCUPIED
        cells = frozenset({(10, 9), (10, 10), (10, 11)})

        pose = observation_pose_on_path(
            cells,
            SlamSnapshot(occupancy, confidence, version=1),
            tuple((x, 10) for x in range(1, 11)),
            origin=(1, 10),
            sensor_range=8.0,
            sensor_fov_deg=60.0,
            confidence_threshold=0.5,
        )

        self.assertIsNotNone(pose)
        self.assertGreater(pose.position[0], 6)

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

    def test_leased_frontier_requires_full_interior_and_no_reservation(self):
        occupancy = np.full((24, 24), FREE, dtype=np.int8)
        confidence = np.ones((24, 24), dtype=np.float32)
        occupancy[10:13, 10:13] = UNKNOWN
        confidence[10:13, 10:13] = 0.0
        slam = SlamSnapshot(occupancy, confidence, version=1)
        all_cells = tuple(
            (x, y) for y in range(24) for x in range(24)
        )

        admitted = leased_local_frontiers(
            slam,
            allowed_cells=all_cells,
            confidence_threshold=0.6,
            minimum_component_cells=1,
            minimum_unknown_support_cells=1,
        )

        self.assertEqual(len(admitted), 1)
        component = admitted[0]
        self.assertFalse(leased_local_frontiers(
            slam,
            allowed_cells=component,
            confidence_threshold=0.6,
            minimum_component_cells=1,
            minimum_unknown_support_cells=1,
        ))
        self.assertFalse(leased_local_frontiers(
            slam,
            allowed_cells=all_cells,
            excluded_geometry=(component,),
            confidence_threshold=0.6,
            minimum_component_cells=1,
            minimum_unknown_support_cells=1,
        ))


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

    @staticmethod
    def _highway_snapshot(version: int = 11):
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        result = build_highway_graph(
            SlamSnapshot(occupancy, confidence, version=version),
            confidence_threshold=0.6,
            macro_cell_size=16,
            maximum_build_ms=250.0,
            maximum_connector_expansions=4096,
        )
        return result.snapshot

    def test_highway_snapshot_is_accepted_only_through_coordination_result(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        snapshot = self._highway_snapshot()
        trace = Mock()
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )

        controller._apply_coordination_result(CoordinationResult(
            arrived=True,
            waiting=True,
            highway_snapshot=snapshot,
        ))

        self.assertIs(controller._highway_snapshot, snapshot)
        event = next(
            call for call in trace.record.call_args_list
            if call.args[0] == "drone_highway_snapshot_received"
        )
        self.assertEqual(event.kwargs["highway_version"], 11)

    def test_active_highway_routes_long_component_check_in_without_astar(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        controller.highway_mode = "active"
        controller.highway_minimum_route_sensor_ranges = 0.0
        controller._highway_snapshot = self._highway_snapshot()

        with patch.object(controller, "_compute_path") as compute:
            reached, encountered = controller._return_to_coordination_point(
                (40, 16),
            )

        self.assertTrue(reached)
        self.assertFalse(encountered)
        self.assertEqual(self.drone.snapshot().position, (40, 16))
        compute.assert_not_called()

    def test_observe_or_local_obstacle_falls_back_from_highway(self) -> None:
        controller = self.drone.movement_controller
        controller.highway_minimum_route_sensor_ranges = 0.0
        controller._highway_snapshot = self._highway_snapshot()
        trace = Mock()
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )

        controller.highway_mode = "observe"
        self.assertFalse(controller._component_check_in_highway_route(
            (16, 16), (40, 16),
        ))
        observe_event = trace.record.call_args_list[-1]
        self.assertFalse(observe_event.kwargs["selected"])
        self.assertEqual(observe_event.kwargs["fallback_reason"], "observe_only")

        advised = controller._highway_snapshot.route(
            (16, 16),
            (40, 16),
            maximum_query_ms=50.0,
            maximum_connector_expansions=4096,
        )
        blocked = advised.path[len(advised.path) // 2]
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[blocked[1], blocked[0]] = OCCUPIED
        confidence[blocked[1], blocked[0]] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        controller.highway_mode = "active"
        self.assertFalse(controller._component_check_in_highway_route(
            (16, 16), (40, 16),
        ))
        occupied_event = trace.record.call_args_list[-1]
        self.assertFalse(occupied_event.kwargs["selected"])
        self.assertEqual(
            occupied_event.kwargs["fallback_reason"],
            "locally_known_occupied",
        )

    @staticmethod
    def _batch_member(task_id: int, entry: tuple[int, int]) -> BatchMember:
        unit = ComponentWorkUnit(
            work_unit_id=task_id,
            component_id=task_id,
            component_revision=0,
            kind=WorkUnitKind.FOCUSED_ANCHOR,
            cells=frozenset({entry}),
            anchor_position=entry,
            scan_headings=(90,),
            estimated_effort=1.0,
            state=WorkUnitState.CLAIMED,
        )
        task = ExplorationTask(
            task_id=task_id,
            component_id=task_id,
            component_revision=0,
            work_unit_ids=(task_id,),
            parent_task_id=None,
            depth=0,
            preferred_entry=entry,
            estimated_effort=1.0,
            state=TaskState.CLAIMED,
        )
        return BatchMember(
            task=task,
            work_units=(unit,),
            claim=ClaimLease(task_id, (task_id,), 0, task_id + 10, 1),
            estimated_service_cost=1.0,
        )

    @staticmethod
    def _incidental_candidate(
        *,
        cells=frozenset({(24, 20), (25, 20)}),
        gateway=(23, 20),
        heading=180,
    ):
        return IncidentalScanCandidate(
            signature=IncidentalPocketSignature(
                support_bins=tuple(sorted({
                    (x // 4, y // 4) for x, y in cells
                })),
                gateway_bin=(gateway[0] // 4, gateway[1] // 4),
                wall_bin=(6, 5),
            ),
            cells=frozenset(cells),
            gateway=gateway,
            wall_contact=(26, 20),
            wall_cells=2,
            heading=heading,
            heading_delta=90.0,
            current_visibility=0,
            proposed_visibility=len(cells),
            side_only_support=len(cells),
        )

    def _install_incidental_execution(self, *, mode="active"):
        controller = self.drone.movement_controller
        controller.incidental_scan_mode = mode
        controller.incidental_maximum_attempts = 1
        controller.incidental_maximum_wait = 6.0
        controller.incidental_attempt_timeout = 3.0
        controller.incidental_maximum_rotation = 180.0
        controller.incidental_cooldown_ranges = 0.0
        controller.incidental_sample_spacing_ranges = 0.0
        execution = _CoordinationExecution(
            directive=ExplorationDirective(
                directive_id=91,
                kind=DirectiveKind.COMPONENT_TASK,
            ),
            phase="transit",
            path_start_index=0,
        )
        controller._coordination_execution = execution
        return controller, execution

    def test_batch_reorders_by_full_local_tour_and_continues_after_failure(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        self.drone.runtime_state.move_to((24, 16))
        members = (
            self._batch_member(0, (20, 16)),
            self._batch_member(1, (30, 16)),
        )
        directive = ExplorationDirective(
            directive_id=70,
            kind=DirectiveKind.COMPONENT_BATCH,
            batch_members=members,
            spatial_lease=SpatialLease(
                lease_id=3,
                owner_drone_id=0,
                directive_id=70,
                issued_revision=1,
                cell_spans=((16, 18, 32),),
                member_task_ids=(0, 1),
            ),
            maximum_total_components=3,
            maximum_detour_distance=100.0,
            minimum_avoided_round_trip=0.0,
            maximum_service_seconds=45.0,
            maximum_total_dfs_nodes=48,
            maximum_consecutive_low_gain_scans=2,
        )

        with patch.object(
            controller,
            "_local_slam_distance",
            side_effect=lambda first, second: math.dist(first, second),
        ):
            controller._install_exploration_directive(directive)
            execution = controller._coordination_execution
            self.assertEqual(execution.batch_current_member.task.task_id, 1)
            controller._prepare_execution_return(
                execution,
                reason="route_unreachable",
            )

        self.assertEqual(execution.batch_current_member.task.task_id, 0)
        self.assertEqual(len(execution.batch_completed_members), 1)
        failed = execution.batch_completed_members[0]
        self.assertEqual(failed.member.task.task_id, 1)
        self.assertEqual(failed.disposition, "unreachable")
        self.assertIsNotNone(failed.suspension)
        self.assertEqual(execution.phase, "transit")

        execution.work_unit_outcomes = (
            WorkUnitOutcome(0, "sensor_gain"),
        )
        controller._capture_batch_member(
            execution,
            disposition="complete",
        )
        execution.return_actual_path = ((24, 16), (16, 16))
        controller._finish_batch_execution(execution)
        report = controller._coordination_report

        self.assertEqual(report.kind, DirectiveKind.COMPONENT_BATCH)
        self.assertIsNone(report.claim_token)
        self.assertFalse(report.work_unit_outcomes)
        self.assertEqual(
            {item.task_id for item in report.batch_member_reports},
            {0, 1},
        )
        self.assertEqual(
            {item.claim_token for item in report.batch_member_reports},
            {10, 11},
        )

    def test_incidental_scan_requires_exact_pose_and_newer_sequence_then_resumes_suffix(self):
        controller, execution = self._install_incidental_execution()
        trace = Mock()
        controller.dependencies = controller.dependencies.__class__(
            **{
                **controller.dependencies.__dict__,
                "runtime_trace": trace,
            }
        )
        candidate = self._incidental_candidate()
        detection = IncidentalScanDetection(candidate, 1, ())
        sensor = self.drone.sensor_controller
        sensor._last_completed_scan = SensorScanCompletion(
            pose=(16, 16, 0.0),
            sequence=4,
            newly_known_cells=0,
            confidence_gain=0.0,
        )
        with patch(
            "agents.drone_movement.incidental_scan_candidate",
            return_value=detection,
        ):
            followed = controller._follow_path(
                ((16, 16), (17, 16), (18, 16)),
                source="component_task_transit",
                incidental_transit=True,
                path_target=(18, 16),
                path_status=PATH_COMPLETE,
            )
            self.assertFalse(followed)
            pending = controller._pending_incidental_scan
            self.assertIsNotNone(pending)
            assert pending is not None
            self.assertEqual(pending.retained_route, ((18, 16),))
            self.assertEqual(self.drone.snapshot().position, (17, 16))
            self.assertEqual(
                controller.activity_snapshot().state,
                "Incidental scan",
            )

            sensor._last_completed_scan = SensorScanCompletion(
                pose=(17, 16, float(candidate.heading)),
                sequence=4,
                newly_known_cells=1,
                confidence_gain=0.2,
            )
            controller._advance_pending_incidental_scan(execution)
            self.assertIsNotNone(controller._pending_incidental_scan)
            sensor._last_completed_scan = SensorScanCompletion(
                pose=(18, 16, float(candidate.heading)),
                sequence=5,
                newly_known_cells=1,
                confidence_gain=0.2,
            )
            controller._advance_pending_incidental_scan(execution)
            self.assertIsNotNone(controller._pending_incidental_scan)
            sensor._last_completed_scan = SensorScanCompletion(
                pose=(17, 16, float(candidate.heading)),
                sequence=5,
                newly_known_cells=3,
                confidence_gain=0.4,
            )
            with patch.object(controller, "rebuild_frontiers"):
                controller._advance_pending_incidental_scan(execution)

        self.assertIsNone(controller._pending_incidental_scan)
        self.assertEqual(self.drone.snapshot().position, (18, 16))
        self.assertEqual(execution.incidental_completions, 1)
        with patch.object(controller, "_compute_path") as compute_path:
            self.assertTrue(controller._advance_directive_transit(
                execution,
                (18, 16),
                source="component_task_transit",
            ))
            compute_path.assert_not_called()
        event_names = [call.args[0] for call in trace.record.call_args_list]
        self.assertIn("drone_incidental_scan_started", event_names)
        self.assertIn("drone_incidental_scan_finished", event_names)
        self.assertIn("drone_incidental_route_resumed", event_names)

    def test_incidental_timeout_resumes_without_route_or_reposition_strike(self):
        controller, execution = self._install_incidental_execution()
        execution.route_attempts = 3
        execution.reposition_attempts = 2
        candidate = self._incidental_candidate()
        with patch(
            "agents.drone_movement.incidental_scan_candidate",
            return_value=IncidentalScanDetection(candidate, 1, ()),
        ):
            controller._follow_path(
                ((16, 16), (17, 16), (18, 16)),
                source="component_task_transit",
                incidental_transit=True,
                path_target=(18, 16),
                path_status=PATH_COMPLETE,
            )
        pending = controller._pending_incidental_scan
        self.assertIsNotNone(pending)
        assert pending is not None
        self.control.now = pending.deadline + 0.01
        controller._advance_pending_incidental_scan(execution)

        self.assertEqual(self.drone.snapshot().position, (18, 16))
        self.assertEqual(execution.route_attempts, 3)
        self.assertEqual(execution.reposition_attempts, 2)
        self.assertEqual(execution.incidental_timeouts, 1)

    def test_incidental_cancellation_discards_retained_suffix(self):
        controller, execution = self._install_incidental_execution()
        trace = Mock()
        controller.dependencies = controller.dependencies.__class__(
            **{
                **controller.dependencies.__dict__,
                "runtime_trace": trace,
            }
        )
        candidate = self._incidental_candidate()
        with patch(
            "agents.drone_movement.incidental_scan_candidate",
            return_value=IncidentalScanDetection(candidate, 1, ()),
        ):
            controller._follow_path(
                ((16, 16), (17, 16), (18, 16)),
                source="component_task_transit",
                incidental_transit=True,
                path_target=(18, 16),
                path_status=PATH_COMPLETE,
            )
        self.control.now += 1.0
        controller._cancel_pending_incidental_scan("energy_reserve")

        self.assertIsNone(controller._pending_incidental_scan)
        self.assertEqual(self.drone.snapshot().position, (17, 16))
        self.assertEqual(execution.incidental_timeouts, 0)
        finished = [
            call.kwargs
            for call in trace.record.call_args_list
            if call.args[0] == "drone_incidental_scan_finished"
        ]
        self.assertEqual(finished[-1]["outcome"], "cancelled")
        self.assertEqual(
            finished[-1]["interruption_reason"],
            "energy_reserve",
        )

    def test_invalid_retained_edge_is_discarded_and_normal_transit_replans(self):
        controller, execution = self._install_incidental_execution()
        candidate = self._incidental_candidate()
        sensor = self.drone.sensor_controller
        with patch(
            "agents.drone_movement.incidental_scan_candidate",
            return_value=IncidentalScanDetection(candidate, 1, ()),
        ), patch.object(
            self.drone.runtime_state,
            "graph_is_valid",
            side_effect=(True, False),
        ):
            controller._follow_path(
                ((16, 16), (17, 16), (18, 16)),
                source="component_task_transit",
                incidental_transit=True,
                path_target=(18, 16),
                path_status=PATH_COMPLETE,
            )
            pending = controller._pending_incidental_scan
            self.assertIsNotNone(pending)
            assert pending is not None
            sensor._last_completed_scan = SensorScanCompletion(
                pose=(17, 16, float(candidate.heading)),
                sequence=pending.minimum_scan_sequence + 1,
                newly_known_cells=1,
                confidence_gain=0.1,
            )
            with patch.object(controller, "rebuild_frontiers"):
                controller._advance_pending_incidental_scan(execution)

        self.assertEqual(self.drone.snapshot().position, (17, 16))
        self.assertIsNone(controller._pending_incidental_scan)
        replanned = PathResult(
            ((17, 16), (18, 16)),
            PATH_COMPLETE,
            1,
            0.0,
        )
        with patch.object(
            controller,
            "_compute_path",
            return_value=replanned,
        ) as compute_path:
            self.assertTrue(controller._advance_directive_transit(
                execution,
                (18, 16),
                source="component_task_transit",
            ))
        compute_path.assert_called_once_with((17, 16), (18, 16))
        self.assertEqual(self.drone.snapshot().position, (18, 16))

    def test_incidental_observe_mode_has_no_rotation_or_pause(self):
        controller, execution = self._install_incidental_execution(
            mode="observe"
        )
        trace = Mock()
        controller.dependencies = controller.dependencies.__class__(
            **{
                **controller.dependencies.__dict__,
                "runtime_trace": trace,
            }
        )
        candidate = self._incidental_candidate()
        with patch(
            "agents.drone_movement.incidental_scan_candidate",
            return_value=IncidentalScanDetection(candidate, 1, ()),
        ):
            followed = controller._follow_path(
                ((16, 16), (17, 16), (18, 16)),
                source="component_task_transit",
                incidental_transit=True,
                path_target=(18, 16),
                path_status=PATH_COMPLETE,
            )

        self.assertTrue(followed)
        self.assertIsNone(controller._pending_incidental_scan)
        self.assertEqual(self.drone.snapshot().position, (18, 16))
        self.assertEqual(self.drone.snapshot().heading_deg, 90.0)
        self.assertEqual(execution.incidental_attempts, 1)
        candidate_events = [
            call.kwargs
            for call in trace.record.call_args_list
            if call.args[0] == "drone_incidental_scan_candidate"
        ]
        self.assertEqual(candidate_events[0]["mode"], "observe")

    def test_incidental_duplicate_suppression_and_material_reopening(self):
        controller, _execution = self._install_incidental_execution()
        original = self._incidental_candidate()
        controller._remember_incidental_attempt(original)
        self.assertFalse(controller._incidental_candidate_is_reopenable(original))

        expanded = self._incidental_candidate(cells=frozenset(
            set(original.cells) | {(30 + index, 20) for index in range(8)}
        ))
        moved = self._incidental_candidate(gateway=(33, 20))
        self.assertTrue(controller._incidental_candidate_is_reopenable(expanded))
        self.assertTrue(controller._incidental_candidate_is_reopenable(moved))

    def test_contact_and_shared_invalidation_precede_incidental_selection(self):
        controller, _execution = self._install_incidental_execution()
        order = []
        controller.dependencies = controller.dependencies.__class__(
            **{
                **controller.dependencies.__dict__,
                "physical_contact_checkpoint": (
                    lambda _drone_id: order.append("contact")
                ),
            }
        )

        def stop_after_share():
            order.append("shared_invalidation")
            return True

        with patch(
            "agents.drone_movement.incidental_scan_candidate"
        ) as detector:
            followed = controller._follow_path(
                ((16, 16), (17, 16)),
                source="component_task_transit",
                stop_when=stop_after_share,
                stop_reason="shared_slam_invalidated",
                incidental_transit=True,
                path_target=(17, 16),
                path_status=PATH_COMPLETE,
            )

        self.assertFalse(followed)
        self.assertEqual(order, ["contact", "shared_invalidation"])
        detector.assert_not_called()

    def test_incidental_sampling_excludes_noneligible_route_contexts(self):
        controller, execution = self._install_incidental_execution()
        cases = (
            (DirectiveKind.COMPONENT_FOLLOW, "following", "component_branch_follow"),
            (DirectiveKind.COMPONENT_TASK, "repositioning", "component_dfs_breadcrumb_fallback"),
            (DirectiveKind.COMPONENT_TASK, "returning", "component_task_transit"),
            (DirectiveKind.RADIAL_PROBE, "transit", "radial_probe_outbound"),
        )
        for kind, phase, source in cases:
            with self.subTest(kind=kind, phase=phase, source=source):
                execution.directive = ExplorationDirective(
                    directive_id=91,
                    kind=kind,
                )
                execution.phase = phase
                with patch(
                    "agents.drone_movement.incidental_scan_candidate"
                ) as detector:
                    selected = controller._maybe_sample_incidental_transit(
                        edge_distance=100.0,
                        retained_route=(),
                        route_target=(18, 16),
                        route_source=source,
                        path_status=PATH_COMPLETE,
                    )
                self.assertFalse(selected)
                detector.assert_not_called()

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

    def test_reporting_stays_at_rover_while_queued_report_is_pending(
        self,
    ) -> None:
        ready = threading.Event()
        arrival = Mock(return_value=CoordinationResult(
            arrived=True,
            report_accepted=False,
            waiting=True,
            directive_ready=ready,
        ))
        missed = Mock(return_value=(30, 30))
        controller = self.drone.movement_controller
        controller.dependencies = controller.dependencies.__class__(
            **{
                **controller.dependencies.__dict__,
                "get_check_in_position": lambda: (16, 16),
                "rendezvous_endpoint_missed": missed,
                "exploration_check_in": arrival,
            }
        )
        controller._coordination_report = CoordinationReport(
            report_id=11,
            directive_id=8,
            kind=DirectiveKind.COMPONENT_TASK,
        )

        self.drone.move()

        arrival.assert_called_once()
        missed.assert_not_called()
        self.assertEqual(controller.activity_snapshot().target, (16, 16))
        self.assertIs(controller._coordination_ready, ready)

    def test_returning_path_stops_at_physical_rover_and_queues_report(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        ready = threading.Event()
        queued = []
        positions_checked = []

        def request_stop(_drone_id):
            position = self.drone.snapshot().position
            positions_checked.append(position)
            return position == (18, 16)

        def check_in(_drone_id, report):
            self.assertEqual(self.drone.snapshot().position, (18, 16))
            queued.append(report)
            return CoordinationResult(
                arrived=True,
                report_accepted=False,
                waiting=True,
                directive_ready=ready,
            )

        controller.dependencies = controller.dependencies.__class__(**{
            **controller.dependencies.__dict__,
            "get_check_in_position": lambda: (24, 16),
            "request_exploration_report_stop": request_stop,
            "exploration_check_in": check_in,
        })
        execution = _CoordinationExecution(
            directive=ExplorationDirective(
                directive_id=40,
                kind=DirectiveKind.COMPONENT_TASK,
            ),
            phase="returning",
            path_start_index=0,
            work_unit_outcomes=[],
            causal_successors=[],
            completed_scan_headings=[],
            timed_out_scan_headings=[],
        )
        controller._coordination_execution = execution
        with patch.object(controller, "_compute_path", return_value=PathResult(
            tuple((x, 16) for x in range(16, 25)),
            PATH_COMPLETE,
            9,
            0.0,
        )):
            controller._advance_component_task(execution)

        self.assertEqual(self.drone.snapshot().position, (18, 16))
        self.assertIn((18, 16), positions_checked)
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0].return_path_source, "rover_encounter")
        self.assertEqual(queued[0].return_actual_path[-1], (18, 16))
        self.assertIs(controller._coordination_ready, ready)
        self.assertIs(controller._coordination_report, queued[0])

    def test_reporting_uses_direct_contact_before_old_endpoint(self) -> None:
        controller = self.drone.movement_controller
        report = CoordinationReport(
            report_id=41,
            directive_id=41,
            kind=DirectiveKind.COMPONENT_TASK,
        )
        controller._coordination_report = report
        check_in = Mock(return_value=CoordinationResult(
            arrived=True,
            report_accepted=False,
            waiting=True,
            directive_ready=threading.Event(),
        ))
        controller.dependencies = controller.dependencies.__class__(**{
            **controller.dependencies.__dict__,
            "get_check_in_position": lambda: (24, 16),
            "exploration_check_in": check_in,
        })

        with patch.object(controller, "_compute_path") as path:
            controller._advance_component_check_in()

        check_in.assert_called_once_with(self.drone.id, report)
        path.assert_not_called()
        self.assertEqual(self.drone.snapshot().position, (16, 16))

    def test_empty_check_in_docks_when_route_crosses_moving_rover(self) -> None:
        controller = self.drone.movement_controller
        ready = threading.Event()
        docked = [False]
        dock_positions = []
        fallback_check_in = Mock()

        def request_dock(_drone_id):
            position = self.drone.snapshot().position
            dock_positions.append(position)
            if position != (18, 16):
                return CoordinationResult(arrived=False)
            docked[0] = True
            return CoordinationResult(
                arrived=True,
                waiting=True,
                directive_ready=ready,
            )

        controller.dependencies = controller.dependencies.__class__(**{
            **controller.dependencies.__dict__,
            "get_check_in_position": lambda: (24, 16),
            "request_exploration_dock": request_dock,
            "is_exploration_docked": lambda _drone_id: docked[0],
            "exploration_check_in": fallback_check_in,
        })
        with patch.object(controller, "_compute_path", return_value=PathResult(
            tuple((x, 16) for x in range(16, 25)),
            PATH_COMPLETE,
            9,
            0.0,
        )):
            controller._advance_component_check_in()

        self.assertEqual(self.drone.snapshot().position, (18, 16))
        self.assertIn((18, 16), dock_positions)
        self.assertIs(controller._coordination_ready, ready)
        self.assertEqual(controller.activity_snapshot().state, "Docked")
        fallback_check_in.assert_not_called()

    def test_docked_drone_skips_local_work_until_assignment_is_ready(self) -> None:
        controller = self.drone.movement_controller
        ready = threading.Event()
        controller._coordination_ready = ready
        controller.dependencies = controller.dependencies.__class__(**{
            **controller.dependencies.__dict__,
            "is_exploration_docked": lambda _drone_id: True,
        })

        with patch.object(
            controller,
            "_refresh_frontiers_before_mission_state",
        ) as refresh:
            self.drone.move()

        refresh.assert_not_called()
        self.assertFalse(self.drone.snapshot().done)

        self.control.directive = ExplorationDirective(
            directive_id=44,
            kind=DirectiveKind.HOME,
            reason="component_work_exhausted",
        )
        ready.set()
        self.drone.move()

        self.assertTrue(self.drone.snapshot().done)

    def test_failed_queued_contact_keeps_report_for_retry(self) -> None:
        controller = self.drone.movement_controller
        report = CoordinationReport(
            report_id=42,
            directive_id=42,
            kind=DirectiveKind.COMPONENT_TASK,
        )
        ready = threading.Event()
        ready.set()
        controller._coordination_report = report
        controller._coordination_ready = ready
        assignment = Mock(return_value=CoordinationResult(arrived=False))
        controller.dependencies = controller.dependencies.__class__(**{
            **controller.dependencies.__dict__,
            "exploration_assignment": assignment,
        })

        controller._advance_component_check_in()

        assignment.assert_called_once_with(self.drone.id)
        self.assertIs(controller._coordination_report, report)
        self.assertIsNone(controller._coordination_ready)

    def test_active_task_transit_does_not_request_report_stop(self) -> None:
        controller = self.drone.movement_controller
        request_stop = Mock(return_value=True)
        controller.dependencies = controller.dependencies.__class__(**{
            **controller.dependencies.__dict__,
            "request_exploration_report_stop": request_stop,
        })
        execution = _CoordinationExecution(
            directive=ExplorationDirective(
                directive_id=43,
                kind=DirectiveKind.COMPONENT_TASK,
            ),
            phase="transit",
            path_start_index=0,
        )

        self.assertTrue(controller._advance_directive_transit(
            execution,
            (17, 16),
            source="component_task_transit",
        ))

        request_stop.assert_not_called()
        self.assertIsNone(controller._coordination_report)

    def test_focused_transit_stops_at_earliest_local_observation_pose(self):
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[:, :25] = FREE
        confidence[:, :25] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(
            occupancy,
            confidence,
            version=1,
        ))
        self.control.path_result = PathResult(
            ((16, 16), (18, 16), (20, 16), (24, 16)),
            PATH_COMPLETE,
            4,
            0.0,
        )
        trace = SimpleNamespace(record=Mock())
        controller = self.drone.movement_controller
        controller.dependencies = controller.dependencies.__class__(**{
            **controller.dependencies.__dict__,
            "runtime_trace": trace,
        })
        execution = _CoordinationExecution(
            directive=ExplorationDirective(
                directive_id=44,
                kind=DirectiveKind.COMPONENT_TASK,
            ),
            phase="transit",
            path_start_index=0,
            root_component_id=7,
        )
        node = LocalComponentNode(
            local_id=3,
            cells=frozenset({(24, 15), (24, 16), (24, 17)}),
            anchor_position=(24, 16),
            scan_heading=90,
            work_unit_id=9,
            allow_standoff=True,
        )

        reached = controller._advance_directive_transit(
            execution,
            node.anchor_position,
            source="component_task_transit",
            observation_node=node,
        )

        self.assertTrue(reached)
        self.assertEqual(self.drone.snapshot().position, (16, 16))
        self.assertEqual(execution.observation_position, (16, 16))
        self.assertEqual(execution.observation_heading, 90)
        selection_calls = [
            call for call in trace.record.call_args_list
            if call.args[0] == "drone_component_observation_pose_selected"
        ]
        self.assertEqual(len(selection_calls), 1)
        self.assertEqual(
            selection_calls[0].kwargs["frontier_position"],
            (24, 16),
        )
        self.assertEqual(
            selection_calls[0].kwargs["observation_position"],
            (16, 16),
        )

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

    def test_visible_stationary_leader_releases_unbranched_follower(self) -> None:
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
        controller = self.drone.movement_controller
        leader = [(1, (16, 16))]
        controller.dependencies = controller.dependencies.__class__(**{
            **controller.dependencies.__dict__,
            "get_drone_positions": lambda: tuple(leader),
        })
        with (
            patch.object(controller, "rebuild_frontiers"),
            patch.object(controller, "_reachable_local_nodes", return_value=()),
        ):
            controller._advance_component_follow(controller._coordination_execution)
            self.control.now = 10.0
            leader[0] = (1, (19, 16))
            controller._advance_component_follow(controller._coordination_execution)
            self.control.now = 24.0
            controller._advance_component_follow(controller._coordination_execution)
            self.assertIsNotNone(controller._coordination_execution)
            self.control.now = 25.1
            controller._advance_component_follow(controller._coordination_execution)

        self.assertIsNone(controller._coordination_execution)
        self.assertEqual(controller._coordination_report.kind, DirectiveKind.COMPONENT_FOLLOW)
        self.assertEqual(controller._coordination_report.causal_transitions, ())

    def test_waiting_drone_rejoins_rover_before_assignment(self) -> None:
        controller = self.drone.movement_controller
        ready = threading.Event()
        assignment = Mock()
        contact = Mock(return_value=False)
        endpoint = [(16, 16)]
        check_in = Mock(return_value=CoordinationResult(arrived=False))
        missed = Mock(side_effect=lambda _id, _point: (30, 30))
        controller.dependencies = controller.dependencies.__class__(**{
            **controller.dependencies.__dict__,
            "exploration_contact": contact,
            "exploration_assignment": assignment,
            "exploration_check_in": check_in,
            "get_check_in_position": lambda: endpoint[0],
            "rendezvous_endpoint_missed": missed,
        })
        controller._coordination_ready = ready
        controller._coordination_wait_started_at = 1.0

        controller._advance_component_check_in()

        assignment.assert_not_called()
        self.assertIsNone(controller._coordination_ready)
        missed.assert_called_once_with(self.drone.id, (16, 16))
        self.assertIsNone(controller._coordination_wait_started_at)

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

    def test_thin_successor_advances_under_same_claim(self) -> None:
        unit = ComponentWorkUnit(
            work_unit_id=13,
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
            task_id=7,
            component_id=3,
            component_revision=0,
            work_unit_ids=(13,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(16, 16),
            estimated_effort=1.0,
            state=TaskState.CLAIMED,
        )
        self._deliver(ExplorationDirective(
            directive_id=15,
            kind=DirectiveKind.COMPONENT_TASK,
            task=task,
            work_units=(unit,),
            claim=ClaimLease(7, (13,), 0, 44, 1),
        ))
        self.drone.move()
        self.drone.sensor_controller._last_completed_scan = (
            SensorScanCompletion(
                pose=(16, 16, 90.0),
                sequence=1,
                newly_known_cells=1,
                confidence_gain=1.0,
            )
        )
        controller = self.drone.movement_controller
        with patch(
            "agents.drone_movement.related_local_successors",
            return_value=(frozenset({(18, 16)}),),
        ), patch.object(controller, "_sweep_anchor_spacing", return_value=8.0):
            self.drone.move()

        execution = controller._coordination_execution
        self.assertEqual(execution.thin_successor_count, 1)
        self.assertEqual(execution.dfs.current.node.anchor_position, (18, 16))
        self.assertFalse(self.control.reports)
        with patch.object(
            controller, "_sweep_anchor_spacing", return_value=8.0,
        ), patch.object(
            controller,
            "_compute_path",
            side_effect=AssertionError("distant successor must not route"),
        ):
            self.assertIsNone(controller._thin_local_successor(
                execution, (frozenset({(40, 16)}),),
            ))

    def test_pending_siblings_use_current_distance_without_route_search(self) -> None:
        controller = self.drone.movement_controller
        mask = np.zeros((64, 64), dtype=bool)
        mask[16, 18] = True
        mask[20, 20] = True
        controller._last_frontier_mask = mask
        controller._frontier_slam_version = self.drone.slam_map.version
        dfs = LocalDFSStack()
        near = dfs.new_node(frozenset({(18, 16)}), (18, 16), 90)
        far = dfs.new_node(frozenset({(20, 20)}), (20, 20), 180)
        stale = dfs.new_node(frozenset({(50, 50)}), (50, 50), 90)
        execution = _CoordinationExecution(
            directive=ExplorationDirective(
                directive_id=16,
                kind=DirectiveKind.COMPONENT_TASK,
            ),
            phase="transit",
            path_start_index=0,
            dfs=dfs,
            work_unit_outcomes=[],
            local_routes={},
        )
        with patch.object(
            controller,
            "_compute_path",
            side_effect=AssertionError("sibling ranking must not run A*"),
        ), patch(
            "agents.drone_movement.related_local_successors",
            return_value=(),
        ):
            ranked = controller._rank_pending_component_nodes(
                execution, (far, stale, near),
            )

        self.assertEqual([node.local_id for node in ranked], [near.local_id, far.local_id])

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
        self.assertEqual(report.outbound_distance, 4.0)
        self.assertEqual(report.service_distance, 0.0)
        self.assertEqual(report.return_distance, 4.0)

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

    def test_shared_slam_retires_target_during_outbound_route(self) -> None:
        unit = ComponentWorkUnit(
            work_unit_id=52,
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
            task_id=11,
            component_id=9,
            component_revision=0,
            work_unit_ids=(52,),
            parent_task_id=None,
            depth=0,
            preferred_entry=(24, 16),
            estimated_effort=1.0,
            state=TaskState.CLAIMED,
        )
        self._deliver(ExplorationDirective(
            directive_id=13,
            kind=DirectiveKind.COMPONENT_TASK,
            task=task,
            work_units=(unit,),
            claim=ClaimLease(11, (52,), 0, 38, 1),
        ))
        self.control.path_result = PathResult(
            ((16, 16), (18, 16), (20, 16), (24, 16)),
            PATH_COMPLETE,
            4,
            0.0,
        )
        controller = self.drone.movement_controller
        shared = False

        def contact_checkpoint(_drone_id: int) -> None:
            nonlocal shared
            if shared or self.drone.snapshot().position != (18, 16):
                return
            occupancy = np.full((64, 64), FREE, dtype=np.int8)
            confidence = np.ones((64, 64), dtype=np.float32)
            self.drone.slam_map.merge_from(SlamSnapshot(
                occupancy,
                confidence,
            ))
            controller.mark_shared_slam_changed()
            shared = True

        controller.dependencies = controller.dependencies.__class__(
            **{
                **controller.dependencies.__dict__,
                "physical_contact_checkpoint": contact_checkpoint,
            }
        )

        self.drone.move()

        execution = controller._coordination_execution
        self.assertTrue(shared)
        self.assertEqual(self.drone.snapshot().position, (18, 16))
        self.assertIsNotNone(execution)
        self.assertEqual(execution.phase, "returning")
        self.assertEqual(
            tuple(
                (outcome.work_unit_id, outcome.disposition)
                for outcome in execution.work_unit_outcomes
            ),
            ((52, "shared_resolved"),),
        )

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

    def _nested_dfs_execution(self) -> _CoordinationExecution:
        controller = self.drone.movement_controller
        dfs = LocalDFSStack()
        root = dfs.new_node(frozenset({(20, 16)}), (20, 16), 90)
        branch = dfs.new_node(frozenset({(24, 16)}), (24, 16), 90)
        sibling = dfs.new_node(frozenset({(20, 20)}), (20, 20), 180)
        leaf = dfs.new_node(frozenset({(28, 16)}), (28, 16), 90)
        dfs.start(root, parent_position=(16, 16))
        dfs.record_outbound(((16, 16), (20, 16)))
        self.assertEqual(dfs.record_scan((branch, sibling)), branch)
        dfs.record_outbound(((20, 16), (24, 16)))
        self.assertEqual(dfs.record_scan((leaf,)), leaf)
        dfs.record_outbound(((24, 16), (28, 16)))
        dfs.record_scan(())
        for point in ((20, 16), (24, 16), (28, 16)):
            self.drone.runtime_state.move_to(point)
        execution = _CoordinationExecution(
            directive=ExplorationDirective(
                directive_id=14,
                kind=DirectiveKind.COMPONENT_TASK,
            ),
            phase="scanning",
            path_start_index=0,
            dfs=dfs,
            work_unit_outcomes=[],
            causal_successors=[],
            completed_scan_headings=[],
            timed_out_scan_headings=[],
            local_routes={},
            visited_successor_anchors=[],
            pending_authoritative_nodes=[],
        )
        controller._coordination_execution = execution
        # This fixture isolates logical DFS navigation; local-map sibling
        # revalidation has its own tests.
        with patch.object(
            controller,
            "_rank_pending_component_nodes",
            side_effect=lambda _execution, nodes: list(nodes),
        ):
            controller._pop_component_frame(execution)
        return execution

    def test_nested_dfs_unwinds_logically_and_astar_goes_to_sibling(self) -> None:
        controller = self.drone.movement_controller
        execution = self._nested_dfs_execution()
        self.assertEqual(self.drone.snapshot().position, (28, 16))
        self.assertEqual(execution.phase, "repositioning")
        self.assertEqual(execution.reposition_target, (20, 20))
        self.assertEqual(len(execution.dfs.frames), 2)
        activity = controller.activity_snapshot()
        self.assertEqual(activity.state, "DFS reposition")
        self.assertEqual(activity.target, (20, 20))
        self.assertEqual(activity.dfs_depth, 2)

        calls = []

        def follow(path, *, source="path", **_kwargs):
            calls.append((source, tuple(path)))
            self.drone.runtime_state.move_to(tuple(path)[-1])
            return True

        with patch.object(controller, "_follow_path", side_effect=follow):
            with patch.object(
                controller,
                "_compute_path",
                return_value=PathResult(
                    ((28, 16), (20, 20)),
                    PATH_COMPLETE,
                    1,
                    0.0,
                ),
            ) as compute:
                controller._advance_component_task(execution)

        compute.assert_called_once_with((28, 16), (20, 20))
        self.assertEqual(calls, [(
            "component_dfs_reposition_astar",
            ((28, 16), (20, 20)),
        )])
        self.assertEqual(self.drone.snapshot().position, (20, 20))
        self.assertEqual(execution.phase, "transit")
        self.assertIsNone(execution.reposition_target)
        controller._coordination_execution = None

    def test_dfs_reposition_uses_recorded_breadcrumb_if_astar_fails(self) -> None:
        controller = self.drone.movement_controller
        execution = self._nested_dfs_execution()
        calls = []

        def follow(path, *, source="path", **_kwargs):
            points = tuple(path)
            calls.append((source, points))
            for point in points:
                self.drone.runtime_state.move_to(point)
            return True

        with patch.object(controller, "_follow_path", side_effect=follow):
            with patch.object(
                controller,
                "_compute_path",
                side_effect=(
                    PathResult((), PATH_UNREACHABLE, 1, 0.0),
                    PathResult((), PATH_UNREACHABLE, 1, 0.0),
                    PathResult(
                        ((20, 16), (20, 20)),
                        PATH_COMPLETE,
                        1,
                        0.0,
                    ),
                ),
            ) as compute:
                controller._advance_component_task(execution)
                self.assertEqual(self.drone.snapshot().position, (28, 16))
                controller._advance_component_task(execution)
                self.assertEqual(self.drone.snapshot().position, (20, 16))
                self.assertEqual(execution.phase, "transit")
                controller._advance_component_task(execution)

        self.assertEqual(compute.call_count, 3)
        self.assertEqual(calls[0], (
            "component_dfs_breadcrumb_fallback",
            ((28, 16), (24, 16), (20, 16)),
        ))
        self.assertEqual(calls[1][0], "component_task_transit")
        self.assertEqual(self.drone.snapshot().position, (20, 20))
        self.assertIsNone(execution.suspension_reason)
        controller._coordination_execution = None

    def test_completed_dfs_returns_to_rover_before_report(self) -> None:
        controller = self.drone.movement_controller
        dfs = LocalDFSStack()
        root = dfs.new_node(frozenset({(20, 16)}), (20, 16), 90)
        dfs.start(root, parent_position=(16, 16))
        dfs.record_outbound(((16, 16), (20, 16)))
        dfs.record_scan(())
        self.drone.runtime_state.move_to((20, 16))
        execution = _CoordinationExecution(
            directive=ExplorationDirective(
                directive_id=15,
                kind=DirectiveKind.COMPONENT_TASK,
            ),
            phase="scanning",
            path_start_index=0,
            dfs=dfs,
            work_unit_outcomes=[],
            causal_successors=[],
            completed_scan_headings=[],
            timed_out_scan_headings=[],
            local_routes={},
            visited_successor_anchors=[],
            pending_authoritative_nodes=[],
        )
        controller._coordination_execution = execution
        controller._pop_component_frame(execution)

        self.assertEqual(execution.phase, "returning")
        self.assertIsNone(controller._coordination_report)
        with patch.object(
            controller,
            "_compute_path",
            return_value=PathResult(
                ((20, 16), (16, 16)),
                PATH_COMPLETE,
                1,
                0.0,
            ),
        ):
            with patch.object(controller, "_follow_path") as follow:
                def finish(
                    path,
                    *,
                    source="path",
                    stop_when=None,
                    stop_reason=None,
                ):
                    self.drone.runtime_state.move_to(tuple(path)[-1])
                    return True
                follow.side_effect = finish
                controller._advance_component_task(execution)

        self.assertEqual(self.drone.snapshot().position, (16, 16))
        self.assertIsNotNone(controller._coordination_report)
        self.assertIsNone(controller._coordination_report.suspension)
        controller._coordination_execution = None


if __name__ == "__main__":
    unittest.main()
