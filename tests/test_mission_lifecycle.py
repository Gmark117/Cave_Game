import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from game import Game
from agents.drone_runtime_state import DroneRuntimeState
from mission.control import MissionControl
from config.simulation_config import (
    ExplorationConfig,
    MissionConfig,
    SimulationConfig,
)
from mapping.terrain_knowledge import TerrainKnowledge
from mapping.exploration_sectors import (
    ExplorationSectorCoordinator,
    SectorAssignment,
    SectorCheckInResult,
    SectorOutcomeReport,
    SectorSuppressionOutcome,
)
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot
from mapping.wall_mapping import WallMappingSnapshot, exposed_wall_mask
from mission.exploration_coordination import (
    CoordinationResult,
    DirectiveKind,
    ExplorationDirective,
    ExplorationPhase,
)
from navigation.pathfinding import PathfindingService
from rendering.mission_renderer import MissionRenderer


class FakeGame:
    def __init__(self) -> None:
        self.sim_settings = SimulationConfig(
            mission_config=MissionConfig(
                seed=7,
                objective=0,
                num_drones=3,
                map_dim="SMALL",
            )
        )
        cave = np.zeros((8, 8), dtype=np.uint8)
        self.cartographer = SimpleNamespace(
            bin_map=cave,
            terrain_roughness=np.zeros_like(cave, dtype=np.float32),
            worm_x=[2],
            worm_y=[2],
        )
        self.width = 1200
        self.height = 750
        self.display = None
        self.window = None
        self.running = True
        self.maximise_calls = 0
        self.windowed_calls = 0

    def to_maximised(self):
        self.maximise_calls += 1
        return self.display

    def to_windowed(self):
        self.windowed_calls += 1
        return self.display


class MissionLifecycleTests(unittest.TestCase):
    @staticmethod
    def _runtime_drone(drone_id: int, position=(4, 4)):
        state = DroneRuntimeState(
            start_position=position,
            cave=np.zeros((16, 16), dtype=np.uint8),
            direction=0,
            frontier_rebuild_cooldown=0.0,
        )
        return SimpleNamespace(
            id=drone_id,
            runtime_state=state,
            snapshot=state.snapshot,
        )

    @staticmethod
    def _rendezvous_protocol(endpoint=(5, 4), drone_count=1):
        protocol = Mock()
        protocol.drone_endpoint.return_value = endpoint
        protocol.snapshot.return_value = SimpleNamespace(
            drone_announcements=tuple(
                SimpleNamespace(epoch=0) for _ in range(drone_count)
            ),
        )
        return protocol

    def test_assignment_requires_current_physical_rover_dock(self) -> None:
        mission = MissionControl(FakeGame())
        coordinator = Mock()
        coordinator.claim_directive.return_value = CoordinationResult(
            arrived=True, waiting=True,
        )
        coordinator.snapshot.return_value = SimpleNamespace(
            waiting_drone_ids=frozenset({0}),
        )
        mission.exploration_coordinator = coordinator
        mission.rovers = [SimpleNamespace(pos=(5, 4))]
        mission.terrain_sharing.drone_at_rover = Mock(return_value=False)
        processed = threading.Event()
        processed.set()
        directive_ready = threading.Event()
        directive_ready.set()
        mission._exploration_check_ins[0] = SimpleNamespace(
            ready=processed,
            report=None,
            result=CoordinationResult(
                arrived=True,
                report_accepted=True,
                waiting=True,
                directive_ready=directive_ready,
            ),
        )

        self.assertFalse(mission.exploration_assignment(0).arrived)
        coordinator.claim_directive.assert_not_called()
        self.assertIn(0, mission._exploration_check_ins)
        mission.terrain_sharing.drone_at_rover.return_value = True
        self.assertFalse(mission.exploration_assignment(0).arrived)
        coordinator.claim_directive.assert_not_called()
        mission._exploration_docked[0] = object()
        self.assertTrue(mission.exploration_assignment(0).arrived)
        coordinator.claim_directive.assert_called_once_with(0)
        self.assertNotIn(0, mission._exploration_check_ins)

    def test_recontact_requires_new_check_in_before_assignment(self) -> None:
        mission = MissionControl(FakeGame())
        coordinator = Mock()
        coordinator.snapshot.return_value = SimpleNamespace(
            waiting_drone_ids=frozenset(),
        )
        mission.exploration_coordinator = coordinator
        mission.terrain_sharing.drone_at_rover = Mock(return_value=True)
        pending_ready = threading.Event()
        pending_ready.set()
        mission._exploration_check_ins[0] = SimpleNamespace(ready=pending_ready)

        self.assertFalse(mission.exploration_assignment(0).arrived)
        self.assertNotIn(0, mission._exploration_check_ins)
        coordinator.claim_directive.assert_not_called()

    def test_exhausted_rover_holds_only_for_fully_docked_team(self) -> None:
        mission = MissionControl(FakeGame())
        coordinator = Mock(drone_count=3)
        coordinator.snapshot.return_value = SimpleNamespace(
            mission_exhausted=True,
        )
        mission.exploration_coordinator = coordinator
        mission.drones = [object(), object(), object()]
        mission._exploration_docked.update({0: object(), 1: object()})

        self.assertFalse(mission._rover_should_hold_position(0))
        mission._exploration_docked[2] = object()
        self.assertTrue(mission._rover_should_hold_position(0))
        mission._exploration_docked.pop(2)
        self.assertFalse(mission._rover_should_hold_position(0))

    def test_waiting_contact_pruned_before_quiescence_check(self) -> None:
        mission = MissionControl(FakeGame())
        coordinator = Mock()
        coordinator.snapshot.return_value = SimpleNamespace(
            waiting_drone_ids=frozenset({0, 1}),
        )
        mission.exploration_coordinator = coordinator
        mission.terrain_sharing.drone_at_rover = Mock(
            side_effect=lambda drone_id, _rover_id: drone_id == 1,
        )

        mission._prune_exploration_waiting_contacts()

        coordinator.waiting_contact_lost.assert_called_once_with(0)
        self.assertFalse(mission.exploration_contact(0))
        self.assertEqual(coordinator.waiting_contact_lost.call_count, 2)

    def test_physical_report_stop_holds_rover_until_check_in_is_queued(
        self,
    ) -> None:
        mission = MissionControl(FakeGame())
        mission.exploration_coordinator = object()
        mission.drones = [SimpleNamespace(
            snapshot=lambda: SimpleNamespace(position=(4, 4)),
            runtime_state=SimpleNamespace(
                carry_to=Mock(return_value=1.0),
                clear_ray_points=Mock(),
            ),
        )]
        mission.rovers = [SimpleNamespace(pos=(5, 4))]
        mission.terrain_sharing.drone_at_rover = Mock(return_value=True)
        mission.rendezvous_protocol = self._rendezvous_protocol()
        report = SimpleNamespace(report_id=12, directive_id=13)

        self.assertTrue(mission.request_exploration_report_stop(0))
        self.assertTrue(mission._rover_should_hold_position(0))
        mission.terrain_sharing.drone_at_rover.return_value = False
        missed = mission.exploration_check_in(0, report)
        self.assertFalse(missed.arrived)
        self.assertTrue(mission._rover_should_hold_position(0))
        mission.terrain_sharing.drone_at_rover.return_value = True
        result = mission.exploration_check_in(0, report)
        duplicate = mission.exploration_check_in(0, report)

        self.assertTrue(result.arrived)
        self.assertIs(result.directive_ready, duplicate.directive_ready)
        self.assertEqual(mission._exploration_check_in_queue.qsize(), 1)
        self.assertNotIn(0, mission._exploration_report_stops)
        self.assertIs(mission._exploration_check_ins[0].report, report)
        self.assertTrue(mission.is_exploration_docked(0))
        self.assertEqual(
            mission._exploration_docked[0].source,
            "report_intercept",
        )
        self.assertEqual(mission._exploration_docked[0].report_id, 12)
        self.assertFalse(mission._rover_should_hold_position(0))
        mission.rendezvous_protocol.drone_rover_contact.assert_called_once_with(0)

    def test_empty_physical_check_in_docks_and_snaps_to_rover(self) -> None:
        mission = MissionControl(FakeGame())
        mission.exploration_coordinator = object()
        drone = self._runtime_drone(0, (4, 4))
        drone.runtime_state.set_ray_points(((4, 2), (5, 2)))
        mission.drones = [drone]
        mission.rovers = [SimpleNamespace(pos=(5, 4))]
        mission.terrain_sharing.drone_at_rover = Mock(return_value=False)
        mission.rendezvous_protocol = self._rendezvous_protocol(
            endpoint=(10, 4),
        )

        missed = mission.request_exploration_dock(0)
        self.assertFalse(missed.arrived)
        self.assertFalse(mission.is_exploration_docked(0))
        self.assertEqual(drone.snapshot().position, (4, 4))

        mission.terrain_sharing.drone_at_rover.return_value = True
        result = mission.request_exploration_dock(0)

        self.assertTrue(result.arrived)
        self.assertTrue(mission.is_exploration_docked(0))
        self.assertEqual(drone.snapshot().position, (5, 4))
        session = mission._exploration_docked[0]
        self.assertEqual(session.source, "check_in_intercept")
        self.assertEqual(session.contact_position, (4, 4))
        self.assertEqual(session.learned_endpoint_epochs, {0})
        self.assertEqual(drone.snapshot().ray_points, ())
        self.assertEqual(mission.exploration_docked_ids(), frozenset({0}))

        drone.runtime_state.set_ray_points(((5, 2),))
        with mission._exploration_check_in_lock:
            mission._undock_exploration_drone_locked(
                0,
                reason="test_release",
            )
        self.assertEqual(drone.snapshot().ray_points, ())
        self.assertFalse(mission.is_exploration_docked(0))

    def test_empty_check_in_dock_is_atomic_with_rover_carriage(self) -> None:
        mission = MissionControl(FakeGame())
        mission.exploration_coordinator = object()
        drone = self._runtime_drone(0, (4, 4))
        mission.drones = [drone]
        rover = SimpleNamespace(pos=(5, 4))
        rover.move = lambda: setattr(rover, "pos", (6, 4))
        mission.rovers = [rover]
        mission.rendezvous_protocol = self._rendezvous_protocol()
        contact_started = threading.Event()
        release_contact = threading.Event()
        request_result = []

        def contact(_drone_id, _rover_id):
            contact_started.set()
            self.assertTrue(release_contact.wait(2.0))
            return True

        mission.terrain_sharing.drone_at_rover = contact

        def rover_step():
            with mission._exploration_check_in_lock:
                rover.move()
                mission._move_docked_drones_locked(0)

        requester = threading.Thread(target=lambda: request_result.append(
            mission.request_exploration_dock(0)
        ))
        mover = threading.Thread(target=rover_step)
        requester.start()
        self.assertTrue(contact_started.wait(2.0))
        mover.start()
        release_contact.set()
        requester.join(2.0)
        mover.join(2.0)

        self.assertFalse(requester.is_alive())
        self.assertFalse(mover.is_alive())
        self.assertTrue(request_result[0].arrived)
        self.assertEqual(rover.pos, (6, 4))
        self.assertEqual(drone.snapshot().position, (6, 4))
        session = mission._exploration_docked[0]
        self.assertEqual(session.carried_steps, 1)
        self.assertEqual(session.carried_distance, 1.0)
        self.assertEqual(session.carried_path, [(5, 4), (6, 4)])

    def test_rover_step_carries_multiple_docked_drones(self) -> None:
        mission = MissionControl(FakeGame())
        mission.exploration_coordinator = object()
        mission.drones = [
            self._runtime_drone(0, (4, 4)),
            self._runtime_drone(1, (5, 3)),
        ]
        mission.rovers = [SimpleNamespace(pos=(5, 4))]
        mission.terrain_sharing.drone_at_rover = Mock(return_value=True)
        mission.rendezvous_protocol = self._rendezvous_protocol(drone_count=2)
        mission.request_exploration_dock(0)
        mission.request_exploration_dock(1)

        with mission._exploration_check_in_lock:
            mission.rovers[0].pos = (7, 4)
            mission._move_docked_drones_locked(0)

        self.assertEqual(
            [drone.snapshot().position for drone in mission.drones],
            [(7, 4), (7, 4)],
        )

    def test_work_assignment_shares_then_releases_at_current_rover_pose(
        self,
    ) -> None:
        mission = MissionControl(FakeGame())
        coordinator = Mock()
        coordinator.snapshot.return_value = SimpleNamespace(
            waiting_drone_ids=frozenset({0}),
        )
        coordinator.claim_directive.return_value = CoordinationResult(
            arrived=True,
            directive=ExplorationDirective(
                directive_id=52,
                kind=DirectiveKind.RADIAL_PROBE,
                probe_target=(12, 4),
            ),
        )
        mission.exploration_coordinator = coordinator
        drone = self._runtime_drone(0, (4, 4))
        mission.drones = [drone]
        mission.rovers = [SimpleNamespace(pos=(5, 4))]
        mission.terrain_sharing.drone_at_rover = Mock(return_value=True)
        mission.rendezvous_protocol = self._rendezvous_protocol()
        mission.request_exploration_dock(0)
        mission._exploration_check_ins.clear()
        with mission._exploration_check_in_lock:
            mission.rovers[0].pos = (8, 4)
            mission._move_docked_drones_locked(0)
        order = []

        def share(_drone_id, _rover_id):
            order.append((
                "share",
                mission.is_exploration_docked(0),
                drone.snapshot().position,
            ))
            return True

        mission.terrain_sharing.share_on_departure = share
        mission.runtime_trace = SimpleNamespace(
            record=lambda event, **_fields: order.append((event,)),
        )

        result = mission.exploration_assignment(0)

        self.assertEqual(result.directive.directive_id, 52)
        self.assertEqual(order[0], ("share", True, (8, 4)))
        self.assertEqual(order[1], ("drone_undocked",))
        self.assertFalse(mission.is_exploration_docked(0))
        self.assertEqual(drone.snapshot().position, (8, 4))

    def test_home_assignment_keeps_drone_docked(self) -> None:
        mission = MissionControl(FakeGame())
        snapshot = SimpleNamespace(
            waiting_drone_ids=frozenset({0}),
            mission_exhausted=True,
            phase=ExplorationPhase.COMPLETE,
            tasks=(),
            claims=(),
        )
        coordinator = Mock()
        coordinator.snapshot.return_value = snapshot
        coordinator.claim_directive.return_value = CoordinationResult(
            arrived=True,
            directive=ExplorationDirective(
                directive_id=53,
                kind=DirectiveKind.HOME,
                reason="component_work_exhausted",
            ),
        )
        mission.exploration_coordinator = coordinator
        mission.drones = [self._runtime_drone(0)]
        mission.rovers = [SimpleNamespace(pos=(5, 4))]
        mission.terrain_sharing.drone_at_rover = Mock(return_value=True)
        mission.terrain_sharing.share_on_departure = Mock()
        mission.rendezvous_protocol = self._rendezvous_protocol()
        mission.request_exploration_dock(0)
        mission._exploration_check_ins.clear()

        result = mission.exploration_assignment(0)

        self.assertEqual(result.directive.kind, DirectiveKind.HOME)
        self.assertTrue(mission.is_exploration_docked(0))
        mission.terrain_sharing.share_on_departure.assert_not_called()

    def test_rover_scan_releases_in_place_without_departure_share(self) -> None:
        mission = MissionControl(FakeGame())
        coordinator = Mock()
        coordinator.snapshot.return_value = SimpleNamespace(
            waiting_drone_ids=frozenset({0}),
        )
        coordinator.claim_directive.return_value = CoordinationResult(
            arrived=True,
            directive=ExplorationDirective(
                directive_id=54,
                kind=DirectiveKind.ROVER_SCAN,
                scan_headings=(0, 90),
            ),
        )
        mission.exploration_coordinator = coordinator
        drone = self._runtime_drone(0)
        mission.drones = [drone]
        mission.rovers = [SimpleNamespace(pos=(5, 4))]
        mission.terrain_sharing.drone_at_rover = Mock(return_value=True)
        mission.terrain_sharing.share_on_departure = Mock()
        mission.rendezvous_protocol = self._rendezvous_protocol()
        mission.request_exploration_dock(0)
        mission._exploration_check_ins.clear()

        result = mission.exploration_assignment(0)

        self.assertEqual(result.directive.kind, DirectiveKind.ROVER_SCAN)
        self.assertFalse(mission.is_exploration_docked(0))
        self.assertEqual(drone.snapshot().position, (5, 4))
        mission.terrain_sharing.share_on_departure.assert_not_called()

    def test_report_stop_is_atomic_with_a_rover_step(self) -> None:
        mission = MissionControl(FakeGame())
        mission.exploration_coordinator = object()
        mission.drones = [SimpleNamespace(
            snapshot=lambda: SimpleNamespace(position=(4, 4)),
        )]
        rover = SimpleNamespace(pos=(5, 4))
        mission.rovers = [rover]
        mission.rendezvous_protocol = Mock()
        contact_started = threading.Event()
        release_contact = threading.Event()
        request_result = []

        def contact(_drone_id, _rover_id):
            contact_started.set()
            self.assertTrue(release_contact.wait(2.0))
            return True

        mission.terrain_sharing.drone_at_rover = contact

        def rover_step():
            with mission._exploration_check_in_lock:
                if not mission._rover_should_hold_position(0):
                    rover.pos = (6, 4)

        requester = threading.Thread(target=lambda: request_result.append(
            mission.request_exploration_report_stop(0)
        ))
        mover = threading.Thread(target=rover_step)
        requester.start()
        self.assertTrue(contact_started.wait(2.0))
        mover.start()
        release_contact.set()
        requester.join(2.0)
        mover.join(2.0)

        self.assertFalse(requester.is_alive())
        self.assertFalse(mover.is_alive())
        self.assertEqual(request_result, [True])
        self.assertEqual(rover.pos, (5, 4))

    def test_report_stop_requires_verified_physical_contact(self) -> None:
        mission = MissionControl(FakeGame())
        mission.exploration_coordinator = object()
        mission.drones = [SimpleNamespace(
            snapshot=lambda: SimpleNamespace(position=(4, 4)),
        )]
        mission.rovers = [SimpleNamespace(pos=(5, 4))]
        mission.terrain_sharing.drone_at_rover = Mock(return_value=False)
        mission.rendezvous_protocol = Mock()

        self.assertFalse(mission.request_exploration_report_stop(0))
        self.assertFalse(mission._rover_should_hold_position(0))
        mission.rendezvous_protocol.drone_rover_contact.assert_not_called()

    def test_epoch_with_standby_exchanges_only_for_actual_arrivals_and_departures(self) -> None:
        shape = (64, 96)
        coordinator = ExplorationSectorCoordinator(shape, 3, (16, 16))
        snapshot = Mock(return_value=SlamSnapshot(
            np.full(shape, UNKNOWN, dtype=np.int8), np.zeros(shape, dtype=np.float32),
        ))
        control = SimpleNamespace(
            rovers=[SimpleNamespace(slam_map=SimpleNamespace(snapshot=snapshot))],
            drones=[], exploration_completion_event=threading.Event(),
            exploration_sectors=coordinator, runtime_trace=SimpleNamespace(record=Mock()),
            simulation_time=lambda: 10.0,
            terrain_sharing=SimpleNamespace(
                check_in_with_rover=Mock(return_value=True),
                share_on_departure=Mock(return_value=True),
            ),
        )
        for drone_id in range(3):
            MissionControl.sector_check_in(control, drone_id, None)
        bootstrap = [MissionControl.sector_assignment(control, i).assignment for i in range(3)]
        occupancy = np.full(shape, OCCUPIED, dtype=np.int8)
        confidence = np.ones(shape, dtype=np.float32)
        occupancy[18:28, 18:28] = UNKNOWN
        confidence[18:28, 18:28] = 0.0
        occupancy[17, 17] = FREE
        snapshot.return_value = SlamSnapshot(occupancy, confidence, version=2)
        for assignment in bootstrap:
            MissionControl.sector_check_in(control, assignment.owner_drone_id, assignment.sector_id)
        assignments = [MissionControl.sector_assignment(control, i).assignment for i in range(3)]
        self.assertEqual([item.standby for item in assignments], [False, True, True])
        self.assertEqual(control.terrain_sharing.check_in_with_rover.call_count, 6)
        self.assertEqual(control.terrain_sharing.share_on_departure.call_count, 4)
        result = MissionControl.sector_check_in(control, 0, assignments[0].sector_id)
        self.assertTrue(result.mission_exhausted)
        for drone_id in range(3):
            self.assertTrue(MissionControl.sector_assignment(control, drone_id).mission_exhausted)
        self.assertEqual(control.terrain_sharing.check_in_with_rover.call_count, 7)
        self.assertEqual(control.terrain_sharing.share_on_departure.call_count, 4)

    def test_standby_delivery_skips_departure_exchange(self) -> None:
        assignment = SectorAssignment(
            sector_id=4, generation=1, owner_drone_id=1, cell_size=32,
            cells=frozenset(), seed=(16, 16), gateway=(16, 16),
            frontier_cells=0, rover_slam_version=1, standby=True,
        )
        result = SectorCheckInResult(arrived=True, assignment=assignment)
        control = SimpleNamespace(
            exploration_completion_event=threading.Event(),
            exploration_sectors=SimpleNamespace(claim_assignment=Mock(return_value=result)),
            terrain_sharing=SimpleNamespace(share_on_departure=Mock()),
        )
        self.assertIs(MissionControl.sector_assignment(control, 1), result)
        control.terrain_sharing.share_on_departure.assert_not_called()

    def test_arrival_forwards_local_sector_outcome_to_coordinator(self) -> None:
        rover_slam = SlamSnapshot(
            occupancy=np.full((8, 8), UNKNOWN, dtype=np.int8),
            confidence=np.zeros((8, 8), dtype=np.float32),
            version=4,
        )
        report = SectorOutcomeReport(
            sector_id=3,
            generation=1,
            suppressions=(SectorSuppressionOutcome(
                component_id=7,
                reasons=("zero_gain_directed_scan",),
                sampled_target_count=3,
            ),),
        )
        report_outcome = Mock(return_value=report)
        coordinator = Mock()
        coordinator.generation = 1
        coordinator.check_in.return_value = SectorCheckInResult(
            arrived=True,
            waiting_for_team=True,
            generation=1,
        )
        control = SimpleNamespace(
            rovers=[SimpleNamespace(
                slam_map=SimpleNamespace(
                    snapshot=Mock(return_value=rover_slam),
                ),
            )],
            drones=[SimpleNamespace(
                movement_controller=SimpleNamespace(
                    sector_outcome_report=report_outcome,
                ),
            )],
            terrain_sharing=SimpleNamespace(
                check_in_with_rover=Mock(return_value=True),
            ),
            exploration_completion_event=SimpleNamespace(
                is_set=lambda: False,
            ),
            exploration_sectors=coordinator,
            runtime_trace=SimpleNamespace(record=Mock()),
            simulation_time=lambda: 12.5,
        )

        result = MissionControl.sector_check_in(control, 0, 3)

        self.assertTrue(result.waiting_for_team)
        report_outcome.assert_called_once_with(3)
        coordinator.check_in.assert_called_once_with(
            0,
            3,
            rover_slam,
            report,
        )

    def test_sensor_update_publishes_discovered_floor_coverage(self) -> None:
        mission = MissionControl(FakeGame())
        cave = np.ones((8, 8), dtype=np.uint8)
        cave[1:7, 1:7] = 0
        mission.map_matrix = cave
        target = exposed_wall_mask(cave)
        occupancy = np.full(cave.shape, UNKNOWN, dtype=np.int8)
        confidence = np.zeros(cave.shape, dtype=np.float32)
        points = tuple(zip(*np.where(target)))
        for y, x in points[: len(points) // 2]:
            occupancy[y, x] = OCCUPIED
            confidence[y, x] = 1.0
        slam = SlamSnapshot(occupancy, confidence, version=3)
        drone = SimpleNamespace(
            update_sensors=Mock(),
            slam_map=SimpleNamespace(
                version=3,
                snapshot=Mock(return_value=slam),
            ),
        )
        mission.drones = [drone]
        mission.control_center = SimpleNamespace(
            set_explored_percent=Mock(),
        )
        mission.simulation_time = Mock(return_value=1.0)
        mission.terrain_knowledge.confidence[:] = 1.0

        mission.update_sensors()

        drone.update_sensors.assert_called_once_with()
        mission.control_center.set_explored_percent.assert_called_once_with(100)
        self.assertAlmostEqual(mission.wall_mapping_progress.ratio, 0.5)
        self.assertAlmostEqual(mission.floor_exploration_ratio, 1.0)

    def test_incomplete_floor_mapping_never_displays_one_hundred(self) -> None:
        mission = MissionControl(FakeGame())
        runtime_state = SimpleNamespace(start_returning_home=Mock())
        drone = SimpleNamespace(
            slam_map=SimpleNamespace(version=1),
            runtime_state=runtime_state,
        )
        mission.drones = [drone]
        mission.control_center = SimpleNamespace(
            set_explored_percent=Mock(),
        )
        mission.simulation_time = Mock(return_value=1.0)
        mission.terrain_knowledge.confidence[:] = 1.0
        mission.terrain_knowledge.confidence[0, 0] = 0.0
        incomplete = WallMappingSnapshot(
            9969,
            10000,
            0.9969,
            False,
            (1,),
        )

        with patch(
            "mission.control.wall_mapping_snapshot",
            return_value=incomplete,
        ):
            mission._update_wall_mapping_progress()

        mission.control_center.set_explored_percent.assert_called_once_with(98)
        runtime_state.start_returning_home.assert_not_called()

    def test_wall_tolerance_does_not_trigger_homing_or_completion(self) -> None:
        mission = MissionControl(FakeGame())
        states = [
            SimpleNamespace(start_returning_home=Mock())
            for _index in range(3)
        ]
        mission.drones = [
            SimpleNamespace(
                slam_map=SimpleNamespace(version=1),
                runtime_state=state,
            )
            for state in states
        ]
        mission.control_center = SimpleNamespace(
            set_explored_percent=Mock(),
        )
        mission.runtime_trace = Mock()
        mission.simulation_time = Mock(return_value=1.0)
        accepted = WallMappingSnapshot(
            2970,
            3000,
            0.99,
            False,
            (1, 1, 1),
        )

        with patch(
            "mission.control.wall_mapping_snapshot",
            return_value=accepted,
        ):
            mission._update_wall_mapping_progress()
            mission._update_wall_mapping_progress()

        for state in states:
            state.start_returning_home.assert_not_called()
        self.assertFalse(mission.exploration_completion_event.is_set())
        mission.runtime_trace.record.assert_not_called()
        mission.control_center.set_explored_percent.assert_called_once_with(0)

    def test_wall_tolerance_never_updates_component_completion(self) -> None:
        mission = MissionControl(FakeGame())
        runtime_state = SimpleNamespace(start_returning_home=Mock())
        mission.drones = [SimpleNamespace(
            slam_map=SimpleNamespace(version=1),
            runtime_state=runtime_state,
        )]
        mission.exploration_coordinator = SimpleNamespace(
            update_wall_goal=Mock(),
        )
        mission.control_center = SimpleNamespace(
            set_explored_percent=Mock(),
        )
        mission.runtime_trace = Mock()
        mission.simulation_time = Mock(return_value=1.0)
        accepted = WallMappingSnapshot(990, 1000, 0.99, False, (1,))

        with patch(
            "mission.control.wall_mapping_snapshot",
            return_value=accepted,
        ):
            mission._update_wall_mapping_progress()

        mission.exploration_coordinator.update_wall_goal.assert_not_called()
        runtime_state.start_returning_home.assert_not_called()
        self.assertFalse(mission.exploration_completion_event.is_set())

    def test_exact_wall_completion_is_diagnostic_only(self) -> None:
        mission = MissionControl(FakeGame())
        states = [
            SimpleNamespace(start_returning_home=Mock())
            for _index in range(3)
        ]
        mission.drones = [
            SimpleNamespace(
                slam_map=SimpleNamespace(version=1),
                runtime_state=state,
            )
            for state in states
        ]
        mission.control_center = SimpleNamespace(
            set_explored_percent=Mock(),
        )
        mission.simulation_time = Mock(return_value=1.0)
        mission.terrain_knowledge.confidence[:] = 1.0
        complete = WallMappingSnapshot(1000, 1000, 1.0, True, (1, 1, 1))

        with patch(
            "mission.control.wall_mapping_snapshot",
            return_value=complete,
        ):
            mission._update_wall_mapping_progress()
            mission._update_wall_mapping_progress()

        for state in states:
            state.start_returning_home.assert_not_called()
        mission.control_center.set_explored_percent.assert_called_once_with(100)

    def test_mission_trace_records_deterministic_exploration_policy(self) -> None:
        game = FakeGame()
        game.sim_settings = SimulationConfig(
            mission_config=game.sim_settings.mission_config,
            exploration=ExplorationConfig(policy="mcts"),
        )

        with patch("mission.control.RuntimeTraceLogger") as trace_logger:
            mission = MissionControl(game)

        constructed = next(
            call
            for call in trace_logger.return_value.record.call_args_list
            if call.args == ("mission_constructed",)
        )
        self.assertEqual(constructed.kwargs["exploration_policy"], "random")
        self.assertEqual(
            constructed.kwargs["frontier_policy"],
            "coordinated_scan_then_frontier_component_dfs",
        )
        self.assertEqual(constructed.kwargs["component_cost_cell_size"], 32)
        self.assertEqual(
            constructed.kwargs["component_check_in"],
            "queued_moving_rover_rendezvous",
        )
        self.assertEqual(
            constructed.kwargs["component_check_in_path"],
            "astar_with_breadcrumb_fallback",
        )
        self.assertEqual(
            constructed.kwargs["component_assignment_delivery"],
            "rover_signalled_no_polling",
        )
        self.assertEqual(
            constructed.kwargs["component_task_routing"],
            "rover_known_free_connectivity_then_drone_astar",
        )
        self.assertEqual(
            constructed.kwargs["component_zero_gain_memory"],
            "individual_anchor_or_subarc_plus_lineage_low_gain",
        )
        self.assertEqual(
            constructed.kwargs["component_workload_estimate"],
            "frontier_cells_plus_unknown_pixels_per_cell_size_squared",
        )
        self.assertEqual(
            constructed.kwargs["component_energy_policy"],
            "unlimited_via_energy_contract",
        )
        self.assertEqual(
            constructed.kwargs["frontier_unknown_basin_rescue"],
            "interior_basins_only",
        )
        self.assertTrue(constructed.kwargs["rover_periodic_sharing"])
        self.assertTrue(constructed.kwargs["rover_drone_pair_sharing"])
        self.assertTrue(
            mission.terrain_sharing.dependencies.periodic_rover_sharing_enabled
        )

        self.assertEqual(
            constructed.kwargs["exploration_completion"],
            "physical_team_quiescence_after_component_exhaustion",
        )
        self.assertEqual(
            constructed.kwargs["exploration_progress"],
            "discovered_floor_coverage_display_only",
        )
        self.assertEqual(
            constructed.kwargs["frontier_minimum_cluster_cells"],
            12,
        )
        self.assertEqual(
            constructed.kwargs[
                "frontier_minimum_unknown_support_cells"
            ],
            64,
        )
        self.assertEqual(constructed.kwargs["frontier_distance_band"], 16.0)
        self.assertEqual(
            constructed.kwargs["frontier_wall_continuation_weight"],
            2.0,
        )
        self.assertEqual(
            constructed.kwargs["frontier_cluster_size_weight"],
            2.0,
        )
        self.assertEqual(
            constructed.kwargs["frontier_cluster_proximity_weight"],
            1.0,
        )
        self.assertEqual(constructed.kwargs["frontier_global_cell_size"], 32)
        self.assertEqual(
            constructed.kwargs["frontier_global_refresh_interval"],
            2.0,
        )
        self.assertEqual(
            constructed.kwargs["frontier_global_ownership_weight"],
            2.0,
        )
        self.assertEqual(
            constructed.kwargs["frontier_maximum_path_circuity"],
            4.0,
        )
        self.assertEqual(constructed.kwargs["wall_direction_bias"], 4.0)
        self.assertEqual(
            constructed.kwargs["unexplored_direction_bias"],
            2.0,
        )
        self.assertEqual(
            constructed.kwargs["separation_direction_bias"],
            1.5,
        )
        self.assertNotIn("mcts_decision_time_budget_ms", constructed.kwargs)

    def test_completion_guard_refuses_a_signalled_sector_assignment(self) -> None:
        mission = MissionControl(FakeGame())
        mission.exploration_completion_event.set()
        mission.exploration_sectors = Mock()
        mission.exploration_sectors.generation = 3

        result = mission.sector_assignment(0)

        self.assertTrue(result.mission_exhausted)
        mission.exploration_sectors.stop.assert_called_once_with()
        mission.exploration_sectors.claim_assignment.assert_not_called()

    def test_construction_does_not_allocate_runtime_resources(self) -> None:
        game = FakeGame()

        with patch(
            "navigation.pathfinding.shared_memory.SharedMemory"
        ) as shared_memory:
            with patch(
                "navigation.pathfinding.ProcessPoolExecutor"
            ) as process_pool:
                mission = MissionControl(game)

        shared_memory.assert_not_called()
        process_pool.assert_not_called()
        self.assertEqual(game.maximise_calls, 0)
        self.assertEqual(mission.drones, [])
        self.assertEqual(mission.rovers, [])
        self.assertIsNone(mission.control_center)
        self.assertIsNone(mission.pathfinding.pool)
        self.assertIsNone(mission.pathfinding.map_shm)
        self.assertIsInstance(mission.terrain_knowledge, TerrainKnowledge)
        self.assertFalse(hasattr(mission, "last_pair_share"))
        self.assertFalse(hasattr(mission, "pair_share_cooldown"))
        self.assertEqual(mission.terrain_sharing.last_drone_share, {})
        self.assertEqual(mission.terrain_sharing.last_pair_share, {})
        self.assertFalse(hasattr(mission, "toggle_terrain_heatmap"))
        self.assertFalse(hasattr(mission, "toggle_drone_heatmap"))
        self.assertFalse(hasattr(mission, "_update_visibility_state"))
        self.assertFalse(hasattr(mission, "draw"))
        self.assertFalse(hasattr(mission, "map_shm"))
        self.assertFalse(hasattr(mission, "known_roughness"))
        self.assertIsInstance(mission.pathfinding, PathfindingService)
        self.assertIsInstance(mission.renderer, MissionRenderer)
        self.assertFalse(hasattr(mission, "waypoint_renderer"))
        self.assertFalse(hasattr(mission, "waypoint_graph"))
        self.assertFalse(hasattr(mission.renderer, "stop_button_rect"))
        self.assertFalse(mission.restart_requested)
        self.assertFalse(mission.exit_requested)
        self.assertFalse(mission.is_paused)
        self.assertTrue(mission.pause_event.is_set())
        self.assertTrue(mission.rover_motion_enabled)
        self.assertFalse(mission._runtime_initialized)
        self.assertFalse(mission._has_run)
        self.assertTrue(hasattr(mission, "compute_path"))
        self.assertFalse(mission.is_mission_over())

        mission.pathfinding.shutdown = Mock()
        mission._shutdown_mission([])
        mission.pathfinding.shutdown.assert_called_once_with()
        self.assertTrue(mission.mission_event.is_set())

    def test_drone_and_rover_pathfinding_delegate_to_owned_service(self) -> None:
        mission = MissionControl(FakeGame())
        mission.pathfinding.compute_path = Mock(
            return_value=[(0, 0), (1, 1)],
        )
        mission.pathfinding.compute_weighted_path = Mock(
            return_value=[(0, 0), (0, 1)],
        )
        rover_terrain = TerrainKnowledge(mission.map_matrix)
        mission.rovers = [SimpleNamespace(
            slam_map=SimpleNamespace(snapshot=Mock(return_value=SlamSnapshot(
                np.full((8, 8), FREE, dtype=np.int8),
                np.ones((8, 8), dtype=np.float32),
                version=1,
            ))),
            terrain_knowledge=rover_terrain,
        )]

        drone_path = mission.compute_path((0, 0), (1, 1))
        rover_path = mission.compute_rover_path(0, (0, 0), (0, 1))

        self.assertEqual(drone_path, [(0, 0), (1, 1)])
        self.assertEqual(rover_path, [(0, 0), (0, 1)])
        mission.pathfinding.compute_path.assert_called_once_with(
            (0, 0),
            (1, 1),
        )
        weighted_args = (
            mission.pathfinding.compute_weighted_path.call_args.args
        )
        np.testing.assert_array_equal(
            weighted_args[0], rover_terrain.roughness,
        )
        np.testing.assert_array_equal(
            weighted_args[1], rover_terrain.confidence,
        )
        self.assertEqual(weighted_args[2:], ((0, 0), (0, 1)))
        traversability = (
            mission.pathfinding.compute_weighted_path.call_args.kwargs[
                "traversability_map"
            ]
        )
        self.assertTrue(np.all(traversability == 0))

    def test_rover_path_rejects_corridor_narrower_than_its_body(self) -> None:
        mission = MissionControl(FakeGame())
        occupancy = np.full((8, 8), OCCUPIED, dtype=np.int8)
        confidence = np.ones((8, 8), dtype=np.float32)
        occupancy[4, 1:7] = FREE
        rover_terrain = TerrainKnowledge(mission.map_matrix)
        rover = SimpleNamespace(
            icon=pygame.Surface((4, 4), pygame.SRCALPHA),
            slam_map=SimpleNamespace(snapshot=Mock(return_value=SlamSnapshot(
                occupancy,
                confidence,
                version=1,
            ))),
            terrain_knowledge=rover_terrain,
        )
        mission.rovers = [rover]
        mission.pathfinding.compute_weighted_path = Mock()

        self.assertEqual(
            mission.compute_rover_path(0, (2, 4), (5, 4)),
            [],
        )
        mission.pathfinding.compute_weighted_path.assert_not_called()

    def test_rover_staging_excludes_diagonal_corner_connection(self) -> None:
        mission = MissionControl(FakeGame())
        occupancy = np.full((8, 8), OCCUPIED, dtype=np.int8)
        occupancy[3, 3] = FREE
        occupancy[4, 4] = FREE
        confidence = np.ones((8, 8), dtype=np.float32)
        mission.rovers = [SimpleNamespace(
            pos=(3, 3),
            icon=pygame.Surface((2, 2), pygame.SRCALPHA),
            slam_map=SimpleNamespace(snapshot=Mock(return_value=SlamSnapshot(
                occupancy, confidence, version=1,
            ))),
            terrain_knowledge=TerrainKnowledge(mission.map_matrix),
        )]

        safe, *_rest = mission._rover_staging_context()

        self.assertTrue(safe[3, 3])
        self.assertFalse(safe[4, 4])

    def test_focused_staging_rejects_move_that_strands_distribution(self) -> None:
        mission = MissionControl(FakeGame())
        shape = (24, 220)
        context = (
            np.ones(shape, dtype=bool),
            np.ones(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.ones(shape, dtype=np.float32),
            (100, 10),
            1.0,
        )

        selected = mission._select_rover_staging(
            (200, 10),
            context,
            outstanding_entries=((0, 10), (200, 10)),
        )

        self.assertIsNone(selected)

    def test_focused_staging_moves_toward_clustered_remaining_work(self) -> None:
        mission = MissionControl(FakeGame())
        shape = (24, 220)
        context = (
            np.ones(shape, dtype=bool),
            np.ones(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.ones(shape, dtype=np.float32),
            (100, 10),
            1.0,
        )

        selected = mission._select_rover_staging(
            (200, 10),
            context,
            outstanding_entries=((180, 10), (200, 10)),
        )

        self.assertIsNotNone(selected)
        self.assertLess(
            selected.remaining_max_distance,
            selected.current_max_distance,
        )

    def test_rover_path_negative_cache_uses_local_slam_version(self) -> None:
        mission = MissionControl(FakeGame())
        occupancy = np.full((8, 8), FREE, dtype=np.int8)
        confidence = np.ones((8, 8), dtype=np.float32)
        snapshots = [
            SlamSnapshot(occupancy, confidence, version=1),
            SlamSnapshot(occupancy, confidence, version=1),
            SlamSnapshot(occupancy, confidence, version=2),
        ]
        rover_terrain = TerrainKnowledge(mission.map_matrix)
        mission.rovers = [SimpleNamespace(
            icon=pygame.Surface((2, 2), pygame.SRCALPHA),
            slam_map=SimpleNamespace(
                snapshot=Mock(side_effect=snapshots),
            ),
            terrain_knowledge=rover_terrain,
        )]
        mission.pathfinding.compute_weighted_path = Mock(return_value=[])
        mission.runtime_trace = Mock()

        self.assertEqual(mission.compute_rover_path(0, (2, 2), (5, 5)), [])
        self.assertEqual(mission.compute_rover_path(0, (2, 2), (5, 5)), [])
        self.assertEqual(mission.compute_rover_path(0, (2, 2), (5, 5)), [])

        self.assertEqual(
            mission.pathfinding.compute_weighted_path.call_count,
            2,
        )
        event_names = tuple(
            call.args[0]
            for call in mission.runtime_trace.record.call_args_list
        )
        self.assertIn("rover_route_negative_cached", event_names)
        self.assertIn("rover_route_negative_cache_hit", event_names)

    def test_game_constructs_then_runs_mission(self) -> None:
        game = object.__new__(Game)
        settings = SimulationConfig(
            mission_config=MissionConfig(seed=11, objective=0)
        )
        game.menu = SimpleNamespace(
            build_sim_settings=Mock(return_value=settings),
        )
        cartographer = object()
        mission = Mock()

        with patch("game.MapGenerator", return_value=cartographer) as generator:
            with patch("game.MissionControl", return_value=mission) as control:
                game.start_mission()

        generator.assert_called_once_with(game, settings)
        control.assert_called_once_with(game)
        mission.run.assert_called_once_with()
        self.assertIs(game.cartographer, cartographer)
        self.assertIs(game.mission_control, mission)

    def test_game_restart_reuses_settings_and_generated_cave(self) -> None:
        game = object.__new__(Game)
        settings = SimulationConfig(
            mission_config=MissionConfig(seed=11, objective=0)
        )
        game.menu = SimpleNamespace(
            build_sim_settings=Mock(return_value=settings),
        )
        cartographer = object()
        first_mission = SimpleNamespace(
            run=Mock(),
            restart_requested=True,
        )
        second_mission = SimpleNamespace(
            run=Mock(),
            restart_requested=False,
        )

        with patch("game.MapGenerator", return_value=cartographer) as generator:
            with patch(
                "game.MissionControl",
                side_effect=[first_mission, second_mission],
            ) as control:
                game.start_mission()

        generator.assert_called_once_with(game, settings)
        self.assertEqual(control.call_count, 2)
        control.assert_any_call(game)
        first_mission.run.assert_called_once_with()
        second_mission.run.assert_called_once_with()
        self.assertIs(game.cartographer, cartographer)
        self.assertIs(game.mission_control, second_mission)

    def test_game_rejects_unimplemented_objective_before_generation(self) -> None:
        game = object.__new__(Game)
        settings = SimulationConfig(
            mission_config=MissionConfig(objective=1)
        )
        game.menu = SimpleNamespace(
            build_sim_settings=Mock(return_value=settings),
        )

        with patch("game.MapGenerator") as generator:
            with self.assertRaisesRegex(
                NotImplementedError,
                "Search and Rescue",
            ):
                game.start_mission()

        generator.assert_not_called()

    def test_run_initializes_executes_and_shuts_down_once(self) -> None:
        mission = MissionControl(FakeGame())
        control_center = SimpleNamespace(start_timer=Mock())

        def initialize_runtime() -> None:
            mission.control_center = control_center
            mission._runtime_initialized = True

        mission._initialize_runtime = Mock(side_effect=initialize_runtime)
        mission._start_agent_threads = Mock(return_value=[])
        mission._run_mission_loop = Mock()
        mission._shutdown_mission = Mock()

        with patch("mission.lifecycle.pygame.get_init", return_value=False):
            mission.run()

        mission._initialize_runtime.assert_called_once_with()
        control_center.start_timer.assert_called_once_with()
        mission._start_agent_threads.assert_called_once_with()
        mission._run_mission_loop.assert_called_once_with()
        mission._shutdown_mission.assert_called_once_with([])
        self.assertFalse(mission._running)
        self.assertTrue(mission._has_run)

        with self.assertRaises(RuntimeError):
            mission.run()

    def test_stop_button_ends_loop_before_simulation_updates(self) -> None:
        mission = MissionControl(FakeGame())
        mission.clock = SimpleNamespace(tick=Mock())
        mission.completed = False
        mission.renderer.draw = Mock()
        mission.update_sensors = Mock()
        mission.terrain_sharing.share_with_rovers = Mock()
        mission.control_center = SimpleNamespace(
            handle_click=Mock(return_value=("mission_stop", None)),
        )
        event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=(1, 1),
        )

        with patch(
            "mission.lifecycle.pygame.event.get",
            return_value=[event],
        ):
            mission._run_mission_loop()

        self.assertTrue(mission.completed)
        mission.terrain_sharing.share_with_rovers.assert_not_called()
        mission.update_sensors.assert_not_called()
        mission.renderer.draw.assert_not_called()

    def test_mute_button_toggles_music_without_stopping_simulation(self) -> None:
        game = FakeGame()
        game.menu = SimpleNamespace(
            music_enabled=Mock(return_value=True),
            toggle_music=Mock(),
        )
        mission = MissionControl(game)
        mission.clock = SimpleNamespace(tick=Mock())
        mission.completed = False
        mission.renderer.draw = Mock()
        mission.update_sensors = Mock()
        mission.terrain_sharing.share_with_rovers = Mock()
        mission.is_mission_over = Mock(return_value=False)
        mission.control_center = SimpleNamespace(
            handle_click=Mock(
                side_effect=[
                    ("mission_music", None),
                    ("mission_stop", None),
                ],
            ),
        )
        mute_event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=(1, 1),
        )
        stop_event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=(1, 1),
        )

        with patch(
            "mission.lifecycle.pygame.event.get",
            side_effect=[[mute_event], [stop_event]],
        ):
            with patch("mission.lifecycle.pygame.display.update"):
                mission._run_mission_loop()

        game.menu.toggle_music.assert_called_once_with()
        # Periodic rover sharing runs on the primary rover worker so the UI
        # loop cannot stall on SLAM merge/reconciliation work.
        mission.terrain_sharing.share_with_rovers.assert_not_called()
        mission.is_mission_over.assert_called_once_with()
        mission.update_sensors.assert_called_once_with()
        mission.renderer.draw.assert_called_once_with()

    def test_restart_button_requests_fresh_mission_before_updates(self) -> None:
        mission = MissionControl(FakeGame())
        mission.clock = SimpleNamespace(tick=Mock())
        mission.completed = False
        mission.renderer.draw = Mock()
        mission.update_sensors = Mock()
        mission.terrain_sharing.share_with_rovers = Mock()
        mission.control_center = SimpleNamespace(
            handle_click=Mock(return_value=("mission_restart", None)),
        )
        event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=(1, 1),
        )

        with patch(
            "mission.lifecycle.pygame.event.get",
            return_value=[event],
        ):
            mission._run_mission_loop()

        self.assertTrue(mission.completed)
        self.assertTrue(mission.restart_requested)
        mission.terrain_sharing.share_with_rovers.assert_not_called()
        mission.update_sensors.assert_not_called()
        mission.renderer.draw.assert_not_called()

    def test_exit_button_closes_program_before_updates(self) -> None:
        game = FakeGame()
        mission = MissionControl(game)
        mission.clock = SimpleNamespace(tick=Mock())
        mission.completed = False
        mission.renderer.draw = Mock()
        mission.update_sensors = Mock()
        mission.terrain_sharing.share_with_rovers = Mock()
        mission.control_center = SimpleNamespace(
            handle_click=Mock(return_value=("mission_exit", None)),
        )
        event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=(1, 1),
        )

        with patch(
            "mission.lifecycle.pygame.event.get",
            return_value=[event],
        ):
            mission._run_mission_loop()

        self.assertTrue(mission.completed)
        self.assertTrue(mission.exit_requested)
        self.assertFalse(game.running)
        mission.terrain_sharing.share_with_rovers.assert_not_called()
        mission.update_sensors.assert_not_called()
        mission.renderer.draw.assert_not_called()

    def test_pause_button_toggles_state_and_skips_simulation_updates(self) -> None:
        mission = MissionControl(FakeGame())
        mission.clock = SimpleNamespace(tick=Mock())
        mission.completed = False
        mission.control_center = SimpleNamespace(
            handle_click=Mock(
                side_effect=[
                    ("mission_pause", None),
                    ("mission_stop", None),
                ],
            ),
            pause_timer=Mock(),
            resume_timer=Mock(),
        )
        mission.renderer.draw = Mock()
        mission.update_sensors = Mock()
        mission.terrain_sharing.share_with_rovers = Mock()
        mission.is_mission_over = Mock()
        pause_event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=(1, 1),
        )
        stop_event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=(1, 1),
        )

        with patch(
            "mission.lifecycle.pygame.event.get",
            side_effect=[[pause_event], [stop_event]],
        ):
            with patch("mission.lifecycle.pygame.display.update"):
                mission._run_mission_loop()

        self.assertTrue(mission.is_paused)
        self.assertFalse(mission.pause_event.is_set())
        mission.control_center.pause_timer.assert_called_once_with()
        mission.control_center.resume_timer.assert_not_called()
        mission.terrain_sharing.share_with_rovers.assert_not_called()
        mission.is_mission_over.assert_not_called()
        mission.update_sensors.assert_not_called()
        mission.renderer.draw.assert_called_once_with()

    def test_pause_toggle_resumes_agents_and_timer(self) -> None:
        mission = MissionControl(FakeGame())
        mission.control_center = SimpleNamespace(
            pause_timer=Mock(),
            resume_timer=Mock(),
        )

        mission.toggle_pause()
        mission.toggle_pause()

        self.assertFalse(mission.is_paused)
        self.assertTrue(mission.pause_event.is_set())
        mission.control_center.pause_timer.assert_called_once_with()
        mission.control_center.resume_timer.assert_called_once_with()

    def test_pause_waits_for_inflight_move_and_blocks_followup_sharing(self) -> None:
        mission = MissionControl(FakeGame())
        mission.delay = 0.01
        move_started = threading.Event()
        release_move = threading.Event()
        pause_finished = threading.Event()

        def move() -> None:
            move_started.set()
            release_move.wait(2.0)

        drone = SimpleNamespace(
            mission_completed=Mock(return_value=False),
            move=move,
        )
        mission.drones = [drone]
        mission.terrain_sharing.share_with_nearby_drones = Mock()
        worker = threading.Thread(target=mission.drone_thread, args=(0,))
        pauser = threading.Thread(
            target=lambda: (
                mission.toggle_pause(),
                pause_finished.set(),
            )
        )

        worker.start()
        self.assertTrue(move_started.wait(2.0))
        pauser.start()
        self.assertFalse(pause_finished.wait(0.05))

        release_move.set()
        self.assertTrue(pause_finished.wait(2.0))

        self.assertTrue(mission.is_paused)
        mission.terrain_sharing.share_with_nearby_drones.assert_not_called()

        mission.mission_event.set()
        mission.pause_coordinator.stop()
        worker.join(2.0)
        pauser.join(2.0)
        self.assertFalse(worker.is_alive())
        self.assertFalse(pauser.is_alive())

    def test_restart_run_does_not_return_to_windowed_mode(self) -> None:
        mission = MissionControl(FakeGame())
        control_center = SimpleNamespace(start_timer=Mock())

        def initialize_runtime() -> None:
            mission.control_center = control_center
            mission._runtime_initialized = True

        def request_restart() -> None:
            mission.restart_requested = True

        mission._initialize_runtime = Mock(side_effect=initialize_runtime)
        mission._start_agent_threads = Mock(return_value=[])
        mission._run_mission_loop = Mock(side_effect=request_restart)
        mission._shutdown_mission = Mock()

        with patch("mission.lifecycle.pygame.get_init", return_value=True):
            mission.run()

        self.assertEqual(mission.game.windowed_calls, 0)

    def test_exit_run_does_not_return_to_windowed_mode(self) -> None:
        mission = MissionControl(FakeGame())
        control_center = SimpleNamespace(start_timer=Mock())

        def initialize_runtime() -> None:
            mission.control_center = control_center
            mission._runtime_initialized = True

        def request_exit() -> None:
            mission.exit_requested = True

        mission._initialize_runtime = Mock(side_effect=initialize_runtime)
        mission._start_agent_threads = Mock(return_value=[])
        mission._run_mission_loop = Mock(side_effect=request_exit)
        mission._shutdown_mission = Mock()

        with patch("mission.lifecycle.pygame.get_init", return_value=True):
            mission.run()

        self.assertEqual(mission.game.windowed_calls, 0)

    def test_run_loop_records_frame_stage_timings(self) -> None:
        mission = MissionControl(FakeGame())
        mission.clock = SimpleNamespace(tick=Mock())
        mission.completed = False
        mission.renderer.draw = Mock()
        mission.update_sensors = Mock()
        mission.terrain_sharing.share_with_rovers = Mock()
        mission.is_mission_over = Mock(return_value=True)
        mission.control_center = SimpleNamespace(
            handle_click=Mock(return_value=("mission_stop", None)),
            pause_timer=Mock(),
        )
        stop_event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=(1, 1),
        )

        timestamps = [
            0.000,
            0.010,
            0.012,
            0.020,
            0.021,
            0.030,
            0.050,
            0.052,
            0.060,
            0.070,
            0.080,
        ]
        with patch(
            "mission.lifecycle.pygame.event.get",
            side_effect=[[], [stop_event]],
        ):
            with patch("mission.lifecycle.pygame.display.update"):
                with patch(
                    "mission.lifecycle.time.perf_counter",
                    side_effect=timestamps,
                ):
                    mission._run_mission_loop()

        timing = mission.frame_profiler.snapshot()
        self.assertEqual(timing.sample_count, 1)
        self.assertAlmostEqual(timing.frame_ms, 52.0)
        self.assertAlmostEqual(timing.wait_ms, 10.0)
        self.assertAlmostEqual(timing.stages_ms["events"], 2.0)
        self.assertAlmostEqual(timing.stages_ms["sharing"], 8.0)
        self.assertAlmostEqual(timing.stages_ms["sensors"], 9.0)
        self.assertAlmostEqual(timing.stages_ms["render"], 20.0)
        self.assertTrue(mission.exploration_complete)
        self.assertTrue(mission.is_paused)
        mission.control_center.pause_timer.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
