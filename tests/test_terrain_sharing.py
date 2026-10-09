import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from config.simulation_config import SharingConfig, SimulationConfig
from mapping.slam_map import UNKNOWN, SlamMap, SlamSnapshot
from agents.drone_runtime_state import DroneRuntimeState
from mapping.terrain_knowledge import TerrainKnowledge
from mapping.terrain_sharing import TerrainSharingService
from contracts import TerrainSharingDependencies


class RecordingTrace:
    def __init__(self) -> None:
        self.events = []

    def record(self, event, **fields) -> None:
        self.events.append((event, fields))


def make_agent(
    agent_id: int,
    position: tuple[int, int],
    shape: tuple[int, int] = (4, 4),
):
    cave = np.zeros(shape, dtype=np.uint8)
    terrain_knowledge = TerrainKnowledge(cave)
    runtime_state = DroneRuntimeState(
        start_position=position,
        cave=cave,
        direction=0,
        frontier_rebuild_cooldown=0.25,
    )
    return SimpleNamespace(
        id=agent_id,
        pos=position,
        radius=4,
        terrain_knowledge=terrain_knowledge,
        runtime_state=runtime_state,
        snapshot=runtime_state.snapshot,
        slam_map=SlamMap(*shape),
        merge_frontiers=runtime_state.merge_frontiers,
        movement_controller=SimpleNamespace(
            mark_shared_slam_changed=Mock(),
            begin_peer_sharing=Mock(),
            begin_rover_sharing=Mock(),
        ),
    )


def make_control():
    cave = np.zeros((4, 4), dtype=np.uint8)
    control = SimpleNamespace(
        settings=SimulationConfig(
            sharing=SharingConfig(
                drone_interval=0.0,
                pair_cooldown=0.0,
                rover_interval=0.5,
                compare_stride=1,
                min_new_info_ratio=0.1,
                min_overlap_diff_ratio=0.25,
                min_roughness_delta=0.1,
            )
        ),
        map_matrix=cave,
        map_h=4,
        map_w=4,
        terrain_knowledge=TerrainKnowledge(cave),
        presentation=SimpleNamespace(terrain_heatmap_dirty=False),
        drones=[],
        rovers=[],
        simulation_time=Mock(return_value=10.0),
    )
    control.dependencies = TerrainSharingDependencies(
        sharing=control.settings.sharing,
        cave_map=control.map_matrix,
        map_width=control.map_w,
        map_height=control.map_h,
        terrain_knowledge=control.terrain_knowledge,
        get_drones=lambda: control.drones,
        get_rovers=lambda: control.rovers,
        presentation=control.presentation,
        simulation_time=control.simulation_time,
    )
    return control


class TerrainSharingTests(unittest.TestCase):
    @staticmethod
    def seed_slam(agent, x: int, y: int, occupancy_value: int, confidence: float):
        shape = agent.slam_map.shape
        occupancy = np.full(shape, UNKNOWN, dtype=np.int8)
        confidence_map = np.zeros(shape, dtype=np.float32)
        occupancy[y, x] = occupancy_value
        confidence_map[y, x] = confidence
        agent.slam_map.merge_from(
            SlamSnapshot(
                occupancy,
                confidence_map,
                point_cloud=((x, y),),
            )
        )

    def test_line_of_sight_rejects_walls_and_out_of_bounds(self) -> None:
        control = make_control()
        service = TerrainSharingService(control.dependencies)
        control.map_matrix[1, 1] = 1

        self.assertFalse(service.has_line_of_sight((0, 0), (2, 2)))
        self.assertFalse(service.has_line_of_sight((0, 0), (8, 8)))
        self.assertTrue(service.has_line_of_sight((2, 0), (3, 0)))

    def test_nearby_drone_receives_slam_and_invalidates_derived_borders(
        self,
    ) -> None:
        control = make_control()
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        source.terrain_knowledge.roughness[1, 1] = 0.8
        source.terrain_knowledge.confidence[1, 1] = 1.0
        source.runtime_state.replace_frontiers(((3, 3),))
        self.seed_slam(source, 2, 2, 1, 0.9)
        control.drones = [source, target]
        service = TerrainSharingService(control.dependencies)

        service.share_with_nearby_drones(0)

        self.assertNotIn((3, 3), target.snapshot().frontiers)
        target.movement_controller.mark_shared_slam_changed.assert_called_once()
        source.movement_controller.mark_shared_slam_changed.assert_not_called()
        self.assertAlmostEqual(
            float(target.terrain_knowledge.roughness[1, 1]),
            0.8,
        )
        self.assertAlmostEqual(
            float(target.terrain_knowledge.confidence[1, 1]),
            1.0,
        )
        target_slam = target.slam_map.snapshot()
        self.assertEqual(int(target_slam.occupancy[2, 2]), 1)
        self.assertAlmostEqual(float(target_slam.confidence[2, 2]), 0.9)
        self.assertEqual(service.last_drone_share[0], 10.0)
        self.assertEqual(service.last_pair_share[(0, 1)], 10.0)
        self.assertTrue(control.presentation.terrain_heatmap_dirty)
        source.movement_controller.begin_peer_sharing.assert_called_once_with(
            1,
            (2, 1),
        )
        target.movement_controller.begin_peer_sharing.assert_called_once_with(
            0,
            (1, 1),
        )

    def test_motion_checkpoint_exchanges_once_on_contact_entry(self) -> None:
        control = make_control()
        contact = Mock()
        object.__setattr__(control.dependencies, "on_drone_contact", contact)
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        self.seed_slam(source, 1, 1, 1, 0.9)
        control.drones = [source, target]
        service = TerrainSharingService(control.dependencies)
        service.drone_share_interval = 30.0
        service.last_drone_share[0] = 10.0

        service.physical_contact_checkpoint(0)
        service.physical_contact_checkpoint(0)

        shared = target.slam_map.snapshot()
        self.assertEqual(int(shared.occupancy[1, 1]), 1)
        self.assertAlmostEqual(float(shared.confidence[1, 1]), 0.9)
        contact.assert_called_once_with(0, 1)
        self.assertEqual(service.last_drone_share[0], 10.0)

    def test_continuous_peer_contact_delivers_new_data_with_only_one_pause(self):
        control = make_control()
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        control.drones = [source, target]
        service = TerrainSharingService(control.dependencies)
        contact = Mock()
        object.__setattr__(control.dependencies, "on_drone_contact", contact)

        for x in range(3):
            self.seed_slam(source, x, 0, 1, 0.9)
            service.share_with_nearby_drones(0)
            self.assertEqual(int(target.slam_map.snapshot().occupancy[0, x]), 1)
        source.movement_controller.begin_peer_sharing.assert_called_once()
        target.movement_controller.begin_peer_sharing.assert_called_once()
        self.assertEqual(contact.call_count, 3)

        source.runtime_state.move_to((12, 1))
        service.physical_contact_checkpoint(0)
        source.runtime_state.move_to((1, 1))
        self.seed_slam(source, 3, 0, 1, 0.9)
        service.physical_contact_checkpoint(0)
        self.assertEqual(source.movement_controller.begin_peer_sharing.call_count, 2)
        self.assertEqual(target.movement_controller.begin_peer_sharing.call_count, 2)

    def test_peer_protocol_contact_requires_proximity_and_line_of_sight(self) -> None:
        control = make_control()
        contact = Mock()
        object.__setattr__(control.dependencies, "on_drone_contact", contact)
        first = make_agent(0, (0, 0))
        second = make_agent(1, (2, 0))
        control.drones = [first, second]
        service = TerrainSharingService(control.dependencies)

        service.share_with_nearby_drones(0)
        contact.assert_called_once_with(0, 1)

        contact.reset_mock()
        second.runtime_state.move_to((3, 3))
        control.map_matrix[1, 1] = 1
        service.share_with_nearby_drones(0)
        contact.assert_not_called()

    def test_visible_peer_positions_use_the_contact_envelope(self) -> None:
        control = make_control()
        first = make_agent(0, (0, 0))
        visible = make_agent(1, (2, 0))
        hidden = make_agent(2, (3, 3))
        control.map_matrix[1, 1] = 1
        control.drones = [first, visible, hidden]

        positions = TerrainSharingService(
            control.dependencies
        ).visible_drone_positions(0)

        self.assertEqual(positions, ((0, (0, 0)), (1, (2, 0))))

    def test_sparse_unsampled_slam_delta_is_still_shared(self) -> None:
        control = make_control()
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        self.seed_slam(source, 1, 1, 1, 0.9)
        control.drones = [source, target]
        service = TerrainSharingService(control.dependencies)
        service.compare_stride = 8
        service.min_new_info_ratio = 1.0

        service.share_with_nearby_drones(0)

        shared = target.slam_map.snapshot()
        self.assertEqual(int(shared.occupancy[1, 1]), 1)
        self.assertAlmostEqual(float(shared.confidence[1, 1]), 0.9)

    def test_nearby_pair_trace_reports_exchange_outcome(self) -> None:
        control = make_control()
        trace = RecordingTrace()
        object.__setattr__(control.dependencies, "runtime_trace", trace)
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        self.seed_slam(source, 1, 1, 1, 0.9)
        control.drones = [source, target]

        TerrainSharingService(control.dependencies).share_with_nearby_drones(0)

        event, fields = next(
            item for item in trace.events
            if item[0] == "drone_sharing_pair"
        )
        self.assertEqual(event, "drone_sharing_pair")
        self.assertEqual(fields["drone_id"], 0)
        self.assertEqual(fields["other_drone_id"], 1)
        self.assertTrue(fields["shared"])
        self.assertEqual(fields["reason"], "exchanged")
        self.assertAlmostEqual(fields["distance"], 1.0)

    def test_pair_cooldown_prevents_duplicate_exchange(self) -> None:
        control = make_control()
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        source.terrain_knowledge.roughness[1, 1] = 0.8
        source.terrain_knowledge.confidence[1, 1] = 1.0
        control.drones = [source, target]
        service = TerrainSharingService(control.dependencies)
        service.pair_share_cooldown = 5.0
        service.last_pair_share[(0, 1)] = 8.0

        service.share_with_nearby_drones(0)

        self.assertEqual(
            float(target.terrain_knowledge.confidence[1, 1]),
            0.0,
        )

    def test_pair_revision_gate_allows_new_information_after_no_delta(self) -> None:
        control = make_control()
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        control.drones = [source, target]
        service = TerrainSharingService(control.dependencies)
        service.pair_share_cooldown = 5.0

        service.share_with_nearby_drones(0)
        source.terrain_knowledge.record_samples(((1, 1, 0.8, 1.0),))
        service.share_with_nearby_drones(0)

        self.assertEqual(service.last_pair_share[(0, 1)], 10.0)
        self.assertAlmostEqual(
            float(target.terrain_knowledge.roughness[1, 1]),
            0.8,
        )
        self.assertAlmostEqual(
            float(target.terrain_knowledge.confidence[1, 1]),
            1.0,
        )

    def test_unchanged_pair_versions_skip_full_map_snapshots(self) -> None:
        control = make_control()
        trace = RecordingTrace()
        object.__setattr__(control.dependencies, "runtime_trace", trace)
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        control.drones = [source, target]
        service = TerrainSharingService(control.dependencies)

        service.share_with_nearby_drones(0)
        source_snapshot = Mock(wraps=source.terrain_knowledge.snapshot)
        target_snapshot = Mock(wraps=target.terrain_knowledge.snapshot)
        source.terrain_knowledge.snapshot = source_snapshot
        target.terrain_knowledge.snapshot = target_snapshot
        service.share_with_nearby_drones(0)

        source_snapshot.assert_not_called()
        target_snapshot.assert_not_called()
        pair_events = [
            fields for event, fields in trace.events
            if event == "drone_sharing_pair"
        ]
        self.assertEqual(pair_events[-1]["reason"], "unchanged_versions")

    def test_service_owns_all_sharing_schedule_state(self) -> None:
        control = make_control()
        service = TerrainSharingService(control.dependencies)

        self.assertFalse(hasattr(control, "last_pair_share"))
        self.assertFalse(hasattr(control, "pair_share_cooldown"))
        self.assertEqual(service.last_drone_share, {})
        self.assertEqual(service.last_pair_share, {})
        self.assertIsNone(service.last_rover_share_time)

    def test_concurrent_workers_process_a_pair_only_once(self) -> None:
        control = make_control()
        control.drones = [
            make_agent(0, (1, 1)),
            make_agent(1, (2, 1)),
        ]
        service = TerrainSharingService(control.dependencies)
        exchange_started = threading.Event()
        release_exchange = threading.Event()
        exchange_calls = []
        worker_errors = []

        def blocking_exchange(
            drone,
            other_drone,
            drone_snapshot,
            other_snapshot,
        ):
            exchange_calls.append((drone.id, other_drone.id))
            exchange_started.set()
            release_exchange.wait(2.0)
            return True

        service._exchange_drone_data = blocking_exchange

        def share(drone_id: int) -> None:
            try:
                service.share_with_nearby_drones(drone_id)
            except BaseException as exc:
                worker_errors.append(exc)

        first = threading.Thread(target=share, args=(0,))
        first.start()
        self.assertTrue(exchange_started.wait(2.0))

        second = threading.Thread(target=share, args=(1,))
        second.start()
        second.join(2.0)

        self.assertFalse(second.is_alive())
        self.assertEqual(exchange_calls, [(0, 1)])

        release_exchange.set()
        first.join(2.0)

        self.assertFalse(first.is_alive())
        self.assertEqual(worker_errors, [])
        self.assertIn((0, 1), service.last_pair_share)

    def test_rover_receives_terrain_only_when_close_enough(self) -> None:
        control = make_control()
        drone = make_agent(0, (1, 1))
        drone.terrain_knowledge.roughness[1, 1] = 0.6
        drone.terrain_knowledge.confidence[1, 1] = 0.5
        rover_knowledge = TerrainKnowledge(np.zeros((4, 4), dtype=np.uint8))
        rover = SimpleNamespace(
            pos=(2, 1),
            radius=4,
            terrain_knowledge=rover_knowledge,
        )
        control.drones = [drone]
        control.rovers = [rover]
        service = TerrainSharingService(control.dependencies)

        service.share_with_rovers()

        self.assertAlmostEqual(float(rover_knowledge.roughness[1, 1]), 0.6)
        self.assertAlmostEqual(float(rover_knowledge.confidence[1, 1]), 0.5)

        rover_knowledge.confidence.fill(0.0)
        rover_knowledge.roughness.fill(-1.0)
        rover.pos = (20, 20)
        service.share_with_rovers()
        self.assertEqual(float(rover_knowledge.confidence[1, 1]), 0.0)

    def test_rover_sharing_skips_full_snapshots_when_versions_are_unchanged(self) -> None:
        control = make_control()
        drone = make_agent(0, (1, 1))
        drone.terrain_knowledge.roughness[1, 1] = 0.6
        drone.terrain_knowledge.confidence[1, 1] = 0.5
        rover = make_agent(1, (2, 1))
        control.drones = [drone]
        control.rovers = [rover]
        service = TerrainSharingService(control.dependencies)
        drone_snapshot = Mock(wraps=drone.terrain_knowledge.snapshot)
        rover_snapshot = Mock(wraps=rover.terrain_knowledge.snapshot)
        drone.terrain_knowledge.snapshot = drone_snapshot
        rover.terrain_knowledge.snapshot = rover_snapshot

        control.simulation_time.side_effect = [10.0, 10.1, 10.6]
        service.share_with_rovers()
        service.share_with_rovers()
        service.share_with_rovers()

        self.assertEqual(drone_snapshot.call_count, 1)
        self.assertEqual(rover_snapshot.call_count, 1)

    def test_physical_rover_check_in_is_bidirectional_for_slam(self) -> None:
        control = make_control()
        drone = make_agent(0, (1, 1))
        rover = make_agent(0, (2, 1))
        self.seed_slam(drone, 0, 0, 1, 0.9)
        self.seed_slam(rover, 3, 3, 0, 0.8)
        control.drones = [drone]
        control.rovers = [rover]
        service = TerrainSharingService(control.dependencies)

        arrived = service.check_in_with_rover(0, 0)

        self.assertTrue(arrived)
        drone_slam = drone.slam_map.snapshot()
        rover_slam = rover.slam_map.snapshot()
        self.assertEqual(int(drone_slam.occupancy[3, 3]), 0)
        self.assertEqual(int(rover_slam.occupancy[0, 0]), 1)
        drone.movement_controller.mark_shared_slam_changed.assert_called_once()
        drone.movement_controller.begin_rover_sharing.assert_called_once_with(
            0,
            (2, 1),
        )

    def test_rover_check_in_requires_physical_proximity(self) -> None:
        control = make_control()
        drone = make_agent(0, (0, 0))
        rover = make_agent(0, (20, 20))
        self.seed_slam(drone, 0, 0, 1, 0.9)
        control.drones = [drone]
        control.rovers = [rover]

        arrived = TerrainSharingService(
            control.dependencies
        ).check_in_with_rover(0, 0)

        self.assertFalse(arrived)
        self.assertEqual(rover.slam_map.version, 0)

    def test_continuous_rover_contact_does_not_renew_pause_or_block_delivery(self):
        control = make_control()
        drone = make_agent(0, (1, 1))
        rover = make_agent(0, (2, 1))
        control.drones = [drone]
        control.rovers = [rover]
        service = TerrainSharingService(control.dependencies)
        contact = Mock()
        object.__setattr__(control.dependencies, "on_drone_rover_contact", contact)

        for x in range(3):
            control.simulation_time.return_value = 10.0 + x * 0.5
            self.seed_slam(rover, x, 0, 1, 0.9)
            service.share_with_rovers()
            self.assertEqual(int(drone.slam_map.snapshot().occupancy[0, x]), 1)
        self.seed_slam(rover, 3, 0, 1, 0.9)
        self.assertTrue(service.share_on_departure(0, 0))
        self.assertEqual(int(drone.slam_map.snapshot().occupancy[0, 3]), 1)
        drone.movement_controller.begin_rover_sharing.assert_called_once()
        self.assertEqual(contact.call_count, 4)

        # Observe a departure on the movement path, between periodic passes.
        drone.runtime_state.move_to((12, 1))
        service.physical_contact_checkpoint(0)
        drone.runtime_state.move_to((1, 1))
        self.seed_slam(rover, 0, 1, 1, 0.9)
        self.assertTrue(service.check_in_with_rover(0, 0))
        self.assertEqual(drone.movement_controller.begin_rover_sharing.call_count, 2)

    def test_rover_pause_rearms_after_line_of_sight_is_lost(self):
        control = make_control()
        drone = make_agent(0, (0, 0))
        rover = make_agent(0, (2, 0))
        control.drones = [drone]
        control.rovers = [rover]
        service = TerrainSharingService(control.dependencies)
        self.seed_slam(rover, 0, 1, 1, 0.9)
        self.assertTrue(service.check_in_with_rover(0, 0))
        control.map_matrix[0, 1] = 1
        self.assertFalse(service.check_in_with_rover(0, 0))
        control.map_matrix[0, 1] = 0
        self.seed_slam(rover, 0, 2, 1, 0.9)
        self.assertTrue(service.check_in_with_rover(0, 0))
        self.assertEqual(drone.movement_controller.begin_rover_sharing.call_count, 2)

    def test_departure_share_refreshes_map_changed_during_standby(self) -> None:
        control = make_control()
        drone = make_agent(0, (1, 1))
        rover = make_agent(0, (2, 1))
        control.drones = [drone]
        control.rovers = [rover]
        service = TerrainSharingService(control.dependencies)

        self.assertTrue(service.check_in_with_rover(0, 0))
        self.seed_slam(rover, 3, 3, 1, 0.9)

        self.assertTrue(service.share_on_departure(0, 0))
        shared = drone.slam_map.snapshot()
        self.assertEqual(int(shared.occupancy[3, 3]), 1)

    def test_drone_pair_sharing_is_suppressed_at_rover(self) -> None:
        control = make_control()
        trace = RecordingTrace()
        object.__setattr__(control.dependencies, "runtime_trace", trace)
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        rover = make_agent(0, (1, 2))
        self.seed_slam(source, 3, 3, 1, 0.9)
        control.drones = [source, target]
        control.rovers = [rover]
        service = TerrainSharingService(control.dependencies)

        service.share_with_nearby_drones(0)
        service.share_with_nearby_drones(0)

        self.assertEqual(target.slam_map.version, 0)
        suppressed = [
            fields for event, fields in trace.events
            if event == "drone_sharing_suppressed"
        ]
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(suppressed[0]["reason"], "rover_arrival_or_standby")

    def test_periodic_rover_sharing_can_be_disabled_for_sector_protocol(self) -> None:
        control = make_control()
        object.__setattr__(
            control.dependencies,
            "periodic_rover_sharing_enabled",
            False,
        )
        drone = make_agent(0, (1, 1))
        rover = make_agent(0, (2, 1))
        self.seed_slam(drone, 3, 3, 1, 0.9)
        control.drones = [drone]
        control.rovers = [rover]
        service = TerrainSharingService(control.dependencies)

        service.share_with_rovers()

        self.assertEqual(rover.slam_map.version, 0)


if __name__ == "__main__":
    unittest.main()
