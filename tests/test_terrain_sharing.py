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
        merge_frontiers=Mock(),
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

    def test_nearby_drone_receives_new_terrain_and_frontiers(self) -> None:
        control = make_control()
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        source.terrain_knowledge.roughness[1, 1] = 0.8
        source.terrain_knowledge.confidence[1, 1] = 1.0
        source.runtime_state.merge_frontiers([(3, 3)])
        self.seed_slam(source, 2, 2, 1, 0.9)
        control.drones = [source, target]
        service = TerrainSharingService(control.dependencies)

        service.share_with_nearby_drones(0)

        target.merge_frontiers.assert_called_once_with(((3, 3),))
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

    def test_pair_without_new_information_does_not_enter_cooldown(self) -> None:
        control = make_control()
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        control.drones = [source, target]
        service = TerrainSharingService(control.dependencies)
        service.pair_share_cooldown = 5.0

        service.share_with_nearby_drones(0)
        source.terrain_knowledge.roughness[1, 1] = 0.8
        source.terrain_knowledge.confidence[1, 1] = 1.0
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

    def test_rover_sharing_skips_full_snapshots_during_cooldown(self) -> None:
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

        self.assertEqual(drone_snapshot.call_count, 2)
        self.assertEqual(rover_snapshot.call_count, 2)


if __name__ == "__main__":
    unittest.main()
