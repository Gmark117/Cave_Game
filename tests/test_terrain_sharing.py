import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from SlamMap import SlamMap
from mapping.terrain_knowledge import TerrainKnowledge
from mapping.terrain_sharing import TerrainSharingService


def make_agent(
    agent_id: int,
    position: tuple[int, int],
    shape: tuple[int, int] = (4, 4),
):
    terrain_knowledge = TerrainKnowledge(np.zeros(shape, dtype=np.uint8))
    return SimpleNamespace(
        id=agent_id,
        pos=position,
        radius=4,
        terrain_knowledge=terrain_knowledge,
        known_roughness=terrain_knowledge.roughness,
        terrain_confidence=terrain_knowledge.confidence,
        terrain_lock=terrain_knowledge.lock,
        exploration_lock=threading.Lock(),
        slam_lock=threading.Lock(),
        border=[],
        slam_map=SlamMap(*shape),
        merge_terrain_data=Mock(),
        merge_exploration_data=Mock(),
    )


def make_control():
    cave = np.zeros((4, 4), dtype=np.uint8)
    return SimpleNamespace(
        settings=SimpleNamespace(
            drone_share_interval=0.0,
            pair_share_cooldown=0.0,
            rover_share_interval=0.5,
        ),
        map_matrix=cave,
        map_h=4,
        map_w=4,
        floor_mask=cave == 0,
        share_compare_stride=1,
        min_share_new_info_ratio=0.1,
        min_share_overlap_diff_ratio=0.25,
        min_share_roughness_delta=0.1,
        presentation=SimpleNamespace(terrain_heatmap_dirty=False),
        drones=[],
        rovers=[],
    )


class TerrainSharingTests(unittest.TestCase):
    def test_line_of_sight_rejects_walls_and_out_of_bounds(self) -> None:
        control = make_control()
        service = TerrainSharingService(control)
        control.map_matrix[1, 1] = 1

        self.assertFalse(service.has_line_of_sight((0, 0), (2, 2)))
        self.assertFalse(service.has_line_of_sight((0, 0), (8, 8)))
        self.assertTrue(service.has_line_of_sight((2, 0), (3, 0)))

    def test_nearby_drone_receives_new_terrain_and_frontiers(self) -> None:
        control = make_control()
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        source.known_roughness[1, 1] = 0.8
        source.terrain_confidence[1, 1] = 1.0
        source.border = [(3, 3)]
        source.slam_map.occupancy[2, 2] = 1
        source.slam_map.confidence[2, 2] = 0.9
        control.drones = [source, target]
        service = TerrainSharingService(control)

        with patch("mapping.terrain_sharing.time.perf_counter", return_value=10.0):
            service.share_with_nearby_drones(0)

        target.merge_terrain_data.assert_not_called()
        target.merge_exploration_data.assert_called_once_with(None, [(3, 3)])
        source.merge_terrain_data.assert_not_called()
        self.assertAlmostEqual(float(target.known_roughness[1, 1]), 0.8)
        self.assertAlmostEqual(float(target.terrain_confidence[1, 1]), 1.0)
        self.assertEqual(int(target.slam_map.occupancy[2, 2]), 1)
        self.assertAlmostEqual(float(target.slam_map.confidence[2, 2]), 0.9)
        self.assertEqual(service.last_drone_share[0], 10.0)
        self.assertEqual(service.last_pair_share[(0, 1)], 10.0)
        self.assertTrue(control.presentation.terrain_heatmap_dirty)

    def test_pair_cooldown_prevents_duplicate_exchange(self) -> None:
        control = make_control()
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        source.known_roughness[1, 1] = 0.8
        source.terrain_confidence[1, 1] = 1.0
        control.drones = [source, target]
        service = TerrainSharingService(control)
        service.pair_share_cooldown = 5.0
        service.last_pair_share[(0, 1)] = 8.0

        with patch("mapping.terrain_sharing.time.perf_counter", return_value=10.0):
            service.share_with_nearby_drones(0)

        self.assertEqual(float(target.terrain_confidence[1, 1]), 0.0)

    def test_pair_without_new_information_does_not_enter_cooldown(self) -> None:
        control = make_control()
        source = make_agent(0, (1, 1))
        target = make_agent(1, (2, 1))
        control.drones = [source, target]
        service = TerrainSharingService(control)
        service.pair_share_cooldown = 5.0

        with patch("mapping.terrain_sharing.time.perf_counter", return_value=10.0):
            service.share_with_nearby_drones(0)
            source.known_roughness[1, 1] = 0.8
            source.terrain_confidence[1, 1] = 1.0
            service.share_with_nearby_drones(0)

        self.assertEqual(service.last_pair_share[(0, 1)], 10.0)
        self.assertAlmostEqual(float(target.known_roughness[1, 1]), 0.8)
        self.assertAlmostEqual(float(target.terrain_confidence[1, 1]), 1.0)

    def test_service_owns_all_sharing_schedule_state(self) -> None:
        control = make_control()
        service = TerrainSharingService(control)

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
        service = TerrainSharingService(control)
        exchange_started = threading.Event()
        release_exchange = threading.Event()
        exchange_calls = []
        worker_errors = []

        def blocking_exchange(drone, other_drone):
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
        drone.known_roughness[1, 1] = 0.6
        drone.terrain_confidence[1, 1] = 0.5
        rover_knowledge = TerrainKnowledge(np.zeros((4, 4), dtype=np.uint8))
        rover = SimpleNamespace(
            pos=(2, 1),
            radius=4,
            terrain_knowledge=rover_knowledge,
            known_roughness=rover_knowledge.roughness,
            terrain_confidence=rover_knowledge.confidence,
        )
        control.drones = [drone]
        control.rovers = [rover]
        service = TerrainSharingService(control)

        service.share_with_rovers()

        self.assertAlmostEqual(float(rover.known_roughness[1, 1]), 0.6)
        self.assertAlmostEqual(float(rover.terrain_confidence[1, 1]), 0.5)

        rover.terrain_confidence.fill(0.0)
        rover.known_roughness.fill(-1.0)
        rover.pos = (20, 20)
        service.share_with_rovers()
        self.assertEqual(float(rover.terrain_confidence[1, 1]), 0.0)

    def test_rover_sharing_skips_full_snapshots_during_cooldown(self) -> None:
        control = make_control()
        drone = make_agent(0, (1, 1))
        drone.known_roughness[1, 1] = 0.6
        drone.terrain_confidence[1, 1] = 0.5
        rover = make_agent(1, (2, 1))
        control.drones = [drone]
        control.rovers = [rover]
        service = TerrainSharingService(control)
        drone_snapshot = Mock(wraps=drone.terrain_knowledge.snapshot)
        rover_snapshot = Mock(wraps=rover.terrain_knowledge.snapshot)
        drone.terrain_knowledge.snapshot = drone_snapshot
        rover.terrain_knowledge.snapshot = rover_snapshot

        with patch(
            "mapping.terrain_sharing.time.perf_counter",
            side_effect=[10.0, 10.1, 10.6],
        ):
            service.share_with_rovers()
            service.share_with_rovers()
            service.share_with_rovers()

        self.assertEqual(drone_snapshot.call_count, 2)
        self.assertEqual(rover_snapshot.call_count, 2)


if __name__ == "__main__":
    unittest.main()
