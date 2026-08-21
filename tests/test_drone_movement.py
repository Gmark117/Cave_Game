import math
import os
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from agents.drone import Drone
from agents.drone_movement import DroneMovementController
from agents.exploration_policy import RandomDirectionPolicy
from asset_config.helpers import next_cell_coords
from config.simulation_config import MissionConfig, SimulationConfig, SlamConfig
from mapping.drone_sensor import SensorScanCompletion
from mapping.slam_map import FREE, UNKNOWN, SlamSnapshot
from mapping.terrain_knowledge import TerrainKnowledge
from mission.pause_control import PauseCoordinator
from navigation.astar_pathfinder import (
    PATH_COMPLETE,
    PATH_PARTIAL_LIMIT,
    PathResult,
)


class ImmediateEvent:
    def wait(self, _timeout: float) -> bool:
        return False


class MovementControl:
    delay = 1 / 15
    terrain_roughness = np.full((64, 64), 0.4, dtype=np.float32)

    def __init__(self) -> None:
        self.mission_event = ImmediateEvent()
        self.paths = {}
        self.path_requests = []
        self.terrain_knowledge = TerrainKnowledge(
            np.zeros((64, 64), dtype=np.uint8)
        )
        self.terrain_fusion = SimpleNamespace(record_scan=lambda samples: None)

    def compute_path(self, start, goal):
        self.path_requests.append((start, goal))
        return list(self.paths.get((start, goal), ()))

    @staticmethod
    def simulation_time() -> float:
        return time.perf_counter()

    @staticmethod
    def pause_checkpoint() -> bool:
        return True

    def wait_simulation_delay(self, duration: float) -> bool:
        self.mission_event.wait(duration)
        return True


class FixedDirectionPolicy:
    def __init__(self, direction: int) -> None:
        self.direction = direction
        self.candidates = ()

    def choose_direction(self, valid_directions) -> int:
        self.candidates = tuple(valid_directions)
        return self.direction


class StrongestWeightedDirectionPolicy:
    def __init__(self) -> None:
        self.weights = {}

    def choose_weighted_direction(self, direction_weights) -> int:
        self.weights = dict(direction_weights)
        return max(self.weights, key=self.weights.get)


class RecordingTrace:
    def __init__(self) -> None:
        self.events = []

    def record(self, event, **fields) -> None:
        self.events.append((event, fields))


class DroneMovementTests(unittest.TestCase):
    def setUp(self) -> None:
        settings = SimulationConfig(
            mission_config=MissionConfig(map_dim="LARGE", seed=19),
            slam=SlamConfig(
                scan_interval=0.0,
                scan_rays=5,
                point_cloud_max_points=50,
            ),
        )
        self.window = pygame.Surface((64, 64), pygame.SRCALPHA)
        game = SimpleNamespace(
            sim_settings=settings,
            window=self.window,
            width=64,
            height=64,
        )
        self.control = MovementControl()
        cave = np.zeros((64, 64), dtype=np.uint8)
        icon = pygame.Surface((4, 4), pygame.SRCALPHA)
        self.drone = Drone(
            game,
            self.control,
            0,
            (16, 16),
            (255, 0, 0),
            icon,
            cave,
        )

    def test_drone_uses_seeded_random_policy(self) -> None:
        self.assertIsInstance(
            self.drone.exploration_policy,
            RandomDirectionPolicy,
        )

    def test_normal_exploration_moves_straight_without_astar(self) -> None:
        policy = FixedDirectionPolicy(0)
        self.drone.exploration_policy = policy

        self.drone.move()
        snapshot = self.drone.snapshot()

        self.assertEqual(snapshot.position, (16, 6))
        self.assertEqual(snapshot.direction, 0)
        self.assertIn(0, policy.candidates)
        self.assertEqual(self.control.path_requests, [])
        self.assertGreater(len(snapshot.path_history), 2)

    def test_random_candidates_are_limited_to_the_current_vision_cone(self) -> None:
        policy = FixedDirectionPolicy(90)
        self.drone.exploration_policy = policy
        self.drone.runtime_state.begin_exploration(90, ((30, 30),))

        valid_directions, _borders, target = (
            self.drone.movement_controller.find_new_node()
        )

        self.assertEqual(valid_directions, list(range(60, 121)))
        self.assertEqual(target, (26, 16))

    def test_vision_cone_heading_filter_wraps_around_north(self) -> None:
        policy = FixedDirectionPolicy(350)
        self.drone.exploration_policy = policy
        self.drone.runtime_state.begin_exploration(350, ((30, 30),))

        valid_directions, _borders, _target = (
            self.drone.movement_controller.find_new_node()
        )

        self.assertEqual(
            valid_directions,
            [*range(0, 21), *range(320, 360)],
        )

    def test_wall_continuation_biases_weighted_random_heading(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        policy = StrongestWeightedDirectionPolicy()
        self.drone.exploration_policy = policy
        self.drone.runtime_state.move_to((32, 32))
        self.drone.runtime_state.reorient(0)
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[16:52, :48] = FREE
        confidence[16:52, :48] = 1.0
        occupancy[15, :40] = 1
        confidence[15, :40] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))

        controller.find_new_node()

        selected = next(
            fields for name, fields in trace.events
            if name == "drone_random_direction_selected"
        )
        self.assertEqual(selected["selection_mode"], "wall_tracking")
        self.assertGreater(selected["maximum_wall_support"], 0.0)
        self.assertGreater(policy.weights[20], policy.weights[340])

    def test_unknown_region_bias_applies_after_wall_support_is_absent(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        self.drone.runtime_state.move_to((32, 32))
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[:, :40] = FREE
        confidence[:, :40] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        directions = (0, 90, 180, 270)
        step_targets = {
            direction: next_cell_coords(
                32,
                32,
                self.drone.step,
                direction,
            )
            for direction in directions
        }

        bias = controller._exploration_heading_bias(
            directions,
            step_targets,
            vision_fov=60.0,
        )

        self.assertEqual(bias.mode, "unexplored_region")
        self.assertEqual(max(bias.wall_support.values()), 0.0)
        self.assertGreater(bias.weights[90], bias.weights[270])

    def test_small_frontier_clusters_do_not_bias_normal_headings(self) -> None:
        controller = self.drone.movement_controller
        self.drone.runtime_state.move_to((32, 32))
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        for x, y in ((40, 30), (40, 34), (24, 30), (24, 34)):
            occupancy[y, x] = UNKNOWN
            confidence[y, x] = 0.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        directions = (90, 270)
        step_targets = {
            direction: next_cell_coords(32, 32, self.drone.step, direction)
            for direction in directions
        }

        bias = controller._exploration_heading_bias(
            directions,
            step_targets,
            vision_fov=120.0,
        )

        self.assertEqual(bias.mode, "distributed_random")
        self.assertEqual(bias.cluster_count, 4)
        self.assertEqual(bias.eligible_cluster_count, 0)
        self.assertEqual(bias.filtered_cluster_count, 4)
        self.assertEqual(bias.selected_cluster_size, 0)

    def test_generic_frontier_score_balances_size_and_proximity(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        self.drone.runtime_state.move_to((32, 32))
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        occupancy[26:38, 26] = UNKNOWN
        confidence[26:38, 26] = 0.0
        occupancy[23:41, 40] = UNKNOWN
        confidence[23:41, 40] = 0.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        directions = (90, 270)
        step_targets = {
            direction: next_cell_coords(32, 32, self.drone.step, direction)
            for direction in directions
        }

        bias = controller._exploration_heading_bias(
            directions,
            step_targets,
            vision_fov=120.0,
        )

        self.assertEqual(bias.mode, "unexplored_region")
        self.assertEqual(bias.selected_cluster_size, 18)
        self.assertEqual(bias.selected_cluster_size_rank, 1.0)
        self.assertEqual(bias.wall_candidate_count, 0)
        self.assertEqual(bias.generic_candidate_count, 2)
        self.assertAlmostEqual(
            bias.selected_cluster_score,
            2.0 + bias.selected_cluster_proximity,
        )
        self.assertGreater(bias.weights[90], bias.weights[270])

    def test_wall_frontier_tier_overrides_a_nearer_generic_cluster(self) -> None:
        controller = self.drone.movement_controller
        self.drone.runtime_state.move_to((32, 32))
        self.drone.runtime_state.reorient(90)
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        occupancy[23:41, 26] = UNKNOWN
        confidence[23:41, 26] = 0.0
        occupancy[25:39, 44] = UNKNOWN
        confidence[25:39, 44] = 0.0
        occupancy[25:39, 45] = 1
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        directions = (90, 270)
        step_targets = {
            direction: next_cell_coords(32, 32, self.drone.step, direction)
            for direction in directions
        }

        bias = controller._exploration_heading_bias(
            directions,
            step_targets,
            vision_fov=120.0,
        )

        self.assertEqual(bias.mode, "wall_tracking")
        self.assertTrue(bias.selected_cluster_touches_wall)
        self.assertEqual(bias.selected_cluster_size, 14)
        self.assertGreater(bias.weights[90], bias.weights[270])

    def test_wall_tier_prefers_current_heading_continuation(self) -> None:
        controller = self.drone.movement_controller
        self.drone.runtime_state.move_to((32, 32))
        self.drone.runtime_state.reorient(0)
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        occupancy[12, 26:38] = UNKNOWN
        confidence[12, 26:38] = 0.0
        occupancy[11, 26:38] = 1
        occupancy[26:38, 40] = UNKNOWN
        confidence[26:38, 40] = 0.0
        occupancy[26:38, 41] = 1
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        directions = (0, 90)
        step_targets = {
            direction: next_cell_coords(32, 32, self.drone.step, direction)
            for direction in directions
        }

        bias = controller._exploration_heading_bias(
            directions,
            step_targets,
            vision_fov=180.0,
        )

        self.assertEqual(bias.mode, "wall_tracking")
        self.assertGreater(bias.selected_cluster_distance, 15.0)
        self.assertGreater(bias.selected_continuation_alignment, 0.98)
        self.assertAlmostEqual(
            bias.selected_cluster_score,
            2.0 * bias.selected_continuation_alignment
            + 2.0 * bias.selected_cluster_size_rank
            + bias.selected_cluster_proximity,
        )
        self.assertGreater(bias.weights[0], bias.weights[90])

    def test_cached_global_region_guides_beyond_local_slam_window(self) -> None:
        controller = self.drone.movement_controller
        self.drone.runtime_state.move_to((8, 32))
        self.drone.runtime_state.reorient(0)
        self.drone.sensor_controller.vision_sensor.max_range = 12
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        occupancy[20:44, 50:62] = UNKNOWN
        confidence[20:44, 50:62] = 0.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        directions = (0, 90, 180, 270)
        step_targets = {
            direction: next_cell_coords(
                8,
                32,
                self.drone.step,
                direction,
            )
            for direction in directions
        }

        bias = controller._exploration_heading_bias(
            directions,
            step_targets,
            vision_fov=60.0,
        )

        self.assertEqual(bias.cluster_count, 0)
        self.assertEqual(bias.mode, "global_unexplored_region")
        self.assertTrue(bias.global_active)
        self.assertGreaterEqual(bias.global_region_size, 12)
        self.assertGreater(bias.global_region_distance, 24.0)
        self.assertGreater(bias.weights[90], bias.weights[270])

    def test_shared_slam_rebuilds_frontiers_before_exhaustion(self) -> None:
        controller = self.drone.movement_controller
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[20, 20] = FREE
        confidence[20, 20] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        self.drone.runtime_state.begin_exploration(0)
        self.assertEqual(self.drone.snapshot().frontiers, ())

        controller.mark_shared_slam_changed()
        controller._refresh_frontiers_before_mission_state()

        self.assertEqual(self.drone.snapshot().frontiers, ((20, 20),))
        self.assertFalse(self.drone.snapshot().returning_home)

    def test_global_frontier_cache_honors_refresh_interval(self) -> None:
        controller = self.drone.movement_controller
        clock = [10.0]
        controller.dependencies = replace(
            controller.dependencies,
            simulation_time=lambda: clock[0],
        )
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        occupancy[20:44, 50:62] = UNKNOWN
        confidence[20:44, 50:62] = 0.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))

        first = controller._ensure_global_frontier_cache(
            current=(8, 32),
            heading=90.0,
        )
        self.drone.slam_map.record_collision((2, 2))
        clock[0] = 11.0
        throttled = controller._ensure_global_frontier_cache(
            current=(8, 32),
            heading=90.0,
        )
        clock[0] = 12.1
        refreshed = controller._ensure_global_frontier_cache(
            current=(8, 32),
            heading=90.0,
        )

        self.assertIs(throttled, first)
        self.assertIsNot(refreshed, first)
        self.assertGreater(refreshed.slam_version, first.slam_version)

    def test_distribution_bias_repels_a_nearby_peer(self) -> None:
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            get_drone_positions=lambda: (
                (0, (16, 16)),
                (1, (26, 16)),
            ),
        )
        directions = (0, 90, 180, 270)
        step_targets = {
            direction: next_cell_coords(
                16,
                16,
                self.drone.step,
                direction,
            )
            for direction in directions
        }

        bias = controller._exploration_heading_bias(
            directions,
            step_targets,
            vision_fov=60.0,
        )

        self.assertEqual(bias.mode, "distributed_random")
        self.assertEqual(bias.peer_count, 1)
        self.assertGreater(
            bias.separation_support[270],
            bias.separation_support[90],
        )
        self.assertGreater(bias.weights[270], bias.weights[90])

    def test_productive_gain_window_keeps_random_exploration(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        self.drone.slam_map.update_from_observations(
            (16, 16),
            free_cells=(
                (x, y)
                for y in range(10)
                for x in range(10)
            ),
            occupied_cells=(),
        )
        controller._stagnation_distance_travelled = 120.0

        recovered = controller._recover_from_stagnation()

        self.assertFalse(recovered)
        self.assertEqual(self.control.path_requests, [])
        window = next(
            fields for name, fields in trace.events
            if name == "drone_stagnation_window"
        )
        self.assertFalse(window["stagnant"])
        self.assertGreaterEqual(window["sensor_newly_known_cells"], 100)

    def test_stagnation_waits_for_one_unknown_facing_sensor_scan(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        self.drone.exploration_policy = FixedDirectionPolicy(0)
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[4, 16] = FREE
        confidence[4, 16] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        controller._stagnation_distance_travelled = 120.0

        recovered = controller._recover_from_stagnation()

        self.assertTrue(recovered)
        self.assertEqual(self.drone.snapshot().position, (16, 16))
        self.assertEqual(self.drone.snapshot().heading_deg, 0.0)
        self.assertEqual(self.control.path_requests, [])
        started = next(
            fields for name, fields in trace.events
            if name == "drone_stagnation_scan_started"
        )
        self.assertEqual(started["frontier_target"], (16, 4))
        self.assertGreater(started["unknown_support_score"], 0.0)

        controller.move()
        self.assertEqual(self.drone.snapshot().position, (16, 16))
        self.assertFalse(any(
            name == "drone_stagnation_scan_completed"
            for name, _fields in trace.events
        ))

        self.drone.sensor_controller.update()
        controller.move()

        completed = next(
            fields for name, fields in trace.events
            if name == "drone_stagnation_scan_completed"
        )
        self.assertTrue(completed["productive"])
        self.assertFalse(completed["frontier_suppressed"])
        self.assertEqual(self.drone.snapshot().position, (16, 16))

    def test_scan_heading_may_face_a_physically_blocked_wall(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        self.drone.exploration_policy = FixedDirectionPolicy(90)
        self.drone.cave[16, 17] = 1
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        occupancy[16, 17] = UNKNOWN
        confidence[16, 17] = 0.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        controller.rebuild_frontiers(
            stride=controller.frontier_stride,
            confidence_threshold=controller.frontier_confidence_threshold,
        )

        self.assertFalse(self.drone.runtime_state.graph_is_valid(
            (16, 16),
            (22, 16),
        ))
        started = controller._start_frontier_scan(
            ((16, 16),),
            reason="test_wall",
        )

        self.assertTrue(started)
        self.assertEqual(self.drone.snapshot().heading_deg, 90.0)
        self.assertEqual(self.drone.snapshot().position, (16, 16))

        self.drone.sensor_controller.update()
        self.drone.exploration_policy = FixedDirectionPolicy(0)
        controller.move()

        completed = next(
            fields for name, fields in trace.events
            if name == "drone_stagnation_scan_completed"
        )
        self.assertGreaterEqual(completed["sensor_newly_known_cells"], 1)
        self.assertFalse(completed["frontier_suppressed"])

    def test_zero_gain_directed_scan_suppresses_unchanged_frontier(
        self,
    ) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        self.drone.exploration_policy = FixedDirectionPolicy(90)
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        occupancy[16, 17] = UNKNOWN
        confidence[16, 17] = 0.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        controller.rebuild_frontiers(
            stride=controller.frontier_stride,
            confidence_threshold=controller.frontier_confidence_threshold,
        )
        self.assertTrue(controller._start_frontier_scan(
            ((16, 16),),
            reason="test_zero_gain",
        ))
        pending = controller._pending_frontier_scan
        self.assertIsNotNone(pending)
        self.drone.sensor_controller._last_completed_scan = (
            SensorScanCompletion(
                pose=(16, 16, 90.0),
                sequence=pending.minimum_scan_sequence + 1,
                newly_known_cells=0,
                confidence_gain=0.0,
            )
        )
        self.drone.exploration_policy = FixedDirectionPolicy(0)

        controller.move()

        completed = next(
            fields for name, fields in trace.events
            if name == "drone_stagnation_scan_completed"
        )
        self.assertFalse(completed["productive"])
        self.assertTrue(completed["frontier_suppressed"])
        self.assertEqual(
            completed["disposition"],
            "unchanged_geometry_suppressed",
        )
        suppressed = next(
            fields for name, fields in trace.events
            if name == "drone_border_target_suppressed"
        )
        self.assertEqual(suppressed["reason"], "zero_gain_directed_scan")

    def test_stagnation_uses_astar_when_local_unknown_is_out_of_range(
        self,
    ) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        self.drone.exploration_policy = FixedDirectionPolicy(0)
        target = (48, 48)
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[target[1], target[0]] = FREE
        confidence[target[1], target[0]] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        self.control.paths[((16, 16), target)] = [
            (16, 16),
            (32, 32),
            target,
        ]
        controller._stagnation_distance_travelled = 120.0

        recovered = controller._recover_from_stagnation()

        self.assertTrue(recovered)
        self.assertEqual(self.drone.snapshot().position, target)
        self.assertEqual(
            self.control.path_requests,
            [((16, 16), target)],
        )
        path = next(
            fields for name, fields in trace.events
            if name == "drone_stagnation_frontier_path"
        )
        self.assertEqual(path["target"], target)

    def test_stagnation_astar_skips_recent_trail_targets(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        self.drone.exploration_policy = FixedDirectionPolicy(0)
        recent_target = (32, 16)
        fresh_target = (48, 48)
        self.drone.runtime_state.move_to(recent_target)
        self.drone.runtime_state.move_to((16, 16))
        self.drone.runtime_state.replace_frontiers((
            recent_target,
            fresh_target,
        ))
        self.control.paths[((16, 16), fresh_target)] = [
            (16, 16),
            (32, 32),
            fresh_target,
        ]

        reached = controller.reach_border(
            avoid_recent_trail=True,
            recovery_reason="stagnation",
        )

        self.assertTrue(reached)
        self.assertEqual(
            self.control.path_requests,
            [((16, 16), fresh_target)],
        )
        filtered = next(
            fields for name, fields in trace.events
            if name == "drone_stagnation_frontier_filter"
        )
        self.assertEqual(filtered["frontier_count"], 2)
        self.assertEqual(filtered["eligible_frontier_count"], 1)

    def test_border_escape_uses_astar_after_no_long_heading_exists(self) -> None:
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[20, 20] = FREE
        confidence[20, 20] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        self.control.paths[((16, 16), (20, 20))] = [
            (16, 16),
            (17, 17),
            (18, 18),
            (19, 19),
            (20, 20),
        ]
        original_validation = self.drone.runtime_state.graph_is_valid

        def cul_de_sac(current, target) -> bool:
            if math.dist(current, target) > self.drone.step:
                return False
            return original_validation(current, target)

        self.drone.runtime_state.graph_is_valid = cul_de_sac

        self.drone.move()

        self.assertEqual(self.drone.snapshot().position, (20, 20))
        self.assertEqual(
            self.control.path_requests,
            [((16, 16), (20, 20))],
        )

    def test_border_escape_reorients_across_the_full_circle(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        self.drone.exploration_policy = FixedDirectionPolicy(270)
        target = (32, 16)
        self.drone.runtime_state.replace_frontiers((target,))
        self.control.paths[((16, 16), target)] = [
            (16, 16),
            (24, 16),
            target,
        ]

        reached = controller.reach_border()
        snapshot = self.drone.snapshot()

        self.assertTrue(reached)
        self.assertEqual(snapshot.position, target)
        self.assertEqual(snapshot.direction, 270)
        self.assertEqual(snapshot.heading_deg, 270.0)
        self.assertIn((21, 16), snapshot.frontiers)
        self.assertFalse(snapshot.returning_home)
        event = next(
            fields for name, fields in trace.events
            if name == "drone_recovery_reoriented"
        )
        self.assertEqual(event["incoming_heading"], 90.0)
        self.assertEqual(event["direction"], 270)
        self.assertEqual(event["border_target"], (21, 16))
        self.assertEqual(event["valid_direction_count"], 360)

    def test_border_already_at_current_position_is_reoriented(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        self.drone.exploration_policy = FixedDirectionPolicy(0)
        self.drone.runtime_state.replace_frontiers(((16, 16),))

        reached = controller.reach_border()

        self.assertTrue(reached)
        self.assertEqual(self.control.path_requests, [])
        self.assertEqual(self.drone.snapshot().heading_deg, 0.0)
        self.assertTrue(any(
            name == "drone_border_target_suppressed"
            for name, _fields in trace.events
        ))
        self.assertTrue(any(
            name == "drone_recovery_reoriented"
            for name, _fields in trace.events
        ))

    def test_reached_border_stays_suppressed_until_geometry_changes(
        self,
    ) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        self.drone.exploration_policy = FixedDirectionPolicy(180)
        target = (28, 28)
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[target[1], target[0]] = FREE
        confidence[target[1], target[0]] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        controller.rebuild_frontiers(stride=1, confidence_threshold=0.6)
        self.control.paths[((16, 16), target)] = [
            (16, 16),
            (22, 22),
            target,
        ]

        self.assertTrue(controller.reach_border())
        controller.rebuild_frontiers(stride=1, confidence_threshold=0.6)

        self.assertEqual(self.drone.snapshot().frontiers, ())
        unchanged = [
            fields for name, fields in trace.events
            if name == "drone_frontiers_rebuilt"
        ][-1]
        self.assertEqual(unchanged["raw_frontier_count"], 1)
        self.assertEqual(unchanged["suppressed_frontier_count"], 1)
        self.assertEqual(unchanged["reactivated_frontier_count"], 0)

        changed_occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        changed_confidence = np.zeros((64, 64), dtype=np.float32)
        changed_occupancy[28, 29] = FREE
        changed_confidence[28, 29] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(
            changed_occupancy,
            changed_confidence,
        ))
        controller.rebuild_frontiers(stride=1, confidence_threshold=0.6)

        self.assertEqual(
            self.drone.snapshot().frontiers,
            ((28, 28), (29, 28)),
        )
        changed = [
            fields for name, fields in trace.events
            if name == "drone_frontiers_rebuilt"
        ][-1]
        self.assertEqual(changed["suppressed_frontier_count"], 0)
        self.assertEqual(changed["reactivated_frontier_count"], 1)

    def test_homing_uses_astar_and_updates_the_rendered_path(self) -> None:
        self.drone.runtime_state.move_to((20, 20))
        self.control.paths[((20, 20), (16, 16))] = [
            (20, 20),
            (18, 18),
            (16, 16),
        ]

        reached = self.drone.movement_controller.reach_start_point()
        snapshot = self.drone.snapshot()

        self.assertTrue(reached)
        self.assertEqual(snapshot.position, (16, 16))
        self.assertEqual(snapshot.path_history[-1], (16, 16))
        self.assertEqual(
            self.control.path_requests,
            [((20, 20), (16, 16))],
        )

    def test_frontier_astar_continues_from_a_capped_progress_segment(self) -> None:
        controller = self.drone.movement_controller
        target = (40, 16)
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[target[1], target[0]] = FREE
        confidence[target[1], target[0]] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        self.drone.runtime_state.replace_frontiers((target,))
        results = iter((
            PathResult(
                ((16, 16), (24, 16)),
                PATH_PARTIAL_LIMIT,
                200000,
                16.0,
            ),
            PathResult(
                ((24, 16), (32, 16), target),
                PATH_COMPLETE,
                100,
                0.0,
            ),
        ))
        controller.dependencies = replace(
            controller.dependencies,
            compute_path_segment=lambda _start, _goal: next(results),
        )

        first_segment = controller.reach_border()

        self.assertTrue(first_segment)
        self.assertEqual(self.drone.snapshot().position, (24, 16))
        self.assertEqual(controller._pending_frontier_route.target, target)
        self.assertIn(target, self.drone.snapshot().frontiers)

        controller.move()

        self.assertEqual(self.drone.snapshot().position, target)
        self.assertIsNone(controller._pending_frontier_route)

    def test_homing_replans_after_a_capped_progress_segment(self) -> None:
        controller = self.drone.movement_controller
        self.drone.runtime_state.move_to((40, 16))
        results = iter((
            PathResult(
                ((40, 16), (28, 16)),
                PATH_PARTIAL_LIMIT,
                200000,
                12.0,
            ),
            PathResult(
                ((28, 16), (20, 16), (16, 16)),
                PATH_COMPLETE,
                100,
                0.0,
            ),
        ))
        controller.dependencies = replace(
            controller.dependencies,
            compute_path_segment=lambda _start, _goal: next(results),
        )

        first_segment = controller.reach_start_point()
        completed = controller.reach_start_point()

        self.assertFalse(first_segment)
        self.assertTrue(completed)
        self.assertEqual(self.drone.snapshot().position, (16, 16))

    def test_frontier_rebuild_uses_only_local_slam(self) -> None:
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[20, 20] = FREE
        confidence[20, 20] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        self.drone.terrain_knowledge.confidence[:] = 1.0
        self.control.terrain_knowledge.confidence[:] = 1.0

        self.drone.movement_controller.rebuild_frontiers(
            stride=1,
            confidence_threshold=0.6,
        )

        self.assertEqual(self.drone.snapshot().frontiers, ((20, 20),))

    def test_low_confidence_free_cell_is_not_a_frontier(self) -> None:
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[20, 20] = FREE
        confidence[20, 20] = 0.1
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        self.drone.terrain_knowledge.confidence[:] = 1.0

        self.drone.movement_controller.rebuild_frontiers(
            stride=1,
            confidence_threshold=0.6,
        )

        self.assertEqual(self.drone.snapshot().frontiers, ())

    def test_path_execution_traces_actual_distance(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )

        followed = controller._follow_path(
            ((16, 16), (19, 20)),
            source="test",
        )

        self.assertTrue(followed)
        motion = next(
            fields for event, fields in trace.events
            if event == "drone_motion"
        )
        self.assertEqual(motion["source"], "test")
        self.assertAlmostEqual(motion["travelled_distance"], 5.0)

    def test_path_traversal_stops_at_pause_barrier(self) -> None:
        stop_event = threading.Event()
        coordinator = PauseCoordinator(stop_event)
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            pause_checkpoint=coordinator.checkpoint,
            wait_simulation_delay=coordinator.wait,
        )
        self.drone.delay = 0.2
        self.drone.speed_factor = 1
        first_node = threading.Event()
        original_move_to = self.drone.runtime_state.move_to

        def record_node(node) -> None:
            original_move_to(node)
            first_node.set()

        self.drone.runtime_state.move_to = record_node

        def follow_path() -> None:
            coordinator.register_current_worker(("drone", 0))
            try:
                controller._follow_path(
                    ((17, 16), (18, 16), (19, 16), (20, 16))
                )
            finally:
                coordinator.unregister_current_worker()

        worker = threading.Thread(target=follow_path)
        worker.start()
        self.assertTrue(first_node.wait(2.0))
        coordinator.pause()
        paused_position = self.drone.snapshot().position
        time.sleep(0.25)

        self.assertEqual(self.drone.snapshot().position, paused_position)
        self.assertNotEqual(paused_position, (20, 16))

        coordinator.resume()
        worker.join(2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.drone.snapshot().position, (20, 16))


if __name__ == "__main__":
    unittest.main()
