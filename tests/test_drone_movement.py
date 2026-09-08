import math
import os
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from agents.drone import Drone
from agents.drone_movement import (
    DroneMovementController,
    _GlobalFrontierRegion,
    _GlobalFrontierTile,
)
from agents.exploration_policy import RandomDirectionPolicy
from asset_config.helpers import next_cell_coords
from config.simulation_config import MissionConfig, SimulationConfig, SlamConfig
from mapping.drone_sensor import SensorScanCompletion
from mapping.exploration_sectors import (
    SectorAssignment,
    SectorCheckInResult,
    SectorFrontierComponent,
)
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


def global_region(
    target: tuple[int, int],
    *,
    size: int = 20,
    touches_wall: bool = False,
) -> _GlobalFrontierRegion:
    """Build one single-tile region for strategic-selection tests."""
    wall_cells = size if touches_wall else 0
    return _GlobalFrontierRegion(
        tiles=(_GlobalFrontierTile(
            target=target,
            size=size,
            wall_cells=wall_cells,
        ),),
        size=size,
        wall_cells=wall_cells,
        tile_count=1,
    )


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

    def test_initial_physical_check_in_installs_sector_assignment(self) -> None:
        controller = self.drone.movement_controller
        controller._suppressed_frontier_geometry[(16, 16)] = ((0, 0),)
        controller.border_retry_until[(16, 16)] = float("inf")
        assignment = SectorAssignment(
            sector_id=0,
            generation=0,
            owner_drone_id=0,
            cell_size=32,
            cells=frozenset({(0, 0), (1, 0), (0, 1), (1, 1)}),
            seed=(48, 16),
            gateway=(16, 16),
            frontier_cells=0,
            rover_slam_version=0,
            bootstrap=True,
        )
        controller.dependencies = replace(
            controller.dependencies,
            sector_check_in=lambda _drone_id, _sector_id: (
                SectorCheckInResult(
                    arrived=True,
                    assignment=assignment,
                    generation=0,
                )
            ),
            get_check_in_position=lambda: (16, 16),
        )
        controller._sector_check_in_required = True

        self.drone.move()

        self.assertEqual(controller._sector_assignment, assignment)
        self.assertFalse(controller._sector_check_in_required)
        self.assertFalse(self.drone.snapshot().returning_home)
        self.assertEqual(controller._suppressed_frontier_geometry, {})
        self.assertEqual(controller.border_retry_until, {})

    def test_new_assignment_keeps_unchanged_frontier_suppression(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        trace = RecordingTrace()
        target = (28, 28)
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[target[1], target[0]] = FREE
        confidence[target[1], target[0]] = 1.0
        occupancy[28, 32] = FREE
        confidence[28, 32] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(
            occupancy,
            confidence,
        ))
        controller.rebuild_frontiers(stride=4, confidence_threshold=0.6)
        controller._suppress_frontier_target(
            target,
            reason="test_cross_generation",
        )
        assignment = SectorAssignment(
            sector_id=3,
            generation=1,
            owner_drone_id=0,
            cell_size=32,
            cells=frozenset({(0, 0)}),
            seed=(48, 48),
            gateway=(16, 16),
            frontier_cells=1,
            rover_slam_version=1,
        )
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
            sector_check_in=lambda _drone_id, _sector_id: (
                SectorCheckInResult(
                    arrived=True,
                    assignment=assignment,
                    generation=1,
                )
            ),
            get_check_in_position=lambda: (16, 16),
        )
        controller._sector_check_in_required = True

        controller._advance_sector_check_in()

        self.assertIn(target, controller._suppressed_frontier_geometry)
        self.assertNotIn(target, self.drone.snapshot().frontiers)
        assigned = next(
            fields for name, fields in trace.events
            if name == "drone_sector_assigned"
        )
        self.assertEqual(assigned["carried_suppression_count"], 1)
        self.assertEqual(assigned["active_suppression_count"], 1)

    def test_sector_check_in_uses_astar_before_breadcrumbs(self) -> None:
        controller = self.drone.movement_controller
        trace = RecordingTrace()
        self.drone.runtime_state.move_to((26, 16))
        self.control.paths[((26, 16), (16, 16))] = [
            (x, 16) for x in range(25, 15, -1)
        ]
        assignment = SectorAssignment(
            sector_id=1,
            generation=1,
            owner_drone_id=0,
            cell_size=32,
            cells=frozenset({(0, 0)}),
            seed=(16, 16),
            gateway=(16, 16),
            frontier_cells=1,
            rover_slam_version=1,
        )

        def check_in(_drone_id, _sector_id):
            return SectorCheckInResult(
                arrived=self.drone.snapshot().position == (16, 16),
                assignment=assignment,
                generation=1,
            )

        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
            sector_check_in=check_in,
            get_check_in_position=lambda: (16, 16),
        )
        controller._sector_check_in_required = True
        controller._completed_sector_id = 0

        self.drone.move()

        self.assertEqual(self.control.path_requests, [((26, 16), (16, 16))])
        sources = [
            fields["source"]
            for event, fields in trace.events
            if event == "drone_sector_check_in_path"
        ]
        self.assertEqual(sources, ["astar"])
        self.assertEqual(self.drone.snapshot().position, (16, 16))

    def test_sector_waiting_uses_rover_notification_without_polling(self) -> None:
        controller = self.drone.movement_controller
        trace = RecordingTrace()
        clock = [10.0]
        calls = []
        assignment_calls = []
        assignment_ready = threading.Event()
        assignment = SectorAssignment(
            sector_id=3,
            generation=1,
            owner_drone_id=0,
            cell_size=32,
            cells=frozenset({(0, 0)}),
            seed=(16, 16),
            gateway=(16, 16),
            frontier_cells=1,
            rover_slam_version=1,
        )

        def check_in(_drone_id, _sector_id):
            calls.append(clock[0])
            if len(calls) == 1:
                return SectorCheckInResult(
                    arrived=True,
                    waiting_for_team=True,
                    generation=0,
                    waiting_generation=0,
                    assignment_ready=assignment_ready,
                )

        def claim_assignment(_drone_id):
            assignment_calls.append(clock[0])
            return SectorCheckInResult(
                arrived=True,
                assignment=assignment,
                generation=1,
            )

        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
            simulation_time=lambda: clock[0],
            sector_check_in=check_in,
            sector_assignment=claim_assignment,
            get_check_in_position=lambda: (16, 16),
        )
        controller._sector_check_in_required = True

        controller._advance_sector_check_in()
        clock[0] = 10.25
        controller._advance_sector_check_in()
        self.assertEqual(calls, [10.0])
        self.assertEqual(assignment_calls, [])

        clock[0] = 10.5
        assignment_ready.set()
        controller._advance_sector_check_in()

        self.assertEqual(calls, [10.0])
        self.assertEqual(assignment_calls, [10.5])
        completed = next(
            fields for name, fields in trace.events
            if name == "drone_sector_wait_completed"
        )
        self.assertEqual(completed["waiting_generation"], 0)
        self.assertEqual(completed["next_generation"], 1)
        self.assertAlmostEqual(completed["waited_seconds"], 0.5)

    def test_sector_check_in_uses_breadcrumb_when_astar_is_unavailable(self) -> None:
        controller = self.drone.movement_controller
        trace = RecordingTrace()
        self.drone.runtime_state.move_to((26, 16))

        def check_in(_drone_id, _sector_id):
            return SectorCheckInResult(
                arrived=self.drone.snapshot().position == (16, 16),
                waiting_for_team=True,
                generation=0,
            )

        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
            sector_check_in=check_in,
            get_check_in_position=lambda: (16, 16),
        )
        controller._sector_check_in_required = True

        self.drone.move()

        sources = [
            fields["source"]
            for event, fields in trace.events
            if event == "drone_sector_check_in_path"
        ]
        self.assertEqual(sources, ["astar", "breadcrumb_fallback"])
        self.assertEqual(self.drone.snapshot().position, (16, 16))

    def test_sector_filters_local_and_global_frontier_evidence(self) -> None:
        controller = self.drone.movement_controller
        controller._sector_assignment = SectorAssignment(
            sector_id=3,
            generation=1,
            owner_drone_id=0,
            cell_size=32,
            cells=frozenset({(0, 0), (0, 1)}),
            seed=(16, 32),
            gateway=(16, 16),
            frontier_cells=10,
            rover_slam_version=1,
        )
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[8:56, 8:56] = FREE
        confidence[8:56, 8:56] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))

        controller.rebuild_frontiers(stride=1, confidence_threshold=0.6)
        regions = controller._coarse_global_frontier_regions(
            self.drone.slam_map.snapshot(point_limit=0)
        )

        self.assertTrue(self.drone.snapshot().frontiers)
        self.assertTrue(
            all(x < 32 for x, _y in self.drone.snapshot().frontiers)
        )
        self.assertTrue(regions)
        self.assertTrue(
            all(tile.target[0] < 32 for region in regions for tile in region.tiles)
        )

    def test_sector_boundary_penalizes_outward_random_steps(self) -> None:
        controller = self.drone.movement_controller
        controller._sector_assignment = SectorAssignment(
            sector_id=3,
            generation=1,
            owner_drone_id=0,
            cell_size=32,
            cells=frozenset({(0, 0)}),
            seed=(16, 16),
            gateway=(16, 16),
            frontier_cells=1,
            rover_slam_version=0,
        )
        self.drone.runtime_state.move_to((28, 16))
        directions = (90, 270)
        step_targets = {
            direction: next_cell_coords(28, 16, 10, direction)
            for direction in directions
        }

        bias = controller._exploration_heading_bias(
            directions,
            step_targets,
            vision_fov=60.0,
        )

        self.assertGreater(bias.weights[270], bias.weights[90])

    def test_sector_exhaustion_survives_drift_outside_entered_sector(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        controller._sector_assignment = SectorAssignment(
            sector_id=3,
            generation=1,
            owner_drone_id=0,
            cell_size=32,
            cells=frozenset({(0, 0)}),
            seed=(16, 16),
            gateway=(16, 16),
            frontier_cells=0,
            rover_slam_version=0,
        )
        controller._sector_assignment_entered = True
        self.drone.runtime_state.begin_exploration(90)
        self.drone.runtime_state.move_to((40, 16))
        self.drone.runtime_state.replace_frontiers(())
        controller._frontier_slam_version = self.drone.slam_map.version

        self.assertFalse(controller._sector_exhausted())
        self.assertTrue(controller._sector_exhausted())

    def test_boxed_out_of_sector_drone_uses_astar_ingress_recovery(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        trace = RecordingTrace()
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        controller._sector_assignment = SectorAssignment(
            sector_id=3,
            generation=1,
            owner_drone_id=0,
            cell_size=32,
            cells=frozenset({(0, 0)}),
            seed=(16, 16),
            gateway=(16, 16),
            frontier_cells=0,
            rover_slam_version=0,
        )
        self.drone.runtime_state.move_to((40, 16))
        self.control.paths[((40, 16), (31, 16))] = [
            (x, 16) for x in range(39, 30, -1)
        ]

        recovered = controller._recover_to_assigned_sector()

        self.assertTrue(recovered)
        self.assertEqual(self.drone.snapshot().position, (31, 16))
        self.assertTrue(controller._sector_assignment_entered)
        recovery = next(
            fields for name, fields in trace.events
            if name == "drone_sector_ingress_recovery"
        )
        self.assertEqual(recovery["source"], "sector_ingress_astar")
        self.assertTrue(recovery["entered_sector"])

    def test_unreachable_sector_ingress_abandons_without_spinning(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        trace = RecordingTrace()
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        controller._sector_assignment = SectorAssignment(
            sector_id=3,
            generation=1,
            owner_drone_id=0,
            cell_size=32,
            cells=frozenset({(0, 0)}),
            seed=(16, 16),
            gateway=(16, 16),
            frontier_cells=0,
            rover_slam_version=0,
        )
        self.drone.runtime_state.begin_exploration(90)
        self.drone.runtime_state.move_to((40, 16))

        with (
            patch.object(
                controller,
                "_sector_ingress_targets",
                return_value=((31, 16),),
            ),
            patch.object(
                controller,
                "_compute_path",
                return_value=PathResult(
                    (),
                    "unreachable",
                    0,
                    9.0,
                ),
            ),
            patch.object(
                controller,
                "_direction_candidates",
                return_value=([], [], {}),
            ),
        ):
            self.assertFalse(controller._recover_to_assigned_sector())
            self.assertTrue(controller._recover_to_assigned_sector())

        self.assertTrue(controller._sector_check_in_required)
        exhausted = next(
            fields for name, fields in trace.events
            if name == "drone_sector_exhausted"
        )
        self.assertEqual(exhausted["reason"], "sector_ingress_unreachable")
        self.assertTrue(any(
            name == "drone_sector_ingress_abandoned"
            for name, _fields in trace.events
        ))

    def test_random_candidates_are_limited_to_the_current_vision_cone(self) -> None:
        policy = FixedDirectionPolicy(90)
        self.drone.exploration_policy = policy
        self.drone.runtime_state.begin_exploration(90, ((30, 30),))

        valid_directions, _borders, target = (
            self.drone.movement_controller.find_new_node()
        )

        self.assertEqual(valid_directions, list(range(60, 121)))
        self.assertEqual(target, (26, 16))

    def test_coverage_memory_prefers_a_fresh_heading(self) -> None:
        controller = self.drone.movement_controller
        current = (16, 32)
        self.drone.runtime_state.move_to(current)
        now = controller._simulation_time()
        controller._record_coverage_transition(
            current,
            (48, 32),
            now,
        )
        self.drone.runtime_state.move_to(current)
        directions = (90, 270)
        step_targets = {
            direction: next_cell_coords(
                *current,
                self.drone.step,
                direction,
            )
            for direction in directions
        }

        bias = controller._exploration_heading_bias(
            directions,
            step_targets,
            vision_fov=120.0,
        )

        self.assertGreater(
            bias.coverage_visit_pressure[90],
            bias.coverage_visit_pressure[270],
        )
        self.assertGreater(
            bias.coverage_edge_pressure[90],
            bias.coverage_edge_pressure[270],
        )
        self.assertLess(
            bias.coverage_penalty_factor[90],
            bias.coverage_penalty_factor[270],
        )
        self.assertGreater(bias.weights[270], bias.weights[90])

    def test_coverage_memory_pressure_decays(self) -> None:
        controller = self.drone.movement_controller
        current = (16, 32)
        now = controller._simulation_time()
        controller._record_coverage_transition(
            current,
            (48, 32),
            now,
        )
        immediate = controller._coverage_heading_penalties(
            (90,),
            current=current,
            apply_penalty=True,
        )[3][90]
        controller.dependencies = replace(
            controller.dependencies,
            simulation_time=lambda: (
                now + 10.0 * controller.coverage_memory_decay_seconds
            ),
        )

        decayed = controller._coverage_heading_penalties(
            (90,),
            current=current,
            apply_penalty=True,
        )[3][90]

        self.assertGreater(decayed, immediate)
        self.assertGreater(decayed, 0.999)

    def test_sector_ingress_is_exempt_from_coverage_penalty(self) -> None:
        controller = self.drone.movement_controller
        controller._sector_assignment = SectorAssignment(
            sector_id=3,
            generation=1,
            owner_drone_id=0,
            cell_size=32,
            cells=frozenset({(0, 0)}),
            seed=(16, 16),
            gateway=(16, 16),
            frontier_cells=1,
            rover_slam_version=0,
        )
        current = (48, 48)
        self.drone.runtime_state.move_to(current)
        now = controller._simulation_time()
        controller._record_coverage_transition(
            current,
            (16, 48),
            now,
        )
        self.drone.runtime_state.move_to(current)
        directions = (0, 270)
        step_targets = {
            direction: next_cell_coords(
                *current,
                self.drone.step,
                direction,
            )
            for direction in directions
        }

        bias = controller._exploration_heading_bias(
            directions,
            step_targets,
            vision_fov=120.0,
        )

        self.assertEqual(bias.mode, "sector_ingress")
        self.assertEqual(
            set(bias.coverage_penalty_factor.values()),
            {1.0},
        )

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
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
            get_drone_positions=lambda: (
                (0, (8, 32)),
                (1, (56, 32)),
            ),
        )
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
        rebuilt = next(
            fields for name, fields in trace.events
            if name == "drone_global_frontiers_rebuilt"
        )
        self.assertIn("selected_requester_distance", rebuilt)
        self.assertIn("selected_nearest_peer_distance", rebuilt)
        self.assertIn("selected_ownership_margin", rebuilt)
        self.assertIn("selected_launch_sector_alignment", rebuilt)
        self.assertIn("selected_ownership_contribution", rebuilt)

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

    def test_local_exhaustion_starts_homing_despite_global_work(self) -> None:
        controller = self.drone.movement_controller
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        occupancy[20:44, 50:62] = UNKNOWN
        confidence[20:44, 50:62] = 0.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        self.drone.runtime_state.begin_exploration(0)

        with (
            patch.object(
                controller,
                "_refresh_frontiers_before_mission_state",
            ),
            patch.object(
                controller,
                "_ensure_global_frontier_cache",
            ) as global_cache,
        ):
            controller.move()

        snapshot = self.drone.snapshot()
        self.assertTrue(snapshot.returning_home)
        self.assertTrue(snapshot.done)
        global_cache.assert_not_called()

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

    def test_global_selection_retains_a_competitive_previous_region(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        current = (32, 32)
        north = global_region((32, 0), touches_wall=True)
        east = global_region((60, 32), touches_wall=True)

        unconstrained = controller._select_global_frontier_region(
            (north, east),
            current=current,
            heading=40.0,
        )
        retained = controller._select_global_frontier_region(
            (north, east),
            current=current,
            heading=40.0,
            preferred_region=east,
        )
        dominated = controller._select_global_frontier_region(
            (north, east),
            current=current,
            heading=0.0,
            preferred_region=east,
        )

        self.assertIs(unconstrained.region, north)
        self.assertIs(retained.region, east)
        self.assertTrue(retained.retained_previous)
        self.assertEqual(retained.previous_region_overlap, 1.0)
        self.assertIs(dominated.region, north)
        self.assertFalse(dominated.retained_previous)

    def test_global_commitment_retains_the_previous_coarse_tile(self) -> None:
        controller = self.drone.movement_controller
        current = (32, 32)
        region = _GlobalFrontierRegion(
            tiles=(
                _GlobalFrontierTile(
                    target=(32, 0),
                    size=20,
                    wall_cells=20,
                ),
                _GlobalFrontierTile(
                    target=(60, 32),
                    size=20,
                    wall_cells=20,
                ),
            ),
            size=40,
            wall_cells=40,
            tile_count=2,
        )

        selected = controller._select_global_frontier_region(
            (region,),
            current=current,
            heading=0.0,
            preferred_region=region,
            preferred_position=(60, 32),
        )

        self.assertIs(selected.region, region)
        self.assertEqual(selected.position, (60, 32))
        self.assertTrue(selected.retained_previous)

    def test_global_selection_prefers_a_spatially_owned_target(self) -> None:
        controller = self.drone.movement_controller
        current = (24, 32)
        controller.dependencies = replace(
            controller.dependencies,
            get_drone_positions=lambda: (
                (0, current),
                (1, (8, 32)),
            ),
        )
        regions = (
            global_region((4, 32)),
            global_region((60, 32)),
        )

        selection = controller._select_global_frontier_region(
            regions,
            current=current,
            heading=0.0,
        )

        self.assertEqual(selection.position, (60, 32))
        self.assertGreater(
            selection.nearest_peer_distance,
            selection.requester_distance,
        )
        self.assertGreater(selection.ownership_margin, 0.0)
        self.assertGreater(selection.ownership_contribution, 0.0)

    def test_global_ownership_keeps_exact_local_wall_tier(self) -> None:
        controller = self.drone.movement_controller
        current = (32, 32)
        controller.dependencies = replace(
            controller.dependencies,
            get_drone_positions=lambda: (
                (0, current),
                (1, (48, 32)),
            ),
        )
        self.drone.runtime_state.move_to(current)
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

        self.assertTrue(bias.selected_cluster_touches_wall)
        self.assertEqual(bias.selected_cluster_size, 14)

    def test_global_wall_representative_keeps_continuation_priority(
        self,
    ) -> None:
        controller = self.drone.movement_controller
        current = (32, 32)
        controller.dependencies = replace(
            controller.dependencies,
            get_drone_positions=lambda: (
                (0, current),
                (1, (60, 32)),
            ),
        )
        wall_region = _GlobalFrontierRegion(
            tiles=(
                _GlobalFrontierTile(
                    target=(32, 20),
                    size=20,
                    wall_cells=20,
                ),
                _GlobalFrontierTile(
                    target=(60, 32),
                    size=20,
                    wall_cells=20,
                ),
            ),
            size=40,
            wall_cells=40,
            tile_count=2,
        )

        selection = controller._select_global_frontier_region(
            (wall_region,),
            current=current,
            heading=90.0,
        )

        self.assertEqual(selection.position, (60, 32))
        self.assertLess(selection.ownership_margin, 0.0)

    def test_global_ownership_batches_peer_geometry(self) -> None:
        controller = self.drone.movement_controller
        current = (32, 32)
        controller.dependencies = replace(
            controller.dependencies,
            get_drone_positions=lambda: (
                (0, current),
                (1, (60, 32)),
                (2, (4, 32)),
            ),
        )
        tiles = tuple(
            _GlobalFrontierTile(
                target=(x, y),
                size=1,
                wall_cells=1,
            )
            for y in range(0, 64, 4)
            for x in range(0, 64, 4)
        )
        wall_region = _GlobalFrontierRegion(
            tiles=tiles,
            size=len(tiles),
            wall_cells=len(tiles),
            tile_count=len(tiles),
        )

        with patch(
            "agents.drone_movement.math.dist",
            wraps=math.dist,
        ) as scalar_distance:
            selection = controller._select_global_frontier_region(
                (wall_region,),
                current=current,
                heading=90.0,
            )

        self.assertIs(selection.region, wall_region)
        self.assertEqual(scalar_distance.call_count, 1)

    def test_peer_positions_change_global_frontier_selection(self) -> None:
        controller = self.drone.movement_controller
        current = (32, 32)
        regions = (
            global_region((4, 32)),
            global_region((60, 32)),
        )

        controller.dependencies = replace(
            controller.dependencies,
            get_drone_positions=lambda: (
                (0, current),
                (1, (8, 32)),
            ),
        )
        peer_on_left = controller._select_global_frontier_region(
            regions,
            current=current,
            heading=0.0,
        )
        controller.dependencies = replace(
            controller.dependencies,
            get_drone_positions=lambda: (
                (0, current),
                (1, (56, 32)),
            ),
        )
        peer_on_right = controller._select_global_frontier_region(
            regions,
            current=current,
            heading=0.0,
        )

        self.assertEqual(peer_on_left.position, (60, 32))
        self.assertEqual(peer_on_right.position, (4, 32))

    def test_overlapping_launch_positions_split_by_drone_sector(self) -> None:
        controller = self.drone.movement_controller
        current = self.drone.start_pos
        regions = (
            global_region((16, 0)),
            global_region((32, 16)),
        )
        positions = lambda: (
            (0, current),
            (1, current),
        )
        controller.dependencies = replace(
            controller.dependencies,
            get_drone_positions=positions,
        )

        drone_zero = controller._select_global_frontier_region(
            regions,
            current=current,
            heading=45.0,
        )
        self.drone.id = 1
        drone_one = controller._select_global_frontier_region(
            regions,
            current=current,
            heading=45.0,
        )

        self.assertEqual(drone_zero.position, (16, 0))
        self.assertEqual(drone_one.position, (32, 16))
        self.assertGreater(drone_zero.launch_sector_alignment, 0.99)
        self.assertGreater(drone_one.launch_sector_alignment, 0.9)

    def test_out_of_sector_global_target_remains_eligible(self) -> None:
        controller = self.drone.movement_controller
        current = self.drone.start_pos
        controller.dependencies = replace(
            controller.dependencies,
            get_drone_positions=lambda: (
                (0, current),
                (1, current),
            ),
        )
        only_region = global_region((16, 32))

        selection = controller._select_global_frontier_region(
            (only_region,),
            current=current,
            heading=0.0,
        )

        self.assertIs(selection.region, only_region)
        self.assertEqual(selection.position, (16, 32))
        self.assertAlmostEqual(selection.launch_sector_alignment, 0.0)

    def test_shared_slam_preserves_global_cache_until_cadence(self) -> None:
        controller = self.drone.movement_controller
        clock = [10.0]
        controller.dependencies = replace(
            controller.dependencies,
            simulation_time=lambda: clock[0],
        )
        first = controller._ensure_global_frontier_cache(
            current=(16, 16),
            heading=0.0,
        )
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[20, 20] = FREE
        confidence[20, 20] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        self.drone.runtime_state.begin_exploration(0)

        controller.mark_shared_slam_changed()
        controller._refresh_frontiers_before_mission_state()
        clock[0] = 11.0
        throttled = controller._ensure_global_frontier_cache(
            current=(16, 16),
            heading=0.0,
        )
        clock[0] = 12.1
        refreshed = controller._ensure_global_frontier_cache(
            current=(16, 16),
            heading=0.0,
        )

        self.assertEqual(self.drone.snapshot().frontiers, ((20, 20),))
        self.assertIs(controller._global_frontier_cache, refreshed)
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
        self.drone.exploration_policy = FixedDirectionPolicy(180)
        controller._global_frontier_cache = None
        with patch.object(
            controller,
            "_ensure_global_frontier_cache",
            wraps=controller._ensure_global_frontier_cache,
        ) as rebuild_global:
            controller.move()

        completed = next(
            fields for name, fields in trace.events
            if name == "drone_stagnation_scan_completed"
        )
        self.assertGreaterEqual(completed["sensor_newly_known_cells"], 1)
        self.assertFalse(completed["frontier_suppressed"])
        self.assertEqual(self.drone.snapshot().heading_deg, 0.0)
        exit_event = next(
            fields for name, fields in trace.events
            if name == "drone_stagnation_scan_exit_reoriented"
        )
        self.assertTrue(exit_event["exact_resume"])
        self.assertEqual(exit_event["resume_heading"], 0.0)
        self.assertEqual(exit_event["direction"], 0)
        self.assertEqual(
            rebuild_global.call_args.kwargs["heading"],
            0.0,
        )

    def test_scan_request_uses_last_fully_published_sensor_sequence(
        self,
    ) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        self.drone.runtime_state.reorient(90)
        self.drone.exploration_policy = FixedDirectionPolicy(90)
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        occupancy[16, 17] = UNKNOWN
        confidence[16, 17] = 0.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))

        slam_advanced = threading.Event()
        publish_completion = threading.Event()

        def finish_in_flight_scan() -> None:
            self.drone.slam_map.update_from_observations(
                (16, 16),
                free_cells=(),
                occupied_cells=(),
            )
            sequence = (
                self.drone.slam_map.progress_snapshot()
                .completed_scan_sequence
            )
            slam_advanced.set()
            if not publish_completion.wait(timeout=2.0):
                return
            self.drone.sensor_controller._last_completed_scan = (
                SensorScanCompletion(
                    pose=(16, 16, 90.0),
                    sequence=sequence,
                    newly_known_cells=0,
                    confidence_gain=0.0,
                )
            )

        worker = threading.Thread(target=finish_in_flight_scan)
        worker.start()
        self.assertTrue(slam_advanced.wait(timeout=2.0))
        try:
            self.assertIsNone(
                self.drone.sensor_controller.last_completed_scan
            )
            self.assertTrue(controller._start_frontier_scan(
                ((16, 16),),
                reason="test_split_scan_publication",
            ))
            pending = controller._pending_frontier_scan
            self.assertIsNotNone(pending)
            assert pending is not None
            self.assertEqual(pending.minimum_scan_sequence, 0)
            started = next(
                fields for name, fields in trace.events
                if name == "drone_stagnation_scan_started"
            )
            self.assertEqual(started["published_scan_sequence"], 0)
            self.assertEqual(started["slam_completed_scan_sequence"], 1)
        finally:
            publish_completion.set()
            worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        controller.move()

        self.assertIsNone(controller._pending_frontier_scan)
        completed = next(
            fields for name, fields in trace.events
            if name == "drone_stagnation_scan_completed"
        )
        self.assertEqual(completed["completed_scan_sequence"], 1)

    def test_pending_scan_timeout_restores_movement_heading(self) -> None:
        trace = RecordingTrace()
        clock = [10.0]
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
            simulation_time=lambda: clock[0],
        )
        self.drone.runtime_state.reorient(90)
        self.drone.exploration_policy = FixedDirectionPolicy(0)
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        occupancy[15, 16] = UNKNOWN
        confidence[15, 16] = 0.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        self.assertTrue(controller._start_frontier_scan(
            ((16, 16),),
            reason="test_timeout",
        ))
        pending = controller._pending_frontier_scan
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(self.drone.snapshot().heading_deg, 0.0)

        clock[0] = pending.deadline + 0.01
        controller.move()

        self.assertIsNone(controller._pending_frontier_scan)
        self.assertEqual(self.drone.snapshot().heading_deg, 90.0)
        timed_out = next(
            fields for name, fields in trace.events
            if name == "drone_stagnation_scan_timed_out"
        )
        self.assertEqual(timed_out["minimum_scan_sequence"], 0)
        self.assertIsNone(timed_out["published_scan_sequence"])
        self.assertGreaterEqual(timed_out["waited_seconds"], 3.0)

    def test_scan_exit_uses_closest_safe_pre_scan_heading(self) -> None:
        controller = self.drone.movement_controller
        self.drone.runtime_state.reorient(90)
        self.drone.exploration_policy = FixedDirectionPolicy(0)
        occupancy = np.full((64, 64), FREE, dtype=np.int8)
        confidence = np.ones((64, 64), dtype=np.float32)
        occupancy[15, 16] = UNKNOWN
        confidence[15, 16] = 0.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        controller.rebuild_frontiers(
            stride=1,
            confidence_threshold=controller.frontier_confidence_threshold,
        )
        self.assertTrue(controller._start_frontier_scan(
            ((16, 16),),
            reason="test_resume_fallback",
        ))
        pending = controller._pending_frontier_scan
        self.assertIsNotNone(pending)
        assert pending is not None

        with patch.object(
            controller,
            "_direction_candidates",
            return_value=([40, 200], [], {}),
        ):
            restored = controller._restore_movement_heading_after_scan(
                pending
            )

        self.assertTrue(restored)
        self.assertEqual(pending.resume_heading, 90.0)
        self.assertEqual(self.drone.snapshot().heading_deg, 40.0)

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

    def test_zero_gain_retirement_covers_the_sampled_component(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        occupancy = np.full((64, 64), UNKNOWN, dtype=np.int8)
        confidence = np.zeros((64, 64), dtype=np.float32)
        occupancy[8:40, 8:40] = FREE
        confidence[8:40, 8:40] = 1.0
        self.drone.slam_map.merge_from(SlamSnapshot(occupancy, confidence))
        controller.rebuild_frontiers(stride=4, confidence_threshold=0.6)
        target = (8, 8)
        frontier_y, frontier_x = np.where(controller._last_frontier_mask)
        rover_component = SectorFrontierComponent(
            component_id=7,
            cells=frozenset(
                (int(x), int(y))
                for y, x in zip(frontier_y, frontier_x)
            ),
        )
        controller._sector_assignment = SectorAssignment(
            sector_id=3,
            generation=1,
            owner_drone_id=0,
            cell_size=32,
            cells=frozenset({(0, 0), (1, 0), (0, 1), (1, 1)}),
            seed=(16, 16),
            gateway=(16, 16),
            frontier_cells=len(rover_component.cells),
            rover_slam_version=1,
            frontier_components=(rover_component,),
        )

        controller._suppress_frontier_target(
            target,
            reason="zero_gain_directed_scan",
            whole_component=True,
        )
        component = controller._frontier_component_targets[target]
        self.assertGreater(len(component), 1)

        self.assertTrue(set(component).issubset(
            controller._suppressed_frontier_geometry
        ))
        self.assertFalse(
            set(component) & set(self.drone.snapshot().frontiers)
        )
        event = next(
            fields for name, fields in trace.events
            if name == "drone_border_target_suppressed"
        )
        self.assertEqual(event["suppressed_target_count"], len(component))
        self.assertEqual(event["assigned_component_id"], 7)
        report = controller.sector_outcome_report(3)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(len(report.suppressions), 1)
        self.assertEqual(report.suppressions[0].component_id, 7)
        self.assertEqual(
            report.suppressions[0].reasons,
            ("zero_gain_directed_scan",),
        )
        self.assertEqual(
            report.suppressions[0].sampled_target_count,
            len(component),
        )

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

    def test_border_escape_rejects_excessively_circuitous_route(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        target = (40, 16)
        self.drone.runtime_state.replace_frontiers((target,))
        self.control.paths[((16, 16), target)] = [
            (16, 16),
            (16, 56),
            (56, 56),
            (56, 16),
            target,
        ]

        reached = controller.reach_border()

        self.assertFalse(reached)
        self.assertEqual(self.drone.snapshot().position, (16, 16))
        self.assertIn(target, controller._suppressed_frontier_geometry)
        rejected = next(
            fields for name, fields in trace.events
            if name == "drone_frontier_route_rejected"
        )
        self.assertGreater(rejected["route_circuity"], 4.0)
        self.assertEqual(rejected["reason"], "excessive_path_circuity")

    def test_reached_frontier_emits_arrival_route_metrics(self) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )
        target = (32, 16)
        self.drone.runtime_state.replace_frontiers((target,))
        self.control.paths[((16, 16), target)] = [
            (16, 16),
            (24, 16),
            target,
        ]

        self.assertTrue(controller.reach_border())

        arrived = next(
            fields for name, fields in trace.events
            if name == "drone_frontier_reached"
        )
        self.assertEqual(arrived["target"], target)
        self.assertEqual(arrived["position"], target)
        self.assertAlmostEqual(arrived["route_distance"], 16.0)
        self.assertAlmostEqual(arrived["direct_distance"], 16.0)
        self.assertAlmostEqual(arrived["route_circuity"], 1.0)

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

    def test_path_trace_counts_coverage_revisits_and_repeated_edges(
        self,
    ) -> None:
        trace = RecordingTrace()
        controller = self.drone.movement_controller
        controller.dependencies = replace(
            controller.dependencies,
            runtime_trace=trace,
        )

        controller._follow_path(((40, 16),), source="outbound")
        controller._follow_path(((16, 16),), source="return")

        motions = [
            fields for event, fields in trace.events
            if event == "drone_motion"
        ]
        self.assertEqual(motions[0]["coverage_cell_entries"], 1)
        self.assertEqual(motions[0]["coverage_new_cell_entries"], 1)
        self.assertEqual(motions[0]["coverage_repeated_edge_entries"], 0)
        self.assertEqual(motions[1]["coverage_cell_entries"], 1)
        self.assertEqual(motions[1]["coverage_revisit_entries"], 1)
        self.assertEqual(motions[1]["coverage_repeated_edge_entries"], 1)

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
