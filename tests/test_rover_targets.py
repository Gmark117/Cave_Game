import threading
import unittest
from types import SimpleNamespace

import numpy as np

from mapping.rover_targets import RoverFrontierTarget, RoverTargetService
from mapping.terrain_knowledge import TerrainKnowledge
from contracts import RoverTargetDependencies


class RoverTargetServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        cave = np.zeros((4, 4), dtype=np.uint8)
        terrain_knowledge = TerrainKnowledge(cave)
        self.control = SimpleNamespace(
            map_matrix=cave,
            terrain_knowledge=terrain_knowledge,
            rover_assignments={},
            completed_rover_targets=set(),
            game=SimpleNamespace(width=4, height=4),
        )
        self.control.rover_assignment_lock = threading.Lock()
        self.control.frontier_candidates = []
        self.control.hold_rovers = set()
        self.service = RoverTargetService(
            RoverTargetDependencies(
                cave_map=self.control.map_matrix,
                terrain_knowledge=self.control.terrain_knowledge,
                assignment_lock=self.control.rover_assignment_lock,
                assignments=self.control.rover_assignments,
                completed_targets=self.control.completed_rover_targets,
                norm_width=self.control.game.width,
                norm_height=self.control.game.height,
                get_frontier_candidates=lambda: tuple(
                    self.control.frontier_candidates
                ),
                should_hold_position=lambda rover_id: (
                    rover_id in self.control.hold_rovers
                ),
            )
        )

    def test_acquire_prefers_claimed_deep_frontier(self) -> None:
        self.control.frontier_candidates = [
            RoverFrontierTarget((1, 1), 1, 1, False, 0, 100.0),
            RoverFrontierTarget((3, 3), 2, 2, True, 2, 10.0),
        ]

        target = self.service.acquire(0, (3, 2))

        self.assertEqual(target, (3, 3))
        self.assertEqual(self.control.rover_assignments[0], (3, 3))

    def test_assignments_and_completed_targets_are_not_reused(self) -> None:
        self.control.frontier_candidates = [
            RoverFrontierTarget((1, 1), 1, 1, True, 1, 5.0),
            RoverFrontierTarget((2, 2), 2, 2, True, 1, 4.0),
        ]
        self.control.rover_assignments[0] = (1, 1)

        target = self.service.acquire(1, (1, 1))
        self.assertEqual(target, (2, 2))

        self.service.release(1, completed=True)
        self.assertIn((2, 2), self.control.completed_rover_targets)
        self.assertNotIn(1, self.control.rover_assignments)

    def test_returns_none_without_frontier_candidates(self) -> None:
        self.assertIsNone(self.service.acquire(0, (0, 0)))

    def test_claimed_targets_use_service_cost_then_asperity(self) -> None:
        self.control.frontier_candidates = [
            RoverFrontierTarget(
                (1, 1), 1, 1, True, 1, 5.0,
                service_cost=10.0,
                rover_route_cost=2.0,
                terrain_roughness=0.1,
                wall_clearance=4.0,
            ),
            RoverFrontierTarget(
                (2, 2), 2, 2, True, 1, 5.0,
                service_cost=4.0,
                rover_route_cost=3.0,
                terrain_roughness=0.8,
                wall_clearance=2.0,
            ),
        ]

        self.assertEqual(self.service.acquire(0, (0, 0)), (2, 2))

        self.service.release(0)
        self.control.frontier_candidates = [
            RoverFrontierTarget(
                (1, 1), 1, 1, True, 1, 5.0,
                service_cost=4.0,
                terrain_roughness=0.2,
                wall_clearance=2.0,
            ),
            RoverFrontierTarget(
                (2, 2), 2, 2, True, 1, 5.0,
                service_cost=4.0,
                terrain_roughness=0.7,
                wall_clearance=6.0,
            ),
        ]
        self.assertEqual(self.service.acquire(0, (0, 0)), (1, 1))

    def test_hold_signal_is_rover_specific(self) -> None:
        self.control.hold_rovers.add(0)

        self.assertTrue(self.service.should_hold(0))
        self.assertFalse(self.service.should_hold(1))

    def test_nearby_lineage_child_requests_new_staging_endpoint(self) -> None:
        self.control.frontier_candidates = [
            RoverFrontierTarget((10, 10), 4, 7, True, 1, 5.0),
        ]
        self.assertEqual(self.service.acquire(0, (0, 0)), (10, 10))
        self.control.frontier_candidates = [
            RoverFrontierTarget(
                (14, 12),
                8,
                9,
                True,
                2,
                4.0,
                parent_component_ids=(4,),
            ),
        ]

        self.assertFalse(self.service.target_is_current(0))

    def test_far_lineage_progress_requests_new_staging_route(self) -> None:
        self.control.frontier_candidates = [
            RoverFrontierTarget((10, 10), 4, 7, True, 1, 5.0),
        ]
        self.service.acquire(0, (0, 0))
        self.control.frontier_candidates = [
            RoverFrontierTarget(
                (100, 10),
                8,
                9,
                True,
                2,
                4.0,
                parent_component_ids=(4,),
            ),
        ]

        self.assertFalse(self.service.target_is_current(0))


if __name__ == "__main__":
    unittest.main()
