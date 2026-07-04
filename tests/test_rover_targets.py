import threading
import unittest
from types import SimpleNamespace

import numpy as np

from mapping.rover_targets import RoverTargetService
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
        self.service = RoverTargetService(
            RoverTargetDependencies(
                cave_map=self.control.map_matrix,
                terrain_knowledge=self.control.terrain_knowledge,
                assignment_lock=self.control.rover_assignment_lock,
                assignments=self.control.rover_assignments,
                completed_targets=self.control.completed_rover_targets,
                norm_width=self.control.game.width,
                norm_height=self.control.game.height,
            )
        )

    def test_acquire_reserves_best_known_rough_target(self) -> None:
        self.control.terrain_knowledge.roughness[1, 1] = 0.5
        self.control.terrain_knowledge.confidence[1, 1] = 0.8
        self.control.terrain_knowledge.roughness[3, 3] = 0.9
        self.control.terrain_knowledge.confidence[3, 3] = 1.0

        target = self.service.acquire(0, (3, 2))

        self.assertEqual(target, (3, 3))
        self.assertEqual(self.control.rover_assignments[0], (3, 3))

    def test_assignments_and_completed_targets_are_not_reused(self) -> None:
        self.control.terrain_knowledge.roughness[1, 1] = 0.9
        self.control.terrain_knowledge.confidence[1, 1] = 1.0
        self.control.terrain_knowledge.roughness[2, 2] = 0.8
        self.control.terrain_knowledge.confidence[2, 2] = 1.0
        self.control.rover_assignments[0] = (1, 1)

        target = self.service.acquire(1, (1, 1))
        self.assertEqual(target, (2, 2))

        self.service.release(1, completed=True)
        self.assertIn((2, 2), self.control.completed_rover_targets)
        self.assertNotIn(1, self.control.rover_assignments)

    def test_returns_none_without_qualified_terrain(self) -> None:
        self.control.terrain_knowledge.roughness[1, 1] = 0.9
        self.control.terrain_knowledge.confidence[1, 1] = 0.1

        self.assertIsNone(self.service.acquire(0, (0, 0)))


if __name__ == "__main__":
    unittest.main()
