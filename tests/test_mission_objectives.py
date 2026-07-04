import unittest
from types import SimpleNamespace

from mission.objectives import ExplorationObjective, build_mission_objective


class MissionObjectiveTests(unittest.TestCase):
    def test_exploration_completes_when_every_drone_is_done(self) -> None:
        objective = ExplorationObjective()

        self.assertFalse(objective.is_complete([], []))
        self.assertTrue(
            objective.is_complete(
                [
                    SimpleNamespace(mission_completed=lambda: True),
                    SimpleNamespace(mission_completed=lambda: True),
                ],
                [],
            )
        )
        self.assertFalse(
            objective.is_complete(
                [
                    SimpleNamespace(mission_completed=lambda: True),
                    SimpleNamespace(mission_completed=lambda: False),
                ],
                [],
            )
        )

    def test_planned_objective_fails_fast(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "Search and Rescue"):
            build_mission_objective(1)


if __name__ == "__main__":
    unittest.main()
