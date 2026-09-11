import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from mapping.exploration_sectors import (
    ExplorationSectorSnapshot,
    SectorAssignment,
)
from rendering.sector_renderer import SectorRenderer


class SectorRendererTests(unittest.TestCase):
    @staticmethod
    def snapshot(*, waiting=frozenset()):
        return ExplorationSectorSnapshot(
            generation=2,
            assignments=(
                SectorAssignment(
                    sector_id=6,
                    generation=2,
                    owner_drone_id=0,
                    cell_size=16,
                    cells=frozenset({(0, 0), (1, 0)}),
                    seed=(8, 8),
                    gateway=(31, 31),
                    frontier_cells=12,
                    rover_slam_version=4,
                ),
                SectorAssignment(
                    sector_id=7,
                    generation=2,
                    owner_drone_id=1,
                    cell_size=16,
                    cells=frozenset({(0, 1), (1, 1)}),
                    seed=(8, 24),
                    gateway=(31, 31),
                    frontier_cells=10,
                    rover_slam_version=4,
                ),
            ),
            waiting_drone_ids=waiting,
            mission_exhausted=False,
        )

    def test_draw_tints_owner_cells_and_marks_boundaries(self) -> None:
        renderer = SectorRenderer(32, 32)
        window = pygame.Surface((32, 32), pygame.SRCALPHA)

        drawn = renderer.draw(
            window,
            self.snapshot(),
            {0: (255, 0, 0), 1: (0, 255, 255)},
        )

        self.assertTrue(drawn)
        self.assertEqual(renderer.surface.get_at((24, 8)).a, 24)
        self.assertEqual(renderer.surface.get_at((24, 24)).a, 24)
        self.assertGreater(renderer.surface.get_at((24, 15)).a, 24)

    def test_waiting_sector_is_dimmed(self) -> None:
        renderer = SectorRenderer(32, 32)
        window = pygame.Surface((32, 32), pygame.SRCALPHA)

        renderer.draw(
            window,
            self.snapshot(waiting=frozenset({0})),
            {0: (255, 0, 0), 1: (0, 255, 255)},
        )

        self.assertEqual(renderer.surface.get_at((24, 8)).a, 8)
        self.assertEqual(renderer.surface.get_at((24, 24)).a, 24)

    def test_empty_snapshot_does_not_blit(self) -> None:
        renderer = SectorRenderer(32, 32)
        window = pygame.Surface((32, 32), pygame.SRCALPHA)
        empty = ExplorationSectorSnapshot(-1, (), frozenset(), False)

        self.assertFalse(renderer.draw(window, empty, {}))

    def test_component_snapshot_draws_work_without_territory_fill(self) -> None:
        renderer = SectorRenderer(32, 32)
        window = pygame.Surface((32, 32), pygame.SRCALPHA)
        snapshot = SimpleNamespace(
            revision=3,
            components=(),
            work_units=(SimpleNamespace(
                work_unit_id=9,
                state="claimed",
                cells=frozenset({(10, 10), (11, 10)}),
                anchor_position=(10, 10),
            ),),
            claims=(SimpleNamespace(
                owner_drone_id=0,
                work_unit_ids=(9,),
            ),),
        )

        self.assertTrue(renderer.draw(window, snapshot, {0: (255, 0, 0)}))
        self.assertGreater(renderer.surface.get_at((10, 10)).a, 0)
        self.assertEqual(renderer.surface.get_at((25, 25)).a, 0)


if __name__ == "__main__":
    unittest.main()
