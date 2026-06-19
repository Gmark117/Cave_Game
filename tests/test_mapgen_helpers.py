import os
import unittest
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from MapGenHelpers import (
    add_wall_transition_noise,
    apply_cv_brush,
    border_control_helper,
    homing_helper,
    make_derangement,
    monitor_worms,
    remove_hermit_caves,
    safe_shm_close,
    safe_shm_create,
    start_worms,
    with_surfarrays,
)
from asset_config.mapgen import MapGen


class FixedRng:
    def integers(self, low, high=None):
        if low == 0 and high == 2:
            return 1
        return low


class MapGenHelperTests(unittest.TestCase):
    def test_brush_removes_wall_cells_without_adding_walls(self) -> None:
        cave = np.ones((9, 9), dtype=np.uint8)

        apply_cv_brush(cave, 4, 4, mode_choice=0, stren=6)

        self.assertEqual(int(cave[4, 4]), 0)
        self.assertGreater(int(np.count_nonzero(cave == 0)), 1)
        self.assertTrue(np.all((cave == 0) | (cave == 1)))

    def test_remove_hermit_caves_keeps_largest_floor_component(self) -> None:
        cave = np.ones((6, 6), dtype=np.uint8)
        cave[1:3, 1:3] = 0
        cave[4, 4] = 0

        cleaned = remove_hermit_caves(cave)

        self.assertTrue(np.all(cleaned[1:3, 1:3] == 0))
        self.assertEqual(int(cleaned[4, 4]), 1)

    def test_wall_transition_noise_is_seeded_and_binary(self) -> None:
        cave = np.ones((20, 20), dtype=np.uint8)
        cave[4:16, 4:16] = 0

        first = add_wall_transition_noise(cave, 20, 20, 11, (8, 4, 3))
        second = add_wall_transition_noise(cave, 20, 20, 11, (8, 4, 3))

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, cave.shape)
        self.assertTrue(np.all((first == 0) | (first == 1)))

    def test_derangement_has_no_fixed_points_and_is_reproducible(self) -> None:
        first = make_derangement(8, np.random.default_rng(17))
        second = make_derangement(8, np.random.default_rng(17))

        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(8)))
        self.assertTrue(all(index != value for index, value in enumerate(first)))
        self.assertEqual(make_derangement(1, np.random.default_rng(1)), [0])

    def test_homing_without_jitter_points_toward_target(self) -> None:
        heading = homing_helper(FixedRng(), 0, 0, 10, 0)
        self.assertEqual(heading, 90)

    def test_border_control_turns_worm_away_from_right_edge(self) -> None:
        heading = border_control_helper(
            FixedRng(),
            x1=90,
            x2=99,
            y1=40,
            y2=50,
            s=10,
            current_dir=0,
            new_dir=True,
            width=100,
            height=100,
            border_thickness=MapGen.BORDER_THICKNESS,
        )
        self.assertGreaterEqual(heading, 180)
        self.assertLess(heading, 360)

    def test_surfarray_context_releases_surface_lock(self) -> None:
        import pygame

        surface = pygame.Surface((3, 3), pygame.SRCALPHA)
        with with_surfarrays(surface) as (rgb, alpha):
            rgb[1, 1] = (10, 20, 30)
            alpha[1, 1] = 200
            self.assertFalse(surface.get_locked())

        self.assertFalse(surface.get_locked())
        self.assertEqual(surface.get_at((1, 1)), (10, 20, 30, 200))

    def test_shared_memory_helpers_copy_and_unlink_array(self) -> None:
        from multiprocessing import shared_memory

        source = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        shm, shared_map = safe_shm_create(source)
        name = shm.name
        try:
            np.testing.assert_array_equal(shared_map, source)
        finally:
            del shared_map
            safe_shm_close(shm)

        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)

    def test_start_worms_builds_and_starts_each_process(self) -> None:
        processes = [Mock(), Mock()]
        with patch("MapGenHelpers.Process", side_effect=processes) as process_cls:
            result = start_worms(
                "map",
                2,
                [1, 2],
                [3, 4],
                (5, 6, 7),
                10,
                [1, 0],
                20,
                30,
            )

        self.assertEqual(result, processes)
        self.assertEqual(process_cls.call_count, 2)
        for process in processes:
            process.start.assert_called_once_with()

    def test_monitor_worms_reports_finished_and_crashed_workers(self) -> None:
        healthy = SimpleNamespace(
            is_alive=Mock(return_value=False),
            join=Mock(),
            terminate=Mock(),
            exitcode=0,
        )
        crashed = SimpleNamespace(
            is_alive=Mock(return_value=False),
            join=Mock(),
            terminate=Mock(),
            exitcode=2,
        )
        callback = Mock()

        with redirect_stdout(io.StringIO()):
            with patch("MapGenHelpers.pygame.event.pump"):
                with patch("MapGenHelpers.time.sleep"):
                    any_crashed = monitor_worms(
                        [healthy, crashed],
                        callback,
                        poll_interval=0,
                    )

        self.assertTrue(any_crashed)
        self.assertEqual(callback.call_count, 2)
        healthy.terminate.assert_not_called()
        crashed.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
