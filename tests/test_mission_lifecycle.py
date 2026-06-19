import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from Game import Game
from MissionControl import MissionControl
from mapping.terrain_knowledge import TerrainKnowledge
from navigation.pathfinding import PathfindingService
from rendering.mission_renderer import MissionRenderer


class FakeGame:
    def __init__(self) -> None:
        self.sim_settings = SimpleNamespace(
            seed=7,
            mission=0,
            num_drones=3,
            map_dim="SMALL",
        )
        cave = np.zeros((8, 8), dtype=np.uint8)
        self.cartographer = SimpleNamespace(
            bin_map=cave,
            terrain_roughness=np.zeros_like(cave, dtype=np.float32),
            worm_x=[2],
            worm_y=[2],
        )
        self.width = 1200
        self.height = 750
        self.display = None
        self.window = None
        self.maximise_calls = 0
        self.windowed_calls = 0

    def to_maximised(self):
        self.maximise_calls += 1
        return self.display

    def to_windowed(self):
        self.windowed_calls += 1
        return self.display


class MissionLifecycleTests(unittest.TestCase):
    def test_construction_does_not_allocate_runtime_resources(self) -> None:
        game = FakeGame()

        with patch(
            "navigation.pathfinding.shared_memory.SharedMemory"
        ) as shared_memory:
            with patch(
                "navigation.pathfinding.ProcessPoolExecutor"
            ) as process_pool:
                mission = MissionControl(game)

        shared_memory.assert_not_called()
        process_pool.assert_not_called()
        self.assertEqual(game.maximise_calls, 0)
        self.assertEqual(mission.drones, [])
        self.assertEqual(mission.rovers, [])
        self.assertIsNone(mission.control_center)
        self.assertIsNone(mission.pool)
        self.assertIsNone(mission.map_shm)
        self.assertIsInstance(mission.terrain_knowledge, TerrainKnowledge)
        self.assertIs(
            mission.known_roughness,
            mission.terrain_knowledge.roughness,
        )
        self.assertIs(
            mission.terrain_confidence,
            mission.terrain_knowledge.confidence,
        )
        self.assertIs(mission.terrain_lock, mission.terrain_knowledge.lock)
        self.assertFalse(hasattr(mission, "last_pair_share"))
        self.assertFalse(hasattr(mission, "pair_share_cooldown"))
        self.assertEqual(mission.terrain_sharing.last_drone_share, {})
        self.assertEqual(mission.terrain_sharing.last_pair_share, {})
        self.assertIsInstance(mission.pathfinding, PathfindingService)
        self.assertIsInstance(mission.renderer, MissionRenderer)
        self.assertIs(
            mission.stop_button_rect,
            mission.renderer.stop_button_rect,
        )
        self.assertIs(
            mission.restart_button_rect,
            mission.renderer.restart_button_rect,
        )
        self.assertIs(
            mission.pause_button_rect,
            mission.renderer.pause_button_rect,
        )
        self.assertFalse(mission.restart_requested)
        self.assertFalse(mission.is_paused)
        self.assertTrue(mission.pause_event.is_set())
        self.assertFalse(mission.rover_motion_enabled)
        self.assertFalse(mission._runtime_initialized)
        self.assertFalse(mission._has_run)
        self.assertEqual(mission.compute_path((0, 0), (1, 1)), [])
        self.assertFalse(mission.is_mission_over())
        with self.assertRaises(RuntimeError):
            mission.draw()

        mission.pathfinding.shutdown = Mock()
        mission._shutdown_mission([])
        mission.pathfinding.shutdown.assert_called_once_with()
        self.assertTrue(mission.mission_event.is_set())

    def test_pathfinding_methods_are_compatibility_facades(self) -> None:
        mission = MissionControl(FakeGame())
        mission.pathfinding.compute_path = Mock(
            return_value=[(0, 0), (1, 1)],
        )
        mission.pathfinding.compute_weighted_path = Mock(
            return_value=[(0, 0), (0, 1)],
        )

        drone_path = mission.compute_path((0, 0), (1, 1))
        rover_path = mission.compute_rover_path((0, 0), (0, 1))

        self.assertEqual(drone_path, [(0, 0), (1, 1)])
        self.assertEqual(rover_path, [(0, 0), (0, 1)])
        mission.pathfinding.compute_path.assert_called_once_with(
            (0, 0),
            (1, 1),
        )
        weighted_args = (
            mission.pathfinding.compute_weighted_path.call_args.args
        )
        np.testing.assert_array_equal(
            weighted_args[0],
            mission.known_roughness,
        )
        np.testing.assert_array_equal(
            weighted_args[1],
            mission.terrain_confidence,
        )
        self.assertEqual(weighted_args[2:], ((0, 0), (0, 1)))

    def test_game_constructs_then_runs_mission(self) -> None:
        game = object.__new__(Game)
        settings = SimpleNamespace(seed=11)
        game.menu = SimpleNamespace(
            build_sim_settings=Mock(return_value=settings),
        )
        cartographer = object()
        mission = Mock()

        with patch("Game.MapGenerator", return_value=cartographer) as generator:
            with patch("Game.MissionControl", return_value=mission) as control:
                game.start_mission()

        generator.assert_called_once_with(game, settings)
        control.assert_called_once_with(game)
        mission.run.assert_called_once_with()
        self.assertIs(game.cartographer, cartographer)
        self.assertIs(game.mission_control, mission)

    def test_game_restart_reuses_settings_and_generated_cave(self) -> None:
        game = object.__new__(Game)
        settings = SimpleNamespace(seed=11)
        game.menu = SimpleNamespace(
            build_sim_settings=Mock(return_value=settings),
        )
        cartographer = object()
        first_mission = SimpleNamespace(
            run=Mock(),
            restart_requested=True,
        )
        second_mission = SimpleNamespace(
            run=Mock(),
            restart_requested=False,
        )

        with patch("Game.MapGenerator", return_value=cartographer) as generator:
            with patch(
                "Game.MissionControl",
                side_effect=[first_mission, second_mission],
            ) as control:
                game.start_mission()

        generator.assert_called_once_with(game, settings)
        self.assertEqual(control.call_count, 2)
        control.assert_any_call(game)
        first_mission.run.assert_called_once_with()
        second_mission.run.assert_called_once_with()
        self.assertIs(game.cartographer, cartographer)
        self.assertIs(game.mission_control, second_mission)

    def test_run_initializes_executes_and_shuts_down_once(self) -> None:
        mission = MissionControl(FakeGame())
        control_center = SimpleNamespace(start_timer=Mock())

        def initialize_runtime() -> None:
            mission.control_center = control_center
            mission._runtime_initialized = True

        mission._initialize_runtime = Mock(side_effect=initialize_runtime)
        mission._start_agent_threads = Mock(return_value=[])
        mission._run_mission_loop = Mock()
        mission._shutdown_mission = Mock()

        with patch("MissionControlLifecycle.pygame.get_init", return_value=False):
            mission.run()

        mission._initialize_runtime.assert_called_once_with()
        control_center.start_timer.assert_called_once_with()
        mission._start_agent_threads.assert_called_once_with()
        mission._run_mission_loop.assert_called_once_with()
        mission._shutdown_mission.assert_called_once_with([])
        self.assertFalse(mission._running)
        self.assertTrue(mission._has_run)

        with self.assertRaises(RuntimeError):
            mission.run()

    def test_stop_button_ends_loop_before_simulation_updates(self) -> None:
        mission = MissionControl(FakeGame())
        mission.clock = SimpleNamespace(tick=Mock())
        mission.completed = False
        mission.renderer.draw = Mock()
        mission.update_sensors = Mock()
        mission._share_terrain_with_rovers = Mock()
        mission.stop_button_rect.x = 0
        mission.stop_button_rect.y = 0
        event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=(1, 1),
        )

        with patch(
            "MissionControlLifecycle.pygame.event.get",
            return_value=[event],
        ):
            mission._run_mission_loop()

        self.assertTrue(mission.completed)
        mission._share_terrain_with_rovers.assert_not_called()
        mission.update_sensors.assert_not_called()
        mission.renderer.draw.assert_not_called()

    def test_restart_button_requests_fresh_mission_before_updates(self) -> None:
        mission = MissionControl(FakeGame())
        mission.clock = SimpleNamespace(tick=Mock())
        mission.completed = False
        mission.renderer.draw = Mock()
        mission.update_sensors = Mock()
        mission._share_terrain_with_rovers = Mock()
        event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=mission.restart_button_rect.center,
        )

        with patch(
            "MissionControlLifecycle.pygame.event.get",
            return_value=[event],
        ):
            mission._run_mission_loop()

        self.assertTrue(mission.completed)
        self.assertTrue(mission.restart_requested)
        mission._share_terrain_with_rovers.assert_not_called()
        mission.update_sensors.assert_not_called()
        mission.renderer.draw.assert_not_called()

    def test_pause_button_toggles_state_and_skips_simulation_updates(self) -> None:
        mission = MissionControl(FakeGame())
        mission.clock = SimpleNamespace(tick=Mock())
        mission.completed = False
        mission.control_center = SimpleNamespace(
            pause_timer=Mock(),
            resume_timer=Mock(),
        )
        mission.renderer.draw = Mock()
        mission.update_sensors = Mock()
        mission._share_terrain_with_rovers = Mock()
        mission.is_mission_over = Mock()
        pause_event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=mission.pause_button_rect.center,
        )
        stop_event = SimpleNamespace(
            type=pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=mission.stop_button_rect.center,
        )

        with patch(
            "MissionControlLifecycle.pygame.event.get",
            side_effect=[[pause_event], [stop_event]],
        ):
            with patch("MissionControlLifecycle.pygame.display.update"):
                mission._run_mission_loop()

        self.assertTrue(mission.is_paused)
        self.assertFalse(mission.pause_event.is_set())
        mission.control_center.pause_timer.assert_called_once_with()
        mission.control_center.resume_timer.assert_not_called()
        mission._share_terrain_with_rovers.assert_not_called()
        mission.is_mission_over.assert_not_called()
        mission.update_sensors.assert_not_called()
        mission.renderer.draw.assert_called_once_with()

    def test_pause_toggle_resumes_agents_and_timer(self) -> None:
        mission = MissionControl(FakeGame())
        mission.control_center = SimpleNamespace(
            pause_timer=Mock(),
            resume_timer=Mock(),
        )

        mission.toggle_pause()
        mission.toggle_pause()

        self.assertFalse(mission.is_paused)
        self.assertTrue(mission.pause_event.is_set())
        mission.control_center.pause_timer.assert_called_once_with()
        mission.control_center.resume_timer.assert_called_once_with()

    def test_restart_run_does_not_return_to_windowed_mode(self) -> None:
        mission = MissionControl(FakeGame())
        control_center = SimpleNamespace(start_timer=Mock())

        def initialize_runtime() -> None:
            mission.control_center = control_center
            mission._runtime_initialized = True

        def request_restart() -> None:
            mission.restart_requested = True

        mission._initialize_runtime = Mock(side_effect=initialize_runtime)
        mission._start_agent_threads = Mock(return_value=[])
        mission._run_mission_loop = Mock(side_effect=request_restart)
        mission._shutdown_mission = Mock()

        with patch("MissionControlLifecycle.pygame.get_init", return_value=True):
            mission.run()

        self.assertEqual(mission.game.windowed_calls, 0)

    def test_run_loop_records_frame_stage_timings(self) -> None:
        mission = MissionControl(FakeGame())
        mission.clock = SimpleNamespace(tick=Mock())
        mission.completed = False
        mission.renderer.draw = Mock()
        mission.update_sensors = Mock()
        mission._share_terrain_with_rovers = Mock()
        mission.is_mission_over = Mock(return_value=True)

        timestamps = [
            0.000,
            0.010,
            0.012,
            0.020,
            0.021,
            0.030,
            0.050,
            0.052,
        ]
        with patch(
            "MissionControlLifecycle.pygame.event.get",
            return_value=[],
        ):
            with patch("MissionControlLifecycle.pygame.display.update"):
                with patch(
                    "MissionControlLifecycle.time.perf_counter",
                    side_effect=timestamps,
                ):
                    mission._run_mission_loop()

        timing = mission.frame_profiler.snapshot()
        self.assertEqual(timing.sample_count, 1)
        self.assertAlmostEqual(timing.frame_ms, 52.0)
        self.assertAlmostEqual(timing.wait_ms, 10.0)
        self.assertAlmostEqual(timing.stages_ms["events"], 2.0)
        self.assertAlmostEqual(timing.stages_ms["sharing"], 8.0)
        self.assertAlmostEqual(timing.stages_ms["sensors"], 9.0)
        self.assertAlmostEqual(timing.stages_ms["render"], 20.0)


if __name__ == "__main__":
    unittest.main()
