import threading
import unittest

from mission.pause_control import PauseCoordinator, SimulationClock


class PauseControlTests(unittest.TestCase):
    def test_simulation_clock_excludes_paused_wall_time(self) -> None:
        wall_time = [100.0]
        clock = SimulationClock(lambda: wall_time[0])

        wall_time[0] = 105.0
        clock.pause()
        self.assertEqual(clock.now(), 105.0)

        wall_time[0] = 1005.0
        self.assertEqual(clock.now(), 105.0)

        clock.resume()
        wall_time[0] = 1010.0
        self.assertEqual(clock.now(), 110.0)

    def test_worker_registering_during_pause_cannot_enter_simulation(self) -> None:
        stop_event = threading.Event()
        coordinator = PauseCoordinator(stop_event)
        coordinator.pause()
        entered = threading.Event()

        def worker() -> None:
            if coordinator.register_current_worker("late-worker"):
                entered.set()
            coordinator.unregister_current_worker()

        thread = threading.Thread(target=worker)
        thread.start()

        self.assertFalse(entered.wait(0.05))
        coordinator.resume()
        self.assertTrue(entered.wait(2.0))
        thread.join(2.0)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
