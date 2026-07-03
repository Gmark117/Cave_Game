import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from generation.worm_process_runner import WormProcessRunner


class WormProcessRunnerTests(unittest.TestCase):
    def test_run_delegates_workers_copies_result_and_cleans_up(self) -> None:
        runner = WormProcessRunner()
        initial_map = np.ones((3, 4), dtype=np.float32)
        shm = SimpleNamespace(name="test-map")
        shared_result = np.zeros((3, 4), dtype=np.uint8)
        worker_finished = Mock()

        def monitor(_processes, callback):
            callback()
            callback()
            return False

        with patch("generation.worm_process_runner.os.cpu_count", return_value=2):
            with patch(
                "generation.worm_process_runner.safe_shm_create",
                return_value=(shm, shared_result),
            ):
                with patch(
                    "generation.worm_process_runner.start_worms",
                    return_value=["p0", "p1"],
                ) as start_worms:
                    with patch(
                        "generation.worm_process_runner.monitor_worms",
                        side_effect=monitor,
                    ):
                        with patch(
                            "generation.worm_process_runner.safe_shm_close"
                        ) as close:
                            result = runner.run(
                                initial_map,
                                4,
                                [1, 2],
                                [1, 2],
                                (3, 4, 5),
                                7,
                                np.random.default_rng(7),
                                worker_finished,
                            )

        self.assertEqual(result.completed_workers, 2)
        self.assertFalse(result.worker_crashed)
        worker_finished.assert_called()
        np.testing.assert_array_equal(result.bin_map, shared_result)
        self.assertEqual(start_worms.call_args.args[1], 2)
        close.assert_called_once_with(shm)


if __name__ == "__main__":
    unittest.main()
