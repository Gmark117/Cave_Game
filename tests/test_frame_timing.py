import unittest

from mission.frame_timing import FrameProfiler


class FrameProfilerTests(unittest.TestCase):
    def test_record_smooths_frame_wait_and_stage_timings(self) -> None:
        profiler = FrameProfiler(smoothing=0.5)

        profiler.record(
            frame_seconds=0.1,
            wait_seconds=0.04,
            stages={"sharing": 0.01, "render": 0.03},
        )
        profiler.record(
            frame_seconds=0.2,
            wait_seconds=0.06,
            stages={"sharing": 0.03, "render": 0.05},
        )

        timing = profiler.snapshot()
        self.assertEqual(timing.sample_count, 2)
        self.assertAlmostEqual(timing.frame_ms, 150.0)
        self.assertAlmostEqual(timing.wait_ms, 50.0)
        self.assertAlmostEqual(timing.work_ms, 100.0)
        self.assertAlmostEqual(timing.fps, 1.0 / 0.15)
        self.assertAlmostEqual(timing.stages_ms["sharing"], 20.0)
        self.assertAlmostEqual(timing.stages_ms["render"], 40.0)

    def test_rejects_invalid_smoothing(self) -> None:
        with self.assertRaises(ValueError):
            FrameProfiler(smoothing=0.0)


if __name__ == "__main__":
    unittest.main()
