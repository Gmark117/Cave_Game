"""Smoothed performance telemetry for the mission frame loop."""

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class FrameTimingSnapshot:
    """Read-only frame timing values expressed in milliseconds."""

    sample_count: int
    fps: float
    frame_ms: float
    work_ms: float
    wait_ms: float
    stages_ms: Mapping[str, float]


class FrameProfiler:
    """Maintain exponentially smoothed frame and stage durations."""

    def __init__(self, smoothing: float = 0.15) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("Frame timing smoothing must be in (0, 1]")

        self.smoothing = float(smoothing)
        self.sample_count = 0
        self._frame_seconds = 0.0
        self._wait_seconds = 0.0
        self._stage_seconds: Dict[str, float] = {}

    def record(
        self,
        frame_seconds: float,
        wait_seconds: float,
        stages: Mapping[str, float],
    ) -> None:
        """Record one frame and update the smoothed timing values."""
        frame_value = max(0.0, float(frame_seconds))
        wait_value = min(frame_value, max(0.0, float(wait_seconds)))
        stage_values = {
            name: max(0.0, float(duration))
            for name, duration in stages.items()
        }

        if self.sample_count == 0:
            self._frame_seconds = frame_value
            self._wait_seconds = wait_value
            self._stage_seconds = stage_values
        else:
            self._frame_seconds = self._smooth(
                self._frame_seconds,
                frame_value,
            )
            self._wait_seconds = self._smooth(
                self._wait_seconds,
                wait_value,
            )
            for name, value in stage_values.items():
                previous = self._stage_seconds.get(name, value)
                self._stage_seconds[name] = self._smooth(previous, value)

        self.sample_count += 1

    def snapshot(self) -> FrameTimingSnapshot:
        """Return the current smoothed timing values."""
        frame_ms = self._frame_seconds * 1000.0
        wait_ms = self._wait_seconds * 1000.0
        fps = 0.0 if self._frame_seconds <= 0.0 else 1.0 / self._frame_seconds
        return FrameTimingSnapshot(
            sample_count=self.sample_count,
            fps=fps,
            frame_ms=frame_ms,
            work_ms=max(0.0, frame_ms - wait_ms),
            wait_ms=wait_ms,
            stages_ms={
                name: duration * 1000.0
                for name, duration in self._stage_seconds.items()
            },
        )

    def _smooth(self, previous: float, current: float) -> float:
        return previous + self.smoothing * (current - previous)
