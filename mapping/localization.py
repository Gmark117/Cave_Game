"""Pose-estimation boundary for SLAM localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


Position = Tuple[int, int]


@dataclass(frozen=True)
class PoseEstimate:
    """Estimated agent pose used by sensing and exploration decisions."""

    position: Position
    heading_deg: float
    confidence: float
    source: str
    timestamp: float

    def __post_init__(self) -> None:
        """Normalize position and validate confidence bounds."""
        x, y = self.position
        object.__setattr__(self, "position", (int(x), int(y)))
        object.__setattr__(self, "heading_deg", float(self.heading_deg))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("pose confidence must be between zero and one")


class PerfectPoseLocalizer:
    """Localizer that exposes the simulator pose as a perfect estimate."""

    source = "perfect-runtime"

    def estimate(
        self,
        runtime_snapshot: Any,
        timestamp: float,
    ) -> PoseEstimate:
        """Return a perfect pose estimate from a detached runtime snapshot."""
        return PoseEstimate(
            position=runtime_snapshot.position,
            heading_deg=runtime_snapshot.heading_deg,
            confidence=1.0,
            source=self.source,
            timestamp=timestamp,
        )
