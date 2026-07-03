"""Typed, immutable configuration for one Cave Explorer simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MissionConfig:
    """Mission selection and cave-generation inputs."""

    objective: int = 0
    map_dim: str = "MEDIUM"
    seed: int = 0
    num_drones: int = 3

    def __post_init__(self) -> None:
        if self.objective < 0:
            raise ValueError("objective must be non-negative")
        if not self.map_dim:
            raise ValueError("map_dim must not be empty")
        if self.num_drones <= 0:
            raise ValueError("num_drones must be positive")


@dataclass(frozen=True)
class SlamConfig:
    """Local occupancy sensing and point-cloud limits."""

    scan_interval: float = 0.25
    scan_rays: int = 60
    point_cloud_max_points: int = 6000

    def __post_init__(self) -> None:
        if self.scan_interval < 0.0:
            raise ValueError("scan_interval must be non-negative")
        if self.scan_rays <= 0:
            raise ValueError("scan_rays must be positive")
        if self.point_cloud_max_points <= 0:
            raise ValueError("point_cloud_max_points must be positive")


@dataclass(frozen=True)
class SharingConfig:
    """Agent sharing schedules and meaningful-difference thresholds."""

    drone_interval: float = 0.5
    pair_cooldown: float = 1.2
    rover_interval: float = 0.5
    compare_stride: int = 8
    min_new_info_ratio: float = 0.04
    min_overlap_diff_ratio: float = 0.18
    min_roughness_delta: float = 0.12

    def __post_init__(self) -> None:
        if min(self.drone_interval, self.pair_cooldown, self.rover_interval) < 0.0:
            raise ValueError("sharing intervals must be non-negative")
        if self.compare_stride <= 0:
            raise ValueError("compare_stride must be positive")
        if not 0.0 <= self.min_new_info_ratio <= 1.0:
            raise ValueError("min_new_info_ratio must be between zero and one")
        if not 0.0 <= self.min_overlap_diff_ratio <= 1.0:
            raise ValueError(
                "min_overlap_diff_ratio must be between zero and one"
            )
        if self.min_roughness_delta < 0.0:
            raise ValueError("min_roughness_delta must be non-negative")


@dataclass(frozen=True)
class FrontierConfig:
    """Frontier sampling and rebuild timing."""

    stride: int = 4
    confidence_threshold: float = 0.6
    rebuild_cooldown: float = 0.25

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between zero and one"
            )
        if self.rebuild_cooldown < 0.0:
            raise ValueError("rebuild_cooldown must be non-negative")


@dataclass(frozen=True)
class RenderingConfig:
    """SLAM rendering cache limits and refresh timing."""

    point_tail: int = 400
    refresh_interval: float = 0.1

    def __post_init__(self) -> None:
        if self.point_tail < 0:
            raise ValueError("point_tail must be non-negative")
        if self.refresh_interval < 0.0:
            raise ValueError("refresh_interval must be non-negative")


@dataclass(frozen=True)
class SimulationConfig:
    """Complete validated configuration passed into a mission."""

    mission_config: MissionConfig = field(default_factory=MissionConfig)
    slam: SlamConfig = field(default_factory=SlamConfig)
    sharing: SharingConfig = field(default_factory=SharingConfig)
    frontier: FrontierConfig = field(default_factory=FrontierConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
