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
        """Validate values that are unsafe or ambiguous at runtime."""
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
        """Validate SLAM sampling and memory limits."""
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
        """Validate sharing cadence and comparison thresholds."""
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
    """Stable frontier extraction and lifecycle settings."""

    confidence_threshold: float = 0.6
    minimum_unknown_support: int = 4
    continuation_min_distance: float = 12.0
    continuation_scan_headings: int = 3
    rebuild_cooldown: float = 0.25
    cluster_match_distance: float = 32.0
    missing_refresh_limit: int = 3
    gateway_min_separation: float = 64.0

    def __post_init__(self) -> None:
        """Validate frontier extraction and lifecycle controls."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between zero and one"
            )
        if not 1 <= self.minimum_unknown_support <= 9:
            raise ValueError(
                "minimum_unknown_support must be between one and nine"
            )
        if self.continuation_min_distance < 0.0:
            raise ValueError("continuation_min_distance must be non-negative")
        if not 1 <= self.continuation_scan_headings <= 6:
            raise ValueError(
                "continuation_scan_headings must be between one and six"
            )
        if self.rebuild_cooldown < 0.0:
            raise ValueError("rebuild_cooldown must be non-negative")
        if self.cluster_match_distance < 0.0:
            raise ValueError("cluster_match_distance must be non-negative")
        if self.missing_refresh_limit < 0:
            raise ValueError("missing_refresh_limit must be non-negative")
        if self.gateway_min_separation < 0.0:
            raise ValueError("gateway_min_separation must be non-negative")


@dataclass(frozen=True)
class WaypointConfig:
    """Strategic trail, highway routing, and local connector limits."""

    enabled: bool = True
    spatial_hash_cell: int = 32
    merge_radius: float = 8.0
    connector_distance: float = 64.0
    gateway_connector_distance: float = 192.0
    route_cache_capacity: int = 64
    connector_limit: int = 8
    turn_threshold_degrees: float = 45.0
    minimum_turn_leg: float = 24.0
    chokepoint_narrow_clearance: float = 8.0
    chokepoint_shoulder_clearance: float = 16.0
    chokepoint_shoulder_length: float = 24.0
    recovery_anchor_interval: float = 128.0

    def __post_init__(self) -> None:
        """Validate strategic-graph and bounded-connector controls."""
        if self.spatial_hash_cell <= 0:
            raise ValueError("spatial_hash_cell must be positive")
        if self.merge_radius < 0.0:
            raise ValueError("waypoint merge_radius must be non-negative")
        if self.merge_radius >= self.spatial_hash_cell:
            raise ValueError(
                "waypoint merge_radius must be smaller than spatial_hash_cell"
            )
        if self.connector_distance <= 0.0:
            raise ValueError("connector_distance must be positive")
        if self.gateway_connector_distance <= 0.0:
            raise ValueError("gateway_connector_distance must be positive")
        if self.gateway_connector_distance < self.connector_distance:
            raise ValueError(
                "gateway_connector_distance must not be below connector_distance"
            )
        if self.route_cache_capacity <= 0:
            raise ValueError("route_cache_capacity must be positive")
        if self.connector_limit <= 0:
            raise ValueError("connector_limit must be positive")
        if min(
            self.turn_threshold_degrees,
            self.minimum_turn_leg,
            self.chokepoint_narrow_clearance,
            self.chokepoint_shoulder_clearance,
            self.chokepoint_shoulder_length,
            self.recovery_anchor_interval,
        ) <= 0.0:
            raise ValueError("strategic waypoint thresholds must be positive")
        if self.chokepoint_narrow_clearance >= self.chokepoint_shoulder_clearance:
            raise ValueError(
                "chokepoint narrow clearance must be below shoulder clearance"
            )


@dataclass(frozen=True)
class ExplorationConfig:
    """Exploration policy selection and MCTS search controls."""

    policy: str = "mcts"
    iterations: int = 256
    horizon: int = 4
    planning_rays: int = 4
    uct_exploration: float = 1.414
    discount: float = 0.95
    decision_time_budget_ms: float = 40.0

    def __post_init__(self) -> None:
        """Validate exploration policy and numeric search settings."""
        policy = self.policy.casefold()
        if policy not in {"mcts", "frontier"}:
            raise ValueError("policy must be 'mcts' or 'frontier'")
        object.__setattr__(self, "policy", policy)

        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.planning_rays <= 0:
            raise ValueError("planning_rays must be positive")
        if self.uct_exploration < 0.0:
            raise ValueError("uct_exploration must be non-negative")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between zero and one")
        if self.decision_time_budget_ms < 0.0:
            raise ValueError(
                "decision_time_budget_ms must be non-negative"
            )


@dataclass(frozen=True)
class RenderingConfig:
    """SLAM rendering cache limits and refresh timing."""

    point_tail: int = 400
    refresh_interval: float = 0.1

    def __post_init__(self) -> None:
        """Validate cached-rendering limits."""
        if self.point_tail < 0:
            raise ValueError("point_tail must be non-negative")
        if self.refresh_interval < 0.0:
            raise ValueError("refresh_interval must be non-negative")


@dataclass(frozen=True)
class TraceConfig:
    """Structured runtime trace controls for diagnosing live missions."""

    enabled: bool = False
    directory: str = "logs"
    mcts_root_visits: int = 6
    frame_interval: float = 1.0

    def __post_init__(self) -> None:
        """Validate trace output settings."""
        if not self.directory:
            raise ValueError("trace directory must not be empty")
        if self.mcts_root_visits < 0:
            raise ValueError("mcts_root_visits must be non-negative")
        if self.frame_interval < 0.0:
            raise ValueError("frame_interval must be non-negative")


@dataclass(frozen=True)
class SimulationConfig:
    """Complete validated configuration passed into a mission."""

    mission_config: MissionConfig = field(default_factory=MissionConfig)
    slam: SlamConfig = field(default_factory=SlamConfig)
    sharing: SharingConfig = field(default_factory=SharingConfig)
    frontier: FrontierConfig = field(default_factory=FrontierConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    waypoints: WaypointConfig = field(default_factory=WaypointConfig)
