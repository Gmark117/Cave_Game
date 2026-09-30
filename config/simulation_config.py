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
    wall_completion_tolerance_pixels: int = 30

    def __post_init__(self) -> None:
        """Validate values that are unsafe or ambiguous at runtime."""
        if self.objective < 0:
            raise ValueError("objective must be non-negative")
        if not self.map_dim:
            raise ValueError("map_dim must not be empty")
        if self.num_drones <= 0:
            raise ValueError("num_drones must be positive")
        if self.wall_completion_tolerance_pixels < 0:
            raise ValueError(
                "wall_completion_tolerance_pixels must be non-negative"
            )


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
    """Local borders and cached whole-map frontier guidance settings."""

    confidence_threshold: float = 0.6
    stride: int = 4
    rebuild_cooldown: float = 0.25
    minimum_cluster_cells: int = 12
    minimum_unknown_support_cells: int = 64
    distance_band: float = 16.0
    wall_continuation_weight: float = 2.0
    cluster_size_weight: float = 2.0
    cluster_proximity_weight: float = 1.0
    global_cell_size: int = 32
    global_refresh_interval: float = 2.0
    global_ownership_weight: float = 2.0
    maximum_path_circuity: float = 4.0

    def __post_init__(self) -> None:
        """Validate border extraction controls."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between zero and one"
            )
        if self.stride <= 0:
            raise ValueError("frontier stride must be positive")
        if self.rebuild_cooldown < 0.0:
            raise ValueError("rebuild_cooldown must be non-negative")
        if self.minimum_cluster_cells <= 0:
            raise ValueError("minimum_cluster_cells must be positive")
        if self.minimum_unknown_support_cells <= 0:
            raise ValueError(
                "minimum_unknown_support_cells must be positive"
            )
        if self.distance_band <= 0.0:
            raise ValueError("frontier distance_band must be positive")
        if self.global_cell_size <= 0:
            raise ValueError("frontier global_cell_size must be positive")
        if self.global_refresh_interval < 0.0:
            raise ValueError(
                "frontier global_refresh_interval must be non-negative"
            )
        if self.maximum_path_circuity < 1.0:
            raise ValueError(
                "frontier maximum_path_circuity must be at least one"
            )
        if min(
            self.wall_continuation_weight,
            self.cluster_size_weight,
            self.cluster_proximity_weight,
            self.global_ownership_weight,
        ) < 0.0:
            raise ValueError("frontier scoring weights must be non-negative")


@dataclass(frozen=True)
class ExplorationConfig:
    """Biased-random exploration and low-information recovery controls."""

    policy: str = "random"
    stagnation_distance: float = 120.0
    stagnation_min_sensor_cells_per_px: float = 0.5
    wall_direction_bias: float = 4.0
    unexplored_direction_bias: float = 2.0
    separation_direction_bias: float = 1.5
    coverage_memory_cell_size: int = 32
    coverage_memory_decay_seconds: float = 180.0
    coverage_visit_weight: float = 1.5
    coverage_edge_weight: float = 2.0

    def __post_init__(self) -> None:
        """Validate and normalize legacy policy names."""
        policy = self.policy.casefold()
        if policy not in {"random", "wall_region", "mcts", "frontier"}:
            raise ValueError(
                "policy must be 'random', 'wall_region', 'mcts', or 'frontier'"
            )
        if self.stagnation_distance <= 0.0:
            raise ValueError("stagnation_distance must be positive")
        if self.stagnation_min_sensor_cells_per_px < 0.0:
            raise ValueError(
                "stagnation_min_sensor_cells_per_px must be non-negative"
            )
        if self.coverage_memory_cell_size <= 0:
            raise ValueError("coverage_memory_cell_size must be positive")
        if self.coverage_memory_decay_seconds <= 0.0:
            raise ValueError(
                "coverage_memory_decay_seconds must be positive"
            )
        if min(
            self.wall_direction_bias,
            self.unexplored_direction_bias,
            self.separation_direction_bias,
            self.coverage_visit_weight,
            self.coverage_edge_weight,
        ) < 0.0:
            raise ValueError("exploration direction biases must be non-negative")
        object.__setattr__(self, "policy", "random")


@dataclass(frozen=True)
class IncidentalScanConfig:
    """Bounded wall-pocket observation during eligible component transit."""

    mode: str = "off"
    maximum_attempts_per_directive: int = 2
    maximum_wait_seconds_per_directive: float = 6.0
    attempt_timeout_seconds: float = 3.0
    maximum_rotation_degrees_per_directive: float = 180.0
    distance_cooldown_sensor_ranges: float = 1.0
    sample_spacing_sensor_ranges: float = 0.5

    def __post_init__(self) -> None:
        mode = str(self.mode).casefold()
        if mode not in {"off", "observe", "active"}:
            raise ValueError(
                "incidental scan mode must be 'off', 'observe', or 'active'"
            )
        if self.maximum_attempts_per_directive < 0:
            raise ValueError("maximum incidental attempts must be non-negative")
        if min(
            self.maximum_wait_seconds_per_directive,
            self.attempt_timeout_seconds,
            self.maximum_rotation_degrees_per_directive,
            self.distance_cooldown_sensor_ranges,
            self.sample_spacing_sensor_ranges,
        ) < 0.0:
            raise ValueError("incidental scan limits must be non-negative")
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class HighwayConfig:
    """Rover-built navigation graph and physically published route advice."""

    mode: str = "off"
    macro_cell_size: int = 32
    minimum_version_delta: int = 4
    maximum_build_ms: float = 250.0
    maximum_query_ms: float = 50.0
    maximum_connector_expansions: int = 4096
    minimum_route_sensor_ranges: float = 2.0
    maximum_route_circuity: float = 3.0

    def __post_init__(self) -> None:
        mode = str(self.mode).casefold()
        if mode not in {"off", "observe", "active"}:
            raise ValueError("highway mode must be 'off', 'observe', or 'active'")
        if self.macro_cell_size < 2:
            raise ValueError("highway macro cell size must be at least two")
        if self.minimum_version_delta < 1:
            raise ValueError("highway minimum version delta must be positive")
        if self.maximum_connector_expansions < 1:
            raise ValueError("highway connector expansion limit must be positive")
        if min(
            self.maximum_build_ms,
            self.maximum_query_ms,
            self.minimum_route_sensor_ranges,
        ) < 0.0:
            raise ValueError("highway budgets must be non-negative")
        if self.maximum_route_circuity < 1.0:
            raise ValueError("highway maximum route circuity must be at least one")
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class FocusedFrontierBatchConfig:
    """Bounded frontier-task batching during the registry-focused phase."""

    mode: str = "off"
    maximum_claimed_components: int = 3
    maximum_total_components: int = 4
    lease_margin_sensor_ranges: float = 1.0
    maximum_detour_sensor_ranges: float = 2.0
    minimum_avoided_round_trip_sensor_ranges: float = 0.25
    maximum_service_seconds: float = 45.0
    maximum_total_dfs_nodes: int = 48
    maximum_consecutive_low_gain_scans: int = 2
    low_gain_maximum_new_cells: int = 1
    low_gain_maximum_confidence_gain: float = 1.0
    maximum_planning_ms: float = 250.0
    maximum_route_queries: int = 128

    def __post_init__(self) -> None:
        mode = str(self.mode).casefold()
        if mode not in {"off", "observe", "active"}:
            raise ValueError(
                "focused frontier batch mode must be 'off', 'observe', or 'active'"
            )
        if self.maximum_claimed_components < 1:
            raise ValueError("maximum claimed components must be positive")
        if self.maximum_total_components < self.maximum_claimed_components:
            raise ValueError(
                "maximum total components must cover claimed components"
            )
        if self.maximum_total_dfs_nodes < 1:
            raise ValueError("maximum total DFS nodes must be positive")
        if self.maximum_consecutive_low_gain_scans < 1:
            raise ValueError("maximum consecutive low-gain scans must be positive")
        if self.low_gain_maximum_new_cells < 0:
            raise ValueError("low-gain cell threshold must be non-negative")
        if self.maximum_route_queries < 1:
            raise ValueError(
                "focused frontier route query limit must be positive"
            )
        if min(
            self.lease_margin_sensor_ranges,
            self.maximum_detour_sensor_ranges,
            self.minimum_avoided_round_trip_sensor_ranges,
            self.maximum_service_seconds,
            self.low_gain_maximum_confidence_gain,
            self.maximum_planning_ms,
        ) < 0.0:
            raise ValueError("focused frontier batch limits must be non-negative")
        object.__setattr__(self, "mode", mode)


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
    frame_interval: float = 1.0

    def __post_init__(self) -> None:
        """Validate trace output settings."""
        if not self.directory:
            raise ValueError("trace directory must not be empty")
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
    incidental_scan: IncidentalScanConfig = field(
        default_factory=IncidentalScanConfig
    )
    highway: HighwayConfig = field(default_factory=HighwayConfig)
    focused_frontier_batch: FocusedFrontierBatchConfig = field(
        default_factory=FocusedFrontierBatchConfig
    )
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
