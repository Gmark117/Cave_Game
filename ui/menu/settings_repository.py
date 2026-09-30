"""INI persistence for menu configuration."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypeVar

from config.simulation_config import (
    FocusedFrontierBatchConfig,
    ExplorationConfig,
    FrontierConfig,
    HighwayConfig,
    IncidentalScanConfig,
    MissionConfig,
    RenderingConfig,
    SharingConfig,
    SimulationConfig,
    SlamConfig,
    TraceConfig,
)
from asset_config.gameplay import GameOptions


T = TypeVar("T")


@dataclass(frozen=True)
class AudioSettings:
    """Menu audio preferences stored in the options INI."""

    volume: int = 100
    music: str = "on"
    button: str = "on"


class MenuSettingsRepository:
    """Persist audio and typed simulation configuration."""

    def __init__(self, game_dir: Path) -> None:
        """Store the project root used to resolve config file paths."""
        self.game_dir = Path(game_dir)

    @property
    def options_path(self) -> Path:
        """Ignored per-user audio settings written at runtime."""
        return self.game_dir / "GameConfig" / "options.local.ini"

    @property
    def options_default_path(self) -> Path:
        """Committed default audio settings used on first run."""
        return self.game_dir / "GameConfig" / "options.default.ini"

    @property
    def simulation_path(self) -> Path:
        """Ignored per-user simulation settings written at runtime."""
        return self.game_dir / "GameConfig" / "simulation.local.ini"

    @property
    def simulation_default_path(self) -> Path:
        """Committed default simulation settings used on first run."""
        return self.game_dir / "GameConfig" / "simulation.default.ini"

    def load_audio(self) -> AudioSettings:
        """Load audio settings with default, then local precedence."""
        config = configparser.ConfigParser()
        config.read(
            [
                self.options_default_path,
                self.options_path,
            ]
        )
        return AudioSettings(
            volume=config.getint("Options", "volume", fallback=100),
            music=config.get("Options", "music", fallback="on"),
            button=config.get("Options", "button", fallback="on"),
        )

    def save_audio(self, settings: AudioSettings) -> None:
        """Persist audio settings to the ignored local options file."""
        config = configparser.ConfigParser()
        config["Options"] = {
            "volume": str(settings.volume),
            "music": settings.music,
            "button": settings.button,
        }
        self.options_path.parent.mkdir(parents=True, exist_ok=True)
        with self.options_path.open("w") as config_file:
            config.write(config_file)

    def load_simulation(
        self,
        defaults: SimulationConfig,
    ) -> Optional[SimulationConfig]:
        """Load sectioned simulation settings with local precedence."""
        for current_path in (
            self.simulation_path,
            self.simulation_default_path,
        ):
            if not current_path.exists():
                continue
            config = configparser.ConfigParser()
            config.read(current_path)
            return self._load_current(config, defaults)

        return None

    def save_simulation(self, settings: SimulationConfig) -> None:
        """Write only the new configuration format."""
        mission = settings.mission_config
        config = configparser.ConfigParser()
        config["MISSION"] = {
            "objective": self._objective_name(mission.objective),
            "map_dimension": self._map_name(mission.map_dim),
            "seed": str(mission.seed),
            "drones": str(mission.num_drones),
            "wall_completion_tolerance_pixels": str(
                mission.wall_completion_tolerance_pixels
            ),
        }
        config["SLAM"] = {
            "scan_interval": str(settings.slam.scan_interval),
            "scan_rays": str(settings.slam.scan_rays),
            "point_cloud_max_points": str(
                settings.slam.point_cloud_max_points
            ),
        }
        config["SHARING"] = {
            "drone_interval": str(settings.sharing.drone_interval),
            "pair_cooldown": str(settings.sharing.pair_cooldown),
            "rover_interval": str(settings.sharing.rover_interval),
            "compare_stride": str(settings.sharing.compare_stride),
            "min_new_info_ratio": str(settings.sharing.min_new_info_ratio),
            "min_overlap_diff_ratio": str(
                settings.sharing.min_overlap_diff_ratio
            ),
            "min_roughness_delta": str(
                settings.sharing.min_roughness_delta
            ),
        }
        config["FRONTIER"] = {
            "confidence_threshold": str(
                settings.frontier.confidence_threshold
            ),
            "stride": str(settings.frontier.stride),
            "rebuild_cooldown": str(settings.frontier.rebuild_cooldown),
            "minimum_cluster_cells": str(
                settings.frontier.minimum_cluster_cells
            ),
            "minimum_unknown_support_cells": str(
                settings.frontier.minimum_unknown_support_cells
            ),
            "distance_band": str(settings.frontier.distance_band),
            "wall_continuation_weight": str(
                settings.frontier.wall_continuation_weight
            ),
            "cluster_size_weight": str(
                settings.frontier.cluster_size_weight
            ),
            "cluster_proximity_weight": str(
                settings.frontier.cluster_proximity_weight
            ),
            "global_cell_size": str(
                settings.frontier.global_cell_size
            ),
            "global_refresh_interval": str(
                settings.frontier.global_refresh_interval
            ),
            "global_ownership_weight": str(
                settings.frontier.global_ownership_weight
            ),
            "maximum_path_circuity": str(
                settings.frontier.maximum_path_circuity
            ),
        }
        config["EXPLORATION"] = {
            "policy": settings.exploration.policy,
            "stagnation_distance": str(
                settings.exploration.stagnation_distance
            ),
            "stagnation_min_sensor_cells_per_px": str(
                settings.exploration.stagnation_min_sensor_cells_per_px
            ),
            "wall_direction_bias": str(
                settings.exploration.wall_direction_bias
            ),
            "unexplored_direction_bias": str(
                settings.exploration.unexplored_direction_bias
            ),
            "separation_direction_bias": str(
                settings.exploration.separation_direction_bias
            ),
            "coverage_memory_cell_size": str(
                settings.exploration.coverage_memory_cell_size
            ),
            "coverage_memory_decay_seconds": str(
                settings.exploration.coverage_memory_decay_seconds
            ),
            "coverage_visit_weight": str(
                settings.exploration.coverage_visit_weight
            ),
            "coverage_edge_weight": str(
                settings.exploration.coverage_edge_weight
            ),
        }
        config["INCIDENTAL_SCAN"] = {
            "mode": settings.incidental_scan.mode,
            "maximum_attempts_per_directive": str(
                settings.incidental_scan.maximum_attempts_per_directive
            ),
            "maximum_wait_seconds_per_directive": str(
                settings.incidental_scan.maximum_wait_seconds_per_directive
            ),
            "attempt_timeout_seconds": str(
                settings.incidental_scan.attempt_timeout_seconds
            ),
            "maximum_rotation_degrees_per_directive": str(
                settings.incidental_scan.maximum_rotation_degrees_per_directive
            ),
            "distance_cooldown_sensor_ranges": str(
                settings.incidental_scan.distance_cooldown_sensor_ranges
            ),
            "sample_spacing_sensor_ranges": str(
                settings.incidental_scan.sample_spacing_sensor_ranges
            ),
        }
        config["HIGHWAY"] = {
            "mode": settings.highway.mode,
            "macro_cell_size": str(settings.highway.macro_cell_size),
            "minimum_version_delta": str(
                settings.highway.minimum_version_delta
            ),
            "maximum_build_ms": str(settings.highway.maximum_build_ms),
            "maximum_query_ms": str(settings.highway.maximum_query_ms),
            "maximum_connector_expansions": str(
                settings.highway.maximum_connector_expansions
            ),
            "minimum_route_sensor_ranges": str(
                settings.highway.minimum_route_sensor_ranges
            ),
            "maximum_route_circuity": str(
                settings.highway.maximum_route_circuity
            ),
        }
        config["FOCUSED_FRONTIER_BATCH"] = {
            "mode": settings.focused_frontier_batch.mode,
            "maximum_claimed_components": str(
                settings.focused_frontier_batch.maximum_claimed_components
            ),
            "maximum_total_components": str(
                settings.focused_frontier_batch.maximum_total_components
            ),
            "lease_margin_sensor_ranges": str(
                settings.focused_frontier_batch.lease_margin_sensor_ranges
            ),
            "maximum_detour_sensor_ranges": str(
                settings.focused_frontier_batch.maximum_detour_sensor_ranges
            ),
            "minimum_avoided_round_trip_sensor_ranges": str(
                settings.focused_frontier_batch
                .minimum_avoided_round_trip_sensor_ranges
            ),
            "maximum_service_seconds": str(
                settings.focused_frontier_batch.maximum_service_seconds
            ),
            "maximum_total_dfs_nodes": str(
                settings.focused_frontier_batch.maximum_total_dfs_nodes
            ),
            "maximum_consecutive_low_gain_scans": str(
                settings.focused_frontier_batch.maximum_consecutive_low_gain_scans
            ),
            "low_gain_maximum_new_cells": str(
                settings.focused_frontier_batch.low_gain_maximum_new_cells
            ),
            "low_gain_maximum_confidence_gain": str(
                settings.focused_frontier_batch.low_gain_maximum_confidence_gain
            ),
            "maximum_planning_ms": str(
                settings.focused_frontier_batch.maximum_planning_ms
            ),
            "maximum_route_queries": str(
                settings.focused_frontier_batch.maximum_route_queries
            ),
        }
        config["RENDERING"] = {
            "slam_point_tail": str(settings.rendering.point_tail),
            "slam_refresh_interval": str(
                settings.rendering.refresh_interval
            ),
        }
        config["TRACE"] = {
            "enabled": str(settings.trace.enabled),
            "directory": settings.trace.directory,
            "frame_interval": str(settings.trace.frame_interval),
        }
        self.simulation_path.parent.mkdir(parents=True, exist_ok=True)
        with self.simulation_path.open("w") as config_file:
            config.write(config_file)

    def _load_current(
        self,
        config: configparser.ConfigParser,
        defaults: SimulationConfig,
    ) -> SimulationConfig:
        """Read the sectioned INI format."""
        return SimulationConfig(
            mission_config=self._section_or_default(
                lambda: self._read_mission(
                    config["MISSION"] if config.has_section("MISSION") else {},
                    defaults.mission_config,
                ),
                defaults.mission_config,
            ),
            slam=self._section_or_default(
                lambda: self._read_slam(
                    config["SLAM"] if config.has_section("SLAM") else {},
                    defaults.slam,
                ),
                defaults.slam,
            ),
            sharing=self._section_or_default(
                lambda: self._read_sharing(
                    config["SHARING"]
                    if config.has_section("SHARING")
                    else {},
                    defaults.sharing,
                ),
                defaults.sharing,
            ),
            frontier=self._section_or_default(
                lambda: self._read_frontier(
                    config["FRONTIER"]
                    if config.has_section("FRONTIER")
                    else {},
                    defaults.frontier,
                ),
                defaults.frontier,
            ),
            exploration=self._section_or_default(
                lambda: self._read_exploration(
                    config["EXPLORATION"]
                    if config.has_section("EXPLORATION")
                    else {},
                    defaults.exploration,
                ),
                defaults.exploration,
            ),
            incidental_scan=self._section_or_default(
                lambda: self._read_incidental_scan(
                    config["INCIDENTAL_SCAN"]
                    if config.has_section("INCIDENTAL_SCAN")
                    else {},
                    defaults.incidental_scan,
                ),
                defaults.incidental_scan,
            ),
            highway=self._section_or_default(
                lambda: self._read_highway(
                    config["HIGHWAY"]
                    if config.has_section("HIGHWAY")
                    else {},
                    defaults.highway,
                ),
                defaults.highway,
            ),
            focused_frontier_batch=self._section_or_default(
                lambda: self._read_focused_frontier_batch(
                    config["FOCUSED_FRONTIER_BATCH"]
                    if config.has_section("FOCUSED_FRONTIER_BATCH")
                    else (
                        config["ENDGAME_BATCH"]
                        if config.has_section("ENDGAME_BATCH")
                        else {}
                    ),
                    defaults.focused_frontier_batch,
                ),
                defaults.focused_frontier_batch,
            ),
            rendering=self._section_or_default(
                lambda: self._read_rendering(
                    config["RENDERING"]
                    if config.has_section("RENDERING")
                    else {},
                    defaults.rendering,
                ),
                defaults.rendering,
            ),
            trace=self._section_or_default(
                lambda: self._read_trace(
                    config["TRACE"] if config.has_section("TRACE") else {},
                    defaults.trace,
                ),
                defaults.trace,
            ),
        )

    def _read_mission(
        self,
        section: object,
        defaults: MissionConfig,
    ) -> MissionConfig:
        """Parse the mission section, using defaults for missing values."""
        return MissionConfig(
            objective=self._objective_index(
                str(
                    section.get(
                        "objective",
                        self._objective_name(defaults.objective),
                    )
                ),
                defaults.objective,
            ),
            map_dim=self._map_dimension(
                str(section.get("map_dimension", defaults.map_dim)),
                defaults.map_dim,
            ),
            seed=int(section.get("seed", defaults.seed)),
            num_drones=int(section.get("drones", defaults.num_drones)),
            wall_completion_tolerance_pixels=int(section.get(
                "wall_completion_tolerance_pixels",
                defaults.wall_completion_tolerance_pixels,
            )),
        )

    @staticmethod
    def _read_slam(section: object, defaults: SlamConfig) -> SlamConfig:
        """Parse the SLAM section."""
        return SlamConfig(
            scan_interval=float(
                section.get("scan_interval", defaults.scan_interval)
            ),
            scan_rays=int(section.get("scan_rays", defaults.scan_rays)),
            point_cloud_max_points=int(
                section.get(
                    "point_cloud_max_points",
                    defaults.point_cloud_max_points,
                )
            ),
        )

    @staticmethod
    def _read_sharing(
        section: object,
        defaults: SharingConfig,
    ) -> SharingConfig:
        """Parse proximity-sharing thresholds and intervals."""
        return SharingConfig(
            drone_interval=float(
                section.get("drone_interval", defaults.drone_interval)
            ),
            pair_cooldown=float(
                section.get("pair_cooldown", defaults.pair_cooldown)
            ),
            rover_interval=float(
                section.get("rover_interval", defaults.rover_interval)
            ),
            compare_stride=int(
                section.get("compare_stride", defaults.compare_stride)
            ),
            min_new_info_ratio=float(
                section.get(
                    "min_new_info_ratio",
                    defaults.min_new_info_ratio,
                )
            ),
            min_overlap_diff_ratio=float(
                section.get(
                    "min_overlap_diff_ratio",
                    defaults.min_overlap_diff_ratio,
                )
            ),
            min_roughness_delta=float(
                section.get(
                    "min_roughness_delta",
                    defaults.min_roughness_delta,
                )
            ),
        )

    @staticmethod
    def _read_frontier(
        section: object,
        defaults: FrontierConfig,
    ) -> FrontierConfig:
        """Parse frontier detection settings."""
        return FrontierConfig(
            confidence_threshold=float(
                section.get(
                    "confidence_threshold",
                    defaults.confidence_threshold,
                )
            ),
            stride=int(section.get("stride", defaults.stride)),
            rebuild_cooldown=float(
                section.get(
                    "rebuild_cooldown",
                    defaults.rebuild_cooldown,
                )
            ),
            minimum_cluster_cells=int(section.get(
                "minimum_cluster_cells",
                defaults.minimum_cluster_cells,
            )),
            minimum_unknown_support_cells=int(section.get(
                "minimum_unknown_support_cells",
                defaults.minimum_unknown_support_cells,
            )),
            distance_band=float(section.get(
                "distance_band",
                defaults.distance_band,
            )),
            wall_continuation_weight=float(section.get(
                "wall_continuation_weight",
                defaults.wall_continuation_weight,
            )),
            cluster_size_weight=float(section.get(
                "cluster_size_weight",
                defaults.cluster_size_weight,
            )),
            cluster_proximity_weight=float(section.get(
                "cluster_proximity_weight",
                defaults.cluster_proximity_weight,
            )),
            global_cell_size=int(section.get(
                "global_cell_size",
                defaults.global_cell_size,
            )),
            global_refresh_interval=float(section.get(
                "global_refresh_interval",
                defaults.global_refresh_interval,
            )),
            global_ownership_weight=float(section.get(
                "global_ownership_weight",
                defaults.global_ownership_weight,
            )),
            maximum_path_circuity=float(section.get(
                "maximum_path_circuity",
                defaults.maximum_path_circuity,
            )),
        )

    @staticmethod
    def _read_exploration(
        section: object,
        defaults: ExplorationConfig,
    ) -> ExplorationConfig:
        """Parse policy while ignoring retired legacy MCTS keys."""
        return ExplorationConfig(
            policy=str(section.get("policy", defaults.policy)),
            stagnation_distance=float(section.get(
                "stagnation_distance",
                defaults.stagnation_distance,
            )),
            stagnation_min_sensor_cells_per_px=float(section.get(
                "stagnation_min_sensor_cells_per_px",
                defaults.stagnation_min_sensor_cells_per_px,
            )),
            wall_direction_bias=float(section.get(
                "wall_direction_bias",
                defaults.wall_direction_bias,
            )),
            unexplored_direction_bias=float(section.get(
                "unexplored_direction_bias",
                defaults.unexplored_direction_bias,
            )),
            separation_direction_bias=float(section.get(
                "separation_direction_bias",
                defaults.separation_direction_bias,
            )),
            coverage_memory_cell_size=int(section.get(
                "coverage_memory_cell_size",
                defaults.coverage_memory_cell_size,
            )),
            coverage_memory_decay_seconds=float(section.get(
                "coverage_memory_decay_seconds",
                defaults.coverage_memory_decay_seconds,
            )),
            coverage_visit_weight=float(section.get(
                "coverage_visit_weight",
                defaults.coverage_visit_weight,
            )),
            coverage_edge_weight=float(section.get(
                "coverage_edge_weight",
                defaults.coverage_edge_weight,
            )),
        )

    @staticmethod
    def _read_incidental_scan(
        section: object,
        defaults: IncidentalScanConfig,
    ) -> IncidentalScanConfig:
        """Parse bounded incidental wall-pocket scan controls."""
        return IncidentalScanConfig(
            mode=str(section.get("mode", defaults.mode)),
            maximum_attempts_per_directive=int(section.get(
                "maximum_attempts_per_directive",
                defaults.maximum_attempts_per_directive,
            )),
            maximum_wait_seconds_per_directive=float(section.get(
                "maximum_wait_seconds_per_directive",
                defaults.maximum_wait_seconds_per_directive,
            )),
            attempt_timeout_seconds=float(section.get(
                "attempt_timeout_seconds",
                defaults.attempt_timeout_seconds,
            )),
            maximum_rotation_degrees_per_directive=float(section.get(
                "maximum_rotation_degrees_per_directive",
                defaults.maximum_rotation_degrees_per_directive,
            )),
            distance_cooldown_sensor_ranges=float(section.get(
                "distance_cooldown_sensor_ranges",
                defaults.distance_cooldown_sensor_ranges,
            )),
            sample_spacing_sensor_ranges=float(section.get(
                "sample_spacing_sensor_ranges",
                defaults.sample_spacing_sensor_ranges,
            )),
        )

    @staticmethod
    def _read_focused_frontier_batch(
        section: object,
        defaults: FocusedFrontierBatchConfig,
    ) -> FocusedFrontierBatchConfig:
        """Parse focused frontier batching and bounded local-work controls."""
        return FocusedFrontierBatchConfig(
            mode=str(section.get("mode", defaults.mode)),
            maximum_claimed_components=int(section.get(
                "maximum_claimed_components",
                defaults.maximum_claimed_components,
            )),
            maximum_total_components=int(section.get(
                "maximum_total_components",
                defaults.maximum_total_components,
            )),
            lease_margin_sensor_ranges=float(section.get(
                "lease_margin_sensor_ranges",
                defaults.lease_margin_sensor_ranges,
            )),
            maximum_detour_sensor_ranges=float(section.get(
                "maximum_detour_sensor_ranges",
                defaults.maximum_detour_sensor_ranges,
            )),
            minimum_avoided_round_trip_sensor_ranges=float(section.get(
                "minimum_avoided_round_trip_sensor_ranges",
                defaults.minimum_avoided_round_trip_sensor_ranges,
            )),
            maximum_service_seconds=float(section.get(
                "maximum_service_seconds",
                defaults.maximum_service_seconds,
            )),
            maximum_total_dfs_nodes=int(section.get(
                "maximum_total_dfs_nodes",
                defaults.maximum_total_dfs_nodes,
            )),
            maximum_consecutive_low_gain_scans=int(section.get(
                "maximum_consecutive_low_gain_scans",
                defaults.maximum_consecutive_low_gain_scans,
            )),
            low_gain_maximum_new_cells=int(section.get(
                "low_gain_maximum_new_cells",
                defaults.low_gain_maximum_new_cells,
            )),
            low_gain_maximum_confidence_gain=float(section.get(
                "low_gain_maximum_confidence_gain",
                defaults.low_gain_maximum_confidence_gain,
            )),
            maximum_planning_ms=float(section.get(
                "maximum_planning_ms",
                defaults.maximum_planning_ms,
            )),
            maximum_route_queries=int(section.get(
                "maximum_route_queries",
                defaults.maximum_route_queries,
            )),
        )

    @staticmethod
    def _read_highway(
        section: object,
        defaults: HighwayConfig,
    ) -> HighwayConfig:
        """Parse bounded rover-highway construction and rollout controls."""
        return HighwayConfig(
            mode=str(section.get("mode", defaults.mode)),
            macro_cell_size=int(section.get(
                "macro_cell_size",
                defaults.macro_cell_size,
            )),
            minimum_version_delta=int(section.get(
                "minimum_version_delta",
                defaults.minimum_version_delta,
            )),
            maximum_build_ms=float(section.get(
                "maximum_build_ms",
                defaults.maximum_build_ms,
            )),
            maximum_query_ms=float(section.get(
                "maximum_query_ms",
                defaults.maximum_query_ms,
            )),
            maximum_connector_expansions=int(section.get(
                "maximum_connector_expansions",
                defaults.maximum_connector_expansions,
            )),
            minimum_route_sensor_ranges=float(section.get(
                "minimum_route_sensor_ranges",
                defaults.minimum_route_sensor_ranges,
            )),
            maximum_route_circuity=float(section.get(
                "maximum_route_circuity",
                defaults.maximum_route_circuity,
            )),
        )

    @staticmethod
    def _read_rendering(
        section: object,
        defaults: RenderingConfig,
    ) -> RenderingConfig:
        """Parse SLAM rendering cache settings."""
        return RenderingConfig(
            point_tail=int(
                section.get("slam_point_tail", defaults.point_tail)
            ),
            refresh_interval=float(
                section.get(
                    "slam_refresh_interval",
                    defaults.refresh_interval,
                )
            ),
        )

    @staticmethod
    def _read_trace(
        section: object,
        defaults: TraceConfig,
    ) -> TraceConfig:
        """Parse structured runtime trace settings."""
        enabled_value = str(section.get("enabled", defaults.enabled))
        enabled = enabled_value.casefold() in {"1", "true", "yes", "on"}
        return TraceConfig(
            enabled=enabled,
            directory=str(section.get("directory", defaults.directory)),
            frame_interval=float(
                section.get("frame_interval", defaults.frame_interval)
            ),
        )

    @staticmethod
    def _section_or_default(
        loader: Callable[[], T],
        default: T,
    ) -> T:
        """Return a parsed section or its default when conversion fails."""
        try:
            return loader()
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _objective_name(objective: int) -> str:
        """Convert a mission index to the menu label when possible."""
        if 0 <= objective < len(GameOptions.MISSION):
            return str(GameOptions.MISSION[objective])
        return str(objective)

    @staticmethod
    def _objective_index(name: str, default: int) -> int:
        """Convert a mission label or numeric string to a mission index."""
        for index, option in enumerate(GameOptions.MISSION):
            if str(option).casefold() == name.casefold():
                return index
        try:
            return max(0, int(name))
        except ValueError:
            return default

    @staticmethod
    def _map_name(map_dim: str) -> str:
        """Normalize a map dimension to the menu spelling when possible."""
        for option in GameOptions.MAP_SIZE:
            if str(option).casefold() == map_dim.casefold():
                return str(option)
        return map_dim

    @staticmethod
    def _map_dimension(name: str, default: str) -> str:
        """Normalize a map dimension to the internal uppercase value."""
        for option in GameOptions.MAP_SIZE:
            if str(option).casefold() == name.casefold():
                return str(option).upper()
        return default
