"""INI persistence for menu configuration."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypeVar

from config.simulation_config import (
    ExplorationConfig,
    FrontierConfig,
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
            "stride": str(settings.frontier.stride),
            "confidence_threshold": str(
                settings.frontier.confidence_threshold
            ),
            "rebuild_cooldown": str(settings.frontier.rebuild_cooldown),
        }
        config["EXPLORATION"] = {
            "policy": settings.exploration.policy,
            "iterations": str(settings.exploration.iterations),
            "horizon": str(settings.exploration.horizon),
            "branching_factor": str(
                settings.exploration.branching_factor
            ),
            "frontier_cluster_limit": str(
                settings.exploration.frontier_cluster_limit
            ),
            "planning_rays": str(settings.exploration.planning_rays),
            "uct_exploration": str(settings.exploration.uct_exploration),
            "discount": str(settings.exploration.discount),
            "rollout_temperature": str(
                settings.exploration.rollout_temperature
            ),
            "decision_time_budget_ms": str(
                settings.exploration.decision_time_budget_ms
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
            "mcts_root_visits": str(settings.trace.mcts_root_visits),
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
            stride=int(section.get("stride", defaults.stride)),
            confidence_threshold=float(
                section.get(
                    "confidence_threshold",
                    defaults.confidence_threshold,
                )
            ),
            rebuild_cooldown=float(
                section.get(
                    "rebuild_cooldown",
                    defaults.rebuild_cooldown,
                )
            ),
        )

    @staticmethod
    def _read_exploration(
        section: object,
        defaults: ExplorationConfig,
    ) -> ExplorationConfig:
        """Parse exploration policy and MCTS search settings."""
        return ExplorationConfig(
            policy=str(section.get("policy", defaults.policy)),
            iterations=int(section.get("iterations", defaults.iterations)),
            horizon=int(section.get("horizon", defaults.horizon)),
            branching_factor=int(
                section.get(
                    "branching_factor",
                    defaults.branching_factor,
                )
            ),
            frontier_cluster_limit=int(
                section.get(
                    "frontier_cluster_limit",
                    defaults.frontier_cluster_limit,
                )
            ),
            planning_rays=int(
                section.get("planning_rays", defaults.planning_rays)
            ),
            uct_exploration=float(
                section.get(
                    "uct_exploration",
                    defaults.uct_exploration,
                )
            ),
            discount=float(section.get("discount", defaults.discount)),
            rollout_temperature=float(
                section.get(
                    "rollout_temperature",
                    defaults.rollout_temperature,
                )
            ),
            decision_time_budget_ms=float(
                section.get(
                    "decision_time_budget_ms",
                    defaults.decision_time_budget_ms,
                )
            ),
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
            mcts_root_visits=int(
                section.get(
                    "mcts_root_visits",
                    defaults.mcts_root_visits,
                )
            ),
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
