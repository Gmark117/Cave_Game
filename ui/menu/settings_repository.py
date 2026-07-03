"""INI persistence and legacy migration for menu configuration."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypeVar

from config.simulation_config import (
    FrontierConfig,
    MissionConfig,
    RenderingConfig,
    SharingConfig,
    SimulationConfig,
    SlamConfig,
)
from asset_config.gameplay import GameOptions


T = TypeVar("T")


@dataclass(frozen=True)
class AudioSettings:
    volume: int = 100
    music: str = "on"
    button: str = "on"


class MenuSettingsRepository:
    """Persist audio and typed simulation configuration."""

    def __init__(self, game_dir: Path) -> None:
        self.game_dir = Path(game_dir)

    @property
    def options_path(self) -> Path:
        return self.game_dir / "GameConfig" / "options.ini"

    @property
    def simulation_path(self) -> Path:
        return self.game_dir / "GameConfig" / "simulation.ini"

    @property
    def legacy_simulation_path(self) -> Path:
        return self.game_dir / "GameConfig" / "symSettings.ini"

    def load_audio(self) -> AudioSettings:
        config = configparser.ConfigParser()
        config.read(self.options_path)
        return AudioSettings(
            volume=config.getint("Options", "volume", fallback=100),
            music=config.get("Options", "music", fallback="on"),
            button=config.get("Options", "button", fallback="on"),
        )

    def save_audio(self, settings: AudioSettings) -> None:
        config = configparser.ConfigParser()
        config["Options"] = {
            "volume": str(settings.volume),
            "music": settings.music,
            "button": settings.button,
        }
        with self.options_path.open("w") as config_file:
            config.write(config_file)

    def load_simulation(
        self,
        defaults: SimulationConfig,
    ) -> Optional[SimulationConfig]:
        """Load the new format, falling back to the legacy file if absent."""
        if self.simulation_path.exists():
            config = configparser.ConfigParser()
            config.read(self.simulation_path)
            return self._load_current(config, defaults)

        if self.legacy_simulation_path.exists():
            config = configparser.ConfigParser()
            config.read(self.legacy_simulation_path)
            return self._load_legacy(config, defaults)

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
        config["RENDERING"] = {
            "slam_point_tail": str(settings.rendering.point_tail),
            "slam_refresh_interval": str(
                settings.rendering.refresh_interval
            ),
        }
        with self.simulation_path.open("w") as config_file:
            config.write(config_file)

    def _load_current(
        self,
        config: configparser.ConfigParser,
        defaults: SimulationConfig,
    ) -> SimulationConfig:
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
            rendering=self._section_or_default(
                lambda: self._read_rendering(
                    config["RENDERING"]
                    if config.has_section("RENDERING")
                    else {},
                    defaults.rendering,
                ),
                defaults.rendering,
            ),
        )

    def _load_legacy(
        self,
        config: configparser.ConfigParser,
        defaults: SimulationConfig,
    ) -> SimulationConfig:
        mission_section = (
            config["symSettings"] if config.has_section("symSettings") else {}
        )
        slam_section = config["SLAM"] if config.has_section("SLAM") else {}
        mission = self._section_or_default(
            lambda: MissionConfig(
                objective=self._objective_index(
                    str(
                        mission_section.get(
                            "Mode",
                            self._objective_name(
                                defaults.mission_config.objective
                            ),
                        )
                    ),
                    defaults.mission_config.objective,
                ),
                map_dim=self._map_dimension(
                    str(
                        mission_section.get(
                            "Map_dimension",
                            defaults.mission_config.map_dim,
                        )
                    ),
                    defaults.mission_config.map_dim,
                ),
                seed=int(
                    mission_section.get(
                        "Seed",
                        defaults.mission_config.seed,
                    )
                ),
                num_drones=int(
                    mission_section.get(
                        "Drones",
                        defaults.mission_config.num_drones,
                    )
                ),
            ),
            defaults.mission_config,
        )
        slam = self._section_or_default(
            lambda: SlamConfig(
                scan_interval=float(
                    slam_section.get(
                        "scan_interval",
                        defaults.slam.scan_interval,
                    )
                ),
                scan_rays=int(
                    slam_section.get(
                        "scan_rays",
                        defaults.slam.scan_rays,
                    )
                ),
                point_cloud_max_points=int(
                    slam_section.get(
                        "point_cloud_max_points",
                        defaults.slam.point_cloud_max_points,
                    )
                ),
            ),
            defaults.slam,
        )
        sharing = self._section_or_default(
            lambda: SharingConfig(
                drone_interval=defaults.sharing.drone_interval,
                pair_cooldown=defaults.sharing.pair_cooldown,
                rover_interval=float(
                    slam_section.get(
                        "rover_share_interval",
                        defaults.sharing.rover_interval,
                    )
                ),
                compare_stride=defaults.sharing.compare_stride,
                min_new_info_ratio=defaults.sharing.min_new_info_ratio,
                min_overlap_diff_ratio=(
                    defaults.sharing.min_overlap_diff_ratio
                ),
                min_roughness_delta=defaults.sharing.min_roughness_delta,
            ),
            defaults.sharing,
        )
        frontier = self._section_or_default(
            lambda: FrontierConfig(
                stride=int(
                    slam_section.get(
                        "frontier_stride",
                        defaults.frontier.stride,
                    )
                ),
                confidence_threshold=float(
                    slam_section.get(
                        "frontier_confidence_threshold",
                        defaults.frontier.confidence_threshold,
                    )
                ),
                rebuild_cooldown=float(
                    slam_section.get(
                        "frontier_rebuild_cooldown",
                        defaults.frontier.rebuild_cooldown,
                    )
                ),
            ),
            defaults.frontier,
        )
        rendering = self._section_or_default(
            lambda: RenderingConfig(
                point_tail=int(
                    slam_section.get(
                        "render_point_tail",
                        defaults.rendering.point_tail,
                    )
                ),
                refresh_interval=float(
                    slam_section.get(
                        "render_interval",
                        defaults.rendering.refresh_interval,
                    )
                ),
            ),
            defaults.rendering,
        )
        return SimulationConfig(
            mission_config=mission,
            slam=slam,
            sharing=sharing,
            frontier=frontier,
            rendering=rendering,
        )

    def _read_mission(
        self,
        section: object,
        defaults: MissionConfig,
    ) -> MissionConfig:
        return MissionConfig(
            objective=self._objective_index(
                str(section.get("objective", self._objective_name(defaults.objective))),
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
    def _read_rendering(
        section: object,
        defaults: RenderingConfig,
    ) -> RenderingConfig:
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
    def _section_or_default(
        loader: Callable[[], T],
        default: T,
    ) -> T:
        try:
            return loader()
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _objective_name(objective: int) -> str:
        if 0 <= objective < len(GameOptions.MISSION):
            return str(GameOptions.MISSION[objective])
        return str(objective)

    @staticmethod
    def _objective_index(name: str, default: int) -> int:
        for index, option in enumerate(GameOptions.MISSION):
            if str(option).casefold() == name.casefold():
                return index
        try:
            return max(0, int(name))
        except ValueError:
            return default

    @staticmethod
    def _map_name(map_dim: str) -> str:
        for option in GameOptions.MAP_SIZE:
            if str(option).casefold() == map_dim.casefold():
                return str(option)
        return map_dim

    @staticmethod
    def _map_dimension(name: str, default: str) -> str:
        for option in GameOptions.MAP_SIZE:
            if str(option).casefold() == name.casefold():
                return str(option).upper()
        return default
