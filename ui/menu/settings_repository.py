"""INI persistence for menu configuration."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping, Optional, TypeVar, get_type_hints

from config.simulation_config import MissionConfig, SimulationConfig
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
        """Write the supported dataclass fields to the local sectioned INI."""
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
        for setting in fields(settings):
            if setting.name == "mission_config":
                continue
            component = getattr(settings, setting.name)
            config[setting.name.upper()] = {
                self._ini_key(setting.name, item.name): str(
                    getattr(component, item.name)
                )
                for item in fields(component)
            }
        self.simulation_path.parent.mkdir(parents=True, exist_ok=True)
        with self.simulation_path.open("w") as config_file:
            config.write(config_file)


    def _load_current(
        self,
        config: configparser.ConfigParser,
        defaults: SimulationConfig,
    ) -> SimulationConfig:
        """Read each section independently; an invalid section uses its fallback."""
        components = {}
        for setting in fields(defaults):
            name = setting.name
            default = getattr(defaults, name)
            section_name = "MISSION" if name == "mission_config" else name.upper()
            if name == "focused_frontier_batch" and not config.has_section(
                section_name
            ):
                section_name = "ENDGAME_BATCH"
            section = config[section_name] if config.has_section(section_name) else {}
            try:
                components[name] = (
                    self._read_mission(section, default)
                    if name == "mission_config"
                    else self._read_section(section, default, name)
                )
            except (TypeError, ValueError):
                components[name] = default
        return SimulationConfig(**components)

    @staticmethod
    def _ini_key(section: str, name: str) -> str:
        """Retain the two historical rendering key names."""
        return f"slam_{name}" if section == "rendering" else name

    @classmethod
    def _read_section(cls, section: Mapping[str, object], defaults: T, name: str) -> T:
        """Convert only declared scalar fields, ignoring retired and unknown keys."""
        converters = get_type_hints(type(defaults))
        values = {}
        for item in fields(defaults):
            value = section.get(
                cls._ini_key(name, item.name), getattr(defaults, item.name)
            )
            convert = converters[item.name]
            values[item.name] = (
                str(value).casefold() in {"1", "true", "yes", "on"}
                if convert is bool
                else convert(value)
            )
        return type(defaults)(**values)


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
