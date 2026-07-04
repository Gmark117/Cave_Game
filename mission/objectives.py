"""Mission objective policies.

Each objective owns mission-completion semantics. Exploration is the only
implemented objective; planned objectives fail fast instead of reusing
exploration behavior by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from asset_config.gameplay import GameOptions


class MissionObjective(Protocol):
    """Completion policy selected by the mission configuration."""

    name: str

    def is_complete(
        self,
        drones: Sequence[Any],
        rovers: Sequence[Any],
    ) -> bool:
        """Return whether the mission has reached its objective."""


@dataclass(frozen=True)
class ExplorationObjective:
    """Complete when every drone has finished exploration and returned home."""

    name: str = "Exploration"

    def is_complete(
        self,
        drones: Sequence[Any],
        rovers: Sequence[Any],
    ) -> bool:
        """Return whether all deployed drones have completed exploration."""
        if not drones:
            return False
        return all(drone.mission_completed() for drone in drones)


def build_mission_objective(objective_index: int) -> MissionObjective:
    """Create the mission objective selected by the menu configuration."""
    try:
        objective_name = str(GameOptions.MISSION[objective_index])
    except IndexError as exc:
        raise ValueError(
            f"Unknown mission objective index: {objective_index}"
        ) from exc

    if objective_name == "Exploration":
        return ExplorationObjective()

    # Search and Rescue is visible in the menu as planned work. Raising here is
    # clearer than pretending it has Exploration's completion rules.
    raise NotImplementedError(
        f"Mission objective '{objective_name}' is planned but not implemented"
    )
