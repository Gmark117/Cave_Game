"""Detached runtime values consumed by the control-center UI."""

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from agents.drone_runtime_state import DroneSnapshot
from asset_config.rendering import DroneColors, RoverColors


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class AgentRosterEntry:
    """Stable presentation identity for one control-center roster slot."""

    id: int
    name: str
    color: Color


DRONE_ROSTER = (
    AgentRosterEntry(0, "Blinky", DroneColors.RED.value),
    AgentRosterEntry(1, "Pinky", DroneColors.PINK.value),
    AgentRosterEntry(2, "Inky", DroneColors.L_BLUE.value),
    AgentRosterEntry(3, "Clyde", DroneColors.ORANGE.value),
    AgentRosterEntry(4, "Sue", DroneColors.PURPLE.value),
    AgentRosterEntry(5, "Tim", DroneColors.BROWN.value),
    AgentRosterEntry(6, "Funky", DroneColors.GREEN.value),
    AgentRosterEntry(7, "Kinky", DroneColors.GOLD.value),
)

ROVER_ROSTER = (
    AgentRosterEntry(0, "Huey", RoverColors.RED.value),
    AgentRosterEntry(1, "Dewey", RoverColors.BLUE.value),
    AgentRosterEntry(2, "Louie", RoverColors.GREEN.value),
)


@dataclass(frozen=True)
class DroneStatusView:
    """Immutable control-center state copied from one runtime drone."""

    id: int
    name: str
    color: Color
    battery: int
    status: str
    show_path: bool
    show_vision: bool


@dataclass(frozen=True)
class RoverStatusView:
    """Immutable control-center state copied from one runtime rover."""

    id: int
    name: str
    color: Color
    battery: int
    status: str


@dataclass(frozen=True)
class ControlCenterViewModel:
    """Complete immutable display data for one control-center frame."""

    elapsed_time: str
    explored_percent: int
    active_tab: str
    drone_statuses: tuple[DroneStatusView, ...]
    rover_statuses: tuple[RoverStatusView, ...]
    show_terrain_heatmap: bool
    selected_drone_heatmap_id: int | None
    debug_lines: tuple[str, ...]

    def __init__(
        self,
        elapsed_time: str,
        explored_percent: int,
        active_tab: str,
        drone_statuses: Iterable[DroneStatusView],
        rover_statuses: Iterable[RoverStatusView],
        show_terrain_heatmap: bool,
        selected_drone_heatmap_id: int | None,
        debug_lines: Iterable[str],
    ) -> None:
        object.__setattr__(self, "elapsed_time", str(elapsed_time))
        object.__setattr__(
            self,
            "explored_percent",
            int(explored_percent),
        )
        object.__setattr__(self, "active_tab", str(active_tab))
        object.__setattr__(
            self,
            "drone_statuses",
            tuple(drone_statuses),
        )
        object.__setattr__(
            self,
            "rover_statuses",
            tuple(rover_statuses),
        )
        object.__setattr__(
            self,
            "show_terrain_heatmap",
            bool(show_terrain_heatmap),
        )
        object.__setattr__(
            self,
            "selected_drone_heatmap_id",
            selected_drone_heatmap_id,
        )
        object.__setattr__(
            self,
            "debug_lines",
            tuple(str(line) for line in debug_lines),
        )


def _roster_name(roster: tuple[AgentRosterEntry, ...], agent_id: int) -> str:
    if 0 <= agent_id < len(roster):
        return roster[agent_id].name
    return f"Agent {agent_id + 1}"


def _drone_status(snapshot: DroneSnapshot) -> str:
    if snapshot.done:
        return "Done"
    if snapshot.returning_home:
        return "Homing"
    if snapshot.explored:
        return "Deployed"
    return "Ready"


def build_drone_status_views(
    drones: Iterable[Any],
    snapshots: Iterable[DroneSnapshot],
) -> tuple[DroneStatusView, ...]:
    """Copy current drone display state into detached immutable values."""
    drone_list = tuple(drones)
    snapshot_list = tuple(snapshots)
    if len(drone_list) != len(snapshot_list):
        raise ValueError("drones and snapshots must have the same length")
    return tuple(
        DroneStatusView(
            id=int(drone.id),
            name=_roster_name(DRONE_ROSTER, int(drone.id)),
            color=tuple(drone.color),
            battery=int(snapshot.battery),
            status=_drone_status(snapshot),
            show_path=snapshot.show_path,
            show_vision=snapshot.show_vision,
        )
        for drone, snapshot in zip(drone_list, snapshot_list)
    )


def build_rover_status_views(
    rovers: Iterable[Any],
) -> tuple[RoverStatusView, ...]:
    """Copy current rover display state into detached immutable values."""
    return tuple(
        RoverStatusView(
            id=int(rover.id),
            name=_roster_name(ROVER_ROSTER, int(rover.id)),
            color=tuple(rover.color),
            battery=int(rover.battery),
            status=str(rover.status),
        )
        for rover in rovers
    )
