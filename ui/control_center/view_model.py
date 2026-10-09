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
    detail: str = ""
    target: tuple[int, int] | None = None


@dataclass(frozen=True)
class RoverStatusView:
    """Immutable control-center state copied from one runtime rover."""

    id: int
    name: str
    color: Color
    battery: int
    status: str
    detail: str = ""
    target: tuple[int, int] | None = None
    show_highway: bool = True


@dataclass(frozen=True)
class ControlCenterViewModel:
    """Complete immutable display data for one control-center frame."""

    elapsed_time: str
    explored_percent: float
    active_tab: str
    drone_statuses: tuple[DroneStatusView, ...]
    rover_statuses: tuple[RoverStatusView, ...]
    show_terrain_heatmap: bool
    selected_drone_heatmap_id: int | None
    selected_rover_heatmap_id: int | None
    show_full_map: bool
    debug_lines: tuple[str, ...]
    system_lines: tuple[str, ...]
    is_paused: bool
    music_enabled: bool
    exploration_complete: bool

    def __init__(
        self,
        elapsed_time: str,
        explored_percent: float,
        active_tab: str,
        drone_statuses: Iterable[DroneStatusView],
        rover_statuses: Iterable[RoverStatusView],
        show_terrain_heatmap: bool,
        selected_drone_heatmap_id: int | None,
        debug_lines: Iterable[str],
        is_paused: bool = False,
        music_enabled: bool = True,
        show_full_map: bool = False,
        selected_rover_heatmap_id: int | None = None,
        system_lines: Iterable[str] = (),
        exploration_complete: bool = False,
    ) -> None:
        """Copy mutable mission values into immutable display values."""
        object.__setattr__(self, "elapsed_time", str(elapsed_time))
        object.__setattr__(
            self,
            "explored_percent",
            float(explored_percent),
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
            "selected_rover_heatmap_id",
            selected_rover_heatmap_id,
        )
        object.__setattr__(self, "show_full_map", bool(show_full_map))
        object.__setattr__(
            self,
            "debug_lines",
            tuple(str(line) for line in debug_lines),
        )
        object.__setattr__(
            self,
            "system_lines",
            tuple(str(line) for line in system_lines),
        )
        object.__setattr__(self, "is_paused", bool(is_paused))
        object.__setattr__(self, "music_enabled", bool(music_enabled))
        object.__setattr__(
            self,
            "exploration_complete",
            bool(exploration_complete),
        )


def _roster_name(roster: tuple[AgentRosterEntry, ...], agent_id: int) -> str:
    """Return a themed roster name, falling back for extra agents."""
    if 0 <= agent_id < len(roster):
        return roster[agent_id].name
    return f"Agent {agent_id + 1}"


def _drone_status(snapshot: DroneSnapshot) -> str:
    """Translate drone runtime flags into a user-facing status label."""
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
    views = []
    for drone, snapshot in zip(drone_list, snapshot_list):
        activity = None
        controller = getattr(drone, "movement_controller", None)
        activity_snapshot = getattr(controller, "activity_snapshot", None)
        if callable(activity_snapshot):
            activity = activity_snapshot()
        views.append(DroneStatusView(
            id=int(drone.id),
            name=_roster_name(DRONE_ROSTER, int(drone.id)),
            color=tuple(drone.color),
            battery=int(snapshot.battery),
            status=(
                _drone_status(snapshot)
                if activity is None
                else str(activity.state)
            ),
            show_path=snapshot.show_path,
            show_vision=snapshot.show_vision,
            detail="" if activity is None else str(activity.detail),
            target=None if activity is None else activity.target,
        ))
    return tuple(views)


def build_rover_status_views(
    rovers: Iterable[Any],
) -> tuple[RoverStatusView, ...]:
    """Copy current rover display state into detached immutable values."""
    views = []
    for rover in rovers:
        snapshot_method = getattr(rover, "snapshot", None)
        snapshot = snapshot_method() if callable(snapshot_method) else rover
        target = getattr(snapshot, "target", getattr(rover, "target", None))
        status = str(getattr(snapshot, "status", rover.status))
        detail = _rover_detail(status, target)
        views.append(RoverStatusView(
            id=int(rover.id),
            name=_roster_name(ROVER_ROSTER, int(rover.id)),
            color=tuple(rover.color),
            battery=int(getattr(snapshot, "battery", rover.battery)),
            status=_rover_status(status),
            detail=detail,
            target=target,
            show_highway=bool(getattr(snapshot, "show_highway", True)),
        ))
    return tuple(views)


def _rover_status(status: str) -> str:
    """Translate rover runtime status labels into compact UI text."""
    if status == "Advancing":
        return "Moving"
    return status


def _rover_detail(
    status: str,
    target: tuple[int, int] | None,
) -> str:
    """Explain the rover state without making the renderer inspect policy."""
    target_text = "none" if target is None else f"{target[0]},{target[1]}"
    if status == "Rendezvous":
        return "holding for drone rendezvous"
    if status == "Ready":
        return "no reachable component target"
    if status == "Updating":
        return "selecting component target"
    if status == "Announcing":
        return f"awaiting endpoint acks for {target_text}"
    if status == "Staging":
        return f"staged at {target_text}"
    if status == "Advancing":
        return f"target {target_text}"
    return f"target {target_text}"
