"""Shared protocol and dependency objects for simulation collaborators.

The mission controller remains the composition root. Collaborators receive
only the small protocol or dependency bundle they need, which keeps ownership
boundaries visible and tests focused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, Sequence, Tuple

import numpy as np

from config.simulation_config import RenderingConfig, SharingConfig
from mapping.terrain_knowledge import TerrainSample


Position = Tuple[int, int]


class TerrainKnowledgeStore(Protocol):
    """Terrain knowledge operations required by mission services."""

    floor_mask: np.ndarray

    def record_samples(self, samples: Iterable[TerrainSample]) -> bool:
        """Fuse visible terrain samples and report whether anything changed."""
        ...

    def snapshot(self) -> Any:
        """Return a detached terrain snapshot for sharing or rendering."""
        ...


class PresentationInvalidator(Protocol):
    """Presentation flag mutated when terrain/SLAM displays become stale."""

    terrain_heatmap_dirty: bool
    selected_drone_heatmap_id: int | None
    selected_rover_heatmap_id: int | None
    show_terrain_heatmap: bool
    show_full_map: bool


class SlamRendererLike(Protocol):
    """Renderer API used by the SLAM view service."""

    surface: Any

    def render(self, *args: Any, **kwargs: Any) -> None:
        """Render into ``surface`` using occupancy or terrain arrays."""
        ...

    def full_map_underlay(self, floor_mask: np.ndarray) -> Any:
        """Return the cached full-cave underlay surface."""
        ...


@dataclass
class TerrainFusionDependencies:
    """Inputs required to fuse rover terrain and invalidate its heatmap."""

    terrain_knowledge: TerrainKnowledgeStore
    presentation: PresentationInvalidator


@dataclass(frozen=True)
class TerrainSharingDependencies:
    """Inputs required by proximity-based terrain and SLAM sharing."""

    sharing: SharingConfig
    cave_map: np.ndarray
    map_width: int
    map_height: int
    terrain_knowledge: TerrainKnowledgeStore
    get_drones: Callable[[], Sequence[Any]]
    get_rovers: Callable[[], Sequence[Any]]
    presentation: PresentationInvalidator
    simulation_time: Callable[[], float]
    runtime_trace: Any | None = None
    periodic_rover_sharing_enabled: bool = True
    on_drone_contact: Callable[[int, int], None] | None = None
    on_drone_rover_contact: Callable[[int], None] | None = None


@dataclass(frozen=True)
class RoverTargetDependencies:
    """Inputs required by rover frontier staging and reservation."""

    cave_map: np.ndarray
    terrain_knowledge: TerrainKnowledgeStore
    assignment_lock: Any
    assignments: dict[int, Position]
    completed_targets: set[Position]
    norm_width: int
    norm_height: int
    get_frontier_candidates: Callable[[], Sequence[Any]] = lambda: ()
    should_hold_position: Callable[[int], bool] = lambda _rover_id: False
    simulation_time: Callable[[], float] = lambda: 0.0
    runtime_trace: Any | None = None


@dataclass(frozen=True)
class SlamViewDependencies:
    """Inputs required to render combined or per-agent SLAM views."""

    rendering: RenderingConfig
    terrain_knowledge: TerrainKnowledgeStore
    presentation: PresentationInvalidator
    slam_renderer: SlamRendererLike
    get_drones: Callable[[], Sequence[Any]]
    get_window: Callable[[], Any]
    get_rovers: Callable[[], Sequence[Any]] = lambda: ()


@dataclass(frozen=True)
class MissionDebugDependencies:
    """Inputs required to build mission debug text."""

    get_drones: Callable[[], Sequence[Any]]
    presentation: PresentationInvalidator
    dirty_map_count: Callable[[], int]
    simulation_time: Callable[[], float]
    frame_profiler: Any | None = None
    runtime_trace: Any | None = None
    get_rovers: Callable[[], Sequence[Any]] = lambda: ()


@dataclass(frozen=True)
class MissionRendererDependencies:
    """Inputs required for full-frame mission rendering."""

    get_window: Callable[[], Any]
    slam_view: Any
    debug_info: Any
    get_control_center: Callable[[], Any]
    get_drones: Callable[[], Sequence[Any]]
    get_rovers: Callable[[], Sequence[Any]]
    presentation: PresentationInvalidator
    is_paused: Callable[[], bool]
    is_music_enabled: Callable[[], bool]
    sector_renderer: Any | None = None
    get_sector_snapshot: Callable[[], Any] = lambda: None
    is_exploration_complete: Callable[[], bool] = lambda: False
    get_docked_drone_ids: Callable[[], frozenset[int]] = frozenset
    highway_renderer: Any | None = None
    get_highway_snapshot: Callable[[int], Any] = lambda _rover_id: None


@dataclass(frozen=True)
class DroneMovementDependencies:
    """Callbacks used by drone movement without retaining mission control."""

    compute_path: Callable[[Position, Position], list[Position]]
    simulation_time: Callable[[], float]
    pause_checkpoint: Callable[[], bool]
    wait_simulation_delay: Callable[[float], bool]
    runtime_trace: Any | None = None
    get_drone_positions: Callable[
        [], Sequence[tuple[int, Position]]
    ] = lambda: ()
    physical_contact_checkpoint: Callable[[int], None] | None = None
    compute_path_segment: Callable[[Position, Position], Any] | None = None
    sector_check_in: Callable[[int, int | None], Any] | None = None
    sector_assignment: Callable[[int], Any] | None = None
    get_check_in_position: Callable[[], Position] | None = None
    rendezvous_endpoint_missed: Callable[
        [int, Position], Position
    ] | None = None
    exploration_check_in: Callable[[int, Any | None], Any] | None = None
    request_exploration_report_stop: Callable[[int], bool] | None = None
    request_exploration_dock: Callable[[int], Any] | None = None
    is_exploration_docked: Callable[[int], bool] | None = None
    exploration_contact: Callable[[int], bool] | None = None
    exploration_assignment: Callable[[int], Any] | None = None
    exploration_energy_return: Callable[
        [int, float, float, float], Any
    ] | None = None


@dataclass(frozen=True)
class DroneSensorDependencies:
    """Inputs used by drone sensing and local terrain sampling."""

    terrain_roughness: np.ndarray
    simulation_time: Callable[[], float]
    record_terrain_scan: Callable[[Iterable[TerrainSample]], None]
    runtime_trace: Any | None = None


@dataclass(frozen=True)
class RoverNavigationDependencies:
    """Callbacks used by rover navigation without retaining mission control."""

    rover_targets: Any
    compute_rover_path: Callable[
        [int, Position, Position], list[Position]
    ]
    simulation_time: Callable[[], float] = lambda: 0.0
    runtime_trace: Any | None = None
    announce_rendezvous: Callable[[Position], Any] = lambda _position: None
    rendezvous_departure_ready: Callable[[Position], bool] = (
        lambda _position: True
    )
    rendezvous_arrived: Callable[[Position], bool] = lambda _position: True
    rendezvous_departed: Callable[[Position], bool] = lambda _position: True
