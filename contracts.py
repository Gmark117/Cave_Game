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


@dataclass(frozen=True)
class RoverTargetDependencies:
    """Inputs required by rover target scoring and reservation."""

    cave_map: np.ndarray
    terrain_knowledge: TerrainKnowledgeStore
    assignment_lock: Any
    assignments: dict[int, Position]
    completed_targets: set[Position]
    norm_width: int
    norm_height: int


@dataclass(frozen=True)
class SlamViewDependencies:
    """Inputs required to render combined or per-drone SLAM views."""

    rendering: RenderingConfig
    terrain_knowledge: TerrainKnowledgeStore
    presentation: PresentationInvalidator
    slam_renderer: SlamRendererLike
    get_drones: Callable[[], Sequence[Any]]
    get_window: Callable[[], Any]


@dataclass(frozen=True)
class MissionDebugDependencies:
    """Inputs required to build mission debug text."""

    get_drones: Callable[[], Sequence[Any]]
    presentation: PresentationInvalidator
    dirty_map_count: Callable[[], int]
    simulation_time: Callable[[], float]
    frame_profiler: Any | None = None
    runtime_trace: Any | None = None


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
    waypoint_renderer: Any | None = None


@dataclass(frozen=True)
class DroneMovementDependencies:
    """Callbacks used by drone movement without retaining mission control."""

    simulation_time: Callable[[], float]
    pause_checkpoint: Callable[[], bool]
    wait_simulation_delay: Callable[[float], bool]
    runtime_trace: Any | None = None
    waypoint_graph: Any | None = None


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
    compute_rover_path: Callable[[Position, Position], list[Position]]
