"""Focused dependency objects for mission services.

`MissionControl` remains the composition root. Services receive only the
collaborators they need, which keeps ownership boundaries visible and tests
small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, Sequence, Tuple

import numpy as np

from SimulationConfig import RenderingConfig, SharingConfig
from mapping.terrain_knowledge import TerrainSample


Position = Tuple[int, int]


class TerrainKnowledgeStore(Protocol):
    """Terrain knowledge operations required by mission services."""

    floor_mask: np.ndarray

    def record_samples(self, samples: Iterable[TerrainSample]) -> bool: ...

    def explored_ratio(self) -> float: ...

    def snapshot(self) -> Any: ...


class ProgressDisplay(Protocol):
    """Control-center progress display boundary."""

    def set_explored_percent(self, value: int) -> None: ...


class PresentationInvalidator(Protocol):
    """Presentation flag mutated when terrain/SLAM displays become stale."""

    terrain_heatmap_dirty: bool
    selected_drone_heatmap_id: int | None
    show_terrain_heatmap: bool


class SlamRendererLike(Protocol):
    """Renderer API used by the SLAM view service."""

    surface: Any

    def render(self, *args: Any, **kwargs: Any) -> None: ...


@dataclass
class TerrainFusionDependencies:
    terrain_knowledge: TerrainKnowledgeStore
    get_control_center: Callable[[], ProgressDisplay]
    presentation: PresentationInvalidator
    simulation_time: Callable[[], float]
    explored_update_interval: float
    last_explored_update: float = 0.0


@dataclass(frozen=True)
class TerrainSharingDependencies:
    sharing: SharingConfig
    cave_map: np.ndarray
    map_width: int
    map_height: int
    terrain_knowledge: TerrainKnowledgeStore
    get_drones: Callable[[], Sequence[Any]]
    get_rovers: Callable[[], Sequence[Any]]
    presentation: PresentationInvalidator
    simulation_time: Callable[[], float]


@dataclass(frozen=True)
class RoverTargetDependencies:
    cave_map: np.ndarray
    terrain_knowledge: TerrainKnowledgeStore
    assignment_lock: Any
    assignments: dict[int, Position]
    completed_targets: set[Position]
    norm_width: int
    norm_height: int


@dataclass(frozen=True)
class SlamViewDependencies:
    rendering: RenderingConfig
    terrain_knowledge: TerrainKnowledgeStore
    presentation: PresentationInvalidator
    slam_renderer: SlamRendererLike
    get_drones: Callable[[], Sequence[Any]]
    get_window: Callable[[], Any]


@dataclass(frozen=True)
class MissionDebugDependencies:
    get_drones: Callable[[], Sequence[Any]]
    presentation: PresentationInvalidator
    dirty_map_count: Callable[[], int]
    simulation_time: Callable[[], float]
    frame_profiler: Any | None = None


@dataclass(frozen=True)
class MissionRendererDependencies:
    get_window: Callable[[], Any]
    slam_view: Any
    debug_info: Any
    get_control_center: Callable[[], Any]
    get_drones: Callable[[], Sequence[Any]]
    get_rovers: Callable[[], Sequence[Any]]
    presentation: PresentationInvalidator
    is_paused: Callable[[], bool]


@dataclass(frozen=True)
class DroneMovementDependencies:
    compute_path: Callable[[Position, Position], list[Position]]
    simulation_time: Callable[[], float]
    pause_checkpoint: Callable[[], bool]
    wait_simulation_delay: Callable[[float], bool]


@dataclass(frozen=True)
class DroneSensorDependencies:
    terrain_roughness: np.ndarray
    simulation_time: Callable[[], float]
    record_terrain_scan: Callable[[Iterable[TerrainSample]], None]


@dataclass(frozen=True)
class RoverNavigationDependencies:
    rover_targets: Any
    compute_rover_path: Callable[[Position, Position], list[Position]]
