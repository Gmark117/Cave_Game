"""Mission-level wall-surface mapping progress from distributed SLAM.

Terrain roughness remains rover navigation data.  Exploration progress instead
measures the red occupied SLAM pixels that cover cave, pillar, and internal-wall
surfaces.  The ground-truth cave mask is used only for mission telemetry; it is
never exposed to a drone planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from mapping.slam_map import OCCUPIED


@dataclass(frozen=True)
class WallMappingSnapshot:
    """Detached team wall-coverage telemetry."""

    mapped_wall_pixels: int
    total_wall_pixels: int
    ratio: float
    complete: bool
    slam_versions: tuple[int, ...] = ()


def exposed_wall_mask(cave_map: np.ndarray) -> np.ndarray:
    """Return wall pixels sharing an orthogonal edge with traversable cave.

    Solid rock interiors are not sensor-visible surfaces and therefore do not
    belong to the exploration objective.  This includes the outer cave shell,
    pillars, and internal walls without requiring a drone to map buried rock.
    """
    cave = np.asarray(cave_map)
    if cave.ndim != 2:
        raise ValueError("cave map must be two-dimensional")
    wall = cave != 0
    floor = ~wall
    adjacent_floor = np.zeros_like(wall, dtype=bool)
    adjacent_floor[1:, :] |= floor[:-1, :]
    adjacent_floor[:-1, :] |= floor[1:, :]
    adjacent_floor[:, 1:] |= floor[:, :-1]
    adjacent_floor[:, :-1] |= floor[:, 1:]
    return wall & adjacent_floor


def wall_mapping_snapshot(
    cave_map: np.ndarray,
    drones: Sequence[Any],
    *,
    confidence_threshold: float = 0.6,
) -> WallMappingSnapshot:
    """Combine local SLAM snapshots into wall-surface coverage telemetry."""
    target = exposed_wall_mask(cave_map)
    observed = np.zeros_like(target, dtype=bool)
    versions: list[int] = []
    threshold = float(confidence_threshold)
    for drone in drones:
        slam = drone.slam_map.snapshot(point_limit=0)
        versions.append(int(slam.version))
        height = min(target.shape[0], slam.occupancy.shape[0])
        width = min(target.shape[1], slam.occupancy.shape[1])
        if height <= 0 or width <= 0:
            continue
        observed[:height, :width] |= (
            (slam.occupancy[:height, :width] == OCCUPIED)
            & (slam.confidence[:height, :width] >= threshold)
        )

    total = int(np.count_nonzero(target))
    mapped = int(np.count_nonzero(target & observed))
    ratio = (mapped / total) if total else 0.0
    return WallMappingSnapshot(
        mapped_wall_pixels=mapped,
        total_wall_pixels=total,
        ratio=ratio,
        complete=bool(total > 0 and mapped == total),
        slam_versions=tuple(versions),
    )
