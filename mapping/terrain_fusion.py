"""Mission-level terrain telemetry fusion for rover route planning/rendering.

Terrain roughness is deliberately separate from exploration completion.  Wall
mapping progress is derived from distributed SLAM occupancy instead.
"""

from typing import Iterable

from mapping.terrain_knowledge import TerrainSample
from contracts import TerrainFusionDependencies


class TerrainFusionService:
    """Fuse observations into mission telemetry without mutating agents."""

    def __init__(self, dependencies: TerrainFusionDependencies) -> None:
        """Store mission-level telemetry dependencies."""
        self.dependencies = dependencies

    def record_scan(self, samples: Iterable[TerrainSample]) -> None:
        """Fuse observations into mission telemetry and update the UI."""
        dependencies = self.dependencies
        map_updated = dependencies.terrain_knowledge.record_samples(samples)

        if map_updated:
            dependencies.presentation.terrain_heatmap_dirty = True
