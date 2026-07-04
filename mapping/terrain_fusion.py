"""Mission-level terrain telemetry fusion.

This service updates the aggregate used for progress reporting and combined
rendering. It does not distribute knowledge to agents and must not influence
active agent decisions.
"""

from typing import Iterable

from mapping.terrain_knowledge import (
    TerrainSample,
    fuse_terrain_samples,
)
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

        now = dependencies.simulation_time()
        # Progress text is relatively expensive to redraw every sensor tick, so
        # update it at a small interval while still marking the heatmap dirty.
        if (
            map_updated
            and (now - dependencies.last_explored_update)
            >= dependencies.explored_update_interval
        ):
            dependencies.get_control_center().set_explored_percent(
                round(
                    dependencies.terrain_knowledge.explored_ratio() * 100
                )
            )
            dependencies.last_explored_update = now
        if map_updated:
            dependencies.presentation.terrain_heatmap_dirty = True
