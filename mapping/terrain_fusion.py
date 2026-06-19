"""Mission-level terrain telemetry fusion.

This service updates the aggregate used for progress reporting and combined
rendering. It does not distribute knowledge to agents and must not influence
active agent decisions.
"""

from typing import Any, Iterable

import pygame

from mapping.terrain_knowledge import (
    TerrainSample,
    fuse_terrain_samples,
)


class TerrainFusionService:
    """Fuse observations into mission telemetry without mutating agents."""

    def __init__(self, control: Any) -> None:
        self.control = control

    def record_scan(self, samples: Iterable[TerrainSample]) -> None:
        """Fuse observations into mission telemetry and update the UI."""
        control = self.control
        map_updated = control.terrain_knowledge.record_samples(samples)

        now = pygame.time.get_ticks() / 1000.0
        if (
            map_updated
            and (now - control.last_explored_update)
            >= control.explored_update_interval
        ):
            control.control_center.explored_percent = round(
                control.terrain_knowledge.explored_ratio() * 100
            )
            control.last_explored_update = now
        if map_updated:
            control.presentation.terrain_heatmap_dirty = True
