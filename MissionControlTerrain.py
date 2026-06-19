"""Terrain, sharing, and SLAM facade for MissionControl.

This mixin keeps the existing MissionControl method names stable while
delegating the actual work to focused services. That lets older call sites
continue to use `record_terrain_scan`, `_share_terrain_with_nearby_drones`,
`draw_terrain_heatmap`, and rover target helpers while the responsibilities
now live in smaller modules.
"""

from typing import List, Optional, Tuple

import numpy as np

from mapping.rover_targets import RoverTargetService
from mapping.terrain_fusion import TerrainFusionService, TerrainSample
from mapping.terrain_sharing import TerrainSharingService
from mission.debug_info import MissionDebugInfo
from rendering.slam_view import SlamViewService


class MissionControlTerrainMixin:
    """Compatibility facade for terrain and SLAM mission services."""

    def _init_terrain_services(self) -> None:
        """Create terrain/SLAM helper services for this MissionControl."""
        self.terrain_fusion = TerrainFusionService(self)
        self.terrain_sharing = TerrainSharingService(self)
        self.rover_targets = RoverTargetService(self)
        self.slam_view = SlamViewService(self)
        self.debug_info = MissionDebugInfo(self)

    def _ensure_terrain_services(self) -> None:
        """Lazily initialize services for tests or legacy construction paths."""
        if not hasattr(self, "terrain_fusion"):
            self._init_terrain_services()

    def build_debug_lines(self) -> List[str]:
        """Build a small set of runtime debug lines for the control panel."""
        self._ensure_terrain_services()
        return self.debug_info.build_lines()

    def record_terrain_scan(self, samples: List[TerrainSample]) -> None:
        """Fuse drone terrain observations into the shared known-terrain maps."""
        self._ensure_terrain_services()
        self.terrain_fusion.record_scan(samples)

    def toggle_terrain_heatmap(self) -> None:
        """Toggle visibility of the SLAM map overlay."""
        self.presentation.toggle_terrain_heatmap()
        self._update_visibility_state()

    def toggle_drone_heatmap(self, drone_id: int) -> None:
        """Toggle per-drone SLAM map for a specific drone id."""
        if drone_id < 0 or drone_id >= len(self.drones):
            return
        self.presentation.toggle_drone_heatmap(drone_id)
        self._update_visibility_state()

    def _update_visibility_state(self) -> None:
        """Update drone path/vision visibility from heatmap selection state."""
        selected_id = self.presentation.selected_drone_heatmap_id

        if selected_id is not None:
            show_path_for_selected = not self.presentation.show_terrain_heatmap
            for i, drone in enumerate(self.drones):
                drone.show_vision = i == selected_id
                drone.show_path = (i == selected_id) and show_path_for_selected
        else:
            show_overlays = not self.presentation.show_terrain_heatmap
            for drone in self.drones:
                drone.show_path = show_overlays
                drone.show_vision = show_overlays

    def _has_line_of_sight(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Return True when segment a->b does not cross cave walls."""
        self._ensure_terrain_services()
        return self.terrain_sharing.has_line_of_sight(a, b)

    def _maps_differ_enough(
        self,
        source_roughness: np.ndarray,
        source_confidence: np.ndarray,
        target_roughness: np.ndarray,
        target_confidence: np.ndarray,
    ) -> bool:
        """Return True when sharing is likely to add meaningful information."""
        self._ensure_terrain_services()
        return self.terrain_sharing.maps_differ_enough(
            source_roughness,
            source_confidence,
            target_roughness,
            target_confidence,
        )

    def _slam_maps_differ_enough(
        self,
        source_occ: np.ndarray,
        source_conf: np.ndarray,
        target_occ: np.ndarray,
        target_conf: np.ndarray,
    ) -> bool:
        """Return True when SLAM occupancy sharing adds meaningful info."""
        self._ensure_terrain_services()
        return self.terrain_sharing.slam_maps_differ_enough(
            source_occ,
            source_conf,
            target_occ,
            target_conf,
        )

    def _share_terrain_with_nearby_drones(self, drone_id: int) -> None:
        """Check for nearby drones and exchange terrain and SLAM data."""
        self._ensure_terrain_services()
        self.terrain_sharing.share_with_nearby_drones(drone_id)

    def _share_terrain_with_rovers(self) -> None:
        """Share terrain knowledge from all drones with rovers."""
        self._ensure_terrain_services()
        self.terrain_sharing.share_with_rovers()

    def _refresh_slam_map(self, drone_id: Optional[int] = None) -> None:
        """Rebuild the cached SLAM map surface."""
        self._ensure_terrain_services()
        self.slam_view.refresh(drone_id)

    def draw_terrain_heatmap(self) -> None:
        """Blit the SLAM map overlay."""
        self._ensure_terrain_services()
        self.slam_view.draw()

    def acquire_rover_target(
        self, rover_id: int, current_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Choose and reserve a discovered rough-terrain target for a rover."""
        self._ensure_terrain_services()
        return self.rover_targets.acquire(rover_id, current_pos)

    def release_rover_target(self, rover_id: int, completed: bool = False) -> None:
        """Release or mark complete a rover terrain target reservation."""
        self._ensure_terrain_services()
        self.rover_targets.release(rover_id, completed)
