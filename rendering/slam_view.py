"""Mission SLAM and terrain heatmap view orchestration."""

import time
from typing import Any, List, Optional, Tuple

import numpy as np


class SlamViewService:
    """Build and draw cached SLAM/terrain map surfaces for MissionControl."""

    def __init__(self, control: Any) -> None:
        self.control = control
        settings = getattr(control, "settings", None)
        self.refresh_interval = max(
            0.0,
            float(getattr(settings, "slam_render_interval", 0.1)),
        )
        self.last_refresh_time: float | None = None

    def refresh(self, drone_id: Optional[int] = None) -> None:
        """Rebuild the cached SLAM map surface.

        Renders occupancy by default. Renders terrain roughness when the global
        terrain heatmap toggle is enabled. If a per-drone heatmap is selected,
        renders only that drone's SLAM/terrain data.
        """
        control = self.control
        if not control.drones:
            control.slam_renderer.surface.fill((0, 0, 0, 0))
            control.presentation.terrain_heatmap_dirty = False
            self.last_refresh_time = time.perf_counter()
            return

        h, w = control.floor_mask.shape
        render_tail = int(getattr(control.settings, "slam_render_point_tail", 400))

        selected_id = control.presentation.selected_drone_heatmap_id
        if selected_id is not None and 0 <= selected_id < len(control.drones):
            self._render_selected_drone(selected_id, h, w, render_tail)
        else:
            self._render_combined(h, w, render_tail)

        control.presentation.terrain_heatmap_dirty = False
        self.last_refresh_time = time.perf_counter()

    def draw(self) -> None:
        """Blit the cached SLAM map overlay, refreshing it when dirty."""
        control = self.control
        if not control.drones:
            return

        any_dirty = control.presentation.terrain_heatmap_dirty
        for drone in control.drones:
            if (
                getattr(drone, "slam_map", None) is not None
                and drone.slam_map.dirty
            ):
                any_dirty = True
                break

        now = time.perf_counter()
        refresh_due = (
            self.last_refresh_time is None
            or now < self.last_refresh_time
            or (now - self.last_refresh_time) >= self.refresh_interval
        )
        if any_dirty and refresh_due:
            self.refresh()

        control.game.window.blit(control.slam_renderer.surface, (0, 0))

    def _render_selected_drone(
        self, selected_id: int, h: int, w: int, render_tail: int
    ) -> None:
        control = self.control
        drone = control.drones[selected_id]

        with drone.slam_lock:
            occ = drone.slam_map.occupancy.copy()
            conf = drone.slam_map.confidence.copy()
            points = drone.slam_map.recent_points(render_tail)
            drone.slam_map.dirty = False

        padded_occ = np.full((h, w), -1, dtype=np.int8)
        padded_conf = np.zeros((h, w), dtype=np.float32)

        eh = min(h, occ.shape[0])
        ew = min(w, occ.shape[1])
        if eh > 0 and ew > 0:
            padded_occ[:eh, :ew] = occ[:eh, :ew]
            padded_conf[:eh, :ew] = conf[:eh, :ew]

        if control.presentation.show_terrain_heatmap:
            terrain = drone.terrain_knowledge.snapshot()
            control.slam_renderer.render(
                None,
                None,
                points,
                draw_points=True,
                roughness=terrain.roughness,
                roughness_conf=terrain.confidence,
            )
        else:
            control.slam_renderer.render(
                padded_occ, padded_conf, points, draw_points=False
            )

    def _render_combined(self, h: int, w: int, render_tail: int) -> None:
        control = self.control
        combined_occ = np.full((h, w), -1, dtype=np.int8)
        combined_conf = np.zeros((h, w), dtype=np.float32)
        combined_points: List[Tuple[int, int]] = []

        for drone in control.drones:
            with drone.slam_lock:
                occ = drone.slam_map.occupancy
                conf = drone.slam_map.confidence
                combined_points.extend(drone.slam_map.recent_points(render_tail))
                drone.slam_map.dirty = False

            eh = min(h, occ.shape[0], conf.shape[0])
            ew = min(w, occ.shape[1], conf.shape[1])
            if eh <= 0 or ew <= 0:
                continue

            target_conf = combined_conf[:eh, :ew]
            source_conf = conf[:eh, :ew]
            higher_conf = source_conf > target_conf
            combined_occ[:eh, :ew][higher_conf] = occ[:eh, :ew][higher_conf]
            target_conf[higher_conf] = source_conf[higher_conf]

        if control.presentation.show_terrain_heatmap:
            terrain = control.terrain_knowledge.snapshot()
            control.slam_renderer.render(
                None,
                None,
                combined_points,
                draw_points=True,
                roughness=terrain.roughness,
                roughness_conf=terrain.confidence,
            )
        else:
            control.slam_renderer.render(
                combined_occ, combined_conf, combined_points, draw_points=False
            )
