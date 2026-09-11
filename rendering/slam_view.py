"""Mission SLAM and terrain heatmap view orchestration."""

import time
from typing import Any, List, Optional, Tuple

import numpy as np

from contracts import SlamViewDependencies


class SlamViewService:
    """Build and draw cached SLAM/terrain map surfaces for MissionControl."""

    def __init__(self, dependencies: SlamViewDependencies) -> None:
        """Store view dependencies and initialize render-cache versions."""
        self.dependencies = dependencies
        self.refresh_interval = dependencies.rendering.refresh_interval
        self.last_refresh_time: float | None = None
        self.rendered_versions: dict[tuple[str, int], int] = {}

    def dirty_map_count(self) -> int:
        """Return newer maps participating in the currently selected view."""
        dependencies = self.dependencies
        drones = dependencies.get_drones()
        rovers = dependencies.get_rovers()
        selected_rover_id = getattr(
            dependencies.presentation,
            "selected_rover_heatmap_id",
            None,
        )
        if (
            selected_rover_id is not None
            and 0 <= selected_rover_id < len(rovers)
        ):
            return int(self._map_is_dirty(
                "rover",
                selected_rover_id,
                rovers[selected_rover_id].slam_map,
            ))
        selected_drone_id = (
            dependencies.presentation.selected_drone_heatmap_id
        )
        if (
            selected_drone_id is not None
            and 0 <= selected_drone_id < len(drones)
        ):
            return int(self._map_is_dirty(
                "drone",
                selected_drone_id,
                drones[selected_drone_id].slam_map,
            ))
        agents = drones if drones else rovers
        agent_kind = "drone" if drones else "rover"
        return sum(
            1
            for agent_id, agent in enumerate(agents)
            if self._map_is_dirty(agent_kind, agent_id, agent.slam_map)
        )

    def refresh(self, drone_id: Optional[int] = None) -> None:
        """Rebuild the cached SLAM map surface.

        Renders occupancy by default. Renders terrain roughness when the global
        terrain heatmap toggle is enabled. If a per-drone heatmap is selected,
        renders only that drone's SLAM/terrain data.
        """
        dependencies = self.dependencies
        drones = dependencies.get_drones()
        rovers = dependencies.get_rovers()
        if not drones and not rovers:
            dependencies.slam_renderer.surface.fill((0, 0, 0, 0))
            dependencies.presentation.terrain_heatmap_dirty = False
            self.rendered_versions.clear()
            self.last_refresh_time = time.perf_counter()
            return

        h, w = dependencies.terrain_knowledge.floor_mask.shape
        render_tail = dependencies.rendering.point_tail

        selected_drone_id = (
            drone_id
            if drone_id is not None
            else dependencies.presentation.selected_drone_heatmap_id
        )
        selected_rover_id = getattr(
            dependencies.presentation,
            "selected_rover_heatmap_id",
            None,
        )
        if drone_id is not None:
            selected_rover_id = None
        if (
            selected_rover_id is not None
            and 0 <= selected_rover_id < len(rovers)
        ):
            self._render_selected_agent(
                "rover",
                selected_rover_id,
                rovers[selected_rover_id],
                h,
                w,
                render_tail,
            )
        elif (
            selected_drone_id is not None
            and 0 <= selected_drone_id < len(drones)
        ):
            self._render_selected_agent(
                "drone",
                selected_drone_id,
                drones[selected_drone_id],
                h,
                w,
                render_tail,
            )
        else:
            self._render_combined(
                h,
                w,
                render_tail,
                drones if drones else rovers,
                "drone" if drones else "rover",
            )

        dependencies.presentation.terrain_heatmap_dirty = False
        self.last_refresh_time = time.perf_counter()

    def draw(self) -> None:
        """Blit the cached SLAM map overlay, refreshing it when dirty."""
        dependencies = self.dependencies
        if (
            not dependencies.get_drones()
            and not dependencies.get_rovers()
        ):
            return

        # This flag also represents a selected/combined view identity change.
        # Occupancy views must therefore rebuild even when their underlying
        # SLAM version was already cached while rendering another view.
        view_dirty = (
            dependencies.presentation.terrain_heatmap_dirty
            or self._current_view_is_dirty()
        )

        now = time.perf_counter()
        refresh_due = (
            self.last_refresh_time is None
            or now < self.last_refresh_time
            or (now - self.last_refresh_time) >= self.refresh_interval
        )
        if view_dirty and refresh_due:
            self.refresh()

        dependencies.get_window().blit(dependencies.slam_renderer.surface, (0, 0))

    def draw_static_background(self) -> bool:
        """Draw the full cave map as an opaque pre-dimmed background."""
        dependencies = self.dependencies
        if not getattr(dependencies.presentation, "show_full_map", False):
            return False

        underlay = dependencies.slam_renderer.full_map_underlay(
            dependencies.terrain_knowledge.floor_mask,
        )
        if underlay is None:
            return False

        dependencies.get_window().blit(underlay, (0, 0))
        return True

    def _render_selected_agent(
        self,
        agent_kind: str,
        selected_id: int,
        agent: Any,
        h: int,
        w: int,
        render_tail: int,
    ) -> None:
        """Render only the selected drone or rover's local knowledge."""
        dependencies = self.dependencies
        slam = agent.slam_map.snapshot(point_limit=render_tail)
        occ = slam.occupancy
        conf = slam.confidence
        points = list(slam.point_cloud)

        padded_occ = np.full((h, w), -1, dtype=np.int8)
        padded_conf = np.zeros((h, w), dtype=np.float32)

        # Drones may have differently shaped maps in tests; pad into the mission
        # terrain shape before handing arrays to the renderer.
        eh = min(h, occ.shape[0])
        ew = min(w, occ.shape[1])
        if eh > 0 and ew > 0:
            padded_occ[:eh, :ew] = occ[:eh, :ew]
            padded_conf[:eh, :ew] = conf[:eh, :ew]

        if dependencies.presentation.show_terrain_heatmap:
            terrain = agent.terrain_knowledge.snapshot()
            dependencies.slam_renderer.render(
                None,
                None,
                points,
                draw_points=True,
                roughness=terrain.roughness,
                roughness_conf=terrain.confidence,
            )
        else:
            dependencies.slam_renderer.render(
                padded_occ,
                padded_conf,
                points,
                draw_points=False,
            )
        self.rendered_versions[(agent_kind, selected_id)] = slam.version

    def _render_combined(
        self,
        h: int,
        w: int,
        render_tail: int,
        agents: Any,
        agent_kind: str,
    ) -> None:
        """Merge the default mission-agent SLAM views."""
        dependencies = self.dependencies
        combined_occ = np.full((h, w), -1, dtype=np.int8)
        combined_conf = np.zeros((h, w), dtype=np.float32)
        combined_points: List[Tuple[int, int]] = []

        snapshots = [
            agent.slam_map.snapshot(point_limit=render_tail)
            for agent in agents
        ]
        for slam in snapshots:
            occ = slam.occupancy
            conf = slam.confidence
            combined_points.extend(slam.point_cloud)

            eh = min(h, occ.shape[0], conf.shape[0])
            ew = min(w, occ.shape[1], conf.shape[1])
            if eh <= 0 or ew <= 0:
                continue

            target_conf = combined_conf[:eh, :ew]
            source_conf = conf[:eh, :ew]
            higher_conf = source_conf > target_conf
            # For each cell, keep the occupancy label with the highest reported
            # confidence among all drone snapshots.
            combined_occ[:eh, :ew][higher_conf] = occ[:eh, :ew][higher_conf]
            target_conf[higher_conf] = source_conf[higher_conf]

        if dependencies.presentation.show_terrain_heatmap:
            terrain = dependencies.terrain_knowledge.snapshot()
            dependencies.slam_renderer.render(
                None,
                None,
                combined_points,
                draw_points=True,
                roughness=terrain.roughness,
                roughness_conf=terrain.confidence,
            )
        else:
            dependencies.slam_renderer.render(
                combined_occ,
                combined_conf,
                combined_points,
                draw_points=False,
            )

        for agent_id, slam in enumerate(snapshots):
            self.rendered_versions[(agent_kind, agent_id)] = slam.version

    def _current_view_is_dirty(self) -> bool:
        """Return whether the active combined/selected view needs refresh."""
        dependencies = self.dependencies
        drones = dependencies.get_drones()
        rovers = dependencies.get_rovers()
        selected_rover_id = getattr(
            dependencies.presentation,
            "selected_rover_heatmap_id",
            None,
        )
        if (
            selected_rover_id is not None
            and 0 <= selected_rover_id < len(rovers)
        ):
            return self._map_is_dirty(
                "rover",
                selected_rover_id,
                rovers[selected_rover_id].slam_map,
            )
        selected_id = dependencies.presentation.selected_drone_heatmap_id
        if selected_id is not None and 0 <= selected_id < len(drones):
            return self._map_is_dirty(
                "drone",
                selected_id,
                drones[selected_id].slam_map,
            )
        agents = drones if drones else rovers
        agent_kind = "drone" if drones else "rover"
        return any(
            self._map_is_dirty(agent_kind, agent_id, agent.slam_map)
            for agent_id, agent in enumerate(agents)
        )

    def _map_is_dirty(
        self,
        agent_kind: str,
        agent_id: int,
        slam_map: Any,
    ) -> bool:
        """Compare an agent SLAM version with its rendered version."""
        rendered_version = self.rendered_versions.get(
            (str(agent_kind), int(agent_id)),
            -1,
        )
        return slam_map.has_changed_since(rendered_version)
