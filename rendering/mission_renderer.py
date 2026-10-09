"""Top-level mission scene composition."""

from asset_config.rendering import Colors
from contracts import MissionRendererDependencies
from ui.control_center.view_model import (
    build_drone_status_views,
    build_rover_status_views,
)


class MissionRenderer:
    """Render the complete mission scene in a stable layer order."""

    def __init__(self, dependencies: MissionRendererDependencies) -> None:
        """Store rendering dependencies and fixed mission-control buttons."""
        self.dependencies = dependencies

    def draw(self) -> None:
        """Render SLAM, agents, control center, and mission controls."""
        dependencies = self.dependencies
        control_center = dependencies.get_control_center()
        if control_center is None:
            raise RuntimeError("Mission runtime is not initialized")

        drones = tuple(dependencies.get_drones())
        rovers = tuple(dependencies.get_rovers())
        drone_snapshots = tuple(drone.snapshot() for drone in drones)
        docked_drone_ids = frozenset(dependencies.get_docked_drone_ids())
        window = dependencies.get_window()
        draw_static_background = getattr(
            dependencies.slam_view,
            "draw_static_background",
            None,
        )
        if draw_static_background is None or not draw_static_background():
            window.fill(Colors.BLACK.value)
        dependencies.slam_view.draw()

        sector_renderer = dependencies.sector_renderer
        if sector_renderer is not None:
            sector_renderer.draw(
                window,
                dependencies.get_sector_snapshot(),
                {drone.id: drone.color for drone in drones},
            )

        if dependencies.highway_renderer is not None:
            for rover in rovers:
                if getattr(rover, "show_highway", True):
                    dependencies.highway_renderer.draw(
                        window, dependencies.get_highway_snapshot(int(rover.id)),
                    )

        # Layer order: SLAM, sectors, highways, historical paths, translucent vision
        # cones, icons, then the control center.
        for drone, snapshot in zip(drones, drone_snapshots):
            drone.renderer.draw_path(snapshot)
        for rover in rovers:
            rover.renderer.draw_path()

        for drone, snapshot in zip(drones, drone_snapshots):
            if drone.id not in docked_drone_ids:
                drone.renderer.draw_vision_overlay(snapshot)

        drones_by_id = {int(drone.id): drone for drone in drones}
        snapshots_by_id = {
            int(drone.id): snapshot
            for drone, snapshot in zip(drones, drone_snapshots)
        }
        rovers_by_id = {int(rover.id): rover for rover in rovers}
        drawn_sharing_pairs: set[tuple[str, int, int]] = set()
        for drone, snapshot in zip(drones, drone_snapshots):
            controller = getattr(drone, "movement_controller", None)
            activity_snapshot = getattr(controller, "activity_snapshot", None)
            activity = (
                activity_snapshot()
                if callable(activity_snapshot)
                else None
            )
            peer_id = getattr(activity, "peer_id", None)
            rover_id = getattr(activity, "rover_id", None)
            if getattr(activity, "state", None) != "Sharing":
                continue
            if peer_id is not None:
                peer = drones_by_id.get(int(peer_id))
                peer_snapshot = snapshots_by_id.get(int(peer_id))
                pair_ids = sorted((int(drone.id), int(peer_id)))
                pair = ("drone", pair_ids[0], pair_ids[1])
                if (
                    pair in drawn_sharing_pairs
                    or peer is None
                    or peer_snapshot is None
                ):
                    continue
                drone.renderer.draw_sharing_cue(
                    snapshot,
                    peer_snapshot.position,
                    self._icon_ring_radius(peer),
                )
                drawn_sharing_pairs.add(pair)
                continue
            if rover_id is None:
                continue
            rover = rovers_by_id.get(int(rover_id))
            pair = ("rover", int(drone.id), int(rover_id))
            if pair in drawn_sharing_pairs or rover is None:
                continue
            rover_snapshot_method = getattr(rover, "snapshot", None)
            rover_position = (
                rover_snapshot_method().position
                if callable(rover_snapshot_method)
                else tuple(rover.pos)
            )
            drone.renderer.draw_sharing_cue(
                snapshot,
                rover_position,
                self._icon_ring_radius(rover),
            )
            drawn_sharing_pairs.add(pair)

        for i, (drone, snapshot) in enumerate(
            zip(drones, drone_snapshots)
        ):
            drone.renderer.draw_icon(snapshot)
            if i < len(rovers):
                rovers[i].renderer.draw_icon()

        debug_lines = dependencies.debug_info.build_debug_lines(
            drone_snapshots
        )
        system_lines = dependencies.debug_info.build_system_lines(
            drone_snapshots
        )
        drone_statuses = build_drone_status_views(
            drones,
            drone_snapshots,
        )
        rover_statuses = build_rover_status_views(rovers)
        control_center.draw_control_center(
            drone_statuses=drone_statuses,
            rover_statuses=rover_statuses,
            show_terrain_heatmap=(
                dependencies.presentation.show_terrain_heatmap
            ),
            selected_drone_heatmap_id=(
                dependencies.presentation.selected_drone_heatmap_id
            ),
            debug_lines=debug_lines,
            system_lines=system_lines,
            is_paused=dependencies.is_paused(),
            music_enabled=dependencies.is_music_enabled(),
            show_full_map=getattr(
                dependencies.presentation,
                "show_full_map",
                False,
            ),
            selected_rover_heatmap_id=getattr(
                dependencies.presentation,
                "selected_rover_heatmap_id",
                None,
            ),
            exploration_complete=dependencies.is_exploration_complete(),
        )

    @staticmethod
    def _icon_ring_radius(agent: object) -> int | None:
        """Return a cue radius just outside an agent sprite, when available."""
        icon = getattr(agent, "icon", None)
        get_size = getattr(icon, "get_size", None)
        if not callable(get_size):
            return None
        return max(get_size()) // 2 + 4
