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
        window = dependencies.get_window()
        draw_static_background = getattr(
            dependencies.slam_view,
            "draw_static_background",
            None,
        )
        if draw_static_background is None or not draw_static_background():
            window.fill(Colors.BLACK.value)
        dependencies.slam_view.draw()

        waypoint_renderer = dependencies.waypoint_renderer
        if waypoint_renderer is not None:
            waypoint_renderer.draw(window)

        # Layer order: waypoint highways, historical paths, translucent vision
        # cones, icons, then the control center.
        for drone, snapshot in zip(drones, drone_snapshots):
            drone.renderer.draw_path(snapshot)
        for rover in rovers:
            rover.renderer.draw_path()

        for drone, snapshot in zip(drones, drone_snapshots):
            drone.renderer.draw_vision_overlay(snapshot)

        for i, (drone, snapshot) in enumerate(
            zip(drones, drone_snapshots)
        ):
            drone.renderer.draw_icon(snapshot)
            if i < len(rovers):
                rovers[i].renderer.draw_icon()

        debug_lines = dependencies.debug_info.build_lines(drone_snapshots)
        drone_statuses = build_drone_status_views(
            drones,
            drone_snapshots,
        )
        rover_statuses = build_rover_status_views(rovers)
        control_center.draw_control_center(
            drone_statuses,
            rover_statuses,
            dependencies.presentation.show_terrain_heatmap,
            dependencies.presentation.selected_drone_heatmap_id,
            debug_lines,
            dependencies.is_paused(),
            dependencies.is_music_enabled(),
            getattr(dependencies.presentation, "show_full_map", False),
        )
