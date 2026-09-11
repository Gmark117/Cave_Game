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

        sector_renderer = dependencies.sector_renderer
        if sector_renderer is not None:
            sector_renderer.draw(
                window,
                dependencies.get_sector_snapshot(),
                {drone.id: drone.color for drone in drones},
            )

        # Layer order: SLAM, sectors, historical paths, translucent vision
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
