"""Top-level mission scene composition."""

import math

import pygame

from asset_config.rendering import Colors
from mission.service_dependencies import MissionRendererDependencies
from ui.control_center.view_model import (
    build_drone_status_views,
    build_rover_status_views,
)


class MissionRenderer:
    """Render the complete mission scene in a stable layer order."""

    def __init__(self, dependencies: MissionRendererDependencies) -> None:
        self.dependencies = dependencies
        self.stop_button_rect = pygame.Rect(10, 10, 40, 40)
        self.restart_button_rect = pygame.Rect(58, 10, 40, 40)
        self.pause_button_rect = pygame.Rect(106, 10, 40, 40)

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
        window.fill(Colors.BLACK.value)
        dependencies.slam_view.draw()

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
        )

        self.draw_stop_button()
        self.draw_restart_button()
        self.draw_pause_button()

    def draw_stop_button(self) -> None:
        """Draw the mission stop control."""
        window = self.dependencies.get_window()
        pygame.draw.rect(window, Colors.RED.value, self.stop_button_rect)
        pygame.draw.rect(
            window,
            Colors.GREY.value,
            self.stop_button_rect,
            2,
        )

        outer_symbol = pygame.Rect(0, 0, 22, 22)
        outer_symbol.center = self.stop_button_rect.center
        pygame.draw.rect(
            window,
            Colors.WHITE.value,
            outer_symbol,
            2,
        )

        inner_symbol = pygame.Rect(0, 0, 12, 12)
        inner_symbol.center = self.stop_button_rect.center
        pygame.draw.rect(window, Colors.WHITE.value, inner_symbol)

    def draw_restart_button(self) -> None:
        """Draw the current-mission restart control."""
        window = self.dependencies.get_window()
        pygame.draw.rect(
            window,
            Colors.BLUE.value,
            self.restart_button_rect,
        )
        pygame.draw.rect(
            window,
            Colors.GREY.value,
            self.restart_button_rect,
            2,
        )

        center_x, center_y = self.restart_button_rect.center
        radius = 10.5
        start_angle = math.radians(55)
        end_angle = math.tau
        point_count = 48
        arc_points = []
        for index in range(point_count):
            ratio = index / (point_count - 1)
            angle = start_angle + ((end_angle - start_angle) * ratio)
            arc_points.append(
                (
                    round(center_x + radius * math.cos(angle)),
                    round(center_y + radius * math.sin(angle)),
                )
            )

        pygame.draw.lines(
            window,
            Colors.WHITE.value,
            False,
            arc_points,
            3,
        )
        pygame.draw.aalines(
            window,
            Colors.WHITE.value,
            False,
            arc_points,
        )

        end_x, end_y = arc_points[-1]
        tangent_angle = end_angle + (math.pi / 2)
        tangent_x = math.cos(tangent_angle)
        tangent_y = math.sin(tangent_angle)
        normal_x = -tangent_y
        normal_y = tangent_x
        tip = (
            round(end_x + tangent_x * 5),
            round(end_y + tangent_y * 5),
        )
        base_center = (
            end_x - tangent_x * 3,
            end_y - tangent_y * 3,
        )
        pygame.draw.polygon(
            window,
            Colors.WHITE.value,
            [
                tip,
                (
                    round(base_center[0] + normal_x * 4),
                    round(base_center[1] + normal_y * 4),
                ),
                (
                    round(base_center[0] - normal_x * 4),
                    round(base_center[1] - normal_y * 4),
                ),
            ],
        )

    def draw_pause_button(self) -> None:
        """Draw a pause icon while running and a play icon while paused."""
        window = self.dependencies.get_window()
        pygame.draw.rect(
            window,
            (
                Colors.GREEN.value
                if self.dependencies.is_paused()
                else Colors.OCHRE.value
            ),
            self.pause_button_rect,
        )
        pygame.draw.rect(
            window,
            Colors.GREY.value,
            self.pause_button_rect,
            2,
        )

        center_x, center_y = self.pause_button_rect.center
        if self.dependencies.is_paused():
            pygame.draw.polygon(
                window,
                Colors.WHITE.value,
                [
                    (center_x - 6, center_y - 11),
                    (center_x - 6, center_y + 11),
                    (center_x + 10, center_y),
                ],
            )
            return

        left_bar = pygame.Rect(0, 0, 6, 22)
        left_bar.center = (center_x - 5, center_y)
        right_bar = pygame.Rect(0, 0, 6, 22)
        right_bar.center = (center_x + 5, center_y)
        pygame.draw.rect(window, Colors.WHITE.value, left_bar)
        pygame.draw.rect(window, Colors.WHITE.value, right_bar)
