"""Panel and status-row helpers for the control-center renderer."""

from typing import Optional

from asset_config.gameplay import Display
from asset_config.rendering import Colors, Fonts, RectHandle
from ui.control_center.view_model import (
    DRONE_ROSTER,
    ROVER_ROSTER,
    AgentRosterEntry,
    ControlCenterViewModel,
    DroneStatusView,
    RoverStatusView,
)


class ControlCenterPanelMixin:
    """Draw tab contents and live agent status rows."""

    def draw_drone_section(
        self,
        statuses: tuple[DroneStatusView, ...],
        selected_drone_heatmap_id: Optional[int],
    ) -> None:
        """Draw the drone roster, status rows, and per-drone toggles."""
        self.draw_section_header("drones")
        statuses_by_id = {status.id: status for status in statuses}
        for roster_entry in DRONE_ROSTER:
            self.draw_status(
                roster_entry,
                status_view=statuses_by_id.get(roster_entry.id),
                selected_drone_heatmap_id=selected_drone_heatmap_id,
                name_height=self.DRONE_NAME_Y,
                data_height=self.DRONE_DATA_Y,
            )

    def draw_rover_section(
        self,
        statuses: tuple[RoverStatusView, ...],
    ) -> None:
        """Draw rover roster rows, including undeployed placeholders."""
        self.draw_section_header("rovers")
        statuses_by_id = {status.id: status for status in statuses}
        for roster_entry in ROVER_ROSTER:
            self.draw_status(
                roster_entry,
                rover=True,
                status_view=statuses_by_id.get(roster_entry.id),
                name_height=self.DRONE_NAME_Y,
                data_height=self.DRONE_DATA_Y,
            )

    def draw_debug_panel(self, debug_lines: tuple[str, ...]) -> None:
        """Draw wrapped mission debug lines in the debug tab."""
        self.draw_section_header("debug")
        lines = debug_lines or ("No debug lines available",)
        font_obj = self._get_font(Fonts.BIG.value, 20)
        max_w = Display.LEGEND_WIDTH - 16
        ypos = self.DRONE_NAME_Y
        index = 0
        line_gap = 8
        for line in lines:
            if ":" in line:
                label, value = line.split(":", 1)
                ypos, index, done = self._draw_label_value_entry(
                    "debug",
                    index,
                    label.strip() + ":",
                    value.strip(),
                    font_obj,
                    max_w,
                    ypos,
                    line_gap,
                    line_gap,
                )
            else:
                ypos, index, done = self._draw_wrapped_text_lines(
                    "debug",
                    index,
                    line,
                    font_obj,
                    max_w,
                    ypos,
                    line_gap,
                )
            if done:
                return

    def draw_system_panel(self, view: ControlCenterViewModel) -> None:
        """Draw aggregate UI/system state in the system tab."""
        self.draw_section_header("system")
        total_drones = len(view.drone_statuses)
        active_vision = sum(
            1 for drone in view.drone_statuses if drone.show_vision
        )
        active_paths = sum(
            1 for drone in view.drone_statuses if drone.show_path
        )
        avg_battery = (
            0
            if total_drones == 0
            else int(
                sum(
                    drone.battery
                    for drone in view.drone_statuses
                )
                / total_drones
            )
        )
        pairs = [
            ("Active tab:", view.active_tab.upper()),
            ("Drones online:", str(total_drones)),
            ("Rovers online:", str(len(view.rover_statuses))),
            (
                "Vision overlays:",
                f"{active_vision}/{total_drones}",
            ),
            ("Path overlays:", f"{active_paths}/{total_drones}"),
            ("Avg drone battery:", f"{avg_battery}%"),
            ("Debug lines:", str(len(view.debug_lines))),
        ]
        font_obj = self._get_font(Fonts.BIG.value, 20)
        max_w = Display.LEGEND_WIDTH - 16
        ypos = self.DRONE_DATA_Y - 25
        index = 0
        for label, value in pairs:
            ypos, index, done = self._draw_label_value_entry(
                "system",
                index,
                label,
                value,
                font_obj,
                max_w,
                ypos,
                8,
                6,
            )
            if done:
                return

    def draw_status(
        self,
        roster_entry: AgentRosterEntry,
        rover: bool = False,
        status_view: Optional[
            DroneStatusView | RoverStatusView
        ] = None,
        selected_drone_heatmap_id: Optional[int] = None,
        name_height: Optional[int] = None,
        data_height: Optional[int] = None,
    ) -> None:
        """Draw one roster row, using N/A when that agent is not deployed."""
        deployed = status_view is not None
        number = roster_entry.id
        label = roster_entry.name
        color = (
            roster_entry.color
            if status_view is None
            else status_view.color
        )
        display_label = (
            label
            if status_view is None
            else status_view.name
        )
        if name_height is None:
            name_height = 760 if rover else 230
        if data_height is None:
            data_height = 790 if rover else 260
        max_battery = 2400 if rover else 100

        key = (
            f"rover_{label}"
            if rover
            else f"drone_{label}"
        )
        if deployed:
            key = ("live_agent_name", display_label, color)
            if key not in self._static_surfaces:
                self._static_surfaces[key] = self._get_font(
                    Fonts.BIG.value,
                    25,
                ).render(
                    display_label,
                    True,
                    color,
                ).convert_alpha()
        name_surf = self._static_surfaces[key]
        y_center = name_height + 60 * number
        name_rect = name_surf.get_rect()
        name_rect.midleft = (8, y_center)
        self.control_surf.blit(name_surf, name_rect)

        if isinstance(status_view, DroneStatusView):
            # Drone rows own three small action buttons: path, vision, terrain.
            self._draw_drone_toggles(
                status_view,
                y_center,
                selected_drone_heatmap_id,
            )

        if status_view is None:
            na_surf = self._static_fragments["N/A"]
            na_rect = na_surf.get_rect()
            na_rect.midleft = (8, data_height + 60 * number)
            self.control_surf.blit(na_surf, na_rect)
            return

        status_color = self._status_color(status_view.status)
        battery_color = self.percent_color(
            status_view.battery,
            max_battery,
        )
        cache_key = (
            f"status_{'rover_' if rover else ''}"
            f"{label}_{status_view.battery}_{status_view.status}"
        )
        data_surf = self._get_cached_status_surface(
            cache_key,
            status_view.battery,
            status_view.status,
            battery_color,
            status_color,
            25,
            Fonts.BIG.value,
        )
        self._blit_cached_surface(
            data_surf,
            self.origin_x,
            data_height + 60 * number,
            RectHandle.MIDLEFT.value,
        )

    def draw_section_header(self, key: str) -> None:
        """Draw a cached tab section title by key."""
        surf = self._static_surfaces.get(key)
        if surf is None:
            return
        rect = surf.get_rect()
        rect.centerx = Display.LEGEND_WIDTH // 2
        rect.centery = self.SECTION_HEADER_Y
        self.control_surf.blit(surf, rect)

    @staticmethod
    def _status_color(status: str) -> tuple[int, int, int]:
        """Choose a text color for an agent status label."""
        if status in ("Ready", "Done"):
            return Colors.GREEN.value
        if status in (
            "Updating",
            "Advancing",
            "Sharing",
            "Charging",
        ):
            return Colors.YELLOW.value
        if status in ("Deployed", "Homing"):
            return Colors.WHITE.value
        return Colors.RED.value

    @staticmethod
    def percent_color(
        value: int,
        max_value: int = 100,
    ) -> tuple[int, int, int]:
        """Return red/yellow/green for low/medium/high percentages."""
        if value < max_value * 20 / 100:
            return Colors.RED.value
        if value < max_value * 80 / 100:
            return Colors.YELLOW.value
        return Colors.GREEN.value
