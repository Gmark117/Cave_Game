"""Drone agent state for the Cave Explorer simulation."""

import random as rand
from typing import Tuple, TYPE_CHECKING

from agents.exploration_policy import RandomDirectionPolicy
from mapping.slam_map import SlamMap
from agents.drone_movement import DroneMovementController
from agents.drone_runtime_state import DroneRuntimeState, DroneSnapshot
from mapping.drone_sensor import DroneSensorController
from mapping.localization import PerfectPoseLocalizer
from mapping.terrain_knowledge import TerrainKnowledge
from contracts import (
    DroneMovementDependencies,
    DroneSensorDependencies,
)
from rendering.agent_renderer import DroneRenderer

if TYPE_CHECKING:
    import pygame


class Drone:
    """Autonomous agent state shared by movement, mapping, and rendering."""

    def __init__(self, game, control, id: int, start_pos: Tuple[int, int],
                 color: Tuple[int, int, int], icon: 'pygame.Surface', cave: list) -> None:
        """Initialize runtime state for a drone.

        Args:
            game: Owner `Game` instance (used for surfaces and settings).
            control: `MissionControl` instance used for pathfinding calls.
            id: Unique integer id for the drone.
            start_pos: (x,y) starting coordinates.
            color: RGB color tuple used for drawing the drone overlays.
            icon: Pygame Surface for drawing the drone icon.
            cave: Binary map matrix used for collision checks.
        """
        self.game = game
        self.settings = game.sim_settings
        self.cave = cave

        # Identity and movement. Radius is a map-size proximity value used by
        # movement, sharing, and as the base for the LiDAR sensor range.
        self.id = id
        self.map_size = self.settings.mission_config.map_dim
        self.radius = self.calculate_radius()
        self.step = 10
        initial_direction = rand.randint(0, 359)

        # Appearance / drawing
        self.color = color
        # Vision cone alpha set to 128 for semi-transparency
        self.alpha = 128
        self.icon = icon

        # State and lifecycle
        self.delay = control.delay

        # Presentation and traversal configuration
        self.speed_factor = 4

        # Local terrain knowledge (distributed mapping). Agent decisions should
        # use this local store; MissionControl keeps a separate aggregate only
        # for progress and combined rendering.
        self.terrain_knowledge = TerrainKnowledge(self.cave)

        # Exploration bookkeeping
        self.start_pos = start_pos
        frontier_rebuild_cooldown = self.settings.frontier.rebuild_cooldown
        self.runtime_state = DroneRuntimeState(
            start_position=start_pos,
            cave=cave,
            direction=initial_direction,
            frontier_rebuild_cooldown=frontier_rebuild_cooldown,
        )
        self.localizer = PerfectPoseLocalizer()
        self.exploration_policy = self._build_exploration_policy()

        # SLAM state
        map_h = len(self.cave)
        map_w = len(self.cave[0]) if map_h else 0
        max_points = self.settings.slam.point_cloud_max_points
        self.slam_map = SlamMap(map_h, map_w, max_points=max_points)
        # Controllers keep movement/sensing logic out of this data container.
        self.movement_controller = DroneMovementController(
            self,
            DroneMovementDependencies(
                compute_path=control.compute_path,
                simulation_time=control.simulation_time,
                pause_checkpoint=control.pause_checkpoint,
                wait_simulation_delay=control.wait_simulation_delay,
                compute_path_segment=getattr(
                    control,
                    "compute_path_segment",
                    None,
                ),
                runtime_trace=getattr(control, "runtime_trace", None),
                get_drone_positions=lambda: tuple(
                    (peer.id, peer.snapshot().position)
                    for peer in getattr(control, "drones", ())
                ),
            ),
        )
        self.sensor_controller = DroneSensorController(
            self,
            DroneSensorDependencies(
                terrain_roughness=control.terrain_roughness,
                simulation_time=control.simulation_time,
                record_terrain_scan=control.terrain_fusion.record_scan,
                runtime_trace=getattr(control, "runtime_trace", None),
            ),
        )
        self.renderer = DroneRenderer(self)

    def _build_exploration_policy(self):
        """Create a reproducible random policy private to this drone."""
        seed = int(self.settings.mission_config.seed) + self.id * 9_973
        return RandomDirectionPolicy(seed=seed)

    
    def calculate_radius(self) -> int:
        """Return the map-size-based proximity radius for this drone."""
        match self.map_size:
            case 'SMALL':
                return 40
            case 'MEDIUM':
                return 20
            case 'LARGE':
                return 10
            case _:
                return 20
        
    
    def move(self) -> None:
        """Advance this drone's exploration state."""
        self.movement_controller.move()

    def mission_completed(self) -> bool:
        """Return whether this drone has completed its mission."""
        return self.movement_controller.mission_completed()

    def update_sensors(self) -> None:
        """Update local SLAM and terrain knowledge without rendering."""
        self.sensor_controller.update()

    def snapshot(self) -> DroneSnapshot:
        """Return detached runtime state for cross-thread consumers."""
        return self.runtime_state.snapshot()

    def merge_frontiers(self, other_border) -> None:
        """Merge simple border coordinates received from another drone."""
        if other_border is None:
            return
        self.runtime_state.merge_frontiers(other_border)

    def toggle_path(self) -> None:
        """Toggle rendering visibility for the drone path overlay."""
        self.runtime_state.toggle_path()


    def toggle_vision(self) -> None:
        """Toggle rendering visibility for the drone vision overlay."""
        self.runtime_state.toggle_vision()

    def set_overlay_visibility(
        self,
        *,
        show_path: bool,
        show_vision: bool,
    ) -> None:
        """Atomically set path and vision presentation visibility."""
        self.runtime_state.set_overlay_visibility(
            show_path=show_path,
            show_vision=show_vision,
        )
