"""Drone agent state for the Cave Explorer simulation."""

import random as rand
from typing import Tuple, TYPE_CHECKING

from agents.exploration_policy import FrontierExplorationPolicy
from agents.mcts_exploration_policy import MctsExplorationPolicy
from mapping.slam_map import SlamMap
from agents.drone_movement import DroneMovementController
from agents.drone_runtime_state import DroneRuntimeState, DroneSnapshot
from mapping.drone_sensor import DroneSensorController
from mapping.localization import PerfectPoseLocalizer
from mapping.terrain_knowledge import TerrainKnowledge
from navigation.waypoint_graph import WaypointGraph
from navigation.frontier_clusters import (
    AssignmentRegistry,
    FrontierClusterRegistry,
    FrontierExtractor,
    FrontierGatewayManager,
)
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
        frontier_settings = self.settings.frontier
        self.frontier_registry = getattr(control, "frontier_registry", None)
        if self.frontier_registry is None:
            self.frontier_registry = FrontierClusterRegistry(
                match_distance=frontier_settings.cluster_match_distance,
                missing_refresh_limit=frontier_settings.missing_refresh_limit,
            )
        self.frontier_assignments = getattr(control, "frontier_assignments", None)
        if self.frontier_assignments is None:
            self.frontier_assignments = AssignmentRegistry()
        self.frontier_extractor = FrontierExtractor(
            frontier_settings.confidence_threshold,
            minimum_unknown_support=(
                frontier_settings.minimum_unknown_support
            ),
        )

        waypoint_settings = getattr(self.settings, "waypoints", None)
        waypoint_graph = getattr(control, "waypoint_graph", None)
        if (
            waypoint_graph is None
            and waypoint_settings is not None
            and waypoint_settings.enabled
        ):
            # Lightweight test/custom controls may not own mission services.
            # Production MissionControl injects one graph shared by all drones.
            waypoint_graph = WaypointGraph(
                merge_radius=waypoint_settings.merge_radius,
                connector_distance=waypoint_settings.connector_distance,
                connector_limit=waypoint_settings.connector_limit,
                spatial_hash_cell=waypoint_settings.spatial_hash_cell,
                route_cache_capacity=waypoint_settings.route_cache_capacity,
            )
        self.frontier_gateway_manager = getattr(
            control, "frontier_gateway_manager", None
        )
        if self.frontier_gateway_manager is None and waypoint_graph is not None:
            self.frontier_gateway_manager = FrontierGatewayManager(
                self.frontier_registry,
                waypoint_graph,
                minimum_separation=frontier_settings.gateway_min_separation,
            )
        self.exploration_coordinator = getattr(
            control, "exploration_coordinator", None
        )

        # SLAM state
        map_h = len(self.cave)
        map_w = len(self.cave[0]) if map_h else 0
        max_points = self.settings.slam.point_cloud_max_points
        self.slam_map = SlamMap(map_h, map_w, max_points=max_points)
        # Controllers keep movement/sensing logic out of this data container.
        self.movement_controller = DroneMovementController(
            self,
            DroneMovementDependencies(
                simulation_time=control.simulation_time,
                pause_checkpoint=control.pause_checkpoint,
                wait_simulation_delay=control.wait_simulation_delay,
                runtime_trace=getattr(control, "runtime_trace", None),
                waypoint_graph=waypoint_graph,
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
        """Create the configured exploration policy for this drone."""
        exploration = self.settings.exploration
        if exploration.policy == "frontier":
            return FrontierExplorationPolicy()

        mission_seed = self.settings.mission_config.seed
        policy_seed = int(mission_seed) + (self.id * 9_973)
        return MctsExplorationPolicy(exploration, seed=policy_seed)

    
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

    def share_frontier_clusters_with(self, other_drone: "Drone") -> tuple[int, ...]:
        """Transfer stable frontier knowledge across an authorized link."""
        if self.frontier_registry is not other_drone.frontier_registry:
            return ()
        transferred = self.frontier_registry.share(self.id, other_drone.id)
        if transferred:
            other_drone.runtime_state.replace_frontier_clusters(
                self.frontier_registry.visible_to(other_drone.id)
            )
        return transferred

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
