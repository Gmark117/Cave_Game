"""Mission orchestration and runtime resource setup.

Constructing `MissionControl` prepares mission state only. Calling `run()`
initializes the mission window, agents, pathfinding resources, and worker
threads before entering the main loop.
"""

import random as rand
from dataclasses import dataclass, replace
from queue import Empty, Queue
import math
import threading
import time
from pathlib import Path
from typing import List, Tuple, Any, Optional

import numpy as np
import cv2
import pygame

from asset_config.helpers import wall_hit
from agents.factory import AgentFactory
from ui.control_center.facade import ControlCenter
from mapping.rover_targets import RoverFrontierTarget, RoverTargetService
from mapping.exploration_sectors import (
    ExplorationSectorCoordinator,
    SectorCheckInResult,
)
from mapping.terrain_fusion import TerrainFusionService
from mapping.terrain_knowledge import TerrainKnowledge
from mapping.terrain_sharing import TerrainSharingService
from mapping.wall_mapping import WallMappingSnapshot, wall_mapping_snapshot
from mapping.slam_map import FREE
from mission.debug_info import MissionDebugInfo
from mission.frame_timing import FrameProfiler
from mission.energy import EnergyState
from mission.exploration_coordination import (
    CoordinationReport,
    CoordinationResult,
    DirectiveKind,
    FrontierTaskCoordinator,
)
from mission.objectives import build_mission_objective
from mission.pause_control import PauseCoordinator, SimulationClock
from mission.rendezvous import RendezvousProtocol
from mission.runtime_trace import RuntimeTraceLogger
from contracts import (
    MissionDebugDependencies,
    MissionRendererDependencies,
    RoverTargetDependencies,
    SlamViewDependencies,
    TerrainFusionDependencies,
    TerrainSharingDependencies,
)
from navigation.pathfinding import PathfindingService
from navigation.astar_pathfinder import PathResult
from mission.presentation_adapter import PresentationAdapter
from rendering.slam_renderer import SlamRenderer
from rendering.mission_renderer import MissionRenderer
from rendering.sector_renderer import SectorRenderer
from rendering.slam_view import SlamViewService
from mission.lifecycle import MissionControlLifecycleMixin


@dataclass
class _PendingExplorationCheckIn:
    """One drone report queued for execution on the primary rover thread."""

    drone_id: int
    report: CoordinationReport | None
    ready: threading.Event
    enqueued_at: float
    result: CoordinationResult | None = None


class MissionControl(MissionControlLifecycleMixin):
    """Orchestrates the simulation mission.

    Construction is side-effect-light and does not start threads, processes,
    shared memory, or the mission loop. Runtime resources are created by
    `run()` through `_initialize_runtime()`.
    """
    def __init__(self, game: Any) -> None:
        """Prepare mission state without starting the mission.

        Args:
            game: The `Game` instance owning this mission (typed as `Any`
                  to avoid circular imports).
        """
        # Seed every mission from the menu settings so generated caves and
        # agent color/order choices remain reproducible for a given seed.
        rand.seed(game.sim_settings.mission_config.seed)

        self.game         = game
        self.settings     = game.sim_settings 
        self.objective    = (
            getattr(game, "mission_objective", None)
            or build_mission_objective(
                self.settings.mission_config.objective
            )
        )
        self.cartographer = game.cartographer
        self.map_matrix   = self.cartographer.bin_map # Get the binary map representation
        self.map_h, self.map_w = np.asarray(self.map_matrix).shape
        # Map generation may be bypassed or mocked in tests, so normalize the
        # optional roughness layer to the cave matrix shape before sensors use it.
        terrain_roughness_src = np.array(
            getattr(self.cartographer, 'terrain_roughness', np.zeros_like(self.map_matrix)),
            dtype=np.float32
        )
        if terrain_roughness_src.shape != np.asarray(self.map_matrix).shape:
            terrain_roughness_src = np.zeros(np.asarray(self.map_matrix).shape, dtype=np.float32)
        self.terrain_roughness = terrain_roughness_src
        # Mission aggregate for telemetry and combined UI rendering only.
        # Active agent decisions must use their own local knowledge.
        self.terrain_knowledge = TerrainKnowledge(self.map_matrix)
        self.rover_assignment_lock = threading.Lock()
        self.rover_assignments = {}
        self.completed_rover_targets = set()
        # Rovers advance from their own received maps on dedicated workers.
        self.rover_motion_enabled = True
        self.rendezvous_protocol: RendezvousProtocol | None = None
        self._rover_staging_cache_key = None
        self._rover_staging_cache = None

        # Runtime resources are initialized explicitly by run().
        # Pathfinding owns external resources (shared memory and a process pool)
        # but does not allocate them until ``run`` calls ``start``.
        self.pathfinding = PathfindingService(
            self.map_matrix,
            self.settings.mission_config.num_drones,
        )
        self.mission_event = threading.Event()
        self.exploration_completion_event = threading.Event()
        self.simulation_clock = SimulationClock()
        self.pause_coordinator = PauseCoordinator(self.mission_event)
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.is_paused = False
        self.clock: Optional[pygame.time.Clock] = None
        self.drones = []
        self.rovers = []
        self.num_drones = self.settings.mission_config.num_drones
        self.num_rovers = 0
        self.control_center: Optional[ControlCenter] = None
        self._runtime_initialized = False
        self._running = False
        self._has_run = False
        self._exploration_check_in_lock = threading.Lock()
        self._exploration_check_in_queue: Queue[
            _PendingExplorationCheckIn
        ] = Queue()
        self._exploration_check_ins: dict[
            int, _PendingExplorationCheckIn
        ] = {}
        self.restart_requested = False
        self.exit_requested = False
        self.frame_profiler = FrameProfiler()
        self.runtime_trace = RuntimeTraceLogger(
            Path(__file__).resolve().parents[1],
            self.settings.trace,
        )
        self.runtime_trace.record(
            "mission_constructed",
            seed=self.settings.mission_config.seed,
            map_dim=self.settings.mission_config.map_dim,
            drones=self.settings.mission_config.num_drones,
            map_width=self.map_w,
            map_height=self.map_h,
            exploration_policy=self.settings.exploration.policy,
            exploration_completion=(
                "physical_team_quiescence_after_component_exhaustion"
            ),
            exploration_progress="discovered_floor_coverage_display_only",
            terrain_role="rover_checkpoint_and_navigation",
            frontier_policy=(
                "coordinated_scan_then_frontier_component_dfs"
            ),
            component_cost_cell_size=self.settings.frontier.global_cell_size,
            component_check_in="queued_moving_rover_rendezvous",
            component_check_in_path="astar_with_breadcrumb_fallback",
            component_assignment_delivery="rover_signalled_no_polling",
            component_task_routing=(
                "rover_known_free_connectivity_then_drone_astar"
            ),
            component_probe_routing=(
                "bootstrap_only_rover_known_free_then_drone_astar"
            ),
            component_follower_policy=(
                "spare_drone_follows_leader_then_reserves_first_split_branch"
            ),
            component_zero_gain_memory="individual_anchor_or_subarc",
            component_workload_estimate=(
                "frontier_cells_plus_unknown_pixels_per_cell_size_squared"
            ),
            component_zero_work_policy=(
                "bootstrap_probe_then_component_team_quiescence"
            ),
            component_claim_lease="token_fenced_explicit_release",
            component_energy_policy="unlimited_via_energy_contract",
            component_wide_classifier=(
                "multiple_footprints_and_wall_contact_count_ratio_run"
            ),
            frontier_unknown_basin_rescue="interior_basins_only",
            rover_periodic_sharing=True,
            rover_drone_pair_sharing=True,
            rover_count=1,
            rover_rendezvous_protocol=(
                "contact_carried_ack_confirmed_target_then_proposal_fallback"
            ),
            frontier_stride=self.settings.frontier.stride,
            frontier_minimum_cluster_cells=(
                self.settings.frontier.minimum_cluster_cells
            ),
            frontier_minimum_unknown_support_cells=(
                self.settings.frontier.minimum_unknown_support_cells
            ),
            frontier_distance_band=self.settings.frontier.distance_band,
            frontier_wall_continuation_weight=(
                self.settings.frontier.wall_continuation_weight
            ),
            frontier_cluster_size_weight=(
                self.settings.frontier.cluster_size_weight
            ),
            frontier_cluster_proximity_weight=(
                self.settings.frontier.cluster_proximity_weight
            ),
            frontier_global_cell_size=(
                self.settings.frontier.global_cell_size
            ),
            frontier_global_refresh_interval=(
                self.settings.frontier.global_refresh_interval
            ),
            frontier_global_ownership_weight=(
                self.settings.frontier.global_ownership_weight
            ),
            frontier_maximum_path_circuity=(
                self.settings.frontier.maximum_path_circuity
            ),
            stagnation_distance=(
                self.settings.exploration.stagnation_distance
            ),
            coverage_memory_cell_size=(
                self.settings.exploration.coverage_memory_cell_size
            ),
            coverage_memory_decay_seconds=(
                self.settings.exploration.coverage_memory_decay_seconds
            ),
            coverage_visit_weight=(
                self.settings.exploration.coverage_visit_weight
            ),
            coverage_edge_weight=(
                self.settings.exploration.coverage_edge_weight
            ),
            stagnation_min_sensor_cells_per_px=(
                self.settings.exploration.stagnation_min_sensor_cells_per_px
            ),
            wall_direction_bias=(
                self.settings.exploration.wall_direction_bias
            ),
            unexplored_direction_bias=(
                self.settings.exploration.unexplored_direction_bias
            ),
            separation_direction_bias=(
                self.settings.exploration.separation_direction_bias
            ),
        )
        
        self.delay = 1/15 # Set a delay for frame updates


        self.completed = False # Track whether the mission is completed
        self.exploration_complete = False

        # Initialize presentation adapter for UI state and map rendering
        self.presentation = PresentationAdapter(self.map_w, self.map_h)
        self.slam_renderer = SlamRenderer(
            self.map_w,
            self.map_h,
            frontier_confidence_threshold=(
                self.settings.frontier.confidence_threshold
            ),
        )
        self.sector_renderer = SectorRenderer(self.map_w, self.map_h)
        self.last_explored_update = 0.0
        self.wall_mapping_progress = WallMappingSnapshot(0, 0, 0.0, False)
        self.floor_exploration_ratio = 0.0
        self.explored_update_interval = 0.5
        # Dependency bundles keep services decoupled from the large
        # MissionControl object while still giving them the callbacks they need.
        self.terrain_fusion_dependencies = TerrainFusionDependencies(
            terrain_knowledge=self.terrain_knowledge,
            presentation=self.presentation,
        )
        self.terrain_fusion = TerrainFusionService(
            self.terrain_fusion_dependencies
        )
        self.terrain_sharing = TerrainSharingService(
            TerrainSharingDependencies(
                sharing=self.settings.sharing,
                cave_map=np.asarray(self.map_matrix),
                map_width=self.map_w,
                map_height=self.map_h,
                terrain_knowledge=self.terrain_knowledge,
                get_drones=lambda: self.drones,
                get_rovers=lambda: self.rovers,
                presentation=self.presentation,
                simulation_time=self.simulation_time,
                runtime_trace=self.runtime_trace,
                periodic_rover_sharing_enabled=True,
                on_drone_contact=self._rendezvous_drone_contact,
                on_drone_rover_contact=self._rendezvous_drone_rover_contact,
            )
        )
        self.rover_targets = RoverTargetService(
            RoverTargetDependencies(
                cave_map=np.asarray(self.map_matrix),
                terrain_knowledge=self.terrain_knowledge,
                assignment_lock=self.rover_assignment_lock,
                assignments=self.rover_assignments,
                completed_targets=self.completed_rover_targets,
                norm_width=self.game.width,
                norm_height=self.game.height,
                get_frontier_candidates=self._rover_frontier_candidates,
                should_hold_position=self._rover_should_hold_position,
            )
        )
        self.slam_view = SlamViewService(
            SlamViewDependencies(
                rendering=self.settings.rendering,
                terrain_knowledge=self.terrain_knowledge,
                presentation=self.presentation,
                slam_renderer=self.slam_renderer,
                get_drones=lambda: self.drones,
                get_window=lambda: self.game.window,
                get_rovers=lambda: self.rovers,
            )
        )
        self.debug_info = MissionDebugInfo(
            MissionDebugDependencies(
                get_drones=lambda: self.drones,
                presentation=self.presentation,
                dirty_map_count=self.slam_view.dirty_map_count,
                simulation_time=self.simulation_time,
                frame_profiler=self.frame_profiler,
                runtime_trace=self.runtime_trace,
                get_rovers=lambda: self.rovers,
            )
        )
        self.renderer = MissionRenderer(
            MissionRendererDependencies(
                get_window=lambda: self.game.window,
                slam_view=self.slam_view,
                debug_info=self.debug_info,
                get_control_center=lambda: self.control_center,
                get_drones=lambda: self.drones,
                get_rovers=lambda: self.rovers,
                presentation=self.presentation,
                is_paused=lambda: self.is_paused,
                is_music_enabled=self.music_enabled,
                sector_renderer=self.sector_renderer,
                get_sector_snapshot=lambda: (
                    self.exploration_coordinator.snapshot(blocking=False)
                    if self.exploration_coordinator is not None
                    else self.exploration_sectors.snapshot()
                ),
                is_exploration_complete=lambda: self.exploration_complete,
            )
        )
        
        # Set the starting position for drones
        self.start_point = None
        self.set_start_point()
        self.rendezvous_protocol = self._new_rendezvous_protocol()
        self.exploration_sectors = ExplorationSectorCoordinator(
            (self.map_h, self.map_w),
            self.num_drones,
            self.start_point,
            cell_size=self.settings.frontier.global_cell_size,
            confidence_threshold=(
                self.settings.frontier.confidence_threshold
            ),
            minimum_frontier_component_cells=(
                self.settings.frontier.minimum_cluster_cells
            ),
            minimum_unknown_support_cells=(
                self.settings.frontier.minimum_unknown_support_cells
            ),
        )
        # Built after drones so the coordinator uses the actual sensor
        # footprint.  The legacy sector coordinator remains available to
        # existing focused tests while production drones use this replacement.
        self.exploration_coordinator: FrontierTaskCoordinator | None = None
        self._traced_discovery_rounds: set[int] = set()

    def _initialize_runtime(self) -> None:
        """Create window, agents, pathfinding resources, and first frame."""
        if self._runtime_initialized:
            return

        self.completed = False
        self.exploration_complete = False
        self.mission_event.clear()
        self.exploration_completion_event.clear()
        self.pause_event.set()
        self.is_paused = False
        self.game.display = self.game.to_maximised()
        self.control_center = ControlCenter(self.game)
        self.rendezvous_protocol = self._new_rendezvous_protocol()

        AgentFactory.build_drones(self)
        AgentFactory.build_rovers(self)

        if not self.drones:
            raise RuntimeError("component exploration requires at least one drone")
        vision_sensor = self.drones[0].sensor_controller.vision_sensor
        self.exploration_coordinator = FrontierTaskCoordinator(
            (self.map_h, self.map_w),
            self.num_drones,
            self.start_point,
            sensor_range=float(vision_sensor.max_range),
            sensor_fov_deg=float(vision_sensor.fov_deg),
            confidence_threshold=(
                self.settings.frontier.confidence_threshold
            ),
            minimum_component_cells=(
                self.settings.frontier.minimum_cluster_cells
            ),
            minimum_unknown_support_cells=(
                self.settings.frontier.minimum_unknown_support_cells
            ),
            frontier_stride=self.settings.frontier.stride,
            global_cell_size=self.settings.frontier.global_cell_size,
        )

        # Reset presentation after agents exist so their path/vision toggles
        # start from a known default each time a mission is run.
        self.presentation.reset(self.drones)

        self.clock = pygame.time.Clock()
        self.pathfinding.start()
        self._runtime_initialized = True

        self.update_sensors()
        self.renderer.draw()
        pygame.display.update()

    def set_start_point(self) -> None:
        """Pick a viable start point from the map generator worm starts.

        Keeps sampling the list of candidate worm starts until a non-wall
        coordinate is found.
        """
        # Continuously search for a valid starting point until one is found
        while self.start_point is None or wall_hit(self.map_matrix, self.start_point):
            # Randomly select one of the initial points of the worms
            # Choose based on available worm starts (don't assume 4)
            i = rand.randrange(len(self.cartographer.worm_x))
            self.start_point = (self.cartographer.worm_x[i], self.cartographer.worm_y[i])
    

# =============================================================================
# Drone threads and pathfinding interface
# =============================================================================

    def drone_thread(self, drone_id: int) -> None:
        """
        Thread function that controls the movement of a single drone during the mission.
        This method runs in a separate thread for each drone and continuously moves the drone
        until either the mission is terminated (via mission_event) or the drone completes its
        assigned mission.
        Notes:
            - The method respects the global mission_event flag, which can stop all drones.
            - Movement speed is controlled by self.delay using an interruptible wait.
            - The wait mechanism allows for immediate response when mission_event is set.
        """
        label = ("drone", drone_id)
        if not self.pause_coordinator.register_current_worker(label):
            return
        try:
            while (
                not self.mission_event.is_set()
                and not self.drones[drone_id].mission_completed()
            ):
                # Worker threads only pause at cooperative checkpoints, which
                # keeps shared drone state from being stopped mid-update.
                if not self.pause_checkpoint():
                    break
                self.drones[drone_id].move()

                if not self.pause_checkpoint():
                    break
                self.terrain_sharing.share_with_nearby_drones(drone_id)

                if not self.wait_simulation_delay(self.delay):
                    break
        finally:
            self.pause_coordinator.unregister_current_worker()


    def compute_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute an A* escape or homing path for a drone."""
        return self.pathfinding.compute_path(start, goal)


    def compute_path_segment(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> PathResult:
        """Compute a complete drone route or one capped progress segment."""
        return self.pathfinding.compute_path_segment(start, goal)

    def _rover_frontier_candidates(self) -> tuple[RoverFrontierTarget, ...]:
        """Expose rover-local registry tasks as stable staging candidates."""
        coordinator = self.exploration_coordinator
        if coordinator is None:
            return ()
        snapshot = coordinator.published_snapshot()
        live_components = {
            component.component_id
            for component in snapshot.components
            if component.state.value == "active"
        }
        components_by_id = {
            component.component_id: component
            for component in snapshot.components
        }
        claimed_task_ids = {claim.task_id for claim in snapshot.claims}
        staging = self._rover_staging_context()
        candidates = []
        for task in snapshot.tasks:
            if not (
                task.component_id in live_components
                and task.state.value in {
                    "ready",
                    "claimed",
                    "active",
                    "suspended",
                }
            ):
                continue
            selected = self._select_rover_staging(
                task.preferred_entry,
                staging,
            )
            if selected is None:
                continue
            position, route_cost, roughness, clearance = selected
            candidates.append(RoverFrontierTarget(
                position=position,
                component_id=task.component_id,
                task_id=task.task_id,
                claimed=task.task_id in claimed_task_ids,
                depth=task.depth,
                estimated_effort=task.estimated_effort,
                parent_component_ids=components_by_id[
                    task.component_id
                ].parent_ids,
                service_cost=(
                    route_cost
                    + math.dist(position, task.preferred_entry)
                ),
                rover_route_cost=route_cost,
                terrain_roughness=roughness,
                wall_clearance=clearance,
            ))
        return tuple(candidates)

    def _rover_staging_context(self):
        """Build clearance and terrain arrays from the primary rover's map."""
        if not self.rovers or self.rovers[0] is None:
            return None
        rover = self.rovers[0]
        slam = rover.slam_map.snapshot(point_limit=0)
        terrain_version = int(rover.terrain_knowledge.version)
        icon = getattr(rover, "icon", None)
        body_radius = max(
            1.0,
            float(max(icon.get_size())) / 2.0 if icon is not None else 1.0,
        )
        cache_key = (int(slam.version), terrain_version, body_radius)
        if self._rover_staging_cache_key == cache_key:
            labels, clearance, asperity, terrain_confidence = (
                self._rover_staging_cache
            )
        else:
            known_free = (
                (np.asarray(slam.occupancy) == FREE)
                & (
                    np.asarray(slam.confidence)
                    >= self.settings.frontier.confidence_threshold
                )
            )
            padded = np.pad(known_free.astype(np.uint8), 1)
            clearance = cv2.distanceTransform(
                padded,
                cv2.DIST_L2,
                5,
            )[1:-1, 1:-1]
            safe = known_free & (clearance >= body_radius)
            _count, labels = cv2.connectedComponents(
                safe.astype(np.uint8),
                connectivity=8,
            )
            terrain = rover.terrain_knowledge.snapshot()
            terrain_confidence = terrain.confidence
            roughness = np.where(
                terrain_confidence > 0.0,
                terrain.roughness,
                1.0,
            ).astype(np.float32)
            diameter = max(1, int(math.ceil(body_radius * 2.0)) + 1)
            asperity = cv2.dilate(
                roughness,
                np.ones((diameter, diameter), dtype=np.uint8),
            )
            self._rover_staging_cache_key = cache_key
            self._rover_staging_cache = (
                labels,
                clearance,
                asperity,
                terrain_confidence,
            )
        rover_x, rover_y = int(rover.pos[0]), int(rover.pos[1])
        if not (
            0 <= rover_y < labels.shape[0]
            and 0 <= rover_x < labels.shape[1]
        ):
            return None
        rover_label = int(labels[rover_y, rover_x])
        if rover_label <= 0:
            return None
        safe = labels == rover_label
        return (
            safe,
            clearance,
            asperity,
            terrain_confidence,
            tuple(rover.pos),
            body_radius,
        )

    def _select_rover_staging(self, entry, context):
        """Choose a low-asperity, wall-clear service point near a task."""
        if context is None:
            return None
        safe, clearance, roughness, terrain_confidence, rover_pos, radius = context
        entry_x, entry_y = int(entry[0]), int(entry[1])
        service_radius = max(
            radius * 4.0,
            float(self.settings.frontier.global_cell_size) * 2.0,
        )
        left = max(0, int(math.floor(entry_x - service_radius)))
        right = min(safe.shape[1], int(math.ceil(entry_x + service_radius + 1)))
        top = max(0, int(math.floor(entry_y - service_radius)))
        bottom = min(safe.shape[0], int(math.ceil(entry_y + service_radius + 1)))
        local_safe = safe[top:bottom, left:right]
        ys, xs = np.nonzero(local_safe)
        if len(xs) == 0:
            return None
        ranked = []
        for local_x, local_y in zip(xs, ys):
            x, y = int(local_x) + left, int(local_y) + top
            point = int(x), int(y)
            service_distance = math.dist(point, entry)
            if service_distance > service_radius:
                continue
            if float(terrain_confidence[y, x]) <= 0.0:
                continue
            known_roughness = float(roughness[y, x])
            rover_distance = math.dist(rover_pos, point)
            ranked.append((
                service_distance + rover_distance,
                known_roughness,
                -float(clearance[y, x]),
                rover_distance,
                int(y),
                int(x),
                point,
            ))
        if not ranked:
            return None
        selected = min(ranked)
        point = selected[-1]
        return point, selected[3], selected[1], -selected[2]

    def _rover_should_hold_position(self, rover_id: int) -> bool:
        """Hold only for reports already delivered by physical contact."""
        if int(rover_id) != 0:
            return False
        with self._exploration_check_in_lock:
            return bool(self._exploration_check_ins)

    def get_check_in_position(
        self,
        drone_id: int | None = None,
    ) -> Tuple[int, int]:
        """Return physical rover position or one drone's selected rendezvous."""
        if drone_id is not None and self.rendezvous_protocol is not None:
            return self.rendezvous_protocol.drone_endpoint(int(drone_id))
        if self.rovers and self.rovers[0] is not None:
            return tuple(self.rovers[0].pos)
        return tuple(self.start_point)

    def rendezvous_endpoint_missed(
        self,
        drone_id: int,
        position: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Advance one drone only after it observes its confirmed point empty."""
        protocol = self.rendezvous_protocol
        if protocol is None:
            return tuple(position)
        return protocol.drone_missed_endpoint(int(drone_id), position)

    def visible_drone_positions(
        self,
        observer_id: int,
    ) -> tuple[tuple[int, Tuple[int, int]], ...]:
        """Expose no peer position outside direct LOS/proximity."""
        return self.terrain_sharing.visible_drone_positions(observer_id)

    def announce_rendezvous(self, position: Tuple[int, int]):
        protocol = self.rendezvous_protocol
        return None if protocol is None else protocol.propose(position)

    def rendezvous_departure_ready(self, position: Tuple[int, int]) -> bool:
        protocol = self.rendezvous_protocol
        return protocol is None or protocol.can_depart(position)

    def rendezvous_arrived(self, position: Tuple[int, int]) -> bool:
        protocol = self.rendezvous_protocol
        return protocol is None or protocol.rover_arrived(position)

    def _rendezvous_drone_contact(self, first_id: int, second_id: int) -> None:
        if self.rendezvous_protocol is not None:
            self.rendezvous_protocol.drone_drone_contact(first_id, second_id)

    def _rendezvous_drone_rover_contact(self, drone_id: int) -> None:
        if self.rendezvous_protocol is not None:
            self.rendezvous_protocol.drone_rover_contact(drone_id)

    def _new_rendezvous_protocol(self) -> RendezvousProtocol:
        return RendezvousProtocol(
            self.num_drones,
            tuple(self.start_point),
            trace=self._trace_rendezvous,
        )

    def _trace_rendezvous(self, event: str, **fields: object) -> None:
        self.runtime_trace.record(
            event,
            sim_time=self.simulation_time(),
            **fields,
        )

    def exploration_check_in(
        self,
        drone_id: int,
        report: CoordinationReport | None = None,
    ) -> CoordinationResult:
        """Queue a physical rendezvous for the primary rover worker."""
        coordinator = self.exploration_coordinator
        normalized_id = int(drone_id)
        if coordinator is None or not self.rovers or self.rovers[0] is None:
            return CoordinationResult(arrived=False)
        if not self.terrain_sharing.drone_at_rover(normalized_id, 0):
            return CoordinationResult(arrived=False)
        with self._exploration_check_in_lock:
            pending = self._exploration_check_ins.get(normalized_id)
            if pending is None:
                pending = _PendingExplorationCheckIn(
                    drone_id=normalized_id,
                    report=report,
                    ready=threading.Event(),
                    enqueued_at=time.perf_counter(),
                )
                self._exploration_check_ins[normalized_id] = pending
                self._exploration_check_in_queue.put(pending)
                self.runtime_trace.record(
                    "drone_component_check_in_queued",
                    sim_time=self.simulation_time(),
                    drone_id=normalized_id,
                    report_id=None if report is None else report.report_id,
                    directive_id=(
                        None if report is None else report.directive_id
                    ),
                )
        return CoordinationResult(
            arrived=True,
            report_accepted=report is None,
            waiting=True,
            directive_ready=pending.ready,
        )

    def _perform_exploration_check_in(
        self,
        drone_id: int,
        report: CoordinationReport | None = None,
    ) -> CoordinationResult:
        """Merge and schedule one queued report on the rover worker."""
        coordinator = self.exploration_coordinator
        if coordinator is None or not self.rovers or self.rovers[0] is None:
            return CoordinationResult(arrived=False)
        coordinator.update_rover_position(self.get_check_in_position())
        arrived = self.terrain_sharing.check_in_with_rover(drone_id, 0)
        if not arrived:
            return CoordinationResult(arrived=False)
        rover_slam = self.rovers[0].slam_map.snapshot(point_limit=0)
        battery = 100.0
        if 0 <= int(drone_id) < len(self.drones):
            battery = float(self.drones[int(drone_id)].snapshot().battery)
        result = coordinator.check_in(
            drone_id,
            rover_slam,
            report=report,
            energy_state=EnergyState(
                remaining_energy=battery,
                capacity=100.0,
                unlimited=True,
            ),
        )
        coordination_snapshot = coordinator.snapshot()
        self.runtime_trace.record(
            "drone_component_check_in",
            sim_time=self.simulation_time(),
            drone_id=int(drone_id),
            report_id=None if report is None else report.report_id,
            directive_id=None if report is None else report.directive_id,
            directive_kind=None if report is None else report.kind.value,
            claim_token=None if report is None else report.claim_token,
            report_accepted=result.report_accepted,
            waiting=result.waiting,
            mission_exhausted=result.mission_exhausted,
            coordination_phase=coordination_snapshot.phase.value,
            rover_slam_version=rover_slam.version,
        )
        if report is not None and result.report_accepted:
            if report.kind in {
                DirectiveKind.COMPONENT_TASK,
                DirectiveKind.COMPONENT_FOLLOW,
            }:
                self.runtime_trace.record(
                    (
                        "rover_task_completed"
                        if report.kind == DirectiveKind.COMPONENT_TASK
                        else "rover_branch_follow_completed"
                    ),
                    sim_time=self.simulation_time(),
                    drone_id=int(drone_id),
                    report_id=report.report_id,
                    task_id=report.task_id,
                    component_id=report.component_id,
                    claim_token=report.claim_token,
                    work_unit_outcomes=report.work_unit_outcomes,
                    causal_transition_count=len(report.causal_transitions),
                    suspended=report.suspension is not None,
                    suspension_reason=(
                        None if report.suspension is None
                        else report.suspension.reason
                    ),
                )
            elif report.kind == DirectiveKind.RADIAL_PROBE:
                self.runtime_trace.record(
                    "drone_probe_returned",
                    sim_time=self.simulation_time(),
                    drone_id=int(drone_id),
                    report_id=report.report_id,
                    completed_scan_headings=(
                        report.completed_scan_headings
                    ),
                    timed_out_scan_headings=(
                        report.timed_out_scan_headings
                    ),
                    sensor_newly_known_cells=(
                        report.sensor_newly_known_cells
                    ),
                    sensor_confidence_gain=report.sensor_confidence_gain,
                    outbound_distance=self._trace_path_distance(
                        report.outbound_actual_path
                    ),
                    return_distance=self._trace_path_distance(
                        report.return_actual_path
                    ),
                    return_path_source=report.return_path_source,
                )
            if (
                report.kind in {
                    DirectiveKind.ROVER_SCAN,
                    DirectiveKind.RADIAL_PROBE,
                }
                and result.reconcile_result is not None
            ):
                self.runtime_trace.record(
                    "rover_discovery_round_completed",
                    sim_time=self.simulation_time(),
                    round_id=report.round_id,
                    round_kind=report.kind.value,
                    registry_revision=result.reconcile_result.revision,
                    active_component_count=len(
                        result.reconcile_result.active_component_ids
                    ),
                    ready_work_unit_count=len(
                        result.reconcile_result.ready_work_unit_ids
                    ),
                    coordination_phase=coordination_snapshot.phase.value,
                )
        if result.reconcile_result is not None:
            reconcile = result.reconcile_result
            snapshot = coordination_snapshot
            self.runtime_trace.record(
                "rover_frontier_registry_reconciled",
                sim_time=self.simulation_time(),
                revision=reconcile.revision,
                elapsed_ms=reconcile.elapsed_ms,
                rover_slam_version=rover_slam.version,
                active_component_ids=reconcile.active_component_ids,
                active_component_count=len(reconcile.active_component_ids),
                ready_work_unit_ids=reconcile.ready_work_unit_ids,
                ready_work_unit_count=len(reconcile.ready_work_unit_ids),
                dormant_component_count=sum(
                    component.state.value == "dormant"
                    for component in snapshot.components
                ),
                resolved_component_count=sum(
                    component.state.value == "resolved"
                    for component in snapshot.components
                ),
                exploration_modes={
                    mode: sum(
                        component.exploration_mode.value == mode
                        for component in snapshot.components
                        if component.state.value == "active"
                    )
                    for mode in ("focused", "wall_follow", "sweep")
                },
            )
            for transition in reconcile.transitions:
                self.runtime_trace.record(
                    "rover_frontier_lineage_changed",
                    sim_time=self.simulation_time(),
                    revision=reconcile.revision,
                    transition_kind=transition.kind.value,
                    parent_ids=transition.parent_ids,
                    child_ids=transition.child_ids,
                    evidence=transition.evidence,
                )
        return result

    def _process_exploration_check_ins(self) -> None:
        """Drain queued component reports without involving drone workers."""
        while True:
            try:
                pending = self._exploration_check_in_queue.get_nowait()
            except Empty:
                return
            started_at = time.perf_counter()
            pending.result = self._perform_exploration_check_in(
                pending.drone_id,
                pending.report,
            )
            pending.ready.set()
            self.runtime_trace.record(
                "rover_component_check_in_processed",
                sim_time=self.simulation_time(),
                drone_id=pending.drone_id,
                queue_wait_ms=(started_at - pending.enqueued_at) * 1000.0,
                processing_ms=(time.perf_counter() - started_at) * 1000.0,
                report_accepted=pending.result.report_accepted,
            )

    def exploration_energy_return(
        self,
        drone_id: int,
        route_home_cost: float,
        next_action_cost: float,
        safety_reserve: float,
    ):
        """Evaluate a movement checkpoint through the coordinator policy."""
        coordinator = self.exploration_coordinator
        if coordinator is None:
            return None
        battery = 100.0
        if 0 <= int(drone_id) < len(self.drones):
            battery = float(self.drones[int(drone_id)].snapshot().battery)
        return coordinator.evaluate_return(
            drone_id,
            EnergyState(
                remaining_energy=battery,
                capacity=100.0,
                unlimited=True,
            ),
            route_home_cost=route_home_cost,
            next_action_cost=next_action_cost,
            safety_reserve=safety_reserve,
        )

    def exploration_assignment(self, drone_id: int) -> CoordinationResult:
        """Deliver one signalled component, scan, probe, or home directive."""
        coordinator = self.exploration_coordinator
        if coordinator is None:
            return CoordinationResult(arrived=False)
        completed = None
        with self._exploration_check_in_lock:
            pending = self._exploration_check_ins.get(int(drone_id))
            if pending is not None and pending.ready.is_set():
                completed = self._exploration_check_ins.pop(int(drone_id))
        if completed is not None:
            processed = completed.result or CoordinationResult(arrived=False)
            if not processed.arrived or (
                completed.report is not None
                and not processed.report_accepted
            ):
                return processed
            if (
                processed.directive_ready is None
                or not processed.directive_ready.is_set()
            ):
                return processed
            claimed = coordinator.claim_directive(drone_id)
            result = replace(
                claimed,
                report_accepted=processed.report_accepted,
                reconcile_result=processed.reconcile_result,
                published_directives=processed.published_directives,
            )
        else:
            result = coordinator.claim_directive(drone_id)
        directive = result.directive
        if directive is None:
            return result
        if directive.kind in {
            DirectiveKind.COMPONENT_TASK,
            DirectiveKind.COMPONENT_FOLLOW,
            DirectiveKind.RADIAL_PROBE,
        }:
            self.terrain_sharing.share_on_departure(drone_id, 0)
        if (
            directive.round_id is not None
            and directive.round_id not in self._traced_discovery_rounds
        ):
            self._traced_discovery_rounds.add(directive.round_id)
            round_state = coordinator.snapshot().discovery_round
            self.runtime_trace.record(
                "rover_discovery_round_started",
                sim_time=self.simulation_time(),
                round_id=directive.round_id,
                round_kind=directive.kind.value,
                participant_ids=(
                    () if round_state is None
                    else round_state.participant_ids
                ),
                scan_plans=(
                    {} if round_state is None else round_state.scan_plans
                ),
                probe_targets=(
                    {} if round_state is None
                    else round_state.probe_targets
                ),
                ring=0 if round_state is None else round_state.ring,
            )
        if directive.kind == DirectiveKind.COMPONENT_TASK:
            self.runtime_trace.record(
                "rover_task_claimed",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                directive_id=directive.directive_id,
                task_id=directive.task.task_id,
                component_id=directive.task.component_id,
                component_revision=directive.task.component_revision,
                parent_task_id=directive.task.parent_task_id,
                dfs_depth=directive.task.depth,
                work_unit_ids=directive.claim.work_unit_ids,
                claim_token=directive.claim.token,
                entry=directive.task.preferred_entry,
                estimated_effort=directive.task.estimated_effort,
                route_distance=self._trace_path_distance(
                    directive.outbound_route
                ),
            )
        elif directive.kind == DirectiveKind.COMPONENT_FOLLOW:
            self.runtime_trace.record(
                "rover_branch_follow_assigned",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                directive_id=directive.directive_id,
                task_id=directive.task.task_id,
                component_id=directive.task.component_id,
                leader_drone_id=directive.leader_drone_id,
                branch_index=directive.follow_branch_index,
            )
        elif directive.kind == DirectiveKind.RADIAL_PROBE:
            self.runtime_trace.record(
                "drone_probe_started",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                directive_id=directive.directive_id,
                round_id=directive.round_id,
                probe_target=directive.probe_target,
                scan_headings=directive.scan_headings,
                route_distance=self._trace_path_distance(
                    directive.outbound_route
                ),
            )
        elif directive.kind == DirectiveKind.HOME:
            snapshot = coordinator.snapshot()
            self.runtime_trace.record(
                "exploration_homing_started",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                reason=directive.reason,
            )
            self.runtime_trace.record(
                "rover_exploration_quiescence_evaluated",
                sim_time=self.simulation_time(),
                drone_id=int(drone_id),
                reason=directive.reason,
                mission_exhausted=snapshot.mission_exhausted,
                coordination_phase=snapshot.phase.value,
                ready_task_count=sum(
                    task.state.value in {"ready", "suspended", "blocked"}
                    for task in snapshot.tasks
                ),
                live_claim_count=len(snapshot.claims),
            )
        return result

    @staticmethod
    def _trace_path_distance(path: tuple[Tuple[int, int], ...]) -> float:
        return sum(
            float(np.hypot(
                current[0] - previous[0],
                current[1] - previous[1],
            ))
            for previous, current in zip(path, path[1:])
        )

    def sector_check_in(
        self,
        drone_id: int,
        completed_sector_id: int | None,
    ) -> SectorCheckInResult:
        """Exchange maps at the primary rover and advance the sector barrier."""
        if not self.rovers or self.rovers[0] is None:
            return SectorCheckInResult(arrived=False)
        arrived = self.terrain_sharing.check_in_with_rover(drone_id, 0)
        if not arrived:
            return SectorCheckInResult(
                arrived=False,
                generation=self.exploration_sectors.generation,
            )

        if self.exploration_completion_event.is_set():
            self.exploration_sectors.stop()
            return SectorCheckInResult(
                arrived=True,
                mission_exhausted=True,
                generation=self.exploration_sectors.generation,
            )

        outcome_report = None
        if 0 <= int(drone_id) < len(self.drones):
            movement = getattr(
                self.drones[int(drone_id)],
                "movement_controller",
                None,
            )
            report_outcome = getattr(
                movement,
                "sector_outcome_report",
                None,
            )
            if callable(report_outcome):
                outcome_report = report_outcome(completed_sector_id)
        rover_slam = self.rovers[0].slam_map.snapshot(point_limit=0)
        result = self.exploration_sectors.check_in(
            drone_id,
            completed_sector_id,
            rover_slam,
            outcome_report,
        )
        if self.exploration_completion_event.is_set():
            self.exploration_sectors.stop()
            result = SectorCheckInResult(
                arrived=True,
                mission_exhausted=True,
                generation=self.exploration_sectors.generation,
            )
        diagnostics = result.frontier_diagnostics
        if result.published_assignments:
            self.runtime_trace.record(
                "rover_sector_epoch_published",
                sim_time=self.simulation_time(),
                generation=result.generation,
                rover_slam_version=rover_slam.version,
                active_drone_ids=tuple(
                    item.owner_drone_id for item in result.published_assignments
                    if not item.standby
                ),
                standby_drone_ids=tuple(
                    item.owner_drone_id for item in result.published_assignments
                    if item.standby
                ),
                assignments=tuple({
                    "drone_id": item.owner_drone_id,
                    "sector_id": item.sector_id,
                    "standby": item.standby,
                    "frontier_cells": item.frontier_cells,
                    "frontier_component_ids": tuple(
                        component.component_id for component in item.frontier_components
                    ),
                    "estimated_effort": item.estimated_effort,
                    "effort_breakdown": item.effort_breakdown,
                    "scope_pixels": (
                        None if item.exploration_mask is None
                        else int(np.count_nonzero(item.exploration_mask))
                    ),
                } for item in result.published_assignments),
            )
        if diagnostics is not None:
            component_sizes = diagnostics.component_sizes
            self.runtime_trace.record(
                "rover_sector_frontiers_filtered",
                sim_time=self.simulation_time(),
                generation=result.generation,
                rover_slam_version=rover_slam.version,
                raw_frontier_pixels=diagnostics.raw_frontier_pixels,
                raw_component_count=diagnostics.raw_component_count,
                component_size_min=(
                    0 if not component_sizes else min(component_sizes)
                ),
                component_size_max=(
                    0 if not component_sizes else max(component_sizes)
                ),
                component_size_sample=component_sizes[:32],
                significant_frontier_pixels=(
                    diagnostics.significant_frontier_pixels
                ),
                significant_component_count=(
                    diagnostics.significant_component_count
                ),
                large_component_count=diagnostics.large_component_count,
                unknown_supported_component_count=(
                    diagnostics.unknown_supported_component_count
                ),
                unknown_supported_candidate_count=(
                    diagnostics.unknown_supported_candidate_count
                ),
                redundant_unknown_supported_component_count=(
                    diagnostics
                    .redundant_unknown_supported_component_count
                ),
                frontier_unknown_basin_count=(
                    diagnostics.frontier_unknown_basin_count
                ),
                significant_unknown_basin_count=(
                    diagnostics.significant_unknown_basin_count
                ),
                border_connected_unknown_basin_count=(
                    diagnostics.border_connected_unknown_basin_count
                ),
                border_connected_unknown_supported_candidate_count=(
                    diagnostics
                    .border_connected_unknown_supported_candidate_count
                ),
                discarded_component_count=(
                    diagnostics.discarded_component_count
                ),
                discarded_frontier_pixels=(
                    diagnostics.discarded_frontier_pixels
                ),
                minimum_component_cells=(
                    diagnostics.minimum_component_cells
                ),
                minimum_unknown_support_cells=(
                    diagnostics.minimum_unknown_support_cells
                ),
                component_diagnostics=diagnostics.components,
                mission_exhausted=result.mission_exhausted,
            )
        workload = result.workload_diagnostics
        if workload is not None:
            initial = workload.initial_frontier_workloads
            balanced = workload.balanced_frontier_workloads
            self.runtime_trace.record(
                "rover_sector_workload_balanced",
                sim_time=self.simulation_time(),
                generation=result.generation,
                initial_frontier_workloads=initial,
                balanced_frontier_workloads=balanced,
                initial_workload_spread=(
                    0 if not initial else max(initial) - min(initial)
                ),
                balanced_workload_spread=(
                    0 if not balanced else max(balanced) - min(balanced)
                ),
                moved_coarse_cell_count=(
                    workload.moved_coarse_cell_count
                ),
                initial_estimated_efforts=(
                    workload.initial_estimated_efforts
                ),
                balanced_estimated_efforts=(
                    workload.balanced_estimated_efforts
                ),
                initial_estimated_effort_spread=(
                    0.0
                    if not workload.initial_estimated_efforts
                    else max(workload.initial_estimated_efforts)
                    - min(workload.initial_estimated_efforts)
                ),
                balanced_estimated_effort_spread=(
                    0.0
                    if not workload.balanced_estimated_efforts
                    else max(workload.balanced_estimated_efforts)
                    - min(workload.balanced_estimated_efforts)
                ),
            )
        outcome = result.outcome_diagnostics
        if outcome is not None:
            self.runtime_trace.record(
                "rover_sector_frontier_outcomes",
                sim_time=self.simulation_time(),
                generation=result.generation,
                previous_generation=outcome.previous_generation,
                confident_occupied_gain=outcome.confident_occupied_gain,
                remembered_component_count=(
                    outcome.remembered_component_count
                ),
                suppressed_component_count=(
                    outcome.suppressed_component_count
                ),
                suppressed_frontier_pixels=(
                    outcome.suppressed_frontier_pixels
                ),
                remaining_component_count=(
                    outcome.remaining_component_count
                ),
                remaining_frontier_pixels=(
                    outcome.remaining_frontier_pixels
                ),
                evaluated_component_count=(
                    outcome.evaluated_component_count
                ),
                locally_unchanged_component_count=(
                    outcome.locally_unchanged_component_count
                ),
                reported_zero_gain_component_count=(
                    outcome.reported_zero_gain_component_count
                ),
                productive_component_count=(
                    outcome.productive_component_count
                ),
                resolved_component_count=(
                    outcome.resolved_component_count
                ),
                component_outcomes=outcome.components,
                mission_exhausted=result.mission_exhausted,
            )
        suppressions = (
            () if outcome_report is None else outcome_report.suppressions
        )
        self.runtime_trace.record(
            "drone_sector_check_in",
            sim_time=self.simulation_time(),
            drone_id=int(drone_id),
            completed_sector_id=completed_sector_id,
            generation=result.generation,
            waiting_for_team=result.waiting_for_team,
            mission_exhausted=result.mission_exhausted,
            assigned_sector_id=(
                None
                if result.assignment is None
                else result.assignment.sector_id
            ),
            assigned_cell_count=(
                0
                if result.assignment is None
                else len(result.assignment.cells)
            ),
            assigned_frontier_cells=(
                0
                if result.assignment is None
                else result.assignment.frontier_cells
            ),
            assigned_estimated_effort=(
                0.0
                if result.assignment is None
                else result.assignment.estimated_effort
            ),
            assigned_frontier_component_ids=(
                ()
                if result.assignment is None
                else tuple(
                    component.component_id
                    for component in result.assignment.frontier_components
                )
            ),
            local_suppression_component_count=len(suppressions),
            local_suppression_component_ids=tuple(
                suppression.component_id
                for suppression in suppressions
            ),
            local_suppression_reasons=tuple(sorted({
                reason
                for suppression in suppressions
                for reason in suppression.reasons
            })),
            rover_slam_version=rover_slam.version,
        )
        return result

    def sector_assignment(self, drone_id: int) -> SectorCheckInResult:
        """Deliver a signalled assignment after one final rover exchange."""
        if self.exploration_completion_event.is_set():
            self.exploration_sectors.stop()
            return SectorCheckInResult(
                arrived=True,
                mission_exhausted=True,
                generation=self.exploration_sectors.generation,
            )
        result = self.exploration_sectors.claim_assignment(drone_id)
        if result.assignment is None or result.mission_exhausted:
            return result
        if self.exploration_completion_event.is_set():
            self.exploration_sectors.stop()
            return SectorCheckInResult(
                arrived=True,
                mission_exhausted=True,
                generation=self.exploration_sectors.generation,
            )
        if result.assignment.standby:
            return result
        departed = self.terrain_sharing.share_on_departure(drone_id, 0)
        if not departed:
            return SectorCheckInResult(
                arrived=False,
                waiting_for_team=True,
                generation=result.generation,
            )
        if self.exploration_completion_event.is_set():
            self.exploration_sectors.stop()
            return SectorCheckInResult(
                arrived=True,
                mission_exhausted=True,
                generation=self.exploration_sectors.generation,
            )
        self.runtime_trace.record(
            "drone_sector_assignment_delivered",
            sim_time=self.simulation_time(),
            drone_id=int(drone_id),
            generation=result.assignment.generation,
            sector_id=result.assignment.sector_id,
            frontier_cells=result.assignment.frontier_cells,
            estimated_effort=result.assignment.estimated_effort,
            frontier_component_ids=tuple(
                component.component_id
                for component in result.assignment.frontier_components
            ),
        )
        return result


    def compute_rover_path(
        self,
        rover_id: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Plan only through traversable cells received by this rover."""
        normalized_id = int(rover_id)
        if not 0 <= normalized_id < len(self.rovers):
            return []
        rover = self.rovers[normalized_id]
        slam = rover.slam_map.snapshot(point_limit=0)
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        known_free = (
            (occupancy == FREE)
            & (
                confidence
                >= self.settings.frontier.confidence_threshold
            )
        )
        icon = getattr(rover, "icon", None)
        body_radius = max(
            1.0,
            float(max(icon.get_size())) / 2.0 if icon is not None else 1.0,
        )
        clearance = cv2.distanceTransform(
            np.pad(known_free.astype(np.uint8), 1),
            cv2.DIST_L2,
            5,
        )[1:-1, 1:-1]
        rover_safe = known_free & (clearance >= body_radius)
        start_x, start_y = int(start[0]), int(start[1])
        goal_x, goal_y = int(goal[0]), int(goal[1])
        height, width = known_free.shape
        if not (
            0 <= start_x < width
            and 0 <= start_y < height
            and 0 <= goal_x < width
            and 0 <= goal_y < height
            and rover_safe[start_y, start_x]
            and rover_safe[goal_y, goal_x]
        ):
            return []
        traversability = np.ones(rover_safe.shape, dtype=np.uint8)
        traversability[rover_safe] = 0
        terrain = rover.terrain_knowledge.snapshot()
        started_at = time.perf_counter()
        path = self.pathfinding.compute_weighted_path(
            terrain.roughness,
            terrain.confidence,
            start,
            goal,
            traversability_map=traversability,
        )
        self.runtime_trace.record(
            "rover_route_planned",
            sim_time=self.simulation_time(),
            rover_id=normalized_id,
            start=start,
            goal=goal,
            path_length=len(path),
            elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
            rover_slam_version=slam.version,
            local_slam_only=True,
            body_radius=body_radius,
            goal_clearance=float(clearance[goal_y, goal_x]),
        )
        return path


    def rover_thread(self, rover_id: int) -> None:
        """Drive rover movement using the terrain-aware weighted planner."""
        label = ("rover", rover_id)
        if not self.pause_coordinator.register_current_worker(label):
            return
        try:
            while not self.mission_event.is_set():
                if not self.pause_checkpoint():
                    break
                if rover_id == 0:
                    self._process_exploration_check_ins()
                self.rovers[rover_id].move()
                coordinator = self.exploration_coordinator
                if coordinator is not None and rover_id == 0:
                    coordinator.update_rover_position(
                        tuple(self.rovers[rover_id].pos)
                    )
                    self.terrain_sharing.share_with_rovers()
                if not self.wait_simulation_delay(self.delay):
                    break
        finally:
            self.pause_coordinator.unregister_current_worker()

    def pause_checkpoint(self) -> bool:
        """Park the calling simulation worker at a safe pause boundary."""
        return self.pause_coordinator.checkpoint()

    def wait_simulation_delay(self, duration: float) -> bool:
        """Wait for active simulation time, excluding paused intervals."""
        return self.pause_coordinator.wait(duration)

    def simulation_time(self) -> float:
        """Return monotonic mission time with paused intervals removed."""
        return self.simulation_clock.now()

    def toggle_pause(self) -> None:
        """Atomically park or resume all mission workers and simulation time."""
        if self.exploration_complete:
            return
        if not self.is_paused:
            self.is_paused = True
            self.pause_event.clear()
            self.simulation_clock.pause()
            if self.control_center is not None:
                self.control_center.pause_timer()
            self.pause_coordinator.pause()
            return

        self.simulation_clock.resume()
        if self.control_center is not None:
            self.control_center.resume_timer()
        self.is_paused = False
        self.pause_event.set()
        self.pause_coordinator.resume()

    def music_enabled(self) -> bool:
        """Return the menu-owned music state when available."""
        menu = getattr(self.game, "menu", None)
        if menu is None or not hasattr(menu, "music_enabled"):
            return True
        return bool(menu.music_enabled())

    def toggle_music(self) -> None:
        """Toggle persisted background music through the menu facade."""
        menu = getattr(self.game, "menu", None)
        if menu is not None and hasattr(menu, "toggle_music"):
            menu.toggle_music()

    def update_sensors(self) -> None:
        """Update local sensing and display-only floor exploration progress."""
        for drone in self.drones:
            drone.update_sensors()
        self._update_wall_mapping_progress()

    def _update_wall_mapping_progress(self) -> WallMappingSnapshot:
        """Publish wall diagnostics and discovered-floor UI progress.

        Component quiescence is the only exploration completion trigger.  The
        floor ratio deliberately remains observational so a complete mission
        can expose its small unexplored remainder.
        """
        was_complete = self.wall_mapping_progress.complete
        versions = tuple(drone.slam_map.version for drone in self.drones)
        if versions == self.wall_mapping_progress.slam_versions:
            progress = self.wall_mapping_progress
        else:
            progress = wall_mapping_snapshot(
                np.asarray(self.map_matrix),
                tuple(self.drones),
                confidence_threshold=(
                    self.settings.frontier.confidence_threshold
                ),
            )
        self.wall_mapping_progress = progress
        missing_wall_pixels = max(
            0,
            progress.total_wall_pixels - progress.mapped_wall_pixels,
        )
        self.floor_exploration_ratio = self.terrain_knowledge.explored_ratio()
        if self.runtime_trace is not None:
            if progress.complete and not was_complete:
                self.runtime_trace.record(
                    "team_wall_mapping_complete",
                    mapped_wall_pixels=progress.mapped_wall_pixels,
                    total_wall_pixels=progress.total_wall_pixels,
                    missing_wall_pixels=missing_wall_pixels,
                    drone_count=len(self.drones),
                )
        now = self.simulation_time()
        if (
            self.control_center is not None
            and (
                self.last_explored_update == 0.0
                or now - self.last_explored_update
                >= self.explored_update_interval
            )
        ):
            displayed_percent = (
                100
                if self.floor_exploration_ratio >= 1.0
                else min(99, int(self.floor_exploration_ratio * 100.0))
            )
            self.control_center.set_explored_percent(displayed_percent)
            self.last_explored_update = now
        return progress
