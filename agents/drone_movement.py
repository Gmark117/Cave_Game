"""Drone exploration, frontier selection, and homing behavior."""

import logging
import math
import time
from typing import Any, List, Tuple

import numpy as np

from agents.exploration_policy import (
    ExplorationContext,
    ExplorationDecision,
    ExplorationDecisionKind,
)
from contracts import DroneMovementDependencies
from mapping.ray_geometry import bresenham_line_points
from mapping.slam_map import FREE
from navigation.waypoint_graph import (
    ROUTE_DISCONNECTED,
    ROUTE_NO_GOAL_CONNECTOR,
)


Position = Tuple[int, int]
logger = logging.getLogger(__name__)


class DroneMovementController:
    """Drive one drone's exploration state machine."""

    def __init__(
        self,
        drone: Any,
        dependencies: DroneMovementDependencies,
    ) -> None:
        """Bind one drone to movement callbacks supplied by MissionControl."""
        self.drone = drone
        self.dependencies = dependencies
        settings = drone.settings

        # Failed frontier paths are cooled down so a drone does not spend every
        # frame retrying the same unreachable target.
        self.border_retry_cooldown = 1.5
        self.border_retry_until: dict[Position, float] = {}
        self.frontier_stride = settings.frontier.stride
        self.frontier_confidence_threshold = (
            settings.frontier.confidence_threshold
        )
        self.waypoint_config = getattr(settings, "waypoints", None)
        self.waypoint_graph = dependencies.waypoint_graph
        self.waypoint_bridge_attempt_limit = 1
        self._waypoint_pending_path: List[Position] = [
            (int(drone.start_pos[0]), int(drone.start_pos[1]))
        ]
        self._seed_home_waypoint()

    def move(self) -> None:
        """Advance the drone's exploration or homing state."""
        drone = self.drone
        done, _ = drone.runtime_state.evaluate_mission_state()
        if done:
            return

        self._trace(
            "drone_move_start",
            state=self._snapshot_summary(drone.snapshot()),
            slam_version=drone.slam_map.version,
        )
        node_found = False
        while not node_found:
            decision = self.choose_exploration_action()
            self._trace_decision("drone_decision", decision)
            if decision.kind == ExplorationDecisionKind.EXHAUSTED:
                self._trace(
                    "drone_policy_exhausted",
                    state=self._snapshot_summary(drone.snapshot()),
                    slam_version=drone.slam_map.version,
                )
                self.rebuild_frontiers(
                    stride=max(1, self.frontier_stride),
                    confidence_threshold=self.frontier_confidence_threshold,
                )
                decision = self.choose_exploration_action()
                self._trace_decision("drone_post_rebuild_decision", decision)
                if decision.kind == ExplorationDecisionKind.EXHAUSTED:
                    drone.runtime_state.start_returning_home()
                    self._trace(
                        "drone_start_homing_after_exhaustion",
                        state=self._snapshot_summary(drone.snapshot()),
                    )
                    decision = ExplorationDecision(
                        kind=ExplorationDecisionKind.HOMING,
                        target=drone.start_pos,
                    )

            node_found = self.execute_exploration_action(decision)
            self._trace(
                "drone_action_result",
                decision_kind=decision.kind.value,
                node_found=node_found,
                state=self._snapshot_summary(drone.snapshot()),
            )
            if (
                not node_found
                and decision.kind == ExplorationDecisionKind.STEP
                and decision.planned_path
            ):
                return
            if decision.kind != ExplorationDecisionKind.STEP:
                return

    def choose_exploration_action(self) -> ExplorationDecision:
        """Ask the exploration policy for the next high-level action."""
        drone = self.drone
        snapshot = drone.snapshot()
        context = self._build_exploration_context(
            snapshot=snapshot,
            slam_snapshot=drone.slam_map.snapshot(point_limit=0),
        )
        return drone.exploration_policy.decide(
            context,
            drone.runtime_state.graph_is_valid,
        )

    def execute_exploration_action(
        self,
        decision: ExplorationDecision,
    ) -> bool:
        """Execute one policy decision by moving or updating runtime state."""
        if decision.kind == ExplorationDecisionKind.HOMING:
            if self.reach_start_point():
                self.drone.runtime_state.mark_done()
                self._trace("drone_homing_done")
            return True

        if decision.kind == ExplorationDecisionKind.FRONTIER:
            return self._reach_frontier_targets(
                decision.frontier_targets or (
                    (decision.target,) if decision.target is not None else ()
                )
            )

        if decision.kind == ExplorationDecisionKind.STEP:
            return self._execute_step_decision(decision)

        if decision.kind == ExplorationDecisionKind.ROTATE:
            return self._execute_rotate_decision(decision)

        return False

    def reach_start_point(self) -> bool:
        """Follow an A* path back to the drone's starting position."""
        drone = self.drone
        snapshot = drone.snapshot()
        if snapshot.position == drone.start_pos:
            self._trace("drone_homing_already_home")
            return True

        path = self._compute_path(snapshot.position, drone.start_pos)
        self._trace(
            "drone_homing_path",
            start=snapshot.position,
            goal=drone.start_pos,
            path_len=len(path),
        )
        if not path:
            return False

        self._follow_path(path)
        return drone.snapshot().position == drone.start_pos

    def find_new_node(
        self,
    ) -> Tuple[List[int], List[Position], Position]:
        """Choose a locally valid exploration step.

        Raises `AssertionError` when no valid direction remains.
        """
        drone = self.drone
        snapshot = drone.snapshot()
        context = self._build_exploration_context(
            snapshot=snapshot,
        )
        decision = drone.exploration_policy.choose_next_step(
            context,
            drone.runtime_state.graph_is_valid,
        )
        if (
            decision.kind != ExplorationDecisionKind.STEP
            or decision.target is None
            or decision.direction is None
        ):
            raise AssertionError

        drone.runtime_state.set_direction(decision.direction)
        return (
            list(decision.valid_directions),
            list(decision.frontier_targets),
            decision.target,
        )

    def _execute_step_decision(
        self,
        decision: ExplorationDecision,
    ) -> bool:
        """Execute a policy-selected exploration step."""
        if decision.target is None or decision.direction is None:
            return False

        self.drone.runtime_state.set_direction(decision.direction)
        if decision.planned_path:
            self.drone.runtime_state.begin_exploration(
                decision.direction,
                decision.frontier_targets,
            )
            return self._follow_policy_path(list(decision.planned_path))

        return self.explore(
            list(decision.valid_directions),
            list(decision.frontier_targets),
            decision.target,
        )

    def _execute_rotate_decision(
        self,
        decision: ExplorationDecision,
    ) -> bool:
        """Execute an in-place policy rotation."""
        if decision.direction is None:
            return False
        self.drone.runtime_state.begin_exploration(
            decision.direction,
            decision.frontier_targets,
        )
        return True

    def explore(
        self,
        valid_dirs: List[int],
        valid_targets: List[Position],
        chosen_target: Position,
    ) -> bool:
        """Attempt exploration toward `chosen_target`."""
        drone = self.drone
        snapshot = drone.snapshot()
        chosen_direction = snapshot.direction
        drone.runtime_state.begin_exploration(
            chosen_direction,
            valid_targets,
        )
        valid_dirs.remove(chosen_direction)

        path = self._compute_path(snapshot.position, chosen_target)
        if not path:
            return False

        self._follow_path(path)
        return True

    def reach_border(self) -> bool:
        """Follow an A* path to the nearest viable frontier."""
        drone = self.drone
        snapshot = drone.snapshot()
        frontiers = list(self._prioritized_frontiers(snapshot))

        if not frontiers:
            self.maybe_rebuild_frontiers()
            snapshot = drone.snapshot()
            frontiers = list(self._prioritized_frontiers(snapshot))
            if not frontiers:
                return False

        return self._reach_frontier_targets(frontiers)

    def _reach_frontier_targets(
        self,
        frontiers: tuple[Position, ...] | list[Position],
    ) -> bool:
        """Reach a frontier directly or advance one sparse highway segment."""
        drone = self.drone
        now = self._simulation_time()
        slam_snapshot = None
        known_free = None
        bridge_budget = [self.waypoint_bridge_attempt_limit]
        for target in frontiers:
            current_position = drone.snapshot().position
            if target == current_position:
                self._trace(
                    "drone_frontier_skip",
                    target=target,
                    reason="current_position",
                )
                continue
            retry_at = self.border_retry_until.get(target, 0.0)
            if now < retry_at:
                self._trace(
                    "drone_frontier_skip",
                    target=target,
                    reason="cooldown",
                    retry_in=retry_at - now,
                )
                continue

            distance = math.dist(current_position, target)
            direct_limit = self._direct_path_limit()
            direct_attempted = (
                self.waypoint_graph is None or distance <= direct_limit
            )
            if direct_attempted:
                path = self._compute_path(current_position, target)
                self._trace(
                    "drone_frontier_path",
                    start=current_position,
                    target=target,
                    distance=distance,
                    attempted=True,
                    path_len=len(path),
                )
                if path and len(path) > 1:
                    followed = self._follow_path(path)
                    if self._frontier_was_reached(target):
                        self._mark_frontier_reached(target)
                        return True
                    # A shutdown or pause-barrier stop must not consume the
                    # target or start another route from the partially moved
                    # position in the same action.
                    if not followed:
                        return False
                    if drone.snapshot().position != current_position:
                        return True

                self._trace(
                    "drone_frontier_direct_path_failed",
                    start=current_position,
                    target=target,
                    distance=distance,
                    path_len=len(path),
                )
            else:
                self._trace(
                    "drone_frontier_direct_path_skipped",
                    start=current_position,
                    target=target,
                    distance=distance,
                    direct_path_limit=direct_limit,
                    reason="far_target",
                )

            if self.waypoint_graph is not None:
                if slam_snapshot is None:
                    slam_snapshot = drone.slam_map.snapshot(point_limit=0)
                    known_free = self._known_free_mask(slam_snapshot)
                    current_x, current_y = current_position
                    if (
                        0 <= current_y < known_free.shape[0]
                        and 0 <= current_x < known_free.shape[1]
                    ):
                        # Occupying the current cell is direct traversal
                        # evidence even if the latest scan confidence is low.
                        known_free[current_y, current_x] = True
                if known_free is not None and self._advance_waypoint_segment(
                    current_position,
                    target,
                    known_free,
                    bridge_budget=bridge_budget,
                ):
                    return True

            self.border_retry_until[target] = (
                now + self.border_retry_cooldown
            )

        self._trace(
            "drone_frontier_targets_exhausted",
            frontier_count=len(frontiers),
            state=self._snapshot_summary(drone.snapshot()),
        )
        return False

    def _advance_waypoint_segment(
        self,
        start: Position,
        target: Position,
        known_free: np.ndarray,
        *,
        bridge_budget: List[int] | None = None,
    ) -> bool:
        """Plan and traverse only the next segment toward a far frontier."""
        graph = self.waypoint_graph
        if graph is None:
            return False

        # Short policy actions are accumulated across calls to preserve sparse
        # spacing. A route needs the live pose anchored, so flush the remaining
        # physically travelled tail before connector search.
        self._flush_pending_waypoint_path(force=True)
        route_started = time.perf_counter()
        route = graph.find_route(start, target, known_free)
        initial_route_status = route.status
        bridge_status = "not_attempted"
        bridge_update = None
        bridge_search_distance = self._waypoint_bridge_distance()
        bridgeable_statuses = {
            ROUTE_NO_GOAL_CONNECTOR,
            ROUTE_DISCONNECTED,
        }
        allow_bridge = (
            bridge_budget is None or bridge_budget[0] > 0
        )
        if route.status in bridgeable_statuses and allow_bridge:
            if bridge_budget is not None:
                bridge_budget[0] -= 1
            bridge_update = graph.connect_known_free_corridor(
                target,
                known_free,
                search_distance=bridge_search_distance,
                source="gateway",
            )
            bridge_status = bridge_update.status
            if bridge_update.status == "ok":
                route = graph.find_route(start, target, known_free)
        elif route.status in bridgeable_statuses:
            bridge_status = "attempt_budget_exhausted"
        gateway_status = "not_attempted"
        gateway_update = None
        if route.found:
            # Route search already uses an ephemeral, requester-validated goal
            # connector. Persist a gateway only for a target that can actually
            # make progress, avoiding graph growth from failed alternatives.
            gateway_update = graph.connect_known_free_waypoint(
                target,
                known_free,
                source="gateway",
            )
            gateway_status = gateway_update.status
        route_elapsed_ms = (time.perf_counter() - route_started) * 1000.0
        graph_nodes, graph_edges = graph.counts()
        if bridge_update is not None:
            self._trace_waypoint_update(bridge_update)
            self._trace(
                "drone_waypoint_bridge",
                start=start,
                target=target,
                initial_route_status=initial_route_status,
                status=bridge_status,
                search_distance=bridge_search_distance,
                sampled_waypoint_count=len(
                    bridge_update.sampled_waypoints
                ),
                sampled_waypoints=self._position_sample(
                    list(bridge_update.sampled_waypoints)
                ),
                added_waypoint_count=len(
                    bridge_update.added_waypoints
                ),
                added_edge_count=len(bridge_update.added_edges),
            )
        if gateway_update is not None:
            self._trace_waypoint_update(gateway_update)
        route_positions = tuple(route.waypoints)
        next_waypoint = (
            route_positions[1] if len(route_positions) > 1 else None
        )
        self._trace(
            "drone_waypoint_route",
            start=start,
            target=target,
            initial_status=initial_route_status,
            status=route.status,
            bridge_status=bridge_status,
            gateway_status=gateway_status,
            route_len=len(route_positions),
            route_sample=self._position_sample(list(route_positions)),
            next_waypoint=next_waypoint,
            route_cost=route.cost if math.isfinite(route.cost) else None,
            route_elapsed_ms=route_elapsed_ms,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )
        if not route.found or next_waypoint is None:
            return False

        astar_path = self._compute_path(start, next_waypoint)
        trusted_path = list(route.first_segment_path)
        segment_path: List[Position] = []
        path_source = "failed"
        if (
            astar_path
            and len(astar_path) > 1
            and tuple(astar_path[0]) == start
            and tuple(astar_path[-1]) == next_waypoint
            and self._segment_path_is_known(
                astar_path,
                known_free,
                trusted_path,
            )
        ):
            segment_path = astar_path
            path_source = "astar"
        elif len(trusted_path) > 1:
            # Every route edge is either physical traversal evidence or was
            # validated against this requester's known-free mask. Following
            # that stored shape is safer than accepting an omniscient A*
            # shortcut through cells this drone has not learned yet.
            segment_path = trusted_path
            path_source = "trusted_route_fallback"

        self._trace(
            "drone_waypoint_segment_path",
            start=start,
            target=target,
            segment_goal=next_waypoint,
            route_source=route.first_segment_source,
            path_source=path_source,
            astar_path_len=len(astar_path),
            path_len=len(segment_path),
        )
        if len(segment_path) <= 1:
            return False

        followed = self._follow_path(segment_path)
        current = self.drone.snapshot().position
        if self._frontier_was_reached(target):
            self._mark_frontier_reached(target)
            return True
        if not followed:
            return False
        if current == start:
            return False

        self.border_retry_until.pop(target, None)
        self._trace(
            "drone_waypoint_segment_complete",
            target=target,
            segment_goal=next_waypoint,
            state=self._snapshot_summary(self.drone.snapshot()),
        )
        return True

    def _frontier_was_reached(self, target: Position) -> bool:
        """Return whether movement ended at the selected frontier."""
        return self.drone.snapshot().position == target

    def _mark_frontier_reached(self, target: Position) -> None:
        """Consume and trace a frontier only after physically reaching it."""
        self.drone.runtime_state.remove_frontier(target)
        self.border_retry_until.pop(target, None)
        self._trace(
            "drone_frontier_reached",
            target=target,
            state=self._snapshot_summary(self.drone.snapshot()),
        )

    def update_borders(self) -> None:
        """Rebuild frontier targets when the cooldown permits."""
        self.maybe_rebuild_frontiers()

    def maybe_rebuild_frontiers(self) -> bool:
        """Rebuild frontiers if the configured cooldown has elapsed."""
        now = self._simulation_time()
        if not self.drone.runtime_state.reserve_frontier_rebuild(now):
            return False

        self.rebuild_frontiers(
            stride=max(1, self.frontier_stride),
            confidence_threshold=self.frontier_confidence_threshold,
        )
        return True

    def rebuild_frontiers(
        self,
        stride: int = 4,
        confidence_threshold: float = 0.6,
    ) -> None:
        """Extract frontier cells from local SLAM and terrain confidence."""
        drone = self.drone
        slam = drone.slam_map.snapshot(point_limit=0)
        terrain = drone.terrain_knowledge.snapshot()
        context = self._build_exploration_context(
            snapshot=drone.snapshot(),
            slam_snapshot=slam,
            terrain_snapshot=terrain,
        )
        frontiers = drone.exploration_policy.extract_frontiers(
            context,
            stride=stride,
            confidence_threshold=confidence_threshold,
        )
        update_registry = getattr(
            drone.exploration_policy,
            "update_priority_frontier_registry",
            None,
        )
        if update_registry is not None:
            update_registry(context, frontiers)

        drone.runtime_state.replace_frontiers(frontiers)
        self.border_retry_until = {}
        self._trace(
            "drone_frontiers_rebuilt",
            frontier_count=len(frontiers),
            frontier_sample=self._position_sample(frontiers),
            slam_version=slam.version,
        )

    def mission_completed(self) -> bool:
        """Return True once exploration is exhausted and the drone is home."""
        drone = self.drone
        snapshot = drone.snapshot()
        if not snapshot.explored:
            return False

        done, _ = drone.runtime_state.evaluate_mission_state()
        if done:
            logger.info("Drone %s has completed the mission", drone.id)
            return True

        return False

    def get_distance(self, target: Position) -> float:
        """Return distance to a frontier, deprioritizing already-visible cells."""
        context = self._build_exploration_context(
            snapshot=self.drone.snapshot(),
        )
        return self.drone.exploration_policy.frontier_distance(
            context,
            target,
        )

    def _distance_from(
        self,
        position: Position,
        target: Position,
    ) -> float:
        """Return distance while pushing already-visible frontiers to the end."""
        distance = math.dist(position, target)
        if distance <= self.drone.radius:
            return float(self.drone.game.width)
        return distance

    def _prioritized_frontiers(
        self,
        snapshot: Any,
    ) -> tuple[Position, ...]:
        """Return frontiers ordered by the drone's exploration policy."""
        context = self._build_exploration_context(snapshot=snapshot)
        return self.drone.exploration_policy.prioritize_frontiers(context)

    def _build_exploration_context(
        self,
        *,
        snapshot: Any,
        slam_snapshot: Any | None = None,
        terrain_snapshot: Any | None = None,
    ) -> ExplorationContext:
        """Build detached policy inputs around the current pose estimate."""
        drone = self.drone
        pose_estimate = drone.localizer.estimate(
            snapshot,
            timestamp=self._simulation_time(),
        )
        cave = np.asarray(drone.cave)
        width = int(cave.shape[1]) if cave.ndim == 2 else int(drone.game.width)
        return ExplorationContext(
            pose_estimate=pose_estimate,
            runtime_snapshot=snapshot,
            cave_map=cave,
            start_position=drone.start_pos,
            step=drone.step,
            radius=drone.radius,
            map_width=width,
            frontier_stride=self.frontier_stride,
            frontier_confidence_threshold=self.frontier_confidence_threshold,
            battery=snapshot.battery,
            slam_snapshot=slam_snapshot,
            terrain_snapshot=terrain_snapshot,
        )

    def _compute_path(
        self,
        start: Position,
        goal: Position,
    ) -> List[Position]:
        """Ask MissionControl's pathfinding service for a route."""
        return self.dependencies.compute_path(start, goal)

    def _direct_path_limit(self) -> float:
        """Return the largest distance that may use one direct A* request."""
        if self.waypoint_config is None:
            return float("inf")
        return float(self.waypoint_config.direct_path_limit)

    def _waypoint_bridge_distance(self) -> float:
        """Return the bounded last-mile search radius for observed frontiers."""
        graph = self.waypoint_graph
        if graph is None:
            return self._direct_path_limit()
        return self._direct_path_limit() + float(graph.connector_distance)

    def _known_free_mask(self, slam_snapshot: Any) -> np.ndarray:
        """Build the strict SLAM-only traversability mask used by highways."""
        return (
            (slam_snapshot.occupancy == FREE)
            & (
                slam_snapshot.confidence
                >= self.frontier_confidence_threshold
            )
        )

    @staticmethod
    def _segment_path_is_known(
        path: List[Position],
        known_free: np.ndarray,
        trusted_path: List[Position],
    ) -> bool:
        """Reject A* shortcuts outside known-free or travelled route cells."""
        trusted_cells = set(trusted_path)
        height, width = known_free.shape

        def allowed(point: Position) -> bool:
            x, y = point
            return (
                0 <= x < width
                and 0 <= y < height
                and (
                    bool(known_free[y, x])
                    or point in trusted_cells
                )
            )

        previous: Position | None = None
        previous_cell: Position | None = None
        for raw_point in path:
            point = (int(raw_point[0]), int(raw_point[1]))
            points = (
                [point]
                if previous is None
                else bresenham_line_points(
                    previous[0],
                    previous[1],
                    point[0],
                    point[1],
                )[1:]
            )
            for cell in points:
                if not allowed(cell):
                    return False
                if (
                    previous_cell is not None
                    and cell[0] != previous_cell[0]
                    and cell[1] != previous_cell[1]
                    and not (
                        allowed((cell[0], previous_cell[1]))
                        or allowed((previous_cell[0], cell[1]))
                    )
                ):
                    return False
                previous_cell = cell
            previous = point
        return previous is not None

    def _simulation_time(self) -> float:
        """Return mission time with paused duration removed."""
        return self.dependencies.simulation_time()

    def _follow_path(self, path: List[Position]) -> bool:
        """Walk a path one node at a time, stopping cleanly on pause/shutdown."""
        drone = self.drone
        executed = [drone.snapshot().position]
        try:
            for node in path:
                normalized = (int(node[0]), int(node[1]))
                if not self.dependencies.pause_checkpoint():
                    return False

                drone.runtime_state.move_to(normalized)
                if normalized != executed[-1]:
                    executed.append(normalized)

                if not self.dependencies.wait_simulation_delay(
                    drone.delay / drone.speed_factor
                ):
                    return False
            return True
        finally:
            self._register_travelled_path(executed)

    def _seed_home_waypoint(self) -> None:
        """Add the shared mission home node before any drone starts moving."""
        graph = self.waypoint_graph
        if graph is None:
            return
        waypoint, added = graph.add_waypoint(
            self.drone.start_pos,
            source="home",
        )
        if added:
            self._trace(
                "waypoint_added",
                waypoint=waypoint,
                source="home",
                node_count=graph.node_count,
            )

    def _register_travelled_path(self, path: List[Position]) -> None:
        """Accumulate executed motion until one sparse interval is available."""
        graph = self.waypoint_graph
        if graph is None or len(path) <= 1:
            return

        normalized: List[Position] = []
        for point in path:
            position = (int(point[0]), int(point[1]))
            if not normalized or normalized[-1] != position:
                normalized.append(position)
        if len(normalized) <= 1:
            return

        pending = self._waypoint_pending_path
        if not pending:
            pending.append(normalized[0])
        elif pending[-1] != normalized[0]:
            # External/test motion may bypass this controller. Do not invent an
            # edge across that discontinuity; retain only the new observed tail.
            self._flush_pending_waypoint_path(force=True)
            pending = [normalized[0]]
            self._waypoint_pending_path = pending
        pending.extend(normalized[1:])
        self._flush_pending_waypoint_path(force=False)

    def _flush_pending_waypoint_path(self, *, force: bool) -> None:
        """Commit the accumulated travelled tail when spacing or routing needs it."""
        graph = self.waypoint_graph
        pending = self._waypoint_pending_path
        if graph is None or len(pending) <= 1:
            return

        dense: List[Position] = [pending[0]]
        for start, end in zip(pending, pending[1:]):
            dense.extend(
                bresenham_line_points(
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                )[1:]
            )
        cumulative_distances = [0.0]
        for start, end in zip(dense, dense[1:]):
            cumulative_distances.append(
                cumulative_distances[-1] + math.dist(start, end)
            )
        total_distance = cumulative_distances[-1]

        if force:
            commit_index = len(dense) - 1
        else:
            completed_intervals = int(
                (total_distance + 1e-9) // graph.spacing
            )
            if completed_intervals <= 0:
                return
            commit_distance = completed_intervals * graph.spacing
            commit_index = next(
                index
                for index, distance in enumerate(cumulative_distances)
                if distance + 1e-9 >= commit_distance
            )

        committed = dense[: commit_index + 1]
        self._waypoint_pending_path = dense[commit_index:]
        update = graph.register_travelled_path(committed)
        self._trace_waypoint_update(update)

    def _trace_waypoint_update(self, update: Any) -> None:
        """Trace actual sparse-graph mutations, excluding duplicate merges."""
        graph = self.waypoint_graph
        if graph is None:
            return
        for node in update.added_waypoints:
            self._trace(
                "waypoint_added",
                waypoint=node.position,
                source=node.source,
                node_count=graph.node_count,
            )
        for edge in update.added_edges:
            self._trace(
                "waypoint_edge_added",
                start=edge.start,
                end=edge.end,
                source=edge.source,
                distance=edge.cost,
                edge_count=graph.edge_count,
            )

    def _trace(self, event: str, **fields: Any) -> None:
        """Write one movement trace event when runtime tracing is enabled."""
        trace = getattr(self.dependencies, "runtime_trace", None)
        if trace is None:
            return
        trace.record(
            event,
            sim_time=self._simulation_time(),
            drone_id=self.drone.id,
            **fields,
        )

    def _trace_decision(
        self,
        event: str,
        decision: ExplorationDecision,
    ) -> None:
        """Trace a policy decision with the latest MCTS diagnostics."""
        self._trace(
            event,
            decision=self._decision_summary(decision),
            mcts=self._mcts_summary(),
        )

    def _decision_summary(
        self,
        decision: ExplorationDecision,
    ) -> dict[str, Any]:
        """Return compact JSON-safe decision fields."""
        return {
            "kind": decision.kind.value,
            "target": decision.target,
            "direction": decision.direction,
            "valid_direction_count": len(decision.valid_directions),
            "frontier_count": len(decision.frontier_targets),
            "frontier_sample": self._position_sample(
                decision.frontier_targets
            ),
            "planned_path_len": len(decision.planned_path),
        }

    def _mcts_summary(self) -> dict[str, Any] | None:
        """Return the latest MCTS diagnostic summary, when present."""
        policy = getattr(self.drone, "exploration_policy", None)
        diagnostics = getattr(policy, "last_search_diagnostics", None)
        if diagnostics is None:
            return None
        config = getattr(policy, "config", None)
        root_limit = getattr(
            self.drone.settings.trace,
            "mcts_root_visits",
            0,
        )
        root_visits = sorted(
            diagnostics.root_visits,
            key=lambda visit: (-visit.visits, -visit.mean_reward),
        )[:root_limit]
        return {
            "iterations": diagnostics.iterations,
            "max_iterations": getattr(config, "iterations", None),
            "generated_nodes": diagnostics.generated_nodes,
            "selected_kind": diagnostics.selected_kind,
            "selected_direction": diagnostics.selected_direction,
            "selected_target": diagnostics.selected_target,
            "selected_reward": diagnostics.selected_reward,
            "slam_version": diagnostics.slam_version,
            "elapsed_ms": diagnostics.elapsed_ms,
            "root_visits": [
                {
                    "kind": visit.kind,
                    "direction": visit.direction,
                    "target": visit.target,
                    "visits": visit.visits,
                    "mean_reward": visit.mean_reward,
                }
                for visit in root_visits
            ],
        }

    @staticmethod
    def _snapshot_summary(snapshot: Any) -> dict[str, Any]:
        """Return compact drone runtime state for trace events."""
        return {
            "position": snapshot.position,
            "direction": snapshot.direction,
            "heading": snapshot.heading_deg,
            "frontier_count": len(snapshot.frontiers),
            "frontier_sample": DroneMovementController._position_sample(
                snapshot.frontiers
            ),
            "returning_home": snapshot.returning_home,
            "done": snapshot.done,
            "explored": snapshot.explored,
            "battery": snapshot.battery,
            "path_len": len(snapshot.path_history),
        }

    @staticmethod
    def _position_sample(
        positions: tuple[Position, ...] | list[Position],
        limit: int = 8,
    ) -> list[Position]:
        """Return the first few positions for readable trace events."""
        return [
            (int(position[0]), int(position[1]))
            for position in tuple(positions)[:limit]
        ]

    def _follow_policy_path(self, path: List[Position]) -> bool:
        """Walk a policy path after simulator collision validation."""
        drone = self.drone
        current = drone.snapshot().position
        executed = [current]
        try:
            for raw_node in path:
                node = (int(raw_node[0]), int(raw_node[1]))
                if node == current:
                    continue
                try:
                    valid_segment = drone.runtime_state.graph_is_valid(
                        current,
                        node,
                    )
                except (IndexError, ValueError):
                    self._trace(
                        "drone_policy_path_invalid",
                        current=current,
                        node=node,
                        reason="validation_exception",
                    )
                    return False
                if not valid_segment:
                    slam_updated = drone.slam_map.record_collision(node)
                    self._trace(
                        "drone_policy_path_invalid",
                        current=current,
                        node=node,
                        reason="blocked_segment",
                        slam_updated=slam_updated,
                        slam_version=drone.slam_map.version,
                    )
                    return False
                if not self.dependencies.pause_checkpoint():
                    return False

                drone.runtime_state.move_to(node)
                current = node
                executed.append(node)

                if not self.dependencies.wait_simulation_delay(
                    drone.delay / drone.speed_factor
                ):
                    return False
            return True
        finally:
            self._register_travelled_path(executed)
