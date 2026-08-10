"""Drone exploration, frontier selection, and homing behavior."""

import logging
import math
import time
from dataclasses import replace
from typing import Any, List, Tuple

import numpy as np

from agents.exploration_policy import (
    ExplorationContext,
    ExplorationDecision,
    ExplorationDecisionKind,
)
from contracts import DroneMovementDependencies
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN
from navigation.waypoint_graph import (
    EDGE_KNOWN_FREE_CONNECTOR,
    EDGE_KNOWN_FREE_CORRIDOR,
    EDGE_SLAM_LOS,
    EDGE_TRAVELLED,
    ROUTE_DISCONNECTED,
    ROUTE_NO_GOAL_CONNECTOR,
    ROUTE_NO_START_CONNECTOR,
    bresenham_path,
    graph_delta_trace_fields,
    validate_known_free_path,
)
from navigation.navigation_intent import (
    MovementMode,
    MovementOutcome,
    NavigationIntent,
    NavigationWatchdog,
    TransitionReason,
)
from navigation.strategic_selection import (
    StrategicCandidate,
    select_strategic_candidates,
)
from navigation.frontier_clusters import select_accessible_frontier_waypoint
from navigation.trail_accumulator import StrategicTrailAccumulator


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
        self.frontier_confidence_threshold = (
            settings.frontier.confidence_threshold
        )
        self.wall_continuation_min_distance = float(
            settings.frontier.continuation_min_distance
        )
        self.wall_continuation_scan_headings = int(
            settings.frontier.continuation_scan_headings
        )
        self.waypoint_config = getattr(settings, "waypoints", None)
        self.waypoint_graph = dependencies.waypoint_graph
        self.waypoint_bridge_attempt_limit = 1
        self._frontier_assignment_tokens: dict[int, int] = {}
        self._retained_cluster_id: int | None = None
        self._frontier_waypoints: dict[int, Position] = {}
        self._scan_suppressed_clusters: dict[int, tuple[Any, ...]] = {}
        self._unreachable_blacklist: dict[
            int,
            tuple[int, int, str, tuple[int, int, int, int], bytes],
        ] = {}
        self._last_waypoint_route_transient_failure = False
        self._last_frontier_selection_status: str | None = None
        self._last_frontier_selection_details: dict[str, Any] = {}
        self._frontier_wait_until = 0.0
        self._frontier_wait_signature: tuple[Any, ...] | None = None
        start_position = (int(drone.start_pos[0]), int(drone.start_pos[1]))
        self._trail_accumulator = StrategicTrailAccumulator(
            start_position,
            turn_threshold_degrees=float(getattr(
                self.waypoint_config, "turn_threshold_degrees", 45.0,
            )),
            minimum_turn_leg=float(getattr(
                self.waypoint_config, "minimum_turn_leg", 24.0,
            )),
            recovery_interval=float(getattr(
                self.waypoint_config, "recovery_anchor_interval", 128.0,
            )),
            chokepoint_narrow_clearance=float(getattr(
                self.waypoint_config, "chokepoint_narrow_clearance", 8.0,
            )),
            chokepoint_shoulder_clearance=float(getattr(
                self.waypoint_config, "chokepoint_shoulder_clearance", 16.0,
            )),
            chokepoint_shoulder_length=float(getattr(
                self.waypoint_config, "chokepoint_shoulder_length", 24.0,
            )),
        )
        self._seed_home_waypoint()

    def move(self) -> None:
        """Advance the drone's exploration or homing state."""
        drone = self.drone
        coordinator = getattr(drone, "exploration_coordinator", None)
        reconciliation = None
        if coordinator is not None:
            reconciliation = coordinator.synchronize()
        done, _ = drone.runtime_state.evaluate_mission_state()
        if done:
            return
        if self._frontier_wait_blocks(reconciliation):
            return

        active_intent = drone.snapshot().navigation_intent
        if active_intent is not None:
            outcome = self._execute_active_navigation_intent(active_intent)
            current_intent = drone.snapshot().navigation_intent
            self._trace(
                "drone_intent_result",
                cluster_id=active_intent.cluster_id,
                intent_before=self._intent_summary(active_intent),
                intent_after=self._intent_summary(current_intent),
                assignment_token=active_intent.assignment_token,
                assignment_id=active_intent.assignment_token,
                route_id=active_intent.route_id,
                transition_reason=outcome.transition_reason.value,
                travelled_distance=outcome.travelled_distance,
                route_progress_delta=outcome.route_progress_delta,
                cursor_progress={
                    "edge_cursor_before": active_intent.edge_cursor,
                    "polyline_cursor_before": active_intent.polyline_cursor,
                    "remaining_cost_before": active_intent.remaining_route_cost,
                    "edge_cursor_after": (
                        None if current_intent is None
                        else current_intent.edge_cursor
                    ),
                    "polyline_cursor_after": (
                        None if current_intent is None
                        else current_intent.polyline_cursor
                    ),
                    "remaining_cost_after": (
                        None if current_intent is None
                        else current_intent.remaining_route_cost
                    ),
                    "route_progress_delta": outcome.route_progress_delta,
                },
                actual_information_gain=outcome.actual_information_gain,
                movement_outcome=self._outcome_summary(outcome),
                invalidated=outcome.invalidated,
            )
            if not outcome.invalidated:
                return
            if drone.snapshot().navigation_intent is not None:
                # The previous intent was invalidated into an explicit local
                # recovery intent; do not fall through into global selection.
                return
        if self._retained_cluster_id is not None:
            retained_id = self._retained_cluster_id
            cluster = next((
                item for item in drone.frontier_registry.visible_to(drone.id)
                if item.id == retained_id
            ), None)
            strict_candidates = self._score_frontier_clusters(
                tuple(drone.snapshot().frontier_cluster_ids)
            )
            if cluster is not None and retained_id in strict_candidates:
                if self._reach_frontier_clusters((cluster.id,)):
                    return
                if (
                    self._last_frontier_selection_status
                    == TransitionReason.NO_ACTIONABLE_FRONTIER.value
                ):
                    self._wait_for_unactionable_frontiers((cluster.id,))
                    return
            # Retention cannot bypass the strict accessible wall tier. A
            # cluster that lost wall gain is reconsidered through the global
            # candidate set when another wall continuation is actionable.
            self._retained_cluster_id = None

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
                )
                decision = self.choose_exploration_action()
                self._trace_decision("drone_post_rebuild_decision", decision)
                if decision.kind == ExplorationDecisionKind.EXHAUSTED:
                    team_exhausted = (
                        coordinator.report_exhausted(drone.id)
                        if coordinator is not None else True
                    )
                    if not team_exhausted:
                        wait_seconds = self._begin_frontier_wait()
                        self._trace(
                            "drone_waiting_for_team_exhaustion",
                            state=self._snapshot_summary(drone.snapshot()),
                            retry_after_seconds=wait_seconds,
                        )
                        return
                    if coordinator is None:
                        drone.runtime_state.start_returning_home()
                    self._trace(
                        "drone_start_homing_after_exhaustion",
                        state=self._snapshot_summary(drone.snapshot()),
                    )
                    decision = ExplorationDecision(
                        kind=ExplorationDecisionKind.HOMING,
                        target=drone.start_pos,
                    )

            outcome = self.execute_exploration_action(decision)
            node_found = bool(outcome)
            self._trace(
                "drone_action_result",
                decision_kind=decision.kind.value,
                node_found=node_found,
                movement_outcome={
                    "travelled_distance": outcome.travelled_distance,
                    "route_progress_delta": outcome.route_progress_delta,
                    "arrived": outcome.arrived,
                    "collision": outcome.collision,
                    "scan_complete": outcome.scan_complete,
                    "actual_information_gain": outcome.actual_information_gain,
                    "invalidated": outcome.invalidated,
                    "transition_reason": outcome.transition_reason.value,
                },
                state=self._snapshot_summary(drone.snapshot()),
            )
            if (
                outcome.transition_reason
                == TransitionReason.NO_ACTIONABLE_FRONTIER
            ):
                requested_ids = tuple(
                    decision.frontier_cluster_ids or (
                        (decision.cluster_id,)
                        if decision.cluster_id is not None else ()
                    )
                )
                self._wait_for_unactionable_frontiers(requested_ids)
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
        policy = drone.exploration_policy
        context = self._build_exploration_context(
            snapshot=snapshot,
            slam_snapshot=(
                None
                if getattr(policy, "uses_strategic_control", False)
                else drone.slam_map.snapshot(point_limit=0)
            ),
        )
        return policy.decide(
            context,
        )

    def _execute_active_navigation_intent(
        self,
        intent: NavigationIntent,
    ) -> MovementOutcome:
        """Use local control only for deviations, scanning, and recovery."""
        invalid = self._validate_navigation_intent(intent)
        if invalid is not None:
            return invalid
        if intent.local_scan_pending:
            return self._complete_local_scan_if_ready(intent)
        policy = self.drone.exploration_policy
        decide_local = getattr(policy, "decide_local", None)
        needs_local = (
            intent.mode in {MovementMode.SCAN, MovementMode.RECOVERY}
            or self._pose_deviates_from_route(intent)
        )
        if decide_local is None or not needs_local:
            primitive = (
                "recovery"
                if intent.mode == MovementMode.RECOVERY
                else "follow_edge"
            )
            updated = replace(intent, previous_primitive=primitive)
            self.drone.runtime_state.replace_navigation_intent(updated)
            return self._execute_navigation_intent(updated)

        snapshot = self.drone.snapshot()
        context = self._build_exploration_context(
            snapshot=snapshot,
        )
        if getattr(policy, "uses_local_mcts", False):
            decision = decide_local(
                context,
                slam_snapshot_provider=self.drone.slam_map.try_snapshot_window,
                slam_shape=self.drone.slam_map.shape,
                slam_version_hint=self.drone.slam_map.version,
            )
        else:
            decision = decide_local(context)
        self._trace_decision("drone_local_mcts_decision", decision)
        return self._execute_local_intent_decision(intent, decision)

    def _execute_local_intent_decision(
        self,
        intent: NavigationIntent,
        decision: ExplorationDecision,
    ) -> MovementOutcome:
        """Execute one local primitive without changing strategic identity."""
        primitive = decision.local_primitive or (
            "rotate_scan"
            if decision.kind == ExplorationDecisionKind.ROTATE
            else "follow_edge"
        )
        updated = replace(intent, previous_primitive=primitive)
        self.drone.runtime_state.replace_navigation_intent(updated)
        invalid = self._validate_navigation_intent(updated)
        if invalid is not None:
            return invalid

        if updated.mode == MovementMode.SCAN:
            return self._execute_navigation_intent(updated)
        if updated.mode == MovementMode.RECOVERY:
            return self._execute_navigation_intent(updated)
        if primitive == "follow_edge":
            return self._execute_navigation_intent(updated)
        if primitive == "recovery":
            return self._start_recovery_intent(
                updated,
                TransitionReason.STALLED,
            )
        if primitive == "rotate_scan":
            if decision.direction is None:
                outcome = MovementOutcome(
                    transition_reason=TransitionReason.STALLED,
                )
            else:
                progress = self.drone.slam_map.progress_snapshot()
                updated = replace(
                    updated,
                    scan_sequence=progress.completed_scan_sequence,
                    scan_start_sensor_newly_known_cells=(
                        progress.sensor_newly_known_cells
                    ),
                    scan_start_sensor_confidence_gain=(
                        progress.sensor_confidence_gain
                    ),
                    local_scan_pending=True,
                )
                self.drone.runtime_state.replace_navigation_intent(updated)
                self.drone.runtime_state.begin_exploration(
                    decision.direction,
                )
                outcome = MovementOutcome(
                    arrived=True,
                    transition_reason=TransitionReason.SCAN_STARTED,
                )
            self.drone.runtime_state.record_movement_outcome(outcome)
            return outcome

        path = tuple(decision.planned_path)
        if not path or path[0] != self.drone.snapshot().position:
            path = (self.drone.snapshot().position, *path)
        distance = sum(
            math.dist(start, end) for start, end in zip(path, path[1:])
        )
        if distance > float(self.drone.step) + 1e-9:
            return self._invalidate_navigation_intent(
                updated,
                TransitionReason.INVALIDATED,
            )
        if decision.direction is not None:
            self.drone.runtime_state.begin_exploration(decision.direction)
        current = path[0]
        for node in path[1:]:
            if not self.drone.runtime_state.graph_is_valid(current, node):
                self.drone.slam_map.record_collision(node)
                outcome = MovementOutcome(
                    collision=True,
                    invalidated=True,
                    transition_reason=TransitionReason.COLLISION,
                )
                self._clear_navigation_intent(updated, outcome)
                if updated.cluster_id is not None:
                    self._release_frontier_assignment(updated.cluster_id)
                return outcome
            current = node
        followed = self._follow_policy_path(list(path))
        if not followed:
            outcome = MovementOutcome(
                transition_reason=TransitionReason.PAUSED,
            )
            self.drone.runtime_state.record_movement_outcome(outcome)
            return outcome
        outcome = MovementOutcome(
            travelled_distance=distance,
            transition_reason=TransitionReason.NONE,
        )
        self.drone.runtime_state.record_movement_outcome(outcome)
        recovery = self._apply_navigation_watchdog(outcome, updated)
        return recovery or outcome

    def _pose_deviates_from_route(self, intent: NavigationIntent) -> bool:
        """Return whether the live pose no longer matches the stored cursor."""
        if intent.mode not in {MovementMode.TRAVEL, MovementMode.HOME}:
            return False
        if intent.edge_cursor >= len(intent.route_paths):
            return False
        vertices = intent.route_paths[intent.edge_cursor]
        raster: list[Position] = []
        for start, goal in zip(vertices, vertices[1:]):
            leg = list(bresenham_path(start, goal))
            raster.extend(leg if not raster else leg[1:])
        if not raster:
            raster = list(vertices)
        if not raster:
            return False
        cursor = min(max(0, intent.polyline_cursor), len(raster) - 1)
        return math.dist(self.drone.snapshot().position, raster[cursor]) > 1.5

    def execute_exploration_action(
        self,
        decision: ExplorationDecision,
    ) -> MovementOutcome:
        """Execute one policy decision by moving or updating runtime state."""
        if decision.kind == ExplorationDecisionKind.HOMING:
            reached = self.reach_start_point()
            if reached:
                self.drone.runtime_state.mark_done()
                self._trace("drone_homing_done")
            return MovementOutcome(
                arrived=reached,
                transition_reason=(
                    TransitionReason.HOME if reached
                    else TransitionReason.STALLED
                ),
            )

        if decision.kind == ExplorationDecisionKind.FRONTIER:
            moved = self._reach_frontier_clusters(
                decision.frontier_cluster_ids or (
                    (decision.cluster_id,)
                    if decision.cluster_id is not None else ()
                )
            )
            outcome = self.drone.snapshot().last_movement_outcome
            return outcome if moved else MovementOutcome(
                transition_reason=(
                    TransitionReason.NO_ACTIONABLE_FRONTIER
                    if self._last_frontier_selection_status
                    == TransitionReason.NO_ACTIONABLE_FRONTIER.value
                    else TransitionReason.STALLED
                ),
            )

        return MovementOutcome(transition_reason=TransitionReason.STALLED)

    def reach_start_point(self) -> bool:
        """Route home through the same belief-only strategic graph machinery."""
        drone = self.drone
        snapshot = drone.snapshot()
        if snapshot.position == drone.start_pos:
            self._trace("drone_homing_already_home")
            return True
        intent = snapshot.navigation_intent
        if intent is not None and intent.mode == MovementMode.HOME:
            return self._execute_navigation_intent(intent).arrived
        graph = self.waypoint_graph
        if graph is None:
            return False
        slam = drone.slam_map.snapshot(point_limit=0)
        known_free = self._known_free_mask(slam)
        for position in (snapshot.position, drone.start_pos):
            x, y = position
            if 0 <= y < known_free.shape[0] and 0 <= x < known_free.shape[1]:
                known_free[y, x] = True
        route = self._find_route_with_live_tail(
            snapshot.position,
            drone.start_pos,
            known_free,
        )
        self._trace(
            "drone_homing_route",
            start=snapshot.position,
            goal=drone.start_pos,
            status=route.status,
            route_id=route.id,
            route_edge_ids=route.edge_ids,
            route_cache_hit=route.cache_hit,
            topology_revision=route.topology_revision,
            requester_knowledge_revision=route.requester_knowledge_revision,
            route=self._route_summary(route),
            replan_reason=TransitionReason.HOME.value,
            route_replanned=True,
            active_intent_valid=False,
        )
        if not route.found:
            return False
        intent = NavigationIntent(
            mode=MovementMode.HOME,
            route_id=route.id,
            target=drone.start_pos,
            topology_revision=route.topology_revision,
            requester_knowledge_revision=route.requester_knowledge_revision,
            route_node_ids=route.node_ids,
            route_edge_ids=route.edge_ids,
            route_paths=route.segment_paths or (route.first_segment_path,),
            route_sources=route.segment_sources or (
                route.first_segment_source or EDGE_KNOWN_FREE_CONNECTOR,
            ),
            route_segment_edge_ids=route.segment_edge_ids or (None,),
            remaining_route_cost=route.remaining_cost,
            selection_slam_version=slam.version,
        )
        intent = drone.runtime_state.set_navigation_intent(
            intent,
            reason=TransitionReason.HOME,
        )
        self._trace_navigation_transition(
            previous=None,
            current=intent,
            reason=TransitionReason.HOME,
            route_replanned=True,
            replan_reason=TransitionReason.HOME.value,
        )
        return self._execute_navigation_intent(intent, known_free=known_free).arrived

    def _reach_frontier_clusters(
        self,
        cluster_ids: tuple[int, ...] | list[int],
    ) -> bool:
        """Reserve and route toward authoritative stable frontier clusters."""
        drone = self.drone
        self._last_frontier_selection_status = None
        self._last_frontier_selection_details = {}
        active = drone.snapshot().navigation_intent
        if (
            active is not None
            and active.mode in {MovementMode.TRAVEL, MovementMode.SCAN}
            and active.target is not None
        ):
            outcome = self._execute_navigation_intent(active)
            return bool(outcome)
        requested_cluster_ids = tuple(dict.fromkeys(
            int(cluster_id) for cluster_id in cluster_ids
        ))
        cluster_ids = self._score_frontier_clusters(requested_cluster_ids)
        if requested_cluster_ids and not cluster_ids:
            accessible_before = tuple(sorted(self._frontier_waypoints))
            self.rebuild_frontiers()
            refreshed_cluster_ids = tuple(
                drone.snapshot().frontier_cluster_ids
            )
            cluster_ids = self._score_frontier_clusters(
                refreshed_cluster_ids
            )
            if not cluster_ids:
                accessible_after = tuple(sorted(self._frontier_waypoints))
                self._last_frontier_selection_status = (
                    TransitionReason.NO_ACTIONABLE_FRONTIER.value
                )
                self._last_frontier_selection_details = {
                    "requested_cluster_count": len(requested_cluster_ids),
                    "requested_cluster_ids": requested_cluster_ids,
                    "refreshed_cluster_count": len(refreshed_cluster_ids),
                    "refreshed_cluster_ids": refreshed_cluster_ids,
                    "accessible_before_refresh": accessible_before,
                    "accessible_after_refresh": accessible_after,
                }
                self._trace(
                    "drone_frontier_targets_exhausted",
                    reason=TransitionReason.NO_ACTIONABLE_FRONTIER.value,
                    frontier_count=0,
                    frontier_cluster_ids=(),
                    requested_cluster_ids=requested_cluster_ids,
                    refreshed_cluster_ids=refreshed_cluster_ids,
                    accessible_cluster_ids=accessible_after,
                    state=self._snapshot_summary(drone.snapshot()),
                )
                return False
        now = self._simulation_time()
        bridge_budget = [self.waypoint_bridge_attempt_limit]
        visible = {
            cluster.id: cluster
            for cluster in drone.frontier_registry.visible_to(drone.id)
        }
        for cluster_id in cluster_ids:
            cluster = visible.get(int(cluster_id))
            if cluster is None:
                continue
            target = self._frontier_waypoints.get(
                cluster.id, cluster.representative
            )
            current_position = drone.snapshot().position
            assignment = drone.frontier_assignments.reserve(
                cluster_id=cluster_id,
                drone_id=drone.id,
                gateway_id=cluster.gateway_id,
            )
            if assignment is None:
                self._trace(
                    "drone_frontier_skip",
                    target=target,
                    cluster_id=cluster_id,
                    reason="reserved",
                )
                continue
            self._frontier_assignment_tokens[cluster_id] = assignment.token
            self._trace(
                "frontier_assignment_reserved",
                assignment_id=assignment.token,
                assignment_token=assignment.token,
                cluster_id=cluster_id,
                gateway_id=assignment.gateway_id,
                expected_gain=cluster.expected_gain,
                wall_gain=cluster.wall_gain,
                wall_directed=bool(cluster.wall_gain),
                continuation_scan=bool(
                    self._retained_cluster_id == cluster_id
                    and cluster.wall_gain
                ),
                scan_heading_count=(
                    self.wall_continuation_scan_headings
                    if (
                        self._retained_cluster_id == cluster_id
                        and cluster.wall_gain
                    ) else 6
                ),
            )
            retry_at = self.border_retry_until.get(target, 0.0)
            if now < retry_at:
                self._trace(
                    "drone_frontier_skip",
                    target=target,
                    cluster_id=cluster_id,
                    reason="cooldown",
                    retry_in=retry_at - now,
                )
                self._release_frontier_assignment(cluster_id)
                continue
            if self.waypoint_graph is not None:
                slam_snapshot = drone.slam_map.snapshot(point_limit=0)
                known_free = self._known_free_mask(slam_snapshot)
                current_x, current_y = current_position
                if (
                    0 <= current_y < known_free.shape[0]
                    and 0 <= current_x < known_free.shape[1]
                ):
                    known_free[current_y, current_x] = True
                if self._advance_waypoint_segment(
                    current_position,
                    target,
                    known_free,
                    cluster_id=cluster_id,
                    bridge_budget=bridge_budget,
                ):
                    return True

            if not self._last_waypoint_route_transient_failure:
                self.border_retry_until[target] = (
                    now + self.border_retry_cooldown
                )
                self._blacklist_frontier(cluster_id, reason="unreachable")
            else:
                self._trace(
                    "drone_frontier_skip",
                    target=target,
                    cluster_id=cluster_id,
                    reason="route_attempt_budget_exhausted",
                )
            self._release_frontier_assignment(cluster_id)

        self._trace(
            "drone_frontier_targets_exhausted",
            frontier_count=len(cluster_ids),
            frontier_cluster_ids=cluster_ids,
            state=self._snapshot_summary(drone.snapshot()),
        )
        return False

    def _frontier_state_signature(
        self,
        reconciliation: Any | None = None,
    ) -> tuple[Any, ...]:
        """Identify changes that can make locally blocked frontier work viable."""
        snapshot = self.drone.snapshot()
        active_cluster_ids = getattr(
            reconciliation,
            "active_cluster_ids",
            None,
        )
        if active_cluster_ids is None:
            active_cluster_ids = tuple(sorted(
                cluster.id
                for cluster in self.drone.frontier_registry.canonical_clusters()
                if cluster.lifecycle != "retired" and cluster.known_by
            ))
        return (
            int(self.drone.slam_map.version),
            tuple(snapshot.frontier_cluster_ids),
            tuple(active_cluster_ids),
            bool(snapshot.returning_home),
        )

    def _frontier_wait_blocks(self, reconciliation: Any | None) -> bool:
        """Suppress unchanged retries while waking immediately on shared state."""
        signature = self._frontier_wait_signature
        if signature is None:
            return False
        now = self._simulation_time()
        current = self._frontier_state_signature(reconciliation)
        if current != signature or current[-1] or now >= self._frontier_wait_until:
            self._frontier_wait_signature = None
            self._frontier_wait_until = 0.0
            return False
        return True

    def _begin_frontier_wait(self) -> float:
        """Back off one locally blocked drone without changing team mission phase."""
        coordinator = getattr(self.drone, "exploration_coordinator", None)
        reconciliation = (
            coordinator.synchronize() if coordinator is not None else None
        )
        configured = float(getattr(
            self.drone.settings.frontier,
            "rebuild_cooldown",
            0.0,
        ))
        wait_seconds = max(
            configured,
            float(self.border_retry_cooldown),
        )
        self._frontier_wait_signature = self._frontier_state_signature(
            reconciliation
        )
        self._frontier_wait_until = self._simulation_time() + wait_seconds
        return wait_seconds

    def _wait_for_unactionable_frontiers(
        self,
        requested_cluster_ids: tuple[int, ...],
    ) -> None:
        """Publish the explicit local-wait transition and arm its backoff."""
        wait_seconds = self._begin_frontier_wait()
        self._trace(
            "drone_waiting_for_team_frontier",
            reason=TransitionReason.NO_ACTIONABLE_FRONTIER.value,
            requested_cluster_ids=tuple(requested_cluster_ids),
            retry_after_seconds=wait_seconds,
            selection=self._last_frontier_selection_details,
            state=self._snapshot_summary(self.drone.snapshot()),
        )

    @staticmethod
    def _scan_suppression_signature(cluster: Any) -> tuple[Any, ...]:
        """Identify the exact frontier geometry already scanned from nearby."""
        return (
            cluster.bounds,
            cluster.cells,
            cluster.wall_cells,
            int(cluster.expected_gain),
            int(cluster.wall_gain),
        )

    def _scan_suppression_applies(self, cluster: Any) -> bool:
        """Keep a tiny continuation dormant only while its shape is unchanged."""
        previous = self._scan_suppressed_clusters.get(cluster.id)
        if previous is None:
            return False
        current = self._scan_suppression_signature(cluster)
        if current == previous:
            return True
        self._scan_suppressed_clusters.pop(cluster.id, None)
        return False

    def _exploration_unknown_mask(self, slam: Any) -> np.ndarray:
        """Return cells not yet supported by confident local vision evidence."""
        return (
            (slam.occupancy == UNKNOWN)
            | (slam.confidence < self.frontier_confidence_threshold)
        )

    def _score_frontier_clusters(
        self,
        cluster_ids: tuple[int, ...] | list[int],
    ) -> tuple[int, ...]:
        """Apply the bounded deterministic global score to stable clusters."""
        requested = {int(cluster_id) for cluster_id in cluster_ids}
        visible = tuple(
            cluster for cluster in self.drone.frontier_registry.visible_to(
                self.drone.id
            )
            if cluster.id in requested
        )
        position = self.drone.snapshot().position
        candidates = []
        slam = self.drone.slam_map.snapshot(point_limit=0)
        known_free = self._known_free_mask(slam)
        unknown = self._exploration_unknown_mask(slam)
        x, y = position
        if 0 <= y < known_free.shape[0] and 0 <= x < known_free.shape[1]:
            known_free[y, x] = True
        self._frontier_waypoints = {}
        for cluster in visible:
            suppressed = self._scan_suppression_applies(cluster)
            minimum_distance = (
                self.wall_continuation_min_distance
                if cluster.id == self._retained_cluster_id and cluster.wall_gain
                else 0.0
            )
            waypoint = select_accessible_frontier_waypoint(
                cluster,
                known_free,
                origin=position,
                minimum_distance=minimum_distance,
                unknown=unknown,
            )
            if waypoint is not None and not suppressed:
                self._frontier_waypoints[cluster.id] = waypoint
            assignment = self.drone.frontier_assignments.assignment_for_cluster(
                cluster.id
            )
            candidates.append(StrategicCandidate(
                cluster_id=cluster.id,
                representative=cluster.representative,
                expected_gain=float(cluster.expected_gain),
                wall_gain=float(cluster.wall_gain),
                route_cost=(
                    math.dist(position, waypoint)
                    if waypoint is not None else math.inf
                ),
                waypoint=waypoint,
                revisit_penalty=cluster.revisit_penalty,
                stall_penalty=(
                    cluster.stall_penalty + cluster.zero_gain_penalty
                ),
                reserved_by_other=(
                    assignment is not None
                    and assignment.drone_id != self.drone.id
                ),
                reachable=waypoint is not None and not suppressed,
                blacklisted=(
                    suppressed or self._blacklist_applies(cluster, slam)
                ),
            ))
        preliminary = select_strategic_candidates(
            candidates,
            position=position,
        )
        if self.waypoint_graph is not None and preliminary:
            routed = []
            for item in preliminary:
                candidate = item.candidate
                route = self._find_route_with_live_tail(
                    position,
                    candidate.waypoint or candidate.representative,
                    known_free,
                )
                bridgeable = route.status in {
                    ROUTE_NO_GOAL_CONNECTOR,
                    ROUTE_DISCONNECTED,
                }
                routed.append(replace(
                    candidate,
                    route_cost=(
                        route.cost if route.found
                        else candidate.route_cost * 1.25
                    ),
                    reachable=route.found or bridgeable,
                ))
            preliminary = select_strategic_candidates(
                routed,
                position=position,
            )
        return tuple(
            item.candidate.cluster_id
            for item in preliminary
        )

    def _blacklist_frontier(
        self,
        cluster_id: int | None,
        *,
        reason: str,
    ) -> None:
        """Persist a belief-region-specific unreachable result."""
        if cluster_id is None:
            return
        cluster = next((
            item for item in self.drone.frontier_registry.visible_to(
                self.drone.id
            )
            if item.id == cluster_id
        ), None)
        if cluster is None:
            return
        slam = self.drone.slam_map.snapshot(point_limit=0)
        self._unreachable_blacklist[cluster_id] = (
            slam.version,
            self.waypoint_graph.topology_revision if self.waypoint_graph else 0,
            str(reason),
            cluster.bounds,
            self._belief_bounds_signature(slam, cluster.bounds),
        )
        self._trace(
            "frontier_cluster_blacklisted",
            cluster=self._cluster_summary(cluster),
            cluster_id=cluster.id,
            gateway_id=cluster.gateway_id,
            reason=str(reason),
            slam_version=slam.version,
            topology_revision=(
                self.waypoint_graph.topology_revision
                if self.waypoint_graph is not None else 0
            ),
        )

    def _blacklist_applies(self, cluster: Any, slam: Any) -> bool:
        record = self._unreachable_blacklist.get(cluster.id)
        if record is None:
            return False
        (
            _slam_revision,
            topology_revision,
            _reason,
            affected_bounds,
            signature,
        ) = record
        current_topology = (
            self.waypoint_graph.topology_revision if self.waypoint_graph else 0
        )
        return bool(
            topology_revision == current_topology
            and affected_bounds == cluster.bounds
            and signature == self._belief_bounds_signature(
                slam, affected_bounds
            )
        )

    @staticmethod
    def _belief_bounds_signature(
        slam: Any,
        bounds: tuple[int, int, int, int],
    ) -> bytes:
        """Fingerprint only belief cells that justified one blacklist entry."""
        x0, y0, x1, y1 = bounds
        occupancy = np.ascontiguousarray(slam.occupancy[y0:y1, x0:x1])
        confidence = np.ascontiguousarray(slam.confidence[y0:y1, x0:x1])
        return occupancy.tobytes() + confidence.tobytes()

    def _release_frontier_assignment(self, cluster_id: int | None) -> None:
        """Deterministically release one owned failed/finished reservation."""
        if cluster_id is None:
            return
        token = self._frontier_assignment_tokens.pop(cluster_id, None)
        if token is not None:
            assignment = self.drone.frontier_assignments.assignment_for_token(
                token
            )
            released = self.drone.frontier_assignments.release(
                token, drone_id=self.drone.id
            )
            self._trace(
                "frontier_assignment_released",
                assignment_id=token,
                assignment_token=token,
                cluster_id=cluster_id,
                gateway_id=(
                    None if assignment is None else assignment.gateway_id
                ),
                released=released,
            )

    def _advance_waypoint_segment(
        self,
        start: Position,
        target: Position,
        known_free: np.ndarray,
        *,
        cluster_id: int,
        bridge_budget: List[int] | None = None,
    ) -> bool:
        """Plan and traverse only the next segment toward a far frontier."""
        self._last_waypoint_route_transient_failure = False
        graph = self.waypoint_graph
        if graph is None:
            return False

        latched = self.drone.snapshot().navigation_intent
        if latched is not None and latched.target == target:
            outcome = self._execute_navigation_intent(
                latched,
                known_free=known_free,
            )
            return bool(
                outcome.travelled_distance > 0.0
                or outcome.arrived
                or outcome.scan_complete
            )

        # The current pose and uncommitted physical tail remain ephemeral.
        # Route lookup may use its requester-known-free connector, but must not
        # mutate topology merely because planning was requested.
        route_started = time.perf_counter()
        route_lookup_elapsed_ms = 0.0
        route_lookup_calls = 0

        def find_route() -> Any:
            nonlocal route_lookup_elapsed_ms, route_lookup_calls
            lookup_started = time.perf_counter()
            result = self._find_route_with_live_tail(
                start,
                target,
                known_free,
            )
            route_lookup_elapsed_ms += (
                time.perf_counter() - lookup_started
            ) * 1000.0
            route_lookup_calls += 1
            return result

        route = find_route()
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
                route = find_route()
        elif route.status in bridgeable_statuses:
            bridge_status = "attempt_budget_exhausted"
            self._last_waypoint_route_transient_failure = True
        gateway_status = "not_attempted"
        # Only a corridor that was required to repair connectivity may become
        # a protected gateway.  A route using an ephemeral goal connector does
        # not mutate or role-protect the graph.
        gateway_manager = getattr(self.drone, "frontier_gateway_manager", None)
        gateway_id = self.drone.frontier_registry.get(cluster_id).gateway_id
        if (
            gateway_manager is not None
            and bridge_update is not None
            and bridge_update.status == "ok"
        ):
            gateway_id = gateway_manager.ensure_gateway(
                cluster_id,
                known_free,
                requester_id=self.drone.id,
                position=target,
            )
            gateway_status = "ok" if gateway_id is not None else "orphan_rejected"
            if gateway_id is not None and not route.found:
                delta = gateway_manager.retire_gateway(cluster_id)
                self._trace_graph_delta(delta)
                gateway_id = None
                gateway_status = "retired_unusable"
        elif gateway_id is not None:
            gateway_status = "reused"
        if route.found and gateway_id is not None:
            token = self._frontier_assignment_tokens.get(cluster_id)
            if token is not None:
                self.drone.frontier_assignments.attach_gateway(
                    token,
                    drone_id=self.drone.id,
                    gateway_id=gateway_id,
                )
        route_elapsed_ms = (time.perf_counter() - route_started) * 1000.0
        route_repair_elapsed_ms = max(
            0.0,
            route_elapsed_ms - route_lookup_elapsed_ms,
        )
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
            route_lookup_elapsed_ms=route_lookup_elapsed_ms,
            route_lookup_calls=route_lookup_calls,
            route_repair_elapsed_ms=route_repair_elapsed_ms,
            route_id=route.id,
            route_cache_hit=route.cache_hit,
            topology_revision=route.topology_revision,
            requester_knowledge_revision=route.requester_knowledge_revision,
            route_node_ids=route.node_ids,
            route_edge_ids=route.edge_ids,
            remaining_route_cost=(
                route.remaining_cost
                if math.isfinite(route.remaining_cost) else None
            ),
            route=self._route_summary(route),
            replan_reason=self._replan_reason(),
            route_replanned=True,
            active_intent_valid=False,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )
        if not route.found:
            return False
        if next_waypoint is None:
            next_waypoint = target

        active = self.drone.snapshot().navigation_intent
        if active is None or active.target != target:
            gateway_id = None
            assignment_token = None
            cluster = self.drone.frontier_registry.get(cluster_id)
            gateway_id = cluster.gateway_id
            assignment_token = self._frontier_assignment_tokens.get(cluster_id)
            continuation_scan = bool(
                self._retained_cluster_id == cluster_id
                and cluster.wall_gain
            )
            segment_paths = route.segment_paths or (route.first_segment_path,)
            segment_sources = route.segment_sources or (
                route.first_segment_source or EDGE_KNOWN_FREE_CONNECTOR,
            )
            segment_edge_ids = route.segment_edge_ids or (None,)
            active = NavigationIntent(
                mode=MovementMode.TRAVEL,
                route_id=route.id,
                cluster_id=cluster_id,
                gateway_id=gateway_id,
                assignment_token=assignment_token,
                target=target,
                topology_revision=route.topology_revision,
                requester_knowledge_revision=route.requester_knowledge_revision,
                route_node_ids=route.node_ids,
                route_edge_ids=route.edge_ids,
                route_paths=tuple(segment_paths),
                route_sources=tuple(segment_sources),
                route_segment_edge_ids=tuple(segment_edge_ids),
                remaining_route_cost=route.remaining_cost,
                selection_slam_version=self.drone.slam_map.version,
                scan_heading_count=(
                    self.wall_continuation_scan_headings
                    if continuation_scan else 6
                ),
            )
            active = self.drone.runtime_state.set_navigation_intent(
                active,
                reason=TransitionReason.SELECTED,
            )
            self._retained_cluster_id = None
            self._trace_navigation_transition(
                previous=None,
                current=active,
                reason=TransitionReason.SELECTED,
                route_replanned=True,
                replan_reason=self._replan_reason(),
            )

        self._trace(
            "drone_waypoint_segment_path",
            start=start,
            target=target,
            segment_goal=next_waypoint,
            route_source=route.first_segment_source,
            path_source="persistent_route",
            astar_path_len=0,
            persistent_edge_astar_calls=0,
            connector_astar_calls=int(route.connector_astar_calls),
            path_len=len(route.first_segment_path),
            route_id=route.id,
            route_cache_hit=route.cache_hit,
            topology_revision=route.topology_revision,
            requester_knowledge_revision=route.requester_knowledge_revision,
        )
        outcome = self._execute_navigation_intent(active, known_free=known_free)
        current = self.drone.snapshot().position
        if outcome.invalidated:
            replacement = self.drone.snapshot().navigation_intent
            if (
                replacement is not None
                and replacement.mode == MovementMode.RECOVERY
            ):
                return True
            return False
        if current == start and not (outcome.arrived or outcome.scan_complete):
            return False

        self.border_retry_until.pop(target, None)
        self._trace(
            "drone_waypoint_segment_complete",
            target=target,
            segment_goal=next_waypoint,
            state=self._snapshot_summary(self.drone.snapshot()),
        )
        return True

    def _validate_navigation_intent(
        self,
        intent: NavigationIntent,
        *,
        known_free: np.ndarray | None = None,
    ) -> MovementOutcome | None:
        """Validate reservation and remaining route before any primitive."""
        if intent.cluster_id is not None:
            try:
                cluster = self.drone.frontier_registry.get(intent.cluster_id)
            except KeyError:
                cluster = None
            if cluster is not None and cluster.lifecycle == "retired":
                return self._invalidate_navigation_intent(
                    intent,
                    TransitionReason.GOAL_RETIRED,
                )
        if intent.assignment_token is not None:
            assignment = self.drone.frontier_assignments.assignment_for_token(
                intent.assignment_token
            )
            if assignment is None or assignment.drone_id != self.drone.id:
                return self._invalidate_navigation_intent(
                    intent,
                    TransitionReason.RESERVATION_LOST,
                )

        graph = self.waypoint_graph
        requires_route = intent.mode != MovementMode.SCAN or bool(
            intent.route_paths
        )
        if graph is None and requires_route:
            return self._invalidate_navigation_intent(
                intent,
                TransitionReason.INVALIDATED,
            )
        if graph is not None and graph.topology_revision != intent.topology_revision:
            active_edges = {edge.id for edge in graph.snapshot().edges}
            remaining_ids = {
                edge_id
                for edge_id in intent.route_segment_edge_ids[intent.edge_cursor:]
                if edge_id is not None
            }
            if not remaining_ids.issubset(active_edges):
                return self._invalidate_navigation_intent(
                    intent,
                    TransitionReason.ROUTE_EDGE_RETIRED,
                )

        belief_sources = {EDGE_SLAM_LOS, EDGE_KNOWN_FREE_CORRIDOR}
        if (
            self.drone.slam_map.version != intent.requester_knowledge_revision
            and any(
                source in belief_sources
                for source in intent.route_sources[intent.edge_cursor:]
            )
        ):
            if known_free is None:
                known_free = self._known_free_mask(
                    self.drone.slam_map.snapshot(point_limit=0)
                )
            for path, source in zip(
                intent.route_paths[intent.edge_cursor:],
                intent.route_sources[intent.edge_cursor:],
            ):
                if (
                    source in belief_sources
                    and not validate_known_free_path(path, known_free)
                ):
                    return self._invalidate_navigation_intent(
                        intent,
                        TransitionReason.BELIEF_CORRIDOR_INVALID,
                    )
        return None

    def _execute_navigation_intent(
        self,
        intent: NavigationIntent,
        *,
        known_free: np.ndarray | None = None,
    ) -> MovementOutcome:
        """Execute one bounded prefix of a latched exact route."""
        invalid = self._validate_navigation_intent(
            intent,
            known_free=known_free,
        )
        if invalid is not None:
            return invalid
        if intent.mode == MovementMode.SCAN:
            return self._execute_scan_intent(intent)
        if intent.mode not in {
            MovementMode.TRAVEL,
            MovementMode.HOME,
            MovementMode.RECOVERY,
        }:
            outcome = MovementOutcome(transition_reason=TransitionReason.NONE)
            self.drone.runtime_state.record_movement_outcome(outcome)
            return outcome

        (
            path,
            cursor_checkpoints,
            edge_cursor,
            polyline_cursor,
            distance,
        ) = self._route_prefix(
            intent, float(self.drone.step)
        )
        if len(path) <= 1 or distance <= 0.0:
            arrived = edge_cursor >= len(intent.route_paths)
            normalized_intent = intent.advanced(
                edge_cursor=edge_cursor,
                polyline_cursor=polyline_cursor,
                remaining_route_cost=(
                    0.0 if arrived else intent.remaining_route_cost
                ),
            )
            outcome = MovementOutcome(
                arrived=arrived,
                transition_reason=(
                    TransitionReason.REACHED if arrived
                    else TransitionReason.STALLED
                ),
            )
            if arrived and intent.mode == MovementMode.HOME:
                self._clear_navigation_intent(intent, outcome)
                self.drone.runtime_state.mark_done()
            elif arrived and intent.mode == MovementMode.RECOVERY:
                outcome = MovementOutcome(
                    arrived=True,
                    transition_reason=TransitionReason.RECOVERY_COMPLETE,
                )
                self._clear_navigation_intent(intent, outcome)
            elif arrived and intent.target is not None:
                if intent.cluster_id is not None:
                    return self._begin_scan(normalized_intent)
                self._clear_navigation_intent(intent, outcome)
            else:
                self.drone.runtime_state.record_movement_outcome(outcome)
            return outcome

        current = path[0]
        for node in path[1:]:
            if not self.drone.runtime_state.graph_is_valid(current, node):
                self.drone.slam_map.record_collision(node)
                outcome = MovementOutcome(
                    collision=True,
                    invalidated=True,
                    transition_reason=TransitionReason.COLLISION,
                )
                self._clear_navigation_intent(intent, outcome)
                if intent.cluster_id is not None:
                    self._release_frontier_assignment(intent.cluster_id)
                return outcome
            current = node
        followed, executed, processed_nodes = self._follow_path_execution(
            list(path),
            register_travelled=False,
        )
        self._ingest_ephemeral_route_motion(
            intent,
            path,
            cursor_checkpoints,
            processed_nodes,
        )
        if not followed:
            actual_distance = sum(
                math.dist(start, end)
                for start, end in zip(executed, executed[1:])
            )
            if processed_nodes > 0:
                checkpoint_index = min(
                    processed_nodes,
                    len(cursor_checkpoints),
                ) - 1
                partial_edge, partial_polyline = cursor_checkpoints[
                    checkpoint_index
                ]
                self.drone.runtime_state.advance_navigation_intent(
                    edge_cursor=partial_edge,
                    polyline_cursor=partial_polyline,
                    remaining_route_cost=max(
                        0.0,
                        intent.remaining_route_cost - actual_distance,
                    ),
                )
            outcome = MovementOutcome(
                travelled_distance=actual_distance,
                route_progress_delta=actual_distance,
                transition_reason=TransitionReason.PAUSED,
            )
            self.drone.runtime_state.record_movement_outcome(outcome)
            return outcome
        remaining = max(0.0, intent.remaining_route_cost - distance)
        advanced_intent = self.drone.runtime_state.advance_navigation_intent(
            edge_cursor=edge_cursor,
            polyline_cursor=polyline_cursor,
            remaining_route_cost=remaining,
        )
        arrived = (
            edge_cursor >= len(intent.route_paths)
            or self._frontier_was_reached(intent.target)
        )
        outcome = MovementOutcome(
            travelled_distance=distance,
            route_progress_delta=distance,
            arrived=arrived,
            transition_reason=(
                TransitionReason.REACHED if arrived
                else TransitionReason.PROGRESS
            ),
        )
        self.drone.runtime_state.record_movement_outcome(outcome)
        recovery = (
            self._apply_navigation_watchdog(
                outcome,
                advanced_intent or intent,
            )
            if intent.mode != MovementMode.RECOVERY and not arrived else None
        )
        if recovery is not None:
            return recovery
        if arrived and intent.mode == MovementMode.HOME:
            self._clear_navigation_intent(intent, outcome)
            self.drone.runtime_state.mark_done()
        elif arrived and intent.mode == MovementMode.RECOVERY:
            outcome = MovementOutcome(
                travelled_distance=distance,
                route_progress_delta=distance,
                arrived=True,
                transition_reason=TransitionReason.RECOVERY_COMPLETE,
            )
            self._clear_navigation_intent(intent, outcome)
        elif arrived and intent.target is not None:
            if intent.cluster_id is not None:
                return self._begin_scan(
                    advanced_intent or intent,
                    travelled_distance=distance,
                    route_progress_delta=distance,
                )
            self._clear_navigation_intent(intent, outcome)
        return outcome

    def _apply_navigation_watchdog(
        self,
        outcome: MovementOutcome,
        intent: NavigationIntent,
    ) -> MovementOutcome | None:
        """Update route progress/revisits and enter explicit recovery."""
        snapshot = self.drone.snapshot()
        cursor = min(intent.edge_cursor, len(intent.route_segment_edge_ids) - 1)
        edge_id = (
            intent.route_segment_edge_ids[cursor]
            if cursor >= 0 and intent.route_segment_edge_ids else None
        )
        position = snapshot.position
        visit = edge_id if edge_id is not None else (
            position[0] // 32,
            position[1] // 32,
        )
        now = self._simulation_time()
        watchdog = snapshot.navigation_watchdog.observe(
            outcome,
            now=now,
            visit=visit,
        )
        self.drone.runtime_state.update_navigation_watchdog(watchdog)
        reason = watchdog.recovery_reason(now=now)
        self._trace(
            "drone_watchdog",
            visit=visit,
            travelled_distance=outcome.travelled_distance,
            route_progress_delta=outcome.route_progress_delta,
            actual_information_gain=outcome.actual_information_gain,
            watchdog=self._watchdog_summary(watchdog),
            triggered_reason=(None if reason is None else reason.value),
        )
        if reason is None:
            return None
        return self._start_recovery_intent(intent, reason)

    def _start_recovery_intent(
        self,
        intent: NavigationIntent,
        reason: TransitionReason,
    ) -> MovementOutcome:
        """Latch a safe, previously travelled prefix for local recovery."""
        snapshot = self.drone.snapshot()
        if intent.cluster_id is not None:
            self.drone.frontier_registry.penalize(
                intent.cluster_id,
                revisit=(1.0 if reason == TransitionReason.REVERSAL else 0.0),
                stall=1.0,
            )
            self._release_frontier_assignment(intent.cluster_id)
        history = snapshot.path_history
        reverse_prefix = tuple(reversed(history[-max(2, int(self.drone.step) + 1):]))
        if len(reverse_prefix) < 2:
            return self._invalidate_navigation_intent(intent, reason)
        recovery_intent = NavigationIntent(
            mode=MovementMode.RECOVERY,
            route_id=0,
            target=reverse_prefix[-1],
            topology_revision=(
                self.waypoint_graph.topology_revision
                if self.waypoint_graph is not None else 0
            ),
            requester_knowledge_revision=self.drone.slam_map.version,
            route_paths=(reverse_prefix,),
            route_sources=("travelled",),
            route_segment_edge_ids=(None,),
            remaining_route_cost=sum(
                math.dist(start, end)
                for start, end in zip(reverse_prefix, reverse_prefix[1:])
            ),
            selection_slam_version=self.drone.slam_map.version,
            previous_primitive="recovery",
        )
        recovery_intent = self.drone.runtime_state.set_navigation_intent(
            recovery_intent,
            reason=reason,
        )
        self.drone.runtime_state.update_navigation_watchdog(
            NavigationWatchdog(last_progress_time=self._simulation_time())
        )
        self._trace_navigation_transition(
            previous=intent,
            current=recovery_intent,
            reason=reason,
            route_replanned=False,
            replan_reason=reason.value,
        )
        return MovementOutcome(
            invalidated=True,
            transition_reason=reason,
        )

    @staticmethod
    def _bounded_scan_heading_count(intent: NavigationIntent) -> int:
        """Return a defensive one-to-six heading count for an intent."""
        return max(1, min(6, int(intent.scan_heading_count)))

    def _wall_continuation_heading(
        self,
        intent: NavigationIntent,
    ) -> float | None:
        """Aim a retained scan at locally unknown cells beside its wall tip."""
        if intent.cluster_id is None:
            return None
        cluster = next((
            item for item in self.drone.frontier_registry.visible_to(
                self.drone.id
            )
            if item.id == intent.cluster_id
        ), None)
        if cluster is None or not cluster.wall_gain or not cluster.wall_cells:
            return None

        slam = self.drone.slam_map.snapshot(point_limit=0)
        unknown = self._exploration_unknown_mask(slam)
        occupied = (
            (slam.occupancy == OCCUPIED)
            & (slam.confidence >= self.frontier_confidence_threshold)
        )
        position = self.drone.snapshot().position
        anchors = sorted(
            cluster.wall_cells,
            key=lambda point: (
                math.dist(position, point),
                point[1],
                point[0],
            ),
        )
        nearest = math.dist(position, anchors[0])
        anchors = tuple(
            point for point in anchors
            if math.dist(position, point) <= nearest + 2.0
        )
        surface_vectors: list[tuple[float, float, float, int, int]] = []
        unknown_vectors: list[tuple[float, float, float, int, int]] = []
        height, width = unknown.shape
        for anchor_x, anchor_y in anchors:
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    if offset_x == 0 and offset_y == 0:
                        continue
                    x = anchor_x + offset_x
                    y = anchor_y + offset_y
                    if not (0 <= x < width and 0 <= y < height):
                        continue
                    if not unknown[y, x]:
                        continue
                    delta_x = float(x - position[0])
                    delta_y = float(y - position[1])
                    distance = math.hypot(delta_x, delta_y)
                    if distance <= 1e-9:
                        continue
                    vector = (1.0 / distance, delta_x, delta_y, y, x)
                    unknown_vectors.append(vector)
                    y0 = max(0, y - 1)
                    y1 = min(height, y + 2)
                    x0 = max(0, x - 1)
                    x1 = min(width, x + 2)
                    if np.any(occupied[y0:y1, x0:x1]):
                        surface_vectors.append(vector)

        vectors = surface_vectors or unknown_vectors
        if not vectors:
            return None
        delta_x = sum(weight * dx for weight, dx, _dy, _y, _x in vectors)
        delta_y = sum(weight * dy for weight, _dx, dy, _y, _x in vectors)
        if math.hypot(delta_x, delta_y) <= 1e-9:
            _weight, delta_x, delta_y, _y, _x = min(
                vectors,
                key=lambda item: (
                    math.hypot(item[1], item[2]),
                    item[3],
                    item[4],
                ),
            )
        return math.degrees(math.atan2(delta_x, -delta_y)) % 360.0

    def _scan_base_heading(self, intent: NavigationIntent) -> float:
        """Center a shortened continuation sweep on the unresolved wall."""
        current = float(self.drone.snapshot().heading_deg) % 360.0
        heading_count = self._bounded_scan_heading_count(intent)
        if heading_count >= 6:
            return current
        directed = self._wall_continuation_heading(intent)
        if directed is None:
            directed = current
        half_span = 30.0 * float(heading_count - 1)
        return (directed - half_span) % 360.0

    def _begin_scan(
        self,
        intent: NavigationIntent,
        *,
        travelled_distance: float = 0.0,
        route_progress_delta: float = 0.0,
    ) -> MovementOutcome:
        """Latch six-heading scanning while retaining the reservation."""
        if intent.cluster_id is not None:
            # En-route sensing often resolves a tiny frontier before the route
            # reaches its representative.  Refresh once at arrival and avoid a
            # six-heading scan when that cluster is absent from the fresh local
            # belief.  Hysteresis may keep it visible, so the per-observer
            # missing count is the authoritative signal here.
            selected_cluster = next((
                cluster
                for cluster in self.drone.frontier_registry.visible_to(
                    self.drone.id
                )
                if cluster.id == intent.cluster_id
            ), None)
            belief_advanced = bool(
                selected_cluster is not None
                and self.drone.slam_map.version
                > selected_cluster.last_seen_revision
            )
            if belief_advanced:
                self.rebuild_frontiers()
            refreshed = next((
                cluster
                for cluster in self.drone.frontier_registry.visible_to(
                    self.drone.id
                )
                if cluster.id == intent.cluster_id
            ), None)
            if belief_advanced and (
                refreshed is None or refreshed.missing_refresh_count > 0
            ):
                stale_cluster = (
                    None
                    if refreshed is None
                    else self._cluster_summary(refreshed)
                )
                coordinator = getattr(
                    self.drone, "exploration_coordinator", None
                )
                if coordinator is None:
                    self.drone.frontier_registry.retire(
                        intent.cluster_id,
                        reason="resolved_at_arrival",
                    )
                else:
                    coordinator.retire_cluster(
                        intent.cluster_id,
                        reason="resolved_at_arrival",
                    )
                self.drone.runtime_state.replace_frontier_clusters(
                    self.drone.frontier_registry.visible_to(self.drone.id)
                )
                outcome = MovementOutcome(
                    travelled_distance=travelled_distance,
                    route_progress_delta=route_progress_delta,
                    arrived=True,
                    scan_complete=True,
                    transition_reason=TransitionReason.SCAN_COMPLETE,
                )
                self._clear_navigation_intent(intent, outcome)
                self._release_frontier_assignment(intent.cluster_id)
                self._trace(
                    "frontier_cluster_retired",
                    cluster_id=intent.cluster_id,
                    gateway_id=intent.gateway_id,
                    assignment_id=intent.assignment_token,
                    intent_id=intent.intent_id,
                    route_id=intent.route_id,
                    reason="resolved_at_arrival",
                    actual_information_gain=0.0,
                    cluster=stale_cluster,
                )
                return outcome

        progress = self.drone.slam_map.progress_snapshot()
        scan_base_heading = self._scan_base_heading(intent)
        scan_intent = replace(
            intent,
            mode=MovementMode.SCAN,
            scan_heading_cursor=0,
            scan_base_heading=scan_base_heading,
            scan_sequence=progress.completed_scan_sequence,
            scan_start_sensor_newly_known_cells=(
                progress.sensor_newly_known_cells
            ),
            scan_start_sensor_confidence_gain=(
                progress.sensor_confidence_gain
            ),
            local_scan_pending=False,
        )
        scan_intent = self.drone.runtime_state.replace_navigation_intent(
            scan_intent,
            reason=TransitionReason.SCAN_STARTED,
        )
        self.drone.runtime_state.begin_exploration(
            int(round(scan_base_heading)) % 360
        )
        self._trace_navigation_transition(
            previous=intent,
            current=scan_intent,
            reason=TransitionReason.SCAN_STARTED,
            route_replanned=False,
            replan_reason=None,
        )
        self._trace(
            "drone_frontier_reached",
            target=intent.target,
            cluster_id=intent.cluster_id,
            gateway_id=intent.gateway_id,
            assignment_id=intent.assignment_token,
            intent_id=intent.intent_id,
            route_id=intent.route_id,
        )
        outcome = MovementOutcome(
            travelled_distance=travelled_distance,
            route_progress_delta=route_progress_delta,
            arrived=True,
            transition_reason=TransitionReason.SCAN_STARTED,
        )
        self.drone.runtime_state.record_movement_outcome(outcome)
        return outcome

    @staticmethod
    def _sensor_scan_gain(
        progress: Any,
        intent: NavigationIntent,
    ) -> tuple[int, float]:
        """Measure only evidence produced by this drone's vision sensor."""
        newly_known = max(
            0,
            int(progress.sensor_newly_known_cells)
            - int(intent.scan_start_sensor_newly_known_cells),
        )
        confidence = max(
            0.0,
            float(progress.sensor_confidence_gain)
            - float(intent.scan_start_sensor_confidence_gain),
        )
        return newly_known, confidence

    def _continuation_waypoints(
        self,
        cluster: Any,
        *,
        origin: Position,
    ) -> tuple[Position | None, Position | None]:
        """Choose nearest and batched wall viewpoints from one SLAM snapshot."""
        slam = self.drone.slam_map.snapshot(point_limit=0)
        known_free = self._known_free_mask(slam)
        x, y = origin
        if 0 <= y < known_free.shape[0] and 0 <= x < known_free.shape[1]:
            known_free[y, x] = True
        unknown = self._exploration_unknown_mask(slam)
        nearest = select_accessible_frontier_waypoint(
            cluster,
            known_free,
            origin=origin,
            unknown=unknown,
        )
        continuation = select_accessible_frontier_waypoint(
            cluster,
            known_free,
            origin=origin,
            minimum_distance=self.wall_continuation_min_distance,
            unknown=unknown,
        )
        return nearest, continuation

    def _complete_local_scan_if_ready(
        self,
        intent: NavigationIntent,
    ) -> MovementOutcome:
        """Gate a one-heading local rotation on a completed sensor sequence."""
        progress = self.drone.slam_map.progress_snapshot()
        if progress.completed_scan_sequence <= intent.scan_sequence:
            outcome = MovementOutcome(
                transition_reason=TransitionReason.SCAN_STARTED,
            )
            self.drone.runtime_state.record_movement_outcome(outcome)
            return outcome
        newly_known_gain, confidence_gain = self._sensor_scan_gain(
            progress,
            intent,
        )
        updated = replace(
            intent,
            scan_sequence=progress.completed_scan_sequence,
            scan_start_sensor_newly_known_cells=(
                progress.sensor_newly_known_cells
            ),
            scan_start_sensor_confidence_gain=(
                progress.sensor_confidence_gain
            ),
            local_scan_pending=False,
        )
        self.drone.runtime_state.replace_navigation_intent(updated)
        outcome = MovementOutcome(
            scan_complete=True,
            actual_information_gain=float(newly_known_gain) + confidence_gain,
            transition_reason=TransitionReason.SCAN_COMPLETE,
        )
        self.drone.runtime_state.record_movement_outcome(outcome)
        recovery = self._apply_navigation_watchdog(outcome, updated)
        return recovery or outcome

    def _execute_scan_intent(self, intent: NavigationIntent) -> MovementOutcome:
        """Rotate only after the sensor sequence advances for each heading."""
        progress = self.drone.slam_map.progress_snapshot()
        if progress.completed_scan_sequence <= intent.scan_sequence:
            outcome = MovementOutcome(
                transition_reason=TransitionReason.SCAN_STARTED,
            )
            self.drone.runtime_state.record_movement_outcome(outcome)
            return outcome
        next_heading = intent.scan_heading_cursor + 1
        if next_heading < self._bounded_scan_heading_count(intent):
            updated = replace(
                intent,
                scan_heading_cursor=next_heading,
                scan_sequence=progress.completed_scan_sequence,
            )
            self.drone.runtime_state.replace_navigation_intent(updated)
            self.drone.runtime_state.begin_exploration(int(round(
                intent.scan_base_heading + next_heading * 60
            )) % 360)
            newly_known_gain, confidence_gain = self._sensor_scan_gain(
                progress,
                intent,
            )
            outcome = MovementOutcome(
                actual_information_gain=(
                    float(newly_known_gain) + max(0.0, confidence_gain)
                ),
                transition_reason=TransitionReason.SCAN_STARTED,
            )
            self.drone.runtime_state.record_movement_outcome(outcome)
            return outcome

        newly_known_gain, confidence_gain = self._sensor_scan_gain(
            progress,
            intent,
        )
        gain = (
            float(newly_known_gain) + max(0.0, confidence_gain)
        )
        reason = (
            TransitionReason.ZERO_GAIN
            if gain <= 0.0 else TransitionReason.SCAN_COMPLETE
        )
        outcome = MovementOutcome(
            scan_complete=True,
            actual_information_gain=gain,
            transition_reason=reason,
        )
        self._clear_navigation_intent(intent, outcome)
        if intent.cluster_id is not None:
            if gain <= 0.0:
                self._scan_suppressed_clusters.pop(intent.cluster_id, None)
                retired_cluster = self.drone.frontier_registry.get(
                    intent.cluster_id
                )
                coordinator = getattr(
                    self.drone, "exploration_coordinator", None
                )
                if coordinator is None:
                    self.drone.frontier_registry.retire(
                        intent.cluster_id,
                        reason="zero_gain",
                    )
                else:
                    coordinator.retire_cluster(
                        intent.cluster_id,
                        reason="zero_gain",
                    )
                self._trace(
                    "frontier_cluster_retired",
                    cluster_id=intent.cluster_id,
                    gateway_id=intent.gateway_id,
                    assignment_id=intent.assignment_token,
                    intent_id=intent.intent_id,
                    route_id=intent.route_id,
                    reason="zero_gain",
                    actual_information_gain=gain,
                    cluster=(
                        None
                        if retired_cluster is None
                        else self._cluster_summary(retired_cluster)
                    ),
                )
                self.drone.runtime_state.replace_frontier_clusters(
                    self.drone.frontier_registry.visible_to(self.drone.id)
                )
                self._release_frontier_assignment(intent.cluster_id)
            else:
                self.rebuild_frontiers()
                continuing = next((
                    cluster
                    for cluster in self.drone.frontier_registry.visible_to(
                        self.drone.id
                    )
                    if cluster.id == intent.cluster_id
                ), None)
                if continuing is not None and continuing.wall_gain:
                    position = self.drone.snapshot().position
                    nearest, continuation = self._continuation_waypoints(
                        continuing,
                        origin=position,
                    )
                    if newly_known_gain > 0 and continuation is not None:
                        self._scan_suppressed_clusters.pop(
                            intent.cluster_id,
                            None,
                        )
                        self._retained_cluster_id = intent.cluster_id
                        self._trace(
                            "frontier_continuation_retained",
                            cluster_id=intent.cluster_id,
                            local_sensor_newly_known_cells=newly_known_gain,
                            local_sensor_confidence_gain=confidence_gain,
                            next_waypoint=continuation,
                            next_waypoint_distance=math.dist(
                                position,
                                continuation,
                            ),
                            minimum_distance=(
                                self.wall_continuation_min_distance
                            ),
                            scan_heading_count=(
                                self.wall_continuation_scan_headings
                            ),
                        )
                    else:
                        self._scan_suppressed_clusters[intent.cluster_id] = (
                            self._scan_suppression_signature(continuing)
                        )
                        self._release_frontier_assignment(intent.cluster_id)
                        self._trace(
                            "frontier_continuation_suppressed",
                            cluster_id=intent.cluster_id,
                            reason=(
                                "no_new_sensor_cells"
                                if newly_known_gain <= 0
                                else "continuation_too_close"
                            ),
                            local_sensor_newly_known_cells=newly_known_gain,
                            local_sensor_confidence_gain=confidence_gain,
                            nearest_waypoint=nearest,
                            nearest_waypoint_distance=(
                                None if nearest is None
                                else math.dist(position, nearest)
                            ),
                            minimum_distance=(
                                self.wall_continuation_min_distance
                            ),
                        )
                elif continuing is not None:
                    self._retained_cluster_id = intent.cluster_id
                else:
                    self._release_frontier_assignment(intent.cluster_id)
        return outcome

    def _invalidate_navigation_intent(
        self,
        intent: NavigationIntent,
        reason: TransitionReason,
    ) -> MovementOutcome:
        """Release an intent reservation and record why replanning is needed."""
        outcome = MovementOutcome(
            invalidated=True,
            transition_reason=reason,
        )
        self._clear_navigation_intent(intent, outcome)
        if intent.cluster_id is not None:
            self._release_frontier_assignment(intent.cluster_id)
        return outcome

    def _clear_navigation_intent(
        self,
        intent: NavigationIntent,
        outcome: MovementOutcome,
    ) -> None:
        """Clear one intent and emit its explicit terminal transition."""
        previous = self.drone.runtime_state.clear_navigation_intent(outcome)
        self._trace_navigation_transition(
            previous=previous or intent,
            current=None,
            reason=outcome.transition_reason,
            route_replanned=False,
            replan_reason=None,
        )

    def _route_prefix(
        self,
        intent: NavigationIntent,
        budget: float,
    ) -> tuple[
        tuple[Position, ...],
        tuple[tuple[int, int], ...],
        int,
        int,
        float,
    ]:
        """Take at most ``budget`` pixels along stored oriented polylines."""
        edge_cursor = intent.edge_cursor
        polyline_cursor = intent.polyline_cursor
        current = self.drone.snapshot().position
        selected = [current]
        cursor_checkpoints = [(edge_cursor, polyline_cursor)]
        travelled = 0.0
        while edge_cursor < len(intent.route_paths):
            vertices = intent.route_paths[edge_cursor]
            raster: list[Position] = []
            for start, goal in zip(vertices, vertices[1:]):
                leg = list(bresenham_path(start, goal))
                raster.extend(leg if not raster else leg[1:])
            if not raster:
                raster = list(vertices)
            if polyline_cursor >= len(raster) - 1:
                edge_cursor += 1
                polyline_cursor = 0
                cursor_checkpoints[-1] = (edge_cursor, polyline_cursor)
                continue
            for index in range(polyline_cursor + 1, len(raster)):
                step_distance = math.dist(selected[-1], raster[index])
                if travelled + step_distance > budget + 1e-9:
                    return (
                        tuple(selected), tuple(cursor_checkpoints), edge_cursor,
                        index - 1, travelled,
                    )
                selected.append(raster[index])
                travelled += step_distance
                polyline_cursor = index
                cursor_checkpoints.append((edge_cursor, polyline_cursor))
            edge_cursor += 1
            polyline_cursor = 0
            cursor_checkpoints[-1] = (edge_cursor, polyline_cursor)
        return (
            tuple(selected),
            tuple(cursor_checkpoints),
            edge_cursor,
            polyline_cursor,
            travelled,
        )

    def _frontier_was_reached(self, target: Position) -> bool:
        """Return whether movement ended at the selected frontier."""
        return self.drone.snapshot().position == target

    def update_borders(self) -> None:
        """Rebuild frontier targets when the cooldown permits."""
        self.maybe_rebuild_frontiers()

    def maybe_rebuild_frontiers(self) -> bool:
        """Rebuild frontiers if the configured cooldown has elapsed."""
        now = self._simulation_time()
        if not self.drone.runtime_state.reserve_frontier_rebuild(now):
            return False

        self.rebuild_frontiers()
        return True

    def rebuild_frontiers(self) -> None:
        """Refresh stable frontier clusters from the local SLAM belief."""
        drone = self.drone
        slam = drone.slam_map.snapshot(point_limit=0)
        extraction = drone.frontier_extractor.refresh(slam)
        coordinator = getattr(drone, "exploration_coordinator", None)
        if coordinator is None:
            clusters = drone.frontier_registry.refresh(
                drone.id,
                extraction.components,
                slam_version=slam.version,
            )
        else:
            clusters = coordinator.refresh_frontiers(
                drone.id,
                extraction.components,
                slam_version=slam.version,
            )
        frontiers = tuple(cluster.representative for cluster in clusters)
        drone.runtime_state.replace_frontier_clusters(clusters)
        self.border_retry_until = {}
        self._trace(
            "drone_frontiers_rebuilt",
            frontier_count=len(frontiers),
            frontier_sample=self._position_sample(frontiers),
            slam_version=slam.version,
            frontier_cluster_ids=tuple(cluster.id for cluster in clusters),
            clusters=tuple(self._cluster_summary(cluster) for cluster in clusters),
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

    def _build_exploration_context(
        self,
        *,
        snapshot: Any,
        slam_snapshot: Any | None = None,
    ) -> ExplorationContext:
        """Build detached policy inputs around the current pose estimate."""
        drone = self.drone
        pose_estimate = drone.localizer.estimate(
            snapshot,
            timestamp=self._simulation_time(),
        )
        return ExplorationContext(
            pose_estimate=pose_estimate,
            runtime_snapshot=snapshot,
            start_position=drone.start_pos,
            step=drone.step,
            radius=drone.radius,
            frontier_confidence_threshold=self.frontier_confidence_threshold,
            slam_snapshot=slam_snapshot,
        )

    def _waypoint_bridge_distance(self) -> float:
        """Return the bounded last-mile search radius for observed frontiers."""
        if self.waypoint_config is None:
            return 192.0
        return float(self.waypoint_config.gateway_connector_distance)

    def _known_free_mask(self, slam_snapshot: Any) -> np.ndarray:
        """Build the strict SLAM-only traversability mask used by highways."""
        return (
            (slam_snapshot.occupancy == FREE)
            & (
                slam_snapshot.confidence
                >= self.frontier_confidence_threshold
            )
        )

    def _find_route_with_live_tail(
        self,
        start: Position,
        target: Position,
        known_free: np.ndarray,
    ) -> Any:
        """Route through an exact uncommitted travelled tail when necessary.

        Recovery anchors intentionally remain 128 px apart while pose-to-graph
        connectors are bounded to 64 px.  Between those thresholds the live
        trail is the authoritative, physically executed connection back to the
        persistent graph; it must remain ephemeral, but it can safely be used
        as the first route segment.
        """
        graph = self.waypoint_graph
        if graph is None:
            raise RuntimeError("waypoint graph is required for route lookup")
        requester_revision = self.drone.slam_map.version
        route = graph.find_route(
            start,
            target,
            known_free,
            requester_id=self.drone.id,
            requester_knowledge_revision=requester_revision,
        )
        if route.status != ROUTE_NO_START_CONNECTOR:
            return route

        normalized_start = (int(start[0]), int(start[1]))
        live_tail = self._trail_accumulator.tail
        if len(live_tail) <= 1 or live_tail[-1] != normalized_start:
            return route
        anchor = live_tail[0]
        anchored = graph.find_route(
            anchor,
            target,
            known_free,
            requester_id=self.drone.id,
            requester_knowledge_revision=requester_revision,
        )
        if not anchored.found:
            return anchored

        entry_path = tuple(reversed(live_tail))
        entry_cost = sum(
            math.dist(first, second)
            for first, second in zip(entry_path, entry_path[1:])
        )
        anchored_waypoints = tuple(anchored.waypoints)
        waypoints = (normalized_start,) + anchored_waypoints
        return replace(
            anchored,
            waypoints=waypoints,
            first_segment_path=entry_path,
            first_segment_source=EDGE_TRAVELLED,
            first_segment_cost=entry_cost,
            cost=entry_cost + anchored.cost,
            entry_connector_path=entry_path,
            remaining_cost=entry_cost + anchored.remaining_cost,
            segment_paths=(entry_path,) + tuple(anchored.segment_paths),
            segment_sources=(EDGE_TRAVELLED,) + tuple(
                anchored.segment_sources
            ),
            segment_edge_ids=(None,) + tuple(anchored.segment_edge_ids),
        )

    def _ingest_ephemeral_route_motion(
        self,
        intent: NavigationIntent,
        path: tuple[Position, ...],
        cursor_checkpoints: tuple[tuple[int, int], ...],
        processed_nodes: int,
    ) -> None:
        """Feed only executed non-persistent connector steps to the trail."""
        limit = min(
            max(0, int(processed_nodes)),
            len(path),
            len(cursor_checkpoints),
        )
        connector_path: list[Position] = []

        def flush_connector_path() -> None:
            if len(connector_path) > 1:
                self._ingest_travelled_motion(connector_path)
            connector_path.clear()

        for index in range(1, limit):
            source_cursor = cursor_checkpoints[index - 1][0]
            if not (0 <= source_cursor < len(intent.route_sources)):
                flush_connector_path()
                continue
            if intent.route_sources[source_cursor] not in {
                EDGE_KNOWN_FREE_CONNECTOR,
                EDGE_SLAM_LOS,
            }:
                flush_connector_path()
                continue
            if not connector_path:
                connector_path.append(path[index - 1])
            elif connector_path[-1] != path[index - 1]:
                flush_connector_path()
                connector_path.append(path[index - 1])
            connector_path.append(path[index])
        flush_connector_path()

    def _simulation_time(self) -> float:
        """Return mission time with paused duration removed."""
        return self.dependencies.simulation_time()

    def _follow_path(
        self,
        path: List[Position],
        *,
        register_travelled: bool = True,
    ) -> bool:
        """Walk a path one node at a time, stopping cleanly on pause/shutdown."""
        completed, _executed, _processed_nodes = self._follow_path_execution(
            path,
            register_travelled=register_travelled,
        )
        return completed

    def _follow_path_execution(
        self,
        path: List[Position],
        *,
        register_travelled: bool = True,
    ) -> tuple[bool, tuple[Position, ...], int]:
        """Walk a path and return the exact prefix processed before a pause."""
        drone = self.drone
        executed = [drone.snapshot().position]
        processed_nodes = 0
        started_sim_time = self._simulation_time()
        completed = False
        try:
            for node in path:
                normalized = (int(node[0]), int(node[1]))
                if not self.dependencies.pause_checkpoint():
                    break

                drone.runtime_state.move_to(normalized)
                processed_nodes += 1
                if normalized != executed[-1]:
                    executed.append(normalized)

                if not self.dependencies.wait_simulation_delay(
                    drone.delay / drone.speed_factor
                ):
                    break
            else:
                completed = True
        finally:
            ended_sim_time = self._simulation_time()
            if register_travelled:
                self._ingest_travelled_motion(executed)
            self._trace_motion(
                executed,
                source="path",
                completed=completed,
                started_sim_time=started_sim_time,
                ended_sim_time=ended_sim_time,
            )
        return completed, tuple(executed), processed_nodes

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
            self._trace_graph_delta(graph.last_delta)

    def _ingest_travelled_motion(self, path: List[Position]) -> None:
        """Accumulate motion and persist only confirmed strategic sections."""
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

        slam_snapshot = self.drone.slam_map.snapshot(point_limit=0)
        known_free = self._known_free_mask(slam_snapshot)
        for section in self._trail_accumulator.append(normalized, known_free):
            update = graph.register_travelled_section(
                section.path,
                end_roles=section.end_roles,
            )
            self._trace_waypoint_update(update)

    def _trace_waypoint_update(self, update: Any) -> None:
        """Trace one complete stable-ID graph mutation and its legacy aliases."""
        self._trace_graph_delta(update.delta)

    def _trace_graph_delta(self, delta: Any) -> None:
        """Emit the canonical topology delta plus temporary Phase 6 aliases."""
        graph = self.waypoint_graph
        if graph is None or int(delta.revision) <= 0:
            return
        fields = graph_delta_trace_fields(graph, delta)
        self._trace(
            "waypoint_graph_delta",
            **fields,
        )

        for node_data in fields["added_nodes"]:
            self._trace(
                "waypoint_added",
                node_id=node_data["node_id"],
                waypoint=node_data["position"],
                source=node_data["source"],
                roles=node_data["roles"],
                topology_revision=int(delta.revision),
                node_count=graph.node_count,
            )
        for edge_data in fields["added_edges"]:
            self._trace(
                "waypoint_edge_added",
                edge_id=edge_data["edge_id"],
                start_node_id=edge_data["start_node_id"],
                end_node_id=edge_data["end_node_id"],
                start=edge_data["start"],
                end=edge_data["end"],
                source=edge_data["source"],
                distance=edge_data["cost"],
                topology_revision=int(delta.revision),
                edge_count=graph.edge_count,
            )

    def _trace_motion(
        self,
        executed: List[Position],
        *,
        source: str,
        completed: bool,
        started_sim_time: float,
        ended_sim_time: float,
    ) -> None:
        """Trace actual travelled distance for progress-efficiency analysis."""
        if len(executed) <= 1:
            return
        travelled_distance = sum(
            math.dist(start, end)
            for start, end in zip(executed, executed[1:])
        )
        if travelled_distance <= 0.0:
            return
        self._trace(
            "drone_motion",
            source=source,
            completed=completed,
            start=executed[0],
            end=executed[-1],
            point_count=len(executed),
            travelled_distance=travelled_distance,
            started_sim_time=started_sim_time,
            ended_sim_time=ended_sim_time,
        )

    def _trace(self, event: str, **fields: Any) -> None:
        """Write one movement trace event when runtime tracing is enabled."""
        trace = getattr(self.dependencies, "runtime_trace", None)
        if trace is None:
            return
        snapshot = self.drone.snapshot()
        intent = self._intent_summary(snapshot.navigation_intent)
        fields.setdefault("intent", intent)
        # Phase 6 keeps the legacy nested alias until a live trace validates
        # every replacement analyzer metric.
        fields.setdefault("navigation_intent", intent)
        fields.setdefault(
            "movement_mode",
            snapshot.movement_mode.value,
        )
        fields.setdefault(
            "watchdog",
            self._watchdog_summary(snapshot.navigation_watchdog),
        )
        trace.record(
            event,
            sim_time=self._simulation_time(),
            drone_id=self.drone.id,
            **fields,
        )

    def _trace_navigation_transition(
        self,
        *,
        previous: NavigationIntent | None,
        current: NavigationIntent | None,
        reason: TransitionReason,
        route_replanned: bool,
        replan_reason: str | None,
    ) -> None:
        """Emit one explicit, stable-ID navigation state transition."""
        previous_summary = self._intent_summary(previous)
        current_summary = self._intent_summary(current)
        self._trace(
            "drone_navigation_transition",
            intent=current_summary,
            navigation_intent=current_summary,
            previous_intent=previous_summary,
            mode_transition={
                "from_mode": None if previous is None else previous.mode.value,
                "to_mode": None if current is None else current.mode.value,
                "reason": reason.value,
            },
            goal_changed=(
                previous is None
                or current is None
                or previous.cluster_id != current.cluster_id
            ),
            replan_reason=replan_reason,
            route_replanned=bool(route_replanned),
            transition_reason=reason.value,
        )

    @staticmethod
    def _intent_summary(
        intent: NavigationIntent | None,
    ) -> dict[str, Any] | None:
        """Return the canonical compact intent telemetry object."""
        if intent is None:
            return None
        remaining = float(intent.remaining_route_cost)
        summary = {
            "intent_id": int(intent.intent_id),
            "mode": intent.mode.value,
            "goal_cluster_id": intent.cluster_id,
            "gateway_id": intent.gateway_id,
            "assignment_id": intent.assignment_token,
            "route_id": int(intent.route_id),
            "topology_revision": int(intent.topology_revision),
            "requester_knowledge_revision": int(
                intent.requester_knowledge_revision
            ),
            "selection_slam_version": int(intent.selection_slam_version),
            "route_node_ids": tuple(intent.route_node_ids),
            "route_edge_ids": tuple(intent.route_edge_ids),
            "edge_cursor": int(intent.edge_cursor),
            "polyline_cursor": int(intent.polyline_cursor),
            "remaining_route_cost": (
                remaining if math.isfinite(remaining) else None
            ),
        }
        if intent.mode == MovementMode.SCAN:
            summary.update({
                "scan_heading_cursor": int(intent.scan_heading_cursor),
                "scan_heading_count": int(intent.scan_heading_count),
                "scan_base_heading": float(intent.scan_base_heading),
            })
        return summary

    @staticmethod
    def _watchdog_summary(watchdog: Any) -> dict[str, Any]:
        """Return every input used by the deterministic watchdog."""
        return {
            "last_progress_time": float(watchdog.last_progress_time),
            "distance_without_progress": float(
                watchdog.distance_without_progress
            ),
            "recent_visits": tuple(watchdog.recent_visits),
            "reversal_count": int(watchdog.reversal_count),
            "revisit_ratio": float(watchdog.revisit_ratio),
        }

    @staticmethod
    def _outcome_summary(outcome: MovementOutcome) -> dict[str, Any]:
        """Return the canonical movement outcome telemetry object."""
        return {
            "travelled_distance": outcome.travelled_distance,
            "route_progress_delta": outcome.route_progress_delta,
            "arrived": outcome.arrived,
            "collision": outcome.collision,
            "scan_complete": outcome.scan_complete,
            "actual_information_gain": outcome.actual_information_gain,
            "invalidated": outcome.invalidated,
            "transition_reason": outcome.transition_reason.value,
        }

    @staticmethod
    def _route_summary(route: Any) -> dict[str, Any]:
        """Return the canonical stable-ID route telemetry object."""
        total = float(route.cost)
        remaining = float(route.remaining_cost)
        return {
            "route_id": int(route.id),
            "cache_hit": bool(route.cache_hit),
            "cache_eligible": bool(route.node_ids),
            "cache_key": (
                (
                    int(route.topology_revision),
                    int(route.node_ids[-1]),
                )
                if route.node_ids else None
            ),
            "topology_revision": int(route.topology_revision),
            "requester_knowledge_revision": int(
                route.requester_knowledge_revision
            ),
            "node_ids": tuple(route.node_ids),
            "edge_ids": tuple(route.edge_ids),
            "connector_astar_calls": int(route.connector_astar_calls),
            "total_cost": total if math.isfinite(total) else None,
            "remaining_cost": (
                remaining if math.isfinite(remaining) else None
            ),
        }

    @staticmethod
    def _cluster_summary(cluster: Any) -> dict[str, Any]:
        """Return stable cluster/gateway identity and lifecycle telemetry."""
        return {
            "cluster_id": int(cluster.id),
            "gateway_id": cluster.gateway_id,
            "representative": cluster.representative,
            "bounds": cluster.bounds,
            "expected_gain": cluster.expected_gain,
            "wall_gain": cluster.wall_gain,
            "wall_cell_count": len(cluster.wall_cells),
            "lifecycle": cluster.lifecycle,
            "first_seen_revision": cluster.first_seen_revision,
            "last_seen_revision": cluster.last_seen_revision,
            "missing_refresh_count": cluster.missing_refresh_count,
        }

    def _replan_reason(self) -> str:
        """Return the explicit prior transition authorizing a route plan."""
        reason = self.drone.snapshot().transition_reason
        if reason in {
            TransitionReason.NONE,
            TransitionReason.PROGRESS,
            TransitionReason.PAUSED,
        }:
            return TransitionReason.SELECTED.value
        return reason.value

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
            "goal_cluster_id": decision.cluster_id,
            "direction": decision.direction,
            "local_primitive": decision.local_primitive,
            "frontier_count": len(decision.frontier_targets),
            "frontier_cluster_ids": decision.frontier_cluster_ids,
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
            "performed": bool(getattr(diagnostics, "performed", False)),
            "iterations": diagnostics.iterations,
            "max_iterations": getattr(config, "iterations", None),
            "generated_nodes": diagnostics.generated_nodes,
            "selected_kind": diagnostics.selected_kind,
            "selected_direction": diagnostics.selected_direction,
            "selected_target": diagnostics.selected_target,
            "selected_reward": diagnostics.selected_reward,
            "slam_version": diagnostics.slam_version,
            "elapsed_ms": diagnostics.elapsed_ms,
            "root_coverage_complete": getattr(
                diagnostics,
                "root_coverage_complete",
                False,
            ),
            "overrun_stage": getattr(diagnostics, "overrun_stage", None),
            "safe_fallback": getattr(diagnostics, "safe_fallback", None),
            "budget_ms": getattr(diagnostics, "budget_ms", None),
            "search_budget_ms": getattr(
                diagnostics,
                "search_budget_ms",
                None,
            ),
            "reserved_budget_ms": getattr(
                diagnostics,
                "reserved_budget_ms",
                None,
            ),
            "window_bounds": getattr(diagnostics, "window_bounds", None),
            "preprocessing_cells": getattr(
                diagnostics,
                "preprocessing_cells",
                None,
            ),
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
        started_sim_time = self._simulation_time()
        completed = False
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
            completed = True
            return True
        finally:
            ended_sim_time = self._simulation_time()
            self._ingest_travelled_motion(executed)
            self._trace_motion(
                executed,
                source="policy_path",
                completed=completed,
                started_sim_time=started_sim_time,
                ended_sim_time=ended_sim_time,
            )
