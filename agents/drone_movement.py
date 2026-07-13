"""Drone exploration, frontier selection, and homing behavior."""

import logging
import math
from typing import Any, List, Tuple

import numpy as np

from agents.exploration_policy import (
    ExplorationContext,
    ExplorationDecision,
    ExplorationDecisionKind,
)
from contracts import DroneMovementDependencies


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
        """Follow an A* path to the first reachable frontier target."""
        drone = self.drone
        now = self._simulation_time()
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

            path = self._compute_path(current_position, target)
            self._trace(
                "drone_frontier_path",
                start=current_position,
                target=target,
                path_len=len(path),
            )
            if not path or len(path) <= 1:
                self.border_retry_until[target] = (
                    now + self.border_retry_cooldown
                )
                continue

            self._follow_path(path)
            drone.runtime_state.remove_frontier(target)
            self.border_retry_until.pop(target, None)
            self._trace(
                "drone_frontier_reached",
                target=target,
                state=self._snapshot_summary(drone.snapshot()),
            )
            return True

        self._trace(
            "drone_frontier_targets_exhausted",
            frontier_count=len(frontiers),
            state=self._snapshot_summary(drone.snapshot()),
        )
        return False

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

    def _simulation_time(self) -> float:
        """Return mission time with paused duration removed."""
        return self.dependencies.simulation_time()

    def _follow_path(self, path: List[Position]) -> bool:
        """Walk a path one node at a time, stopping cleanly on pause/shutdown."""
        drone = self.drone
        for node in path:
            if not self.dependencies.pause_checkpoint():
                return False

            drone.runtime_state.move_to(node)

            if not self.dependencies.wait_simulation_delay(
                drone.delay / drone.speed_factor
            ):
                return False
        return True

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
        for node in path:
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

            if not self.dependencies.wait_simulation_delay(
                drone.delay / drone.speed_factor
            ):
                return False
        return True
