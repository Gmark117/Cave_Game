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

        node_found = False
        while not node_found:
            decision = self.choose_exploration_action()
            if decision.kind == ExplorationDecisionKind.EXHAUSTED:
                self.update_borders()
                decision = self.choose_exploration_action()

            node_found = self.execute_exploration_action(decision)
            if decision.kind != ExplorationDecisionKind.STEP:
                return

    def choose_exploration_action(self) -> ExplorationDecision:
        """Ask the exploration policy for the next high-level action."""
        drone = self.drone
        snapshot = drone.snapshot()
        context = self._build_exploration_context(snapshot=snapshot)
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
            return True

        if decision.kind == ExplorationDecisionKind.FRONTIER:
            return self._reach_frontier_targets(
                decision.frontier_targets or (
                    (decision.target,) if decision.target is not None else ()
                )
            )

        if decision.kind == ExplorationDecisionKind.STEP:
            return self._execute_step_decision(decision)

        return False

    def reach_start_point(self) -> bool:
        """Follow an A* path back to the drone's starting position."""
        drone = self.drone
        snapshot = drone.snapshot()
        if snapshot.position == drone.start_pos:
            return True

        path = self._compute_path(snapshot.position, drone.start_pos)
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
        return self.explore(
            list(decision.valid_directions),
            list(decision.frontier_targets),
            decision.target,
        )

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
                continue
            retry_at = self.border_retry_until.get(target, 0.0)
            if now < retry_at:
                continue

            path = self._compute_path(current_position, target)
            if not path or len(path) <= 1:
                self.border_retry_until[target] = (
                    now + self.border_retry_cooldown
                )
                continue

            self._follow_path(path)
            drone.runtime_state.remove_frontier(target)
            self.border_retry_until.pop(target, None)
            return True

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

        drone.runtime_state.replace_frontiers(frontiers)
        self.border_retry_until = {}

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
