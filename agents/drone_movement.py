"""Drone exploration, frontier selection, and homing behavior."""

import logging
import math
import random as rand
from typing import Any, List, Tuple

import numpy as np

from mapping.slam_map import FREE
from asset_config.helpers import next_cell_coords
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
        done, returning_home = (
            drone.runtime_state.evaluate_mission_state()
        )
        if done:
            return

        if returning_home:
            if self.reach_start_point():
                drone.runtime_state.mark_done()
            return

        node_found = False
        while not node_found:
            try:
                valid_dirs, valid_targets, chosen_target = self.find_new_node()
            except AssertionError:
                self.update_borders()
                node_found = self.reach_border()
                if not node_found:
                    return
            else:
                node_found = self.explore(
                    valid_dirs,
                    valid_targets,
                    chosen_target,
                )

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
        current_position = snapshot.position
        # The exploration model still samples every integer heading. Build the
        # usable headings directly so validity, direction, and frontier target
        # stay aligned without a separate blacklist pass.
        valid_dirs: List[int] = []
        valid_targets: List[Position] = []
        for direction in range(360):
            frontier_target = next_cell_coords(
                *current_position,
                drone.radius + 1,
                direction,
            )
            if drone.runtime_state.graph_is_valid(
                current_position,
                frontier_target,
            ):
                valid_dirs.append(direction)
                valid_targets.append(frontier_target)
        assert valid_dirs

        chosen_direction = rand.choice(valid_dirs)
        target = next_cell_coords(
            *current_position,
            drone.step,
            chosen_direction,
        )
        while not drone.runtime_state.graph_is_valid(
            current_position,
            target,
        ):
            rejected_index = valid_dirs.index(chosen_direction)
            valid_dirs.pop(rejected_index)
            valid_targets.pop(rejected_index)
            assert valid_dirs
            chosen_direction = rand.choice(valid_dirs)
            target = next_cell_coords(
                *current_position,
                drone.step,
                chosen_direction,
            )

        drone.runtime_state.set_direction(chosen_direction)
        return valid_dirs, valid_targets, target

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
        frontiers = sorted(
            snapshot.frontiers,
            key=lambda target: self._distance_from(
                snapshot.position,
                target,
            ),
        )

        if not frontiers:
            self.maybe_rebuild_frontiers()
            snapshot = drone.snapshot()
            frontiers = sorted(
                snapshot.frontiers,
                key=lambda target: self._distance_from(
                    snapshot.position,
                    target,
                ),
            )
            if not frontiers:
                return False

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
        occupancy = slam.occupancy
        slam_confidence = slam.confidence

        height, width = occupancy.shape
        cave = np.asarray(drone.cave)
        terrain_confidence = drone.terrain_knowledge.snapshot().confidence
        floor_mask = cave == 0

        # A frontier is a known free floor cell touching at least one still
        # unknown floor cell. That gives the drone a useful target at the edge
        # of its current knowledge.
        known_mask = (
            (slam_confidence >= confidence_threshold)
            | (terrain_confidence > 0.0)
        )
        free_known = floor_mask & known_mask & (occupancy == FREE)
        unknown = floor_mask & (~known_mask)

        neighbor_unknown = np.zeros_like(unknown, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ys_src = slice(max(0, -dy), height - max(0, dy))
                ys_dst = slice(max(0, dy), height - max(0, -dy))
                xs_src = slice(max(0, -dx), width - max(0, dx))
                xs_dst = slice(max(0, dx), width - max(0, -dx))
                neighbor_unknown[ys_dst, xs_dst] |= unknown[ys_src, xs_src]

        frontier_mask = free_known & neighbor_unknown
        stride = max(1, int(stride))
        if stride > 1:
            sampled = frontier_mask[::stride, ::stride]
            ys, xs = np.where(sampled)
            ys = ys * stride
            xs = xs * stride
        else:
            ys, xs = np.where(frontier_mask)

        frontiers = [
            (int(x), int(y))
            for y, x in zip(ys, xs)
        ]

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
        return self._distance_from(
            self.drone.snapshot().position,
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
