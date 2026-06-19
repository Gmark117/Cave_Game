"""Drone exploration, frontier selection, and homing behavior."""

import math
import random as rand
import time
from typing import Any, List, Tuple

import numpy as np

from SlamMap import FREE
from asset_config.helpers import next_cell_coords


Position = Tuple[int, int]


class DroneMovementController:
    """Drive one drone's exploration state machine."""

    def __init__(self, drone: Any) -> None:
        self.drone = drone
        settings = drone.settings

        self.border_retry_cooldown = 1.5
        self.border_retry_until: dict[Position, float] = {}
        self.frontier_rebuild_cooldown = float(
            getattr(settings, "frontier_rebuild_cooldown", 0.25)
        )
        self.last_frontier_rebuild = 0.0
        self.frontier_stride = int(getattr(settings, "frontier_stride", 4))
        self.frontier_confidence_threshold = float(
            getattr(settings, "frontier_confidence_threshold", 0.6)
        )

    def move(self) -> None:
        """Advance the drone's exploration or homing state."""
        drone = self.drone
        if drone.done:
            return

        if drone.returning_home or (drone.explored and not drone.border):
            drone.returning_home = True
            if self.reach_start_point():
                drone.done = True
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
        if drone.pos == drone.start_pos:
            return True

        path = self._compute_path(drone.pos, drone.start_pos)
        if not path:
            return False

        self._follow_path(path)
        return drone.pos == drone.start_pos

    def find_new_node(
        self,
    ) -> Tuple[List[int], List[Position], Position]:
        """Choose a locally valid exploration step.

        Raises `AssertionError` when no valid direction remains.
        """
        drone = self.drone
        directions = 360
        all_dirs = list(range(directions))
        targets = [[0, 0] for _ in all_dirs]
        dir_res = int(360 / len(all_dirs))

        dir_blacklist = []
        for direction in all_dirs:
            targets[direction][0], targets[direction][1] = next_cell_coords(
                *drone.pos,
                drone.radius + 1,
                direction * dir_res,
            )
            if not drone.graph.is_valid(
                drone.pos,
                (*targets[direction],),
            ):
                dir_blacklist.append(direction)

        valid_dirs = [
            direction
            for direction in all_dirs
            if direction not in dir_blacklist
        ]
        valid_targets = [
            (*targets[valid_direction],)
            for valid_direction in valid_dirs
        ]
        assert valid_dirs

        drone.dir = rand.choice(valid_dirs)
        target = next_cell_coords(*drone.pos, drone.step, drone.dir)
        while not drone.graph.is_valid(drone.pos, target):
            valid_dirs.remove(drone.dir)
            valid_targets.remove((*targets[drone.dir],))
            assert valid_dirs
            drone.dir = rand.choice(valid_dirs)
            target = next_cell_coords(*drone.pos, drone.step, drone.dir)

        return valid_dirs, valid_targets, target

    def explore(
        self,
        valid_dirs: List[int],
        valid_targets: List[Position],
        chosen_target: Position,
    ) -> bool:
        """Attempt exploration toward `chosen_target`."""
        drone = self.drone
        drone.explored = True
        drone.dir_log.append(drone.dir)
        drone.border.extend(valid_targets)
        drone.border = list(set(drone.border))
        valid_dirs.remove(drone.dir)

        path = self._compute_path(drone.pos, chosen_target)
        if not path:
            return False

        self._follow_path(path)
        return True

    def reach_border(self) -> bool:
        """Follow an A* path to the nearest viable frontier."""
        drone = self.drone
        drone.border.sort(key=self.get_distance)

        if not drone.border:
            self.maybe_rebuild_frontiers()
            if not drone.border:
                return False

        now = time.perf_counter()
        for target in list(drone.border):
            if target == drone.pos:
                continue
            retry_at = self.border_retry_until.get(target, 0.0)
            if now < retry_at:
                continue

            path = self._compute_path(drone.pos, target)
            if not path or len(path) <= 1:
                self.border_retry_until[target] = (
                    now + self.border_retry_cooldown
                )
                continue

            self._follow_path(path)
            if target in drone.border:
                drone.border.remove(target)
            self.border_retry_until.pop(target, None)
            return True

        return False

    def update_borders(self) -> None:
        """Rebuild frontier targets when the cooldown permits."""
        self.maybe_rebuild_frontiers()

    def maybe_rebuild_frontiers(self) -> bool:
        """Rebuild frontiers if the configured cooldown has elapsed."""
        now = time.perf_counter()
        if (
            now - self.last_frontier_rebuild
        ) < self.frontier_rebuild_cooldown:
            return False

        self.last_frontier_rebuild = now
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
        with drone.slam_lock:
            occupancy = drone.slam_map.occupancy.copy()
            slam_confidence = drone.slam_map.confidence.copy()

        height, width = occupancy.shape
        cave = np.asarray(drone.cave)
        terrain_confidence = drone.terrain_knowledge.snapshot().confidence
        floor_mask = cave == 0

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

        with drone.exploration_lock:
            drone.border = frontiers
            self.border_retry_until = {}

    def mission_completed(self) -> bool:
        """Return True once exploration is exhausted and the drone is home."""
        drone = self.drone
        if not drone.explored:
            return False

        if not drone.border and not drone.done:
            drone.returning_home = True
            return False

        if drone.done:
            print(f"Drone {drone.id} has completed the mission!")
            return True

        return False

    def get_distance(self, target: Position) -> float:
        """Return distance to a frontier, deprioritizing already-visible cells."""
        drone = self.drone
        distance = math.dist(drone.pos, target)
        if distance <= drone.radius:
            return float(drone.game.width)
        return distance

    def update_heading(self, previous: Position, current: Position) -> None:
        """Update the drone heading from two consecutive positions."""
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        if dx == 0 and dy == 0:
            return
        drone = self.drone
        drone.heading_deg = math.degrees(math.atan2(dx, -dy))

    def _compute_path(
        self,
        start: Position,
        goal: Position,
    ) -> List[Position]:
        control = self.drone.control
        if not hasattr(control, "compute_path"):
            return []
        return control.compute_path(start, goal)

    def _follow_path(self, path: List[Position]) -> None:
        drone = self.drone
        for node in path:
            previous = drone.pos
            drone.pos = node
            self.update_heading(previous, drone.pos)
            drone.graph.add_node(node)
            if hasattr(drone.control, "mission_event"):
                drone.control.mission_event.wait(
                    drone.delay / drone.speed_factor
                )
            else:
                time.sleep(drone.delay / drone.speed_factor)
