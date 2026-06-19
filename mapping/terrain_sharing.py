"""Terrain and SLAM sharing rules between mission agents."""

import math
import threading
import time
from typing import Any, Tuple

import numpy as np


class TerrainSharingService:
    """Coordinate proximity-limited sharing between drones and rovers."""

    def __init__(self, control: Any) -> None:
        self.control = control
        settings = getattr(control, "settings", None)
        self.drone_share_interval = max(
            0.0,
            float(getattr(settings, "drone_share_interval", 0.5)),
        )
        self.pair_share_cooldown = max(
            0.0,
            float(getattr(settings, "pair_share_cooldown", 1.2)),
        )
        self.rover_share_interval = max(
            0.0,
            float(getattr(settings, "rover_share_interval", 0.5)),
        )
        self.last_drone_share: dict[int, float] = {}
        self.last_pair_share: dict[Tuple[int, int], float] = {}
        self.last_rover_share_time: float | None = None
        self._active_pairs: set[Tuple[int, int]] = set()
        self._cooldown_lock = threading.Lock()

    def _reserve_drone_schedule(self, drone_id: int, now: float) -> bool:
        """Atomically reserve one drone's periodic sharing pass."""
        with self._cooldown_lock:
            last_share = self.last_drone_share.get(drone_id, 0.0)
            if (now - last_share) < self.drone_share_interval:
                return False
            self.last_drone_share[drone_id] = now
            return True

    def _reserve_pair(
        self,
        pair_key: Tuple[int, int],
        now: float,
    ) -> bool:
        """Atomically reserve a pair that is off cooldown and not in flight."""
        with self._cooldown_lock:
            if pair_key in self._active_pairs:
                return False
            last_share = self.last_pair_share.get(pair_key, 0.0)
            if (now - last_share) < self.pair_share_cooldown:
                return False
            self._active_pairs.add(pair_key)
            return True

    def _release_pair(
        self,
        pair_key: Tuple[int, int],
        now: float,
        shared: bool,
    ) -> None:
        """Release an in-flight pair and record successful exchange time."""
        with self._cooldown_lock:
            self._active_pairs.discard(pair_key)
            if shared:
                self.last_pair_share[pair_key] = now

    def _reserve_rover_schedule(self, now: float) -> bool:
        """Atomically reserve the periodic drone-to-rover sharing pass."""
        with self._cooldown_lock:
            if (
                self.last_rover_share_time is not None
                and (now - self.last_rover_share_time)
                < self.rover_share_interval
            ):
                return False
            self.last_rover_share_time = now
            return True

    def has_line_of_sight(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Return True when segment a->b does not cross cave walls."""
        control = self.control
        x0, y0 = int(a[0]), int(a[1])
        x1, y1 = int(b[0]), int(b[1])

        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return True

        for i in range(steps + 1):
            t = i / steps
            x = int(round(x0 + dx * t))
            y = int(round(y0 + dy * t))

            if y < 0 or y >= control.map_h or x < 0 or x >= control.map_w:
                return False
            if control.map_matrix[y][x] != 0:
                return False

        return True

    def maps_differ_enough(
        self,
        source_roughness: np.ndarray,
        source_confidence: np.ndarray,
        target_roughness: np.ndarray,
        target_confidence: np.ndarray,
    ) -> bool:
        """Return True when sharing is likely to add meaningful terrain info."""
        control = self.control
        stride = max(1, int(control.share_compare_stride))

        src_conf = source_confidence[::stride, ::stride]
        tgt_conf = target_confidence[::stride, ::stride]
        src_rough = source_roughness[::stride, ::stride]
        tgt_rough = target_roughness[::stride, ::stride]
        floor = control.floor_mask[::stride, ::stride]

        src_known = floor & (src_conf > 0.0)
        if not np.any(src_known):
            return False

        tgt_known = floor & (tgt_conf > 0.0)
        src_known_count = int(np.count_nonzero(src_known))
        if src_known_count == 0:
            return False

        new_info = src_known & (~tgt_known)
        new_info_ratio = np.count_nonzero(new_info) / src_known_count
        if new_info_ratio >= control.min_share_new_info_ratio:
            return True

        overlap = src_known & tgt_known
        overlap_count = int(np.count_nonzero(overlap))
        if overlap_count == 0:
            return False

        overlap_delta = np.abs(src_rough - tgt_rough)
        meaningful_delta = overlap & (
            overlap_delta >= control.min_share_roughness_delta
        )
        overlap_diff_ratio = np.count_nonzero(meaningful_delta) / overlap_count
        return overlap_diff_ratio >= control.min_share_overlap_diff_ratio

    def slam_maps_differ_enough(
        self,
        source_occ: np.ndarray,
        source_conf: np.ndarray,
        target_occ: np.ndarray,
        target_conf: np.ndarray,
    ) -> bool:
        """Return True when SLAM occupancy sharing adds meaningful info."""
        control = self.control
        stride = max(1, int(control.share_compare_stride))

        src_conf = source_conf[::stride, ::stride]
        tgt_conf = target_conf[::stride, ::stride]
        src_occ = source_occ[::stride, ::stride]
        tgt_occ = target_occ[::stride, ::stride]

        src_known = src_conf > 0.0
        if not np.any(src_known):
            return False

        tgt_known = tgt_conf > 0.0
        src_known_count = int(np.count_nonzero(src_known))
        if src_known_count == 0:
            return False

        new_info = src_known & (~tgt_known)
        new_info_ratio = np.count_nonzero(new_info) / src_known_count
        if new_info_ratio >= control.min_share_new_info_ratio:
            return True

        overlap = src_known & tgt_known
        overlap_count = int(np.count_nonzero(overlap))
        if overlap_count == 0:
            return False

        diff = src_occ != tgt_occ
        overlap_diff_ratio = np.count_nonzero(diff & overlap) / overlap_count
        return overlap_diff_ratio >= control.min_share_overlap_diff_ratio

    def share_with_nearby_drones(self, drone_id: int) -> None:
        """Check for nearby drones and exchange terrain and SLAM data."""
        control = self.control
        drone = control.drones[drone_id]
        now = time.perf_counter()

        if not self._reserve_drone_schedule(drone_id, now):
            return

        for other_id, other_drone in enumerate(control.drones):
            if other_id == drone_id:
                continue

            pair_key = (min(drone_id, other_id), max(drone_id, other_id))
            dx = drone.pos[0] - other_drone.pos[0]
            dy = drone.pos[1] - other_drone.pos[1]
            distance = math.sqrt(dx * dx + dy * dy)

            proximity_threshold = min(drone.radius, other_drone.radius)
            if distance >= 2 * proximity_threshold:
                continue
            if not self.has_line_of_sight(drone.pos, other_drone.pos):
                continue

            if not self._reserve_pair(pair_key, now):
                continue

            shared = False
            try:
                shared = self._exchange_drone_data(drone, other_drone)
                if shared:
                    control.presentation.terrain_heatmap_dirty = True
            finally:
                self._release_pair(pair_key, now, shared)

    def _exchange_drone_data(self, drone: Any, other_drone: Any) -> bool:
        """Exchange meaningful terrain, frontier, and SLAM data for one pair."""
        drone_terrain = drone.terrain_knowledge.snapshot()
        with drone.exploration_lock:
            drone_border = list(drone.border)
        other_terrain = other_drone.terrain_knowledge.snapshot()
        with other_drone.exploration_lock:
            other_border = list(other_drone.border)

        with drone.slam_lock:
            drone_occ = drone.slam_map.occupancy.copy()
            drone_conf = drone.slam_map.confidence.copy()
            drone_points = list(drone.slam_map.point_cloud)
        with other_drone.slam_lock:
            other_occ = other_drone.slam_map.occupancy.copy()
            other_conf = other_drone.slam_map.confidence.copy()
            other_points = list(other_drone.slam_map.point_cloud)

        should_other_receive = self.maps_differ_enough(
            drone_terrain.roughness,
            drone_terrain.confidence,
            other_terrain.roughness,
            other_terrain.confidence,
        )
        should_drone_receive = self.maps_differ_enough(
            other_terrain.roughness,
            other_terrain.confidence,
            drone_terrain.roughness,
            drone_terrain.confidence,
        )
        should_other_receive_slam = self.slam_maps_differ_enough(
            drone_occ,
            drone_conf,
            other_occ,
            other_conf,
        )
        should_drone_receive_slam = self.slam_maps_differ_enough(
            other_occ,
            other_conf,
            drone_occ,
            drone_conf,
        )

        if not (
            should_other_receive
            or should_drone_receive
            or should_other_receive_slam
            or should_drone_receive_slam
        ):
            return False

        if should_other_receive:
            other_drone.terrain_knowledge.merge_from(drone_terrain)
            other_drone.merge_exploration_data(None, drone_border)
        if should_drone_receive:
            drone.terrain_knowledge.merge_from(other_terrain)
            drone.merge_exploration_data(None, other_border)

        if should_other_receive_slam:
            with other_drone.slam_lock:
                other_drone.slam_map.merge_from_arrays(
                    drone_occ, drone_conf, drone_points
                )
        if should_drone_receive_slam:
            with drone.slam_lock:
                drone.slam_map.merge_from_arrays(
                    other_occ, other_conf, other_points
                )

        return True

    def share_with_rovers(self) -> None:
        """Share terrain knowledge from drones to nearby rovers."""
        control = self.control
        now = time.perf_counter()
        if not self._reserve_rover_schedule(now):
            return

        for rover in control.rovers:
            if rover is None:
                continue

            for drone in control.drones:
                dx = rover.pos[0] - drone.pos[0]
                dy = rover.pos[1] - drone.pos[1]
                distance = math.sqrt(dx * dx + dy * dy)

                proximity_threshold = min(rover.radius, drone.radius)
                if distance >= proximity_threshold:
                    continue
                if not self.has_line_of_sight(rover.pos, drone.pos):
                    continue

                drone_terrain = drone.terrain_knowledge.snapshot()
                rover_terrain = rover.terrain_knowledge.snapshot()
                should_rover_receive = self.maps_differ_enough(
                    drone_terrain.roughness,
                    drone_terrain.confidence,
                    rover_terrain.roughness,
                    rover_terrain.confidence,
                )
                if not should_rover_receive:
                    continue

                if rover.terrain_knowledge.merge_from(drone_terrain):
                    control.presentation.terrain_heatmap_dirty = True
