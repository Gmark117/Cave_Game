"""Terrain and SLAM sharing rules between mission agents."""

import math
import threading
import time
from typing import Any, Tuple

import numpy as np

from contracts import TerrainSharingDependencies
from mapping.slam_map import OCCUPIED


class TerrainSharingService:
    """Coordinate proximity-limited sharing between drones and rovers."""

    def __init__(self, dependencies: TerrainSharingDependencies) -> None:
        """Copy sharing thresholds and initialize cooldown bookkeeping."""
        self.dependencies = dependencies
        sharing = dependencies.sharing
        self.drone_share_interval = sharing.drone_interval
        self.pair_share_cooldown = sharing.pair_cooldown
        self.rover_share_interval = sharing.rover_interval
        self.compare_stride = sharing.compare_stride
        self.min_new_info_ratio = sharing.min_new_info_ratio
        self.min_overlap_diff_ratio = sharing.min_overlap_diff_ratio
        self.min_roughness_delta = sharing.min_roughness_delta
        self.last_drone_share: dict[int, float] = {}
        self.last_pair_share: dict[Tuple[int, int], float] = {}
        self.last_rover_share_time: float | None = None
        self._active_pairs: set[Tuple[int, int]] = set()
        self._physical_contact_pairs: set[Tuple[int, int]] = set()
        self._last_pair_versions: dict[
            Tuple[int, int], tuple[tuple[int, int], tuple[int, int]]
        ] = {}
        self._last_rover_versions: dict[
            Tuple[int, int], tuple[tuple[int, int], tuple[int, int]]
        ] = {}
        self._rover_suppressed_drones: set[int] = set()
        self._cooldown_lock = threading.Lock()
        self._rover_exchange_lock = threading.Lock()

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

    def _simulation_time(self) -> float:
        """Return pause-adjusted mission time for cooldown checks."""
        return self.dependencies.simulation_time()

    def _trace(self, event: str, **fields: Any) -> None:
        """Emit compact pair-level sharing evidence when tracing is enabled."""
        trace = getattr(self.dependencies, "runtime_trace", None)
        if trace is not None:
            trace.record(event, sim_time=self._simulation_time(), **fields)

    def has_line_of_sight(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Return True when segment a->b does not cross cave walls."""
        dependencies = self.dependencies
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

            if (
                y < 0
                or y >= dependencies.map_height
                or x < 0
                or x >= dependencies.map_width
            ):
                return False
            if dependencies.cave_map[y][x] != 0:
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
        dependencies = self.dependencies
        stride = self.compare_stride

        # Compare a strided subset to keep sharing checks cheap on large maps.
        # A share happens when the source has enough new cells or meaningfully
        # different roughness for cells both agents already know.
        src_conf = source_confidence[::stride, ::stride]
        tgt_conf = target_confidence[::stride, ::stride]
        src_rough = source_roughness[::stride, ::stride]
        tgt_rough = target_roughness[::stride, ::stride]
        floor = dependencies.terrain_knowledge.floor_mask[::stride, ::stride]

        src_known = floor & (src_conf > 0.0)
        if not np.any(src_known):
            return False

        tgt_known = floor & (tgt_conf > 0.0)
        src_known_count = int(np.count_nonzero(src_known))
        if src_known_count == 0:
            return False

        new_info = src_known & (~tgt_known)
        new_info_ratio = np.count_nonzero(new_info) / src_known_count
        if new_info_ratio >= self.min_new_info_ratio:
            return True

        overlap = src_known & tgt_known
        overlap_count = int(np.count_nonzero(overlap))
        if overlap_count == 0:
            return False

        overlap_delta = np.abs(src_rough - tgt_rough)
        meaningful_delta = overlap & (
            overlap_delta >= self.min_roughness_delta
        )
        overlap_diff_ratio = np.count_nonzero(meaningful_delta) / overlap_count
        return overlap_diff_ratio >= self.min_overlap_diff_ratio

    def slam_maps_differ_enough(
        self,
        source_occ: np.ndarray,
        source_conf: np.ndarray,
        target_occ: np.ndarray,
        target_conf: np.ndarray,
    ) -> bool:
        """Return whether the source contains any cell the merge can improve.

        SLAM is mission state, not optional terrain telemetry.  A strided ratio
        filter could permanently miss a small late frontier, so mirror the
        exact confidence-dominance and occupied-tie rules used by SlamMap.
        """
        height = min(source_occ.shape[0], target_occ.shape[0])
        width = min(source_occ.shape[1], target_occ.shape[1])
        if height <= 0 or width <= 0:
            return False
        src_occ = source_occ[:height, :width]
        src_conf = source_conf[:height, :width]
        tgt_occ = target_occ[:height, :width]
        tgt_conf = target_conf[:height, :width]
        higher_confidence = src_conf > tgt_conf
        occupied_ties = (
            (src_occ == OCCUPIED)
            & (tgt_occ != OCCUPIED)
            & (src_conf >= tgt_conf - 1e-4)
        )
        return bool(np.any(higher_confidence | occupied_ties))

    def share_with_nearby_drones(self, drone_id: int) -> None:
        """Check for nearby drones and exchange terrain and SLAM data."""
        dependencies = self.dependencies
        drones = dependencies.get_drones()
        drone = drones[drone_id]
        drone_snapshot = drone.snapshot()
        now = self._simulation_time()

        if self._at_any_rover(drone, drone_snapshot):
            self._clear_physical_contacts(drone_id)
            with self._cooldown_lock:
                first_suppression = drone_id not in self._rover_suppressed_drones
                self._rover_suppressed_drones.add(drone_id)
            if first_suppression:
                self._trace(
                    "drone_sharing_suppressed",
                    drone_id=int(drone_id),
                    reason="rover_arrival_or_standby",
                )
            return
        with self._cooldown_lock:
            self._rover_suppressed_drones.discard(drone_id)

        if not self._reserve_drone_schedule(drone_id, now):
            return

        for other_id, other_drone in enumerate(drones):
            if other_id == drone_id:
                continue

            other_snapshot = other_drone.snapshot()
            pair_key = (min(drone_id, other_id), max(drone_id, other_id))
            dx = (
                drone_snapshot.position[0]
                - other_snapshot.position[0]
            )
            dy = (
                drone_snapshot.position[1]
                - other_snapshot.position[1]
            )
            distance = math.sqrt(dx * dx + dy * dy)

            proximity_threshold = min(drone.radius, other_drone.radius)
            if distance >= 2 * proximity_threshold:
                self._set_physical_contact(pair_key, False)
                continue
            if self._at_any_rover(other_drone, other_snapshot):
                self._set_physical_contact(pair_key, False)
                continue
            # Agents need to be close and unobstructed; walls block data sharing.
            if not self.has_line_of_sight(
                drone_snapshot.position,
                other_snapshot.position,
            ):
                self._set_physical_contact(pair_key, False)
                self._trace(
                    "drone_sharing_pair",
                    drone_id=drone_id,
                    other_drone_id=other_id,
                    pair_key=pair_key,
                    distance=distance,
                    shared=False,
                    reason="no_line_of_sight",
                )
                continue

            self._set_physical_contact(pair_key, True)
            self._exchange_visible_pair(
                drone_id,
                other_id,
                drone,
                other_drone,
                drone_snapshot,
                other_snapshot,
                pair_key,
                distance,
                now,
            )

    def physical_contact_checkpoint(self, drone_id: int) -> None:
        """Exchange immediately when translation enters a peer contact envelope.

        Long path followers do not return to the ordinary periodic sharing pass
        between points.  Track contact edges here so a crossing is observed
        without repeating full-map comparisons for every path pixel.
        """
        dependencies = self.dependencies
        drones = dependencies.get_drones()
        if not 0 <= int(drone_id) < len(drones):
            return
        drone = drones[int(drone_id)]
        drone_snapshot = drone.snapshot()
        if self._at_any_rover(drone, drone_snapshot):
            self._clear_physical_contacts(drone_id)
            return
        now = self._simulation_time()
        for other_id, other_drone in enumerate(drones):
            if other_id == int(drone_id):
                continue
            other_snapshot = other_drone.snapshot()
            pair_key = (
                min(int(drone_id), other_id),
                max(int(drone_id), other_id),
            )
            distance = math.dist(
                drone_snapshot.position,
                other_snapshot.position,
            )
            proximity_threshold = min(drone.radius, other_drone.radius)
            visible_contact = bool(
                distance < 2 * proximity_threshold
                and not self._at_any_rover(other_drone, other_snapshot)
                and self.has_line_of_sight(
                    drone_snapshot.position,
                    other_snapshot.position,
                )
            )
            entered = self._set_physical_contact(pair_key, visible_contact)
            if not entered:
                continue
            self._exchange_visible_pair(
                int(drone_id),
                other_id,
                drone,
                other_drone,
                drone_snapshot,
                other_snapshot,
                pair_key,
                distance,
                now,
            )

    def _set_physical_contact(
        self,
        pair_key: Tuple[int, int],
        in_contact: bool,
    ) -> bool:
        """Record a contact edge and return whether it has just been entered."""
        with self._cooldown_lock:
            present = pair_key in self._physical_contact_pairs
            if not in_contact:
                self._physical_contact_pairs.discard(pair_key)
                return False
            self._physical_contact_pairs.add(pair_key)
            return not present

    def _clear_physical_contacts(self, drone_id: int) -> None:
        """Forget peer-contact edges involving one rover-adjacent drone."""
        normalized_id = int(drone_id)
        with self._cooldown_lock:
            self._physical_contact_pairs = {
                pair for pair in self._physical_contact_pairs
                if normalized_id not in pair
            }

    def _exchange_visible_pair(
        self,
        drone_id: int,
        other_id: int,
        drone: Any,
        other_drone: Any,
        drone_snapshot: Any,
        other_snapshot: Any,
        pair_key: Tuple[int, int],
        distance: float,
        now: float,
    ) -> None:
        """Run the existing ACK and bidirectional map exchange for one contact."""
        contact_callback = self.dependencies.on_drone_contact
        if callable(contact_callback):
            contact_callback(int(drone_id), int(other_id))

        if not self._reserve_pair(pair_key, now):
            self._trace(
                "drone_sharing_pair",
                drone_id=drone_id,
                other_drone_id=other_id,
                pair_key=pair_key,
                distance=distance,
                shared=False,
                reason="cooldown_or_active",
            )
            return

        shared = False
        exchange_started = time.perf_counter()
        try:
            versions = self._agent_pair_versions(drone, other_drone)
            with self._cooldown_lock:
                unchanged = self._last_pair_versions.get(pair_key) == versions
            if unchanged:
                self._trace(
                    "drone_sharing_pair",
                    drone_id=drone_id,
                    other_drone_id=other_id,
                    pair_key=pair_key,
                    distance=distance,
                    shared=False,
                    reason="unchanged_versions",
                    elapsed_ms=(
                        time.perf_counter() - exchange_started
                    ) * 1000.0,
                )
                return
            shared = self._exchange_drone_data(
                drone,
                other_drone,
                drone_snapshot,
                other_snapshot,
            )
            if shared:
                self.dependencies.presentation.terrain_heatmap_dirty = True
            self._trace(
                "drone_sharing_pair",
                drone_id=drone_id,
                other_drone_id=other_id,
                pair_key=pair_key,
                distance=distance,
                shared=bool(shared),
                reason=("exchanged" if shared else "no_delta"),
                elapsed_ms=(
                    time.perf_counter() - exchange_started
                ) * 1000.0,
                drone_slam_version=drone.slam_map.version,
                other_slam_version=other_drone.slam_map.version,
            )
        finally:
            final_versions = self._agent_pair_versions(drone, other_drone)
            with self._cooldown_lock:
                self._last_pair_versions[pair_key] = final_versions
            self._release_pair(pair_key, now, shared)

    @staticmethod
    def _agent_versions(agent: Any) -> tuple[int, int]:
        """Return cheap terrain and SLAM revisions for exchange gating."""
        terrain_version = int(getattr(agent.terrain_knowledge, "version", -1))
        slam_map = getattr(agent, "slam_map", None)
        slam_version = int(getattr(slam_map, "version", -1))
        return terrain_version, slam_version

    def _agent_pair_versions(
        self,
        first: Any,
        second: Any,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return ordered revisions for a sharing pair."""
        if int(first.id) <= int(second.id):
            return self._agent_versions(first), self._agent_versions(second)
        return self._agent_versions(second), self._agent_versions(first)

    def _at_any_rover(self, drone: Any, drone_snapshot: Any) -> bool:
        """Return whether a drone is inside any rover's exchange radius."""
        for rover in self.dependencies.get_rovers():
            if rover is None:
                continue
            rover_position = tuple(rover.pos)
            distance = math.dist(drone_snapshot.position, rover_position)
            if distance < min(drone.radius, rover.radius):
                return True
        return False

    def _exchange_drone_data(
        self,
        drone: Any,
        other_drone: Any,
        drone_snapshot: Any,
        other_snapshot: Any,
    ) -> bool:
        """Exchange meaningful terrain and SLAM data for one pair.

        Frontier coordinates are derived state.  Sharing their old coordinates
        used to mix borders extracted from different map versions; recipients
        now rebuild them from the merged local SLAM on their movement thread.
        """
        _ = drone_snapshot, other_snapshot
        drone_terrain = drone.terrain_knowledge.snapshot()
        other_terrain = other_drone.terrain_knowledge.snapshot()

        drone_slam = drone.slam_map.snapshot()
        other_slam = other_drone.slam_map.snapshot()

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
            drone_slam.occupancy,
            drone_slam.confidence,
            other_slam.occupancy,
            other_slam.confidence,
        )
        should_drone_receive_slam = self.slam_maps_differ_enough(
            other_slam.occupancy,
            other_slam.confidence,
            drone_slam.occupancy,
            drone_slam.confidence,
        )

        if not (
            should_other_receive
            or should_drone_receive
            or should_other_receive_slam
            or should_drone_receive_slam
        ):
            return False

        changed = False
        if should_other_receive:
            changed |= bool(
                other_drone.terrain_knowledge.merge_from(drone_terrain)
            )
        if should_drone_receive:
            changed |= bool(
                drone.terrain_knowledge.merge_from(other_terrain)
            )

        other_slam_changed = False
        drone_slam_changed = False
        if should_other_receive_slam:
            other_slam_changed = bool(
                other_drone.slam_map.merge_from(drone_slam)
            )
        if should_drone_receive_slam:
            drone_slam_changed = bool(
                drone.slam_map.merge_from(other_slam)
            )
        if other_slam_changed:
            self._notify_shared_slam_changed(other_drone)
        if drone_slam_changed:
            self._notify_shared_slam_changed(drone)

        self._trace(
            "drone_slam_exchange",
            drone_id=int(drone.id),
            other_drone_id=int(other_drone.id),
            drone_slam_changed=drone_slam_changed,
            other_slam_changed=other_slam_changed,
            raw_frontier_coordinates_shared=False,
        )
        return bool(changed or other_slam_changed or drone_slam_changed)

    @staticmethod
    def _notify_shared_slam_changed(drone: Any) -> None:
        """Invalidate derived navigation state without cross-thread rebuilds."""
        controller = getattr(drone, "movement_controller", None)
        callback = getattr(controller, "mark_shared_slam_changed", None)
        if callable(callback):
            callback()

    def share_with_rovers(self) -> None:
        """Exchange accumulated terrain and SLAM with nearby rovers."""
        dependencies = self.dependencies
        if not dependencies.periodic_rover_sharing_enabled:
            return
        now = self._simulation_time()
        if not self._reserve_rover_schedule(now):
            return

        for rover_id, rover in enumerate(dependencies.get_rovers()):
            if rover is None:
                continue

            for drone in dependencies.get_drones():
                self._share_if_at_rover(
                    drone,
                    rover,
                    rover_id=rover_id,
                    trace_event="drone_rover_proximity_share",
                )

    def check_in_with_rover(
        self,
        drone_id: int,
        rover_id: int = 0,
    ) -> bool:
        """Perform an uncached physical rendezvous with one rover.

        Returning ``True`` means the agents are in range with line of sight;
        it does not require either map to contain a new delta.  This distinction
        lets an already-synchronized drone complete the sector barrier.
        """
        drones = self.dependencies.get_drones()
        rovers = self.dependencies.get_rovers()
        if not (0 <= int(drone_id) < len(drones)):
            return False
        if not (0 <= int(rover_id) < len(rovers)):
            return False
        rover = rovers[int(rover_id)]
        if rover is None:
            return False
        return self._share_if_at_rover(
            drones[int(drone_id)],
            rover,
            rover_id=int(rover_id),
            trace_event="drone_rover_check_in",
        )

    def drone_at_rover(
        self,
        drone_id: int,
        rover_id: int = 0,
    ) -> bool:
        """Return whether a drone can currently rendezvous with a rover."""
        drones = self.dependencies.get_drones()
        rovers = self.dependencies.get_rovers()
        if not (0 <= int(drone_id) < len(drones)):
            return False
        if not (0 <= int(rover_id) < len(rovers)):
            return False
        rover = rovers[int(rover_id)]
        if rover is None:
            return False
        drone = drones[int(drone_id)]
        position = drone.snapshot().position
        distance = math.dist(tuple(rover.pos), position)
        if distance >= min(rover.radius, drone.radius):
            return False
        return self.has_line_of_sight(tuple(rover.pos), position)

    def share_on_departure(
        self,
        drone_id: int,
        rover_id: int = 0,
    ) -> bool:
        """Refresh one departing drone from the rover's team checkpoint."""
        drones = self.dependencies.get_drones()
        rovers = self.dependencies.get_rovers()
        if not (0 <= int(drone_id) < len(drones)):
            return False
        if not (0 <= int(rover_id) < len(rovers)):
            return False
        rover = rovers[int(rover_id)]
        if rover is None:
            return False
        return self._share_if_at_rover(
            drones[int(drone_id)],
            rover,
            rover_id=int(rover_id),
            trace_event="drone_rover_departure_share",
        )

    def _share_if_at_rover(
        self,
        drone: Any,
        rover: Any,
        *,
        rover_id: int,
        trace_event: str,
    ) -> bool:
        """Exchange both local stores when a drone physically reaches a rover."""
        drone_snapshot = drone.snapshot()
        dx = rover.pos[0] - drone_snapshot.position[0]
        dy = rover.pos[1] - drone_snapshot.position[1]
        distance = math.sqrt(dx * dx + dy * dy)
        proximity_threshold = min(rover.radius, drone.radius)
        if distance >= proximity_threshold:
            return False
        if not self.has_line_of_sight(rover.pos, drone_snapshot.position):
            self._trace(
                trace_event,
                drone_id=int(drone.id),
                rover_id=int(getattr(rover, "id", rover_id)),
                distance=distance,
                arrived=False,
                reason="no_line_of_sight",
            )
            return False

        contact_callback = self.dependencies.on_drone_rover_contact
        if callable(contact_callback):
            contact_callback(int(drone.id))

        pair_key = (int(drone.id), int(rover_id))
        exchange_started = time.perf_counter()
        with self._rover_exchange_lock:
            versions = (
                self._agent_versions(drone),
                self._agent_versions(rover),
            )
            if self._last_rover_versions.get(pair_key) == versions:
                changed = False
                reason = "unchanged_versions"
            else:
                changed = self._exchange_drone_rover_data(drone, rover)
                self._last_rover_versions[pair_key] = (
                    self._agent_versions(drone),
                    self._agent_versions(rover),
                )
                reason = "exchanged" if changed else "no_delta"
        self._trace(
            trace_event,
            drone_id=int(drone.id),
            rover_id=int(getattr(rover, "id", rover_id)),
            distance=distance,
            arrived=True,
            changed=changed,
            reason=reason,
            elapsed_ms=(time.perf_counter() - exchange_started) * 1000.0,
            map_comparison_skipped=(reason == "unchanged_versions"),
            drone_slam_version=getattr(
                getattr(drone, "slam_map", None),
                "version",
                None,
            ),
            rover_slam_version=getattr(
                getattr(rover, "slam_map", None),
                "version",
                None,
            ),
        )
        return True

    def visible_drone_positions(
        self,
        observer_id: int,
    ) -> tuple[tuple[int, tuple[int, int]], ...]:
        """Return peers directly observable under the radio contact rules."""
        drones = self.dependencies.get_drones()
        normalized = int(observer_id)
        if not 0 <= normalized < len(drones):
            return ()
        observer = drones[normalized]
        observer_position = tuple(observer.snapshot().position)
        visible = [(normalized, observer_position)]
        for peer_id, peer in enumerate(drones):
            if peer_id == normalized:
                continue
            peer_position = tuple(peer.snapshot().position)
            threshold = 2 * min(observer.radius, peer.radius)
            if math.dist(observer_position, peer_position) >= threshold:
                continue
            if not self.has_line_of_sight(observer_position, peer_position):
                continue
            visible.append((int(peer_id), peer_position))
        return tuple(visible)

    def _exchange_drone_rover_data(self, drone: Any, rover: Any) -> bool:
        """Upload drone knowledge and download the rover's team checkpoint."""
        drone_terrain = drone.terrain_knowledge.snapshot()
        rover_terrain = rover.terrain_knowledge.snapshot()
        rover_receives_terrain = self.maps_differ_enough(
            drone_terrain.roughness,
            drone_terrain.confidence,
            rover_terrain.roughness,
            rover_terrain.confidence,
        )
        drone_receives_terrain = self.maps_differ_enough(
            rover_terrain.roughness,
            rover_terrain.confidence,
            drone_terrain.roughness,
            drone_terrain.confidence,
        )

        changed = False
        if rover_receives_terrain:
            changed |= bool(rover.terrain_knowledge.merge_from(drone_terrain))
        if drone_receives_terrain:
            changed |= bool(drone.terrain_knowledge.merge_from(rover_terrain))
        if rover_receives_terrain or drone_receives_terrain:
            self.dependencies.presentation.terrain_heatmap_dirty = True

        drone_slam_map = getattr(drone, "slam_map", None)
        rover_slam_map = getattr(rover, "slam_map", None)
        if drone_slam_map is None or rover_slam_map is None:
            return changed

        drone_slam = drone_slam_map.snapshot()
        rover_slam = rover_slam_map.snapshot()
        rover_receives_slam = self.slam_maps_differ_enough(
            drone_slam.occupancy,
            drone_slam.confidence,
            rover_slam.occupancy,
            rover_slam.confidence,
        )
        drone_receives_slam = self.slam_maps_differ_enough(
            rover_slam.occupancy,
            rover_slam.confidence,
            drone_slam.occupancy,
            drone_slam.confidence,
        )
        if rover_receives_slam:
            changed |= bool(rover_slam_map.merge_from(drone_slam))
        drone_slam_changed = False
        if drone_receives_slam:
            drone_slam_changed = bool(drone_slam_map.merge_from(rover_slam))
            changed |= drone_slam_changed
        if drone_slam_changed:
            self._notify_shared_slam_changed(drone)
        return changed
