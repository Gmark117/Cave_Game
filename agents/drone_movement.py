"""Random drone exploration with A* escape and homing routes."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np

from asset_config.helpers import next_cell_coords
from contracts import DroneMovementDependencies
from mapping.ray_geometry import bresenham_line_points
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN
from navigation.astar_pathfinder import (
    PATH_COMPLETE,
    PATH_PARTIAL_LIMIT,
    PATH_UNREACHABLE,
    PathResult,
)


Position = Tuple[int, int]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PendingFrontierScan:
    """One wall-facing sensor pose that temporarily blocks translation."""

    position: Position
    heading: int
    frontier_target: Position
    unknown_target: Position
    reason: str
    minimum_scan_sequence: int
    baseline_geometry: tuple[Position, ...] | None


@dataclass(frozen=True)
class _PendingFrontierRoute:
    """A capped A* frontier route waiting for its next segment."""

    target: Position
    recovery_reason: str


@dataclass(frozen=True)
class _FrontierCluster:
    """One connected local unknown boundary used for heading selection."""

    cells: tuple[Position, ...]
    size: int
    touches_wall: bool
    distance: float
    continuation_alignment: float


@dataclass(frozen=True)
class _ClusterSelection:
    """Selected component and normalized terms used to rank it."""

    cluster: _FrontierCluster | None
    support: dict[int, float]
    score: float
    size_rank: float
    proximity: float
    wall_candidate_count: int
    generic_candidate_count: int


@dataclass(frozen=True)
class _GlobalFrontierTile:
    """Frontier evidence accumulated inside one coarse SLAM cell."""

    target: Position
    size: int
    wall_cells: int

    @property
    def touches_wall(self) -> bool:
        """Return whether this cell contains a wall continuation."""
        return self.wall_cells > 0


@dataclass(frozen=True)
class _GlobalFrontierRegion:
    """One connected set of occupied coarse frontier cells."""

    tiles: tuple[_GlobalFrontierTile, ...]
    size: int
    wall_cells: int
    tile_count: int

    @property
    def touches_wall(self) -> bool:
        """Return whether any unknown boundary cell continues a known wall."""
        return self.wall_cells > 0


@dataclass(frozen=True)
class _GlobalFrontierCache:
    """Coarse whole-map regions and one stable strategic selection."""

    regions: tuple[_GlobalFrontierRegion, ...]
    target: _GlobalFrontierRegion | None
    target_position: Position | None
    target_score: float
    target_size_rank: float
    target_proximity: float
    eligible_region_count: int
    filtered_region_count: int
    wall_candidate_count: int
    generic_candidate_count: int
    slam_version: int
    built_at: float


@dataclass(frozen=True)
class _GlobalFrontierGuidance:
    """Directional evidence from the cached whole-map target."""

    support: dict[int, float]
    active: bool
    target: Position | None
    size: int
    distance: float | None
    bearing: float | None
    touches_wall: bool
    score: float
    size_rank: float
    proximity: float
    region_count: int
    eligible_region_count: int
    filtered_region_count: int
    wall_candidate_count: int
    generic_candidate_count: int
    slam_version: int


@dataclass(frozen=True)
class _HeadingBias:
    """Weighted-random evidence attached to valid movement headings."""

    weights: dict[int, float]
    wall_support: dict[int, float]
    frontier_support: dict[int, float]
    global_support: dict[int, float]
    separation_support: dict[int, float]
    mode: str
    peer_count: int
    cluster_count: int
    eligible_cluster_count: int
    filtered_cluster_count: int
    selected_cluster_size: int
    selected_cluster_distance: float | None
    selected_cluster_touches_wall: bool
    selected_continuation_alignment: float
    selected_cluster_score: float
    selected_cluster_size_rank: float
    selected_cluster_proximity: float
    wall_candidate_count: int
    generic_candidate_count: int
    global_active: bool
    global_target: Position | None
    global_region_size: int
    global_region_distance: float | None
    global_region_bearing: float | None
    global_region_touches_wall: bool
    global_region_score: float
    global_region_size_rank: float
    global_region_proximity: float
    global_region_count: int
    global_eligible_region_count: int
    global_filtered_region_count: int
    global_wall_candidate_count: int
    global_generic_candidate_count: int
    global_slam_version: int


class DroneMovementController:
    """Explore locally at random and use A* only for escape or homing."""

    def __init__(
        self,
        drone: Any,
        dependencies: DroneMovementDependencies,
    ) -> None:
        self.drone = drone
        self.dependencies = dependencies
        frontier = drone.settings.frontier
        self.border_retry_cooldown = 1.5
        self.border_retry_until: dict[Position, float] = {}
        self.frontier_stride = max(1, int(getattr(frontier, "stride", 4)))
        self.frontier_confidence_threshold = float(
            frontier.confidence_threshold
        )
        self.minimum_frontier_cluster_cells = max(
            1,
            int(getattr(frontier, "minimum_cluster_cells", 12)),
        )
        self.frontier_distance_band = max(
            1.0,
            float(getattr(frontier, "distance_band", 16.0)),
        )
        self.wall_continuation_weight = max(
            0.0,
            float(getattr(frontier, "wall_continuation_weight", 2.0)),
        )
        self.frontier_cluster_size_weight = max(
            0.0,
            float(getattr(frontier, "cluster_size_weight", 2.0)),
        )
        self.frontier_cluster_proximity_weight = max(
            0.0,
            float(getattr(frontier, "cluster_proximity_weight", 1.0)),
        )
        self.global_frontier_cell_size = max(
            1,
            int(getattr(frontier, "global_cell_size", 32)),
        )
        self.global_frontier_refresh_interval = max(
            0.0,
            float(getattr(frontier, "global_refresh_interval", 2.0)),
        )
        self._last_raw_frontiers: frozenset[Position] = frozenset()
        self._suppressed_frontier_geometry: dict[
            Position,
            tuple[Position, ...] | None,
        ] = {}
        exploration = drone.settings.exploration
        self.stagnation_distance = float(
            exploration.stagnation_distance
        )
        self.stagnation_min_sensor_cells_per_px = float(
            exploration.stagnation_min_sensor_cells_per_px
        )
        self.wall_direction_bias = float(
            exploration.wall_direction_bias
        )
        self.unexplored_direction_bias = float(
            exploration.unexplored_direction_bias
        )
        self.separation_direction_bias = float(
            exploration.separation_direction_bias
        )
        progress = drone.slam_map.progress_snapshot()
        self._stagnation_sensor_baseline = (
            progress.sensor_newly_known_cells
        )
        self._stagnation_distance_travelled = 0.0
        self._pending_frontier_scan: _PendingFrontierScan | None = None
        self._pending_frontier_route: _PendingFrontierRoute | None = None
        self._partial_route_endpoints: dict[
            Position,
            tuple[Position, ...],
        ] = {}
        self._global_frontier_cache: _GlobalFrontierCache | None = None
        self._shared_slam_changed = threading.Event()
        self._frontier_slam_version = -1

    def mark_shared_slam_changed(self) -> None:
        """Request a frontier refresh on the owning movement thread."""
        self._shared_slam_changed.set()

    def _refresh_frontiers_before_mission_state(self) -> None:
        """Apply shared or late SLAM changes before exhaustion can start home."""
        drone = self.drone
        state = drone.snapshot()
        shared_change = self._shared_slam_changed.is_set()
        if shared_change:
            self._shared_slam_changed.clear()
            self._global_frontier_cache = None
        if state.done or state.returning_home:
            return

        slam_version = drone.slam_map.version
        empty_changed_map = (
            state.explored
            and not state.frontiers
            and slam_version != self._frontier_slam_version
        )
        if not shared_change and not empty_changed_map:
            return

        self.rebuild_frontiers(
            stride=self.frontier_stride,
            confidence_threshold=self.frontier_confidence_threshold,
        )
        self._trace(
            "drone_slam_frontiers_refreshed",
            reason=(
                "shared_slam_changed"
                if shared_change
                else "empty_frontiers_on_changed_slam"
            ),
            previous_slam_version=slam_version,
            rebuilt_slam_version=self._frontier_slam_version,
            frontier_count=len(drone.snapshot().frontiers),
        )

    def move(self) -> None:
        """Advance one random step, escape route, or homing route."""
        drone = self.drone
        self._refresh_frontiers_before_mission_state()
        if self._pending_frontier_scan is not None:
            self._advance_pending_frontier_scan()
            return
        done, returning_home = drone.runtime_state.evaluate_mission_state()
        if done:
            return
        if returning_home:
            self._pending_frontier_route = None
            if self.reach_start_point():
                drone.runtime_state.mark_done()
            return

        pending_route = self._pending_frontier_route
        if pending_route is not None:
            self.maybe_rebuild_frontiers()
            target_active = pending_route.target in drone.snapshot().frontiers
            if target_active:
                if self.reach_border(
                    recovery_reason=pending_route.recovery_reason,
                    preferred_target=pending_route.target,
                ):
                    return
            else:
                self._clear_partial_route(pending_route.target)
            self._trace(
                "drone_partial_frontier_route_cancelled",
                target=pending_route.target,
                reason=(
                    "route_failed" if target_active else "frontier_changed"
                ),
            )
            self._pending_frontier_route = None

        self._trace("drone_move_start", state=self._snapshot_summary())
        if self._recover_from_stagnation():
            return
        try:
            valid_directions, border_targets, chosen_target = (
                self.find_new_node()
            )
        except AssertionError:
            self.update_borders()
            if not self.reach_border():
                self._trace(
                    "drone_no_reachable_border",
                    state=self._snapshot_summary(),
                )
            return

        self.explore(valid_directions, border_targets, chosen_target)

    def find_new_node(
        self,
    ) -> tuple[list[int], list[Position], Position]:
        """Choose a clear random heading inside the current vision cone."""
        drone = self.drone
        snapshot = drone.snapshot()
        cone_center = float(snapshot.heading_deg) % 360.0
        vision_sensor = getattr(
            getattr(drone, "sensor_controller", None),
            "vision_sensor",
            None,
        )
        vision_fov = float(getattr(vision_sensor, "fov_deg", 60.0))
        half_fov = max(0.0, min(180.0, vision_fov / 2.0))
        valid_directions, border_targets, step_targets = (
            self._direction_candidates(
                cone_center=cone_center,
                half_fov=half_fov,
            )
        )

        assert valid_directions
        bias = self._exploration_heading_bias(
            valid_directions,
            step_targets,
            vision_fov=vision_fov,
        )
        weighted_chooser = getattr(
            drone.exploration_policy,
            "choose_weighted_direction",
            None,
        )
        if callable(weighted_chooser):
            chosen_direction = weighted_chooser(bias.weights)
        else:
            chosen_direction = drone.exploration_policy.choose_direction(
                valid_directions
            )
        if chosen_direction not in step_targets:
            raise ValueError("exploration policy selected an invalid direction")
        drone.runtime_state.set_direction(chosen_direction)
        chosen_target = step_targets[chosen_direction]
        self._trace(
            "drone_random_direction_selected",
            direction=chosen_direction,
            target=chosen_target,
            valid_direction_count=len(valid_directions),
            border_count=len(border_targets),
            vision_cone_center=cone_center,
            vision_fov_deg=vision_fov,
            selection_mode=bias.mode,
            selected_weight=bias.weights[chosen_direction],
            selected_wall_support=bias.wall_support[chosen_direction],
            selected_frontier_support=(
                bias.frontier_support[chosen_direction]
            ),
            selected_global_frontier_support=(
                bias.global_support[chosen_direction]
            ),
            selected_separation_support=(
                bias.separation_support[chosen_direction]
            ),
            maximum_wall_support=max(bias.wall_support.values()),
            maximum_frontier_support=max(bias.frontier_support.values()),
            maximum_global_frontier_support=max(
                bias.global_support.values()
            ),
            peer_count=bias.peer_count,
            frontier_cluster_count=bias.cluster_count,
            eligible_frontier_cluster_count=bias.eligible_cluster_count,
            filtered_frontier_cluster_count=bias.filtered_cluster_count,
            selected_frontier_cluster_size=bias.selected_cluster_size,
            selected_frontier_cluster_distance=(
                bias.selected_cluster_distance
            ),
            selected_frontier_touches_wall=(
                bias.selected_cluster_touches_wall
            ),
            selected_wall_continuation_alignment=(
                bias.selected_continuation_alignment
            ),
            selected_frontier_cluster_score=bias.selected_cluster_score,
            selected_frontier_cluster_size_rank=(
                bias.selected_cluster_size_rank
            ),
            selected_frontier_cluster_proximity=(
                bias.selected_cluster_proximity
            ),
            wall_frontier_candidate_count=bias.wall_candidate_count,
            generic_frontier_candidate_count=bias.generic_candidate_count,
            minimum_frontier_cluster_cells=(
                self.minimum_frontier_cluster_cells
            ),
            global_frontier_active=bias.global_active,
            global_frontier_target=bias.global_target,
            global_frontier_region_size=bias.global_region_size,
            global_frontier_region_distance=bias.global_region_distance,
            global_frontier_region_bearing=bias.global_region_bearing,
            global_frontier_touches_wall=(
                bias.global_region_touches_wall
            ),
            global_frontier_region_score=bias.global_region_score,
            global_frontier_region_size_rank=(
                bias.global_region_size_rank
            ),
            global_frontier_region_proximity=(
                bias.global_region_proximity
            ),
            global_frontier_region_count=bias.global_region_count,
            global_frontier_eligible_region_count=(
                bias.global_eligible_region_count
            ),
            global_frontier_filtered_region_count=(
                bias.global_filtered_region_count
            ),
            global_wall_frontier_candidate_count=(
                bias.global_wall_candidate_count
            ),
            global_generic_frontier_candidate_count=(
                bias.global_generic_candidate_count
            ),
            global_frontier_slam_version=bias.global_slam_version,
        )
        return valid_directions, border_targets, chosen_target

    def _exploration_heading_bias(
        self,
        valid_directions: Iterable[int],
        step_targets: dict[int, Position],
        *,
        vision_fov: float,
    ) -> _HeadingBias:
        """Score wall tips, generic unknown borders, and team separation."""
        directions = tuple(int(direction) for direction in valid_directions)
        drone = self.drone
        current = drone.snapshot().position
        vision_sensor = getattr(
            getattr(drone, "sensor_controller", None),
            "vision_sensor",
            None,
        )
        sensor_range = float(getattr(
            vision_sensor,
            "max_range",
            drone.radius * 4,
        ))
        margin = int(math.ceil(sensor_range + drone.step + 2.0))
        slam = drone.slam_map.snapshot_window((
            current[0] - margin,
            current[1] - margin,
            current[0] + margin + 1,
            current[1] + margin + 1,
        ))
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        known = confidence >= self.frontier_confidence_threshold
        known_free = known & (occupancy == FREE)
        known_occupied = known & (occupancy == OCCUPIED)
        unknown = (~known) | (occupancy == UNKNOWN)
        frontier_unknown = unknown & self._neighbor_adjacency(known_free)
        wall_unknown = (
            frontier_unknown & self._neighbor_adjacency(known_occupied)
        )

        clusters = self._frontier_clusters(
            frontier_unknown,
            wall_unknown,
            slam.origin,
            current=current,
            heading=float(drone.snapshot().heading_deg),
        )
        eligible_clusters = tuple(
            cluster
            for cluster in clusters
            if cluster.size >= self.minimum_frontier_cluster_cells
        )
        selection = self._select_frontier_cluster(
            eligible_clusters,
            directions,
            step_targets,
            sensor_range=sensor_range,
            half_fov=max(0.0, min(180.0, vision_fov / 2.0)),
        )
        selected_cluster = selection.cluster
        frontier_support = dict(selection.support)
        wall_support = {
            direction: (
                selection.support[direction]
                if selected_cluster is not None
                and selected_cluster.touches_wall
                else 0.0
            )
            for direction in directions
        }
        separation_support, peer_count = self._separation_scores(
            directions,
            sensor_range=sensor_range,
        )
        global_guidance = self._global_frontier_guidance(
            directions,
            current=current,
            heading=float(drone.snapshot().heading_deg),
            sensor_range=sensor_range,
        )
        normalized_wall = self._normalize_scores(wall_support)
        normalized_frontier = self._normalize_scores(frontier_support)
        wall_tracking = (
            selected_cluster is not None
            and selected_cluster.touches_wall
        )
        frontier_tracking = selected_cluster is not None
        if global_guidance.active and global_guidance.touches_wall:
            mode = "global_wall_tracking"
        elif global_guidance.active:
            mode = "global_unexplored_region"
        elif wall_tracking:
            mode = "wall_tracking"
        elif frontier_tracking:
            mode = "unexplored_region"
        else:
            mode = "distributed_random"

        weights: dict[int, float] = {}
        for direction in directions:
            local_information_bias = (
                self.wall_direction_bias * normalized_wall[direction]
                if wall_tracking
                else self.unexplored_direction_bias
                * normalized_frontier[direction]
            )
            information_bias = local_information_bias
            if global_guidance.active:
                global_weight = (
                    self.wall_direction_bias
                    if global_guidance.touches_wall
                    else self.unexplored_direction_bias
                )
                # A whole-map target is strategic.  Nearby exact geometry
                # remains useful for collision-safe wall following, but does
                # not get to pull the drone back into a small local hole.
                information_bias = (
                    0.35 * local_information_bias
                    + global_weight * global_guidance.support[direction]
                )
            weights[direction] = max(
                1e-6,
                1.0
                + information_bias
                + self.separation_direction_bias
                * separation_support[direction],
            )
        return _HeadingBias(
            weights=weights,
            wall_support=wall_support,
            frontier_support=frontier_support,
            global_support=global_guidance.support,
            separation_support=separation_support,
            mode=mode,
            peer_count=peer_count,
            cluster_count=len(clusters),
            eligible_cluster_count=len(eligible_clusters),
            filtered_cluster_count=(
                len(clusters) - len(eligible_clusters)
            ),
            selected_cluster_size=(
                selected_cluster.size if selected_cluster is not None else 0
            ),
            selected_cluster_distance=(
                selected_cluster.distance
                if selected_cluster is not None
                else None
            ),
            selected_cluster_touches_wall=(
                selected_cluster.touches_wall
                if selected_cluster is not None
                else False
            ),
            selected_continuation_alignment=(
                selected_cluster.continuation_alignment
                if selected_cluster is not None
                else 0.0
            ),
            selected_cluster_score=selection.score,
            selected_cluster_size_rank=selection.size_rank,
            selected_cluster_proximity=selection.proximity,
            wall_candidate_count=selection.wall_candidate_count,
            generic_candidate_count=selection.generic_candidate_count,
            global_active=global_guidance.active,
            global_target=global_guidance.target,
            global_region_size=global_guidance.size,
            global_region_distance=global_guidance.distance,
            global_region_bearing=global_guidance.bearing,
            global_region_touches_wall=global_guidance.touches_wall,
            global_region_score=global_guidance.score,
            global_region_size_rank=global_guidance.size_rank,
            global_region_proximity=global_guidance.proximity,
            global_region_count=global_guidance.region_count,
            global_eligible_region_count=(
                global_guidance.eligible_region_count
            ),
            global_filtered_region_count=(
                global_guidance.filtered_region_count
            ),
            global_wall_candidate_count=(
                global_guidance.wall_candidate_count
            ),
            global_generic_candidate_count=(
                global_guidance.generic_candidate_count
            ),
            global_slam_version=global_guidance.slam_version,
        )

    def _global_frontier_guidance(
        self,
        directions: tuple[int, ...],
        *,
        current: Position,
        heading: float,
        sensor_range: float,
    ) -> _GlobalFrontierGuidance:
        """Return cheap per-step bearings from a periodically rebuilt cache."""
        cache = self._ensure_global_frontier_cache(
            current=current,
            heading=heading,
        )
        zero_support = {direction: 0.0 for direction in directions}
        region = cache.target
        target_position = cache.target_position
        if region is None or target_position is None:
            return _GlobalFrontierGuidance(
                support=zero_support,
                active=False,
                target=None,
                size=0,
                distance=None,
                bearing=None,
                touches_wall=False,
                score=0.0,
                size_rank=0.0,
                proximity=0.0,
                region_count=len(cache.regions),
                eligible_region_count=cache.eligible_region_count,
                filtered_region_count=cache.filtered_region_count,
                wall_candidate_count=cache.wall_candidate_count,
                generic_candidate_count=cache.generic_candidate_count,
                slam_version=cache.slam_version,
            )

        distance = math.dist(current, target_position)
        bearing = self._bearing(current, target_position)
        local_window_radius = sensor_range + self.drone.step + 2.0
        active = distance > local_window_radius
        support = (
            self._directional_progress_scores(directions, bearing)
            if active
            else zero_support
        )
        return _GlobalFrontierGuidance(
            support=support,
            active=active,
            target=target_position,
            size=region.size,
            distance=distance,
            bearing=bearing,
            touches_wall=region.touches_wall,
            score=cache.target_score,
            size_rank=cache.target_size_rank,
            proximity=cache.target_proximity,
            region_count=len(cache.regions),
            eligible_region_count=cache.eligible_region_count,
            filtered_region_count=cache.filtered_region_count,
            wall_candidate_count=cache.wall_candidate_count,
            generic_candidate_count=cache.generic_candidate_count,
            slam_version=cache.slam_version,
        )

    def _ensure_global_frontier_cache(
        self,
        *,
        current: Position,
        heading: float,
        slam: Any | None = None,
    ) -> _GlobalFrontierCache:
        """Rebuild coarse whole-map regions only after the refresh cadence."""
        now = self._simulation_time()
        cache = self._global_frontier_cache
        current_version = self.drone.slam_map.version
        if cache is not None and (
            cache.slam_version == current_version
            or now - cache.built_at < self.global_frontier_refresh_interval
        ):
            return cache

        started = time.perf_counter()
        if slam is None:
            slam = self.drone.slam_map.snapshot(point_limit=0)
        regions = self._coarse_global_frontier_regions(slam)
        (
            target,
            target_position,
            target_score,
            target_size_rank,
            target_proximity,
            eligible_count,
            wall_count,
            generic_count,
        ) = self._select_global_frontier_region(
            regions,
            current=current,
            heading=heading,
        )
        cache = _GlobalFrontierCache(
            regions=regions,
            target=target,
            target_position=target_position,
            target_score=target_score,
            target_size_rank=target_size_rank,
            target_proximity=target_proximity,
            eligible_region_count=eligible_count,
            filtered_region_count=len(regions) - eligible_count,
            wall_candidate_count=wall_count,
            generic_candidate_count=generic_count,
            slam_version=int(slam.version),
            built_at=now,
        )
        self._global_frontier_cache = cache
        self._trace(
            "drone_global_frontiers_rebuilt",
            slam_version=cache.slam_version,
            coarse_cell_size=self.global_frontier_cell_size,
            region_count=len(regions),
            eligible_region_count=eligible_count,
            filtered_region_count=cache.filtered_region_count,
            wall_candidate_count=wall_count,
            generic_candidate_count=generic_count,
            selected_target=target_position,
            selected_region_size=(target.size if target is not None else 0),
            selected_region_touches_wall=(
                target.touches_wall if target is not None else False
            ),
            selected_score=target_score,
            selected_size_rank=target_size_rank,
            selected_proximity=target_proximity,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return cache

    def _coarse_global_frontier_regions(
        self,
        slam: Any,
    ) -> tuple[_GlobalFrontierRegion, ...]:
        """Aggregate full-SLAM frontier pixels before connected components."""
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        known = confidence >= self.frontier_confidence_threshold
        known_free = known & (occupancy == FREE)
        known_occupied = known & (occupancy == OCCUPIED)
        unknown = (~known) | (occupancy == UNKNOWN)
        frontier_unknown = unknown & self._neighbor_adjacency(known_free)
        rows, columns = np.nonzero(frontier_unknown)
        if len(rows) == 0:
            return ()

        wall_unknown = (
            frontier_unknown & self._neighbor_adjacency(known_occupied)
        )
        cell_size = self.global_frontier_cell_size
        height, width = occupancy.shape
        tile_columns = max(1, math.ceil(width / cell_size))
        tile_rows = max(1, math.ceil(height / cell_size))
        tile_x = columns // cell_size
        tile_y = rows // cell_size
        flat_indices = tile_y * tile_columns + tile_x
        tile_total = tile_rows * tile_columns
        counts = np.bincount(flat_indices, minlength=tile_total)
        wall_counts = np.bincount(
            flat_indices,
            weights=wall_unknown[rows, columns].astype(np.float64),
            minlength=tile_total,
        )
        x_sums = np.bincount(
            flat_indices,
            weights=columns.astype(np.float64),
            minlength=tile_total,
        )
        y_sums = np.bincount(
            flat_indices,
            weights=rows.astype(np.float64),
            minlength=tile_total,
        )

        remaining = {int(index) for index in np.flatnonzero(counts)}
        regions: list[_GlobalFrontierRegion] = []
        for seed in sorted(remaining):
            if seed not in remaining:
                continue
            remaining.remove(seed)
            stack = [seed]
            members: list[int] = []
            while stack:
                flat_index = stack.pop()
                members.append(flat_index)
                member_y, member_x = divmod(flat_index, tile_columns)
                for offset_y in (-1, 0, 1):
                    for offset_x in (-1, 0, 1):
                        if offset_x == 0 and offset_y == 0:
                            continue
                        neighbor_x = member_x + offset_x
                        neighbor_y = member_y + offset_y
                        if not (
                            0 <= neighbor_x < tile_columns
                            and 0 <= neighbor_y < tile_rows
                        ):
                            continue
                        neighbor = neighbor_y * tile_columns + neighbor_x
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            stack.append(neighbor)

            member_indices = np.asarray(members, dtype=np.int64)
            size = int(np.sum(counts[member_indices]))
            tiles = tuple(sorted(
                (
                    _GlobalFrontierTile(
                        target=(
                            int(round(x_sums[index] / counts[index])),
                            int(round(y_sums[index] / counts[index])),
                        ),
                        size=int(counts[index]),
                        wall_cells=int(round(float(wall_counts[index]))),
                    )
                    for index in members
                ),
                key=lambda tile: tile.target,
            ))
            regions.append(_GlobalFrontierRegion(
                tiles=tiles,
                size=size,
                wall_cells=int(round(float(np.sum(
                    wall_counts[member_indices]
                )))),
                tile_count=len(members),
            ))
        return tuple(sorted(
            regions,
            key=lambda region: (region.tiles[0].target, -region.size),
        ))

    def _select_global_frontier_region(
        self,
        regions: tuple[_GlobalFrontierRegion, ...],
        *,
        current: Position,
        heading: float,
    ) -> tuple[
        _GlobalFrontierRegion | None,
        Position | None,
        float,
        float,
        float,
        int,
        int,
        int,
    ]:
        """Use the local policy hierarchy on coarse whole-map regions."""
        eligible = tuple(
            region
            for region in regions
            if region.size >= self.minimum_frontier_cluster_cells
        )
        wall_regions = tuple(
            region for region in eligible if region.touches_wall
        )
        generic_regions = tuple(
            region for region in eligible if not region.touches_wall
        )
        tier = wall_regions or generic_regions
        if not tier:
            return None, None, 0.0, 0.0, 0.0, len(eligible), 0, 0

        sizes = sorted({region.size for region in tier})
        size_ranks = (
            {sizes[0]: 1.0}
            if len(sizes) == 1
            else {
                size: index / float(len(sizes) - 1)
                for index, size in enumerate(sizes)
            }
        )
        scored = []
        for region in tier:
            representative_tiles = (
                tuple(tile for tile in region.tiles if tile.touches_wall)
                if region.touches_wall
                else region.tiles
            )
            tile_terms = []
            for tile in representative_tiles:
                distance = math.dist(current, tile.target)
                bearing = self._bearing(current, tile.target)
                alignment = max(
                    0.0,
                    1.0
                    - self._angular_distance(bearing, heading) / 180.0,
                )
                tile_terms.append((alignment, distance, tile))
            if region.touches_wall:
                alignment, distance, representative = min(
                    tile_terms,
                    key=lambda item: (
                        -item[0],
                        item[1],
                        -item[2].size,
                        item[2].target,
                    ),
                )
            else:
                alignment, distance, representative = min(
                    tile_terms,
                    key=lambda item: (
                        item[1],
                        -item[2].size,
                        item[2].target,
                    ),
                )
            proximity = 1.0 / (
                1.0 + distance / self.frontier_distance_band
            )
            size_rank = size_ranks[region.size]
            score = (
                self.frontier_cluster_size_weight * size_rank
                + self.frontier_cluster_proximity_weight * proximity
            )
            if region.touches_wall:
                score += self.wall_continuation_weight * alignment
            scored.append((
                score,
                size_rank,
                proximity,
                distance,
                region,
                representative.target,
            ))

        score, size_rank, proximity, _distance, target, target_position = min(
            scored,
            key=lambda item: (
                -item[0],
                -item[4].size,
                item[3],
                item[5],
            ),
        )
        return (
            target,
            target_position,
            score,
            size_rank,
            proximity,
            len(eligible),
            len(wall_regions),
            len(generic_regions),
        )

    @classmethod
    def _directional_progress_scores(
        cls,
        directions: Iterable[int],
        target_bearing: float,
    ) -> dict[int, float]:
        """Normalize which valid headings turn most toward a remote target."""
        closeness = {
            int(direction): 180.0 - cls._angular_distance(
                direction,
                target_bearing,
            )
            for direction in directions
        }
        if not closeness:
            return {}
        minimum = min(closeness.values())
        maximum = max(closeness.values())
        if maximum - minimum <= 1e-9:
            return {direction: 1.0 for direction in closeness}
        return {
            direction: (value - minimum) / (maximum - minimum)
            for direction, value in closeness.items()
        }

    @staticmethod
    def _bearing(origin: Position, target: Position) -> float:
        """Return simulator heading from origin to target."""
        return math.degrees(math.atan2(
            target[0] - origin[0],
            -(target[1] - origin[1]),
        )) % 360.0

    def _frontier_clusters(
        self,
        frontier_mask: np.ndarray,
        wall_mask: np.ndarray,
        window_origin: Position,
        *,
        current: Position,
        heading: float,
    ) -> tuple[_FrontierCluster, ...]:
        """Extract deterministic eight-connected local boundary components."""
        rows, columns = np.nonzero(frontier_mask)
        remaining = {
            (int(column), int(row))
            for row, column in zip(rows, columns)
        }
        clusters: list[_FrontierCluster] = []
        for seed in sorted(remaining):
            if seed not in remaining:
                continue
            remaining.remove(seed)
            stack = [seed]
            local_cells: list[Position] = []
            touches_wall = False
            while stack:
                local_x, local_y = stack.pop()
                local_cells.append((local_x, local_y))
                touches_wall = touches_wall or bool(
                    wall_mask[local_y, local_x]
                )
                for offset_y in (-1, 0, 1):
                    for offset_x in (-1, 0, 1):
                        if offset_x == 0 and offset_y == 0:
                            continue
                        neighbor = (
                            local_x + offset_x,
                            local_y + offset_y,
                        )
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            stack.append(neighbor)

            global_cells = tuple(sorted(
                (
                    local_x + int(window_origin[0]),
                    local_y + int(window_origin[1]),
                )
                for local_x, local_y in local_cells
            ))
            distance = min(
                math.dist(current, cell) for cell in global_cells
            )
            centroid_x = sum(cell[0] for cell in global_cells) / len(
                global_cells
            )
            centroid_y = sum(cell[1] for cell in global_cells) / len(
                global_cells
            )
            bearing = math.degrees(math.atan2(
                centroid_x - current[0],
                -(centroid_y - current[1]),
            )) % 360.0
            angular_delta = self._angular_distance(bearing, heading)
            alignment = max(0.0, 1.0 - angular_delta / 180.0)
            clusters.append(_FrontierCluster(
                cells=global_cells,
                size=len(global_cells),
                touches_wall=touches_wall,
                distance=distance,
                continuation_alignment=alignment,
            ))
        return tuple(clusters)

    def _select_frontier_cluster(
        self,
        clusters: Iterable[_FrontierCluster],
        directions: tuple[int, ...],
        step_targets: dict[int, Position],
        *,
        sensor_range: float,
        half_fov: float,
    ) -> _ClusterSelection:
        """Apply strict wall/generic tiers and weighted within-tier scoring."""
        candidates = tuple(clusters)
        wall_clusters = tuple(
            cluster for cluster in candidates if cluster.touches_wall
        )
        generic_clusters = tuple(
            cluster for cluster in candidates if not cluster.touches_wall
        )

        wall_candidates = self._actionable_cluster_support(
            wall_clusters,
            directions,
            step_targets,
            sensor_range=sensor_range,
            half_fov=half_fov,
        )
        tier = wall_candidates
        if not tier:
            tier = self._actionable_cluster_support(
                generic_clusters,
                directions,
                step_targets,
                sensor_range=sensor_range,
                half_fov=half_fov,
            )
        if not tier:
            return _ClusterSelection(
                cluster=None,
                support={direction: 0.0 for direction in directions},
                score=0.0,
                size_rank=0.0,
                proximity=0.0,
                wall_candidate_count=len(wall_clusters),
                generic_candidate_count=len(generic_clusters),
            )

        size_ranks = self._cluster_size_ranks(
            candidate[0] for candidate in tier
        )
        scored: list[
            tuple[float, float, float, _FrontierCluster, dict[int, float]]
        ] = []
        for cluster, support in tier:
            size_rank = size_ranks[cluster.size]
            proximity = 1.0 / (
                1.0 + cluster.distance / self.frontier_distance_band
            )
            score = (
                self.frontier_cluster_size_weight * size_rank
                + self.frontier_cluster_proximity_weight * proximity
            )
            if cluster.touches_wall:
                score += (
                    self.wall_continuation_weight
                    * cluster.continuation_alignment
                )
            scored.append((score, size_rank, proximity, cluster, support))

        score, size_rank, proximity, cluster, support = min(
            scored,
            key=lambda item: (
                -item[0],
                -item[3].size,
                item[3].distance,
                item[3].cells[0],
            ),
        )
        return _ClusterSelection(
            cluster=cluster,
            support=support,
            score=score,
            size_rank=size_rank,
            proximity=proximity,
            wall_candidate_count=len(wall_clusters),
            generic_candidate_count=len(generic_clusters),
        )

    def _actionable_cluster_support(
        self,
        clusters: Iterable[_FrontierCluster],
        directions: tuple[int, ...],
        step_targets: dict[int, Position],
        *,
        sensor_range: float,
        half_fov: float,
    ) -> tuple[tuple[_FrontierCluster, dict[int, float]], ...]:
        """Attach heading support to components visible from a candidate step."""
        actionable = []
        for cluster in clusters:
            support = self._cone_point_support_scores(
                directions,
                step_targets,
                cluster.cells,
                sensor_range=sensor_range,
                half_fov=half_fov,
            )
            if max(support.values(), default=0.0) > 0.0:
                actionable.append((cluster, support))
        return tuple(actionable)

    @staticmethod
    def _cluster_size_ranks(
        clusters: Iterable[_FrontierCluster],
    ) -> dict[int, float]:
        """Return normalized ordinal ranks with the largest size at one."""
        sizes = sorted({cluster.size for cluster in clusters})
        if len(sizes) == 1:
            return {sizes[0]: 1.0}
        denominator = float(len(sizes) - 1)
        return {
            size: index / denominator
            for index, size in enumerate(sizes)
        }

    @staticmethod
    def _neighbor_adjacency(mask: np.ndarray) -> np.ndarray:
        """Return cells adjacent to at least one true eight-neighbor."""
        height, width = mask.shape
        adjacent = np.zeros_like(mask, dtype=bool)
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                if offset_x == 0 and offset_y == 0:
                    continue
                source_y = slice(
                    max(0, -offset_y),
                    height - max(0, offset_y),
                )
                target_y = slice(
                    max(0, offset_y),
                    height - max(0, -offset_y),
                )
                source_x = slice(
                    max(0, -offset_x),
                    width - max(0, offset_x),
                )
                target_x = slice(
                    max(0, offset_x),
                    width - max(0, -offset_x),
                )
                adjacent[target_y, target_x] |= mask[source_y, source_x]
        return adjacent

    def _cone_point_support_scores(
        self,
        directions: Iterable[int],
        step_targets: dict[int, Position],
        support_points: Iterable[Position],
        *,
        sensor_range: float,
        half_fov: float,
    ) -> dict[int, float]:
        """Estimate boundary evidence visible after each candidate step."""
        points = np.asarray(tuple(support_points), dtype=np.float32)
        if points.size == 0:
            return {int(direction): 0.0 for direction in directions}
        point_x = points[:, 0]
        point_y = points[:, 1]
        scores: dict[int, float] = {}
        current = self.drone.snapshot().position
        for direction in directions:
            target = step_targets[direction]
            lookahead_x = (float(current[0]) + float(target[0])) / 2.0
            lookahead_y = (float(current[1]) + float(target[1])) / 2.0
            delta_x = point_x - lookahead_x
            delta_y = point_y - lookahead_y
            distance = np.hypot(delta_x, delta_y)
            angles = np.degrees(np.arctan2(delta_x, -delta_y))
            angle_delta = (
                angles - float(direction) + 180.0
            ) % 360.0 - 180.0
            visible = (
                (distance > 0.0)
                & (distance <= sensor_range)
                & (np.abs(angle_delta) <= half_fov)
            )
            if not np.any(visible):
                scores[direction] = 0.0
                continue
            scores[direction] = float(np.sum(
                1.0 - 0.5 * distance[visible] / max(sensor_range, 1.0),
                dtype=np.float64,
            ))
        return scores

    def _separation_scores(
        self,
        directions: Iterable[int],
        *,
        sensor_range: float,
    ) -> tuple[dict[int, float], int]:
        """Prefer headings away from nearby peers and initial launch overlap."""
        drone = self.drone
        current = drone.snapshot().position
        positions = tuple(self.dependencies.get_drone_positions())
        peer_positions = tuple(
            position
            for drone_id, position in positions
            if int(drone_id) != int(drone.id)
        )
        separation_radius = max(sensor_range * 2.0, drone.step * 8.0)
        vector_x = 0.0
        vector_y = 0.0
        for peer_position in peer_positions:
            delta_x = float(current[0] - peer_position[0])
            delta_y = float(current[1] - peer_position[1])
            distance = math.hypot(delta_x, delta_y)
            if distance <= 1e-9 or distance >= separation_radius:
                continue
            strength = (1.0 - distance / separation_radius) ** 2
            vector_x += delta_x / distance * strength
            vector_y += delta_y / distance * strength

        drone_count = int(drone.settings.mission_config.num_drones)
        if drone_count > 1:
            launch_distance = math.dist(current, drone.start_pos)
            launch_strength = max(
                0.0,
                1.0 - launch_distance / separation_radius,
            )
            sector_heading = 360.0 * float(drone.id) / drone_count
            sector_radians = math.radians(sector_heading)
            vector_x += math.sin(sector_radians) * launch_strength
            vector_y -= math.cos(sector_radians) * launch_strength

        magnitude = math.hypot(vector_x, vector_y)
        if magnitude <= 1e-9:
            scores = {int(direction): 0.0 for direction in directions}
            return scores, len(peer_positions)
        preferred_heading = math.degrees(
            math.atan2(vector_x, -vector_y)
        ) % 360.0
        scores = {
            int(direction): (
                1.0
                + math.cos(math.radians(
                    self._angular_distance(direction, preferred_heading)
                ))
            ) / 2.0
            for direction in directions
        }
        return scores, len(peer_positions)

    @staticmethod
    def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
        """Scale nonnegative heading evidence into the unit interval."""
        maximum = max(scores.values(), default=0.0)
        if maximum <= 0.0:
            return {direction: 0.0 for direction in scores}
        return {
            direction: max(0.0, value) / maximum
            for direction, value in scores.items()
        }

    def _direction_candidates(
        self,
        *,
        cone_center: float,
        half_fov: float,
    ) -> tuple[list[int], list[Position], dict[int, Position]]:
        """Return collision-free headings and their look-ahead coordinates."""
        drone = self.drone
        current = drone.snapshot().position
        valid_directions: list[int] = []
        border_targets: list[Position] = []
        step_targets: dict[int, Position] = {}

        for direction in range(360):
            if self._angular_distance(direction, cone_center) > half_fov:
                continue
            border = next_cell_coords(
                *current,
                drone.radius + 1,
                direction,
            )
            step_target = next_cell_coords(
                *current,
                drone.step,
                direction,
            )
            if not drone.runtime_state.graph_is_valid(current, border):
                continue
            if not drone.runtime_state.graph_is_valid(current, step_target):
                continue
            valid_directions.append(direction)
            border_targets.append(border)
            step_targets[direction] = step_target
        return valid_directions, border_targets, step_targets

    @staticmethod
    def _angular_distance(first: float, second: float) -> float:
        """Return the shortest absolute distance between two headings."""
        return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)

    def _recover_from_stagnation(self) -> bool:
        """Redirect a low-information random walk toward local unknown space."""
        distance = self._stagnation_distance_travelled
        if distance < self.stagnation_distance:
            return False

        progress = self.drone.slam_map.progress_snapshot()
        sensor_cells = max(
            0,
            progress.sensor_newly_known_cells
            - self._stagnation_sensor_baseline,
        )
        sensor_cells_per_px = sensor_cells / max(distance, 1e-9)
        stagnant = (
            sensor_cells_per_px
            < self.stagnation_min_sensor_cells_per_px
        )
        self._trace(
            "drone_stagnation_window",
            travelled_distance=distance,
            sensor_newly_known_cells=sensor_cells,
            sensor_cells_per_px=sensor_cells_per_px,
            minimum_sensor_cells_per_px=(
                self.stagnation_min_sensor_cells_per_px
            ),
            stagnant=stagnant,
            slam_version=progress.version,
        )
        self._reset_stagnation_window(progress)
        if not stagnant:
            return False

        self.rebuild_frontiers(
            stride=self.frontier_stride,
            confidence_threshold=self.frontier_confidence_threshold,
        )
        frontiers = self.drone.snapshot().frontiers
        self._trace(
            "drone_stagnation_detected",
            travelled_distance=distance,
            sensor_newly_known_cells=sensor_cells,
            sensor_cells_per_px=sensor_cells_per_px,
            frontier_count=len(frontiers),
        )
        if self._start_frontier_scan(
            frontiers,
            reason="local_unknown_frontier",
        ):
            return True
        if self.reach_border(
            avoid_recent_trail=True,
            recovery_reason="stagnation",
        ):
            return True

        self._trace(
            "drone_stagnation_unresolved",
            frontier_count=len(frontiers),
            state=self._snapshot_summary(),
        )
        return False

    def _reset_stagnation_window(self, progress: Any | None = None) -> None:
        """Start a fresh sensor-local productivity window."""
        current = progress or self.drone.slam_map.progress_snapshot()
        self._stagnation_sensor_baseline = (
            current.sensor_newly_known_cells
        )
        self._stagnation_distance_travelled = 0.0

    def _start_frontier_scan(
        self,
        frontiers: Iterable[Position],
        *,
        reason: str,
    ) -> bool:
        """Face one visible unknown boundary and wait for its sensor scan."""
        drone = self.drone
        snapshot = drone.snapshot()
        slam = drone.slam_map.snapshot(point_limit=0)
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        unknown = (
            (occupancy == UNKNOWN)
            | (confidence < self.frontier_confidence_threshold)
        )
        vision_sensor = getattr(
            getattr(drone, "sensor_controller", None),
            "vision_sensor",
            None,
        )
        sensor_range = float(getattr(
            vision_sensor,
            "max_range",
            drone.radius * 4,
        ))
        current_x, current_y = snapshot.position
        height, width = unknown.shape
        supported: list[
            tuple[float, Position, Position, int, float, int]
        ] = []

        for target in frontiers:
            target_x, target_y = target
            if not (0 <= target_x < width and 0 <= target_y < height):
                continue
            distance = math.dist(snapshot.position, target)
            if distance > sensor_range:
                continue
            if (
                target != snapshot.position
                and not drone.runtime_state.graph_is_valid(
                    snapshot.position,
                    target,
                )
            ):
                continue

            unknown_neighbors: list[Position] = []
            for neighbor_y in range(
                max(0, target_y - 1),
                min(height, target_y + 2),
            ):
                for neighbor_x in range(
                    max(0, target_x - 1),
                    min(width, target_x + 2),
                ):
                    if unknown[neighbor_y, neighbor_x]:
                        unknown_neighbors.append((neighbor_x, neighbor_y))
            if not unknown_neighbors:
                continue

            distance_weight = 1.0 - 0.5 * min(
                1.0,
                distance / max(sensor_range, 1.0),
            )
            weight = len(unknown_neighbors) * distance_weight
            for unknown_target in unknown_neighbors:
                delta_x = unknown_target[0] - current_x
                delta_y = unknown_target[1] - current_y
                if delta_x == 0 and delta_y == 0:
                    continue
                angle = math.degrees(math.atan2(delta_x, -delta_y)) % 360.0
                supported.append((
                    weight,
                    target,
                    unknown_target,
                    len(unknown_neighbors),
                    distance,
                    int(round(angle)) % 360,
                ))

        if not supported:
            return False

        best_score = max(item[0] for item in supported)
        if best_score <= 0.0:
            return False
        strongest = [
            item for item in supported
            if math.isclose(item[0], best_score, rel_tol=1e-9)
        ]
        nearest_distance = min(item[4] for item in strongest)
        strongest = [
            item for item in strongest
            if math.isclose(item[4], nearest_distance, abs_tol=1e-9)
        ]
        best_directions = sorted({item[5] for item in strongest})
        chosen_direction = drone.exploration_policy.choose_direction(
            best_directions
        )
        selected = max(
            (item for item in strongest if item[5] == chosen_direction),
            key=lambda item: (item[3], item[1], item[2]),
        )
        geometry = None
        if selected[1] in self._last_raw_frontiers:
            geometry = self._frontier_geometry_signature(
                selected[1],
                self._last_raw_frontiers,
            )
        progress = drone.slam_map.progress_snapshot()
        sensor = getattr(drone, "sensor_controller", None)
        last_scan = getattr(sensor, "last_completed_scan", None)
        expected_pose = (
            int(snapshot.position[0]),
            int(snapshot.position[1]),
            round(float(chosen_direction) % 360.0, 3),
        )
        reuse_completed_scan = bool(
            last_scan is not None
            and last_scan.pose == expected_pose
            and last_scan.sequence == progress.completed_scan_sequence
        )
        minimum_sequence = progress.completed_scan_sequence
        if reuse_completed_scan:
            minimum_sequence -= 1

        self._pending_frontier_scan = _PendingFrontierScan(
            position=snapshot.position,
            heading=chosen_direction,
            frontier_target=selected[1],
            unknown_target=selected[2],
            reason=reason,
            minimum_scan_sequence=minimum_sequence,
            baseline_geometry=geometry,
        )
        drone.runtime_state.reorient(chosen_direction)
        self._trace(
            "drone_stagnation_scan_started",
            position=snapshot.position,
            incoming_heading=snapshot.heading_deg,
            direction=chosen_direction,
            frontier_target=selected[1],
            unknown_target=selected[2],
            frontier_distance=selected[4],
            unknown_neighbor_count=selected[3],
            unknown_support_score=best_score,
            candidate_heading_count=len(best_directions),
            minimum_scan_sequence=minimum_sequence,
            reused_completed_scan=reuse_completed_scan,
            slam_version=slam.version,
            reason=reason,
        )
        return True

    def _advance_pending_frontier_scan(self) -> None:
        """Finish a requested scan, then restore a movement-safe heading."""
        pending = self._pending_frontier_scan
        if pending is None:
            return
        sensor = getattr(self.drone, "sensor_controller", None)
        completion = getattr(sensor, "last_completed_scan", None)
        expected_pose = (
            pending.position[0],
            pending.position[1],
            round(float(pending.heading) % 360.0, 3),
        )
        if (
            completion is None
            or completion.sequence <= pending.minimum_scan_sequence
            or completion.pose != expected_pose
        ):
            return

        sensor_cells = max(0, int(completion.newly_known_cells))
        confidence_gain = max(0.0, float(completion.confidence_gain))
        productive = sensor_cells > 0 or confidence_gain > 1e-9
        self.rebuild_frontiers(
            stride=self.frontier_stride,
            confidence_threshold=self.frontier_confidence_threshold,
        )
        current_geometry = None
        if pending.frontier_target in self._last_raw_frontiers:
            current_geometry = self._frontier_geometry_signature(
                pending.frontier_target,
                self._last_raw_frontiers,
            )

        suppressed = False
        disposition = "sensor_gain"
        if not productive:
            if current_geometry is None:
                disposition = "frontier_resolved"
            elif (
                pending.baseline_geometry is not None
                and current_geometry != pending.baseline_geometry
            ):
                disposition = "geometry_changed"
            else:
                self._suppress_frontier_target(
                    pending.frontier_target,
                    reason="zero_gain_directed_scan",
                )
                suppressed = True
                disposition = "unchanged_geometry_suppressed"

        self._pending_frontier_scan = None
        self._trace(
            "drone_stagnation_scan_completed",
            position=pending.position,
            direction=pending.heading,
            frontier_target=pending.frontier_target,
            unknown_target=pending.unknown_target,
            reason=pending.reason,
            completed_scan_sequence=completion.sequence,
            sensor_newly_known_cells=sensor_cells,
            sensor_confidence_gain=confidence_gain,
            productive=productive,
            frontier_suppressed=suppressed,
            disposition=disposition,
            slam_version=self.drone.slam_map.version,
        )
        self._restore_movement_heading_after_scan(pending)

    def _restore_movement_heading_after_scan(
        self,
        pending: _PendingFrontierScan,
    ) -> bool:
        """Leave a scan-only pose through a collision-safe full-circle turn."""
        drone = self.drone
        snapshot = drone.snapshot()
        valid_directions, _border_targets, _step_targets = (
            self._direction_candidates(
                cone_center=snapshot.heading_deg,
                half_fov=180.0,
            )
        )
        if not valid_directions:
            self._trace(
                "drone_stagnation_scan_no_safe_exit",
                position=snapshot.position,
                scan_direction=pending.heading,
                frontier_target=pending.frontier_target,
            )
            self._reset_stagnation_window()
            return False

        chosen_direction = drone.exploration_policy.choose_direction(
            valid_directions
        )
        drone.runtime_state.reorient(chosen_direction)
        self._trace(
            "drone_stagnation_scan_exit_reoriented",
            position=snapshot.position,
            incoming_heading=snapshot.heading_deg,
            direction=chosen_direction,
            scan_direction=pending.heading,
            frontier_target=pending.frontier_target,
            valid_direction_count=len(valid_directions),
        )
        self._reset_stagnation_window()
        return True

    def explore(
        self,
        valid_directions: list[int],
        border_targets: list[Position],
        chosen_target: Position,
    ) -> bool:
        """Walk one collision-checked straight step without invoking A*."""
        drone = self.drone
        snapshot = drone.snapshot()
        drone.runtime_state.begin_exploration(
            snapshot.direction,
            border_targets,
        )
        path = bresenham_line_points(
            snapshot.position[0],
            snapshot.position[1],
            chosen_target[0],
            chosen_target[1],
        )
        followed = self._follow_path(path, source="random_step")
        self._trace(
            "drone_random_step",
            direction=snapshot.direction,
            target=chosen_target,
            valid_direction_count=len(valid_directions),
            completed=followed,
        )
        return followed

    def reach_border(
        self,
        *,
        avoid_recent_trail: bool = False,
        recovery_reason: str = "boxed_in",
        preferred_target: Position | None = None,
    ) -> bool:
        """Use A* to reach the nearest viable SLAM frontier."""
        drone = self.drone
        snapshot = drone.snapshot()
        frontiers = sorted(
            snapshot.frontiers,
            key=lambda target: self._distance_from(snapshot.position, target),
        )
        if preferred_target is not None:
            frontiers = [
                target for target in frontiers
                if target == preferred_target
            ]
        if not frontiers:
            return False

        recent_trail = (
            self._recent_trail_positions()
            if avoid_recent_trail else ()
        )
        if avoid_recent_trail:
            eligible = tuple(
                target
                for target in frontiers
                if target not in self._suppressed_frontier_geometry
                and not self._target_near_recent_trail(
                    target,
                    recent_trail,
                )
            )
            self._trace(
                "drone_stagnation_frontier_filter",
                frontier_count=len(frontiers),
                eligible_frontier_count=len(eligible),
                recent_trail_point_count=len(recent_trail),
                recent_trail_clearance=self._recent_trail_clearance(),
            )
            frontiers = list(eligible)
            if not frontiers:
                return False

        now = self._simulation_time()
        for target in frontiers:
            if target in self._suppressed_frontier_geometry:
                continue
            current = drone.snapshot().position
            if target == current:
                if (
                    recovery_reason == "stagnation"
                    and self._start_frontier_scan(
                        (target,),
                        reason="frontier_arrival_unknown",
                    )
                ):
                    return True
                self._suppress_reached_border(target)
                self._reorient_after_border(target)
                return True
            if now < self.border_retry_until.get(target, 0.0):
                continue

            result = self._compute_path(current, target)
            path = result.path
            self._trace(
                "drone_border_path",
                start=current,
                target=target,
                path_length=len(path),
                path_status=result.status,
                path_iterations=result.iterations,
                path_remaining_distance=result.remaining_distance,
                segment_endpoint=path[-1] if path else None,
                recovery_reason=recovery_reason,
            )
            if recovery_reason == "stagnation":
                self._trace(
                    "drone_stagnation_frontier_path",
                    start=current,
                    target=target,
                    path_length=len(path),
                    path_status=result.status,
                )
            if len(path) <= 1:
                self.border_retry_until[target] = (
                    now + self.border_retry_cooldown
                )
                continue
            if result.status == PATH_PARTIAL_LIMIT:
                if not self._accept_partial_endpoint(current, target, path[-1]):
                    self.border_retry_until[target] = (
                        now + self.border_retry_cooldown
                    )
                    continue
                self._pending_frontier_route = _PendingFrontierRoute(
                    target=target,
                    recovery_reason=recovery_reason,
                )
                return self._follow_path(
                    path,
                    source="border_astar_partial",
                )
            if result.status != PATH_COMPLETE:
                self.border_retry_until[target] = (
                    now + self.border_retry_cooldown
                )
                continue
            self._clear_partial_route(target)
            self._pending_frontier_route = None
            if not self._follow_path(path, source="border_astar"):
                return False

            if (
                recovery_reason == "stagnation"
                and self._start_frontier_scan(
                    (target,),
                    reason="frontier_arrival_unknown",
                )
            ):
                return True
            self._suppress_reached_border(target)
            self._reorient_after_border(target)
            return True
        return False

    def _recent_trail_positions(self) -> tuple[Position, ...]:
        """Return the recent breadcrumb suffix covering one gain window."""
        history = self.drone.snapshot().path_history
        if not history:
            return ()
        recent = [history[-1]]
        accumulated = 0.0
        for point in reversed(history[:-1]):
            accumulated += math.dist(point, recent[-1])
            recent.append(point)
            if accumulated >= self.stagnation_distance:
                break
        return tuple(recent)

    def _recent_trail_clearance(self) -> float:
        """Return the minimum separation for a stagnation A* target."""
        return float(max(self.drone.step * 2, self.frontier_stride * 2))

    def _target_near_recent_trail(
        self,
        target: Position,
        recent_trail: Iterable[Position],
    ) -> bool:
        """Return whether a target would send recovery back onto recent path."""
        clearance = self._recent_trail_clearance()
        return any(
            math.dist(target, point) <= clearance
            for point in recent_trail
        )

    def _suppress_reached_border(self, target: Position) -> None:
        """Retire a reached target until its local geometry changes."""
        self._suppress_frontier_target(
            target,
            reason="reached_unchanged_local_geometry",
        )

    def _suppress_frontier_target(
        self,
        target: Position,
        *,
        reason: str,
    ) -> None:
        """Retire one target until its sampled local geometry changes."""
        geometry = None
        if target in self._last_raw_frontiers:
            geometry = self._frontier_geometry_signature(
                target,
                self._last_raw_frontiers,
            )
        self._suppressed_frontier_geometry[target] = geometry
        self.drone.runtime_state.remove_frontier(target)
        self.border_retry_until.pop(target, None)
        self._clear_partial_route(target)
        if (
            self._pending_frontier_route is not None
            and self._pending_frontier_route.target == target
        ):
            self._pending_frontier_route = None
        self._trace(
            "drone_border_target_suppressed",
            target=target,
            reason=reason,
            local_geometry_point_count=(
                0 if geometry is None else len(geometry)
            ),
            slam_version=self.drone.slam_map.version,
        )

    def _reorient_after_border(self, target: Position) -> bool:
        """Turn toward a usable exit after A* reaches an escape border."""
        drone = self.drone
        snapshot = drone.snapshot()
        valid_directions, _border_targets, step_targets = (
            self._direction_candidates(
                cone_center=snapshot.heading_deg,
                half_fov=180.0,
            )
        )
        if not valid_directions:
            self._trace(
                "drone_recovery_no_outgoing_heading",
                target=target,
                position=snapshot.position,
                incoming_heading=snapshot.heading_deg,
            )
            return False

        chosen_direction = drone.exploration_policy.choose_direction(
            valid_directions
        )
        self._apply_reorientation(
            chosen_direction,
            step_targets,
            event="drone_recovery_reoriented",
            valid_direction_count=len(valid_directions),
            target=target,
        )
        return True

    def _apply_reorientation(
        self,
        chosen_direction: int,
        step_targets: dict[int, Position],
        *,
        event: str,
        valid_direction_count: int,
        **fields: Any,
    ) -> None:
        """Rotate in place, retain one exit border, and reset gain tracking."""
        drone = self.drone
        snapshot = drone.snapshot()
        drone.runtime_state.reorient(chosen_direction)
        chosen_border = next_cell_coords(
            *snapshot.position,
            drone.radius + 1,
            chosen_direction,
        )
        drone.runtime_state.merge_frontiers((chosen_border,))
        self._trace(
            event,
            position=snapshot.position,
            incoming_heading=snapshot.heading_deg,
            direction=chosen_direction,
            step_target=step_targets[chosen_direction],
            border_target=chosen_border,
            valid_direction_count=valid_direction_count,
            **fields,
        )
        self._reset_stagnation_window()

    def reach_start_point(self) -> bool:
        """Use A* to return to the drone's starting position."""
        drone = self.drone
        current = drone.snapshot().position
        if current == drone.start_pos:
            return True
        result = self._compute_path(current, drone.start_pos)
        path = result.path
        self._trace(
            "drone_homing_path",
            start=current,
            target=drone.start_pos,
            path_length=len(path),
            path_status=result.status,
            path_iterations=result.iterations,
            path_remaining_distance=result.remaining_distance,
            segment_endpoint=path[-1] if path else None,
        )
        if not path:
            return False
        if result.status == PATH_PARTIAL_LIMIT:
            if not self._accept_partial_endpoint(
                current,
                drone.start_pos,
                path[-1],
            ):
                return False
            self._follow_path(path, source="home_astar_partial")
            return False
        if result.status != PATH_COMPLETE:
            return False
        self._clear_partial_route(drone.start_pos)
        return self._follow_path(
            path,
            source="home_astar",
        ) and drone.snapshot().position == drone.start_pos

    def update_borders(self) -> None:
        """Refresh SLAM frontier targets when the cooldown permits."""
        self.maybe_rebuild_frontiers()

    def maybe_rebuild_frontiers(self) -> bool:
        """Rebuild frontiers at most once per configured cooldown."""
        now = self._simulation_time()
        if not self.drone.runtime_state.reserve_frontier_rebuild(now):
            return False
        self.rebuild_frontiers(
            stride=self.frontier_stride,
            confidence_threshold=self.frontier_confidence_threshold,
        )
        return True

    def rebuild_frontiers(
        self,
        *,
        stride: int = 4,
        confidence_threshold: float = 0.6,
    ) -> None:
        """Extract known-free cells bordering unknown local SLAM cells."""
        slam = self.drone.slam_map.snapshot(point_limit=0)
        occupancy = np.asarray(slam.occupancy)
        confidence = np.asarray(slam.confidence)
        threshold = float(confidence_threshold)
        known_free = (occupancy == FREE) & (confidence >= threshold)
        unknown = (occupancy == UNKNOWN) | (confidence < threshold)

        neighbor_unknown = self._neighbor_adjacency(unknown)

        frontier_mask = known_free & neighbor_unknown
        sampling_stride = max(1, int(stride))
        sampled = frontier_mask[::sampling_stride, ::sampling_stride]
        rows, columns = np.where(sampled)
        raw_frontiers = tuple(
            (
                int(column * sampling_stride),
                int(row * sampling_stride),
            )
            for row, column in zip(rows, columns)
        )
        raw_frontier_set = frozenset(raw_frontiers)
        reactivated: list[Position] = []
        for target, stored_geometry in tuple(
            self._suppressed_frontier_geometry.items()
        ):
            if target not in raw_frontier_set:
                self._suppressed_frontier_geometry.pop(target, None)
                continue
            current_geometry = self._frontier_geometry_signature(
                target,
                raw_frontier_set,
            )
            if stored_geometry is None:
                self._suppressed_frontier_geometry[target] = current_geometry
            elif current_geometry != stored_geometry:
                self._suppressed_frontier_geometry.pop(target, None)
                reactivated.append(target)

        suppressed = tuple(
            target
            for target in raw_frontiers
            if target in self._suppressed_frontier_geometry
        )
        frontiers = tuple(
            target
            for target in raw_frontiers
            if target not in self._suppressed_frontier_geometry
        )
        self._last_raw_frontiers = raw_frontier_set
        self._frontier_slam_version = int(slam.version)
        self.drone.runtime_state.replace_frontiers(frontiers)
        self.border_retry_until = {
            target: retry_until
            for target, retry_until in self.border_retry_until.items()
            if target in raw_frontier_set
        }
        self._trace(
            "drone_frontiers_rebuilt",
            frontier_count=len(frontiers),
            frontier_sample=frontiers[:12],
            raw_frontier_count=len(raw_frontiers),
            suppressed_frontier_count=len(suppressed),
            suppressed_frontier_sample=suppressed[:12],
            reactivated_frontier_count=len(reactivated),
            reactivated_frontier_sample=tuple(reactivated[:12]),
            slam_version=slam.version,
        )
        if self._global_frontier_cache is None:
            state = self.drone.snapshot()
            self._ensure_global_frontier_cache(
                current=state.position,
                heading=float(state.heading_deg),
                slam=slam,
            )

    def _frontier_geometry_signature(
        self,
        target: Position,
        frontiers: Iterable[Position],
    ) -> tuple[Position, ...]:
        """Describe sampled frontier geometry local to one reached target."""
        local_radius = max(
            float(self.drone.radius) * 2.0,
            float(self.frontier_stride) * 2.0,
        )
        radius_squared = local_radius * local_radius
        target_x, target_y = target
        offsets = (
            (point[0] - target_x, point[1] - target_y)
            for point in frontiers
            if (
                (point[0] - target_x) ** 2
                + (point[1] - target_y) ** 2
            ) <= radius_squared
        )
        return tuple(sorted(offsets))

    def mission_completed(self) -> bool:
        """Return whether the drone has completed exploration and homing."""
        done, _returning_home = self.drone.runtime_state.evaluate_mission_state()
        if done:
            logger.info("Drone %s has completed the mission", self.drone.id)
        return done

    def get_distance(self, target: Position) -> float:
        """Return the current border-priority distance for compatibility."""
        return self._distance_from(self.drone.snapshot().position, target)

    def _distance_from(self, position: Position, target: Position) -> float:
        distance = math.dist(position, target)
        if distance <= self.drone.radius:
            return float(self.drone.game.width)
        return distance

    def _compute_path(self, start: Position, goal: Position) -> PathResult:
        """Ask for a complete physical route or one capped route segment."""
        segment_planner = self.dependencies.compute_path_segment
        if callable(segment_planner):
            result = segment_planner(start, goal)
            if isinstance(result, PathResult):
                return result

        path = tuple(self.dependencies.compute_path(start, goal))
        status = (
            PATH_COMPLETE
            if path and path[-1] == goal
            else PATH_UNREACHABLE
        )
        remaining = (
            0.0
            if status == PATH_COMPLETE
            else math.dist(path[-1] if path else start, goal)
        )
        return PathResult(path, status, 0, remaining)

    def _accept_partial_endpoint(
        self,
        start: Position,
        goal: Position,
        endpoint: Position,
    ) -> bool:
        """Accept a capped segment only when it advances to a fresh endpoint."""
        advances = math.dist(endpoint, goal) < math.dist(start, goal) - 1e-6
        recent = self._partial_route_endpoints.get(goal, ())
        accepted = advances and endpoint not in recent
        self._trace(
            "drone_astar_partial_segment",
            start=start,
            goal=goal,
            endpoint=endpoint,
            accepted=accepted,
            advances=advances,
            repeated_endpoint=endpoint in recent,
            remaining_distance=math.dist(endpoint, goal),
        )
        if accepted:
            self._partial_route_endpoints[goal] = (*recent[-7:], endpoint)
        return accepted

    def _clear_partial_route(self, goal: Position) -> None:
        """Forget loop protection after completing or retiring a route."""
        self._partial_route_endpoints.pop(goal, None)

    def _simulation_time(self) -> float:
        return float(self.dependencies.simulation_time())

    def _follow_path(
        self,
        path: Iterable[Position],
        *,
        source: str = "path",
    ) -> bool:
        """Walk a path while recording the breadcrumb history used by rendering."""
        points = tuple((int(point[0]), int(point[1])) for point in path)
        started = self._simulation_time()
        start = self.drone.snapshot().position
        end = start
        moved_points = 0
        completed = True
        for node in points:
            if node == self.drone.snapshot().position:
                continue
            if not self.dependencies.pause_checkpoint():
                completed = False
                break
            previous = self.drone.snapshot().position
            if not self.drone.runtime_state.graph_is_valid(previous, node):
                completed = False
                break
            self.drone.runtime_state.move_to(node)
            end = node
            moved_points += 1
            if not self.dependencies.wait_simulation_delay(
                self.drone.delay / self.drone.speed_factor
            ):
                completed = False
                break

        travelled_distance = self._path_distance(
            start,
            points,
            moved_points,
        )
        self._stagnation_distance_travelled += travelled_distance
        self._trace(
            "drone_motion",
            source=source,
            completed=completed,
            start=start,
            end=end,
            point_count=moved_points,
            travelled_distance=travelled_distance,
            started_sim_time=started,
            ended_sim_time=self._simulation_time(),
        )
        return completed

    @staticmethod
    def _path_distance(
        start: Position,
        points: tuple[Position, ...],
        moved_points: int,
    ) -> float:
        if moved_points <= 0:
            return 0.0
        moved: list[Position] = [start]
        for point in points:
            if point == moved[-1]:
                continue
            moved.append(point)
            if len(moved) - 1 >= moved_points:
                break
        return sum(
            math.dist(previous, current)
            for previous, current in zip(moved, moved[1:])
        )

    def _trace(self, event: str, **fields: Any) -> None:
        trace = getattr(self.dependencies, "runtime_trace", None)
        if trace is not None:
            trace.record(
                event,
                sim_time=self._simulation_time(),
                drone_id=self.drone.id,
                **fields,
            )

    def _snapshot_summary(self) -> dict[str, Any]:
        snapshot = self.drone.snapshot()
        return {
            "position": snapshot.position,
            "direction": snapshot.direction,
            "frontier_count": len(snapshot.frontiers),
            "returning_home": snapshot.returning_home,
            "done": snapshot.done,
        }
