"""Belief-only frontier extraction, stable identity, and coordination.

The objects in this module deliberately know nothing about the ground-truth cave
map.  A frontier is derived only from a detached SLAM occupancy/confidence
snapshot.  Coordinates remain available as a compatibility view, while stable
integer cluster IDs are the authoritative identity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import threading
from typing import Iterable, Sequence

import numpy as np

from mapping.slam_map import FREE, OCCUPIED, SlamSnapshot


Position = tuple[int, int]
Bounds = tuple[int, int, int, int]
FrontierClusterId = int


@dataclass(frozen=True)
class FrontierMasks:
    """Canonical mutually exclusive SLAM belief masks."""

    free: np.ndarray
    occupied: np.ndarray
    unknown: np.ndarray
    frontier: np.ndarray


@dataclass(frozen=True)
class FrontierComponent:
    """One connected component from a single SLAM refresh."""

    cells: frozenset[Position]
    bounds: Bounds
    representative: Position
    expected_gain: int
    wall_gain: int = 0
    wall_cells: frozenset[Position] = frozenset()


@dataclass(frozen=True)
class FrontierExtraction:
    """Cached extraction result for one SLAM version."""

    version: int
    masks: FrontierMasks
    components: tuple[FrontierComponent, ...]
    full_rebuild: bool
    dirty_regions: tuple[Bounds, ...] = ()


def _shift_neighbor_mask(mask: np.ndarray) -> np.ndarray:
    """Return cells adjacent (8-connected) to any true cell in ``mask``."""
    height, width = mask.shape
    neighbors = np.zeros_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ys_src = slice(max(0, -dy), height - max(0, dy))
            ys_dst = slice(max(0, dy), height - max(0, -dy))
            xs_src = slice(max(0, -dx), width - max(0, dx))
            xs_dst = slice(max(0, dx), width - max(0, -dx))
            neighbors[ys_dst, xs_dst] |= mask[ys_src, xs_src]
    return neighbors


def _neighborhood_support(mask: np.ndarray) -> np.ndarray:
    """Count true cells in each clipped 3x3 neighborhood."""
    height, width = mask.shape
    support = mask.astype(np.uint8, copy=True)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ys_src = slice(max(0, -dy), height - max(0, dy))
            ys_dst = slice(max(0, dy), height - max(0, -dy))
            xs_src = slice(max(0, -dx), width - max(0, dx))
            xs_dst = slice(max(0, dx), width - max(0, -dx))
            support[ys_dst, xs_dst] += mask[ys_src, xs_src]
    return support


def _normalize_bounds(bounds: Sequence[int], shape: tuple[int, int]) -> Bounds:
    x0, y0, x1, y1 = (int(value) for value in bounds)
    height, width = shape
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def _expanded(bounds: Bounds, shape: tuple[int, int], halo: int = 1) -> Bounds:
    x0, y0, x1, y1 = bounds
    return _normalize_bounds((x0 - halo, y0 - halo, x1 + halo, y1 + halo), shape)


class FrontierExtractor:
    """Cache and incrementally update one drone's canonical frontier mask."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        *,
        large_update_ratio: float = 0.25,
        minimum_unknown_support: int = 4,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.large_update_ratio = max(0.0, min(1.0, float(large_update_ratio)))
        self.minimum_unknown_support = max(
            1, min(9, int(minimum_unknown_support))
        )
        self._lock = threading.RLock()
        self._result: FrontierExtraction | None = None
        self._occupancy: np.ndarray | None = None
        self._confidence: np.ndarray | None = None
        self._exploration_unknown: np.ndarray | None = None
        self._surface_unknown: np.ndarray | None = None
        self._wall_frontier: np.ndarray | None = None

    def _derive_frontiers(
        self,
        free: np.ndarray,
        occupied: np.ndarray,
        unknown: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Separate coherent discovery space from wall-surface continuation."""
        adjacent_free = _shift_neighbor_mask(free)
        adjacent_occupied = _shift_neighbor_mask(occupied)
        surface_unknown = unknown & adjacent_free & adjacent_occupied
        coherent_unknown = unknown & (
            _neighborhood_support(unknown) >= self.minimum_unknown_support
        )
        exploration_unknown = coherent_unknown | surface_unknown
        frontier = free & _shift_neighbor_mask(exploration_unknown)
        wall_frontier = frontier & _shift_neighbor_mask(surface_unknown)
        return (
            exploration_unknown,
            surface_unknown,
            frontier,
            wall_frontier,
        )

    def refresh(
        self,
        snapshot: SlamSnapshot,
        *,
        dirty_regions: Iterable[Bounds] | None = None,
        force_full: bool = False,
    ) -> FrontierExtraction:
        """Refresh once per SLAM version, using dirty regions plus a halo."""
        occupancy = np.asarray(snapshot.occupancy, dtype=np.int8)
        confidence = np.asarray(snapshot.confidence, dtype=np.float32)
        if occupancy.shape != confidence.shape:
            raise ValueError("SLAM occupancy and confidence shapes differ")

        with self._lock:
            if self._result is not None and self._result.version == snapshot.version:
                return self._result

            needs_full = (
                force_full
                or self._result is None
                or self._occupancy is None
                or self._occupancy.shape != occupancy.shape
            )
            regions: list[Bounds] = []
            if not needs_full:
                changed = (
                    (occupancy != self._occupancy)
                    | (np.abs(confidence - self._confidence) > 1e-6)
                )
                changed_count = int(np.count_nonzero(changed))
                if changed_count == 0:
                    self._result = replace(self._result, version=int(snapshot.version))
                    self._occupancy = occupancy.copy()
                    self._confidence = confidence.copy()
                    return self._result
                if changed_count > occupancy.size * self.large_update_ratio:
                    needs_full = True
                else:
                    ys, xs = np.where(changed)
                    regions.append((
                        int(xs.min()), int(ys.min()),
                        int(xs.max()) + 1, int(ys.max()) + 1,
                    ))
                    if dirty_regions is not None:
                        regions.extend(
                            _normalize_bounds(region, occupancy.shape)
                            for region in dirty_regions
                        )

            threshold = self.confidence_threshold
            confidently_known = confidence >= threshold
            free = confidently_known & (occupancy == FREE)
            occupied = confidently_known & (occupancy == OCCUPIED)
            unknown = ~(free | occupied)

            if needs_full:
                (
                    exploration_unknown,
                    surface_unknown,
                    frontier,
                    wall_frontier,
                ) = self._derive_frontiers(free, occupied, unknown)
                normalized_regions: tuple[Bounds, ...] = ()
            else:
                assert self._exploration_unknown is not None
                assert self._surface_unknown is not None
                assert self._wall_frontier is not None
                frontier = self._result.masks.frontier.copy()
                exploration_unknown = self._exploration_unknown.copy()
                surface_unknown = self._surface_unknown.copy()
                wall_frontier = self._wall_frontier.copy()
                normalized_regions = tuple(
                    _expanded(region, occupancy.shape, halo=2)
                    for region in regions
                )
                for x0, y0, x1, y1 in normalized_regions:
                    crop = _expanded(
                        (x0, y0, x1, y1), occupancy.shape, halo=2
                    )
                    crop_x0, crop_y0, crop_x1, crop_y1 = crop
                    local = self._derive_frontiers(
                        free[crop_y0:crop_y1, crop_x0:crop_x1],
                        occupied[crop_y0:crop_y1, crop_x0:crop_x1],
                        unknown[crop_y0:crop_y1, crop_x0:crop_x1],
                    )
                    offset_y = y0 - crop_y0
                    offset_x = x0 - crop_x0
                    local_slice = (
                        slice(offset_y, offset_y + (y1 - y0)),
                        slice(offset_x, offset_x + (x1 - x0)),
                    )
                    destination = (slice(y0, y1), slice(x0, x1))
                    exploration_unknown[destination] = local[0][local_slice]
                    surface_unknown[destination] = local[1][local_slice]
                    frontier[destination] = local[2][local_slice]
                    wall_frontier[destination] = local[3][local_slice]

            masks = FrontierMasks(free, occupied, unknown, frontier)
            result = FrontierExtraction(
                version=int(snapshot.version),
                masks=masks,
                components=self._components(
                    frontier,
                    exploration_unknown,
                    wall_frontier,
                    surface_unknown,
                ),
                full_rebuild=needs_full,
                dirty_regions=normalized_regions,
            )
            self._result = result
            self._occupancy = occupancy.copy()
            self._confidence = confidence.copy()
            self._exploration_unknown = exploration_unknown
            self._surface_unknown = surface_unknown
            self._wall_frontier = wall_frontier
            return result

    @staticmethod
    def _components(
        frontier: np.ndarray,
        exploration_unknown: np.ndarray,
        wall_frontier: np.ndarray,
        surface_unknown: np.ndarray,
    ) -> tuple[FrontierComponent, ...]:
        """Extract deterministic 8-connected components from the cached mask."""
        height, width = frontier.shape
        remaining = frontier.copy()
        components: list[FrontierComponent] = []
        for y in range(height):
            for x in range(width):
                if not remaining[y, x]:
                    continue
                remaining[y, x] = False
                stack = [(x, y)]
                cells: list[Position] = []
                while stack:
                    cx, cy = stack.pop()
                    cells.append((cx, cy))
                    for ny in range(max(0, cy - 1), min(height, cy + 2)):
                        for nx in range(max(0, cx - 1), min(width, cx + 2)):
                            if remaining[ny, nx]:
                                remaining[ny, nx] = False
                                stack.append((nx, ny))
                frozen = frozenset(cells)
                min_x = min(point[0] for point in cells)
                min_y = min(point[1] for point in cells)
                max_x = max(point[0] for point in cells)
                max_y = max(point[1] for point in cells)
                wall_cells = frozenset(
                    point for point in cells if wall_frontier[point[1], point[0]]
                )
                representative_cells = tuple(wall_cells) or tuple(cells)
                centroid = (
                    sum(point[0] for point in representative_cells)
                    / len(representative_cells),
                    sum(point[1] for point in representative_cells)
                    / len(representative_cells),
                )
                representative = min(
                    representative_cells,
                    key=lambda point: (
                        math.dist(point, centroid), point[1], point[0]
                    ),
                )
                gain_cells: set[Position] = set()
                wall_gain_cells: set[Position] = set()
                for cx, cy in cells:
                    for ny in range(max(0, cy - 1), min(height, cy + 2)):
                        for nx in range(max(0, cx - 1), min(width, cx + 2)):
                            if exploration_unknown[ny, nx]:
                                gain_cells.add((nx, ny))
                            if surface_unknown[ny, nx]:
                                wall_gain_cells.add((nx, ny))
                components.append(FrontierComponent(
                    cells=frozen,
                    bounds=(min_x, min_y, max_x + 1, max_y + 1),
                    representative=representative,
                    expected_gain=len(gain_cells),
                    wall_gain=len(wall_gain_cells),
                    wall_cells=wall_cells,
                ))
        return tuple(components)


@dataclass(frozen=True)
class FrontierCluster:
    """Detached canonical cluster state, filtered by registry visibility."""

    id: FrontierClusterId
    cells: frozenset[Position]
    bounds: Bounds
    representative: Position
    expected_gain: int
    known_by: frozenset[int]
    lifecycle: str
    first_seen_revision: int
    last_seen_revision: int
    wall_gain: int = 0
    wall_cells: frozenset[Position] = frozenset()
    missing_refresh_count: int = 0
    gateway_id: int | None = None
    revisit_penalty: float = 0.0
    stall_penalty: float = 0.0
    zero_gain_penalty: float = 0.0


def select_accessible_frontier_waypoint(
    cluster: FrontierCluster,
    known_free: np.ndarray,
    *,
    origin: Position,
    minimum_distance: float = 0.0,
    unknown: np.ndarray | None = None,
) -> Position | None:
    """Choose one informative requester-known-free cell for a stable cluster.

    Cluster representatives are canonical matching/scoring geometry, not a
    command that every drone must reach.  A shared cluster can therefore keep
    its stable ID while each requester selects a cell supported by its own
    SLAM. Retained wall work may require a minimum displacement so a shifting
    one-pixel boundary is not rescanned from effectively the same pose.
    """
    mask = np.asarray(known_free)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise TypeError("known_free must be a two-dimensional boolean mask")
    minimum = float(minimum_distance)
    if minimum < 0.0:
        raise ValueError("minimum_distance must be non-negative")
    unknown_mask = None
    if unknown is not None:
        unknown_mask = np.asarray(unknown)
        if unknown_mask.shape != mask.shape or unknown_mask.dtype != np.bool_:
            raise TypeError(
                "unknown must be a boolean mask matching known_free"
            )
    preferred_cells = cluster.wall_cells if cluster.wall_gain else cluster.cells
    candidates = []
    for point in preferred_cells:
        x, y = point
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
            candidates.append((int(x), int(y)))
    if not candidates and cluster.wall_gain:
        for point in cluster.cells:
            x, y = point
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                candidates.append((int(x), int(y)))
    if not candidates:
        return None
    candidates = [
        point for point in candidates
        if math.dist(origin, point) + 1e-9 >= minimum
    ]
    if not candidates:
        return None

    def local_unknown_support(point: Position) -> int:
        if unknown_mask is None or not cluster.wall_gain:
            return 0
        x, y = point
        radius = 4
        x0 = max(0, x - radius)
        x1 = min(unknown_mask.shape[1], x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(unknown_mask.shape[0], y + radius + 1)
        return int(np.count_nonzero(unknown_mask[y0:y1, x0:x1]))

    return min(
        candidates,
        key=lambda point: (
            -local_unknown_support(point),
            math.dist(origin, point),
            math.dist(cluster.representative, point),
            point[1],
            point[0],
        ),
    )


@dataclass(frozen=True)
class FrontierLifecycleEvent:
    cluster_id: FrontierClusterId
    lifecycle: str
    reason: str


class FrontierClusterRegistry:
    """Canonicalize clusters while retaining per-drone knowledge boundaries."""

    def __init__(
        self,
        *,
        match_distance: float = 32.0,
        missing_refresh_limit: int = 3,
    ) -> None:
        self.match_distance = float(match_distance)
        self.missing_refresh_limit = max(0, int(missing_refresh_limit))
        self._lock = threading.RLock()
        self._next_id = 1
        self._clusters: dict[int, FrontierCluster] = {}
        self._views: dict[tuple[int, int], FrontierCluster] = {}
        self._missing: dict[tuple[int, int], int] = {}
        self._tombstones: set[frozenset[Position]] = set()
        self._events: list[FrontierLifecycleEvent] = []

    def refresh(
        self,
        drone_id: int,
        components: Iterable[FrontierComponent],
        *,
        slam_version: int,
    ) -> tuple[FrontierCluster, ...]:
        """Match one drone's refreshed components to stable canonical IDs."""
        observer = int(drone_id)
        refreshed = tuple(components)
        with self._lock:
            seen: set[int] = set()
            for component in refreshed:
                if component.cells in self._tombstones:
                    continue
                cluster_id = self._best_match(component, exclude=seen)
                if cluster_id is None:
                    cluster_id = self._next_id
                    self._next_id += 1
                    cluster = FrontierCluster(
                        id=cluster_id,
                        cells=component.cells,
                        bounds=component.bounds,
                        representative=component.representative,
                        expected_gain=component.expected_gain,
                        known_by=frozenset({observer}),
                        lifecycle="active",
                        first_seen_revision=int(slam_version),
                        last_seen_revision=int(slam_version),
                        wall_gain=component.wall_gain,
                        wall_cells=component.wall_cells,
                    )
                else:
                    previous = self._clusters[cluster_id]
                    cluster = replace(
                        previous,
                        cells=component.cells,
                        bounds=component.bounds,
                        representative=component.representative,
                        expected_gain=component.expected_gain,
                        wall_gain=component.wall_gain,
                        wall_cells=component.wall_cells,
                        known_by=previous.known_by | {observer},
                        lifecycle="active",
                        last_seen_revision=int(slam_version),
                    )
                self._clusters[cluster_id] = cluster
                self._views[(cluster_id, observer)] = cluster
                self._missing[(cluster_id, observer)] = 0
                seen.add(cluster_id)

            known_ids = tuple(
                cluster.id for cluster in self._clusters.values()
                if observer in cluster.known_by and cluster.lifecycle != "retired"
            )
            for cluster_id in known_ids:
                if cluster_id in seen:
                    continue
                key = (cluster_id, observer)
                count = self._missing.get(key, 0) + 1
                self._missing[key] = count
                if count > self.missing_refresh_limit:
                    cluster = self._clusters[cluster_id]
                    remaining = cluster.known_by - {observer}
                    lifecycle = "active" if remaining else "retired"
                    self._clusters[cluster_id] = replace(
                        cluster, known_by=remaining, lifecycle=lifecycle,
                    )
                    if lifecycle == "retired":
                        self._tombstones.add(cluster.cells)
                        self._events.append(FrontierLifecycleEvent(
                            cluster_id, "retired", "missing"
                        ))
            return self.visible_to(observer)

    def _best_match(
        self,
        component: FrontierComponent,
        *,
        exclude: set[int],
    ) -> int | None:
        candidates: list[tuple[float, int]] = []
        for cluster in self._clusters.values():
            if cluster.id in exclude or cluster.lifecycle == "retired":
                continue
            overlap = len(component.cells & cluster.cells)
            overlap_ratio = overlap / max(1, min(len(component.cells), len(cluster.cells)))
            distance = math.dist(component.representative, cluster.representative)
            bounds_overlap = self._bounds_overlap(
                self._expand_bounds(component.bounds, self.match_distance),
                cluster.bounds,
            )
            if overlap_ratio >= 0.25 or (
                distance <= self.match_distance and bounds_overlap
            ):
                candidates.append((-overlap_ratio, distance, cluster.id))
        return min(candidates)[2] if candidates else None

    @staticmethod
    def _expand_bounds(bounds: Bounds, distance: float) -> Bounds:
        amount = int(math.ceil(distance))
        x0, y0, x1, y1 = bounds
        return (x0 - amount, y0 - amount, x1 + amount, y1 + amount)

    @staticmethod
    def _bounds_overlap(a: Bounds, b: Bounds) -> bool:
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    def visible_to(self, drone_id: int) -> tuple[FrontierCluster, ...]:
        """Return only clusters explicitly known by ``drone_id``."""
        observer = int(drone_id)
        with self._lock:
            visible = []
            for cluster in self._clusters.values():
                if observer not in cluster.known_by or cluster.lifecycle == "retired":
                    continue
                missing = self._missing.get((cluster.id, observer), 0)
                local_view = self._views.get((cluster.id, observer), cluster)
                visible.append(replace(
                    local_view,
                    known_by=cluster.known_by,
                    lifecycle=cluster.lifecycle,
                    gateway_id=cluster.gateway_id,
                    missing_refresh_count=missing,
                    revisit_penalty=cluster.revisit_penalty,
                    stall_penalty=cluster.stall_penalty,
                    zero_gain_penalty=cluster.zero_gain_penalty,
                ))
            return tuple(sorted(visible, key=lambda cluster: cluster.id))

    def share(self, source_drone_id: int, target_drone_id: int) -> tuple[int, ...]:
        """Explicitly transfer stable cluster knowledge across communication."""
        source = int(source_drone_id)
        target = int(target_drone_id)
        with self._lock:
            transferred = []
            for cluster in tuple(self._clusters.values()):
                if source not in cluster.known_by or cluster.lifecycle == "retired":
                    continue
                if target not in cluster.known_by:
                    self._clusters[cluster.id] = replace(
                        cluster, known_by=cluster.known_by | {target}
                    )
                    source_view = self._views.get((cluster.id, source), cluster)
                    self._views[(cluster.id, target)] = source_view
                    self._missing[(cluster.id, target)] = 0
                    transferred.append(cluster.id)
            return tuple(transferred)

    def retire(self, cluster_id: int, *, reason: str) -> bool:
        """Retire and tombstone a cluster with an explicit lifecycle event."""
        with self._lock:
            cluster = self._clusters.get(int(cluster_id))
            if cluster is None or cluster.lifecycle == "retired":
                return False
            self._clusters[cluster.id] = replace(
                cluster,
                lifecycle="retired",
                known_by=frozenset(),
                zero_gain_penalty=(
                    cluster.zero_gain_penalty + 1.0
                    if reason == "zero_gain" else cluster.zero_gain_penalty
                ),
            )
            self._tombstones.add(cluster.cells)
            self._events.append(FrontierLifecycleEvent(
                cluster.id, "retired", str(reason)
            ))
            return True

    def set_gateway(self, cluster_id: int, gateway_id: int) -> None:
        """Associate one protected graph gateway with the canonical cluster."""
        with self._lock:
            cluster = self._clusters[int(cluster_id)]
            self._clusters[cluster.id] = replace(cluster, gateway_id=int(gateway_id))

    def clear_gateway(self, cluster_id: int, gateway_id: int | None = None) -> bool:
        """Clear a gateway association when its graph role has been retired."""
        with self._lock:
            cluster = self._clusters.get(int(cluster_id))
            if cluster is None or cluster.gateway_id is None:
                return False
            if gateway_id is not None and cluster.gateway_id != int(gateway_id):
                return False
            self._clusters[cluster.id] = replace(cluster, gateway_id=None)
            return True

    def penalize(
        self,
        cluster_id: int,
        *,
        revisit: float = 0.0,
        stall: float = 0.0,
        zero_gain: float = 0.0,
    ) -> bool:
        """Atomically accumulate deterministic strategic scoring penalties."""
        with self._lock:
            cluster = self._clusters.get(int(cluster_id))
            if cluster is None:
                return False
            self._clusters[cluster.id] = replace(
                cluster,
                revisit_penalty=(
                    cluster.revisit_penalty + max(0.0, float(revisit))
                ),
                stall_penalty=(
                    cluster.stall_penalty + max(0.0, float(stall))
                ),
                zero_gain_penalty=(
                    cluster.zero_gain_penalty + max(0.0, float(zero_gain))
                ),
            )
            return True

    def get(self, cluster_id: int) -> FrontierCluster:
        """Return detached canonical state for coordination internals/tests."""
        with self._lock:
            return self._clusters[int(cluster_id)]

    def lifecycle_events(self) -> tuple[FrontierLifecycleEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def canonical_clusters(self) -> tuple[FrontierCluster, ...]:
        """Return detached canonical state for coordination services."""
        with self._lock:
            return tuple(self._clusters.values())


@dataclass(frozen=True)
class FrontierAssignment:
    token: int
    cluster_id: int
    drone_id: int
    gateway_id: int | None = None


class AssignmentRegistry:
    """Atomic deterministic reservation ownership for clusters/gateways."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_token = 1
        self._by_cluster: dict[int, FrontierAssignment] = {}
        self._by_gateway: dict[int, FrontierAssignment] = {}
        self._by_token: dict[int, FrontierAssignment] = {}

    def reserve(
        self,
        *,
        cluster_id: int,
        drone_id: int,
        gateway_id: int | None = None,
    ) -> FrontierAssignment | None:
        """Atomically reserve if neither canonical goal key is owned."""
        cluster_key = int(cluster_id)
        gateway_key = None if gateway_id is None else int(gateway_id)
        owner = int(drone_id)
        with self._lock:
            existing = self._by_cluster.get(cluster_key)
            if existing is not None:
                return existing if existing.drone_id == owner else None
            if gateway_key is not None:
                existing = self._by_gateway.get(gateway_key)
                if existing is not None:
                    return existing if existing.drone_id == owner else None
            assignment = FrontierAssignment(
                token=self._next_token,
                cluster_id=cluster_key,
                drone_id=owner,
                gateway_id=gateway_key,
            )
            self._next_token += 1
            self._by_cluster[cluster_key] = assignment
            if gateway_key is not None:
                self._by_gateway[gateway_key] = assignment
            self._by_token[assignment.token] = assignment
            return assignment

    def release(self, token: int, *, drone_id: int) -> bool:
        """Release only when token and owner both match."""
        with self._lock:
            assignment = self._by_token.get(int(token))
            if assignment is None or assignment.drone_id != int(drone_id):
                return False
            self._by_token.pop(assignment.token, None)
            self._by_cluster.pop(assignment.cluster_id, None)
            if assignment.gateway_id is not None:
                self._by_gateway.pop(assignment.gateway_id, None)
            return True

    def attach_gateway(
        self,
        token: int,
        *,
        drone_id: int,
        gateway_id: int,
    ) -> FrontierAssignment | None:
        """Atomically extend an owned cluster reservation to its lazy gateway."""
        with self._lock:
            assignment = self._by_token.get(int(token))
            gateway_key = int(gateway_id)
            conflict = self._by_gateway.get(gateway_key)
            if (
                assignment is None
                or assignment.drone_id != int(drone_id)
                or (conflict is not None and conflict.token != assignment.token)
            ):
                return None
            updated = replace(assignment, gateway_id=gateway_key)
            self._by_token[updated.token] = updated
            self._by_cluster[updated.cluster_id] = updated
            self._by_gateway[gateway_key] = updated
            return updated

    def assignment_for_cluster(self, cluster_id: int) -> FrontierAssignment | None:
        with self._lock:
            return self._by_cluster.get(int(cluster_id))

    def assignment_for_token(self, token: int) -> FrontierAssignment | None:
        """Return an assignment only while its exact token remains active."""
        with self._lock:
            return self._by_token.get(int(token))

    def release_cluster(self, cluster_id: int) -> FrontierAssignment | None:
        """Release a canonical goal after registry retirement."""
        with self._lock:
            assignment = self._by_cluster.get(int(cluster_id))
            if assignment is None:
                return None
            self._by_token.pop(assignment.token, None)
            self._by_cluster.pop(assignment.cluster_id, None)
            if assignment.gateway_id is not None:
                self._by_gateway.pop(assignment.gateway_id, None)
            return assignment

    def reconcile_active_clusters(
        self,
        active_cluster_ids: Iterable[int],
    ) -> tuple[FrontierAssignment, ...]:
        """Release reservations whose canonical clusters are no longer active."""
        active = {int(cluster_id) for cluster_id in active_cluster_ids}
        with self._lock:
            retired = tuple(
                assignment
                for cluster_id, assignment in tuple(self._by_cluster.items())
                if cluster_id not in active
            )
            for assignment in retired:
                self._by_token.pop(assignment.token, None)
                self._by_cluster.pop(assignment.cluster_id, None)
                if assignment.gateway_id is not None:
                    self._by_gateway.pop(assignment.gateway_id, None)
            return retired


class FrontierGatewayManager:
    """Create and reuse graph gateways only for their canonical cluster."""

    def __init__(
        self,
        registry: FrontierClusterRegistry,
        waypoint_graph: object,
        *,
        minimum_separation: float = 64.0,
    ) -> None:
        self.registry = registry
        self.waypoint_graph = waypoint_graph
        self.minimum_separation = max(0.0, float(minimum_separation))
        self._lock = threading.RLock()

    def ensure_gateway(
        self,
        cluster_id: int,
        known_free: np.ndarray,
        *,
        requester_id: int | None = None,
        position: Position | None = None,
    ) -> int | None:
        """Adopt a required existing corridor endpoint; never create one."""
        from navigation.waypoint_graph import WaypointRole

        with self._lock:
            cluster = self.registry.get(cluster_id)
            if requester_id is not None:
                cluster = next(
                    (
                        view for view in self.registry.visible_to(requester_id)
                        if view.id == int(cluster_id)
                    ),
                    cluster,
                )
            nodes = {node.id: node for node in self.waypoint_graph.snapshot().nodes}
            if cluster.gateway_id is not None:
                node = nodes.get(cluster.gateway_id)
                if (
                    node is not None
                    and WaypointRole.FRONTIER_GATEWAY in node.roles
                    and self._known_free(node.position, known_free)
                ):
                    return node.id
                return None

            target = cluster.representative if position is None else (
                int(position[0]), int(position[1])
            )
            gateway = next((
                node for node in nodes.values()
                if node.position == target
                and WaypointRole.FRONTIER_GATEWAY in node.roles
                and self._known_free(node.position, known_free)
            ), None)
            if gateway is None:
                return None
            self.registry.set_gateway(cluster.id, gateway.id)
            return gateway.id

    def retire_gateway(
        self,
        cluster_id: int,
        *,
        active_route_node_ids: Iterable[int] = (),
        active_route_edge_ids: Iterable[int] = (),
    ) -> object:
        """Retire an orphan cluster corridor while respecting active routes."""
        cluster = self.registry.get(cluster_id)
        if cluster.gateway_id is None:
            from navigation.waypoint_graph import GraphDelta
            return GraphDelta(revision=self.waypoint_graph.topology_revision)
        delta = self.waypoint_graph.retire_frontier_gateway(
            cluster.gateway_id,
            active_route_node_ids=active_route_node_ids,
            active_route_edge_ids=active_route_edge_ids,
        )
        node = next((
            item for item in self.waypoint_graph.snapshot().nodes
            if item.id == cluster.gateway_id
        ), None)
        from navigation.waypoint_graph import WaypointRole
        if node is None or WaypointRole.FRONTIER_GATEWAY not in node.roles:
            self.registry.clear_gateway(cluster.id, cluster.gateway_id)
        return delta

    @staticmethod
    def _known_free(position: Position, known_free: np.ndarray) -> bool:
        mask = np.asarray(known_free)
        if mask.ndim != 2 or mask.dtype != np.bool_:
            raise TypeError("known_free must be a two-dimensional boolean mask")
        x, y = position
        return bool(
            0 <= y < mask.shape[0]
            and 0 <= x < mask.shape[1]
            and mask[y, x]
        )
