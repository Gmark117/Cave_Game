"""Sparse, SLAM-safe waypoint routing for long drone journeys.

The graph deliberately has no cave-map dependency.  Persistent connections are
created only from paths that a drone actually travelled or from paths validated
against a boolean ``known_free`` mask supplied by SLAM.  Travelled edges remain
trusted; SLAM-derived edges are revalidated against every route requester's
current mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
import threading
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Position = Tuple[int, int]
WaypointPath = Tuple[Position, ...]

EDGE_TRAVELLED = "travelled"
EDGE_SLAM_LOS = "slam_los"
EDGE_KNOWN_FREE_CORRIDOR = "known_free_corridor"
EDGE_KNOWN_FREE_CONNECTOR = "known_free_connector"

ROUTE_OK = "ok"
ROUTE_START_UNKNOWN = "start_unknown"
ROUTE_GOAL_UNKNOWN = "goal_unknown"
ROUTE_NO_START_CONNECTOR = "no_start_connector"
ROUTE_NO_GOAL_CONNECTOR = "no_goal_connector"
ROUTE_DISCONNECTED = "disconnected"

LOCAL_CONNECTOR_MAX_ITERATIONS = 4_000
LOCAL_CONNECTOR_TIME_BUDGET_SECONDS = 0.012


@dataclass(frozen=True)
class WaypointNode:
    """One persistent waypoint and the source that first registered it."""

    position: Position
    source: str


@dataclass(frozen=True)
class WaypointEdge:
    """One undirected persistent edge stored in ``start`` -> ``end`` order."""

    start: Position
    end: Position
    cost: float
    source: str
    path: WaypointPath


@dataclass(frozen=True)
class GraphUpdate:
    """Detached record of the graph objects created by one operation."""

    status: str = ROUTE_OK
    added_waypoints: Tuple[WaypointNode, ...] = ()
    added_edges: Tuple[WaypointEdge, ...] = ()
    sampled_waypoints: Tuple[Position, ...] = ()

    @property
    def added_nodes(self) -> Tuple[WaypointNode, ...]:
        """Alias useful to callers that use generic graph terminology."""
        return self.added_waypoints

    @property
    def changed(self) -> bool:
        """Return whether this operation added or improved graph state."""
        return bool(self.added_waypoints or self.added_edges)


@dataclass(frozen=True)
class WaypointGraphSnapshot:
    """Immutable, consistently ordered view of a graph version."""

    waypoints: Tuple[WaypointNode, ...]
    edges: Tuple[WaypointEdge, ...]
    version: int = 0

    @property
    def nodes(self) -> Tuple[WaypointNode, ...]:
        """Alias for callers that use generic graph terminology."""
        return self.waypoints

    @property
    def node_count(self) -> int:
        """Return the number of persistent waypoints in this snapshot."""
        return len(self.waypoints)

    @property
    def edge_count(self) -> int:
        """Return the number of persistent edges in this snapshot."""
        return len(self.edges)


@dataclass(frozen=True)
class WaypointRoute:
    """Result of a route search through persistent and ephemeral edges."""

    status: str
    waypoints: Tuple[Position, ...] = ()
    first_segment_path: WaypointPath = ()
    first_segment_source: Optional[str] = None
    first_segment_cost: float = math.inf
    cost: float = math.inf

    @property
    def found(self) -> bool:
        """Return whether this result contains a usable route."""
        return self.status == ROUTE_OK


@dataclass(frozen=True)
class _RouteEdge:
    """An oriented edge used only by one Dijkstra invocation."""

    destination: object
    cost: float
    source: str
    path: WaypointPath


def _position(value: Sequence[int]) -> Position:
    """Normalize a two-coordinate position into integer map coordinates."""
    if len(value) != 2:
        raise ValueError("waypoint positions must have exactly two coordinates")
    return (int(value[0]), int(value[1]))


def _known_free_array(known_free: np.ndarray) -> np.ndarray:
    """Validate the explicit boolean-mask contract used by this module.

    Requiring a boolean dtype is intentional: a uint8 cave map uses the inverse
    convention (zero is free), and accepting one could make wall cells look like
    trusted free-space observations.
    """
    mask = np.asarray(known_free)
    if mask.ndim != 2:
        raise ValueError("known_free must be a two-dimensional boolean mask")
    if mask.dtype != np.bool_:
        raise TypeError("known_free must have boolean dtype")
    return mask


def _cell_is_known_free(mask: np.ndarray, position: Position) -> bool:
    """Return whether a coordinate is inside and true in ``mask``."""
    x, y = position
    return (
        0 <= y < mask.shape[0]
        and 0 <= x < mask.shape[1]
        and bool(mask[y, x])
    )


def bresenham_path(start: Sequence[int], goal: Sequence[int]) -> WaypointPath:
    """Return an inclusive, ordered 8-connected Bresenham raster line."""
    x0, y0 = _position(start)
    x1, y1 = _position(goal)
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    points: List[Position] = []

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return tuple(points)
        doubled_error = 2 * error
        if doubled_error >= dy:
            error += dy
            x0 += sx
        if doubled_error <= dx:
            error += dx
            y0 += sy


def _densify_polyline(path: Iterable[Sequence[int]]) -> WaypointPath:
    """Rasterize every coarse polyline leg, preserving its orientation."""
    coarse = tuple(_position(point) for point in path)
    if not coarse:
        return ()

    dense: List[Position] = [coarse[0]]
    for start, goal in zip(coarse, coarse[1:]):
        leg = bresenham_path(start, goal)
        dense.extend(leg[1:])
    return tuple(dense)


def validate_known_free_path(
    path: Iterable[Sequence[int]],
    known_free: np.ndarray,
) -> bool:
    """Validate an oriented polyline against known free space.

    Coarse input is first rasterized.  A diagonal move is permitted only if at
    least one of its two orthogonally adjacent cells is also known free, matching
    the project's A* corner rule and rejecting a squeeze between two obstacles.
    """
    mask = _known_free_array(known_free)
    dense = _densify_polyline(path)
    if not dense:
        return False

    for point in dense:
        if not _cell_is_known_free(mask, point):
            return False

    for previous, current in zip(dense, dense[1:]):
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        if abs(dx) > 1 or abs(dy) > 1:
            return False
        if dx != 0 and dy != 0:
            side_a = (current[0], previous[1])
            side_b = (previous[0], current[1])
            if not (
                _cell_is_known_free(mask, side_a)
                or _cell_is_known_free(mask, side_b)
            ):
                return False
    return True


def known_free_path(
    start: Sequence[int],
    goal: Sequence[int],
    known_free: np.ndarray,
) -> WaypointPath:
    """Build a Bresenham path, returning empty when SLAM cannot validate it."""
    path = bresenham_path(start, goal)
    if validate_known_free_path(path, known_free):
        return path
    return ()


def _bounded_known_free_path_to_any(
    start: Position,
    goals: Sequence[Position],
    known_free: np.ndarray,
    search_distance: float,
) -> WaypointPath:
    """Find a local SLAM-only path from ``start`` to any nearby goal.

    A direct visibility check handles the common open-cavern case cheaply.
    The bounded A* fallback handles bends and chokepoints without ever
    expanding into unknown cells or turning into a full-map path search.
    """
    mask = _known_free_array(known_free)
    deadline = (
        time.perf_counter() + LOCAL_CONNECTOR_TIME_BUDGET_SECONDS
    )
    ordered_goals = tuple(
        sorted(
            set(goals),
            key=lambda goal: (math.dist(start, goal), goal),
        )
    )
    for goal in ordered_goals:
        if time.perf_counter() >= deadline:
            break
        direct = known_free_path(start, goal, mask)
        if direct:
            return direct

    # A bounded set keeps the per-cell heuristic cheap while retaining several
    # alternatives when the nearest highway node lies across a wall.
    search_goals = ordered_goals[:32]
    goal_set = set(search_goals)
    if not goal_set:
        return ()

    def heuristic(position: Position) -> float:
        return min(math.dist(position, goal) for goal in search_goals)

    radius_squared = float(search_distance) ** 2
    # The circle-area bound keeps a malformed or unexpectedly open mask from
    # monopolising a movement tick. It is deliberately local, unlike the
    # simulator's full-resolution A*.
    max_iterations = max(
        1,
        min(
            LOCAL_CONNECTOR_MAX_ITERATIONS,
            int(math.ceil(math.pi * radius_squared)),
        ),
    )
    sequence = itertools.count()
    costs: Dict[Position, float] = {start: 0.0}
    previous: Dict[Position, Position] = {}
    heap: List[Tuple[float, float, int, Position]] = [
        (heuristic(start), 0.0, next(sequence), start)
    ]
    iterations = 0
    neighbor_steps = (
        (-1, -1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, 1, 1.0),
        (1, 1, math.sqrt(2.0)),
    )

    reached: Position | None = None
    while heap and iterations < max_iterations:
        if iterations % 64 == 0 and time.perf_counter() >= deadline:
            break
        _estimate, cost, _order, current = heapq.heappop(heap)
        if cost > costs.get(current, math.inf) + 1e-9:
            continue
        iterations += 1
        if current in goal_set:
            reached = current
            break

        for dx, dy, step_cost in neighbor_steps:
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                (neighbor[0] - start[0]) ** 2
                + (neighbor[1] - start[1]) ** 2
                > radius_squared
            ):
                continue
            if not _cell_is_known_free(mask, neighbor):
                continue
            if dx != 0 and dy != 0 and not (
                _cell_is_known_free(mask, (neighbor[0], current[1]))
                or _cell_is_known_free(mask, (current[0], neighbor[1]))
            ):
                continue
            candidate_cost = cost + step_cost
            if candidate_cost + 1e-9 >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = candidate_cost
            previous[neighbor] = current
            heapq.heappush(
                heap,
                (
                    candidate_cost + heuristic(neighbor),
                    candidate_cost,
                    next(sequence),
                    neighbor,
                ),
            )

    if reached is None:
        return ()

    reversed_path = [reached]
    current = reached
    while current != start:
        current = previous[current]
        reversed_path.append(current)
    return tuple(reversed(reversed_path))


def _path_cost(path: Sequence[Position]) -> float:
    """Return Euclidean step cost along an ordered dense polyline."""
    return sum(
        math.hypot(end[0] - start[0], end[1] - start[1])
        for start, end in zip(path, path[1:])
    )


def _join_paths(*paths: WaypointPath) -> WaypointPath:
    """Join consistently oriented paths without duplicate boundary cells."""
    result: List[Position] = []
    for path in paths:
        if not path:
            continue
        if result and result[-1] == path[0]:
            result.extend(path[1:])
        else:
            result.extend(path)
    return tuple(result)


class WaypointGraph:
    """Thread-safe sparse graph backed only by travelled and SLAM-safe edges."""

    def __init__(
        self,
        spacing: float = 32.0,
        merge_radius: float = 4.0,
        connector_distance: float = 64.0,
        connector_limit: int = 8,
    ) -> None:
        """Create an empty graph with bounded sampling and connector policies."""
        if float(spacing) <= 0.0:
            raise ValueError("spacing must be positive")
        if float(merge_radius) < 0.0:
            raise ValueError("merge_radius cannot be negative")
        if float(merge_radius) >= float(spacing):
            raise ValueError("merge_radius must be smaller than spacing")
        if float(connector_distance) < 0.0:
            raise ValueError("connector_distance cannot be negative")
        if int(connector_limit) < 1:
            raise ValueError("connector_limit must be at least one")

        self.spacing = float(spacing)
        self.merge_radius = float(merge_radius)
        self.connector_distance = float(connector_distance)
        self.connector_limit = int(connector_limit)
        self._lock = threading.RLock()
        self._nodes: Dict[Position, WaypointNode] = {}
        self._edges: Dict[Tuple[Position, Position, str], WaypointEdge] = {}
        self._version = 0

    @property
    def version(self) -> int:
        """Return the monotonic graph revision used by render caches."""
        with self._lock:
            return self._version

    @property
    def node_count(self) -> int:
        """Return the current persistent waypoint count atomically."""
        with self._lock:
            return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Return the current persistent edge count atomically."""
        with self._lock:
            return len(self._edges)

    def counts(self) -> Tuple[int, int]:
        """Return node and edge counts from one lock acquisition."""
        with self._lock:
            return (len(self._nodes), len(self._edges))

    def snapshot(self) -> WaypointGraphSnapshot:
        """Return a detached, deterministic snapshot suitable for tracing."""
        with self._lock:
            waypoints = tuple(
                self._nodes[position]
                for position in sorted(self._nodes)
            )
            edges = tuple(self._edges[key] for key in sorted(self._edges))
            version = self._version
        return WaypointGraphSnapshot(
            waypoints=waypoints,
            edges=edges,
            version=version,
        )

    def add_waypoint(
        self,
        position: Sequence[int],
        source: str = "home",
        known_free: Optional[np.ndarray] = None,
    ) -> Tuple[Position, bool]:
        """Add or safely merge a waypoint, returning canonical position/added.

        Without a mask, only an exact duplicate is merged.  A nearby waypoint is
        reused only when a boolean known-free mask validates line of sight from
        the requested position to that existing waypoint.
        """
        normalized = _position(position)
        mask = None if known_free is None else _known_free_array(known_free)
        with self._lock:
            canonical, node = self._resolve_or_add_waypoint_locked(
                normalized,
                str(source),
                mask,
            )
        return (canonical, node is not None)

    def register_travelled_path(
        self,
        path: Iterable[Sequence[int]],
        known_free: Optional[np.ndarray] = None,
    ) -> GraphUpdate:
        """Sample a travelled polyline and add oriented trusted edge segments.

        Coarse successive positions are Bresenham-densified before cumulative
        distance sampling.  Both original endpoints are always retained.
        ``known_free`` is optional and is used only to permit safe nearby-node
        merging; the registered trajectory itself remains authoritative.
        """
        dense = _densify_polyline(path)
        if not dense:
            return GraphUpdate(status="empty_path")
        mask = None if known_free is None else _known_free_array(known_free)
        sample_indices = self._sample_indices(dense)

        added_waypoints: List[WaypointNode] = []
        added_edges: List[WaypointEdge] = []
        canonical_samples: List[Position] = []

        with self._lock:
            raw_samples = [dense[index] for index in sample_indices]
            for raw_position in raw_samples:
                canonical, node = self._resolve_or_add_waypoint_locked(
                    raw_position,
                    EDGE_TRAVELLED,
                    mask,
                )
                canonical_samples.append(canonical)
                if node is not None:
                    added_waypoints.append(node)

            for sample_number in range(len(sample_indices) - 1):
                start_index = sample_indices[sample_number]
                end_index = sample_indices[sample_number + 1]
                raw_start = dense[start_index]
                raw_end = dense[end_index]
                start = canonical_samples[sample_number]
                end = canonical_samples[sample_number + 1]
                if start == end:
                    continue

                segment = dense[start_index : end_index + 1]
                if start != raw_start:
                    # Nearby merging was allowed only by this same mask.
                    assert mask is not None
                    prefix = tuple(
                        reversed(known_free_path(raw_start, start, mask))
                    )
                    segment = _join_paths(prefix, segment)
                if end != raw_end:
                    assert mask is not None
                    suffix = known_free_path(raw_end, end, mask)
                    segment = _join_paths(segment, suffix)

                edge = self._add_edge_locked(
                    start,
                    end,
                    segment,
                    EDGE_TRAVELLED,
                )
                if edge is not None:
                    added_edges.append(edge)

        return GraphUpdate(
            added_waypoints=tuple(added_waypoints),
            added_edges=tuple(added_edges),
            sampled_waypoints=tuple(canonical_samples),
        )

    def connect_known_free_waypoint(
        self,
        position: Sequence[int],
        known_free: np.ndarray,
        source: str = "gateway",
    ) -> GraphUpdate:
        """Add a gateway and connect it to bounded nearest visible waypoints."""
        normalized = _position(position)
        mask = _known_free_array(known_free)
        if not _cell_is_known_free(mask, normalized):
            return GraphUpdate(status="position_unknown")

        added_waypoints: List[WaypointNode] = []
        added_edges: List[WaypointEdge] = []
        with self._lock:
            canonical, node = self._resolve_or_add_waypoint_locked(
                normalized,
                str(source),
                mask,
            )
            if node is not None:
                added_waypoints.append(node)

            candidates = self._nearest_positions_locked(
                canonical,
                exclude={canonical},
            )
            existing_neighbors = {
                edge.end if edge.start == canonical else edge.start
                for edge in self._edges.values()
                if edge.source == EDGE_SLAM_LOS
                and (edge.start == canonical or edge.end == canonical)
            }
            connected = len(existing_neighbors)
            for candidate in candidates:
                if connected >= self.connector_limit:
                    break
                if candidate in existing_neighbors:
                    continue
                path = known_free_path(canonical, candidate, mask)
                if not path:
                    continue
                edge = self._add_edge_locked(
                    canonical,
                    candidate,
                    path,
                    EDGE_SLAM_LOS,
                )
                connected += 1
                if edge is not None:
                    added_edges.append(edge)

            if node is not None and connected == 0:
                # A speculative frontier gateway without a known-free highway
                # connection is not useful and would accumulate indefinitely as
                # frontier candidates change.  Existing nodes are never removed.
                self._nodes.pop(canonical, None)
                self._version += 1
                return GraphUpdate(status="no_connector")

        return GraphUpdate(
            added_waypoints=tuple(added_waypoints),
            added_edges=tuple(added_edges),
            sampled_waypoints=(canonical,),
        )

    def connect_known_free_corridor(
        self,
        position: Sequence[int],
        known_free: np.ndarray,
        *,
        search_distance: float,
        source: str = "gateway",
    ) -> GraphUpdate:
        """Join a frontier to the highway with short SLAM-safe segments.

        Frontier cells can be much farther from the travelled graph than the
        normal segment connector radius because LiDAR observes ahead of the
        drone.  This method searches only a bounded known-free neighbourhood,
        then samples the validated path at normal waypoint spacing.  No graph
        mutation occurs unless a complete corridor to an existing node exists.
        """
        normalized = _position(position)
        mask = _known_free_array(known_free)
        max_distance = float(search_distance)
        if max_distance <= 0.0:
            raise ValueError("search_distance must be positive")
        if not _cell_is_known_free(mask, normalized):
            return GraphUpdate(status="position_unknown")

        with self._lock:
            backbone = {
                position
                for position, node in self._nodes.items()
                if node.source in {"home", EDGE_TRAVELLED}
            }
            for edge in self._edges.values():
                if edge.source == EDGE_TRAVELLED:
                    backbone.add(edge.start)
                    backbone.add(edge.end)
            candidates = tuple(
                candidate
                for candidate in backbone
                if candidate != normalized
                and math.dist(normalized, candidate) <= max_distance
                and _cell_is_known_free(mask, candidate)
            )
        if not candidates:
            return GraphUpdate(status="no_connector")

        target_to_highway = _bounded_known_free_path_to_any(
            normalized,
            candidates,
            mask,
            max_distance,
        )
        if len(target_to_highway) <= 1:
            return GraphUpdate(status="no_connector")

        corridor = tuple(reversed(target_to_highway))
        highway_position = corridor[0]
        sample_indices = self._sample_indices(
            corridor,
            spacing=min(self.spacing, self.connector_distance),
        )
        added_waypoints: List[WaypointNode] = []
        added_edges: List[WaypointEdge] = []
        canonical_samples: List[Position] = []

        with self._lock:
            # A concurrent isolated-gateway rollback is the only operation
            # that can remove a node. Retry on a later movement tick rather
            # than attaching a corridor to stale state.
            if highway_position not in self._nodes:
                return GraphUpdate(status="stale_connector")

            for sample_offset, path_index in enumerate(sample_indices):
                sample = corridor[path_index]
                node_source = (
                    str(source)
                    if sample_offset == len(sample_indices) - 1
                    else "known_free"
                )
                canonical, node = self._resolve_or_add_waypoint_locked(
                    sample,
                    node_source,
                    None,
                )
                canonical_samples.append(canonical)
                if node is not None:
                    added_waypoints.append(node)

            for offset, (start_index, end_index) in enumerate(
                zip(sample_indices, sample_indices[1:])
            ):
                start = canonical_samples[offset]
                end = canonical_samples[offset + 1]
                segment = corridor[start_index : end_index + 1]
                edge = self._add_edge_locked(
                    start,
                    end,
                    segment,
                    EDGE_KNOWN_FREE_CORRIDOR,
                )
                if edge is not None:
                    added_edges.append(edge)

        return GraphUpdate(
            added_waypoints=tuple(added_waypoints),
            added_edges=tuple(added_edges),
            sampled_waypoints=tuple(canonical_samples),
        )

    def find_route(
        self,
        start: Sequence[int],
        goal: Sequence[int],
        known_free: np.ndarray,
    ) -> WaypointRoute:
        """Find a shortest safe route with ephemeral start/goal connectors."""
        start_position = _position(start)
        goal_position = _position(goal)
        mask = _known_free_array(known_free)

        if not _cell_is_known_free(mask, start_position):
            return WaypointRoute(status=ROUTE_START_UNKNOWN)
        if not _cell_is_known_free(mask, goal_position):
            return WaypointRoute(status=ROUTE_GOAL_UNKNOWN)
        if start_position == goal_position:
            return WaypointRoute(
                status=ROUTE_OK,
                waypoints=(start_position,),
                first_segment_path=(start_position,),
                first_segment_source=EDGE_KNOWN_FREE_CONNECTOR,
                first_segment_cost=0.0,
                cost=0.0,
            )

        with self._lock:
            positions = tuple(self._nodes)
            persistent_edges = tuple(self._edges.values())

        # Connector absence is the overwhelmingly common cheap failure mode.
        # Resolve it before rebuilding and revalidating the persistent graph.
        start_connectors = self._ephemeral_connectors(
            start_position,
            positions,
            mask,
        )
        goal_connectors = self._ephemeral_connectors(
            goal_position,
            positions,
            mask,
        )
        direct_path = ()
        if (
            math.dist(start_position, goal_position)
            <= self.connector_distance
        ):
            direct_path = known_free_path(start_position, goal_position, mask)

        if not start_connectors and not direct_path:
            return WaypointRoute(status=ROUTE_NO_START_CONNECTOR)
        if not goal_connectors and not direct_path:
            return WaypointRoute(status=ROUTE_NO_GOAL_CONNECTOR)

        start_token = object()
        goal_token = object()
        adjacency: Dict[object, List[_RouteEdge]] = {
            position: [] for position in positions
        }
        adjacency[start_token] = []
        adjacency[goal_token] = []

        for edge in persistent_edges:
            if (
                edge.source
                in {EDGE_SLAM_LOS, EDGE_KNOWN_FREE_CORRIDOR}
                and not validate_known_free_path(edge.path, mask)
            ):
                continue
            self._append_route_edge(
                adjacency,
                edge.start,
                edge.end,
                edge.cost,
                edge.source,
                edge.path,
            )

        for waypoint, path in start_connectors:
            self._append_route_edge(
                adjacency,
                start_token,
                waypoint,
                _path_cost(path),
                EDGE_KNOWN_FREE_CONNECTOR,
                path,
            )
        for waypoint, goal_to_waypoint_path in goal_connectors:
            waypoint_to_goal_path = tuple(reversed(goal_to_waypoint_path))
            self._append_route_edge(
                adjacency,
                waypoint,
                goal_token,
                _path_cost(waypoint_to_goal_path),
                EDGE_KNOWN_FREE_CONNECTOR,
                waypoint_to_goal_path,
            )

        if direct_path:
            self._append_route_edge(
                adjacency,
                start_token,
                goal_token,
                _path_cost(direct_path),
                EDGE_KNOWN_FREE_CONNECTOR,
                direct_path,
            )

        return self._dijkstra_route(
            adjacency,
            start_token,
            goal_token,
            start_position,
            goal_position,
        )

    def _resolve_or_add_waypoint_locked(
        self,
        position: Position,
        source: str,
        known_free: Optional[np.ndarray],
    ) -> Tuple[Position, Optional[WaypointNode]]:
        """Resolve exact/visible-near duplicates or insert a new waypoint."""
        if position in self._nodes:
            return (position, None)

        if known_free is not None and self.merge_radius > 0.0:
            candidates = sorted(
                (
                    (math.dist(position, existing), existing)
                    for existing in self._nodes
                    if math.dist(position, existing) <= self.merge_radius
                ),
                key=lambda item: (item[0], item[1]),
            )
            for _distance, existing in candidates:
                if known_free_path(position, existing, known_free):
                    return (existing, None)

        node = WaypointNode(position=position, source=source)
        self._nodes[position] = node
        self._version += 1
        return (position, node)

    def _sample_indices(
        self,
        dense: Sequence[Position],
        *,
        spacing: float | None = None,
    ) -> Tuple[int, ...]:
        """Choose cumulative-distance samples while retaining both endpoints."""
        if len(dense) <= 1:
            return (0,)

        sample_spacing = self.spacing if spacing is None else float(spacing)
        if sample_spacing <= 0.0:
            raise ValueError("sample spacing must be positive")
        indices = [0]
        cumulative = 0.0
        next_sample_distance = sample_spacing
        for index, (start, end) in enumerate(zip(dense, dense[1:]), start=1):
            cumulative += math.hypot(end[0] - start[0], end[1] - start[1])
            if cumulative + 1e-9 >= next_sample_distance:
                indices.append(index)
                next_sample_distance += sample_spacing

        last_index = len(dense) - 1
        if indices[-1] != last_index:
            indices.append(last_index)
        return tuple(indices)

    def _add_edge_locked(
        self,
        start: Position,
        end: Position,
        path: Sequence[Position],
        source: str,
    ) -> Optional[WaypointEdge]:
        """Add or shorten one source-specific undirected persistent edge."""
        if start == end:
            return None
        oriented_path = tuple(path)
        if not oriented_path or oriented_path[0] != start or oriented_path[-1] != end:
            raise ValueError("edge path must be oriented from start to end")

        ordered_start, ordered_end = sorted((start, end))
        if start != ordered_start:
            oriented_path = tuple(reversed(oriented_path))
        key = (ordered_start, ordered_end, source)
        edge = WaypointEdge(
            start=ordered_start,
            end=ordered_end,
            cost=_path_cost(oriented_path),
            source=source,
            path=oriented_path,
        )
        previous = self._edges.get(key)
        if previous is not None and previous.cost <= edge.cost + 1e-9:
            return None
        self._edges[key] = edge
        self._version += 1
        return edge

    def _nearest_positions_locked(
        self,
        origin: Position,
        exclude: set[Position],
    ) -> Tuple[Position, ...]:
        """Return in-range persistent positions sorted by distance and position."""
        candidates = (
            (math.dist(origin, position), position)
            for position in self._nodes
            if position not in exclude
        )
        return tuple(
            position
            for distance, position in sorted(candidates)
            if distance <= self.connector_distance
        )

    def _ephemeral_connectors(
        self,
        origin: Position,
        positions: Sequence[Position],
        known_free: np.ndarray,
    ) -> Tuple[Tuple[Position, WaypointPath], ...]:
        """Return at most ``connector_limit`` nearest visible connectors."""
        candidates = sorted(
            (
                (math.dist(origin, position), position)
                for position in positions
                if math.dist(origin, position) <= self.connector_distance
            ),
            key=lambda item: (item[0], item[1]),
        )
        connectors: List[Tuple[Position, WaypointPath]] = []
        for _distance, position in candidates:
            path = known_free_path(origin, position, known_free)
            if not path:
                continue
            connectors.append((position, path))
            if len(connectors) >= self.connector_limit:
                break
        return tuple(connectors)

    @staticmethod
    def _append_route_edge(
        adjacency: Dict[object, List[_RouteEdge]],
        start: object,
        end: object,
        cost: float,
        source: str,
        path: WaypointPath,
    ) -> None:
        """Append both orientations of an edge to a route-local adjacency map."""
        adjacency[start].append(
            _RouteEdge(
                destination=end,
                cost=cost,
                source=source,
                path=path,
            )
        )
        adjacency[end].append(
            _RouteEdge(
                destination=start,
                cost=cost,
                source=source,
                path=tuple(reversed(path)),
            )
        )

    @staticmethod
    def _dijkstra_route(
        adjacency: Mapping[object, Sequence[_RouteEdge]],
        start_token: object,
        goal_token: object,
        start_position: Position,
        goal_position: Position,
    ) -> WaypointRoute:
        """Run Dijkstra and materialize the rich first-segment result."""
        sequence = itertools.count()
        distances: Dict[object, float] = {start_token: 0.0}
        previous: Dict[object, Tuple[object, _RouteEdge]] = {}
        heap: List[Tuple[float, int, object]] = [
            (0.0, next(sequence), start_token)
        ]

        while heap:
            distance, _order, current = heapq.heappop(heap)
            if distance > distances.get(current, math.inf) + 1e-9:
                continue
            if current is goal_token:
                break
            for edge in adjacency[current]:
                candidate_distance = distance + edge.cost
                if candidate_distance + 1e-9 >= distances.get(
                    edge.destination,
                    math.inf,
                ):
                    continue
                distances[edge.destination] = candidate_distance
                previous[edge.destination] = (current, edge)
                heapq.heappush(
                    heap,
                    (candidate_distance, next(sequence), edge.destination),
                )

        if goal_token not in distances:
            return WaypointRoute(status=ROUTE_DISCONNECTED)

        reversed_nodes: List[object] = [goal_token]
        reversed_edges: List[_RouteEdge] = []
        current = goal_token
        while current is not start_token:
            parent, edge = previous[current]
            reversed_edges.append(edge)
            reversed_nodes.append(parent)
            current = parent
        nodes = tuple(reversed(reversed_nodes))
        edges = tuple(reversed(reversed_edges))

        route_positions: List[Position] = []
        for node in nodes:
            if node is start_token:
                position = start_position
            elif node is goal_token:
                position = goal_position
            else:
                position = node  # type: ignore[assignment]
            if not route_positions or route_positions[-1] != position:
                route_positions.append(position)

        first_edge = next(
            (
                edge
                for edge in edges
                if edge.cost > 1e-9 and len(set(edge.path)) > 1
            ),
            edges[0],
        )
        return WaypointRoute(
            status=ROUTE_OK,
            waypoints=tuple(route_positions),
            first_segment_path=first_edge.path,
            first_segment_source=first_edge.source,
            first_segment_cost=first_edge.cost,
            cost=distances[goal_token],
        )
