"""Sparse, SLAM-safe waypoint routing for long drone journeys.

The graph deliberately has no cave-map dependency.  Persistent connections are
created only from paths that a drone actually travelled or from paths validated
against a boolean ``known_free`` mask supplied by SLAM.  Travelled edges remain
trusted; SLAM-derived edges are revalidated against every route requester's
current mask.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import heapq
import itertools
import math
import threading
import time
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


Position = Tuple[int, int]
WaypointPath = Tuple[Position, ...]
WaypointId = int
EdgeId = int

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
LOCAL_CONNECTOR_TIME_BUDGET_SECONDS = 0.004


class WaypointRole(str, Enum):
    HOME = "home"
    JUNCTION = "junction"
    CHOKEPOINT = "chokepoint"
    TURN = "turn"
    FRONTIER_GATEWAY = "frontier_gateway"
    RECOVERY_ANCHOR = "recovery_anchor"


@dataclass(frozen=True)
class WaypointNode:
    """One persistent waypoint and the source that first registered it."""

    id: WaypointId
    position: Position
    source: str
    roles: FrozenSet[WaypointRole] = frozenset()
    created_revision: int = 0
    updated_revision: int = 0


@dataclass(frozen=True)
class WaypointEdge:
    """One undirected persistent edge stored in ``start`` -> ``end`` order."""

    id: EdgeId
    start_id: WaypointId
    end_id: WaypointId
    start: Position
    end: Position
    cost: float
    source: str
    path: WaypointPath
    created_revision: int = 0
    retired_revision: Optional[int] = None
    owner: Optional[object] = None


@dataclass(frozen=True)
class GraphDelta:
    """All identity changes committed by one logical topology mutation."""

    revision: int = 0
    added_node_ids: Tuple[WaypointId, ...] = ()
    updated_node_ids: Tuple[WaypointId, ...] = ()
    retired_node_ids: Tuple[WaypointId, ...] = ()
    added_edge_ids: Tuple[EdgeId, ...] = ()
    updated_edge_ids: Tuple[EdgeId, ...] = ()
    retired_edge_ids: Tuple[EdgeId, ...] = ()
    edge_replacements: Mapping[EdgeId, Tuple[EdgeId, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphUpdate:
    """Detached record of the graph objects created by one operation."""

    status: str = ROUTE_OK
    added_waypoints: Tuple[WaypointNode, ...] = ()
    added_edges: Tuple[WaypointEdge, ...] = ()
    sampled_waypoints: Tuple[Position, ...] = ()
    delta: GraphDelta = GraphDelta()

    @property
    def added_nodes(self) -> Tuple[WaypointNode, ...]:
        """Alias useful to callers that use generic graph terminology."""
        return self.added_waypoints

    @property
    def changed(self) -> bool:
        """Return whether this operation added or improved graph state."""
        return bool(
            self.added_waypoints
            or self.added_edges
            or self.delta.updated_node_ids
            or self.delta.retired_node_ids
            or self.delta.retired_edge_ids
        )


@dataclass(frozen=True)
class WaypointGraphSnapshot:
    """Immutable, consistently ordered view of a graph version."""

    waypoints: Tuple[WaypointNode, ...]
    edges: Tuple[WaypointEdge, ...]
    version: int = 0

    @property
    def topology_revision(self) -> int:
        return self.version

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
    topology_revision: int = 0
    requester_knowledge_revision: int = 0
    node_ids: Tuple[WaypointId, ...] = ()
    edge_ids: Tuple[EdgeId, ...] = ()
    cache_hit: bool = False
    entry_connector_path: WaypointPath = ()
    exit_connector_path: WaypointPath = ()
    remaining_cost: float = math.inf
    segment_paths: Tuple[WaypointPath, ...] = ()
    segment_sources: Tuple[str, ...] = ()
    segment_edge_ids: Tuple[Optional[EdgeId], ...] = ()
    connector_astar_calls: int = 0
    id: int = 0

    @property
    def found(self) -> bool:
        """Return whether this result contains a usable route."""
        return self.status == ROUTE_OK

    @property
    def route_id(self) -> int:
        """Return the stable trace-facing identity of this route result."""
        return self.id


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


@dataclass(frozen=True)
class _ReverseTree:
    distances: Mapping[WaypointId, float]
    next_edges: Mapping[WaypointId, EdgeId]
    roots: Mapping[WaypointId, WaypointId]


class WaypointGraph:
    """Thread-safe ID graph with a coordinate-compatible public facade."""

    def __init__(
        self,
        merge_radius: float = 8.0,
        connector_distance: float = 64.0,
        connector_limit: int = 8,
        spatial_hash_cell: int = 32,
        route_cache_capacity: int = 64,
    ) -> None:
        """Create an empty strategic graph with bounded connector policies."""
        if float(merge_radius) < 0.0:
            raise ValueError("merge_radius cannot be negative")
        if float(connector_distance) < 0.0:
            raise ValueError("connector_distance cannot be negative")
        if int(connector_limit) < 1:
            raise ValueError("connector_limit must be at least one")
        if int(spatial_hash_cell) < 1:
            raise ValueError("spatial_hash_cell must be positive")
        if int(route_cache_capacity) < 1:
            raise ValueError("route_cache_capacity must be positive")

        self.merge_radius = float(merge_radius)
        self.connector_distance = float(connector_distance)
        self.connector_limit = int(connector_limit)
        self.spatial_hash_cell = int(spatial_hash_cell)
        self.route_cache_capacity = int(route_cache_capacity)
        self._lock = threading.RLock()
        self._nodes: Dict[WaypointId, WaypointNode] = {}
        self._position_ids: Dict[Position, WaypointId] = {}
        self._edges: Dict[EdgeId, WaypointEdge] = {}
        self._edge_keys: Dict[Tuple[WaypointId, WaypointId, str], EdgeId] = {}
        self._adjacency: Dict[WaypointId, Set[EdgeId]] = defaultdict(set)
        self._node_hash: Dict[Tuple[int, int], Set[WaypointId]] = defaultdict(set)
        self._edge_hash: Dict[Tuple[int, int], Set[EdgeId]] = defaultdict(set)
        self._components: Dict[WaypointId, int] = {}
        self._next_node_id = 1
        self._next_edge_id = 1
        self._next_route_id = 1
        self._version = 0
        self._last_delta = GraphDelta()
        self._route_cache: OrderedDict[Tuple[object, ...], _ReverseTree] = OrderedDict()
        self._route_tree_builds = 0
        self._route_cache_hits = 0

    @property
    def version(self) -> int:
        """Return the monotonic graph revision used by render caches."""
        with self._lock:
            return self._version

    @property
    def topology_revision(self) -> int:
        return self.version

    @property
    def last_delta(self) -> GraphDelta:
        with self._lock:
            return self._last_delta

    @property
    def route_tree_builds(self) -> int:
        with self._lock:
            return self._route_tree_builds

    @property
    def route_cache_hits(self) -> int:
        with self._lock:
            return self._route_cache_hits

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
                self._nodes[node_id] for node_id in sorted(self._nodes)
            )
            edges = tuple(self._edges[edge_id] for edge_id in sorted(self._edges))
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
            added_nodes: List[WaypointNode] = []
            updated_node_ids: List[WaypointId] = []
            canonical_id, node = self._resolve_or_add_waypoint_locked(
                normalized,
                str(source),
                mask,
            )
            if node is not None:
                added_nodes.append(node)
            role = self._role_for_source(str(source))
            if (
                node is None
                and role is not None
                and self._add_role_locked(canonical_id, role)
            ):
                updated_node_ids.append(canonical_id)
            self._commit_locked(
                added_nodes, [], updated_node_ids=updated_node_ids,
            )
            canonical = self._nodes[canonical_id].position
        return (canonical, bool(added_nodes))

    def register_travelled_section(
        self,
        path: Iterable[Sequence[int]],
        *,
        start_role: Optional[WaypointRole] = None,
        end_role: Optional[WaypointRole] = None,
        end_roles: Iterable[WaypointRole] = (),
    ) -> GraphUpdate:
        """Store one complete travelled section between strategic endpoints.

        This method never samples fixed-interval interior nodes. Exact
        intersections with existing travelled geometry are the only extra
        points that may become nodes.
        """
        dense = _densify_polyline(path)
        if not dense:
            return GraphUpdate(status="empty_path")
        if len(dense) == 1:
            return GraphUpdate(status="short_path", sampled_waypoints=(dense[0],))

        added_node_ids: List[WaypointId] = []
        added_edges: List[WaypointEdge] = []
        retired_edges: List[EdgeId] = []
        replacements: Dict[EdgeId, Tuple[EdgeId, ...]] = {}
        updated_node_ids: List[WaypointId] = []

        with self._lock:
            strategic_indices = {0, len(dense) - 1}
            # Endpoints landing on an existing trail consolidate overlapping
            # sections. Interior crossings require a transverse intersection;
            # sharing a hash bucket or merely running nearby is never enough.
            for incoming_index, point in enumerate(dense):
                candidate_ids = tuple(
                    self._edge_hash.get(self._hash_cell(point), ())
                )
                for edge_id in candidate_ids:
                    edge = self._edges.get(edge_id)
                    if (
                        edge is None
                        or edge.source != EDGE_TRAVELLED
                        or point not in edge.path[1:-1]
                    ):
                        continue
                    endpoint_hit = incoming_index in {0, len(dense) - 1}
                    if not endpoint_hit and not self._is_transverse_intersection(
                        dense, edge.path, point,
                    ):
                        continue
                    had_junction_role = (
                        point in self._position_ids
                        and WaypointRole.JUNCTION in self._nodes[
                            self._position_ids[point]
                        ].roles
                    )
                    junction_id = self._split_edge_locked(
                        edge_id, point, [], added_edges, retired_edges,
                        replacements,
                    )
                    if (
                        junction_id is not None
                        and not had_junction_role
                        and self._nodes[junction_id].created_revision
                        != self._version + 1
                        and junction_id not in updated_node_ids
                    ):
                        updated_node_ids.append(junction_id)
                    strategic_indices.add(incoming_index)

            ordered_indices = tuple(sorted(strategic_indices))
            canonical_ids: List[WaypointId] = []
            for index in ordered_indices:
                canonical_id, node = self._resolve_or_add_waypoint_locked(
                    dense[index], EDGE_TRAVELLED, None,
                )
                canonical_ids.append(canonical_id)
                if node is not None:
                    added_node_ids.append(canonical_id)

            endpoint_roles = [(canonical_ids[0], start_role), (canonical_ids[-1], end_role)]
            endpoint_roles.extend((canonical_ids[-1], role) for role in end_roles)
            for node_id, role in endpoint_roles:
                if role is not None and self._add_role_locked(node_id, role):
                    if node_id not in added_node_ids and node_id not in updated_node_ids:
                        updated_node_ids.append(node_id)

            # Split-created nodes may have been inserted through a temporary
            # list, so include all newly created IDs from this transaction.
            existing_added = set(added_node_ids)
            for node_id in canonical_ids:
                node = self._nodes[node_id]
                if node.created_revision == self._version + 1:
                    existing_added.add(node_id)
            for edge in added_edges:
                for node_id in (edge.start_id, edge.end_id):
                    node = self._nodes[node_id]
                    if node.created_revision == self._version + 1:
                        existing_added.add(node_id)
            added_node_ids = sorted(existing_added)

            for offset, (start_index, end_index) in enumerate(
                zip(ordered_indices, ordered_indices[1:])
            ):
                edge = self._add_edge_locked(
                    canonical_ids[offset], canonical_ids[offset + 1],
                    dense[start_index : end_index + 1], EDGE_TRAVELLED,
                )
                if edge is not None:
                    added_edges.append(edge)

            # Multiple overlap endpoints can split a replacement created
            # earlier in this same transaction. Report only final live edges
            # and flatten replacement chains to those final IDs.
            added_edges = [
                edge for edge in added_edges if edge.id in self._edges
            ]
            def final_replacements(edge_id: EdgeId) -> Tuple[EdgeId, ...]:
                result: List[EdgeId] = []
                pending = list(replacements.get(edge_id, ()))
                while pending:
                    candidate = pending.pop(0)
                    if candidate in replacements:
                        pending[0:0] = replacements[candidate]
                    elif candidate in self._edges:
                        result.append(candidate)
                return tuple(dict.fromkeys(result))
            replacements = {
                edge_id: final_replacements(edge_id)
                for edge_id in replacements
            }
            added_nodes = [self._nodes[node_id] for node_id in added_node_ids]
            canonical_positions = tuple(
                self._nodes[node_id].position for node_id in canonical_ids
            )
            delta = self._commit_locked(
                added_nodes, added_edges,
                updated_node_ids=updated_node_ids,
                retired_edge_ids=retired_edges,
                replacements=replacements,
            )

        return GraphUpdate(
            added_waypoints=tuple(added_nodes),
            added_edges=tuple(added_edges),
            sampled_waypoints=canonical_positions,
            delta=delta,
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
        updated_node_ids: List[WaypointId] = []
        with self._lock:
            canonical_id, node = self._resolve_or_add_waypoint_locked(
                normalized,
                str(source),
                mask,
            )
            canonical = self._nodes[canonical_id].position
            if node is not None:
                added_waypoints.append(node)
            elif self._add_role_locked(
                canonical_id, WaypointRole.FRONTIER_GATEWAY,
            ):
                updated_node_ids.append(canonical_id)

            candidates = self._nearest_node_ids_locked(
                canonical,
                exclude={canonical_id},
            )
            existing_neighbors = {
                edge.end_id if edge.start_id == canonical_id else edge.start_id
                for edge_id in self._adjacency[canonical_id]
                for edge in (self._edges[edge_id],)
                if edge.source == EDGE_SLAM_LOS
            }
            connected = len(existing_neighbors)
            for candidate_id in candidates:
                if connected >= self.connector_limit:
                    break
                if candidate_id in existing_neighbors:
                    continue
                candidate = self._nodes[candidate_id].position
                path = known_free_path(canonical, candidate, mask)
                if not path:
                    continue
                edge = self._add_edge_locked(
                    canonical_id,
                    candidate_id,
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
                self._remove_uncommitted_node_locked(canonical_id)
                return GraphUpdate(status="no_connector")

            delta = self._commit_locked(
                added_waypoints, added_edges,
                updated_node_ids=updated_node_ids,
            )

        return GraphUpdate(
            added_waypoints=tuple(added_waypoints),
            added_edges=tuple(added_edges),
            sampled_waypoints=(canonical,),
            delta=delta,
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
        then stores the validated corridor as one complete polyline. No graph
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
                node.position
                for node in self._nodes.values()
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
        added_waypoints: List[WaypointNode] = []
        added_edges: List[WaypointEdge] = []
        updated_node_ids: List[WaypointId] = []
        canonical_samples: List[Position] = [highway_position]

        with self._lock:
            # A concurrent isolated-gateway rollback is the only operation
            # that can remove a node. Retry on a later movement tick rather
            # than attaching a corridor to stale state.
            if highway_position not in self._position_ids:
                return GraphUpdate(status="stale_connector")

            gateway_id, node = self._resolve_or_add_waypoint_locked(
                normalized, str(source), None,
            )
            role_added = self._add_role_locked(
                gateway_id, WaypointRole.FRONTIER_GATEWAY,
            )
            canonical_gateway = self._nodes[gateway_id].position
            canonical_samples.append(canonical_gateway)
            if node is not None:
                added_waypoints.append(self._nodes[gateway_id])
            elif role_added:
                updated_node_ids.append(gateway_id)
            edge = self._add_edge_locked(
                self._position_ids[highway_position], gateway_id,
                corridor, EDGE_KNOWN_FREE_CORRIDOR,
            )
            if edge is not None:
                added_edges.append(edge)
            delta = self._commit_locked(
                added_waypoints, added_edges,
                updated_node_ids=updated_node_ids,
            )

        return GraphUpdate(
            added_waypoints=tuple(added_waypoints),
            added_edges=tuple(added_edges),
            sampled_waypoints=tuple(canonical_samples),
            delta=delta,
        )

    def find_route(
        self,
        start: Sequence[int],
        goal: Sequence[int],
        known_free: np.ndarray,
        *,
        requester_id: object = None,
        requester_knowledge_revision: int = 0,
    ) -> WaypointRoute:
        """Find a shortest route using persistent adjacency and cached trees."""
        with self._lock:
            route_id = self._next_route_id
            self._next_route_id += 1
        start_position = _position(start)
        goal_position = _position(goal)
        mask = _known_free_array(known_free)

        if not _cell_is_known_free(mask, start_position):
            return WaypointRoute(status=ROUTE_START_UNKNOWN, id=route_id)
        if not _cell_is_known_free(mask, goal_position):
            return WaypointRoute(status=ROUTE_GOAL_UNKNOWN, id=route_id)
        if start_position == goal_position:
            return WaypointRoute(
                status=ROUTE_OK,
                id=route_id,
                waypoints=(start_position,),
                first_segment_path=(start_position,),
                first_segment_source=EDGE_KNOWN_FREE_CONNECTOR,
                first_segment_cost=0.0,
                cost=0.0,
                topology_revision=self.version,
                requester_knowledge_revision=requester_knowledge_revision,
                remaining_cost=0.0,
                segment_paths=((start_position,),),
                segment_sources=(EDGE_KNOWN_FREE_CONNECTOR,),
                segment_edge_ids=(None,),
            )

        with self._lock:
            positions = tuple(node.position for node in self._nodes.values())

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
        direct_distance = math.dist(start_position, goal_position)
        direct_path = ()
        direct_astar_attempted = False
        far_direct_path = ()
        if direct_distance <= self.connector_distance:
            direct_path = known_free_path(start_position, goal_position, mask)
            if not direct_path:
                # Close targets behind a locally observed bend must not detour
                # through an arbitrarily long global highway.  This bounded A*
                # runs only while constructing the exact ephemeral connector;
                # execution later consumes the stored polyline verbatim.
                direct_astar_attempted = True
                direct_path = _bounded_known_free_path_to_any(
                    start_position,
                    (goal_position,),
                    mask,
                    self.connector_distance,
                )
        else:
            # A long LOS is a route simplification, not a substitute for graph
            # connectivity.  Keep connector/component failure semantics intact
            # and use it only when it shortens an otherwise valid graph route.
            far_direct_path = known_free_path(
                start_position, goal_position, mask,
            )

        if not start_connectors and not direct_path:
            return WaypointRoute(
                status=ROUTE_NO_START_CONNECTOR,
                id=route_id,
                connector_astar_calls=int(direct_astar_attempted),
            )
        if not goal_connectors and not direct_path:
            return WaypointRoute(
                status=ROUTE_NO_GOAL_CONNECTOR,
                id=route_id,
                connector_astar_calls=int(direct_astar_attempted),
            )

        if direct_path:
            return WaypointRoute(
                status=ROUTE_OK,
                id=route_id,
                waypoints=(start_position, goal_position),
                first_segment_path=direct_path,
                first_segment_source=EDGE_KNOWN_FREE_CONNECTOR,
                first_segment_cost=_path_cost(direct_path),
                cost=_path_cost(direct_path),
                topology_revision=self.version,
                requester_knowledge_revision=requester_knowledge_revision,
                entry_connector_path=direct_path,
                remaining_cost=_path_cost(direct_path),
                segment_paths=(direct_path,),
                segment_sources=(EDGE_KNOWN_FREE_CONNECTOR,),
                segment_edge_ids=(None,),
                connector_astar_calls=int(direct_astar_attempted),
            )

        with self._lock:
            start_options = tuple(
                (self._position_ids[position], path)
                for position, path in start_connectors
                if position in self._position_ids
            )
            goal_options = tuple(
                (self._position_ids[position], path)
                for position, path in goal_connectors
                if position in self._position_ids
            )
            # Reject disconnected connector component sets before any tree work.
            start_components = {self._components.get(node_id) for node_id, _ in start_options}
            goal_components = {self._components.get(node_id) for node_id, _ in goal_options}
            if not (start_components & goal_components):
                return WaypointRoute(
                    status=ROUTE_DISCONNECTED,
                    id=route_id,
                    topology_revision=self._version,
                    requester_knowledge_revision=requester_knowledge_revision,
                    connector_astar_calls=int(direct_astar_attempted),
                )

            best: Optional[
                Tuple[
                    float,
                    WaypointId,
                    WaypointPath,
                    _ReverseTree,
                    bool,
                ]
            ] = None
            belief_signature = self._belief_validity_signature_locked(mask)
            goal_signature = tuple(goal_options)
            key = (
                "multi_goal",
                goal_signature,
                self._version,
                requester_id,
                belief_signature,
            )
            tree = self._route_cache.get(key)
            cache_hit = tree is not None
            if tree is not None and not all(
                self._tree_route_valid_locked(start_id, tree, mask)
                for start_id, _path in start_options
                if start_id in tree.distances
            ):
                self._route_cache.pop(key, None)
                tree = None
                cache_hit = False
            if tree is None:
                tree = self._build_multi_goal_tree_locked(
                    goal_options,
                    mask,
                )
                self._route_tree_builds += 1
                self._route_cache[key] = tree
                self._trim_route_cache_locked()
            else:
                self._route_cache.move_to_end(key)
                self._route_cache_hits += 1

            for start_id, start_path in start_options:
                graph_cost = tree.distances.get(start_id)
                if graph_cost is None:
                    continue
                total = _path_cost(start_path) + graph_cost
                candidate = (
                    total,
                    start_id,
                    start_path,
                    tree,
                    cache_hit,
                )
                if best is None or candidate[0] < best[0] - 1e-9:
                    best = candidate

            if best is None:
                return WaypointRoute(
                    status=ROUTE_DISCONNECTED,
                    id=route_id,
                    topology_revision=self._version,
                    requester_knowledge_revision=requester_knowledge_revision,
                    connector_astar_calls=int(direct_astar_attempted),
                )
            route = self._materialize_cached_route_locked(
                start_position,
                goal_position,
                best,
                dict(goal_options),
                requester_knowledge_revision,
                route_id,
                connector_astar_calls=int(direct_astar_attempted),
            )
            far_direct_cost = _path_cost(far_direct_path)
            if (
                far_direct_path
                and far_direct_cost < route.cost - 1e-9
            ):
                return WaypointRoute(
                    status=ROUTE_OK,
                    id=route_id,
                    waypoints=(start_position, goal_position),
                    first_segment_path=far_direct_path,
                    first_segment_source=EDGE_SLAM_LOS,
                    first_segment_cost=far_direct_cost,
                    cost=far_direct_cost,
                    topology_revision=self._version,
                    requester_knowledge_revision=(
                        requester_knowledge_revision
                    ),
                    cache_hit=route.cache_hit,
                    remaining_cost=far_direct_cost,
                    segment_paths=(far_direct_path,),
                    segment_sources=(EDGE_SLAM_LOS,),
                    segment_edge_ids=(None,),
                    connector_astar_calls=int(direct_astar_attempted),
                )
            return route

    def _resolve_or_add_waypoint_locked(
        self,
        position: Position,
        source: str,
        known_free: Optional[np.ndarray],
    ) -> Tuple[WaypointId, Optional[WaypointNode]]:
        """Resolve exact/visible-near duplicates or insert a new waypoint."""
        if position in self._position_ids:
            return (self._position_ids[position], None)

        if known_free is not None and self.merge_radius > 0.0:
            candidates = sorted(
                (
                    (math.dist(position, self._nodes[node_id].position), node_id)
                    for node_id in self._candidate_node_ids_locked(position, self.merge_radius)
                    if math.dist(position, self._nodes[node_id].position) <= self.merge_radius
                ),
                key=lambda item: (item[0], self._nodes[item[1]].position),
            )
            for _distance, node_id in candidates:
                if known_free_path(position, self._nodes[node_id].position, known_free):
                    return (node_id, None)

        node_id = self._next_node_id
        self._next_node_id += 1
        revision = self._version + 1
        role = self._role_for_source(source)
        roles = frozenset() if role is None else frozenset({role})
        node = WaypointNode(
            id=node_id, position=position, source=source, roles=roles,
            created_revision=revision, updated_revision=revision,
        )
        self._nodes[node_id] = node
        self._position_ids[position] = node_id
        self._adjacency[node_id]
        self._node_hash[self._hash_cell(position)].add(node_id)
        return (node_id, node)

    @staticmethod
    def _role_for_source(source: str) -> Optional[WaypointRole]:
        source_roles = {
            "home": WaypointRole.HOME,
            "junction": WaypointRole.JUNCTION,
            "gateway": WaypointRole.FRONTIER_GATEWAY,
            "frontier_gateway": WaypointRole.FRONTIER_GATEWAY,
            "turn": WaypointRole.TURN,
            "chokepoint": WaypointRole.CHOKEPOINT,
            "recovery_anchor": WaypointRole.RECOVERY_ANCHOR,
        }
        return source_roles.get(source)

    def _add_role_locked(
        self, node_id: WaypointId, role: WaypointRole,
    ) -> bool:
        """Add a strategic role without changing node identity."""
        node = self._nodes[node_id]
        normalized = WaypointRole(role)
        if normalized in node.roles:
            return False
        self._nodes[node_id] = WaypointNode(
            id=node.id,
            position=node.position,
            source=node.source,
            roles=node.roles | {normalized},
            created_revision=node.created_revision,
            updated_revision=self._version + 1,
        )
        return True

    def _add_edge_locked(
        self,
        start_id: WaypointId,
        end_id: WaypointId,
        path: Sequence[Position],
        source: str,
    ) -> Optional[WaypointEdge]:
        """Add or shorten one source-specific undirected persistent edge."""
        if start_id == end_id:
            return None
        start = self._nodes[start_id].position
        end = self._nodes[end_id].position
        oriented_path = tuple(path)
        if not oriented_path or oriented_path[0] != start or oriented_path[-1] != end:
            raise ValueError("edge path must be oriented from start to end")

        ordered_start_id, ordered_end_id = sorted((start_id, end_id))
        ordered_start = self._nodes[ordered_start_id].position
        ordered_end = self._nodes[ordered_end_id].position
        if start_id != ordered_start_id:
            oriented_path = tuple(reversed(oriented_path))
        key = (ordered_start_id, ordered_end_id, source)
        edge_id = self._next_edge_id
        edge = WaypointEdge(
            id=edge_id,
            start_id=ordered_start_id,
            end_id=ordered_end_id,
            start=ordered_start,
            end=ordered_end,
            cost=_path_cost(oriented_path),
            source=source,
            path=oriented_path,
            created_revision=self._version + 1,
        )
        previous_id = self._edge_keys.get(key)
        previous = None if previous_id is None else self._edges[previous_id]
        if previous is not None and previous.cost <= edge.cost + 1e-9:
            return None
        if previous is not None:
            self._remove_edge_locked(previous.id)
        self._next_edge_id += 1
        self._edges[edge_id] = edge
        self._edge_keys[key] = edge_id
        self._adjacency[ordered_start_id].add(edge_id)
        self._adjacency[ordered_end_id].add(edge_id)
        for point in oriented_path:
            self._edge_hash[self._hash_cell(point)].add(edge_id)
        return edge

    def _nearest_node_ids_locked(
        self,
        origin: Position,
        exclude: Set[WaypointId],
    ) -> Tuple[WaypointId, ...]:
        """Return in-range persistent positions sorted by distance and position."""
        candidates = (
            (math.dist(origin, self._nodes[node_id].position), node_id)
            for node_id in self._candidate_node_ids_locked(origin, self.connector_distance)
            if node_id not in exclude
        )
        return tuple(
            node_id
            for distance, node_id in sorted(candidates, key=lambda item: (item[0], self._nodes[item[1]].position))
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

    def split_edge(self, edge_id: EdgeId, junction: Sequence[int]) -> GraphDelta:
        """Atomically replace an edge by two exact polyline slices."""
        position = _position(junction)
        with self._lock:
            added_nodes: List[WaypointNode] = []
            added_edges: List[WaypointEdge] = []
            retired_edges: List[EdgeId] = []
            replacements: Dict[EdgeId, Tuple[EdgeId, ...]] = {}
            self._split_edge_locked(
                int(edge_id), position, added_nodes, added_edges,
                retired_edges, replacements,
            )
            return self._commit_locked(
                added_nodes, added_edges,
                retired_edge_ids=retired_edges,
                replacements=replacements,
            )

    def collapse_node(
        self,
        node_id: WaypointId,
        *,
        active_route_node_ids: Iterable[WaypointId] = (),
        active_route_edge_ids: Iterable[EdgeId] = (),
        collapsible_roles: Iterable[WaypointRole] = (),
    ) -> GraphDelta:
        """Collapse an unprotected degree-two node without altering geometry."""
        protected = set(active_route_node_ids)
        protected_edges = set(active_route_edge_ids)
        allowed_roles = set(collapsible_roles)
        with self._lock:
            node = self._nodes.get(int(node_id))
            if node is None:
                raise KeyError(node_id)
            if node.id in protected or (set(node.roles) - allowed_roles):
                return GraphDelta(revision=self._version)
            incident = tuple(self._adjacency[node.id])
            if len(incident) != 2 or protected_edges.intersection(incident):
                return GraphDelta(revision=self._version)
            first, second = (self._edges[edge_id] for edge_id in incident)
            if first.source != second.source:
                return GraphDelta(revision=self._version)
            first_neighbor = first.end_id if first.start_id == node.id else first.start_id
            second_neighbor = second.end_id if second.start_id == node.id else second.start_id
            if first_neighbor == second_neighbor:
                return GraphDelta(revision=self._version)
            path_to_node = (
                first.path if first.end_id == node.id else tuple(reversed(first.path))
            )
            path_from_node = (
                second.path if second.start_id == node.id else tuple(reversed(second.path))
            )
            joined = _join_paths(path_to_node, path_from_node)
            retired = [first.id, second.id]
            self._remove_edge_locked(first.id)
            self._remove_edge_locked(second.id)
            self._remove_node_locked(node.id)
            replacement = self._add_edge_locked(
                first_neighbor, second_neighbor, joined, first.source,
            )
            additions = [] if replacement is None else [replacement]
            return self._commit_locked(
                [], additions,
                retired_node_ids=[node.id],
                retired_edge_ids=retired,
                replacements={
                    first.id: tuple(edge.id for edge in additions),
                    second.id: tuple(edge.id for edge in additions),
                },
            )

    def collapse_inactive_degree_two_nodes(
        self,
        *,
        active_route_node_ids: Iterable[WaypointId] = (),
        active_route_edge_ids: Iterable[EdgeId] = (),
        active_gateway_ids: Iterable[WaypointId] = (),
        limit: int = 64,
    ) -> tuple[GraphDelta, ...]:
        """Batch-collapse safe inactive turn/junction geometry.

        Home, chokepoint, recovery-anchor, and frontier-gateway roles remain
        protected.  Exact edge polylines are joined by ``collapse_node`` so the
        simplification never regenerates a route or changes its geometry.
        """
        protected_nodes = {
            int(node_id) for node_id in active_route_node_ids
        } | {int(node_id) for node_id in active_gateway_ids}
        protected_edges = {int(edge_id) for edge_id in active_route_edge_ids}
        allowed_roles = {WaypointRole.TURN, WaypointRole.JUNCTION}
        maximum = max(0, int(limit))
        if maximum == 0:
            return ()
        deltas: list[GraphDelta] = []
        for node_id in tuple(node.id for node in self.snapshot().nodes):
            if len(deltas) >= maximum:
                break
            try:
                delta = self.collapse_node(
                    node_id,
                    active_route_node_ids=protected_nodes,
                    active_route_edge_ids=protected_edges,
                    collapsible_roles=allowed_roles,
                )
            except KeyError:
                continue
            if delta.retired_node_ids:
                deltas.append(delta)
        return tuple(deltas)

    def retire_inactive_orphan_trail_leaves(
        self,
        *,
        active_route_node_ids: Iterable[WaypointId] = (),
        active_route_edge_ids: Iterable[EdgeId] = (),
        active_gateway_ids: Iterable[WaypointId] = (),
        limit: int = 64,
    ) -> tuple[GraphDelta, ...]:
        """Retire roleless inactive leaves that cannot carry graph transit.

        A roleless degree-zero node or the terminal edge of a roleless
        travelled leaf has no strategic identity and cannot lie between two
        route anchors. Removing it therefore cannot alter any surviving route
        geometry. Active route/gateway identities and every role-bearing node
        remain protected.
        """
        protected_nodes = {
            int(node_id) for node_id in active_route_node_ids
        } | {int(node_id) for node_id in active_gateway_ids}
        protected_edges = {int(edge_id) for edge_id in active_route_edge_ids}
        maximum = max(0, int(limit))
        if maximum == 0:
            return ()
        deltas: list[GraphDelta] = []
        while len(deltas) < maximum:
            with self._lock:
                selected: tuple[WaypointId, EdgeId | None] | None = None
                for node_id in sorted(self._nodes):
                    node = self._nodes[node_id]
                    incident = tuple(self._adjacency[node_id])
                    if (
                        node_id in protected_nodes
                        or node.roles
                        or node.source != EDGE_TRAVELLED
                        or len(incident) > 1
                    ):
                        continue
                    edge_id = incident[0] if incident else None
                    if edge_id is not None and (
                        edge_id in protected_edges
                        or self._edges[edge_id].source != EDGE_TRAVELLED
                    ):
                        continue
                    selected = (node_id, edge_id)
                    break
                if selected is None:
                    break
                node_id, edge_id = selected
                retired_edges: tuple[EdgeId, ...] = ()
                replacements: dict[EdgeId, tuple[EdgeId, ...]] = {}
                if edge_id is not None:
                    self._remove_edge_locked(edge_id)
                    retired_edges = (edge_id,)
                    replacements[edge_id] = ()
                self._remove_node_locked(node_id)
                delta = self._commit_locked(
                    [],
                    [],
                    retired_node_ids=(node_id,),
                    retired_edge_ids=retired_edges,
                    replacements=replacements,
                )
            deltas.append(delta)
        return tuple(deltas)

    def collapse_inactive_nearby_nodes(
        self,
        *,
        active_route_node_ids: Iterable[WaypointId] = (),
        active_route_edge_ids: Iterable[EdgeId] = (),
        active_gateway_ids: Iterable[WaypointId] = (),
        radius: float = 8.0,
        limit: int = 64,
    ) -> tuple[GraphDelta, ...]:
        """Contract short connected inactive nodes while preserving polylines.

        Spatial proximity alone is never sufficient: candidates must share a
        stored edge shorter than ``radius``, and every edge incident to the
        removable node must have the connector's source.  A protected anchor
        may absorb an ordinary turn/junction, but two protected anchors are
        never consolidated.  Active route identities remain untouched.
        """
        distance_limit = max(0.0, float(radius))
        maximum = max(0, int(limit))
        if distance_limit <= 0.0 or maximum == 0:
            return ()
        protected_nodes = {
            int(node_id) for node_id in active_route_node_ids
        } | {int(node_id) for node_id in active_gateway_ids}
        protected_edges = {int(edge_id) for edge_id in active_route_edge_ids}
        special_roles = {
            WaypointRole.HOME,
            WaypointRole.CHOKEPOINT,
            WaypointRole.RECOVERY_ANCHOR,
            WaypointRole.FRONTIER_GATEWAY,
        }
        removable_roles = {WaypointRole.TURN, WaypointRole.JUNCTION}
        deltas: list[GraphDelta] = []
        while len(deltas) < maximum:
            snapshot = self.snapshot()
            candidates = sorted(
                (
                    edge.cost,
                    edge.id,
                )
                for edge in snapshot.edges
                if edge.cost < distance_limit
                and math.dist(edge.start, edge.end) < distance_limit
            )
            contracted = False
            for _cost, edge_id in candidates:
                with self._lock:
                    edge = self._edges.get(edge_id)
                    if edge is None:
                        continue
                    left = self._nodes[edge.start_id]
                    right = self._nodes[edge.end_id]
                    if (
                        left.id in protected_nodes
                        or right.id in protected_nodes
                        or edge.id in protected_edges
                    ):
                        continue
                    left_special = bool(set(left.roles) & special_roles)
                    right_special = bool(set(right.roles) & special_roles)
                    if left_special and right_special:
                        continue
                    if left_special:
                        if (
                            not right.roles
                            or not set(right.roles).issubset(removable_roles)
                        ):
                            continue
                        survivor_id, removed_id = left.id, right.id
                    elif right_special:
                        if (
                            not left.roles
                            or not set(left.roles).issubset(removable_roles)
                        ):
                            continue
                        survivor_id, removed_id = right.id, left.id
                    else:
                        removable = tuple(
                            node.id for node in (left, right)
                            if node.roles
                            and set(node.roles).issubset(removable_roles)
                        )
                        if not removable:
                            continue
                        if len(removable) == 1:
                            removed_id = removable[0]
                            survivor_id = (
                                right.id if removed_id == left.id else left.id
                            )
                        else:
                            survivor_id = min(
                                (left.id, right.id),
                                key=lambda node_id: (
                                    -len(self._adjacency[node_id]),
                                    node_id,
                                ),
                            )
                            removed_id = (
                                right.id if survivor_id == left.id else left.id
                            )
                    incident_ids = tuple(self._adjacency[removed_id])
                    if protected_edges.intersection(incident_ids):
                        continue
                    incident_edges = tuple(
                        self._edges[item] for item in incident_ids
                    )
                    if any(
                        candidate.source != edge.source
                        or candidate.owner != edge.owner
                        for candidate in incident_edges
                    ):
                        continue
                    if any(
                        (
                            candidate.end_id
                            if candidate.start_id == removed_id
                            else candidate.start_id
                        ) == survivor_id
                        and candidate.id != edge.id
                        for candidate in incident_edges
                    ):
                        continue
                    delta = self._contract_connected_node_locked(
                        edge.id,
                        survivor_id=survivor_id,
                        removed_id=removed_id,
                    )
                if delta.retired_node_ids:
                    deltas.append(delta)
                    contracted = True
                    break
            if not contracted:
                break
        return tuple(deltas)

    def _contract_connected_node_locked(
        self,
        connector_edge_id: EdgeId,
        *,
        survivor_id: WaypointId,
        removed_id: WaypointId,
    ) -> GraphDelta:
        """Contract one already-validated connected node under ``_lock``."""
        connector = self._edges[connector_edge_id]
        connector_path = (
            connector.path
            if connector.start_id == survivor_id
            else tuple(reversed(connector.path))
        )
        incident = tuple(
            self._edges[edge_id]
            for edge_id in tuple(self._adjacency[removed_id])
        )
        rewired: list[tuple[WaypointEdge, WaypointId, WaypointPath]] = []
        for old_edge in incident:
            if old_edge.id == connector_edge_id:
                continue
            neighbor_id = (
                old_edge.end_id
                if old_edge.start_id == removed_id
                else old_edge.start_id
            )
            path_from_removed = (
                old_edge.path
                if old_edge.start_id == removed_id
                else tuple(reversed(old_edge.path))
            )
            rewired.append((
                old_edge,
                neighbor_id,
                _join_paths(connector_path, path_from_removed),
            ))

        retired_edge_ids = [edge.id for edge in incident]
        for old_edge in incident:
            self._remove_edge_locked(old_edge.id)
        removed = self._nodes[removed_id]
        survivor = self._nodes[survivor_id]
        self._remove_node_locked(removed_id)
        updated_node_ids: list[WaypointId] = []
        combined_roles = survivor.roles | removed.roles
        if combined_roles != survivor.roles:
            self._nodes[survivor_id] = WaypointNode(
                id=survivor.id,
                position=survivor.position,
                source=survivor.source,
                roles=combined_roles,
                created_revision=survivor.created_revision,
                updated_revision=self._version + 1,
            )
            updated_node_ids.append(survivor_id)

        additions: list[WaypointEdge] = []
        replacements: dict[EdgeId, tuple[EdgeId, ...]] = {
            connector_edge_id: (),
        }
        for old_edge, neighbor_id, joined_path in rewired:
            key = (*sorted((survivor_id, neighbor_id)), old_edge.source)
            previous_id = self._edge_keys.get(key)
            replacement = self._add_edge_locked(
                survivor_id,
                neighbor_id,
                joined_path,
                old_edge.source,
            )
            if replacement is not None:
                additions.append(replacement)
                replacements[old_edge.id] = (replacement.id,)
                if (
                    previous_id is not None
                    and previous_id not in retired_edge_ids
                    and previous_id not in self._edges
                ):
                    retired_edge_ids.append(previous_id)
                    replacements[previous_id] = (replacement.id,)
            else:
                existing_id = self._edge_keys.get(key)
                replacements[old_edge.id] = (
                    () if existing_id is None else (existing_id,)
                )
        return self._commit_locked(
            [],
            additions,
            updated_node_ids=updated_node_ids,
            retired_node_ids=(removed_id,),
            retired_edge_ids=retired_edge_ids,
            replacements=replacements,
        )

    def retire_frontier_gateway(
        self,
        gateway_id: WaypointId,
        *,
        active_route_node_ids: Iterable[WaypointId] = (),
        active_route_edge_ids: Iterable[EdgeId] = (),
    ) -> GraphDelta:
        """Remove one inactive cluster-only gateway and its orphan corridors."""
        protected_nodes = {int(node_id) for node_id in active_route_node_ids}
        protected_edges = {int(edge_id) for edge_id in active_route_edge_ids}
        with self._lock:
            node = self._nodes.get(int(gateway_id))
            if node is None:
                return GraphDelta(revision=self._version)
            incident = tuple(self._adjacency[node.id])
            if node.id in protected_nodes or protected_edges.intersection(incident):
                return GraphDelta(revision=self._version)

            orphan_sources = {EDGE_KNOWN_FREE_CORRIDOR, EDGE_SLAM_LOS}
            retired_edges = [
                edge_id for edge_id in incident
                if self._edges[edge_id].source in orphan_sources
            ]
            for edge_id in retired_edges:
                self._remove_edge_locked(edge_id)

            remaining_roles = frozenset(
                role for role in node.roles
                if role != WaypointRole.FRONTIER_GATEWAY
            )
            retired_nodes: list[WaypointId] = []
            updated_nodes: list[WaypointId] = []
            if not self._adjacency[node.id] and not remaining_roles:
                self._remove_node_locked(node.id)
                retired_nodes.append(node.id)
            elif remaining_roles != node.roles:
                self._nodes[node.id] = WaypointNode(
                    id=node.id,
                    position=node.position,
                    source=node.source,
                    roles=remaining_roles,
                    created_revision=node.created_revision,
                    updated_revision=self._version + 1,
                )
                updated_nodes.append(node.id)

            return self._commit_locked(
                [], [],
                updated_node_ids=updated_nodes,
                retired_node_ids=retired_nodes,
                retired_edge_ids=retired_edges,
            )

    def _split_edge_locked(
        self,
        edge_id: EdgeId,
        position: Position,
        added_nodes: List[WaypointNode],
        added_edges: List[WaypointEdge],
        retired_edges: List[EdgeId],
        replacements: Dict[EdgeId, Tuple[EdgeId, ...]],
    ) -> Optional[WaypointId]:
        edge = self._edges.get(edge_id)
        if edge is None:
            raise KeyError(edge_id)
        try:
            split_index = edge.path.index(position)
        except ValueError as exc:
            raise ValueError("junction must lie exactly on the edge polyline") from exc
        if split_index in {0, len(edge.path) - 1}:
            return None
        node_id, node = self._resolve_or_add_waypoint_locked(
            position, "junction", None,
        )
        if node is not None:
            self._add_role_locked(node.id, WaypointRole.JUNCTION)
            added_nodes.append(self._nodes[node.id])
        else:
            self._add_role_locked(node_id, WaypointRole.JUNCTION)
        self._remove_edge_locked(edge.id)
        first = self._add_edge_locked(
            edge.start_id, node_id, edge.path[: split_index + 1], edge.source,
        )
        second = self._add_edge_locked(
            node_id, edge.end_id, edge.path[split_index:], edge.source,
        )
        created = tuple(candidate for candidate in (first, second) if candidate is not None)
        added_edges.extend(created)
        retired_edges.append(edge.id)
        replacements[edge.id] = tuple(candidate.id for candidate in created)
        return node_id

    def _commit_locked(
        self,
        added_nodes: Sequence[WaypointNode],
        added_edges: Sequence[WaypointEdge],
        *,
        updated_node_ids: Sequence[WaypointId] = (),
        retired_node_ids: Sequence[WaypointId] = (),
        retired_edge_ids: Sequence[EdgeId] = (),
        replacements: Optional[Mapping[EdgeId, Tuple[EdgeId, ...]]] = None,
    ) -> GraphDelta:
        changed = bool(
            added_nodes or added_edges or updated_node_ids
            or retired_node_ids or retired_edge_ids
        )
        if not changed:
            return GraphDelta(revision=self._version)
        self._version += 1
        if retired_node_ids or retired_edge_ids:
            self._rebuild_components_locked()
        else:
            self._update_components_for_additions_locked(added_nodes, added_edges)
        delta = GraphDelta(
            revision=self._version,
            added_node_ids=tuple(node.id for node in added_nodes),
            updated_node_ids=tuple(updated_node_ids),
            retired_node_ids=tuple(retired_node_ids),
            added_edge_ids=tuple(edge.id for edge in added_edges),
            retired_edge_ids=tuple(retired_edge_ids),
            edge_replacements=dict(replacements or {}),
        )
        self._last_delta = delta
        return delta

    def _hash_cell(self, position: Position) -> Tuple[int, int]:
        return (
            position[0] // self.spatial_hash_cell,
            position[1] // self.spatial_hash_cell,
        )

    @staticmethod
    def _is_transverse_intersection(
        incoming: WaypointPath,
        existing: WaypointPath,
        point: Position,
    ) -> bool:
        """Reject collinear overlap; only a real crossing creates a junction."""
        incoming_index = incoming.index(point)
        existing_index = existing.index(point)
        incoming_vector = (
            incoming[incoming_index + 1][0] - incoming[incoming_index - 1][0],
            incoming[incoming_index + 1][1] - incoming[incoming_index - 1][1],
        )
        existing_vector = (
            existing[existing_index + 1][0] - existing[existing_index - 1][0],
            existing[existing_index + 1][1] - existing[existing_index - 1][1],
        )
        return (
            incoming_vector[0] * existing_vector[1]
            - incoming_vector[1] * existing_vector[0]
        ) != 0

    def _candidate_node_ids_locked(
        self, position: Position, radius: float,
    ) -> Set[WaypointId]:
        cell_x, cell_y = self._hash_cell(position)
        span = int(math.ceil(radius / self.spatial_hash_cell))
        result: Set[WaypointId] = set()
        for y in range(cell_y - span, cell_y + span + 1):
            for x in range(cell_x - span, cell_x + span + 1):
                result.update(self._node_hash.get((x, y), ()))
        return result

    def _remove_uncommitted_node_locked(self, node_id: WaypointId) -> None:
        self._remove_node_locked(node_id)

    def _remove_node_locked(self, node_id: WaypointId) -> None:
        node = self._nodes.pop(node_id)
        self._position_ids.pop(node.position, None)
        bucket = self._node_hash.get(self._hash_cell(node.position))
        if bucket is not None:
            bucket.discard(node_id)
            if not bucket:
                self._node_hash.pop(self._hash_cell(node.position), None)
        self._adjacency.pop(node_id, None)

    def _remove_edge_locked(self, edge_id: EdgeId) -> WaypointEdge:
        edge = self._edges.pop(edge_id)
        self._edge_keys.pop((edge.start_id, edge.end_id, edge.source), None)
        self._adjacency[edge.start_id].discard(edge_id)
        self._adjacency[edge.end_id].discard(edge_id)
        for point in edge.path:
            cell = self._hash_cell(point)
            bucket = self._edge_hash.get(cell)
            if bucket is not None:
                bucket.discard(edge_id)
                if not bucket:
                    self._edge_hash.pop(cell, None)
        return edge

    def _rebuild_components_locked(self) -> None:
        self._components.clear()
        component = 0
        for root in self._nodes:
            if root in self._components:
                continue
            component += 1
            stack = [root]
            self._components[root] = component
            while stack:
                current = stack.pop()
                for edge_id in self._adjacency[current]:
                    edge = self._edges[edge_id]
                    neighbor = edge.end_id if edge.start_id == current else edge.start_id
                    if neighbor not in self._components:
                        self._components[neighbor] = component
                        stack.append(neighbor)

    def _update_components_for_additions_locked(
        self,
        added_nodes: Sequence[WaypointNode],
        added_edges: Sequence[WaypointEdge],
    ) -> None:
        """Maintain components incrementally for the common addition-only case."""
        next_component = max(self._components.values(), default=0) + 1
        for node in added_nodes:
            if node.id not in self._components:
                self._components[node.id] = next_component
                next_component += 1
        for edge in added_edges:
            start_component = self._components[edge.start_id]
            end_component = self._components[edge.end_id]
            if start_component == end_component:
                continue
            keep = min(start_component, end_component)
            replace = max(start_component, end_component)
            for node_id, component in tuple(self._components.items()):
                if component == replace:
                    self._components[node_id] = keep

    def _build_reverse_tree_locked(
        self, root: WaypointId, known_free: np.ndarray,
    ) -> _ReverseTree:
        """Build the one-root compatibility form of a reverse route tree."""
        return self._build_multi_goal_tree_locked(
            ((root, (self._nodes[root].position,)),),
            known_free,
        )

    def _build_multi_goal_tree_locked(
        self,
        goals: Sequence[tuple[WaypointId, WaypointPath]],
        known_free: np.ndarray,
    ) -> _ReverseTree:
        """Build one exact reverse tree seeded by all goal connectors.

        Seed distances include each ephemeral goal-connector cost. Dijkstra
        therefore selects the same minimum over connector/tree combinations as
        separate one-root searches while traversing persistent edges only once.
        """
        distances: Dict[WaypointId, float] = {}
        next_edges: Dict[WaypointId, EdgeId] = {}
        roots: Dict[WaypointId, WaypointId] = {}
        sequence = itertools.count()
        heap: List[Tuple[float, int, WaypointId]] = []
        for root, connector_path in goals:
            connector_cost = _path_cost(connector_path)
            previous = distances.get(root)
            if previous is not None and previous <= connector_cost + 1e-9:
                continue
            distances[root] = connector_cost
            roots[root] = root
            heapq.heappush(
                heap,
                (connector_cost, next(sequence), root),
            )
        while heap:
            distance, _order, current = heapq.heappop(heap)
            if distance > distances.get(current, math.inf) + 1e-9:
                continue
            for edge_id in self._adjacency[current]:
                edge = self._edges[edge_id]
                if (
                    edge.source in {EDGE_SLAM_LOS, EDGE_KNOWN_FREE_CORRIDOR}
                    and not validate_known_free_path(edge.path, known_free)
                ):
                    continue
                neighbor = edge.end_id if edge.start_id == current else edge.start_id
                candidate = distance + edge.cost
                if candidate + 1e-9 >= distances.get(neighbor, math.inf):
                    continue
                distances[neighbor] = candidate
                next_edges[neighbor] = edge_id
                roots[neighbor] = roots[current]
                heapq.heappush(heap, (candidate, next(sequence), neighbor))
        return _ReverseTree(
            distances=distances,
            next_edges=next_edges,
            roots=roots,
        )

    def _belief_validity_signature_locked(
        self,
        known_free: np.ndarray,
    ) -> tuple[EdgeId, ...]:
        """Key cached trees by requester-relevant persistent edge validity.

        Unrelated SLAM version changes can reuse a tree, while any corridor
        becoming valid or invalid produces a different key and a fresh build.
        Travelled edges need no belief signature because executed geometry is
        authoritative.
        """
        return tuple(
            edge.id
            for edge in self._edges.values()
            if edge.source in {EDGE_SLAM_LOS, EDGE_KNOWN_FREE_CORRIDOR}
            and validate_known_free_path(edge.path, known_free)
        )

    def _trim_route_cache_locked(self) -> None:
        while len(self._route_cache) > self.route_cache_capacity:
            self._route_cache.popitem(last=False)

    def _tree_route_valid_locked(
        self,
        start: WaypointId,
        tree: _ReverseTree,
        known_free: np.ndarray,
    ) -> bool:
        """Optimistically reuse a tree, checking only scoped selected edges."""
        current = start
        root = tree.roots.get(start)
        if root is None:
            return False
        seen: Set[WaypointId] = set()
        while current != root:
            if current in seen or current not in tree.next_edges:
                return False
            seen.add(current)
            edge = self._edges.get(tree.next_edges[current])
            if edge is None:
                return False
            if (
                edge.source in {EDGE_SLAM_LOS, EDGE_KNOWN_FREE_CORRIDOR}
                and not validate_known_free_path(edge.path, known_free)
            ):
                return False
            current = edge.end_id if edge.start_id == current else edge.start_id
        return True

    def _materialize_cached_route_locked(
        self,
        start_position: Position,
        goal_position: Position,
        selected: Tuple[
            float,
            WaypointId,
            WaypointPath,
            _ReverseTree,
            bool,
        ],
        goal_paths: Mapping[WaypointId, WaypointPath],
        requester_knowledge_revision: int,
        route_id: int,
        *,
        connector_astar_calls: int = 0,
    ) -> WaypointRoute:
        total, current, start_path, tree, cache_hit = selected
        goal_id = tree.roots[current]
        goal_to_node = goal_paths[goal_id]
        node_ids: List[WaypointId] = [current]
        edge_ids: List[EdgeId] = []
        persistent_paths: List[Tuple[WaypointPath, str, float]] = []
        while current != goal_id:
            edge_id = tree.next_edges[current]
            edge = self._edges[edge_id]
            path = edge.path if edge.start_id == current else tuple(reversed(edge.path))
            current = edge.end_id if edge.start_id == current else edge.start_id
            edge_ids.append(edge_id)
            node_ids.append(current)
            persistent_paths.append((path, edge.source, edge.cost))
        goal_path = tuple(reversed(goal_to_node))
        segments: List[Tuple[WaypointPath, str, float, Optional[EdgeId]]] = []
        if len(set(start_path)) > 1:
            segments.append((
                start_path, EDGE_KNOWN_FREE_CONNECTOR,
                _path_cost(start_path), None,
            ))
        segments.extend(
            (path, source, cost, edge_id)
            for (path, source, cost), edge_id in zip(
                persistent_paths, edge_ids
            )
        )
        if len(set(goal_path)) > 1:
            segments.append((
                goal_path, EDGE_KNOWN_FREE_CONNECTOR,
                _path_cost(goal_path), None,
            ))
        first_path, first_source, first_cost, _first_edge_id = segments[0]
        waypoints = [start_position]
        waypoints.extend(self._nodes[node_id].position for node_id in node_ids)
        waypoints.append(goal_position)
        deduplicated = tuple(dict.fromkeys(waypoints))
        return WaypointRoute(
            status=ROUTE_OK,
            id=route_id,
            waypoints=deduplicated,
            first_segment_path=first_path,
            first_segment_source=first_source,
            first_segment_cost=first_cost,
            cost=total,
            topology_revision=self._version,
            requester_knowledge_revision=requester_knowledge_revision,
            node_ids=tuple(node_ids),
            edge_ids=tuple(edge_ids),
            cache_hit=cache_hit,
            entry_connector_path=start_path,
            exit_connector_path=goal_path,
            remaining_cost=total,
            segment_paths=tuple(segment[0] for segment in segments),
            segment_sources=tuple(segment[1] for segment in segments),
            segment_edge_ids=tuple(segment[3] for segment in segments),
            connector_astar_calls=connector_astar_calls,
        )

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
            segment_paths=tuple(edge.path for edge in edges),
            segment_sources=tuple(edge.source for edge in edges),
            segment_edge_ids=tuple(None for _edge in edges),
        )


def graph_delta_trace_fields(
    graph: WaypointGraph,
    delta: GraphDelta,
) -> dict[str, object]:
    """Serialize one committed graph mutation for canonical runtime tracing."""
    snapshot = graph.snapshot()
    nodes = {node.id: node for node in snapshot.nodes}
    edges = {edge.id: edge for edge in snapshot.edges}

    def node_record(node: WaypointNode) -> dict[str, object]:
        return {
            "node_id": int(node.id),
            "position": node.position,
            "source": node.source,
            "roles": tuple(sorted(role.value for role in node.roles)),
            "created_revision": int(node.created_revision),
            "updated_revision": int(node.updated_revision),
        }

    def edge_record(edge: WaypointEdge) -> dict[str, object]:
        return {
            "edge_id": int(edge.id),
            "start_node_id": int(edge.start_id),
            "end_node_id": int(edge.end_id),
            "start": edge.start,
            "end": edge.end,
            "source": edge.source,
            "cost": float(edge.cost),
            "path": edge.path,
            "created_revision": int(edge.created_revision),
        }

    return {
        "topology_revision": int(delta.revision),
        "added_nodes": tuple(
            node_record(nodes[node_id])
            for node_id in delta.added_node_ids
            if node_id in nodes
        ),
        "updated_nodes": tuple(
            node_record(nodes[node_id])
            for node_id in delta.updated_node_ids
            if node_id in nodes
        ),
        "retired_node_ids": tuple(delta.retired_node_ids),
        "added_edges": tuple(
            edge_record(edges[edge_id])
            for edge_id in delta.added_edge_ids
            if edge_id in edges
        ),
        "updated_edges": tuple(
            edge_record(edges[edge_id])
            for edge_id in delta.updated_edge_ids
            if edge_id in edges
        ),
        "retired_edge_ids": tuple(delta.retired_edge_ids),
        "edge_replacements": {
            int(edge_id): tuple(replacements)
            for edge_id, replacements in delta.edge_replacements.items()
        },
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
    }
