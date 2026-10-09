"""Versioned rover-SLAM highway graphs and bounded route queries.

The highway is navigation advice derived only from a detached rover SLAM
snapshot. It never owns exploration work. Corridor snapshots contain a sparse
branching backbone and exact, locally validated polylines. The tiled builder
is retained for offline comparisons and historical snapshot compatibility.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, fields, replace
from typing import Iterable
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np

from mapping.slam_map import FREE, SlamSnapshot


Position = tuple[int, int]

HIGHWAY_COMPLETE = "complete"
HIGHWAY_UNAVAILABLE = "unavailable"
HIGHWAY_UNREACHABLE = "unreachable"
HIGHWAY_STALE = "stale"
HIGHWAY_BUDGET_EXHAUSTED = "budget_exhausted"

_NEIGHBORS = (
    (-1, -1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, 1, 1.0),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class HighwayEdge:
    """One directed graph edge with an exact rover-known-free polyline."""

    target_area: int
    cost: float
    path: tuple[Position, ...]


@dataclass(frozen=True)
class HighwayRoute:
    """One bounded route-query result."""

    status: str
    path: tuple[Position, ...] = ()
    cost: float = math.inf
    snapshot_version: int = -1
    elapsed_ms: float = 0.0
    expanded_nodes: int = 0
    graph_edges: int = 0

    @property
    def complete(self) -> bool:
        return self.status == HIGHWAY_COMPLETE


@dataclass(frozen=True)
class HighwayBuildResult:
    """Outcome of one bounded graph build."""

    status: str
    snapshot: HighwayGraphSnapshot | None
    elapsed_ms: float
    area_count: int = 0
    edge_count: int = 0
    known_free_cells: int = 0
    connector_expansions: int = 0
    source_version: int | None = None
    build_kind: str | None = None


@dataclass(frozen=True)
class HighwaySegment:
    """One undirected backbone branch, including its intermediate pixels."""

    source: int
    target: int
    path: tuple[Position, ...]


@dataclass(frozen=True)
class HighwayGraphSnapshot:
    """Immutable macro-area graph published at verified physical contact."""

    version: int
    origin: Position
    full_shape: tuple[int, int]
    macro_cell_size: int
    confidence_threshold: float
    area_labels: np.ndarray
    anchors: tuple[Position, ...]
    adjacency: tuple[tuple[HighwayEdge, ...], ...]
    known_free_cells: int
    build_elapsed_ms: float
    access_predecessors: np.ndarray | None = None
    segments: tuple[HighwaySegment, ...] = ()
    network_locations: Mapping[Position, tuple[int, int]] | None = None
    network_nodes: Mapping[Position, int] | None = None
    maximum_access_distance: float = 0.0
    measured_access_distance: float = 0.0
    component_count: int = 0
    pruned_branches: int = 0
    capillary_branches: int = 0
    skeleton_scale: int = 1
    skeleton_method: str = "medial_axis"

    def __post_init__(self) -> None:
        labels = np.asarray(self.area_labels, dtype=np.int32)
        if labels.ndim != 2:
            raise ValueError("highway area labels must be two-dimensional")
        if len(self.anchors) != len(self.adjacency):
            raise ValueError("highway anchors and adjacency must align")
        if labels.flags.writeable:
            labels = labels.copy()
            labels.setflags(write=False)
            object.__setattr__(self, "area_labels", labels)
        if self.access_predecessors is not None:
            predecessors = np.asarray(self.access_predecessors, dtype=np.int8)
            if predecessors.shape != labels.shape:
                raise ValueError("highway access field must match its SLAM window")
            if predecessors.flags.writeable:
                predecessors = predecessors.copy()
                predecessors.setflags(write=False)
            object.__setattr__(self, "access_predecessors", predecessors)
        for name in ("network_locations", "network_nodes"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name) or {})))

    def __reduce__(self):
        # MappingProxyType is deliberately immutable but is not picklable.
        values = [dict(getattr(self, field.name)) if field.name in
                  {"network_locations", "network_nodes"} else getattr(self, field.name)
                  for field in fields(self)]
        return (type(self), tuple(values))

    @property
    def area_count(self) -> int:
        return max(0, len(self.anchors) - 1)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.adjacency) // 2

    def route(
        self,
        start: Position,
        goal: Position,
        *,
        maximum_query_ms: float,
        maximum_connector_expansions: int,
        required_version: int | None = None,
    ) -> HighwayRoute:
        """Return a bounded exact-polyline route through this snapshot."""
        if self.access_predecessors is not None:
            from navigation.highway_backbone import route_backbone
            return route_backbone(
                self, start, goal, maximum_query_ms=maximum_query_ms,
                maximum_connector_expansions=maximum_connector_expansions,
                required_version=required_version,
            )
        started = time.perf_counter()
        if required_version is not None and self.version != int(required_version):
            return HighwayRoute(
                HIGHWAY_STALE,
                snapshot_version=self.version,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        budget_ms = max(0.0, float(maximum_query_ms))
        deadline = started + budget_ms / 1000.0
        start_local = self._local(start)
        goal_local = self._local(goal)
        start_area = self._area_at(start_local)
        goal_area = self._area_at(goal_local)
        if start_area <= 0 or goal_area <= 0:
            return self._result(HIGHWAY_UNAVAILABLE, started)
        if start == goal:
            return self._result(
                HIGHWAY_COMPLETE,
                started,
                path=(tuple(start),),
                cost=0.0,
            )
        direct = _direct_labeled_path(
            self.area_labels,
            start_local,
            goal_local,
            area_id=None,
        )
        if direct:
            world_path = self._world_path(direct)
            return self._result(
                HIGHWAY_COMPLETE,
                started,
                path=world_path,
                cost=_path_cost(world_path),
            )

        maximum_expansions = max(1, int(maximum_connector_expansions))
        if start_area == goal_area:
            path, expansions, exhausted = _area_path(
                self.area_labels,
                start_area,
                start_local,
                goal_local,
                deadline=deadline,
                maximum_expansions=maximum_expansions,
            )
            if exhausted:
                return self._result(
                    HIGHWAY_BUDGET_EXHAUSTED,
                    started,
                    expanded_nodes=expansions,
                )
            if not path:
                return self._result(
                    HIGHWAY_UNREACHABLE,
                    started,
                    expanded_nodes=expansions,
                )
            world_path = self._world_path(path)
            return self._result(
                HIGHWAY_COMPLETE,
                started,
                path=world_path,
                cost=_path_cost(world_path),
                expanded_nodes=expansions,
            )

        start_anchor = self._local(self.anchors[start_area])
        goal_anchor = self._local(self.anchors[goal_area])
        start_connector, start_expansions, start_exhausted = _area_path(
            self.area_labels,
            start_area,
            start_local,
            start_anchor,
            deadline=deadline,
            maximum_expansions=maximum_expansions,
        )
        remaining_expansions = max(1, maximum_expansions - start_expansions)
        goal_connector, goal_expansions, goal_exhausted = _area_path(
            self.area_labels,
            goal_area,
            goal_anchor,
            goal_local,
            deadline=deadline,
            maximum_expansions=remaining_expansions,
        )
        connector_expansions = start_expansions + goal_expansions
        if start_exhausted or goal_exhausted or time.perf_counter() > deadline:
            return self._result(
                HIGHWAY_BUDGET_EXHAUSTED,
                started,
                expanded_nodes=connector_expansions,
            )
        if not start_connector or not goal_connector:
            return self._result(
                HIGHWAY_UNREACHABLE,
                started,
                expanded_nodes=connector_expansions,
            )

        graph_cost = {start_area: 0.0}
        parents: dict[int, tuple[int, HighwayEdge]] = {}
        queue: list[tuple[float, int]] = [(0.0, start_area)]
        graph_expansions = 0
        while queue:
            if time.perf_counter() > deadline:
                return self._result(
                    HIGHWAY_BUDGET_EXHAUSTED,
                    started,
                    expanded_nodes=connector_expansions + graph_expansions,
                )
            cost, area_id = heapq.heappop(queue)
            if cost > graph_cost.get(area_id, math.inf) + 1e-9:
                continue
            graph_expansions += 1
            if area_id == goal_area:
                break
            for edge in self.adjacency[area_id]:
                next_cost = cost + edge.cost
                if next_cost + 1e-9 >= graph_cost.get(
                    edge.target_area,
                    math.inf,
                ):
                    continue
                graph_cost[edge.target_area] = next_cost
                parents[edge.target_area] = (area_id, edge)
                heapq.heappush(queue, (next_cost, edge.target_area))
        if goal_area not in graph_cost:
            return self._result(
                HIGHWAY_UNREACHABLE,
                started,
                expanded_nodes=connector_expansions + graph_expansions,
            )

        edges: list[HighwayEdge] = []
        current = goal_area
        while current != start_area:
            parent, edge = parents[current]
            edges.append(edge)
            current = parent
        edges.reverse()
        path: list[Position] = list(self._world_path(start_connector))
        for edge in edges:
            _extend_path(path, edge.path)
        _extend_path(path, self._world_path(goal_connector))
        final_path = tuple(path)
        return self._result(
            HIGHWAY_COMPLETE,
            started,
            path=final_path,
            cost=_path_cost(final_path),
            expanded_nodes=connector_expansions + graph_expansions,
            graph_edges=len(edges),
        )

    def _local(self, point: Position) -> Position:
        return (
            int(point[0]) - int(self.origin[0]),
            int(point[1]) - int(self.origin[1]),
        )

    def _world_path(self, path: Iterable[Position]) -> tuple[Position, ...]:
        offset_x, offset_y = self.origin
        return tuple(
            (int(x) + int(offset_x), int(y) + int(offset_y))
            for x, y in path
        )

    def _area_at(self, point: Position) -> int:
        x, y = point
        if not (
            0 <= y < self.area_labels.shape[0]
            and 0 <= x < self.area_labels.shape[1]
        ):
            return 0
        return int(self.area_labels[y, x])

    def _result(
        self,
        status: str,
        started: float,
        *,
        path: tuple[Position, ...] = (),
        cost: float = math.inf,
        expanded_nodes: int = 0,
        graph_edges: int = 0,
    ) -> HighwayRoute:
        return HighwayRoute(
            status=status,
            path=path,
            cost=float(cost),
            snapshot_version=self.version,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            expanded_nodes=int(expanded_nodes),
            graph_edges=int(graph_edges),
        )


def build_highway_graph(
    slam: SlamSnapshot,
    *,
    confidence_threshold: float,
    macro_cell_size: int,
    maximum_build_ms: float,
    maximum_connector_expansions: int,
    maximum_access_distance: float = 80.0,
    preparation_cache=None,
) -> HighwayBuildResult:
    """Build bounded corridor advice; the old tile size remains readable."""
    from navigation.highway_backbone import build_backbone
    return build_backbone(
        slam, confidence_threshold=confidence_threshold,
        macro_cell_size=macro_cell_size, maximum_build_ms=maximum_build_ms,
        maximum_access_distance=maximum_access_distance,
        preparation_cache=preparation_cache,
    )


def build_tiled_highway_graph(
    slam: SlamSnapshot,
    *,
    confidence_threshold: float,
    macro_cell_size: int,
    maximum_build_ms: float,
    maximum_connector_expansions: int,
) -> HighwayBuildResult:
    """Build a complete immutable graph or discard the bounded attempt."""
    started = time.perf_counter()
    budget_ms = max(0.0, float(maximum_build_ms))
    deadline = started + budget_ms / 1000.0
    occupancy = np.asarray(slam.occupancy)
    confidence = np.asarray(slam.confidence)
    known_free = (
        (occupancy == FREE)
        & (confidence >= float(confidence_threshold))
    )
    height, width = known_free.shape
    labels = np.zeros((height, width), dtype=np.int32)
    anchors: list[Position] = [(0, 0)]
    cell_size = max(2, int(macro_cell_size))
    offset_x, offset_y = (int(value) for value in slam.origin)

    for top in range(0, height, cell_size):
        for left in range(0, width, cell_size):
            if time.perf_counter() > deadline:
                return _aborted_build(started, known_free, len(anchors) - 1)
            bottom = min(height, top + cell_size)
            right = min(width, left + cell_size)
            tile = known_free[top:bottom, left:right]
            if not np.any(tile):
                continue
            count, tile_labels = cv2.connectedComponents(
                tile.astype(np.uint8),
                connectivity=4,
            )
            for local_label in range(1, count):
                ys, xs = np.nonzero(tile_labels == local_label)
                if not len(xs):
                    continue
                area_id = len(anchors)
                labels[top:bottom, left:right][tile_labels == local_label] = area_id
                mean_x = float(xs.mean())
                mean_y = float(ys.mean())
                nearest = int(np.argmin(
                    (xs.astype(float) - mean_x) ** 2
                    + (ys.astype(float) - mean_y) ** 2
                ))
                anchors.append((
                    int(left + xs[nearest] + offset_x),
                    int(top + ys[nearest] + offset_y),
                ))

    adjacency: list[list[HighwayEdge]] = [list() for _ in anchors]
    connector_expansions = 0
    edge_count = 0
    for first_area, second_area, first_point, second_point in _portal_runs(
        labels,
        cell_size,
    ):
        if time.perf_counter() > deadline:
            return _aborted_build(
                started,
                known_free,
                len(anchors) - 1,
                edge_count=edge_count,
                connector_expansions=connector_expansions,
            )
        first_anchor = (
            anchors[first_area][0] - offset_x,
            anchors[first_area][1] - offset_y,
        )
        second_anchor = (
            anchors[second_area][0] - offset_x,
            anchors[second_area][1] - offset_y,
        )
        first_path, first_expansions, first_exhausted = _area_path(
            labels,
            first_area,
            first_anchor,
            first_point,
            deadline=deadline,
            maximum_expansions=max(1, int(maximum_connector_expansions)),
        )
        connector_expansions += first_expansions
        remaining = max(
            1,
            int(maximum_connector_expansions) - first_expansions,
        )
        second_path, second_expansions, second_exhausted = _area_path(
            labels,
            second_area,
            second_point,
            second_anchor,
            deadline=deadline,
            maximum_expansions=remaining,
        )
        connector_expansions += second_expansions
        if first_exhausted or second_exhausted:
            return _aborted_build(
                started,
                known_free,
                len(anchors) - 1,
                edge_count=edge_count,
                connector_expansions=connector_expansions,
            )
        if not first_path or not second_path:
            continue
        local_path = tuple((*first_path, *second_path))
        world_path = tuple(
            (x + offset_x, y + offset_y) for x, y in local_path
        )
        cost = _path_cost(world_path)
        adjacency[first_area].append(HighwayEdge(
            target_area=second_area,
            cost=cost,
            path=world_path,
        ))
        adjacency[second_area].append(HighwayEdge(
            target_area=first_area,
            cost=cost,
            path=tuple(reversed(world_path)),
        ))
        edge_count += 1

    labels.setflags(write=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    snapshot = HighwayGraphSnapshot(
        version=int(slam.version),
        origin=(offset_x, offset_y),
        full_shape=(
            tuple(int(value) for value in slam.full_shape)
            if slam.full_shape is not None
            else (height, width)
        ),
        macro_cell_size=cell_size,
        confidence_threshold=float(confidence_threshold),
        area_labels=labels,
        anchors=tuple(anchors),
        adjacency=tuple(
            tuple(sorted(
                edges,
                key=lambda edge: (edge.target_area, edge.cost, edge.path),
            ))
            for edges in adjacency
        ),
        known_free_cells=int(np.count_nonzero(known_free)),
        build_elapsed_ms=elapsed_ms,
    )
    return HighwayBuildResult(
        status=HIGHWAY_COMPLETE,
        snapshot=snapshot,
        elapsed_ms=elapsed_ms,
        area_count=snapshot.area_count,
        edge_count=snapshot.edge_count,
        known_free_cells=snapshot.known_free_cells,
        connector_expansions=connector_expansions,
    )


class HighwayService:
    """Own the latest complete rover graph; never publish partial rebuilds."""

    def __init__(
        self,
        *,
        confidence_threshold: float,
        macro_cell_size: int,
        maximum_build_ms: float,
        maximum_query_ms: float,
        maximum_connector_expansions: int,
        maximum_access_distance_sensor_ranges: float = 2.0,
        sensor_range: float = 40.0,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.macro_cell_size = max(2, int(macro_cell_size))
        self.maximum_build_ms = max(0.0, float(maximum_build_ms))
        self.maximum_query_ms = max(0.0, float(maximum_query_ms))
        self.maximum_connector_expansions = max(
            1,
            int(maximum_connector_expansions),
        )
        self.maximum_access_distance_sensor_ranges = float(maximum_access_distance_sensor_ranges)
        self.sensor_range = float(sensor_range)
        self._snapshot: HighwayGraphSnapshot | None = None
        self._worker = None

    def start(self) -> None:
        """Enable nonblocking throttled full rebuilds for a running mission."""
        if self._worker is None:
            from navigation.highway_worker import HighwayWorker
            self._worker = HighwayWorker(dict(
                confidence_threshold=self.confidence_threshold,
                macro_cell_size=self.macro_cell_size,
                maximum_build_ms=self.maximum_build_ms,
                maximum_access_distance=self.maximum_access_distance_sensor_ranges * self.sensor_range,
            ))

    def poll(self) -> HighwayBuildResult | None:
        result = None if self._worker is None else self._worker.poll(self._snapshot)
        if result is not None and result.snapshot is not None:
            self._snapshot = result.snapshot
        return result

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None

    @property
    def snapshot(self) -> HighwayGraphSnapshot | None:
        return self._snapshot

    def refresh(self, slam: SlamSnapshot) -> HighwayBuildResult:
        if self._worker is not None:
            result = self._worker.request(slam, self._snapshot)
            if result.snapshot is not None:
                self._snapshot = result.snapshot
            return result
        current = self._snapshot
        if current is not None and current.version == int(slam.version):
            return HighwayBuildResult(
                status="cached",
                snapshot=current,
                elapsed_ms=0.0,
                area_count=current.area_count,
                edge_count=current.edge_count,
                known_free_cells=current.known_free_cells,
            )
        result = build_highway_graph(
            slam,
            confidence_threshold=self.confidence_threshold,
            macro_cell_size=self.macro_cell_size,
            maximum_build_ms=self.maximum_build_ms,
            maximum_connector_expansions=self.maximum_connector_expansions,
            maximum_access_distance=(
                self.maximum_access_distance_sensor_ranges * self.sensor_range
            ),
        )
        if result.snapshot is not None:
            self._snapshot = result.snapshot
        return result

    def route(
        self,
        start: Position,
        goal: Position,
        *,
        required_version: int | None = None,
    ) -> HighwayRoute:
        snapshot = self._snapshot
        if snapshot is None:
            return HighwayRoute(HIGHWAY_UNAVAILABLE)
        return snapshot.route(
            start,
            goal,
            maximum_query_ms=self.maximum_query_ms,
            maximum_connector_expansions=self.maximum_connector_expansions,
            required_version=required_version,
        )


def _portal_runs(
    labels: np.ndarray,
    cell_size: int,
) -> Iterable[tuple[int, int, Position, Position]]:
    """Yield one representative crossing per contiguous area-pair boundary."""
    height, width = labels.shape
    for x in range(cell_size, width, cell_size):
        run: list[tuple[int, int, int]] = []
        previous_pair: tuple[int, int] | None = None
        for y in range(height):
            pair = (int(labels[y, x - 1]), int(labels[y, x]))
            valid = pair[0] > 0 and pair[1] > 0 and pair[0] != pair[1]
            if valid and pair == previous_pair:
                run.append((y, pair[0], pair[1]))
                continue
            if run:
                middle = run[len(run) // 2]
                yield middle[1], middle[2], (x - 1, middle[0]), (x, middle[0])
            run = [(y, pair[0], pair[1])] if valid else []
            previous_pair = pair if valid else None
        if run:
            middle = run[len(run) // 2]
            yield middle[1], middle[2], (x - 1, middle[0]), (x, middle[0])

    for y in range(cell_size, height, cell_size):
        run = []
        previous_pair = None
        for x in range(width):
            pair = (int(labels[y - 1, x]), int(labels[y, x]))
            valid = pair[0] > 0 and pair[1] > 0 and pair[0] != pair[1]
            if valid and pair == previous_pair:
                run.append((x, pair[0], pair[1]))
                continue
            if run:
                middle = run[len(run) // 2]
                yield middle[1], middle[2], (middle[0], y - 1), (middle[0], y)
            run = [(x, pair[0], pair[1])] if valid else []
            previous_pair = pair if valid else None
        if run:
            middle = run[len(run) // 2]
            yield middle[1], middle[2], (middle[0], y - 1), (middle[0], y)


def _area_path(
    labels: np.ndarray,
    area_id: int,
    start: Position,
    goal: Position,
    *,
    deadline: float,
    maximum_expansions: int,
) -> tuple[tuple[Position, ...], int, bool]:
    """Run bounded A* inside one macro-area with strict safe diagonals."""
    height, width = labels.shape
    if not all(
        0 <= x < width and 0 <= y < height and int(labels[y, x]) == area_id
        for x, y in (start, goal)
    ):
        return (), 0, False
    if start == goal:
        return (start,), 0, False
    direct = _direct_labeled_path(
        labels,
        start,
        goal,
        area_id=area_id,
    )
    if direct:
        return direct, 0, False

    def heuristic(point: Position) -> float:
        dx = abs(point[0] - goal[0])
        dy = abs(point[1] - goal[1])
        return dx + dy + (math.sqrt(2.0) - 2.0) * min(dx, dy)

    queue: list[tuple[float, float, int, int]] = [
        (heuristic(start), 0.0, start[0], start[1]),
    ]
    best = {start: 0.0}
    parents: dict[Position, Position] = {}
    expansions = 0
    while queue:
        if time.perf_counter() > deadline or expansions >= maximum_expansions:
            return (), expansions, True
        _score, cost, x, y = heapq.heappop(queue)
        current = (x, y)
        if cost > best.get(current, math.inf) + 1e-9:
            continue
        expansions += 1
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(parents[path[-1]])
            return tuple(reversed(path)), expansions, False
        for dx, dy, step in _NEIGHBORS:
            next_x, next_y = x + dx, y + dy
            if not (
                0 <= next_x < width
                and 0 <= next_y < height
                and int(labels[next_y, next_x]) == area_id
            ):
                continue
            if dx != 0 and dy != 0 and (
                int(labels[y, next_x]) != area_id
                or int(labels[next_y, x]) != area_id
            ):
                continue
            next_point = (next_x, next_y)
            next_cost = cost + step
            if next_cost + 1e-9 >= best.get(next_point, math.inf):
                continue
            best[next_point] = next_cost
            parents[next_point] = current
            heapq.heappush(
                queue,
                (
                    next_cost + heuristic(next_point),
                    next_cost,
                    next_x,
                    next_y,
                ),
            )
    return (), expansions, False


def _direct_labeled_path(
    labels: np.ndarray,
    start: Position,
    goal: Position,
    *,
    area_id: int | None,
    deadline: float = math.inf,
) -> tuple[Position, ...]:
    """Return an exact straight connector through authorized labeled cells."""
    x0, y0 = start
    x1, y1 = goal
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    points: list[Position] = []
    height, width = labels.shape

    def inside(point: Position) -> bool:
        x, y = point
        label = int(labels[y, x]) if 0 <= x < width and 0 <= y < height else 0
        return bool(
            label > 0
            and (area_id is None or label == area_id)
        )

    while True:
        if len(points) % 64 == 0 and time.perf_counter() > deadline:
            return ()
        point = (x0, y0)
        if not inside(point):
            return ()
        points.append(point)
        if point == goal:
            return tuple(points)
        previous = point
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += step_x
        if doubled <= dx:
            error += dx
            y0 += step_y
        if x0 != previous[0] and y0 != previous[1] and not (
            inside((x0, previous[1]))
            and inside((previous[0], y0))
        ):
            return ()


def _path_cost(path: Iterable[Position]) -> float:
    points = tuple(path)
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


def _extend_path(target: list[Position], extension: Iterable[Position]) -> None:
    for point in extension:
        normalized = (int(point[0]), int(point[1]))
        if not target or target[-1] != normalized:
            target.append(normalized)


def _aborted_build(
    started: float,
    known_free: np.ndarray,
    area_count: int,
    *,
    edge_count: int = 0,
    connector_expansions: int = 0,
) -> HighwayBuildResult:
    return HighwayBuildResult(
        status=HIGHWAY_BUDGET_EXHAUSTED,
        snapshot=None,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        area_count=int(area_count),
        edge_count=int(edge_count),
        known_free_cells=int(np.count_nonzero(known_free)),
        connector_expansions=int(connector_expansions),
    )
