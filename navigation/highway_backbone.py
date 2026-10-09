"""Corridor skeletons, bounded capillary coverage, and safe route shortcuts."""

from __future__ import annotations

import heapq
import math
import time

import cv2
import numpy as np
from skimage.graph import MCP
from skimage.morphology import medial_axis, skeletonize

from mapping.slam_map import FREE, SlamSnapshot
from navigation.highway import (
    HIGHWAY_BUDGET_EXHAUSTED, HIGHWAY_COMPLETE, HIGHWAY_STALE,
    HIGHWAY_UNAVAILABLE, HIGHWAY_UNREACHABLE, HighwayBuildResult,
    HighwayEdge, HighwayGraphSnapshot, HighwayRoute, HighwaySegment,
    _direct_labeled_path, _extend_path, _path_cost,
)


_OFFSETS = ((-1, 0), (0, -1), (0, 1), (1, 0))  # MCP uses row, column.
_DIRECTIONS = ((-1, 0), (0, -1), (0, 1), (1, 0),
               (-1, -1), (-1, 1), (1, -1), (1, 1))


class _BudgetExceeded(Exception):
    pass


class _TopologyUnavailable(Exception):
    pass


def _check(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise _BudgetExceeded


def _shortcut(path, labels, deadline):
    """Replace bends with validated chords using a bounded number of probes."""
    if len(path) < 3:
        return tuple(path)
    result = [path[0]]
    current = 0
    while current < len(path) - 1:
        _check(deadline)
        high = len(path) - 1
        low = current + 1
        chord = _direct_labeled_path(
            labels, path[current], path[high], area_id=None, deadline=deadline,
        )
        if chord:
            _extend_path(result, chord)
            break
        # Visibility need not be monotonic. Every accepted chord is validated;
        # binary probing bounds work but does not claim shortest-path optimality.
        best = (path[current], path[low])
        while low + 1 < high:
            _check(deadline)
            middle = (low + high) // 2
            chord = _direct_labeled_path(
                labels, path[current], path[middle], area_id=None,
                deadline=deadline,
            )
            if chord:
                low, best = middle, chord
            else:
                high = middle
        _extend_path(result, best)
        current = low
    _check(deadline)
    return tuple(result)


def _pixel_graph(skeleton, free, deadline):
    """Repair one-cell elbows and reject diagonal-only wall crossings."""
    height, width = free.shape
    for y, x in np.argwhere(skeleton):
        _check(deadline)
        for dx, dy in ((1, 1), (1, -1)):
            nx, ny = int(x + dx), int(y + dy)
            if not (0 <= nx < width and 0 <= ny < height and skeleton[ny, nx]):
                continue
            horizontal, vertical = bool(free[y, nx]), bool(free[ny, x])
            if horizontal != vertical:
                if horizontal:
                    skeleton[y, nx] = True
                else:
                    skeleton[ny, x] = True
    graph = {}
    for y, x in np.argwhere(skeleton):
        _check(deadline)
        x, y = int(x), int(y)
        neighbors = []
        for dx, dy in _DIRECTIONS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height and skeleton[ny, nx]):
                continue
            if dx and dy:
                if not (free[y, nx] and free[ny, x]):
                    continue
                # An existing cardinal elbow supersedes its diagonal shortcut.
                if skeleton[y, nx] or skeleton[ny, x]:
                    continue
            neighbors.append((nx, ny))
        graph[(x, y)] = tuple(neighbors)
    return graph


def _compress(pixels, deadline):
    """Contract chains, retaining junctions, loops, and bounded edge lengths."""
    nodes = {point for point, neighbors in pixels.items() if len(neighbors) != 2}
    unseen = set(pixels)
    components = 0
    while unseen:
        _check(deadline)
        seed = min(unseen)
        pending, component = [seed], set()
        while pending:
            point = pending.pop()
            if point in component:
                continue
            component.add(point)
            pending.extend(pixels[point])
        unseen.difference_update(component)
        components += 1
        if not nodes.intersection(component):
            nodes.add(seed)  # A closed loop has no natural junction.
    anchors = [(0, 0), *sorted(nodes)]
    node_ids = {point: index for index, point in enumerate(anchors) if index}
    visited = set()
    segments = []
    for start in sorted(nodes):
        for neighbor in pixels[start]:
            _check(deadline)
            if (start, neighbor) in visited:
                continue
            path = [start, neighbor]
            visited.update(((start, neighbor), (neighbor, start)))
            previous, current = start, neighbor
            while current not in nodes:
                following = next(point for point in pixels[current] if point != previous)
                visited.update(((current, following), (following, current)))
                path.append(following)
                previous, current = current, following
            # Sampling along a chain changes neither its geometry nor topology.
            # It permits local entry/exit without routing via distant junctions.
            cuts = list(range(0, len(path) - 1, 64)) + [len(path) - 1]
            if path[0] == path[-1] and len(cuts) == 2:
                cuts.insert(1, len(path) // 2)
            for first, last in zip(cuts, cuts[1:]):
                for point in (path[first], path[last]):
                    if point not in node_ids:
                        node_ids[point] = len(anchors)
                        anchors.append(point)
                segments.append(HighwaySegment(
                    node_ids[path[first]], node_ids[path[last]],
                    tuple(path[first:last + 1]),
                ))
    return anchors, segments, components


def _trim_spurs(skeleton, pixels, reach, deadline):
    """Trim short terminal spurs while keeping all links between junctions."""
    removed = 0
    for tip in sorted(point for point, edges in pixels.items() if len(edges) == 1):
        _check(deadline)
        path = [tip]
        previous, current = tip, pixels[tip][0]
        while len(pixels[current]) == 2:
            path.append(current)
            following = next(point for point in pixels[current] if point != previous)
            previous, current = current, following
        if len(pixels[current]) == 1:
            continue  # Keep a standalone corridor, including its endpoints.
        walked = 0.0
        for index, point in enumerate(path):
            if index:
                walked += math.dist(path[index - 1], point)
            if walked >= reach:
                break
            skeleton[point[1], point[0]] = False
        removed += 1
    return removed


def _access_field(free, skeleton, deadline):
    _check(deadline)
    starts = [tuple(map(int, point)) for point in np.argwhere(skeleton)]
    if not starts:
        return np.full(free.shape, np.inf), np.full(free.shape, -2, np.int8)
    solver = MCP(np.where(free, 1.0, np.inf), offsets=_OFFSETS)
    distances, predecessors = solver.find_costs(starts)
    _check(deadline)
    return distances - 1.0, predecessors.astype(np.int8)


def _access_path(point, predecessors, deadline, maximum_steps):
    path = [point]
    x, y = point
    while True:
        _check(deadline)
        direction = int(predecessors[y, x])
        if direction == -1:
            return tuple(path)
        if direction < 0:
            return ()
        if len(path) >= maximum_steps:
            raise _BudgetExceeded
        dy, dx = _OFFSETS[direction]
        x, y = x - dx, y - dy
        path.append((x, y))


def _cover(free, skeleton, reach, deadline, labels):
    """Add capillaries until every free cell has bounded geodesic access."""
    added = 0
    while True:
        distances, predecessors = _access_field(free, skeleton, deadline)
        far = free & (distances > reach + 1e-6)
        if not np.any(far):
            return predecessors, float(np.max(distances[free], initial=0.0)), added
        if not np.all(np.isfinite(distances[free])):
            raise _TopologyUnavailable
        # Choose separated peaks in one pass rather than flooding once per tip.
        values = np.where(far, distances, 0.0).astype(np.float32)
        size = max(3, min(129, int(reach) * 2 + 1))
        peaks = far & (values == cv2.dilate(values, np.ones((size, size), np.uint8)))
        _, regions = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=4)
        flat = np.flatnonzero(peaks)
        if not len(flat):
            flat = np.array([int(np.argmax(values))])
        _, positions = np.unique(regions.flat[flat], return_index=True)
        for position in flat[positions]:
            _check(deadline)
            y, x = divmod(int(position), free.shape[1])
            path = _access_path((x, y), predecessors, deadline, free.size + 1)
            for px, py in _shortcut(path, labels, deadline):
                skeleton[py, px] = True
            added += 1


def _skeleton_grid(free, labels, component_count, reach, deadline):
    """Reduce skeleton work only when free components and holes survive.

    A coarse cell is usable only when its entire original block is free.
    Routing, chord checks, and the access field still use original pixels.
    Narrow bridges or loops that cannot survive reduction use full resolution.
    """
    maximum_scale = min(4, int(reach // 4), math.ceil(math.sqrt(free.size / 100_000)))
    if maximum_scale > 1:
        background_count = cv2.connectedComponents(
            (~np.pad(free, 1)).astype(np.uint8), connectivity=8,
        )[0]
        height, width = free.shape
        for scale in range(maximum_scale, 1, -1):
            _check(deadline)
            padded = np.pad(free, ((0, (-height) % scale), (0, (-width) % scale)))
            coarse = padded.reshape(
                padded.shape[0] // scale, scale, padded.shape[1] // scale, scale,
            ).all(axis=(1, 3))
            coarse_count = cv2.connectedComponents(
                coarse.astype(np.uint8), connectivity=4,
            )[0] - 1
            if coarse_count != component_count:
                continue
            holes = cv2.connectedComponents(
                (~np.pad(coarse, 1)).astype(np.uint8), connectivity=8,
            )[0]
            if holes != background_count:
                continue
            # Equal counts alone could conceal a lost region and a split.
            cy, cx = np.nonzero(coarse)
            represented = np.unique(labels[cy * scale + scale // 2,
                                           cx * scale + scale // 2])
            if len(represented) != component_count or np.any(represented == 0):
                continue
            return coarse, scale
    return free, 1


def build_backbone(
    slam: SlamSnapshot, *, confidence_threshold: float, macro_cell_size: int,
    maximum_build_ms: float, maximum_access_distance: float, preparation_cache=None,
) -> HighwayBuildResult:
    """Publish a complete immutable corridor graph or retain the old snapshot."""
    started = time.perf_counter()
    deadline = started + max(0.0, maximum_build_ms) / 1000.0
    free = ((slam.occupancy == FREE) & (slam.confidence >= confidence_threshold))
    known = int(np.count_nonzero(free))
    rows, columns = np.flatnonzero(np.any(free, axis=1)), np.flatnonzero(np.any(free, axis=0))
    top, left = (int(rows[0]), int(columns[0])) if known else (0, 0)
    bottom, right = (int(rows[-1]) + 1, int(columns[-1]) + 1) if known else (1, 1)
    free = free[top:bottom, left:right]
    try:
        _check(deadline)
        # Bound uninterruptible native work. The normal 1615x1010 cave now
        # fits; substantially larger inputs still retain the previous graph.
        if known > 750_000 or free.size > 2_000_000:
            raise _BudgetExceeded
        if not math.isfinite(maximum_access_distance) or maximum_access_distance <= 0:
            raise ValueError("maximum highway access distance must be positive and finite")
        count, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
        _check(deadline)
        identity = (slam.origin, left, top, confidence_threshold, maximum_access_distance)
        cached = preparation_cache or {}
        if cached.get("identity") == identity and np.array_equal(cached.get("free"), free):
            smooth = cached["smooth"].copy()
            scale, pruned, method = cached["scale"], cached["pruned"], cached["method"]
        else:
            if preparation_cache is not None:
                preparation_cache.clear()
            skeleton_free, scale = _skeleton_grid(
                free, labels, count - 1, maximum_access_distance, deadline,
            )
            _check(deadline)
            # A narrow bridge or small hole can require original resolution.
            # Native thinning avoids the medial-axis size guard while keeping
            # the original topology and all subsequent pixel safety checks.
            method = "thinning" if skeleton_free.size > 500_000 else "medial_axis"
            padded = np.pad(skeleton_free, 1)
            skeleton = (skeletonize(padded) if method == "thinning" else
                        medial_axis(padded, rng=0))[1:-1, 1:-1].copy()
            _check(deadline)
            pixels = _pixel_graph(skeleton, skeleton_free, deadline)
            pruned = _trim_spurs(skeleton, pixels, maximum_access_distance / scale, deadline)
            pixels = _pixel_graph(skeleton, skeleton_free, deadline)
            _, segments, components = _compress(pixels, deadline)
            if components != count - 1:
                raise _TopologyUnavailable
            smooth = np.zeros_like(free)
            lift = lambda point: (point[0] * scale + scale // 2,
                                  point[1] * scale + scale // 2)
            for point in pixels:
                if not pixels[point]:
                    x, y = lift(point)
                    smooth[y, x] = True
            for segment in segments:
                path = segment.path
                if scale > 1:
                    path = [lift(segment.path[0])]
                    for first, last in zip(segment.path, segment.path[1:]):
                        chord = _direct_labeled_path(
                            labels, lift(first), lift(last), area_id=None, deadline=deadline,
                        )
                        if not chord:
                            raise _TopologyUnavailable
                        _extend_path(path, chord)
                for x, y in _shortcut(path, labels, deadline):
                    smooth[y, x] = True
            if preparation_cache is not None:
                preparation_cache.update(identity=identity, free=free.copy(),
                                         smooth=smooth.copy(), scale=scale, pruned=pruned, method=method)
        predecessors, measured, capillaries = _cover(
            free, smooth, maximum_access_distance, deadline, labels,
        )
        pixels = _pixel_graph(smooth, free, deadline)
        anchors, segments, components = _compress(pixels, deadline)
        if components != count - 1:
            raise _TopologyUnavailable
        # Repairing elbows can add access sources. Reusing the earlier field
        # is safe: all its original sources are still network pixels.
        ox, oy = slam.origin[0] + left, slam.origin[1] + top
        world = lambda point: (int(point[0] + ox), int(point[1] + oy))
        anchors = tuple(world(point) for point in anchors)
        segments = tuple(HighwaySegment(
            segment.source, segment.target, tuple(world(point) for point in segment.path),
        ) for segment in segments)
        adjacency = [[] for _ in anchors]
        locations = {}
        for segment_id, segment in enumerate(segments):
            cost = _path_cost(segment.path)
            adjacency[segment.source].append(HighwayEdge(segment.target, cost, segment.path))
            adjacency[segment.target].append(HighwayEdge(segment.source, cost, tuple(reversed(segment.path))))
            for index, point in enumerate(segment.path):
                locations[point] = (segment_id, index)
        nodes = {point: index for index, point in enumerate(anchors) if index}
        labels.setflags(write=False)
        predecessors.setflags(write=False)
        _check(deadline)
        elapsed = (time.perf_counter() - started) * 1000.0
        graph = HighwayGraphSnapshot(
            version=int(slam.version), origin=(ox, oy),
            full_shape=slam.full_shape or slam.occupancy.shape,
            macro_cell_size=int(macro_cell_size), confidence_threshold=confidence_threshold,
            area_labels=labels, anchors=anchors,
            adjacency=tuple(tuple(edges) for edges in adjacency),
            known_free_cells=known, build_elapsed_ms=elapsed,
            access_predecessors=predecessors, segments=segments,
            network_locations=locations, network_nodes=nodes,
            maximum_access_distance=maximum_access_distance,
            measured_access_distance=measured, component_count=count - 1,
            pruned_branches=pruned, capillary_branches=capillaries,
            skeleton_scale=scale,
            skeleton_method=method,
        )
        _check(deadline)
        return HighwayBuildResult(
            HIGHWAY_COMPLETE, graph, (time.perf_counter() - started) * 1000.0,
            graph.area_count, graph.edge_count, known,
        )
    except _BudgetExceeded:
        return HighwayBuildResult(
            HIGHWAY_BUDGET_EXHAUSTED, None,
            (time.perf_counter() - started) * 1000.0, known_free_cells=known,
        )
    except _TopologyUnavailable:
        return HighwayBuildResult(
            HIGHWAY_UNAVAILABLE, None,
            (time.perf_counter() - started) * 1000.0, known_free_cells=known,
        )


def route_backbone(
    graph, start, goal, *, maximum_query_ms, maximum_connector_expansions,
    required_version=None,
) -> HighwayRoute:
    """Attach to nearby branch interiors, route, and validate safe shortcuts."""
    started = time.perf_counter()
    deadline = started + max(0.0, maximum_query_ms) / 1000.0
    expansions = 0

    def result(status, path=(), edges=0):
        return HighwayRoute(
            status, path, _path_cost(path) if path else math.inf, graph.version,
            (time.perf_counter() - started) * 1000.0, expansions, edges,
        )

    if required_version is not None and graph.version != int(required_version):
        return result(HIGHWAY_STALE)
    local_start, local_goal = graph._local(start), graph._local(goal)
    start_area, goal_area = graph._area_at(local_start), graph._area_at(local_goal)
    if not start_area or not goal_area:
        return result(HIGHWAY_UNAVAILABLE)
    if start_area != goal_area:
        return result(HIGHWAY_UNREACHABLE)
    try:
        _check(deadline)
        direct = _direct_labeled_path(
            graph.area_labels, local_start, local_goal,
            area_id=None, deadline=deadline,
        )
        if direct:
            _check(deadline)
            return result(HIGHWAY_COMPLETE, graph._world_path(direct))
        maximum = max(1, int(maximum_connector_expansions))
        first = _access_path(local_start, graph.access_predecessors, deadline, maximum + 1)
        expansions = max(0, len(first) - 1)
        last = _access_path(
            local_goal, graph.access_predecessors, deadline,
            max(1, maximum - expansions) + 1,
        )
        expansions += max(0, len(last) - 1)
        if not first or not last:
            return result(HIGHWAY_UNAVAILABLE)

        def attachments(local_connector):
            connector = graph._world_path(local_connector)
            point = connector[-1]
            if point in graph.network_nodes:
                return {graph.network_nodes[point]: connector}
            segment_id, index = graph.network_locations[point]
            segment = graph.segments[segment_id]
            options = {}
            for node, partial in (
                (segment.source, tuple(reversed(segment.path[:index + 1]))),
                (segment.target, segment.path[index:]),
            ):
                combined = list(connector)
                _extend_path(combined, partial)
                options[node] = tuple(combined)
            return options

        starts, goals = attachments(first), attachments(last)

        def add_visible_entries(point, options):
            # Evaluate distinct nearby branches by whole-route cost. A single
            # nearest attachment can choose the wrong side of an obstacle.
            candidates = []
            for segment in graph.segments:
                _check(deadline)
                nearest = min(segment.path, key=lambda p: math.dist(point, p))
                candidates.append((math.dist(point, nearest), nearest))
            for _, candidate in heapq.nsmallest(8, candidates):
                local = _direct_labeled_path(
                    graph.area_labels, graph._local(point), graph._local(candidate),
                    area_id=None, deadline=deadline,
                )
                if not local:
                    continue
                for node, path in attachments(local).items():
                    if node not in options or _path_cost(path) < _path_cost(options[node]):
                        options[node] = path

        add_visible_entries(tuple(start), starts)
        add_visible_entries(tuple(goal), goals)
        best_path, best_cost, edge_count = (), math.inf, 0
        first_location = graph.network_locations.get(graph._world_path(first)[-1])
        last_location = graph.network_locations.get(graph._world_path(last)[-1])
        if first_location and last_location and first_location[0] == last_location[0]:
            segment = graph.segments[first_location[0]]
            a, b = first_location[1], last_location[1]
            middle = segment.path[a:b + 1] if a <= b else tuple(reversed(segment.path[b:a + 1]))
            candidate = list(graph._world_path(first))
            _extend_path(candidate, middle)
            _extend_path(candidate, reversed(graph._world_path(last)))
            best_path, best_cost, edge_count = tuple(candidate), _path_cost(candidate), 1
        costs = {node: _path_cost(path) for node, path in starts.items()}
        queue = [(cost, node) for node, cost in costs.items()]
        heapq.heapify(queue)
        parents = {}
        while queue:
            _check(deadline)
            cost, node = heapq.heappop(queue)
            if cost > costs.get(node, math.inf) + 1e-9:
                continue
            if cost >= best_cost:
                break
            expansions += 1
            if node in goals and cost + _path_cost(goals[node]) < best_cost:
                chain, current = [], node
                while current in parents:
                    previous, edge = parents[current]
                    chain.append(edge)
                    current = previous
                candidate = list(starts[current])
                for edge in reversed(chain):
                    _extend_path(candidate, edge.path)
                _extend_path(candidate, reversed(goals[node]))
                best_path, best_cost = tuple(candidate), _path_cost(candidate)
                edge_count = len(chain)
            for edge in graph.adjacency[node]:
                following = cost + edge.cost
                if following + 1e-9 < costs.get(edge.target_area, math.inf):
                    costs[edge.target_area] = following
                    parents[edge.target_area] = (node, edge)
                    heapq.heappush(queue, (following, edge.target_area))
        if not best_path:
            return result(HIGHWAY_UNREACHABLE)
        local = tuple(graph._local(point) for point in best_path)
        path = graph._world_path(_shortcut(local, graph.area_labels, deadline))
        _check(deadline)
        return result(HIGHWAY_COMPLETE, path, edge_count)
    except _BudgetExceeded:
        return result(HIGHWAY_BUDGET_EXHAUSTED)
