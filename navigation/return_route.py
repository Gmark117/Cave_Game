"""Bounded return alternatives proven traversable by drone-local SLAM."""

import heapq
import math
import time

import numpy as np

from mapping.slam_map import FREE
from navigation.highway import HighwayRoute, _direct_labeled_path, _extend_path, _path_cost
from navigation.highway_backbone import _BudgetExceeded


def local_return_alternative(slam, start, goal, *, confidence_threshold,
                             incumbent_cost, maximum_ms, maximum_expansions,
                             cached_path=(), allow_partial=False):
    started = time.perf_counter()
    deadline = started + max(0.0, maximum_ms) / 1000
    expanded = 0

    def result(status, path=()):
        return HighwayRoute(status, path, _path_cost(path) if path else math.inf,
                            slam.version, (time.perf_counter() - started) * 1000,
                            expanded)

    if maximum_ms <= 0:
        return result("budget_exhausted")
    free = (slam.occupancy == FREE) & (slam.confidence >= confidence_threshold)
    ox, oy = slam.origin
    first, last = (start[0] - ox, start[1] - oy), (goal[0] - ox, goal[1] - oy)
    height, width = free.shape

    def valid(point):
        x, y = point
        return 0 <= x < width and 0 <= y < height and free[y, x]

    if not valid(first) or not valid(last):
        return result("unavailable")
    try:
        direct = _direct_labeled_path(free, first, last, area_id=None, deadline=deadline)
        if direct:
            path = tuple((x + ox, y + oy) for x, y in direct)
            return result("complete", path) if _path_cost(path) < incumbent_cost - 1e-6 else result("no_shorter_route")
        if cached_path and tuple(cached_path[0]) == tuple(start) and tuple(cached_path[-1]) == tuple(goal):
            candidate = []
            for a, b in zip(cached_path, cached_path[1:]):
                chord = _direct_labeled_path(free, (a[0] - ox, a[1] - oy),
                                            (b[0] - ox, b[1] - oy), area_id=None,
                                            deadline=deadline)
                if not chord:
                    candidate = []
                    break
                _extend_path(candidate, chord)
            if candidate:
                path = tuple((x + ox, y + oy) for x, y in candidate)
                if _path_cost(path) < incumbent_cost - 1e-6:
                    return result("complete", path)
        queue = [(math.dist(first, last), 0.0, first)]
        costs, parents = {first: 0.0}, {}
        closest = first

        def reconstructed(point):
            path = [point]
            while point in parents:
                point = parents[point]
                path.append(point)
            return tuple((x + ox, y + oy) for x, y in reversed(path))

        while queue:
            if time.perf_counter() >= deadline or expanded >= maximum_expansions:
                if allow_partial and closest != first and math.isinf(incumbent_cost):
                    return result("partial_limit", reconstructed(closest))
                return result("budget_exhausted")
            estimate, cost, point = heapq.heappop(queue)
            if cost != costs.get(point):
                continue
            if estimate >= incumbent_cost - 1e-6:
                break
            expanded += 1
            if math.dist(point, last) < math.dist(closest, last):
                closest = point
            if point == last:
                return result("complete", reconstructed(point))
            x, y = point
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                following = (x + dx, y + dy)
                if not valid(following):
                    continue
                if dx and dy and not (valid((x + dx, y)) and valid((x, y + dy))):
                    continue
                next_cost = cost + (math.sqrt(2) if dx and dy else 1.0)
                if next_cost < costs.get(following, math.inf):
                    costs[following], parents[following] = next_cost, (x, y)
                    heapq.heappush(queue, (next_cost + math.dist(following, last), next_cost, following))
        return result("no_shorter_route")
    except _BudgetExceeded:
        return result("budget_exhausted")
