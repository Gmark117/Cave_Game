"""Bounded, versioned route work for drone-local batch decisions."""

from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
import math
import time

from navigation.highway import HighwayRoute


def bounded_local_planning(source):
    """Share one planning allowance across a batch decision's route queries."""
    def decorate(function):
        @wraps(function)
        def call(controller, execution, *args, **kwargs):
            if execution.directive.kind.value != "component_batch":
                return function(controller, execution, *args, **kwargs)
            with controller._local_slam_planner.planning(source):
                return function(controller, execution, *args, **kwargs)
        return call
    return decorate


class LocalSlamRoutePlanner:
    MAXIMUM_PLANNING_MS = 250.0
    MAXIMUM_QUERIES = 128
    MAXIMUM_QUERY_MS = 50.0
    MAXIMUM_EXPANSIONS = 4096
    MAXIMUM_CACHE_ENTRIES = 256

    def __init__(self, slam_map, threshold, *, trace=None, now=time.perf_counter):
        self.slam_map = slam_map
        self.threshold = threshold
        self.trace = trace
        self.now = now
        self.snapshot = None
        self.cache = {}
        self.active = None

    @contextmanager
    def planning(self, source):
        if self.active is not None:
            yield
            return
        started = self.now()
        state = dict(deadline=started + self.MAXIMUM_PLANNING_MS / 1000,
                     queries=0, cache_hits=0, expanded_nodes=0, budget_exhausted=False)
        self.active = state
        try:
            yield
        finally:
            self.active = None
            if self.trace is not None:
                self.trace("drone_local_route_planning_completed", source=source,
                           elapsed_ms=(self.now() - started) * 1000,
                           route_queries=state["queries"], route_cache_hits=state["cache_hits"],
                           expanded_nodes=state["expanded_nodes"],
                           status="budget_exhausted" if state["budget_exhausted"] else "complete")

    def route(self, start, goal):
        from navigation.return_route import local_return_alternative

        started = self.now()
        state = self.active
        if state is not None and started >= state["deadline"]:
            state["budget_exhausted"] = True
            return HighwayRoute("budget_exhausted")
        if self.snapshot is None or self.snapshot.version != self.slam_map.version:
            self.snapshot = self.slam_map.snapshot(point_limit=0)
            self.cache.clear()
        key = (tuple(start), tuple(goal))
        if key in self.cache:
            if state is not None:
                state["cache_hits"] += 1
                state["budget_exhausted"] |= self.cache[key].status == "partial_limit"
            return self.cache[key]
        if state is not None:
            if state["queries"] >= self.MAXIMUM_QUERIES:
                state["budget_exhausted"] = True
                return HighwayRoute("budget_exhausted")
            state["queries"] += 1
        deadline = started + self.MAXIMUM_QUERY_MS / 1000
        if state is not None:
            deadline = min(deadline, state["deadline"])
        route = local_return_alternative(
            self.snapshot, key[0], key[1], confidence_threshold=self.threshold,
            incumbent_cost=math.inf, maximum_ms=max(0, (deadline - self.now()) * 1000),
            maximum_expansions=self.MAXIMUM_EXPANSIONS, allow_partial=True,
        )
        if state is not None:
            state["expanded_nodes"] += route.expanded_nodes
            state["budget_exhausted"] |= route.status in {"budget_exhausted", "partial_limit"}
        if route.status != "budget_exhausted":
            if len(self.cache) >= self.MAXIMUM_CACHE_ENTRIES:
                self.cache.clear()
            self.cache[key] = route
            if route.complete or route.status in {"unavailable", "no_shorter_route"}:
                self.cache[(key[1], key[0])] = replace(route, path=tuple(reversed(route.path)))
        return route
