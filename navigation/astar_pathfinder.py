"""A* pathfinder optimized for numpy-backed cave maps."""

import heapq
import math
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Callable, List, Tuple

import numpy as np


Position = Tuple[int, int]
CellCostPolicy = Callable[[int, int], float]

PATH_COMPLETE = "complete"
PATH_PARTIAL_LIMIT = "partial_limit"
PATH_UNREACHABLE = "unreachable"
PATH_INVALID_ENDPOINT = "invalid_endpoint"
PATH_RESOURCE_UNAVAILABLE = "resource_unavailable"


@dataclass(frozen=True)
class PathResult:
    """One complete route, useful capped segment, or explicit failure."""

    path: tuple[Position, ...]
    status: str
    iterations: int
    remaining_distance: float

    @property
    def reached_goal(self) -> bool:
        """Return whether this result contains a complete route."""
        return self.status == PATH_COMPLETE

ORTH_COST = 1.0
DIAG_COST = math.sqrt(2)
NEIGHBOR_OFFSETS = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def compute_path(
    shm_name: str,
    shape: tuple,
    start: tuple,
    goal: tuple,
    max_iters: int = 200000,
) -> List[Position]:
    """Compute A* path on a shared-memory numpy map where 1==wall, 0==free.
    Returns a list of (x,y) tuples from start to goal, or empty list if no path.
    """
    result = compute_path_segment(
        shm_name,
        shape,
        start,
        goal,
        max_iters,
    )
    if not result.reached_goal:
        return []
    return list(result.path)


def compute_path_segment(
    shm_name: str,
    shape: tuple,
    start: tuple,
    goal: tuple,
    max_iters: int = 200000,
) -> PathResult:
    """Compute a complete path or the best progress segment at the cap."""
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return PathResult((), PATH_RESOURCE_UNAVAILABLE, 0, math.inf)

    arr = None
    try:
        arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
        return _compute_astar_result(
            arr,
            start,
            goal,
            max_iters,
            _unit_cell_cost,
        )
    finally:
        if arr is not None:
            del arr
        try:
            shm.close()
        except (BufferError, OSError):
            pass


def compute_weighted_path(
    cave_map: np.ndarray,
    roughness_map: np.ndarray,
    confidence_map: np.ndarray,
    start: tuple,
    goal: tuple,
    max_iters: int = 200000,
    roughness_weight: float = 4.0,
    unknown_penalty: float = 2.5,
    low_confidence_penalty: float = 1.5,
) -> List[Position]:
    """Compute a weighted A* path for rovers using known terrain roughness.

    Drones continue to use the shared-memory adapter, while rovers use
    terrain-aware costs derived from known roughness and confidence maps.
    """
    arr = np.asarray(cave_map, dtype=np.uint8)
    roughness = np.asarray(roughness_map, dtype=np.float32)
    confidence = np.asarray(confidence_map, dtype=np.float32)

    def terrain_cost(x: int, y: int) -> float:
        terrain_penalty = 1.0
        cell_confidence = float(confidence[y, x])
        if cell_confidence <= 0.0:
            terrain_penalty += unknown_penalty
        else:
            terrain_penalty += roughness_weight * float(
                max(0.0, roughness[y, x])
            )
            terrain_penalty += low_confidence_penalty * max(
                0.0,
                1.0 - cell_confidence,
            )
        return terrain_penalty

    return _compute_astar_path(
        arr,
        start,
        goal,
        max_iters,
        terrain_cost,
    )


def _unit_cell_cost(_x: int, _y: int) -> float:
    """Return the normal cost for unweighted drone movement."""
    return 1.0


def _compute_astar_path(
    arr: np.ndarray,
    start: tuple,
    goal: tuple,
    max_iters: int,
    cell_cost: CellCostPolicy,
) -> List[Position]:
    """Run grid A* with 8-way movement and optional per-cell cost."""
    result = _compute_astar_result(
        arr,
        start,
        goal,
        max_iters,
        cell_cost,
    )
    if not result.reached_goal:
        return []
    return list(result.path)


def _compute_astar_result(
    arr: np.ndarray,
    start: tuple,
    goal: tuple,
    max_iters: int,
    cell_cost: CellCostPolicy,
) -> PathResult:
    """Run A* and retain the best goal-directed path when capped."""
    height, width = arr.shape
    size = width * height

    def inside(x: int, y: int) -> bool:
        """Return whether a coordinate is inside the map."""
        return 0 <= x < width and 0 <= y < height

    def idx(x: int, y: int) -> int:
        """Flatten a 2D map coordinate into a 1D array index."""
        return y * width + x

    start_x, start_y = int(start[0]), int(start[1])
    goal_x, goal_y = int(goal[0]), int(goal[1])

    initial_remaining = math.hypot(goal_x - start_x, goal_y - start_y)
    if not (inside(start_x, start_y) and inside(goal_x, goal_y)):
        return PathResult(
            (), PATH_INVALID_ENDPOINT, 0, initial_remaining,
        )
    if arr[start_y, start_x] != 0 or arr[goal_y, goal_x] != 0:
        return PathResult(
            (), PATH_INVALID_ENDPOINT, 0, initial_remaining,
        )

    start_i = idx(start_x, start_y)
    goal_i = idx(goal_x, goal_y)

    closed = np.zeros(size, dtype=bool)
    parent = np.full(size, -1, dtype=np.int32)
    g_score = np.full(size, np.inf, dtype=np.float32)

    def heuristic_flat(i: int) -> float:
        """Octile-distance heuristic for 8-way grid movement."""
        x = i % width
        y = i // width
        dx = abs(x - goal_x)
        dy = abs(y - goal_y)
        return ORTH_COST * (dx + dy) + (DIAG_COST - 2 * ORTH_COST) * min(dx, dy)

    def reconstruct_path(end_i: int) -> tuple[Position, ...]:
        """Reconstruct the discovered parent chain ending at ``end_i``."""
        path = []
        cur = end_i
        while cur != -1:
            path.append((cur % width, cur // width))
            cur = int(parent[cur])
        return tuple(reversed(path))

    initial_h = heuristic_flat(start_i)
    g_score[start_i] = 0.0
    open_heap = [(initial_h, start_i)]
    best_i = start_i
    best_h = initial_h
    best_g = 0.0

    iters = 0
    while open_heap and iters < max_iters:
        iters += 1
        _, curr_i = heapq.heappop(open_heap)

        if closed[curr_i]:
            continue

        if curr_i == goal_i:
            return PathResult(
                reconstruct_path(curr_i),
                PATH_COMPLETE,
                iters,
                0.0,
            )

        closed[curr_i] = True
        cx = curr_i % width
        cy = curr_i // width
        cg = float(g_score[curr_i])
        current_h = heuristic_flat(curr_i)
        if (
            current_h < best_h - 1e-9
            or (
                abs(current_h - best_h) <= 1e-9
                and cg < best_g
            )
        ):
            best_i = curr_i
            best_h = current_h
            best_g = cg

        for dx, dy in NEIGHBOR_OFFSETS:
            nx = cx + dx
            ny = cy + dy
            if not inside(nx, ny):
                continue

            ni = idx(nx, ny)
            if closed[ni] or arr[ny, nx] != 0:
                continue

            move_cost = DIAG_COST if (dx != 0 and dy != 0) else ORTH_COST
            if dx != 0 and dy != 0:
                # Prevent diagonal moves that squeeze through two touching
                # wall corners.
                adj1_x, adj1_y = cx + dx, cy
                adj2_x, adj2_y = cx, cy + dy
                ok1 = inside(adj1_x, adj1_y) and (arr[adj1_y, adj1_x] == 0)
                ok2 = inside(adj2_x, adj2_y) and (arr[adj2_y, adj2_x] == 0)
                if not (ok1 or ok2):
                    continue

            tentative_g = cg + (move_cost * cell_cost(nx, ny))
            if tentative_g >= float(g_score[ni]):
                continue

            parent[ni] = curr_i
            g_score[ni] = tentative_g
            heapq.heappush(open_heap, (tentative_g + heuristic_flat(ni), ni))

    partial_i = None
    partial_h = math.inf
    partial_key = (math.inf, math.inf, size)
    for _queued_f, candidate_i in open_heap:
        if closed[candidate_i]:
            continue
        candidate_h = heuristic_flat(candidate_i)
        if candidate_h >= initial_h - 1e-9:
            continue
        candidate_g = float(g_score[candidate_i])
        candidate_key = (
            candidate_g + candidate_h,
            candidate_h,
            candidate_i,
        )
        if candidate_key < partial_key:
            partial_i = candidate_i
            partial_h = candidate_h
            partial_key = candidate_key

    if partial_i is None and best_i != start_i and best_h < initial_h:
        partial_i = best_i
        partial_h = best_h
    if open_heap and partial_i is not None:
        if partial_i == goal_i:
            return PathResult(
                reconstruct_path(partial_i),
                PATH_COMPLETE,
                iters,
                0.0,
            )
        return PathResult(
            reconstruct_path(partial_i),
            PATH_PARTIAL_LIMIT,
            iters,
            partial_h,
        )
    return PathResult(
        (),
        PATH_UNREACHABLE if not open_heap else PATH_PARTIAL_LIMIT,
        iters,
        best_h,
    )
