"""A* pathfinder optimized for numpy-backed cave maps."""

import heapq
import math
from multiprocessing import shared_memory
from typing import Callable, List, Tuple

import numpy as np


Position = Tuple[int, int]
CellCostPolicy = Callable[[int, int], float]

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
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return []

    arr = None
    try:
        arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
        return _compute_astar_path(
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
    return 1.0


def _compute_astar_path(
    arr: np.ndarray,
    start: tuple,
    goal: tuple,
    max_iters: int,
    cell_cost: CellCostPolicy,
) -> List[Position]:
    height, width = arr.shape
    size = width * height

    def inside(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    def idx(x: int, y: int) -> int:
        return y * width + x

    start_x, start_y = int(start[0]), int(start[1])
    goal_x, goal_y = int(goal[0]), int(goal[1])

    if not (inside(start_x, start_y) and inside(goal_x, goal_y)):
        return []
    if arr[start_y, start_x] != 0 or arr[goal_y, goal_x] != 0:
        return []

    start_i = idx(start_x, start_y)
    goal_i = idx(goal_x, goal_y)

    closed = np.zeros(size, dtype=bool)
    parent = np.full(size, -1, dtype=np.int32)
    g_score = np.full(size, np.inf, dtype=np.float32)

    def heuristic_flat(i: int) -> float:
        x = i % width
        y = i // width
        dx = abs(x - goal_x)
        dy = abs(y - goal_y)
        return ORTH_COST * (dx + dy) + (DIAG_COST - 2 * ORTH_COST) * min(dx, dy)

    g_score[start_i] = 0.0
    open_heap = [(heuristic_flat(start_i), start_i)]

    iters = 0
    while open_heap and iters < max_iters:
        iters += 1
        _, curr_i = heapq.heappop(open_heap)

        if closed[curr_i]:
            continue

        if curr_i == goal_i:
            path = []
            cur = curr_i
            while cur != -1:
                x = cur % width
                y = cur // width
                path.append((x, y))
                cur = int(parent[cur])
            return path[::-1]

        closed[curr_i] = True
        cx = curr_i % width
        cy = curr_i // width
        cg = float(g_score[curr_i])

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

    return []
