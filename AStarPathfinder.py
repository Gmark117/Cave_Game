"""A* pathfinder optimized for shared-memory numpy maps.

`compute_path` attaches to a SharedMemory numpy array containing a
binary map (1==wall, 0==free) and runs an efficient A* search using
flat-index arrays for performance. Returns a list of (x,y) tuples.
"""

import heapq
import math
from typing import List, Tuple

import numpy as np
from multiprocessing import shared_memory


def compute_path(shm_name: str, shape: tuple, start: tuple, goal: tuple, max_iters: int = 200000) -> List[Tuple[int, int]]:
    """Compute A* path on a shared-memory numpy map where 1==wall, 0==free.
    Returns a list of (x,y) tuples from start to goal, or empty list if no path.
    """
    # Attach to the existing shared-memory buffer created by MissionControl.
    # If the shared memory is not found, return an empty path immediately.
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return []

    # Create a NumPy view onto the shared buffer (row-major: arr[y, x]).
    arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)

    # Dimensions and helpers: convert 2D coordinates to flat indices for
    # compact arrays (fewer Python objects and faster membership checks).
    height, width = arr.shape
    size = width * height

    def inside(x, y):
        return 0 <= x < width and 0 <= y < height

    # 8-neighborhood offsets (dx,dy) including diagonals
    pos_mods = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    # Flat-index helper: convert (x,y) to single index for compact arrays
    def idx(x, y):
        return y * width + x

    start_x, start_y = int(start[0]), int(start[1])
    goal_x, goal_y = int(goal[0]), int(goal[1])

    # Quick bounds and obstacle checks for start/goal
    if not (inside(start_x, start_y) and inside(goal_x, goal_y)):
        return []
    if arr[start_y, start_x] != 0 or arr[goal_y, goal_x] != 0:
        return []

    start_i = idx(start_x, start_y)
    goal_i = idx(goal_x, goal_y)

    # Compact arrays used by the search
    closed = np.zeros(size, dtype=bool)            # visited flag per cell
    parent = np.full(size, -1, dtype=np.int32)     # parent flat-index for path reconstruction
    g_score = np.full(size, np.inf, dtype=np.float32)  # best-known g-score per cell

    # Movement costs and heuristic (octile) for 8-neighborhood
    ORTH_COST = 1.0
    DIAG_COST = math.sqrt(2)

    def heuristic_flat(i):
        # Octile heuristic using integer coordinates
        x = i % width
        y = i // width
        dx = abs(x - goal_x)
        dy = abs(y - goal_y)
        D1 = ORTH_COST
        D2 = DIAG_COST
        return D1 * (dx + dy) + (D2 - 2 * D1) * min(dx, dy)

    # Initialize open heap with start
    g_score[start_i] = 0.0
    start_f = heuristic_flat(start_i)
    open_heap = [(start_f, start_i)]

    # Main search loop (A*). Pop the lowest-f node, expand neighbors.
    iters = 0
    while open_heap and iters < max_iters:
        iters += 1
        _, curr_i = heapq.heappop(open_heap)

        # Skip if we already processed this node with a better cost
        if closed[curr_i]:
            continue

        # Found goal: reconstruct path by following parents
        if curr_i == goal_i:
            path = []
            cur = curr_i
            while cur != -1:
                x = cur % width
                y = cur // width
                path.append((x, y))
                cur = int(parent[cur])
            return path[::-1]

        # Mark current node as closed/visited
        closed[curr_i] = True

        cx = curr_i % width
        cy = curr_i // width
        cg = float(g_score[curr_i])

        # Expand 8 neighbors
        for dx, dy in pos_mods:
            nx = cx + dx
            ny = cy + dy

            # Bounds check
            if not inside(nx, ny):
                continue

            ni = idx(nx, ny)

            # Skip already-closed or blocked cells
            if closed[ni] or arr[ny, nx] != 0:
                continue

            # Movement cost: diagonal vs orthogonal
            move_cost = DIAG_COST if (dx != 0 and dy != 0) else ORTH_COST

            # Corner-cutting rule: allow diagonal movement if at least one
            # of the adjacent orthogonal neighbors is free. This prevents
            # moving through tight diagonal corners where both adjacent
            # orthogonals are walls.
            if dx != 0 and dy != 0:
                adj1_x, adj1_y = cx + dx, cy
                adj2_x, adj2_y = cx, cy + dy
                ok1 = inside(adj1_x, adj1_y) and (arr[adj1_y, adj1_x] == 0)
                ok2 = inside(adj2_x, adj2_y) and (arr[adj2_y, adj2_x] == 0)
                if not (ok1 or ok2):
                    continue

            tentative_g = cg + move_cost

            # If this path to neighbor is not better, skip
            if tentative_g >= float(g_score[ni]):
                continue

            # Record best parent and scores, push to open heap
            parent[ni] = curr_i
            g_score[ni] = tentative_g
            f = tentative_g + heuristic_flat(ni)
            heapq.heappush(open_heap, (f, ni))

    # No path found within iteration limit
    return []
