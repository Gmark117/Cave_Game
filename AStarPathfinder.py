import heapq
import math
import numpy as np
from multiprocessing import shared_memory


def compute_path(shm_name: str, shape: tuple, start: tuple, goal: tuple, max_iters: int = 200000):
    """Compute A* path on a shared-memory numpy map where 1==wall, 0==free.
    Returns a list of (x,y) tuples from start to goal, or empty list if no path.
    """
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return []

    arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)

    height, width = arr.shape
    size = width * height

    def inside(x, y):
        return 0 <= x < width and 0 <= y < height

    # precompute neighbor offsets (8-neighborhood)
    pos_mods = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    # Flat-index helpers
    def idx(x, y):
        return y * width + x

    start_x, start_y = int(start[0]), int(start[1])
    goal_x, goal_y = int(goal[0]), int(goal[1])

    if not (inside(start_x, start_y) and inside(goal_x, goal_y)):
        return []
    if arr[start_y, start_x] != 0 or arr[goal_y, goal_x] != 0:
        return []

    start_i = idx(start_x, start_y)
    goal_i = idx(goal_x, goal_y)

    closed = np.zeros(size, dtype=bool)       # bitset membership
    parent = np.full(size, -1, dtype=np.int32)  # parent flat-index
    g_score = np.full(size, np.inf, dtype=np.float32)

    # costs
    ORTH_COST = 1.0
    DIAG_COST = math.sqrt(2)

    def heuristic_flat(i):
        x = i % width
        y = i // width
        dx = abs(x - goal_x)
        dy = abs(y - goal_y)
        D1 = ORTH_COST
        D2 = DIAG_COST
        return D1 * (dx + dy) + (D2 - 2 * D1) * min(dx, dy)

    g_score[start_i] = 0.0
    start_f = heuristic_flat(start_i)

    open_heap = [(start_f, start_i)]

    iters = 0
    while open_heap and iters < max_iters:
        iters += 1
        _, curr_i = heapq.heappop(open_heap)
        if closed[curr_i]:
            continue
        if curr_i == goal_i:
            # reconstruct path
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

        for dx, dy in pos_mods:
            nx = cx + dx
            ny = cy + dy
            if not inside(nx, ny):
                continue
            ni = idx(nx, ny)
            if closed[ni]:
                continue
            if arr[ny, nx] != 0:
                continue

            # determine move cost (diag vs orthogonal)
            move_cost = DIAG_COST if (dx != 0 and dy != 0) else ORTH_COST

            # corner-cutting rule: allow diagonal if at least one adjacent orthogonal cell is free
            if dx != 0 and dy != 0:
                adj1_x, adj1_y = cx + dx, cy
                adj2_x, adj2_y = cx, cy + dy
                ok1 = inside(adj1_x, adj1_y) and (arr[adj1_y, adj1_x] == 0)
                ok2 = inside(adj2_x, adj2_y) and (arr[adj2_y, adj2_x] == 0)
                if not (ok1 or ok2):
                    continue

            tentative_g = cg + move_cost
            if tentative_g >= float(g_score[ni]):
                continue

            parent[ni] = curr_i
            g_score[ni] = tentative_g
            f = tentative_g + heuristic_flat(ni)
            heapq.heappush(open_heap, (f, ni))

    # no path found or exceeded iters
    return []
