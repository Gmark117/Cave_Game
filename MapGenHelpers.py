"""Helpers for map generation used by `MapGenerator`.

This module contains performance-focused helpers used by the generator:
- OpenCV-based brush application (`apply_cv_brush`)
- SharedMemory lifecycle helpers and a module-level `worm` for
    spawn-safe multiprocessing on Windows
- Post-processing helpers for cleaning and adding noise to the map
"""
import math
from contextlib import contextmanager
import time
from multiprocessing import Process
from multiprocessing import shared_memory as mp_shm
import numpy as np
import cv2
import pygame
import Assets
from Assets import next_cell_coords


# =============================================================================
# Brush application and map post-processing helpers
# =============================================================================

@contextmanager
def with_surfarrays(surface: 'pygame.Surface'):
    """Context manager yielding `(rgb_arr, alpha_arr)` surfarray views.

    Ensures the surfarray views are released (deleted) when exiting the
    context to avoid locking the `pygame.Surface`.
    """
    rgb_arr = pygame.surfarray.pixels3d(surface)
    alpha_arr = pygame.surfarray.pixels_alpha(surface)
    try:
        yield rgb_arr, alpha_arr
    finally:
        del rgb_arr
        del alpha_arr


def apply_cv_brush(sub: np.ndarray, cx: float, cy: float, mode_choice: int, stren: int, rng=None) -> None:
    """Apply a brush mask into `sub` using OpenCV primitives.

    `sub` is a view of the global binary map; pixels with value 1 are
    considered walls. This writes zeros into `sub` where the brush
    removes wall pixels. `mode_choice` selects the brush shape.
    """
    h, w = sub.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = max(1, int(0.5 * stren))
    icx = int(round(cx))
    icy = int(round(cy))
    try:
        if mode_choice == 0:
            cv2.circle(mask, (icx, icy), radius, 1, thickness=-1)
        elif mode_choice == 1:
            rx = max(1, radius)
            ry = max(1, int(max(1, radius / 2)))
            cv2.ellipse(mask, (icx, icy), (rx, ry), 0, 0, 360, 1, thickness=-1)
        elif mode_choice == 2:
            pts = np.array([[[icx + radius, icy], [icx, icy + radius], [icx - radius, icy], [icx, icy - radius]]], dtype=np.int32)
            cv2.fillPoly(mask, pts, 1)
        elif mode_choice == 3:
            r = radius
            pts = np.array([[[icx + int(0.6 * r), icy + r], [icx + r, icy + int(0.6 * r)], [icx + r, icy - int(0.6 * r)], [icx + int(0.6 * r), icy - r], [icx - int(0.6 * r), icy - r], [icx - r, icy - int(0.6 * r)], [icx - r, icy + int(0.6 * r)], [icx - int(0.6 * r), icy + r]]], dtype=np.int32)
            cv2.fillPoly(mask, pts, 1)
        else:
            local_rng = rng if rng is not None else np.random.default_rng()
            factor = float(local_rng.uniform(Assets.MapGen.CHAOTIC_FACTOR_LOW, Assets.MapGen.CHAOTIC_FACTOR_HIGH))
            rr = max(1, int(round(factor * stren)))
            x1 = max(0, icx - rr)
            x2 = min(w - 1, icx + rr)
            y1 = max(0, icy - rr)
            y2 = min(h - 1, icy + rr)
            xs = np.arange(x1, x2 + 1)
            ys = np.arange(y1, y2 + 1)
            XX, YY = np.meshgrid(xs, ys, indexing='xy')
            local_mask = (XX - icx) ** 2 + (YY - icy) ** 2 <= (rr ** 2)
            mask[y1:y2+1, x1:x2+1][local_mask] = 1
    except Exception:
        xs = np.arange(0, w)
        ys = np.arange(0, h)
        XX, YY = np.meshgrid(xs, ys, indexing='xy')
        if mode_choice == 4:
            local_rng = rng if rng is not None else np.random.default_rng()
            factor = float(local_rng.uniform(Assets.MapGen.CHAOTIC_FACTOR_LOW, Assets.MapGen.CHAOTIC_FACTOR_HIGH))
            mask = ((XX - cx) ** 2 + (YY - cy) ** 2) <= (factor * stren) ** 2
        else:
            mask = ((XX - cx) ** 2 + (YY - cy) ** 2) <= (0.5 * stren) ** 2
            mask = mask.astype(np.uint8)
    sub[(mask == 1) & (sub == 1)] = 0
        

def remove_hermit_caves(image: np.ndarray) -> np.ndarray:
    """
    Remove isolated cave regions (hermit caves) from a binary image, keeping only the largest connected component.
    
    This function inverts the input image, identifies all connected components in the inverted space,
    and removes all components except the largest one. This effectively eliminates small, isolated
    cave regions while preserving the main cave system.
    
    Args:
        image (np.ndarray): A binary image where 0 represents cave/empty space and 1 represents walls.
    
    Returns:
        np.ndarray: A cleaned binary image with isolated caves removed. Only the largest connected
                    component of caves is retained, with all other cave regions replaced by walls (1).
    
    Note:
        - Uses 8-connectivity for connected component analysis.
        - If the image contains no caves or only one connected component, the original image is returned.
    """
    inverted_image = np.where(image == 0, 1, 0).astype('uint8')
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_image, connectivity=8)
    if num_labels <= 1:
        return image
    biggest_blob_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    mask_to_keep = np.ones_like(labels, dtype=bool)
    mask_to_keep[labels != biggest_blob_index] = False
    cleaned_image = np.where(mask_to_keep, image, 1)
    return cleaned_image


def add_wall_transition_noise(image: np.ndarray, width: int, height: int, seed: int, worm_inputs: tuple) -> np.ndarray:
    """
    Add noise and random spikes to wall-floor transition regions in a cave map.

    This function creates a more natural-looking cave by adding noise near wall boundaries
    and generating random spike-like protrusions from existing walls.

    Args:
        image (np.ndarray): Input 2D array representing the cave map where 0 is floor and 1 is wall.
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.
        seed (int): Random seed for reproducible noise generation.
        worm_inputs (tuple): Tuple containing worm generation parameters, where the first element
                            affects the depth of the noise transition region.

    Returns:
        np.ndarray: Modified image array with added noise and spike features along wall transitions.

    Notes:
        - Uses distance transform to identify wall-floor boundaries
        - Applies probabilistic noise based on distance from walls
        - Generates random spike protrusions from existing walls
        - The noise depth is calculated as max(4, worm_inputs[0] // 2)
        - Spikes are drawn with random angles and lengths
        - Falls back to manual distance calculation if cv2.distanceTransform fails
    """
    rng = np.random.default_rng(seed=(seed or 0) + 2)
    img = image.copy()
    floor_mask = (img == 0).astype(np.uint8)
    try:
        dt = cv2.distanceTransform(floor_mask, cv2.DIST_L2, 5)
    except Exception:
        dt = np.full_like(img, 999.0, dtype=float)
        ys, xs = np.nonzero(floor_mask)
        for y, x in zip(ys, xs):
            dists = np.abs(np.arange(width) - x)[None, :] + np.abs(np.arange(height) - y)[:, None]
            dt[y, x] = dists.min()
    noise_depth = max(4, int(max(4, (worm_inputs[0] // 2))))
    base_prob = 0.6
    region = (dt > 0) & (dt <= float(noise_depth))
    if region.any():
        regional_dt = dt[region]
        probs = base_prob * (1.0 - (regional_dt / float(noise_depth)))
        rand = rng.random(probs.shape)
        sel = rand < probs
        coords = np.where(region)
        if sel.any():
            img[coords[0][sel], coords[1][sel]] = 1
    wall_coords = np.transpose(np.nonzero(img == 1))
    if wall_coords.shape[0] > 0:
        N = max(24, (width + height) // 25)
        N = min(N, wall_coords.shape[0])
        picks = rng.choice(wall_coords.shape[0], size=N, replace=False)
        spike_mask = np.zeros_like(img, dtype=np.uint8)
        for p in picks:
            y0, x0 = int(wall_coords[p, 0]), int(wall_coords[p, 1])
            length = int(rng.integers(2, noise_depth + 6))
            angle = float(rng.uniform(0, 2 * math.pi))
            dx = int(round(math.cos(angle) * length))
            dy = int(round(math.sin(angle) * length))
            x1 = max(0, min(width - 1, x0 + dx))
            y1 = max(0, min(height - 1, y0 + dy))
            try:
                cv2.line(spike_mask, (x0, y0), (x1, y1), 1, thickness=2)
            except Exception:
                continue
        try:
            kernel = np.ones((3, 3), dtype=np.uint8)
            spike_mask = cv2.dilate(spike_mask, kernel, iterations=1)
        except Exception:
            pass
        img[spike_mask == 1] = 1
    return img


# =============================================================================
# SharedMemory and multiprocessing helpers
# =============================================================================

def safe_shm_create(init_map: np.ndarray):
    """Create a SharedMemory segment and copy `init_map` into it.

    Returns `(shm, shm_arr)` where `shm_arr` is a NumPy view on the
    shared buffer. Caller is responsible for cleanup via `safe_shm_close`.
    """
    shm = mp_shm.SharedMemory(create=True, size=init_map.nbytes)
    shm_arr = np.ndarray(init_map.shape, dtype=np.uint8, buffer=shm.buf)
    shm_arr[:] = init_map[:]
    return shm, shm_arr


def safe_shm_close(shm) -> None:
    """Safely close and unlink a SharedMemory object.

    Silently ignores errors raised during close/unlink to avoid
    crashing the generator during cleanup.
    """
    if shm is None:
        return
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def worm(shm_name: str, height: int, width: int, start_x: int, start_y: int,
                step: int, stren: int, life: int, wid: int, seed: int,
                worm_x_list: list, worm_y_list: list, targets_list: list) -> None:
    """Module-level worker used by multiprocessing to erode the shared map.

    Attaches to an existing SharedMemory segment named `shm_name` and
    performs `life` iterations of erosion starting at `(start_x, start_y)`.
    This function is intentionally module-level so it is picklable on
    Windows when passed to `multiprocessing.Process`.
    """
    try:
        shm = mp_shm.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return
    bin_map = np.ndarray((height, width), dtype=np.uint8, buffer=shm.buf)
    rng = np.random.default_rng(seed)

    # Local direction helpers bound to this worker's RNG and context
    def border_control_local(x1: int, x2: int, y1: int, y2: int, s: int, current_dir: int, new_dir: bool = True) -> int:
        return border_control_helper(rng, x1, x2, y1, y2, s, current_dir, new_dir, width, height, Assets.MapGen.BORDER_THICKNESS)
    def homing_local(x: int, y: int, target_idx: int) -> int:
        return homing_helper(rng, x, y, worm_x_list[target_idx], worm_y_list[target_idx])
    
    dir_local = 0
    x = start_x
    y = start_y
    while life > 0:
        x1 = max(x - int(0.5 * stren), 0)
        y1 = max(y - int(0.5 * stren), 0)
        x2 = min(x + int(0.5 * stren), width-1)
        y2 = min(y + int(0.5 * stren), height-1)
        mode_choice = int(rng.integers(0, 5))
        sub = bin_map[y1:y2+1, x1:x2+1]
        apply_cv_brush(sub, x - x1, y - y1, mode_choice, stren, rng)
        dir_local = border_control_local(x1, x2, y1, y2, stren, dir_local)
        x, y = next_cell_coords(x, y, step, dir_local)
        life -= 1
    target = targets_list[wid]
    life2 = 100
    while ((x < (worm_x_list[target] - 2*step) or x > (worm_x_list[target] + 2*step) or
            y < (worm_y_list[target] - 2*step) or y > (worm_y_list[target] + 2*step)) and life2 > 0):
        dir_local = homing_local(x, y, target)
        x1 = max(x - stren, 0)
        y1 = max(y - stren, 0)
        x2 = min(x + stren, width-1)
        y2 = min(y + stren, height-1)
        strong = int(Assets.MapGen.BLUR_KERNEL_MULTIPLIER_STRENGTH * stren)
        mode_choice = int(rng.integers(0, 5))
        sub = bin_map[y1:y2+1, x1:x2+1]
        apply_cv_brush(sub, x - x1, y - y1, mode_choice, strong, rng)
        dir_local = border_control_local(x1, x2, y1, y2, stren, dir_local, new_dir=False)
        x, y = next_cell_coords(x, y, step, dir_local)
        life2 -= 1
    try:
        shm.close()
    except Exception:
        pass


def start_worms(shm_name: str, worker_count: int, worm_x: list, worm_y: list, worm_inputs: tuple,
                seed_base: int, targets: list, height: int, width: int) -> list:
    """
    Initializes and starts multiple worm processes for map generation.

    This function creates worker processes that simulate "worms" digging through a cave map.
    Each worm operates on shared memory and carves out passages based on its starting position
    and configuration parameters.

    Args:
        shm_name (str): Name of the shared memory block containing the map data.
        worker_count (int): Number of worm processes to create and start.
        worm_x (list): List of x-coordinates for each worm's starting position.
        worm_y (list): List of y-coordinates for each worm's starting position.
        worm_inputs (tuple): Tuple containing three integer parameters for worm behavior:
            - [0]: First worm configuration parameter (e.g., length or steps)
            - [1]: Second worm configuration parameter (e.g., direction change frequency)
            - [2]: Third worm configuration parameter (e.g., width or radius)
        seed_base (int): Base seed value for random number generation. Each worm gets seed_base + i.
        targets (list): List of target positions or objectives for the worms.
        height (int): Height of the map in the shared memory.
        width (int): Width of the map in the shared memory.

    Returns:
        list: List of Process objects representing the started worm worker processes.

    Note:
        - All worm processes are started before the function returns.
        - The caller is responsible for joining/terminating the returned processes.
        - Each worm receives its own unique seed for reproducible randomness.
    """
    proc_list = []
    for i in range(worker_count):
        args = (
            shm_name,
            height,
            width,
            int(worm_x[i]),
            int(worm_y[i]),
            int(worm_inputs[0]),
            int(worm_inputs[1]),
            int(worm_inputs[2]),
            i,
            int(seed_base + i),
            list(map(int, worm_x)),
            list(map(int, worm_y)),
            targets,
        )
        p = Process(target=worm, args=args)
        proc_list.append(p)
        p.start()
    return proc_list


def monitor_worms(proc_list: list, update_callback, poll_interval: float = 0.05) -> bool:
    """Poll worker processes, pump Pygame events, and report crashes.

    Calls `update_callback()` each time a worker finishes so callers can
    update progress UI. Returns True if any worker crashed or raised a
    non-zero exit code.
    """
    finished = [False] * len(proc_list)
    any_crashed = False
    try:
        while not all(finished):
            try:
                pygame.event.pump()
            except Exception:
                pass
            for idx, p in enumerate(proc_list):
                if finished[idx]:
                    continue
                if not p.is_alive():
                    try:
                        p.join(timeout=0)
                    except Exception:
                        pass
                    finished[idx] = True
                    exitcode = p.exitcode
                    if exitcode is not None and exitcode != 0:
                        any_crashed = True
                        print(f"MapGenerator: worker {idx} exited with code {exitcode}")
                    try:
                        update_callback()
                    except Exception:
                        pass
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print('MapGenerator: generation interrupted by user')
        any_crashed = True
    except Exception as e:
        print(f'MapGenerator: watchdog failed: {e}')
        any_crashed = True
    finally:
        for p in proc_list:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
            try:
                p.join(timeout=1)
            except Exception:
                pass
    return any_crashed


# =============================================================================
# Map generation logic helpers
# =============================================================================

def border_control_helper(rng: 'np.random.Generator', x1: int, x2: int, y1: int, y2: int, s: int,
                          current_dir: int, new_dir: bool, width: int, height: int,
                          border_thickness: int) -> int:
    """Return a new direction when a worm nears the map border.

    Uses `rng` to pick a corrective heading that keeps worms away from
    the hard border region.
    """
    if x2+(0.5*s) > width - border_thickness:
        return int(rng.integers(180, 360))
    if x1-(0.5*s) < border_thickness:
        return int(rng.integers(0, 180))
    if y2+(0.5*s) > height - border_thickness:
        if int(rng.integers(0, 2)):
            return int(rng.integers(0, 91))
        else:
            return int(rng.integers(270, 360))
    if y1-(0.5*s) < border_thickness:
        return int(rng.integers(90, 270))
    if new_dir:
        return int(rng.integers(0, 360))
    return current_dir


def homing_helper(rng: 'np.random.Generator', x: int, y: int, target_x: int, target_y: int) -> int:
    """Return a heading (degrees) that roughly points from `(x,y)` to the target.

    Adds a small randomized perturbation to the heading to avoid perfectly
    straight paths.
    """
    rad_dir = math.atan2((y - target_y), (x - target_x))
    deg_dir = (rad_dir if rad_dir >= 0 else (2*math.pi + rad_dir)) * 180 / math.pi
    target_dir = int(deg_dir - 90 if deg_dir >= 90 else deg_dir + 270)
    if int(rng.integers(0, 2)) == 0:
        target_dir = (target_dir + int(rng.integers(-90, 91))) % 360
    return target_dir % 360


def make_derangement(n: int, rng: 'np.random.Generator') -> list:
    """Return a derangement (permutation with no fixed points) of size `n`.

    Uses the provided `rng` for deterministic results.
    """
    if n <= 1:
        return list(range(n))
    while True:
        perm = rng.permutation(n).tolist()
        if all(i != perm[i] for i in range(n)):
            return list(map(int, perm))

