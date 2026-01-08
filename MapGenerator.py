"""MapGenerator: procedural cave map generation utilities.

This module generates cave maps by simulating multiple "worms" that erode
an initially solid grid. It uses multiprocessing with a shared-memory
buffer for performance and OpenCV/NumPy for fast brush operations.

Key features:
- SharedMemory-based multiprocessing (Windows-safe module-level worker)
- Numpy-based masks and OpenCV drawing for brush operations
- Deterministic behavior via `np.random.Generator` seeds
"""

import os
import math
from typing import Tuple
import numpy as np
import cv2
import pygame
import Assets
from Assets import sqr, next_cell_coords
from multiprocessing import Process
from multiprocessing import shared_memory as mp_shm
import time

# Cache for precomputed brush masks: key = (mode_int, stren)
_BRUSH_MASK_CACHE = {}

from contextlib import contextmanager


def make_derangement(n: int, rng: 'np.random.Generator') -> list:
    """Return a derangement (no fixed points) of range(n) using `rng`.

    This keeps derangement logic in one place and avoids duplicated code.
    """
    if n <= 1:
        return list(range(n))
    while True:
        perm = rng.permutation(n).tolist()
        if all(i != perm[i] for i in range(n)):
            return list(map(int, perm))


@contextmanager
def with_surfarrays(surface: 'pygame.Surface'):
    """Context manager that yields (rgb_arr, alpha_arr) from a surface and
    ensures the arrays are released (deleted) afterward.
    """
    rgb_arr = pygame.surfarray.pixels3d(surface)
    alpha_arr = pygame.surfarray.pixels_alpha(surface)
    try:
        yield rgb_arr, alpha_arr
    finally:
        del rgb_arr
        del alpha_arr


def get_brush_mask(mode_choice: int, stren: int):
    """Return (mask, radius) for given brush mode and strength.
    mode_choice: 0=CIRCLE,1=ELLIPSE,2=DIAMOND,3=OCTAGON,4=CHAOTIC
    For chaotic (4) we return None to indicate on-the-fly generation.
    """
    if mode_choice == 4:
        return None, None

    key = (int(mode_choice), int(stren))
    if key in _BRUSH_MASK_CACHE:
        return _BRUSH_MASK_CACHE[key]

    # radius: choose a safe radius to cover shapes (use 0.75*stren)
    r = max(1, int(math.ceil(0.75 * float(stren))))
    dx = np.arange(-r, r + 1)
    dy = np.arange(-r, r + 1)
    XX, YY = np.meshgrid(dx, dy, indexing='xy')  # rows: len(dy), cols: len(dx)

    if mode_choice == 0:  # CIRCLE
        mask = (XX ** 2 + YY ** 2) <= (0.5 * stren) ** 2
    elif mode_choice == 1:  # ELLIPSE (scaled Y)
        mask = (XX ** 2 + 6 * YY ** 2) <= (0.5 * stren) ** 2
    elif mode_choice == 2:  # DIAMOND (Manhattan)
        mask = (np.abs(XX) + np.abs(YY)) <= (0.5 * stren)
    elif mode_choice == 3:  # OCTAGON (approx with Manhattan)
        mask = (np.abs(XX) + np.abs(YY)) <= (0.75 * stren)
    else:
        mask = np.ones_like(XX, dtype=bool)

    _BRUSH_MASK_CACHE[key] = (mask, r)
    return mask, r


def apply_cv_brush(sub: np.ndarray, cx: float, cy: float, mode_choice: int, stren: int, rng=None) -> None:
    """Apply a filled brush onto `sub` (2D uint8 array) using OpenCV drawing.
    sub is modified in-place; pixels equal to 1 will be set to 0 under the brush.
    cx,cy are coordinates relative to `sub` (0..w-1, 0..h-1).
    mode_choice: 0..4 as in get_brush_mask. stren: strength parameter.
    rng: optional numpy Generator for chaotic mode.
    """
    h, w = sub.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    # radius for drawing (approximate)
    radius = max(1, int(0.5 * stren))

    icx = int(round(cx))
    icy = int(round(cy))

    try:
        if mode_choice == 0:  # circle
            cv2.circle(mask, (icx, icy), radius, 1, thickness=-1)
        elif mode_choice == 1:  # ellipse (scaled Y)
            # approximate ellipse axes
            rx = max(1, radius)
            ry = max(1, int(max(1, radius / 2)))
            cv2.ellipse(mask, (icx, icy), (rx, ry), 0, 0, 360, 1, thickness=-1)
        elif mode_choice == 2:  # diamond
            pts = np.array([[[icx + radius, icy], [icx, icy + radius], [icx - radius, icy], [icx, icy - radius]]], dtype=np.int32)
            cv2.fillPoly(mask, pts, 1)
        elif mode_choice == 3:  # octagon (approx)
            r = radius
            pts = np.array([[[icx + int(0.6 * r), icy + r], [icx + r, icy + int(0.6 * r)], [icx + r, icy - int(0.6 * r)], [icx + int(0.6 * r), icy - r], [icx - int(0.6 * r), icy - r], [icx - r, icy - int(0.6 * r)], [icx - r, icy + int(0.6 * r)], [icx - int(0.6 * r), icy + r]]], dtype=np.int32)
            cv2.fillPoly(mask, pts, 1)
        else:
            # chaotic: generate radial random factor
            local_rng = rng if rng is not None else np.random.default_rng()
            factor = float(local_rng.uniform(Assets.MapGen.CHAOTIC_FACTOR_LOW, Assets.MapGen.CHAOTIC_FACTOR_HIGH))
            rr = max(1, int(round(factor * stren)))
            # create small bbox to limit work
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
        # drawing failed; fallback to simple numpy radial mask
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

    # Apply mask: clear only where value==1 (wall)
    sub[(mask == 1) & (sub == 1)] = 0


def safe_shm_create(init_map: np.ndarray):
    """Create a SharedMemory block and return (shm, shm_array).

    Caller is responsible for calling `safe_shm_close(shm)` when done.
    """
    shm = mp_shm.SharedMemory(create=True, size=init_map.nbytes)
    shm_arr = np.ndarray(init_map.shape, dtype=np.uint8, buffer=shm.buf)
    shm_arr[:] = init_map[:]
    return shm, shm_arr


def safe_shm_close(shm) -> None:
    """Close and unlink a SharedMemory block, ignoring harmless errors."""
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


def start_workers(shm_name: str, worker_count: int, worm_x: list, worm_y: list, worm_inputs: tuple,
                  seed_base: int, targets: list, height: int, width: int) -> list:
    """Start `worker_count` `worm_worker` processes and return the Process list."""
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
        p = Process(target=worm_worker, args=args)
        proc_list.append(p)
        p.start()
    return proc_list


def monitor_workers(proc_list: list, update_callback, poll_interval: float = 0.05) -> bool:
    """Monitor processes, call `update_callback()` when each finishes.

    Returns True if any child exited with a non-zero exit code or if
    an interruption/error occurred.
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


def choose_brush_helper(rng: 'np.random.Generator', x1: int, y1: int, x2: int, y2: int, s: int) -> bool:
    """Return True if a brush of strength `s` at (x1,y1)-(x2,y2) should clear (x2,y2).

    `rng` is expected to be a NumPy `Generator`.
    """
    mode = Assets.Brush(int(rng.integers(0, 5))).name
    if mode == 'CIRCLE':
        dist = math.sqrt(sqr(x2-x1) + sqr(y2-y1))
        return dist < (0.5*s)
    if mode == 'ELLIPSE':
        dist = sqr(x2-x1) + 6*sqr(y2-y1)
        return dist < sqr(0.5*s)
    if mode == 'DIAMOND':
        dist = abs(x2-x1) + abs(y2-y1)
        return dist < (0.5*s)
    if mode == 'OCTAGON':
        dist = abs(x2-x1) + abs(y2-y1)
        return dist < (0.75*s)
    if mode == 'CHAOTIC':
        dist = math.sqrt(sqr(x2-x1) + sqr(y2-y1))
        return dist < (float(rng.uniform(Assets.MapGen.CHAOTIC_FACTOR_LOW, Assets.MapGen.CHAOTIC_FACTOR_HIGH)) * s)
    return True


def border_control_helper(rng: 'np.random.Generator', x1: int, x2: int, y1: int, y2: int, s: int,
                          current_dir: int, new_dir: bool, width: int, height: int,
                          border_thickness: int) -> int:
    """Return a direction (0-359) that keeps worms away from map borders.

    Uses the provided `rng` for any random decisions so behavior is reproducible.
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


def homing_sys_helper(rng: 'np.random.Generator', x: int, y: int, target_x: int, target_y: int) -> int:
    """Compute a heading (degrees 0-359) from (x,y) towards (target_x,target_y).

    A small random perturbation is applied via `rng` to avoid perfectly straight paths.
    """
    rad_dir = math.atan2((y - target_y), (x - target_x))
    deg_dir = (rad_dir if rad_dir >= 0 else (2*math.pi + rad_dir)) * 180 / math.pi
    target_dir = int(deg_dir - 90 if deg_dir >= 90 else deg_dir + 270)
    if int(rng.integers(0, 2)) == 0:
        target_dir = (target_dir + int(rng.integers(-90, 91))) % 360
    return target_dir % 360


def worm_worker(shm_name: str, height: int, width: int, start_x: int, start_y: int,
                step: int, stren: int, life: int, wid: int, seed: int,
                worm_x_list: list, worm_y_list: list, targets_list: list) -> None:
    """Module-level worker for map erosion. Runs in a child process and
    modifies the shared memory array in-place. Avoids capturing pygame/game
    objects so it's picklable on Windows."""
    # Use NumPy Generator for fast, reproducible RNG in child processes
    from multiprocessing import shared_memory as _mp_shm
    import numpy as _np

    # Attach to shared memory
    try:
        shm = _mp_shm.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return

    bin_map = _np.ndarray((height, width), dtype=_np.uint8, buffer=shm.buf)

    rng = _np.random.default_rng(seed)

    # use module-level pure helpers to avoid duplicating logic
    def choose_brush_local(x1: int, y1: int, x2: int, y2: int, s: int) -> bool:
        return choose_brush_helper(rng, x1, y1, x2, y2, s)

    def border_control_local(x1: int, x2: int, y1: int, y2: int, s: int, current_dir: int, new_dir: bool = True) -> int:
        return border_control_helper(rng, x1, x2, y1, y2, s, current_dir, new_dir, width, height, Assets.MapGen.BORDER_THICKNESS)

    def homing_sys_local(x: int, y: int, target_idx: int) -> int:
        return homing_sys_helper(rng, x, y, worm_x_list[target_idx], worm_y_list[target_idx])

    # Worm main loop
    dir_local = 0
    x = start_x
    y = start_y
    while life > 0:
        x1 = max(x - int(0.5 * stren), 0)
        y1 = max(y - int(0.5 * stren), 0)
        x2 = min(x + int(0.5 * stren), width-1)
        y2 = min(y + int(0.5 * stren), height-1)

        # Choose a brush mode once per step to approximate original stochastic behavior
        mode_choice = int(rng.integers(0, 5))
        sub = bin_map[y1:y2+1, x1:x2+1]
        # Use OpenCV-based brush drawing for speed; falls back inside helper if needed
        apply_cv_brush(sub, x - x1, y - y1, mode_choice, stren, rng)

        dir_local = border_control_local(x1, x2, y1, y2, stren, dir_local)
        x, y = next_cell_coords(x, y, step, dir_local)
        life -= 1

    # Connect rooms: worker computes its target from provided targets_list
    target = targets_list[wid]
    life2 = 100
    while ((x < (worm_x_list[target] - 2*step) or x > (worm_x_list[target] + 2*step) or
            y < (worm_y_list[target] - 2*step) or y > (worm_y_list[target] + 2*step)) and life2 > 0):
        dir_local = homing_sys_local(x, y, target)
        x1 = max(x - stren, 0)
        y1 = max(y - stren, 0)
        x2 = min(x + stren, width-1)
        y2 = min(y + stren, height-1)
        # Vectorized connect-room brush (stronger brush). Use precomputed masks when possible.
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


class MapGenerator():
    """High-level generator orchestrating worm-based erosion and post-processing.

    Public attributes:
    - `bin_map`: 2D NumPy array (uint8) representing the map (1=wall, 0=floor)
    - `rng`: NumPy `Generator` used for deterministic randomness
    """

    # Configuration constants (moved to Assets.MapGen where appropriate)
    NUM_PROCESSES = getattr(Assets.MapGen, 'DEFAULT_NUM_PROCESSES', 8)
    # Color aliases
    COLOR_WHITE = Assets.Colors.WHITE.value
    COLOR_BLACK = Assets.Colors.BLACK.value
    
    def __init__(self, game, settings) -> None:
        self.game     = game
        self.settings = settings
        self.width    = Assets.Display.FULL_W - Assets.Display.LEGEND_WIDTH
        self.height   = Assets.Display.FULL_H
        
        # Use a NumPy Generator for reproducible RNG
        self.rng = np.random.default_rng(self.settings.seed)

        # Initialise worm management settings
        self.worm_inputs  = Assets.WormInputs[self.settings.map_dim].value # Inputs for worm behavior
        self.targets      = list(map(int, np.linspace(0, self.NUM_PROCESSES - 1, self.NUM_PROCESSES))) # Targets for each worm
        self.proc_counter = 0 # Counter for completed processes
        self.dir          = 0 # Initialize direction (updated during worm movement)
        self.set_starts()  # Set the starting positions of the worms

        if self.settings.prefab: # Load existing map
            self.bin_map = np.loadtxt(Assets.Images.CAVE_MATRIX.value)
        else: # Generate new map
            # Initialize binary map with all walls (1s)
            self.bin_map = np.ones([self.height,self.width])
            self.dig_map(self.NUM_PROCESSES)

            # Perform image processing on the generated map
            self.process_map()

            # Save and display the processed map
            self.save_map()
            self._extract_cave_layer(self.COLOR_WHITE, 'CAVE_WALLS')
            self._extract_cave_layer(self.COLOR_BLACK, 'CAVE_FLOOR')
    

    
    # Create multiple worms and let them eat away the map simultaneously
    # while displaying the loading screen
    def dig_map(self, proc_num: int) -> None:
        """Spawn `proc_num` workers and run them against a SharedMemory buffer.

        This implementation delegates process start/monitor and shared
        memory lifecycle management to small helpers to keep the method
        concise and readable.
        """

        # Limit worker count to available CPUs
        num_cpus = os.cpu_count() or 1
        worker_count = min(proc_num, num_cpus)

        try:
            init_map = self.bin_map.astype(np.uint8)
            shm, shm_arr = safe_shm_create(init_map)

            # Show a simple, non-process-dependent loading message
            try:
                self.game.menu.blit_loading(['Digging...'])
            except Exception:
                pass

            # Derangement targets
            targets = list(range(worker_count))
            shuffled = False
            while not shuffled:
                targets = make_derangement(len(targets), self.rng)
                shuffled = True

            proc_list = start_workers(shm.name, worker_count, self.worm_x, self.worm_y, self.worm_inputs, self.settings.seed, targets, self.height, self.width)

            def _update_finished():
                self.proc_counter += 1

            any_crashed = monitor_workers(proc_list, _update_finished)

            if any_crashed:
                print('MapGenerator: one or more workers crashed during generation')

            # Copy result back
            self.bin_map = np.array(shm_arr, dtype=np.uint8)
        finally:
            try:
                safe_shm_close(shm)
            except Exception:
                pass

    # Model a worm that randomly eats away at the map
    def worm(self, shm_name: str, height: int, width: int, x: int, y: int, step: int, stren: int, life: int, id: int, seed: int) -> None:
        """Class-bound worm used for single-process testing.

        Attaches to the SharedMemory buffer named `shm_name` and performs
        erosion in-place on the shared array. This is the non-multiprocess
        variant kept for debugging and single-process runs.
        """
        # Attach to shared memory and expose it as `self.bin_map` so
        # helper methods can operate on the shared array.
        try:
            shm = mp_shm.SharedMemory(name=shm_name)
        except FileNotFoundError:
            return

        self.bin_map = np.ndarray((height, width), dtype=np.uint8, buffer=shm.buf)

        # Seed local NumPy RNG to avoid identical randomness across workers
        rng = np.random.default_rng(seed)

        while life > 0: # Continue while the worm has life
            # Calculate the borders for the worm's eating area
            x1 = max(x - int(0.5 * stren), 0) # Adjust for legend
            y1 = max(y - int(0.5 * stren), 0)
            x2 = min(x + int(0.5 * stren), self.width-1)
            y2 = min(y + int(0.5 * stren), self.height-1)
            # Use OpenCV brush helper to apply the brush into the subarray
            mode_choice = int(rng.integers(0, 5))
            sub = self.bin_map[y1:y2+1, x1:x2+1]
            apply_cv_brush(sub, x - x1, y - y1, mode_choice, stren, rng)
            
            self.border_control(x1, x2, y1, y2, stren) # Manage borders to keep worms in bounds
            
            # Move to the next cell based on the current direction
            x, y = next_cell_coords(x, y, step, self.dir)
            life -= 1 # Decrease life of the worm

        # Connect rooms after worm finishes eating
        self.connect_rooms(x, y, step, stren, id)
        
        # Update the loading screen with the current process count
        self.proc_counter += 1
        # process-dependent loading screen removed; keep counter for bookkeeping

        # Detach shared memory reference in this worker
        try:
            shm.close()
        except Exception:
            pass


    
    # Set starting positions for the worms
    def set_starts(self) -> None:
        """Initialize worm starting coordinates around the map.

        Uses `self.rng` for deterministic random starts when applicable.
        """

        self.worm_x = list(map(int, [self.width/4,      # Top Left
                                     3*self.width/4,    # Top Right
                                     3*self.width/4,    # Bottom Right
                                     self.width/4,      # Bottom Left
                                     self.width/2,      # Center
                                     int(self.rng.integers(Assets.MapGen.BORDER_THICKNESS, self.width - Assets.MapGen.BORDER_THICKNESS + 1)),
                                     int(self.rng.integers(Assets.MapGen.BORDER_THICKNESS, self.width - Assets.MapGen.BORDER_THICKNESS + 1)),
                                     int(self.rng.integers(Assets.MapGen.BORDER_THICKNESS, self.width - Assets.MapGen.BORDER_THICKNESS + 1))]))
        
        self.worm_y = list(map(int, [self.height/4,     # Top Left
                                     self.height/4,     # Top Right
                                     3*self.height/4,   # Bottom Right
                                     3*self.height/4,   # Bottom Left
                                     self.height/2,     # Center
                                     # Random
                                     int(self.rng.integers(Assets.MapGen.BORDER_THICKNESS, self.height - Assets.MapGen.BORDER_THICKNESS + 1)),
                                     int(self.rng.integers(Assets.MapGen.BORDER_THICKNESS, self.height - Assets.MapGen.BORDER_THICKNESS + 1)),
                                     int(self.rng.integers(Assets.MapGen.BORDER_THICKNESS, self.height - Assets.MapGen.BORDER_THICKNESS + 1))]))

    # Avoid collision with the window borders
    def border_control(self, x1: int, x2: int, y1: int, y2: int, stren: int, new_dir: bool = True) -> None:
        # Delegate to module-level helper
        self.dir = border_control_helper(self.rng, x1, x2, y1, y2, stren, self.dir, new_dir, self.width, self.height, Assets.MapGen.BORDER_THICKNESS)

    # Choose the shape of the eaten part
    def choose_brush(self, x1: int, y1: int, x2: int, y2: int, stren: int) -> bool:
        return choose_brush_helper(self.rng, x1, y1, x2, y2, stren)

    
    # Ensure there are no inaccessible rooms
    def connect_rooms(self, x: int, y: int, step: int, stren: int, id: int) -> None:
        # Get the target coordinates and the boundaries for room connection
        x_min, x_max, y_min, y_max, target = self.assign_target(step, id)
        life = 100 # Counter to limit iterations

        # Continue connecting rooms while target not reached and iterations remain
        while ((x < x_min or x > x_max or y < y_min or y > y_max) and life > 0):
            self.dir = self.homing_sys(x, y, target) # Determine direction towards target

            # Define the border limits for "eating" surrounding pixels
            x1 = max(x - stren, 0) 
            y1 = max(y - stren, 0)
            x2 = min(x + stren, self.width-1)
            y2 = min(y + stren, self.height-1)

            # Use OpenCV brush helper to apply the connect-room brush
            strong = int(self.BLUR_KERNEL_MULTIPLIER_STRENGTH * stren)
            mode_choice = int(self.rng.integers(0, 5))
            sub = self.bin_map[y1:y2+1, x1:x2+1]
            apply_cv_brush(sub, x - x1, y - y1, mode_choice, strong, self.rng)
            
            # Control the borders after eating
            self.border_control(x1, x2, y1, y2, stren, new_dir=False)
            
            # Update coordinates for the next cell based on direction
            x, y = next_cell_coords(x, y, step, self.dir)
            life -= 1 # Decrease the life counter

    # Set the course for the starting point of the closest worm
    def homing_sys(self, x: int, y: int, target: int) -> int:
        return homing_sys_helper(self.rng, x, y, self.worm_x[target], self.worm_y[target])

    # Assign the target where the worm goes to die
    def assign_target(self, step: int, id: int) -> Tuple[int, int, int, int, int]:
        # Shuffle the targets list to randomize selection (derangement)
        shuffled = False
        while not shuffled:
            self.targets = make_derangement(len(self.targets), self.rng)
            shuffled = True
        
        target = self.targets[id] # Get the selected target
        
        # Set boundaries based on target's position
        x_min = self.worm_x[target] - 2*step
        x_max = self.worm_x[target] + 2*step
        y_min = self.worm_y[target] - 2*step
        y_max = self.worm_y[target] + 2*step

        # Return boundaries and target
        return x_min, x_max, y_min, y_max, target

    # Remove isolated caves
    def remove_hermit_caves(self, image: np.ndarray) -> np.ndarray:
        # Create a binary mask by inverting the image
        inverted_image = np.where(image == 0, 1, 0).astype('uint8')
        # Find connected components with statistics in the inverted image
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_image, connectivity=8)

        # If only background exists, return the original image
        if num_labels <= 1:
            return image

         # Identify the index of the largest connected region
        biggest_blob_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        
        # Create a mask to keep only the biggest blob
        mask_to_keep = np.ones_like(labels, dtype=bool)
        mask_to_keep[labels != biggest_blob_index] = False
        
        # Apply the mask to the original image
        cleaned_image = np.where(mask_to_keep, image, 1) # Keep the largest region, clear others

        return cleaned_image
    
    # Change the color of all the pixels within a frame of thickness self.BORDER_THICKNESS to BLACK
    def add_frame(self, image: np.ndarray) -> np.ndarray:
        # Create border mask
        border_mask = np.zeros((self.height, self.width), dtype=bool)
        border_mask[:self.BORDER_THICKNESS, :] = True
        border_mask[-self.BORDER_THICKNESS:, :] = True
        border_mask[:, :self.BORDER_THICKNESS] = True
        border_mask[:, -self.BORDER_THICKNESS:] = True
        
        # Set border pixels to 1 if they are 0
        image[(image == 0) & border_mask] = 1

        return self.mask_frame(image)
    
    # Add stalactites randomly along the frame (only in white areas within [0:7] pixels from the border)
    def mask_frame(self, image: np.ndarray) -> np.ndarray:
        # Define the random generator
        rng = np.random.default_rng(seed=self.settings.seed)

        # Define how far into the map the stalactites are added
        # (Randomly between configured min and max changing value every chunk)
        depth_range = list(range(Assets.MapGen.STALACTITE_MIN_DEPTH, Assets.MapGen.STALACTITE_MAX_DEPTH + 1))
        mask_h = np.repeat(rng.choice(depth_range, size=math.ceil(self.width / Assets.MapGen.STALACTITE_CHUNK_SIZE)), Assets.MapGen.STALACTITE_CHUNK_SIZE)[:self.width]
        mask_v = np.repeat(rng.choice(depth_range, size=math.ceil(self.height / Assets.MapGen.STALACTITE_CHUNK_SIZE)), Assets.MapGen.STALACTITE_CHUNK_SIZE)[:self.height]

        # Create grids
        i_arr = np.arange(self.width)
        j_arr = np.arange(self.height)
        I_grid, J_grid = np.meshgrid(i_arr, j_arr, indexing='ij')

        # Conditions
        cond1 = (I_grid >= self.BORDER_THICKNESS) & (I_grid < self.BORDER_THICKNESS + mask_h[I_grid])
        cond2 = (I_grid > self.width - self.BORDER_THICKNESS - mask_h[I_grid]) & (I_grid <= self.width - self.BORDER_THICKNESS)
        cond3 = (J_grid >= self.BORDER_THICKNESS) & (J_grid < self.BORDER_THICKNESS + mask_v[J_grid])
        cond4 = (J_grid > self.height - self.BORDER_THICKNESS - mask_v[J_grid]) & (J_grid <= self.height - self.BORDER_THICKNESS)
        
        total_mask = cond1 | cond2 | cond3 | cond4
        
        # Generate random values for the masked area
        random_vals = rng.choice([0, 1], size=(self.height, self.width), p=[Assets.MapGen.STALACTITE_PROBABILITY_AIR, Assets.MapGen.STALACTITE_PROBABILITY_STONE])
        
        # Apply where image == 0 and in mask
        mask = (image == 0) & total_mask
        image[mask] = random_vals[mask]
        
        return image

    # Add roughness near walls to create natural floor-wall transitions and small spikes
    def add_wall_transition_noise(self, image: np.ndarray) -> np.ndarray:
        """Introduce noisy floor-to-wall transitions and small stalactite/stalagmite spikes.

        This modifies a binary map (0=floor,1=wall) and returns a new array with
        added roughness. Deterministic with respect to `self.settings.seed`.
        """
        rng = np.random.default_rng(seed=(self.settings.seed or 0) + 2)

        img = image.copy()
        # floor mask: 1 where floor
        floor_mask = (img == 0).astype(np.uint8)

        # Distance (in pixels) from each floor pixel to nearest wall
        try:
            dt = cv2.distanceTransform(floor_mask, cv2.DIST_L2, 5)
        except Exception:
            # fallback: simple Manhattan distance approximation using convolution
            dt = np.full_like(img, 999.0, dtype=float)
            ys, xs = np.nonzero(floor_mask)
            for y, x in zip(ys, xs):
                # minimal brute-force distance (only for small maps or fallback cases)
                dists = np.abs(np.arange(self.width) - x)[None, :] + np.abs(np.arange(self.height) - y)[:, None]
                dt[y, x] = dists.min()

        # noise depth controls how far from the wall we add transitional noise
        # increase for a stronger, more visible transition
        noise_depth = max(4, int(max(4, (self.worm_inputs[0] // 2))))
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

        # Add small spike protrusions (stalactites/stalagmites) from wall pixels
        wall_coords = np.transpose(np.nonzero(img == 1))
        if wall_coords.shape[0] > 0:
            # more spikes for stronger appearance
            N = max(24, (self.width + self.height) // 25)
            N = min(N, wall_coords.shape[0])
            picks = rng.choice(wall_coords.shape[0], size=N, replace=False)
            spike_mask = np.zeros_like(img, dtype=np.uint8)
            for p in picks:
                y0, x0 = int(wall_coords[p, 0]), int(wall_coords[p, 1])
                length = int(rng.integers(2, noise_depth + 6))
                angle = float(rng.uniform(0, 2 * math.pi))
                dx = int(round(math.cos(angle) * length))
                dy = int(round(math.sin(angle) * length))
                x1 = max(0, min(self.width - 1, x0 + dx))
                y1 = max(0, min(self.height - 1, y0 + dy))
                try:
                    cv2.line(spike_mask, (x0, y0), (x1, y1), 1, thickness=2)
                except Exception:
                    # ignore drawing errors
                    continue
            # Slightly thicken spikes for visibility
            try:
                kernel = np.ones((3, 3), dtype=np.uint8)
                spike_mask = cv2.dilate(spike_mask, kernel, iterations=1)
            except Exception:
                pass
            img[spike_mask == 1] = 1

        return img


    
    # Perform image processing of the raw map
    def process_map(self) -> None:
        self.game.menu.blit_loading(['Breeding bats...']) # Display loading message

        # Define the kernel dimensions for image processing
        kernel_dim = self.worm_inputs[1] - Assets.MapGen.MEDIAN_FILTER_REDUCTION
        # Ensure kernel is odd and at least 1 for cv2.medianBlur
        kernel_dim = int(max(1, kernel_dim | 1))
        input_map  = self.bin_map.astype("uint8") # Convert binary map to unsigned 8-bit integer
        
        # Apply a median blur filter to smooth out borders
        processed_map = cv2.medianBlur(input_map, kernel_dim)

        # Remove isolated caves from the processed map
        clean_map = self.remove_hermit_caves(processed_map)
        
        # Add stalactites to the smoothed cave by combining maps
        stalac_map = cv2.bitwise_or(input_map, clean_map)
        # Introduce small border stalactites / rough transitions
        try:
            stalac_map = self.mask_frame(stalac_map)
        except Exception:
            pass

        # Add additional wall-adjacent noise for rough floor-wall transitions
        try:
            stalac_map = self.add_wall_transition_noise(stalac_map)
        except Exception:
            pass

        # Apply a smaller median blur to avoid creating single pixel stalactites
        self.bin_map = cv2.medianBlur(stalac_map, Assets.MapGen.BLUR_KERNEL_FINAL)  # Update the binary map

        # Add black frame in case worms were too greedy
        # (Should not happen, but let's keep it)
        #self.bin_map = self.add_frame(self.bin_map)

    def _extract_cave_layer(self, color_to_remove: Tuple[int, int, int], output_key: str) -> None:
        """
        Extract a cave layer by making specified color transparent and saving result.
        
        Args:
            color_to_remove: RGB tuple of color to make transparent.
            output_key: Key in Assets.Images to save the result to.
        """
        # Load the cave map image
        cave_map = pygame.image.load(Assets.Images.CAVE_MAP.value).convert_alpha()
        
        # Use context manager to safely access and release surfarray views
        with with_surfarrays(cave_map) as (rgb_arr, alpha_arr):
            mask = (rgb_arr == list(color_to_remove)).all(axis=2)
            alpha_arr[mask] = 0
        
        # Save the modified map
        pygame.image.save(cave_map, Assets.Images[output_key].value)

    # Save the generated map
    def save_map(self) -> None:
        # Ensure the output folder exists, create if it does not
        map_dir = os.path.join(Assets.GAME_DIR, 'Assets', 'Map')
        os.makedirs(map_dir, exist_ok=True)

        # Convert binary map values from [0,1] to [0,255] for image representation
        byte_map = np.where(self.bin_map == 1, 0, 255)

        # Save the map as a PNG image
        cv2.imwrite(os.path.join(map_dir, 'map.png'), byte_map)

        # Save the map matrix as a text file 
        np.savetxt(os.path.join(map_dir, 'map_matrix.txt'), self.bin_map)
