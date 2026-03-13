"""Procedural cave map generation utilities.

This module produces cave maps by simulating multiple "worms" that erode
an initial solid grid. It delegates low-level operations (brush drawing,
shared-memory workers and post-processing) to helpers in
`MapGenHelpers.py` so this file remains focused on orchestration.

Key behaviors:
- Spawn spawn-safe multiprocessing workers that operate on a SharedMemory
- Use OpenCV/NumPy for fast brush operations and post-processing
- Maintain deterministic results via `np.random.Generator` seeds
"""

import os
from typing import Tuple
import numpy as np
import cv2
import pygame
import Assets
from Assets import next_cell_coords
from MapGenHelpers import (
    make_derangement,
    with_surfarrays,
    apply_cv_brush,
    safe_shm_create,
    safe_shm_close,
    start_worms,
    monitor_worms,
    border_control_helper,
    homing_helper,
    remove_hermit_caves,
    add_wall_transition_noise,
)


class MapGenerator:
    """Orchestrates multi-process worm erosion and post-processing.

    Attributes:
    - `bin_map` (np.ndarray): binary map (1=wall, 0=floor)
    - `rng` (np.random.Generator): deterministic RNG for reproducible maps
    - `worm_x`, `worm_y`: lists of worm start coordinates
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

        self.build_terrain_roughness()
    

    def dig_map(self, proc_num: int) -> None:
        """Spawn worms to erode `self.bin_map` using a SharedMemory buffer.

        Uses helpers for creating the shared buffer, starting monitorable
        worms processes, and copying the final result back into
        `self.bin_map`.
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

            proc_list = start_worms(shm.name, worker_count, self.worm_x, self.worm_y, self.worm_inputs, self.settings.seed, targets, self.height, self.width)

            def _update_finished():
                self.proc_counter += 1

            any_crashed = monitor_worms(proc_list, _update_finished)

            if any_crashed:
                print('MapGenerator: one or more worms crashed during generation')

            # Copy result back
            self.bin_map = np.array(shm_arr, dtype=np.uint8)
        finally:
            try:
                safe_shm_close(shm)
            except Exception:
                pass


    def set_starts(self) -> None:
        """Initialize deterministic starting coordinates for worms.

        Positions include four corners, the center, and three RNG-based
        locations inside the map bounds (respecting border thickness).
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


    def set_ends(self, step: int, id: int) -> Tuple[int, int, int, int, int]:
        """Select a target worm and return its bounding connection region.

        Returns (x_min, x_max, y_min, y_max, target_index) describing the
        rectangle (expanded by `2*step`) around the chosen target.
        """
        # Produce a derangement of the targets for randomized pairing
        self.targets = make_derangement(len(self.targets), self.rng)
        target = self.targets[id]
        x_min = self.worm_x[target] - 2 * step
        x_max = self.worm_x[target] + 2 * step
        y_min = self.worm_y[target] - 2 * step
        y_max = self.worm_y[target] + 2 * step
        return x_min, x_max, y_min, y_max, target
    

    def connect_rooms(self, x: int, y: int, step: int, stren: int, id: int) -> None:
        """Eat toward a target region and carve a connection if necessary.

        Repeatedly casts an OpenCV brush around the current position and
        advances toward the assigned target until the connection area
        is reached or a life counter expires.
        """
        x_min, x_max, y_min, y_max, target = self.set_ends(step, id)
        life = 100 # Counter to limit iterations

        # Continue connecting rooms while target not reached and iterations remain
        while ((x < x_min or x > x_max or y < y_min or y > y_max) and life > 0):
            self.dir = homing_helper(self.rng, x, y, self.worm_x[target], self.worm_y[target]) # Determine direction towards target

            # Define the border limits for "eating" surrounding pixels
            x1 = max(x - stren, 0) 
            y1 = max(y - stren, 0)
            x2 = min(x + stren, self.width-1)
            y2 = min(y + stren, self.height-1)

            # Use OpenCV brush helper to apply the connect-room brush
            strong = int(Assets.MapGen.BLUR_KERNEL_MULTIPLIER_STRENGTH * stren)
            mode_choice = int(self.rng.integers(0, 5))
            sub = self.bin_map[y1:y2+1, x1:x2+1]
            apply_cv_brush(sub, x - x1, y - y1, mode_choice, strong, self.rng)
            
            # Control the borders after eating
            self.dir = border_control_helper(self.rng, x1, x2, y1, y2, stren, self.dir, False, self.width, self.height, Assets.MapGen.BORDER_THICKNESS)
            
            # Update coordinates for the next cell based on direction
            x, y = next_cell_coords(x, y, step, self.dir)
            life -= 1 # Decrease the life counter


    # Perform image processing of the raw map
    def process_map(self) -> None:
        """Post-process the raw binary map into the final cave layout.

        Steps:
        1. Apply a median blur to smooth worm edges
        2. Remove small isolated cavities
        3. Merge with the original to preserve strong walls
        4. Optionally add frame/stalactite masks and wall-transition noise
        5. Apply a final small blur to reduce single-pixel artifacts
        """
        self.game.menu.blit_loading(['Breeding bats...'])
        # Smooth the raw binary map and remove tiny isolated caves
        kernel_dim = int(max(1, (self.worm_inputs[1] - Assets.MapGen.MEDIAN_FILTER_REDUCTION) | 1))
        raw = self.bin_map.astype('uint8')
        smoothed = cv2.medianBlur(raw, kernel_dim)
        cleaned = remove_hermit_caves(smoothed)

        # Merge cleaned result with original to preserve strong walls, then
        # optionally add stalactite and wall-transition noise.
        stalac = cv2.bitwise_or(raw, cleaned)
        try:
            stalac = add_wall_transition_noise(stalac, self.width, self.height, self.settings.seed, self.worm_inputs)
        except Exception:
            pass

        # Final small blur to avoid single-pixel artifacts
        self.bin_map = cv2.medianBlur(stalac, Assets.MapGen.BLUR_KERNEL_FINAL)


    def build_terrain_roughness(self) -> None:
        """Create a floor-only asperity map used by drones and rovers.

        The generated map stays in `[0, 1]` where higher values represent
        rougher terrain. Roughness is synthesized from smooth noise,
        clustered bumps, and a wall-distance bias so chokepoints and cave
        edges tend to be harsher for rovers to traverse.
        """
        floor_mask = (self.bin_map == 0).astype(np.uint8)

        base_noise = self.rng.random((self.height, self.width), dtype=np.float32)
        base_noise = cv2.GaussianBlur(base_noise, (0, 0), sigmaX=18, sigmaY=18)
        base_noise = cv2.normalize(base_noise, None, 0.0, 1.0, cv2.NORM_MINMAX)

        cluster_noise = self.rng.random((self.height, self.width), dtype=np.float32)
        cluster_noise = cv2.GaussianBlur(cluster_noise, (0, 0), sigmaX=6, sigmaY=6)
        cluster_noise = cv2.normalize(cluster_noise, None, 0.0, 1.0, cv2.NORM_MINMAX)

        wall_bias = np.zeros((self.height, self.width), dtype=np.float32)
        if np.any(floor_mask):
            wall_distance = cv2.distanceTransform(floor_mask, cv2.DIST_L2, 5)
            max_distance = float(wall_distance.max()) or 1.0
            wall_bias = 1.0 - np.clip(wall_distance / max_distance, 0.0, 1.0)

        roughness = (0.45 * base_noise) + (0.35 * wall_bias) + (0.20 * cluster_noise)
        roughness = np.clip(roughness, 0.0, 1.0).astype(np.float32)
        roughness *= floor_mask.astype(np.float32)
        self.terrain_roughness = roughness


    def _extract_cave_layer(self, color_to_remove: Tuple[int, int, int], output_key: str) -> None:
        """Make `color_to_remove` transparent in the saved cave image.

        Loads the cave image, clears the alpha channel for matching pixels,
        and writes the result to `Assets.Images[output_key]`.
        """
        # Load the cave map image
        cave_map = pygame.image.load(Assets.Images.CAVE_MAP.value).convert_alpha()
        
        # Use context manager to safely access and release surfarray views
        with with_surfarrays(cave_map) as (rgb_arr, alpha_arr):
            mask = (rgb_arr == list(color_to_remove)).all(axis=2)
            alpha_arr[mask] = 0
        
        # Save the modified map
        pygame.image.save(cave_map, Assets.Images[output_key].value)


    def save_map(self) -> None:
        """Persist the generated map image and matrix to disk.

        Writes `map.png` (visual) and `map_matrix.txt` (numeric matrix) into
        `Assets.GAME_DIR/Assets/Map`.
        """
        map_dir = os.path.join(Assets.GAME_DIR, 'Assets', 'Map')
        os.makedirs(map_dir, exist_ok=True)
        byte_map = np.where(self.bin_map == 1, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(map_dir, 'map.png'), byte_map)
        np.savetxt(os.path.join(map_dir, 'map_matrix.txt'), self.bin_map)
