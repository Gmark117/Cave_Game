# MapGenerator module

Overview
-------
`MapGenerator.py` implements procedural cave generation by simulating
multiple concurrent "worms" that erode solid terrain. Key design points:

- Uses `multiprocessing.shared_memory` to allow workers to modify the map
  in-place without heavy copying.
- Uses `numpy` for array operations and `cv2` (OpenCV) to accelerate brush
  drawing via rasterized shapes, reducing Python-level loops.
- Randomness is controlled with `numpy.random.Generator` to ensure
  deterministic behavior per-seed and per-worker.

Important functions/classes
---------------------------
- `worm_worker(...)` — module-level multiprocessing worker attached to
  SharedMemory (picklable on Windows). Modifies the shared binary map in-place.
- `MapGenerator` — orchestrates generation, contains `dig_map`, `worm`, and
  post-processing routines (`process_map`, `remove_hermit_caves`, `mask_frame`).
- `apply_cv_brush(...)` — fast OpenCV-based brush application used by both
  worker and class-bound worms.

Usage notes
-----------
- Run the game as usual with `python main.py`; the generator will run during
  startup if `settings.prefab` is false.
- If you want to debug generation without multiprocessing, call
  `MapGenerator.worm(...)` directly (it attaches to SharedMemory for parity).

Performance & tuning
--------------------
- The number of processes is capped at CPU count to avoid oversubscription.
- Brush sizes and worm parameters are controlled by `Assets.WormInputs`.
- For further speedups, consider:
  - Offloading more routines to OpenCV/Numba.
  - Profiling with a small harness (TODO).

Notes about recent changes
-------------------------
- Replaced Python `random` usage with `numpy.random.Generator` across hot
  paths for reproducibility and performance.
- Added `apply_cv_brush` to use OpenCV drawing primitives instead of
  large NumPy meshgrid computations.
- Added robust watchdog and SharedMemory cleanup in `dig_map`.

Next steps
----------
- Add a small benchmarking script to quantify CPU/shared-memory/OpenCV gains.
- Add unit tests that exercise `worm_worker` on tiny SharedMemory arrays.

