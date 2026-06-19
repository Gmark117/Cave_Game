# Session Handoff: Codebase Refactor Through Step 3

Date: June 19, 2026

## Start Here

The repository is in the middle of a large, uncommitted refactor. Do not
discard, reset, or revert the existing dirty worktree. Read these files first:

1. `docs/SESSION_HANDOFF_REFACTOR_STEP4.md`
2. `CODEFLOW.md`
3. `TESTING.md`
4. `README.md`
5. `MissionControl.py`
6. `MissionControlLifecycle.py`
7. `MissionControlTerrain.py`

The next planned task is Step 4: simplify `TerrainSharingService`.

## Current Verification

Latest automated result:

```text
python -m unittest discover -s tests -v
Ran 108 tests
OK
```

Additional checks passed:

```text
python -m compileall -q .
git diff --check
```

No interactive mission smoke run was performed after the final Step 3
documentation and test changes.

## Worktree Warning

The refactor is not committed. Tracked files are modified and the new package
directories, tests, `CODEFLOW.md`, and `TESTING.md` are untracked.

Current high-level dirty state:

- Modified: `AgentFactory.py`, `Drone.py`, `Game.py`, `Graph.py`,
  `MapGenHelpers.py`, `Menu.py`, `MissionControl.py`,
  `MissionControlLifecycle.py`, `MissionControlTerrain.py`, `README.md`,
  `Rover.py`, `SimSettings.py`, and `GameConfig/symSettings.ini`.
- New: `agents/`, `mapping/`, `mission/`, `navigation/`, `rendering/`,
  `tests/`, `CODEFLOW.md`, and `TESTING.md`.

`GameConfig/symSettings.ini` now includes:

```ini
render_interval = 0.1
rover_share_interval = 0.5
```

Work with these changes. Do not assume uncommitted files are disposable.

## Refactor Progress

### Foundation Work

The codebase was mapped in `CODEFLOW.md`, then responsibilities were extracted
from the large mission and agent classes:

- `agents/drone_movement.py`: drone exploration and homing.
- `mapping/drone_sensor.py`: ray casting, local SLAM, terrain sampling.
- `mapping/terrain_fusion.py`: mission telemetry aggregation.
- `mapping/terrain_sharing.py`: proximity and sharing decisions.
- `mapping/rover_targets.py`: provisional rover target reservation.
- `navigation/pathfinding.py`: shared memory and worker pool lifecycle.
- `rendering/mission_renderer.py`: complete frame composition.
- `rendering/slam_view.py`: cached SLAM and heatmap views.
- `rendering/agent_renderer.py`: drone and rover drawing.
- `mission/debug_info.py`: control-center debug text.
- `mission/frame_timing.py`: smoothed frame-stage telemetry.

`MissionControl` construction is setup-only. `run()` owns runtime allocation,
thread startup, the main loop, and teardown. Controllers remain single-use.

### Step 1: Project Test Suite

A broad `unittest` suite was added across the project. Testing placement and
intent are documented in `TESTING.md`.

### Step 2: Terrain Knowledge Ownership

`mapping/terrain_knowledge.py` now owns:

- roughness and confidence arrays,
- synchronization,
- snapshots,
- terrain observation fusion,
- explored ratios,
- confidence-weighted merging.

Mission control, every drone, and every rover own separate
`TerrainKnowledge` instances.

### Step 3: Distributed Semantics Contract

No dormant rover behavior was rewritten. Instead, the intended distributed
model was documented and protected by characterization tests:

- Active agent decisions use agent-local knowledge.
- Mission-global terrain is telemetry and combined UI aggregation only.
- Sharing is the only way local knowledge transfers between agents.
- Rover motion remains disabled.
- Existing rover target and weighted-route code is provisional and must use
  rover-local received knowledge before rover motion is enabled.

Important tests:

- `test_frontier_rebuild_ignores_mission_terrain_telemetry`
- `test_telemetry_fusion_does_not_mutate_agent_local_knowledge`
- terrain sharing characterization tests
- lifecycle assertion that `rover_motion_enabled` is false

## Remaining Checklist

### Step 4: Simplify TerrainSharingService

Original intent:

- Let `TerrainSharingService` decide whether and when agents share.
- Let `TerrainKnowledge` perform the actual terrain merge.
- Add synchronization around pair cooldown state.

Current problem:

- Drone worker threads can call
  `TerrainSharingService.share_with_nearby_drones()` concurrently.
- Pair cooldown data is stored in `MissionControl.last_pair_share`.
- Cooldown check and update are not protected as one atomic operation.
- Per-drone sharing timing is split between `Drone` and the service.
- Rover timing is already service-owned.

Recommended scope for Step 4:

1. Move sharing schedule state into `TerrainSharingService`.
2. Add a service-owned lock around cooldown check/reservation/update.
3. Ensure only one worker can process a drone pair during a cooldown window.
4. Keep proximity, line-of-sight, comparison thresholds, and merge behavior
   unchanged.
5. Continue using `TerrainKnowledge.snapshot()` and `merge_from()` for terrain
   data transfer.
6. Add deterministic tests for pair state ownership and concurrent duplicate
   suppression without using timing sleeps.
7. Update `CODEFLOW.md` and `TESTING.md`.

Avoid broad algorithm or performance changes during this step.

### Step 5: Consolidate Presentation State

- Make `PresentationAdapter` solely responsible for heatmap selection and
  path/vision visibility.
- Remove duplicated implementations from `MissionControl` and
  `MissionControlTerrainMixin`.

### Step 6: Reduce Compatibility Scaffolding

- After call sites and tests use the new boundaries, remove unused `Drone`
  wrappers and redundant terrain-facade methods.
- Keep only intentional public agent-facing APIs.

### Step 7: Defer UI Restructuring

`ControlCenter.py` and `Menu.py` remain large, but they are separate concerns
and are not currently blocking SLAM, terrain, or mission-policy work.

## Decisions and Deferred Work

### Rover Behavior

Rovers are created and rendered, but rover worker threads are disabled through:

```python
self.rover_motion_enabled = False
```

Do not enable the provisional rover target or route policy unchanged. Rover
decisions must eventually use the rover's received local knowledge.

### Performance

Observed live performance is around 3 to 5 FPS on the user's machine.

- Sensing is the dominant stage, roughly twice rendering time.
- The main loop still targets 15 FPS.
- SLAM surface rebuilding is throttled to 0.1 seconds.
- Rover sharing is throttled to 0.5 seconds.
- Frame-stage timings appear in the control-center debug panel.

The user explicitly chose to defer deeper optimization until exploration,
battery management, and mission behavior are more complete. Do not cap sensing
frequency as a performance shortcut.

### Mission Controls

The mission has compact symbol controls:

- STOP: red button with a hollow white square around a filled white square.
- RESTART: light ochre yellow button with a clean clockwise circular arrow.
- PAUSE/PLAY: blue toggle that freezes or resumes agent threads, mission
  updates, and elapsed mission time.

Restart performs normal cleanup, reuses the same settings and generated cave,
then creates a fresh `MissionControl` and fresh agent state.

### SLAM Encapsulation

Encapsulating `SlamMap` arrays and locks behind snapshots is a possible future
cleanup, but it is not part of the current numbered checklist and must not be
confused with Step 3 or Step 4.

## Architectural Contracts

- Pygame event handling and drawing stay on the main thread.
- Drone movement and drone-to-drone sharing run on worker threads.
- `MissionControl` retains orchestration and lifecycle ownership.
- `PathfindingService` owns shared memory, the process pool, and its semaphore.
- `TerrainKnowledge` owns terrain data synchronization and merging.
- Mission terrain telemetry does not distribute knowledge to agents.
- Explicit sharing is the agent-to-agent transfer boundary.
- Existing user changes in the dirty worktree must be preserved.

## Suggested New-Chat Prompt

Use this prompt in the next chat:

```text
Please read docs/SESSION_HANDOFF_REFACTOR_STEP4.md, CODEFLOW.md, and TESTING.md.
Continue the existing uncommitted refactor without reverting any worktree
changes. Start with Step 4: simplify TerrainSharingService by making it own and
synchronize sharing cooldown state while preserving current sharing behavior.
Run focused tests and the full suite when complete.
```
