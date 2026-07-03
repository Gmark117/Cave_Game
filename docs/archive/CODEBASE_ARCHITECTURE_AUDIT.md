# Codebase Architecture Audit and Refactoring Handoff

Date: June 21, 2026

## Purpose

This document records a repository-wide assessment of the Cave Game codebase
against three goals:

1. Clear structure.
2. Readable and understandable code.
3. Properly compartmentalized responsibilities and state.

The purpose of the proposed cleanup is to make the codebase easier for the
project owner to understand, review, and steer. It is not an authorization to
remove or redesign behavior merely because it looks incomplete.

The repository currently contains a large uncommitted refactor. Preserve all
existing worktree changes and do not reset, revert, or discard them.

## Mandatory Check-In Before Every Step

Every future refactoring step requires explicit approval from the project owner
before implementation begins.

Before editing code for a step, the assistant must:

1. State the exact scope and files expected to change.
2. Explain the structural problem being addressed.
3. Identify behavior that must remain unchanged.
4. List any code that appears incorrect, obsolete, unused, duplicated, or
   unfinished.
5. Ask whether any of those items represent intentional or unfinished
   features.
6. Present the proposed implementation and important alternatives.
7. Wait for explicit approval before making changes.

The assistant must not classify code as dead, wrong, accidental, or safe to
remove solely because it is unused, incomplete, inconsistent, or not covered
by the current runtime flow. It may be scaffolding for unfinished work.

If new questionable code is discovered after a step has begun, pause before
changing or removing it and perform another check-in. Previously approved,
safe, atomic work may be completed first when necessary to leave the worktree
in a coherent state.

Approval applies only to the stated step. It does not automatically authorize
later steps in the roadmap.

## Overall Assessment

The codebase is in improving condition and does not need a rewrite. Recent
work established strong ownership patterns for SLAM, terrain knowledge,
sharing schedules, pausing, pathfinding resources, mission lifecycle, and
rendering composition.

The weakest areas are:

- UI state and rendering boundaries.
- Cross-thread agent-state access.
- Configuration ownership.
- Service dependencies on the complete `MissionControl` object.
- Map-generation responsibilities.
- Inconsistent project organization and obsolete documentation.

The automated suite currently contains 127 passing tests and provides a useful
safety net for incremental refactoring.

## Healthy Architectural Areas

### SLAM ownership

`SlamMap.py` privately owns its occupancy and confidence arrays, point cloud,
synchronization, merge rules, and version. Callers consume detached
`SlamSnapshot` values.

### Terrain ownership

`mapping/terrain_knowledge.py` centralizes terrain data, synchronization,
observation fusion, snapshots, merging, and explored-ratio calculations.

### Pathfinding resources

`navigation/pathfinding.py` owns the process pool, semaphore, shared-memory
allocation, and cleanup used by mission path requests.

### Pause coordination

`mission/pause_control.py` cleanly separates pause-aware simulation time from
the worker barrier.

### Tests and documentation

`TESTING.md` describes test placement and architectural contracts, while the
test suite covers most important service and concurrency boundaries.

## Findings and Proposed Solutions

### 1. Control-center data is not a reliable mission view

`ControlCenter.py` calculates rover count with:

```python
self.num_rovers = 1 + (4 % num_drones)
```

The actual factory uses:

```python
math.ceil(control.settings.num_drones / 4)
```

For three drones, the control center reports two rovers while the mission
constructs one.

The control center also initializes hard-coded drone names, battery values,
and statuses. Rover records are refreshed from live objects, but drone records
are not updated in the same way.

Proposed solution:

- Build immutable `DroneStatusView` and `RoverStatusView` values from actual
  runtime agents.
- Pass those values into the control center each frame.
- Remove agent-count formulas and placeholder runtime status from the UI.
- Preserve display names only if the project owner confirms that the named
  agents are an intentional feature.

### 2. The ControlCenter and renderer split is incomplete

`ControlCenter.py` still owns layout, rendering caches, drawing helpers, timer
state, input handling, and hit rectangles. `ControlCenterRenderer.py` reaches
into private control-center state and calls control-center drawing methods.
The two objects therefore depend on each other in both directions.

Proposed solution:

- `ControlCenterController`: timer, selected tab, and action handling.
- `ControlCenterViewModel`: detached values to display.
- `ControlCenterRenderer`: all Pygame resources, layout, drawing, and caching.
- `ControlHitMap`: hit rectangles produced by layout/rendering and consumed by
  the controller.

### 3. Cross-thread drone state is not consistently synchronized

Movement writes position, graph history, frontiers, and related state while
rendering and sharing read those values on other threads. Some reads use
`exploration_lock`, but the corresponding writes do not consistently acquire
that lock. The lock therefore does not define a complete ownership boundary.

Proposed solution:

- Introduce one synchronized `DroneRuntimeState` owner.
- Return a detached `DroneSnapshot` for rendering, sharing, and debug output.
- Keep behavior controllers responsible for decisions while the state object
  owns mutation and synchronization.
- Establish and document which thread may mutate each category of agent
  state.

This should be resolved before relying on broader UI snapshotting.

### 4. Extracted services still depend on a broad control object

Many services accept `control: Any` and retrieve numerous fields from
`MissionControl`. Their actual dependencies are therefore implicit, and type
checking cannot protect service boundaries.

Examples include terrain sharing, fusion, rover targets, SLAM views, debug
information, movement, sensing, and rendering.

Proposed solution:

- Keep `MissionControl` as the composition root.
- Inject small dependency objects or typed protocols into services.
- Candidate boundaries include `SharingConfig`, `MissionAgents`,
  `SimulationClock`, `TerrainTelemetry`, `PathPlanner`, and
  `PresentationInvalidator`.
- Remove defensive `hasattr()` and `getattr()` fallbacks only after tests and
  explicit interfaces make the dependency mandatory.

### 5. Configuration has several competing owners

`SimSettings.py` owns some settings. `MissionControl.py` owns sharing
thresholds. `Menu.py` mirrors each setting into separate attributes and
manually reads and writes the INI representation.

`TerrainSharingService` also reads `drone_share_interval` and
`pair_share_cooldown`, although those fields are not declared by
`SimSettings`.

Proposed solution:

```text
SimulationConfig
├── MissionConfig
├── SlamConfig
├── SharingConfig
├── FrontierConfig
└── RenderingConfig
```

- Make configuration typed and validated.
- Add a `SettingsRepository` for INI serialization.
- Let the menu edit one configuration model instead of mirroring fields.
- Preserve existing INI compatibility unless a migration is explicitly
  approved.

### 6. Menu owns too many unrelated concerns

`Menu.py` contains menu-item modeling, Pygame rendering, input polling,
navigation, audio control, configuration persistence, simulation-settings
conversion, loading screens, and mission startup.

`MenuItem` also represents several different item types through one loosely
typed constructor with many optional parameters.

Proposed solution:

- Typed menu-item variants or dataclasses.
- `MenuController` for navigation and state transitions.
- `MenuRenderer` for drawing.
- `SettingsRepository` for persistence.
- `AudioService` for mixer operations.
- Named navigation methods instead of `lambda` expressions containing
  `setattr()` calls.

### 7. Map generation combines several responsibilities

`MapGenerator.py` performs generation during construction and also manages
progress UI, multiprocessing, post-processing, roughness generation, image
extraction, and file output.

`MapGenHelpers.py` combines brush algorithms, process workers, event pumping,
resource cleanup, progress callbacks, and diagnostics. Broad exception
fallbacks make failures difficult to classify.

Proposed solution:

- `CaveGenerator`: generation API returning a domain result.
- `WormProcessRunner`: multiprocessing and shared memory.
- `CavePostProcessor`: OpenCV cleanup.
- `TerrainRoughnessGenerator`: terrain synthesis.
- `MapArtifactWriter`: optional image and matrix persistence.
- Progress callbacks supplied by the application/UI layer.

`set_ends()` and `connect_rooms()` currently appear disconnected from the
active generation path. Before removing or changing them, perform the
mandatory check-in to establish whether they are unfinished generation
features.

### 8. A* implementations duplicate the search algorithm

`AStarPathfinder.py` contains separate ordinary and weighted A* functions that
repeat most of the same search loop. This makes movement and corner rules easy
to change in one planner but not the other.

The shared-memory worker attaches to a memory block but does not explicitly
close that attachment.

Proposed solution:

- One internal A* implementation with a configurable cell-cost policy.
- Separate public adapters for ordinary drone routing and terrain-weighted
  rover routing.
- Explicitly close worker-side shared-memory attachments with `try/finally`.
- Preserve path selection and corner-cutting behavior through characterization
  tests before consolidation.

### 9. Sensor timing is unclear

`DroneSensorController.update()` casts rays and updates SLAM every main-loop
frame. `slam_scan_interval` throttles only terrain sampling later in the
operation.

Current naming and documentation imply that the entire scan is throttled.

Proposed solution:

- Decide whether the intended setting controls all sensing or terrain sampling
  only.
- If it controls only terrain, rename it to `terrain_scan_interval`.
- If it controls the full sensor update, move the timing gate before ray
  casting.

This is a behavioral decision and requires a mandatory check-in before code is
changed.

### 10. Some state and features appear unused or incomplete

Examples include:

- `PresentationAdapter.terrain_heatmap_surf`.
- `Drone.statuses`.
- `MissionControl.mission`, which currently does not select mission policy.
- `POI`, which is documented and tested but not integrated into runtime flow.
- Map-generation constants and brush variants that do not match active code.

These items must not be removed automatically. They may represent planned
features. Each one requires confirmation from the project owner before
cleanup.

### 11. Documentation contains historical architecture

Some handoffs and `docs/IMPLEMENTATION_CHECKLIST.md` refer to the removed
`MissionControlTerrain` design or planned features that are not yet active.
README statements also occasionally describe intended behavior as if it were
fully integrated.

Proposed solution:

- Keep current architecture in `README.md`, `CODEFLOW.md`, and `TESTING.md`.
- Move superseded handoffs and historical plans into `docs/archive/`.
- Add explicit labels such as `implemented`, `provisional`, `planned`, and
  `deferred`.
- Do not archive a plan that still guides unfinished work without owner
  confirmation.

### 12. Repository organization is transitional

Root-level PascalCase modules coexist with lower-case packages. New package
modules still import older root modules, leaving the project between two
organizational styles.

There is also no declared dependency manifest, standard project configuration,
formatter/linter configuration, or active continuous-integration workflow.

A possible long-term layout is:

```text
app/             game shell, mission composition and lifecycle
domain/          agents, SLAM, terrain models and settings
services/        sensing, sharing, pathfinding and generation
ui/              menus, control center, presentation and renderers
infrastructure/  Pygame, files, multiprocessing and assets
```

Do not begin with broad file moves. Stabilize ownership boundaries first, then
move modules in small, approved steps.

## Recommended Refactoring Order

Every numbered step below is individually gated by the mandatory check-in.

### Step 1: Establish truthful control-center data

- Add characterization tests for actual rover counts and live drone status.
- Introduce detached agent-status view models.
- Remove duplicate UI calculations only after confirming intended display
  names and placeholder behavior.

### Step 2: Establish synchronized agent snapshots

- Define the state that crosses thread boundaries.
- Introduce synchronized mutation and detached snapshots.
- Move rendering, sharing, and debug reads to those snapshots.

### Step 3: Complete the ControlCenter separation

- Separate controller state, view models, renderer resources, and hit maps.
- Preserve all current presentation action tokens and visible behavior.

### Step 4: Restructure Menu

- Separate menu state, rendering, persistence, and audio.
- Replace loosely typed menu-item configuration with typed definitions.

### Step 5: Consolidate configuration ownership

- Introduce typed nested configuration.
- Preserve existing INI files through an explicit repository/migration layer.

### Step 6: Tighten service dependencies

- Replace broad `Any` control dependencies with focused protocols and injected
  collaborators.
- Keep `MissionControl` as the composition root.

### Step 7: Split map generation

- Separate pure generation, process management, post-processing, output, and
  UI progress.
- Resolve unfinished generation methods through owner check-in.

### Step 8: Consolidate pathfinding

- Share one A* engine between drone and rover policies.
- Add explicit worker shared-memory cleanup.

### Step 9: Remove confirmed dead code and reconcile documentation

- Review every apparently unused or incomplete feature with the owner.
- Remove only confirmed dead code.
- Archive only confirmed historical documents.

### Step 10: Normalize packages and project tooling

- Move modules incrementally into a consistent package layout.
- Add dependency and development-tool configuration.
- Add automated test execution if desired.

## Verification Baseline

At the time of this audit:

```text
python -m unittest discover -s tests -q
Ran 127 tests
OK
```

Additional checks:

```text
python -m compileall -q .
git diff --check
```

Both passed. `git diff --check` reported existing line-ending conversion
warnings but no whitespace errors.

No interactive mission smoke run was performed during the audit.

The audit itself made no source-code changes.

## Suggested Opening Prompt for the New Session

```text
Please read:

1. docs/CODEBASE_ARCHITECTURE_AUDIT.md
2. docs/SESSION_HANDOFF_REFACTOR_STEP4.md
3. CODEFLOW.md
4. TESTING.md

Continue from the existing uncommitted refactor without reverting, resetting,
or discarding any worktree changes.

Begin with Step 1 from the architecture audit: establish truthful
control-center data using characterization tests and detached agent-status view
models.

Mandatory workflow:
- Before every refactoring step, check in with me.
- State the exact scope and expected files.
- Explain the issue, proposed implementation, alternatives, and behavior that
  must remain unchanged.
- Identify anything that appears wrong, unused, obsolete, duplicated, or
  incomplete, and ask whether it is intentional or an unfinished feature.
- Do not edit code until I explicitly approve that step.
- If questionable code is discovered during implementation, pause and check in
  again before changing or removing it.
- Approval for one step does not authorize later steps.

For the first step:
- Add tests exposing the real control-center rover count and live drone status.
- Confirm with me whether the named agents and placeholder records are intended
  future behavior before removing them.
- Introduce detached status view models.
- Preserve current presentation actions and visible behavior.
- Do not restructure Menu or broadly rename/move packages yet.

After approved changes, run focused tests, the full suite,
`python -m compileall -q .`, and `git diff --check`.
```
