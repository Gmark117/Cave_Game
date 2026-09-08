# Cave Game

Cave Game is a distributed-systems simulation built with Pygame. It models a team of autonomous drones and rovers exploring a procedurally generated cave while coordinating through limited communication, local perception, and shared mission state. The repository is intentionally structured to show how concurrency, pathfinding, terrain generation, and UI composition work together in a small but non-trivial simulation.

## Overview

The codebase is organized around a simple control chain: `main.py` creates the game shell, `game.py` manages menus and mission startup, and `mission/control.py` coordinates the live simulation. From there, multiple subsystems work together:

- `generation/map_generator.py` builds the cave using multiprocessing and shared memory.
- `agents/drone.py` models local exploration, vision, and terrain knowledge.
- `agents/rover.py` acts as a mobile or stationary support agent depending on mission setup.
- `mapping/terrain_knowledge.py` owns terrain arrays, synchronization, snapshots, observation fusion, and merging.
- `mapping/wall_mapping.py` measures exposed cave/pillar/internal-wall surface
  coverage from combined occupied SLAM evidence.
- `agents/exploration_policy.py` owns reproducible weighted-random selection.
- `navigation/pathfinding.py` and `navigation/astar_pathfinder.py` provide A*
  for cul-de-sac escape, homing, and the disabled rover flow.
- `agents/graph.py` tracks valid movement and exploration connectivity.
- `ui/control_center/facade.py` is the UI facade, `ui/control_center/controller.py` owns timer
  and input state, and `ui/control_center/renderer.py` owns Pygame layout and
  drawing.
- `mission/presentation_adapter.py` keeps UI state and presentation toggles isolated from mission logic.
- `asset_config/` contains the enums and constants that keep gameplay, rendering, media, and map generation consistent.

The design goal is realism through constraint: agents do not start with omniscient knowledge, terrain is discovered incrementally, sharing is event-driven, and the visualization is built from the same distributed data model the agents use.

## Getting Started

### Prerequisites

- Python 3.11 or newer
- `pip` available in your Python installation
- A desktop environment capable of running Pygame

### Quick Start

Install dependencies into your system Python:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the simulation:

```bash
python main.py
```

Run the automated test suite:

```bash
python -m unittest discover -s tests -v
```

See [`TESTING.md`](docs/TESTING.md) for the test matrix, placement rules, and manual
smoke checklist.

## System Architecture

The runtime flow is intentionally layered so that each file owns one part of the simulation lifecycle.

### Startup Path

1. `main.py` instantiates `Game` and starts the application.
2. `game.py` creates the UI window, handles menu navigation, and collects mission settings.
3. When the player starts a mission, `Game` prepares an immutable nested
   `SimulationConfig`, generates the cave, and constructs `MissionControl`.
4. `Game` calls `MissionControl.run()`, which creates agents and runtime resources, launches worker threads, and enters the main loop.
5. A restart request cleanly shuts down that controller and constructs a new
   one over the same settings and generated cave.

### Mission Orchestration

`MissionControl` is the central coordinator during play. Construction prepares mission state without starting the simulation; `run()` explicitly owns runtime initialization and teardown. The main thread handles window events, sensing, and frame updates, while per-agent threads handle movement and nearby data sharing.

The compact square-icon control ends the run and returns to the menu. The
circular-arrow control restarts through the same cleanup path, then `Game`
creates fresh agents, mapping state, timers, threads, and pathfinding resources
without regenerating the cave. The pause/play control freezes agent movement,
mission updates, and elapsed mission time while keeping rendering and input
responsive. PAUSE closes a worker barrier and returns only after every agent
thread reaches a safe checkpoint; PLAY releases the barrier. Behavioral
cooldowns use a pause-aware simulation clock, so wall time spent paused does
not change sensing, sharing, or frontier timing.

That separation matters because the simulation mixes three different execution models:

- The main Pygame loop handles input, timing, and rendering.
- Drone and rover behavior can run concurrently in worker threads.
- Cave generation uses multiprocessing so the map can be carved efficiently at startup.

The main loop targets 15 FPS. `FrameProfiler` records smoothed frame, wait,
sharing, sensing, rendering, and display durations, which are exposed in the
control-center debug panel. SLAM surfaces rebuild at most every 0.1 seconds,
and rover terrain exchange runs at most every 0.5 seconds; cached visuals are
still blitted every frame.

### Agent Responsibilities

`agents/drone.py` composes each drone's local mapping, weighted-random policy,
movement controller, sensor, and renderer. During ordinary exploration the
controller groups local unknown boundaries into connected components. It
weights collision-free headings toward sufficiently large wall-touching
components first, generic components second, and separation from nearby
teammates throughout. A cached 32-pixel coarse index periodically summarizes
frontiers inside the rover-assigned portion of the drone's SLAM map. When its selected region lies
beyond the local scoring window, that bearing becomes the strategic signal
and local geometry remains a lower-weight tactical correction. The previous
global region remains committed while it still overlaps current geometry and
is competitively scored. Tiny components do not bias normal movement but remain
available to bounded stagnation cleanup. A per-drone 32-pixel coverage memory
records coarse cell entries and undirected edge traversals. Its exponentially
decaying visit and edge pressure softly reduces ordinary heading weights toward
recently repeated ground without constraining sector ingress, rover check-in,
or homing routes. A confirmed zero-gain scan retires
the whole connected sampled component until its geometry changes. It then
walks a ten-pixel raster step. If no
radius-length heading is open, it extracts border cells from the drone's local
SLAM and uses A* to reach the nearest viable border. Reaching an
escape border suppresses that exact target until its local frontier geometry
changes, then performs a recovery-only full-circle turn toward a physically
usable heading. A distance-based sensor-gain window also redirects stagnant
random walks toward local unknown SLAM boundaries. Its scan-only heading may
face a wall even though translation in that direction is invalid; movement
waits for one completed scan, then restores the pre-scan travel heading when it
is safe or the closest safe alternative. It falls back to a frontier outside
the recent trail when A* is needed. Sector exhaustion triggers a physical
rover return using A* by default; the flown breadcrumb trail remains a safety
fallback when a route cannot be planned or completed.
Once a drone has entered its assignment, sector exhaustion remains valid even
if its final step drifted across a coarse ownership boundary. A boxed-in drone
that has not exhausted first uses A* to reach the nearest viable owned cell,
then a full-circle direct step; two complete ingress failures abandon the
assignment and begin rover check-in instead of spinning indefinitely.
Team wall completion uses A* homing independently of the sector cycle.
Every physical point is appended to the path history consumed by the renderer.

`agents/rover.py` serves as a stationary support agent and physical team
checkpoint. It accumulates both terrain and occupancy SLAM from arriving
drones, returns the rover's current team snapshot, and anchors synchronized
exploration-sector transitions.

### Support Systems

`navigation/pathfinding.py` owns a cave-map-backed worker pool for drone escape
and homing routes. A search that reaches its fixed work cap returns a tagged
progress segment; the drone follows it and replans toward the unchanged goal.
The service also retains terrain-weighted routing for the disabled rover flow.
`agents/graph.py` remains the physical collision/history boundary, while
`ui/control_center/facade.py` and `mission/presentation_adapter.py` keep UI
concerns outside the simulation core.

## Runtime and Data Flow

The simulation works as a feedback loop:

1. The map is generated.
2. Agents are placed into the world.
3. Drones move and scan every visible cell in their current cone.
4. Dense visibility updates every observed local-SLAM occupancy cell at
   uniform confidence, while sparse sample rays update the separate
   rover-oriented terrain knowledge with distance-weighted confidence.
5. Nearby agents exchange data when the mission rules allow it. A sector
   transition requires a physical rendezvous with the primary rover. Each
   drone exchanges SLAM and terrain once on arrival, stands by for an
   assignment-ready signal, and exchanges once more immediately before
   departure so it receives knowledge uploaded by later arrivals.
6. The UI reports exposed wall-surface coverage from combined occupied SLAM;
   roughness remains an optional terrain heatmap.
7. The rover assigns disjoint 32-pixel exploration sectors. Ordinary movement
   makes a seeded weighted-random choice inside the current vision cone, but
   local and cached whole-map frontier evidence is restricted to the assigned
   sector. Connected wall-frontier components outrank generic components.
   Wall candidates combine continuation, size rank, and proximity with 2:2:1
   weights; generic candidates combine size rank and proximity with 2:1
   weights. A coarse whole-map cache applies the same tiering every two seconds
   and steers toward remote regions without rescoring every frontier pixel on
   every step. Positional separation spreads nearby drones. Decaying coarse
   cell and edge memory reduces the weight of recently revisited directions
   while leaving mandatory transit paths available.
8. When its sector is exhausted, a drone uses A* to return to the rover, with
   its flown breadcrumb path retained as a safety fallback. Once the whole
   team has checked in, the rover signals each waiting drone rather than
   requiring repeated check-in polls, and a new contiguous partition is built
   from accumulated rover SLAM. Before partitioning, frontier
   components smaller than 12 pixels are discarded unless they open into at
   least 64 connected unknown cells in an interior basin. Unknown basins
   connected to the map boundary cannot rescue a small component. Only one
   small gateway is retained per eligible basin. Sector seeds and safe boundary
   transfers minimize estimated scan, approach, component-dispersion, and
   terrain effort while keeping territories connected. Each assignment carries
   stable frontier-component IDs. On arrival, the drone reports its local
   suppression outcomes with the existing check-in; an unchanged component is
   suppressed immediately when its own local region gained no confident cells
   or when a directed scan confirmed zero gain.
   If no significant novel frontier work remains, exploration ends.
   Coordinated homing can
   also start at exact team wall
   completion or when no more than the configured 30 wall pixels remain.
9. Mission completion requires every drone to finish its return.

That flow is important because the game does not use a single global terrain oracle. Instead, knowledge is built from observations and exchanged through explicit events. This makes the heatmap, the agent behavior, and the mission state all consistent with one another.

### Distributed Terrain Knowledge

Terrain state is represented by `TerrainKnowledge`. Mission control, every
drone, and every rover own separate instances containing roughness, confidence,
a floor mask, and synchronization. Drones update their local instance while
the same observations are separately recorded for rover routing and the
optional terrain heatmap. Terrain coverage does not define exploration
progress or completion.

Snapshots provide detached data for rendering and belief-scoped value
estimation. Sharing decides when knowledge moves between agents, while
`TerrainKnowledge.merge_from()` provides the single confidence-weighted merge
rule.

Local occupancy mapping follows the same ownership pattern. Each `SlamMap`
privately owns its occupancy grid, confidence grid, point cloud, lock, and
monotonic version. `TerrainKnowledge` likewise exposes a monotonic revision so
sharing can reject unchanged pairs before copying arrays. Sensing updates both
stores through methods; sharing, frontier selection, and rendering consume
detached snapshots. Rendering
tracks consumed versions so an update arriving during frame composition
remains pending for the next refresh.

Vision and terrain sampling deliberately have different resolutions.
`VisionSensor.scan_cone()` produces a collision-bounded `VisionScan` covering
every grid cell in the cone; those free and occupied observations are the only
sensor inputs to SLAM exploration gain and mission completion. The fixed ray
set remains available for the overlay and samples roughness every second ray
cell. Uniform occupancy and scan-local terrain samples are fused with bulk
array operations, and each `sensor_scan` trace event reports vision, SLAM,
terrain, and total elapsed time. Terrain may therefore remain interpolably
sparse without creating frontiers or delaying wall-mapping completion.

The distributed-semantics contract is:

- Agent-local knowledge drives active agent decisions.
- Mission-global terrain is telemetry and UI aggregation only.
- Exploration progress is occupied-SLAM coverage of exposed wall surfaces;
  buried solid rock is excluded.
- Exact team wall completion or the configured 30-pixel residual tolerance is
  a mission-level homing trigger. The tolerance is capped at one percent of
  the exposed wall total, and the UI still reserves 100% for exact coverage.
  The exposed ground-truth wall mask remains confined to mission evaluation
  and is never supplied to a drone's heading scorer.
- SLAM-derived border selection is local. The physical cave map is consulted
  by collision checks, sensor simulation, communication line of sight, and
  the deliberately simple A* escape/homing service.
- Sharing is the only mechanism that transfers local knowledge between agents;
  sector assignment itself carries no hidden occupancy data.
- Rover movement remains disabled. Its existing target and route code is
  disabled until it uses rover-local received knowledge before activation.

### Proximity-Based Sharing

The sharing model is intentionally limited.

- Drone-to-drone exchange is triggered by proximity and revision-gated, but is
  suppressed while either drone is at a rover.
- SLAM is shared, but derived frontier coordinates are not. A recipient
  invalidates and rebuilds its own borders before mission-exhaustion checks.
- Rover sharing is bidirectional for both terrain and SLAM and belongs to the
  sector rendezvous protocol: once on arrival and once before departure.
  Waiting for the team performs no map exchange.
- The heatmap refresh is also throttled so rendering stays responsive.

This makes the simulation feel distributed rather than centralized. Agents learn locally first, then synchronize when they actually meet.

## Cave Generation

The cave is produced by `generation/map_generator.py`, which uses parallel worker processes to erode an initially solid map into a navigable cave system. The generator uses shared memory so the workers can write into the same binary terrain buffer without copying the entire map between processes.

The generation pipeline is roughly:

1. Create a shared map buffer.
2. Spawn multiple erosion workers.
3. Let those workers carve and refine the terrain concurrently.
4. Apply post-processing to smooth or clean the result.
5. Build the final terrain data used by the simulation.

The reason for this architecture is practical: cave generation is naturally parallel, and shared-memory multiprocessing is a good fit when several workers need to modify the same large array.

## Exploration Behavior

Drone movement intentionally uses a small baseline policy.

The agent typically does the following:

- scan nearby terrain and update private SLAM/terrain knowledge,
- test integer headings inside the current 60-degree vision cone for a clear
  radius-length look-ahead and ten-pixel short step,
- group local unknown boundaries into eight-connected components and exclude
  components below the configured 12-cell normal-heading size threshold,
- periodically aggregate the assigned portion of local SLAM into 32-pixel coarse
  frontier regions, retain one remote strategic target for two seconds, and
  bias the current collision-safe cone toward its bearing,
- keep wall-touching components as the strict first tier and score them from
  continuation alignment, normalized size rank, and proximity at 2:2:1; when
  none are actionable, score generic components from size rank and proximity
  at 2:1,
- add a positional separation score that repels nearby teammates; a rover
  assignment filters out-of-sector local and global targets and strongly
  penalizes steps that leave an already-entered sector,
- multiply ordinary heading weights by a decaying coarse coverage factor based
  on prior cell entries and repeated undirected edge traversals; ingress,
  check-in, and homing ignore this preference,
- choose by the resulting weights with a per-drone seeded random generator,
- walk the chosen line directly and record every point for path rendering,
- after 120 travelled pixels, keep weighted-random movement when sensor-local
  gain is productive; otherwise rotate directly toward an unknown cell beside a
  reachable SLAM border and hold position for exactly one completed scan,
- retain refreshed work after sensor gain, or suppress unchanged local
  frontier components after a zero-gain directed scan, then restore a
  collision-safe movement heading; scan requests use the last fully published
  sensor completion as their sequence baseline, and a three-second
  simulation-time watchdog restores movement if no matching completion arrives,
- when no local border can be viewed, use A* to reach a border outside the
  recent breadcrumb trail,
- when no local heading is available, rebuild known-free/unknown boundaries
  from local SLAM and use A* to reach the nearest viable border,
- after escape A*, retire the reached border while its local geometry is
  unchanged, including across sector assignments, and rotate in place toward
  a usable outgoing heading; reject a complete frontier route when its routed
  distance exceeds the configured multiple of direct displacement,
- use A* by default for the mandatory rover check-in after sector exhaustion,
  and retrace the flown breadcrumb path only as its safety fallback,
- preserve the fact that an assignment was entered after a boundary crossing;
  if an out-of-sector drone is boxed in, route it to a nearby viable owned cell
  or abandon after two failed ingress recoveries and check in,
- use the A* service for homing after accepted team wall completion; capped
  searches return a useful partial route and replan from its endpoint instead
  of being reported as unreachable.

Border extraction consumes local SLAM only. The true map remains the physical
simulation boundary for collision checks, sensor observations, communication
line of sight, and A* escape/homing routes.

## Rendering and UI

The rendering path is deliberately separated from mission logic.

`rendering/mission_renderer.py` owns complete frame composition and the
stop-button visual. A cached sector layer tints each rover-assigned 32-pixel
territory with its drone color, draws only ownership boundaries, marks each
seed and the rover rendezvous, and dims drones waiting at the team barrier.
`rendering/slam_view.py` builds the selected SLAM or terrain
view. Each agent renderer owns paths, vision, and icons; the drone breadcrumb
path is the only navigation overlay. The control-center facade builds immutable frame
data, its controller owns timer/tab/input state, and its renderer owns all
Pygame resources and hit geometry. `mission/presentation_adapter.py` keeps
presentation state isolated so toggles do not contaminate the simulation
model.

Rendering is layered so the visual output stays readable:

1. Black canvas and SLAM or terrain surface
2. Agent paths
3. Drone vision
4. Agent icons
5. Control center and stop button

Runtime code delegates frame composition directly to `MissionRenderer.draw()`.

## Configuration and Assets

The project keeps its runtime settings and visual assets in predictable locations.

- `GameConfig/options.default.ini` and `GameConfig/simulation.default.ini`
  store committed defaults. Navigation exposes local-SLAM border confidence,
  sampling stride, rebuild cooldown, frontier component and unknown-support
  thresholds/proximity scale/score weights, coarse global cell size/refresh
  interval, maximum frontier-route circuity, the weighted `random` policy, its
  wall/unexplored/separation biases, its coarse coverage-memory cell size,
  decay, visit weight, and edge weight, and its
  stagnation distance/gain threshold. Older waypoint and MCTS INI keys are ignored when loading
  existing files.
- `GameConfig/options.local.ini` and `GameConfig/simulation.local.ini` store
  user changes and are ignored by Git.
- `Assets/` contains the audio, fonts, images, backgrounds, and map resources used by the game.
- `Assets/Map/` contains cave images generated at runtime and ignored by Git.
- `asset_config/` provides typed constants and enums so gameplay values, colors, asset paths, and map-generation parameters stay consistent across modules.

This is a deliberate structural choice. Hard-coding file names and magic numbers across the codebase would make the simulation harder to tune and more brittle to change.

## Controls

In menus:

- `Up` / `Down`: move selection
- `Left` / `Right`: change selector or slider values
- `Enter`: confirm, open a submenu, or start a mission
- Number keys `0-9`: edit the seed field
- `Backspace`: return from submenus, except on the seed field where it deletes digits

Simulation settings available in-game:

- Objective: `Exploration` or `Search and Rescue`
- Cave size: `Small`, `Medium`, `Large`
- Seed: custom numeric seed or the default for the selected cave size
- Drones: from 3 to 8

## Project Status

| Feature | Status | Notes |
|---|---|---|
| Terrain roughness map | Implemented | Available in current simulation flow |
| Known map visualization | Implemented | Available in current simulation flow |
| Distributed map sharing | Implemented | Drone pairs and physical rover rendezvous exchange terrain and SLAM |
| Dynamic exploration sectors | Implemented | Rover SLAM drives contiguous sector epochs after mandatory check-ins |
| POI and path sharing | Planned | POI model exists; runtime integration is deferred |
| Drone path rendering | Implemented | Each drone's complete travelled breadcrumb path is rendered incrementally |
| Battery management | Planned | Not yet implemented |
| Random exploration with A* escape | Implemented | Random local steps are direct; A* is reserved for cul-de-sac borders and homing |
| Search & Rescue mission logic | Planned | Objective exists in UI; starting it fails fast instead of running Exploration behavior |
| Drift modeling | Planned | Not yet implemented |

## Troubleshooting

- If dependencies fail to install, upgrade `pip` first and retry.
- If `pygame` audio initialization fails, check that your system audio device is available and not locked by another app.
- If `cv2` import fails, reinstall OpenCV:

```bash
python -m pip install --force-reinstall opencv-python
```

## Notes

- This repository is under active development.
- Some modules contain extension points for future mission logic.
