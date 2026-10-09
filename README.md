# Cave Game

Cave Game is a distributed-systems simulation built with Pygame. It models a team of autonomous drones and rovers exploring a procedurally generated cave while coordinating through limited communication, local perception, and shared mission state. The repository is intentionally structured to show how concurrency, pathfinding, terrain generation, and UI composition work together in a small but non-trivial simulation.

## Overview

The codebase is organized around a simple control chain: `main.py` creates the game shell, `game.py` manages menus and mission startup, and `mission/control.py` coordinates the live simulation. From there, multiple subsystems work together:

- `generation/map_generator.py` builds the cave using multiprocessing and shared memory.
- `agents/drone.py` models local exploration, vision, and terrain knowledge.
- `agents/rover.py` acts as a mobile frontier-staging and rendezvous agent.
- `mapping/terrain_knowledge.py` owns terrain arrays, synchronization, snapshots, observation fusion, and merging.
- `mapping/wall_mapping.py` measures exposed cave/pillar/internal-wall surface
  coverage from combined occupied SLAM evidence.
- `mission/exploration_coordination.py` owns discovery rounds, task claims,
  reachability assignment, and mission quiescence.
- `mapping/frontier_registry.py` owns significant frontier components, stable
  lineage, and bounded wall-follow/sweep work units.
- `agents/component_explorer.py` owns each drone's local component DFS stack.
- `navigation/pathfinding.py` and `navigation/astar_pathfinder.py` provide A*
  for drone transit/homing and rover-local terrain-aware staging.
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

`agents/drone.py` composes each drone's local mapping, movement controller,
sensor, and renderer. The rover first distributes a full-circle scan across the
team. It merges the returned observations, extracts significant eight-connected
frontier components, and assigns bundled component anchors or wall sub-arcs.
One known-free connected-component pass establishes rover-side eligibility;
the assigned drone computes and validates the exact A* route on its own worker.
Claims prevent duplicate work but never own map territory: task and return
transit may cross any mapped traversable space.

At a claimed anchor the drone waits for the ordinary sensor scheduler's exact
scan completion, discovers causally related local successors, and explores them
depth-first. Completed frames unwind logically; the drone uses A* directly to
the next sibling rather than physically revisiting each ancestor. Recorded
movement remains a breadcrumb fallback to the last visited parent if direct
A* fails, while rover return is also A*-first with breadcrumb fallback. A
failed logical reposition suspends the claim instead of retrying indefinitely.
The rover
reconciles one-to-one continuation, split, merge, resegmentation, dormancy, and
resolution into explicit component lineage. Wide wall-connected frontiers are
divided into wall sub-arcs; wide open frontiers use sensor-footprint sweep
anchors. A zero-gain scan retires only the visited anchor or sub-arc.

If the rover scan exposes too little work, idle drones receive bounded,
separated radial probes. Each probe performs a full-circle scan and physically
returns to upload before another ring can be scheduled. Probing is a bootstrap
operation only: as soon as significant component work exists, the coordinator
enters its monotonic component-exploration phase and never starts an endgame
probe. If bootstrap exposes fewer components than drones, spare drones follow
assigned leaders and take deterministic reserved branches when a component
first separates. Tiny gateways into large enclosed unknown areas retain the
existing significance eligibility. Homing starts only after every drone has
physically checked in and all component work and claims are exhausted (or
bootstrap discovery itself is exhausted).
Every physical point is appended to the path history consumed by the renderer.

`agents/rover.py` is the single mobile team checkpoint and runs on its own
worker. It stages near claimed work using only its received map: candidate and
route cells must clear the rover body, while service distance, footprint
asperity, and wall clearance rank the safe choices. Before moving, the rover
announces an immutable endpoint. It cannot depart until every drone has
acknowledged that endpoint and the complete acknowledgement set has returned
to the rover through direct or relayed LOS/proximity contacts. An announcement
does not immediately replace a drone's rendezvous target, even after universal
acknowledgement. Physical contact or relay of an actual departure or newer
confirmed stop advances that target immediately, including during a return
flight. A proposal alone requires finding the remembered stop physically empty
before falling forward. A queued check-in is
not an empty endpoint; the drone waits for the rover worker to accept its
report. This keeps an acknowledgement carrier from waiting at a destination
the rover cannot yet legally depart toward.

### Support Systems

`navigation/pathfinding.py` owns a cave-map-backed worker pool for drone escape
and homing routes. A search that reaches its fixed work cap returns a tagged
progress segment; the drone follows it and replans toward the unchanged goal.
Rovers plan terrain-weighted paths only through confidently free cells in their
own received SLAM.
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
5. Nearby drones exchange data while exploring and immediately invalidate
   stale local targets when received SLAM changes them. Discovery and component
   reports require a physical rendezvous with the moving primary rover.
   Each drone exchanges SLAM and terrain once on arrival, waits on its own
   assignment-ready signal, and exchanges once more immediately before a task
   or probe departure.
6. The UI reports discovered-floor coverage. Exposed wall coverage remains a
   trace diagnostic and is not a completion trigger; roughness remains an
   optional terrain heatmap.
7. At startup the rover partitions a full 360-degree scan across the drones.
   Returned observations feed the frontier registry, which preserves tiny and
   migrated gateway eligibility while assigning stable identities to significant
   eight-connected components. During bootstrap, when no component work is
   available, idle drones receive evenly spaced, known-free-connected radial
   probes with bounded outward rings.
8. The coordinator chooses a deterministic maximum-cardinality matching among
   targets in the rover's known-free connected region, preferring deeper
   lineage continuations and then the rover-to-target distance lower bound.
   Exact task and probe A* belongs to the assigned drone worker; an unreachable
   route is reported at the rover and deferred for that drone/component
   revision. Claims
   are token-fenced and attach to work units, not geography. A drone may cross
   any mapped traversable space, scans its claimed anchor, follows local
   component successors depth-first, and routes directly to the next logical
   target by A* rather than physically backtracking. The rover records split and merge lineage explicitly and
   reassigns released or battery-suspended claims with a new token.
   Unknown support contributes relative effort as pixels divided by the square
   of the global cell size. The initial runtime uses unlimited energy, but both
   assignment and executor checkpoints already go through the energy contract.
   Bootstrap discovery continues until it finds component work or its bounded
   probes are exhausted. Once component exploration begins, absence of open
   component work ends exploration without further probing.
9. Mission completion requires final physical rover check-in by every drone.
   The simulation then pauses on the completed control center until the user
   chooses stop, restart, or exit.

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
- Exploration progress is discovered-floor coverage. A 98-99% final value is
  valid when the significant frontier registry is exhausted.
- Exposed wall coverage is diagnostic only. The compatibility tolerance
  setting does not trigger homing or mission completion, and the ground-truth
  wall mask is never supplied to a drone's heading scorer.
- SLAM-derived border selection is local. The physical cave map is consulted
  by collision checks, sensor simulation, communication line of sight, and
  the deliberately simple A* escape/homing service.
- Sharing is the only mechanism that transfers local knowledge between agents;
  component directives carry claimed frontier geometry, not hidden occupancy
  data or a rover-calculated route.
- The rover moves on a dedicated worker using only received SLAM and terrain.
  Drones keep the newest proposal they learned through contact separate from
  their last confirmed rendezvous target; neither endpoint knowledge nor peer
  positions are broadcast out of range or through walls.

### Proximity-Based Sharing

The sharing model is intentionally limited.

- Drone-to-drone exchange is triggered by proximity and revision-gated, but is
  suppressed while either drone is at a rover.
- SLAM is shared, but derived frontier coordinates are not. A recipient
  invalidates and rebuilds its own borders before mission-exhaustion checks.
- Rover sharing is bidirectional for both terrain and SLAM. Periodic proximity
  exchange runs on the primary rover worker; component/probe rendezvous also
  guarantees an arrival exchange and one departure refresh.
- The first meaningful exchange in each continuous peer or rover encounter
  pauses translation for 0.75 simulation seconds. Further map updates and ACKs
  continue without renewing that pause; overlapping encounters share its
  deadline. Separation rearms the pause for the next encounter.
- A checked-in drone is mechanically docked and carried at the rover's logical
  position while it awaits a directive. Its movement, route planning, rotation,
  and sensors remain inactive, while physical rover contact can continue map
  and rendezvous-acknowledgement exchange.
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

Drone movement executes rover-issued discovery or component directives while
retaining local SLAM decisions and the existing sensor schedule.

The agent typically does the following:

- participate in a coordinated rover-centered full-circle scan;
- accept a token-fenced component claim after rover-known-free eligibility and
  the energy contract pass;
- compute and follow the exact A* route on the drone worker through unrestricted
  traversable space, or report the target unreachable without departing;
- hold the work-unit pose until the existing sensor controller publishes the
  exact requested scan, with the three-second simulation-time watchdog intact;
- classify significant successor frontiers from local SLAM and the completed
  scan footprint, then push them on a bounded DFS stack even when an immediately
  preceding scheduled scan made the directed scan's incremental gain zero;
- choose successor anchors by route reachability and cost, record every actual
  outbound path, and use a recorded suffix only if direct A* reposition fails;
- report sensor gain, the claimed unit's disposition, and causal successor
  lineage at the next physical rover check-in;
- suspend unfinished work through the energy interface when return reserve is
  required, returning its DFS frames and breadcrumb so a fresh claim can resume;
- perform a separated, rover-connected radial probe, validate its exact route
  locally, and run a full 360-degree scan only during bootstrap;
- follow an assigned leader when bootstrap has fewer components than drones,
  taking a reserved child only when the lineage actually splits; and
- finish at the rover after component quiescence or bootstrap discovery
  exhaustion, without an extra return to the historical launch pixel.

Border extraction consumes local SLAM only. The true map remains the physical
simulation boundary for collision checks, sensor observations, communication
line of sight, and A* escape/homing routes.

## Rendering and UI

The rendering path is deliberately separated from mission logic.

`rendering/mission_renderer.py` owns complete frame composition and the
stop-button visual. The exploration overlay draws ready, claimed, active, and
blocked component work cells plus their anchors; it does not tint or imply
territorial ownership.
`rendering/slam_view.py` builds the selected SLAM or terrain
view. Drone and rover rows both provide a local-map selector, so the rover's
received occupancy and terrain knowledge can be inspected independently of the
combined or per-drone view. Drone rows display the live directive phase, while
the debug tab adds directive, task, component, DFS depth, and target context;
rover rows expose navigation/hold state and their current target. Each agent
renderer owns paths, vision, and icons. With Highway in `observe` or `active`
mode, a cached overlay draws the rover's latest complete highway graph as
grey polylines at 85% opacity. The path-icon toggle beside the rover's
local-map selector controls this overlay and starts enabled; Highway itself
remains default-off. The control-center facade builds immutable frame
data, its controller owns timer/tab/input state, and its renderer owns all
Pygame resources and hit geometry. `mission/presentation_adapter.py` keeps
presentation state isolated so toggles do not contaminate the simulation
model.

The rover builds a branching corridor backbone from its received SLAM using
a medial-axis skeleton, retaining loops and narrow connections. Short terminal
spurs are pruned, then capillaries are added until every known free cell has
bounded access through free space. `[HIGHWAY]
maximum_access_distance_sensor_ranges = 2.0` controls that bound in drone LiDAR
ranges: smaller values produce a denser network. The old `macro_cell_size` key
remains readable but does not shape this backbone. Routes compare nearby
entries and exits and safely straighten bends; costs reflect the followed path.
Eligible active returns also check for a strictly shorter route through the
drone's current known free space. Cached paths are revalidated, and the bounded
comparison retains highway advice if no complete better route is found.
The primary rover worker submits each newly received local SLAM version to a
separate highway process and refreshes the yellow frontier registry immediately
from the same input. Full-map rebuilds run at most once every two wall-clock
seconds, with one job in flight and newer uploads coalesced. Each complete graph
replaces the previous graph; regional stitching has been removed. Unchanged
free geometry needs no rebuild. New free cells can follow in a later snapshot;
corrections invalidate affected in-flight builds. Assignments and drone highway delivery still require
physical check-ins. The legacy `minimum_version_delta` key remains readable;
refresh follows every new version. Large full-map builds use a conservative
reduced skeleton only when connected regions and holes survive. Large inputs
requiring full resolution use native thinning, with one preparation retained
for one retry of the identical map if another bounded attempt is needed.
New additions stay queued; corrections cancel an unsafe retry. All routes and access distances are
validated at original resolution. Build and query budgets retain the last
complete graph or use the existing fallback.
See [the frozen-map comparison](docs/HIGHWAY_BACKBONE_VALIDATION.md).

Drone-side Focused Frontier Batching route work also has a 250 ms decision
budget and 128-query cap. Drone-local A* queries use at most 50 ms and 4,096
expansions, with routes cached only for the current local SLAM version. An
incomplete cost cannot justify provisional work. Safe partial paths can advance
claimed transit, and incomplete tour optimization retains the rover's order.

Rendering is layered so the visual output stays readable:

1. Black canvas and SLAM or terrain surface
2. Component markers, highways, and agent paths
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

In a running mission, use the drone or rover tab's `T` button to select that
agent's own map. The global LIDAR button switches the selected map between
occupancy SLAM and terrain knowledge; selecting a rover or drone clears the
other agent-type selection.

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
| Frontier-component discovery and DFS | Implemented | Rover SLAM drives bounded scan/probe rounds and token-fenced component work |
| Rover docking | Implemented | Checked-in drones are carried with inactive movement and sensing until work is assigned or HOME completes them |
| POI and path sharing | Planned | POI model exists; runtime integration is deferred |
| Drone path rendering | Implemented | Each drone's complete travelled breadcrumb path is rendered incrementally |
| Battery management | Contract implemented | Unlimited runtime policy uses route-to-task, next-action, route-home, reserve, accept, return, and suspension hooks; drain/charging remain deferred |
| Route-based component transit | Implemented | A* routes are unrestricted by territory and actual breadcrumbs remain the return fallback |
| Rover highway network | Implemented, default-off | Rover-SLAM graph, physical publication, bounded return-route advice, and a grey overlay with a rover-row visibility toggle |
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
