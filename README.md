# Cave Game

Cave Game is a distributed-systems simulation built with Pygame. It models a team of autonomous drones and rovers exploring a procedurally generated cave while coordinating through limited communication, local perception, and shared mission state. The repository is intentionally structured to show how concurrency, pathfinding, terrain generation, and UI composition work together in a small but non-trivial simulation.

## Overview

The codebase is organized around a simple control chain: `main.py` creates the game shell, `game.py` manages menus and mission startup, and `mission/control.py` coordinates the live simulation. From there, multiple subsystems work together:

- `generation/map_generator.py` builds the cave using multiprocessing and shared memory.
- `agents/drone.py` models local exploration, vision, and terrain knowledge.
- `agents/rover.py` acts as a mobile or stationary support agent depending on mission setup.
- `mapping/terrain_knowledge.py` owns terrain arrays, synchronization, snapshots, observation fusion, and merging.
- `navigation/waypoint_graph.py` owns the shared strategic topology, stable
  node/edge/route IDs, exact edge polylines, and revision-keyed route cache.
- `navigation/frontier_clusters.py` owns stable frontier identity,
  reservations, and belief-validated gateways.
- `mission/exploration_coordination.py` reconciles canonical retirements into
  every drone, confirms team-wide exhaustion, and starts coordinated homing.
- `mapping/wall_mapping.py` measures exposed cave/pillar/internal-wall surface
  coverage from combined occupied SLAM evidence.
- `navigation/navigation_intent.py` and `agents/local_mcts_controller.py` own
  persistent route execution state and bounded goal-conditioned local control.
- `navigation/pathfinding.py` and `navigation/astar_pathfinder.py` remain
  physical-simulator and disabled-rover infrastructure; drone planning does
  not call their ground-truth cave-map A*.
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

`agents/drone.py` composes each drone's local mapping, strategic policy,
movement state machine, sensor, and renderer. Frontier extraction produces
stable cluster IDs from that drone's SLAM belief. A shared assignment registry
reserves a cluster, while each drone selects a requester-known-free waypoint
from that stable cluster's cells. A gateway is protected only when a required
belief corridor actually connects the cluster to the strategic graph. The
selected goal and exact oriented route are latched in a stable-ID navigation
intent. Normal movement advances the stored polyline cursor by one bounded
prefix with no per-prefix A* or goal reselection; only route construction may
run bounded known-free connector A*.

`agents/rover.py` serves as a support agent. In the current architecture it is primarily valuable as a rendezvous and accumulation point for terrain knowledge, which makes it useful for centralizing observations without replacing the distributed model.

### Support Systems

`navigation/waypoint_graph.py` provides the thread-safe strategic highway. Its
monotonic IDs survive role changes, its travelled and requester-scoped
belief-corridor edges retain complete physical polylines, and reverse route
trees are cached by topology, requester, and requester-scoped belief-edge
validity. Unrelated SLAM revisions can therefore reuse a valid tree.
`navigation/pathfinding.py` still owns a cave-map-backed worker pool used at the
physical simulator/standalone algorithm boundary, and terrain-weighted routing
for the currently disabled rover flow. It is not injected into drone movement.
`agents/graph.py` remains the physical collision/history boundary, while
`ui/control_center/facade.py` and `mission/presentation_adapter.py` keep UI
concerns outside the simulation core.

## Runtime and Data Flow

The simulation works as a feedback loop:

1. The map is generated.
2. Agents are placed into the world.
3. Drones move and scan every visible cell in their current cone.
4. Dense visibility updates local SLAM, while sparse sample rays update the
   separate rover-oriented terrain knowledge.
5. Nearby agents exchange data when the mission rules allow it.
6. The UI reports exposed wall-surface coverage from combined occupied SLAM;
   roughness remains an optional terrain heatmap.
7. Belief-only frontier extraction prioritizes continuation of already
   observed wall surfaces. Coherent open unknown regions are used only to find
   another wall; the coherence threshold remains a defensive filter rather
   than compensating for gaps in the visibility cone. A retained wall target
   must offer a meaningfully displaced next viewpoint; unchanged pixel-scale
   tips are locally suppressed, and valid continuations use a directed partial
   sweep instead of another full rotation.
8. Canonical frontier retirement is reconciled across the team. All drones
   home together only after every drone confirms the shared registry is empty.
9. Mission completion requires every drone to finish that coordinated return.

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
monotonic version. Sensing updates it through methods; sharing, frontier
selection, and rendering consume detached `SlamSnapshot` values. Rendering
tracks consumed versions so an update arriving during frame composition
remains pending for the next refresh.

Vision and terrain sampling deliberately have different resolutions.
`VisionSensor.scan_cone()` produces a collision-bounded `VisionScan` covering
every grid cell in the cone; those free and occupied observations are the only
sensor inputs to SLAM exploration gain and mission completion. The fixed ray
set remains available for the overlay and samples roughness every second ray
cell. Terrain may therefore remain interpolably sparse without creating
frontiers or delaying wall-mapping completion.

The distributed-semantics contract is:

- Agent-local knowledge drives active agent decisions.
- Mission-global terrain is telemetry and UI aggregation only.
- Exploration progress is occupied-SLAM coverage of exposed wall surfaces;
  buried solid rock is excluded.
- Exploration exhaustion is belief-only and team-coordinated through the
  canonical frontier registry; the cave map is not exposed to planners.
- Sharing is the only mechanism that transfers local knowledge between agents.
- Rover movement remains disabled. Its existing target and route code is
  disabled until it uses rover-local received knowledge before activation.

### Proximity-Based Sharing

The sharing model is intentionally limited.

- Drone-to-drone exchange is triggered by proximity and throttled so the system does not spend all of its time copying arrays.
- Rover sharing acts as a support mechanism for collecting knowledge near a persistent reference point.
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

Drone movement separates strategic selection from bounded local execution.

The agent typically does the following:

- scan nearby terrain and update private SLAM/terrain knowledge,
- refresh stable frontier clusters, prefer an unassigned wall-continuation
  cluster, and use coherent unknown space only as a wall-discovery fallback,
- select an informative locally known-free cell within the stable cluster,
  batch retained wall progress by at least the configured continuation
  distance, and construct a cached strategic route,
- persist a gateway only when a required bounded belief corridor was actually
  connected; retire its orphan corridor with the cluster,
- persist cluster, gateway, assignment, route, and intent IDs plus exact route
  cursors across movement ticks,
- follow the stored oriented edge polyline deterministically during normal
  travel, and use bounded goal-conditioned local MCTS only when local
  deviations provide a real choice; single-choice scan and recovery modes use
  deterministic stored-intent fast paths,
- invalidate or change goals only for explicit reservation, topology, belief,
  collision, scan, or watchdog reasons.
- reconcile peer-retired IDs before selection and begin homing only after all
  drones have reported a frontier-free team epoch.

Planner-facing types contain SLAM belief and stable IDs, never the true cave
map. The true map is still used where the simulator must enforce physical
collision, generate sensor observations, test communication line of sight, or
exercise the disabled rover/pathfinding scaffolding.

## Rendering and UI

The rendering path is deliberately separated from mission logic.

`rendering/mission_renderer.py` owns complete frame composition and the
stop-button visual. `rendering/slam_view.py` builds the selected SLAM or terrain
view. `rendering/waypoint_renderer.py` rebuilds one persistent overlay per
committed topology revision and renders HOME, JUNCTION, CHOKEPOINT, TURN,
FRONTIER_GATEWAY, and RECOVERY_ANCHOR roles distinctly. Each agent renderer
owns paths, vision, and icons. The control-center facade builds immutable frame
data, its controller owns timer/tab/input state, and its renderer owns all
Pygame resources and hit geometry. `mission/presentation_adapter.py` keeps
presentation state isolated so toggles do not contaminate the simulation
model.

Rendering is layered so the visual output stays readable:

1. Black canvas and SLAM or terrain surface
2. Waypoint highway edges and nodes
3. Agent paths
4. Drone vision
5. Agent icons
6. Control center and stop button

Runtime code delegates frame composition directly to `MissionRenderer.draw()`.

## Configuration and Assets

The project keeps its runtime settings and visual assets in predictable locations.

- `GameConfig/options.default.ini` and `GameConfig/simulation.default.ini`
  store committed defaults. Strategic navigation settings use a 32 px spatial
  hash, an 8 px merge radius, 64/192 px local/gateway connector bounds, a
  64-entry route cache, a four-cell local unknown-coherence threshold, a 12 px
  retained-wall displacement, a three-heading continuation sweep,
  trail-promotion thresholds, and the bounded local MCTS budget. Retired
  breadcrumb/frontier-fallback settings are neither typed nor persisted.
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
| Distributed terrain sharing | Implemented | Drone-to-drone and rover terrain sharing are active |
| POI and path sharing | Planned | POI model exists; runtime integration is deferred |
| Strategic navigation graph | Implemented | Stable IDs, exact stored edge polylines, belief-scoped connectors, persistent cursor execution, and a revision-batched role overlay are active |
| Battery management | Planned | Not yet implemented |
| Stable frontier exploration | Implemented | Stable clusters, reservations, persistent goals, deterministic route following, and bounded local MCTS are active |
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
