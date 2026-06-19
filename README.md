# Cave Game

Cave Game is a distributed-systems simulation built with Pygame. It models a team of autonomous drones and rovers exploring a procedurally generated cave while coordinating through limited communication, local perception, and shared mission state. The repository is intentionally structured to show how concurrency, pathfinding, terrain generation, and UI composition work together in a small but non-trivial simulation.

## Overview

The codebase is organized around a simple control chain: `main.py` creates the game shell, `Game.py` manages menus and mission startup, and `MissionControl.py` coordinates the live simulation. From there, multiple subsystems work together:

- `MapGenerator.py` builds the cave using multiprocessing and shared memory.
- `Drone.py` models local exploration, vision, and terrain knowledge.
- `Rover.py` acts as a mobile or stationary support agent depending on mission setup.
- `mapping/terrain_knowledge.py` owns terrain arrays, synchronization, snapshots, observation fusion, and merging.
- `navigation/pathfinding.py` owns pathfinding workers, shared memory, and mission-facing route requests.
- `AStarPathfinder.py` contains the drone and terrain-weighted A* algorithms.
- `Graph.py` tracks valid movement and exploration connectivity.
- `ControlCenter.py` renders simulation status and control widgets.
- `PresentationAdapter.py` keeps UI state and presentation toggles isolated from mission logic.
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
python -m pip install pygame numpy opencv-python
```

Run the simulation:

```bash
python main.py
```

Run the automated test suite:

```bash
python -m unittest discover -s tests -v
```

See [`TESTING.md`](TESTING.md) for the test matrix, placement rules, and manual
smoke checklist.

## System Architecture

The runtime flow is intentionally layered so that each file owns one part of the simulation lifecycle.

### Startup Path

1. `main.py` instantiates `Game` and starts the application.
2. `Game.py` creates the UI window, handles menu navigation, and collects mission settings.
3. When the player starts a mission, `Game` prepares `SimSettings`, generates the cave, and constructs `MissionControl`.
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
responsive.

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

`Drone.py` focuses on exploration. Each drone moves through the cave, scans its surroundings, records terrain observations locally, and uses pathfinding when it needs to reach a frontier or a chosen target.

`Rover.py` serves as a support agent. In the current architecture it is primarily valuable as a rendezvous and accumulation point for terrain knowledge, which makes it useful for centralizing observations without replacing the distributed model.

### Support Systems

`navigation/pathfinding.py` owns the runtime resources used for route requests. It sends drone A* work to a bounded process pool backed by a shared cave map and computes rover terrain-weighted routes in-process. `AStarPathfinder.py` contains the algorithms themselves. `Graph.py` stores and validates exploration connectivity, while `ControlCenter.py` and `PresentationAdapter.py` keep UI concerns outside the simulation core.

## Runtime and Data Flow

The simulation works as a feedback loop:

1. The map is generated.
2. Agents are placed into the world.
3. Drones move and scan.
4. Each drone updates its local terrain knowledge and the mission telemetry aggregate.
5. Nearby agents exchange data when the mission rules allow it.
6. The UI aggregates that knowledge into a shared visualization.
7. Mission completion is checked continuously until all requirements are satisfied.

That flow is important because the game does not use a single global terrain oracle. Instead, knowledge is built from observations and exchanged through explicit events. This makes the heatmap, the agent behavior, and the mission state all consistent with one another.

### Distributed Terrain Knowledge

Terrain state is represented by `TerrainKnowledge`. Mission control, every drone, and every rover own separate instances containing roughness, confidence, a floor mask, and synchronization. Drones update their local instance while the same observations are separately recorded in the mission aggregate for telemetry and combined rendering.

Snapshots provide detached data for rendering and path planning. Sharing decides when knowledge moves between agents, while `TerrainKnowledge.merge_from()` provides the single confidence-weighted merge rule.

The distributed-semantics contract is:

- Agent-local knowledge drives active agent decisions.
- Mission-global terrain is telemetry and UI aggregation only.
- Sharing is the only mechanism that transfers local knowledge between agents.
- Rover movement remains disabled. Its existing target and route code is
  provisional and must use rover-local received knowledge before activation.

### Proximity-Based Sharing

The sharing model is intentionally limited.

- Drone-to-drone exchange is triggered by proximity and throttled so the system does not spend all of its time copying arrays.
- Rover sharing acts as a support mechanism for collecting knowledge near a persistent reference point.
- The heatmap refresh is also throttled so rendering stays responsive.

This makes the simulation feel distributed rather than centralized. Agents learn locally first, then synchronize when they actually meet.

## Cave Generation

The cave is produced by `MapGenerator.py`, which uses parallel worker processes to erode an initially solid map into a navigable cave system. The generator uses shared memory so the workers can write into the same binary terrain buffer without copying the entire map between processes.

The generation pipeline is roughly:

1. Create a shared map buffer.
2. Spawn multiple erosion workers.
3. Let those workers carve and refine the terrain concurrently.
4. Apply post-processing to smooth or clean the result.
5. Build the final terrain data used by the simulation.

The reason for this architecture is practical: cave generation is naturally parallel, and shared-memory multiprocessing is a good fit when several workers need to modify the same large array.

## Exploration Behavior

Drone movement is a mix of local perception, frontier selection, and path navigation.

The agent typically does the following:

- scan nearby terrain with vision-based sensing,
- update its own local knowledge,
- choose a new exploration target,
- validate the move against the graph and cave geometry,
- use pathfinding when the target is not reachable by direct local motion.

This is why the game feels like an exploration system rather than a simple sprite simulation. The drones are reacting to the cave structure, not just wandering randomly across a flat map.

## Rendering and UI

The rendering path is deliberately separated from mission logic.

`rendering/mission_renderer.py` owns complete frame composition and the stop-button visual. `rendering/slam_view.py` builds the selected SLAM or terrain view, while each agent renderer owns paths, vision, and icons. `ControlCenter.py` is responsible for the mission status display, agent statistics, and interaction widgets. `PresentationAdapter.py` keeps presentation state isolated so toggles do not contaminate the simulation model.

Rendering is layered so the visual output stays readable:

1. Black canvas and SLAM or terrain surface
2. Agent paths
3. Drone vision
4. Agent icons
5. Control center and stop button

`MissionControl.draw()` remains as a compatibility wrapper, but runtime code delegates directly to `MissionRenderer.draw()`.

## Configuration and Assets

The project keeps its runtime settings and visual assets in predictable locations.

- `GameConfig/options.ini` stores audio and user preference settings.
- `GameConfig/symSettings.ini` stores the last selected simulation parameters.
- `Assets/` contains the audio, fonts, images, backgrounds, and map resources used by the game.
- `asset_config/` provides typed constants and enums so gameplay values, colors, asset paths, and map-generation parameters stay consistent across modules.

This is a deliberate structural choice. Hard-coding file names and magic numbers across the codebase would make the simulation harder to tune and more brittle to change.

## Controls

In menus:

- `Up` / `Down`: move selection
- `Left` / `Right`: change selector or slider values
- `Enter`: confirm, open a submenu, or start a mission
- Number keys `0-9`: edit the seed field
- `Backspace`: delete seed digits

Simulation settings available in-game:

- Objective: `Exploration` or `Search and Rescue`
- Cave size: `Small`, `Medium`, `Large`
- Seed: custom numeric seed or the default for the selected cave size
- Drones: from 3 to 8
- Demo Cave: `Yes` or `No`

## Project Status

| Feature | Status | Notes |
|---|---|---|
| Terrain roughness map | Implemented | Available in current simulation flow |
| Known map visualization | Implemented | Available in current simulation flow |
| Distributed data sharing (terrain, POI, paths) | In progress | Core sharing exists, integration is ongoing |
| Waypoints for optimal path segmentation | Planned | Not yet implemented |
| Battery management | Planned | Not yet implemented |
| Non-random exploration logic | Planned | Current behavior includes random exploration components |
| Search & Rescue mission logic | Planned | Objective exists in UI; full mission logic is pending |
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
- The standalone `docs/` files were consolidated into this README so this file is now the primary documentation entry point.
