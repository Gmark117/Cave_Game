# Testing Strategy

The project uses Python's built-in `unittest` framework. The suite is designed
to support refactoring by protecting behavior at module boundaries without
requiring a graphical desktop, full cave generation, or a complete mission run
for every change.

## Running Tests

Run the complete automated suite:

```bash
python -m unittest discover -s tests -v
```

Compile every Python module:

```bash
python -m compileall -q .
```

Run one subsystem while developing:

```bash
python -m unittest tests.test_terrain_sharing -v
python -m unittest tests.test_mission_lifecycle -v
```

## Where Tests Belong

Tests live in `tests/` and are named after the module or cohesive subsystem
they protect.

| Area | Test location | Test level | Why it is tested there |
|---|---|---|---|
| Configuration, resource paths, simple models | `test_helpers_and_models.py` | Unit | These values are shared widely; failures should be caught without constructing the game. |
| Movement geometry | `test_graph.py`, `test_helpers_and_models.py` | Unit | Coordinate conventions and wall crossing are foundational to every agent. |
| Vision and roughness sampling | `test_vision_sensor.py`, `test_roughness_sampler.py` | Unit | Ray endpoints, wall stopping, and confidence decay are deterministic sensor rules. |
| Local SLAM state | `test_slam_map.py` | Unit | Occupancy, confidence dominance, and point-cloud bounds belong to `SlamMap`. |
| SLAM visualization | `test_slam_renderer.py`, `test_slam_view.py` | Surface/service | Pixel data and local-versus-combined view selection can be tested without opening a window. |
| Terrain knowledge model | `test_terrain_knowledge.py` | Unit | Shape validation, snapshots, floor masking, observation fusion, explored ratios, and merging belong to one domain object. |
| Terrain fusion | `test_terrain_fusion.py` | Unit/service | Confidence-weighted updates and mission progress are terrain-domain rules. |
| Distributed knowledge semantics | `test_drone_movement.py`, `test_terrain_fusion.py`, `test_terrain_sharing.py` | Characterization | Drone decisions stay local, telemetry stays isolated, and sharing is the explicit transfer boundary. |
| Terrain and SLAM sharing | `test_terrain_sharing.py` | Characterization/concurrency | Proximity, line of sight, service-owned cooldowns, duplicate pair suppression, and transfer direction span multiple agents. |
| Rover target reservation | `test_rover_targets.py` | Unit/service | Scoring, reservation, and completion must remain independent of rendering or threads. |
| A* algorithms | `test_astar_pathfinder.py` | Unit/integration | Tests use real NumPy maps and shared memory, but no worker pool. |
| Pathfinding resources | `test_pathfinding_service.py` | Service | Pool creation, bounded submission, fallback, and cleanup belong to the resource owner. |
| Drone behavior | `test_drone_movement.py`, `test_drone_sensor.py` | Characterization | Controller behavior is tested through the stable `Drone` API and local state. |
| Rover behavior | `test_rover.py` | Characterization | Planning, advancing, and target release form one rover workflow. |
| Agent construction | `test_agent_factory.py` | Interaction | Asset loading is mocked while constructor arguments and initialized agent state are verified. |
| Agent rendering | `test_agent_renderer.py` | Surface | Renderer-owned surfaces and non-empty drawing output are more stable than screenshots. |
| Mission construction and loop | `test_mission_lifecycle.py` | Interaction | Tests protect setup-only construction, explicit run lifecycle, stop/restart/pause behavior, cave reuse, and cleanup. |
| Frame performance telemetry | `test_frame_timing.py`, `test_mission_lifecycle.py` | Unit/interaction | Smoothing and lifecycle stage boundaries are deterministic and should not require real-time sleeps. |
| Mission frame composition | `test_mission_renderer.py` | Interaction/surface | Draw order is a contract; individual visual details remain in focused renderer tests. |
| Debug information | `test_debug_info.py` | Unit | Debug text should summarize state without requiring the control-center renderer. |
| Presentation transitions | `test_presentation_adapter.py` | State-machine | One action should produce one consistent heatmap/path/vision state. |
| Control-center input and state | `test_control_center.py` | Unit | Hit testing, timer formatting, and color thresholds do not need a live display. |
| Menu input and persistence | `test_menu.py` | Unit/file integration | Values, navigation, and INI round trips are tested without audio or the menu loop. |
| Game event flags | `test_game.py` | Unit | Keyboard-to-flag mapping is independent from window creation. |
| Map-generation helpers | `test_mapgen_helpers.py` | Unit/resource | Brushes, cleanup, seeded noise, process monitoring, and shared-memory helpers are isolated. |
| Map-generator orchestration | `test_map_generator.py` | Interaction/file integration | Worker calls are mocked; deterministic roughness and temporary output files are real. |

## When To Apply Each Test Type

### Pure Unit Tests

Use a pure unit test when inputs and outputs fully describe the behavior.
Examples include coordinate helpers, A* costs, confidence fusion, target
scoring, and map-processing functions.

Add these tests before changing formulas or algorithms. They should use small
arrays and exact assertions whenever practical.

For distributed behavior, also add a characterization test whenever a decision
source changes. The test should distinguish agent-local knowledge, mission
telemetry, and explicitly shared knowledge rather than populating all three
with identical data.

### Characterization Tests

Use characterization tests before restructuring behavior that already works
but is distributed across several objects. Terrain sharing, drone movement,
rover planning, and presentation transitions fall into this category.

These tests should describe externally meaningful behavior rather than current
private implementation steps. They allow collaborators to move code without
silently changing the rules.

### Interaction Tests

Use interaction tests at ownership boundaries such as factories, mission
lifecycle, rendering composition, and pathfinding resource management.

Mock expensive or external collaborators, then verify:

- which collaborator was called,
- the arguments crossing the boundary,
- the resulting public state,
- cleanup after success or failure.

### Pygame Surface Tests

Use in-memory `pygame.Surface` objects for renderers and visual data. Assert
stable properties such as alpha, representative colors, non-empty pixels, and
layer order.

Do not use pixel-perfect full-window screenshots for ordinary unit tests.
Font rasterization, display scaling, and platform rendering can make those
tests noisy without protecting simulation behavior.

### File Tests

Redirect configuration and generated-map output to `TemporaryDirectory`.
Never let automated tests overwrite `GameConfig/` or `Assets/Map/`.

### Multiprocessing and Thread Tests

Unit tests should verify worker arguments, shared-memory copying, cooldowns,
stop events, and cleanup deterministically. Avoid timing-based sleeps.
Sharing concurrency tests coordinate workers with events so pair reservation
and duplicate suppression are deterministic.

Use a separate smoke run for real process spawning and long-running mission
threads. Those checks answer platform and lifecycle questions rather than
individual function correctness.

## Manual Smoke Checklist

Run this after changes to Pygame initialization, map generation, threading,
audio, or full-frame layout:

1. Run `python main.py`.
2. Navigate every menu and confirm selector, slider, seed, and audio behavior.
3. Start a small mission with three drones.
4. Confirm cave generation completes without worker or shared-memory warnings.
5. Toggle global terrain, per-drone terrain, path, and vision controls.
6. Confirm drones move, sense, share, return home, and remain visually aligned.
7. Confirm rover information changes after nearby terrain sharing.
8. Press `PAUSE` and verify agents, mission updates, and the timer freeze while
   the window remains responsive; press `PLAY` and verify they resume.
9. Press `RESTART` and verify the same cave restarts with fresh mission state.
10. Press `STOP` and verify worker shutdown and return to the windowed menu.
11. Close the application through the window control.

## Intentional Automated-Test Gaps

The following remain smoke or manual checks:

- Full `MapGenerator.__init__()` with all worm processes and production-size maps.
- Long-running concurrent missions and race/stress behavior.
- Mixer initialization, music playback, and speaker volume.
- Full control-center visual layout across display scaling and operating systems.
- Window maximize/windowed transitions on a real desktop.
- Offline asset-generation scripts in `tools/`.
- Subjective cave quality and visual roughness distribution.

These gaps are intentional. If one becomes a frequent source of regressions,
promote it to a focused automated test rather than adding a broad end-to-end
test that is slow or unreliable.

## Adding Tests During Future Work

1. For a bug, write a failing reproduction first.
2. For a refactor, add characterization coverage before moving ownership.
3. For a new algorithm, test boundary values and failure cases with small maps.
4. For a new service, test lifecycle and cleanup as well as successful output.
5. For new UI state, test action-to-state transitions separately from drawing.
6. For rendering, test data and representative pixels, then perform one manual visual smoke run.
