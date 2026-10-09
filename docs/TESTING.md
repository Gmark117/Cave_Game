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
python -m unittest tests.test_terrain_fusion tests.test_terrain_sharing tests.test_rover_targets tests.test_debug_info tests.test_drone_movement tests.test_drone_sensor tests.test_slam_view tests.test_mission_renderer -v
python -m unittest tests.test_menu tests.test_menu_controller tests.test_menu_settings_repository tests.test_menu_audio_service tests.test_menu_renderer -v
```

### Highway and Focused Frontier Batching rollout validation

Keep the committed `[HIGHWAY] mode = off` and `[FOCUSED_FRONTIER_BATCH] mode = off`.
For live validation, first compare matched `HIGHWAY=observe` and `active`
missions with both Focused Frontier Batching and `[INCIDENTAL_SCAN]` off. Then
keep `HIGHWAY=active` and compare `FOCUSED_FRONTIER_BATCH=observe` with
`active`, still with Patch C off. Start with seeds 0 and 5, preserve each JSONL
trace, and analyze it with:

```powershell
python .\tools\analyze_runtime_trace.py <trace.jsonl>
```

For the highway stage compare build status/time, retained snapshots, physical
publications, eligible/selected routes, fallback reasons, query time,
circuity/distance, A* request count/time, check-in processing maxima, and report
delivery. For Focused Frontier Batching also compare planning pass time/status,
route queries and cache hits, opportunities and rejection reasons, seed count
before attachments, batch size, estimated separate/combined/avoided distance,
member and return distance, service time, provisional gain, bounds, stale or
failed members, rejected reports, replays, open leases, accepted reports,
quiescence, coverage, and universal rendezvous acknowledgements. A
`planning_budget` result must produce only ordinary singular assignments and
no new batch claim or lease.
`maximum_planning_ms` is measured as wall-clock time across graph queries,
cache hits, candidate ordering, and lease-preview rasterization.
Only after the isolated stage passes should
`FOCUSED_FRONTIER_BATCH=active` be paired with `INCIDENTAL_SCAN=active`, keeping
the highway setting identical. Full missions remain manual evidence runs
because they generate map assets and have asynchronous timing; automated tests
must not overwrite `Assets/Map/`.

## Where Tests Belong

The component-policy regressions cover stable continuation, split and merge
lineage, targeted wide-component retirement, wall-follow versus sweep
classification, token-fenced claims, rover-known-free eligibility with
drone-owned exact routing, bounded discovery rounds, scan-footprint local DFS,
direct sibling A* and breadcrumb recovery, energy suspension, service-cost rover staging,
contact-carried rendezvous acknowledgement and confirmed-stop relay, pending
check-in retention, endgame gain/travel telemetry, per-agent state projection,
rover-local map selection, atomic rover docking and carriage, inactive docked
sensing, release ordering, component quiescence, and default-off Focused
Frontier Batching. Its coverage includes seed-first parallelism,
rover-known route economics, disjoint spatial leases, all-or-none member
claims, whole-report preflight, per-member stale/suspension isolation, replay,
local member reordering, lease-local frontier admission, hard bounds, and
provisional-work fencing. Legacy sector tests remain to protect the compatibility surface
while production composition bypasses it. A live trace remains necessary to
assess completion time and wall yield; passing unit tests does not establish a
runtime speedup.

Tests live in `tests/` and are named after the module or cohesive subsystem
they protect.

| Area | Test location | Test level | Why it is tested there |
|---|---|---|---|
| Configuration, resource paths, simple models | `test_helpers_and_models.py` | Unit | Nested simulation sections are immutable and validated; shared resources are checked without constructing the game. |
| Movement geometry | `test_graph.py`, `test_helpers_and_models.py` | Unit | Coordinate conventions and wall crossing are foundational to every agent. |
| Vision and roughness sampling | `test_vision_sensor.py`, `test_roughness_sampler.py` | Unit | Dense cone coverage, wall occlusion, sparse ray endpoints, two-cell terrain stride, and confidence decay are deterministic sensor rules. |
| Local SLAM state | `test_slam_map.py` | Unit/concurrency | Private-state snapshots, versions, confidence dominance, point-cloud bounds, and concurrent reads/updates belong to `SlamMap`. |
| SLAM visualization | `test_slam_renderer.py`, `test_slam_view.py` | Surface/service | Pixel data, drone/rover-local versus combined selection, version throttling, and updates racing with rendering can be tested without opening a window. |
| Terrain knowledge model | `test_terrain_knowledge.py` | Unit | Shape validation, snapshots, floor masking, observation fusion, explored ratios, and merging belong to one domain object. |
| Focused service dependencies | `test_terrain_fusion.py`, `test_terrain_sharing.py`, `test_rover_targets.py`, `test_debug_info.py`, `test_slam_view.py`, `test_mission_renderer.py` | Unit/service/interaction | Services receive explicit dependency objects from mission setup; tests build those objects directly to protect narrow collaborator boundaries. |
| Terrain fusion | `test_terrain_fusion.py` | Unit/service | Confidence-weighted updates, injected simulation time, and mission progress are terrain-domain rules. |
| Distributed knowledge semantics | `test_drone_movement.py`, `test_terrain_fusion.py`, `test_terrain_sharing.py` | Characterization | Drone decisions stay local, telemetry stays isolated, and sharing is the explicit transfer boundary. |
| Terrain and SLAM sharing | `test_terrain_sharing.py` | Characterization/concurrency | Proximity, line of sight, service-owned cooldowns, duplicate pair suppression, and transfer direction span multiple agents. |
| Rover target reservation | `test_rover_targets.py` | Unit/service | Service-cost and asperity scoring, reservation, exact staging identity, and material retargeting remain independent of rendering or threads. |
| A* and highway algorithms | `test_astar_pathfinder.py`, `test_highway.py` | Unit/integration | Tests use real NumPy maps and shared memory, but no worker pool; highway tests cover disconnected same-tile regions, strict safe connectors, version fences, hard build budgets, and last-complete snapshot retention. |
| Pathfinding resources | `test_pathfinding_service.py` | Service | Pool creation, bounded submission, fallback, and cleanup belong to the resource owner. |
| Weighted-random movement and A* escape | `test_drone_movement.py`, `test_exploration_policy.py`, `test_astar_pathfinder.py` | Unit/concurrency/interaction | Seeded weighted choice, wall-continuation priority, generic unknown fallback, teammate separation, direct raster steps, sensor-local stagnation windows, scan-only wall-facing turns, exact-pose scan waits, zero-gain suppression, local-SLAM borders, cul-de-sac A*, homing, pause barriers, and path history are protected without a live mission. |
| Frontier registry and lineage | `test_frontier_registry.py` | Unit | Significant eight-connected components, stable continuation, explicit split/merge identity, wide-work classification, normalized effort, anchor-local retirement, atomic multi-group claims, and provisional high-water fencing are deterministic registry rules. |
| Component coordination and energy | `test_exploration_coordination.py` | Unit/state-machine | Full-circle and bootstrap-only radial rounds, component-phase monotonicity, follower assignment, known-free connected eligibility, claim fencing, suspension/reassignment, quiescence, unlimited/finite policy boundaries, Patch D seed-first batching, bounded highway economics, planning-budget singular fallback, observe purity, atomic report rejection, stale-member isolation, lease release, and replay are coordinator rules. |
| Local component DFS execution | `test_component_explorer.py` | Unit/interaction | Exact sensor completion, deterministic follower branch claims, logical LIFO unwinding, direct sibling A*, breadcrumb fallback, physical rover return, physically received highway advice, observe/active route selection, local occupied-edge rejection, anchor-local zero gain, energy/route return reports, default-off incidental wall-pocket scans, Patch D lease containment, exact local tour ordering, and member-failure continuation cross the movement/coordinator boundary. |
| Drone runtime state | `test_drone_runtime_state.py` | Unit/concurrency | Immutable snapshots, atomic self-propelled and carried movement/path updates, frontier timing, and concurrent read consistency belong to the synchronized state owner. |
| Drone behavior | `test_drone_movement.py`, `test_drone_sensor.py` | Characterization/unit | Mission-facing actions use the small `Drone` API; detailed movement, docked idling, inactive docked sensing, terrain, and SLAM behavior is tested through owned collaborators with injected pathfinding, pause, clock, and terrain callbacks. |
| Rover behavior | `test_rover.py` | Characterization | Local-SLAM reachability, acknowledgement-gated departure, advancing, rendezvous hold, and target release form one rover workflow through explicit navigation dependencies. |
| Rendezvous protocol | `test_rendezvous.py` | Unit/state-machine | Endpoint announcements, direct and relayed acknowledgements, universal rover receipt, and arrival commit are deterministic without radio broadcasts. |
| Agent construction | `test_agent_factory.py` | Interaction | Asset loading is mocked while constructor arguments, initialized agent state, and the single-rover policy are verified. |
| Agent rendering | `test_agent_renderer.py` | Surface | Renderer-owned surfaces consume detached agent snapshots; non-empty drawing output is more stable than screenshots. |
| Mission construction and loop | `test_mission_lifecycle.py`, `test_pause_control.py` | Interaction/concurrency | Tests protect setup-only construction, explicit run lifecycle, physical docking, rover-step carriage, assignment release ordering, stop/restart behavior, pause barriers, pause-aware time, cave reuse, and cleanup. |
| Frame performance telemetry | `test_frame_timing.py`, `test_mission_lifecycle.py` | Unit/interaction | Smoothing and lifecycle stage boundaries are deterministic and should not require real-time sleeps. |
| Mission frame composition | `test_mission_renderer.py` | Interaction/surface | Draw order, one coherent drone snapshot per frame, and detached status values crossing into the control center are contracts. |
| Debug information | `test_debug_info.py` | Unit | Debug text should summarize state without requiring the control-center renderer. |
| Presentation transitions | `test_presentation_adapter.py` | State-machine | One owner resets and applies every heatmap/path/vision transition, including invalid action handling. |
| Control-center frame models | `test_control_center_view_model.py` | Characterization | Immutable complete-frame, drone, and rover values expose live display state without retaining mutable references. |
| Control-center controller | `test_control_center_controller.py` | Unit | Timer/pause accounting, active tabs, immutable hit maps, and existing action tokens are independent from rendering. |
| Control-center facade | `test_control_center.py` | Interaction | The mission-facing object builds one frame model, delegates rendering, retains hit geometry, and exposes stable timer/progress methods. |
| Control-center rendering | `test_control_center_renderer.py` | Surface/interaction | The renderer consumes only an immutable frame model, owns Pygame resources and layout, and returns detached hit geometry. |
| Menu facade and settings conversion | `test_menu.py` | Unit/file integration | The `Game`-facing API, mission start, default seed ownership, settings conversion, and simulation INI round trips remain stable. |
| Menu controller | `test_menu_controller.py` | Unit/state-machine | Navigation, named actions, selector/slider bounds, and top-row/numpad seed entry are independent from rendering. |
| Menu rendering | `test_menu_renderer.py` | Surface/interaction | Background and row composition are renderer-owned and consume typed menu rows. |
| Menu settings repository | `test_menu_settings_repository.py` | File integration | Dataclass field round trips, shipped-active versus constructor-off modes, whole-file local precedence, legacy batch sections, historical rendering keys, scalar conversion, and section-level malformed-value fallback are protected. |
| Menu audio | `test_menu_audio_service.py` | Unit/service | Mixer initialization, volume, music, and button-sound behavior are isolated behind mocks. |
| Game event flags | `test_game.py` | Unit | Keyboard-to-flag mapping is independent from window creation. |
| Map-generation helpers | `test_mapgen_helpers.py` | Unit/resource | Brush application, cleanup primitives, process monitoring, and shared-memory helpers are isolated. The disabled wall-transition-noise experiment has been removed. |
| Map-generation services | `test_cave_generator.py`, `test_worm_process_runner.py`, `test_cave_post_processor.py`, `test_terrain_roughness_generator.py`, `test_map_artifact_writer.py` | Unit/resource/file integration | Pure orchestration, process ownership, post-processing fallback, deterministic roughness, and artifact output are tested at their owning modules. |
| Trace replay | `test_runtime_trace_analysis.py` | Replay/characterization | Current summaries and historical waypoint/MCTS/batch telemetry remain readable; structured calculations live in `tools/trace_metrics.py`. |
| Map-generator facade | `test_map_generator.py` | Interaction | The game-facing facade delegates generation and output while preserving mission-facing generated data. |

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

For the control center, test controller actions and hit maps separately from
surface drawing. Renderer tests should pass a complete immutable
`ControlCenterViewModel` and verify the returned `ControlHitMap`, without
constructing or mocking a callback-rich `ControlCenter`.

Do not use pixel-perfect full-window screenshots for ordinary unit tests.
Font rasterization, display scaling, and platform rendering can make those
tests noisy without protecting simulation behavior.

### File Tests

Redirect configuration and generated-map output to `TemporaryDirectory`.
Never let automated tests overwrite committed defaults in `GameConfig/` or
runtime map output in `Assets/Map/`.

### Multiprocessing and Thread Tests

Unit tests should verify worker arguments, shared-memory copying, cooldowns,
stop events, and cleanup deterministically. Avoid timing-based sleeps.
Sharing concurrency tests coordinate workers with events so pair reservation
and duplicate suppression are deterministic.
SLAM concurrency tests use barriers and snapshot/version assertions; callers
must not reach into map arrays or coordinate an external SLAM lock.
Drone-runtime concurrency tests repeatedly mutate and snapshot state, asserting
that position, heading, and path history cannot be observed in a torn
combination. Cross-thread consumers should receive `DroneSnapshot` values
rather than read mutable drone attributes.

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
5. Toggle global terrain, per-drone and per-rover terrain/SLAM, path, and vision controls; confirm each selected map is local to that agent and every drone breadcrumb path remains aligned with the cave map.
6. Confirm drones move, sense, share, return home, and remain visually aligned.
7. Confirm rover information changes after nearby terrain sharing.
8. Press `PAUSE` and verify agents, mission updates, and the timer freeze while
   the window remains responsive. Leave it paused longer than the sharing and
   sensing cooldowns, then press `PLAY` and verify no burst of expired work.
9. Press `RESTART` and verify the same cave restarts with fresh mission state.
10. Press `STOP` and verify worker shutdown and return to the windowed menu.
11. Close the application through the window control.

## Intentional Automated-Test Gaps

The following remain smoke or manual checks:

- Full `generation/map_generator.py` runtime facade with all worm processes and production-size maps.
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
7. Do not preserve a forwarding wrapper solely for tests; move the test to the
   collaborator that owns the behavior unless the wrapper is an intentional
   runtime API.
