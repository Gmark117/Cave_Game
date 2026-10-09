# Behavior-preserving cleanup audit — 2026-10-09

## Baseline and scope

The audit started from clean commit
`05a2d8a1d2c6161196beed4c29f40b7e47453bdf` (Highway system optimization).
The baseline suite passed 608 tests. No on-disk `AGENTS.md` was found in the
project or its ancestors; the supplied thread-handoff and token-efficiency
instructions applied. Cleanup changes are left for review and commit.

The inventory covered all 150 tracked Python files: 90 production/package/tool
modules, 59 test modules, and the tests package initializer. Review combined
AST import/caller/state inventories, dependency and duplication checks,
targeted source/contract review, existing tests, and differential replay.
Documentation, INI settings, requirements, asset paths, and generated artifacts
were included. This is a maintenance audit, not an exhaustive correctness proof.

| Area reviewed | Responsibilities and cleanup decision |
|---|---|
| Startup and composition | `main.py`, `game.py`, `contracts.py`, `agents/factory.py`, and `mission/control.py` retain their composition, public callbacks, initialization order, and restart boundaries. Removed unused presentation dimensions. |
| Drone and rover agents | `drone.py`, `rover.py`, `drone_runtime_state.py`, `graph.py`, `component_explorer.py`, `drone_movement.py`, and `exploration_policy.py`: extracted coverage accounting into `CoverageMemory`; removed uncalled movement helpers and unused state/locals. Preserved all execution, scan, return, collision, and ownership rules. |
| Mission coordination | `exploration_coordination.py`, `rendezvous.py`, `energy.py`, and `objectives.py`: removed write-only reconciliation state and unused imports. Report preflight/commit, claims, lineage, leases, focused activation, universal ACK, and energy hooks remain intact. |
| Lifecycle and diagnostics | `lifecycle.py`, `pause_control.py`, `frame_timing.py`, `runtime_trace.py`, `debug_info.py`, and `presentation_adapter.py`: retained barrier, clock, teardown, and resource ownership; removed uncalled combined-debug and disabled-logger wrappers. The dynamically invoked dock cleanup is retained. |
| Mapping and sensing | All `mapping/` modules: retained versioned SLAM/terrain ownership, exact merge rules, dense occupancy versus sparse terrain sensing, frontier significance/lineage, localization and POI boundaries. Removed a redundant terrain comparison and unused rover-target local. |
| Physical sharing | `terrain_sharing.py` retains contact predicates, revision gates, lock ordering, callbacks, arrival/departure exchanges, and non-extending encounter pauses. No communication predicate was consolidated across different encounter roles. |
| Navigation | All `navigation/` modules: shared the process-pool request lifecycle between complete and segmented A* APIs. Removed unused highway worker arguments/state and a dead spur-distance accumulator. Kept the public builder signature and all routing limits, tie-breaks, validation, and fallback rules. |
| Rendering | All `rendering/` modules retain snapshot consumption, layer ordering, cache invalidation, colours, opacity, and visibility behavior. The combined-view occupancy defect below is recorded separately. |
| UI | All menu/control-center controllers, facades, view models, panels, widgets, renderers, text helpers, and audio/settings services: removed the obsolete tab-icon loader/fallback and unused wrappers/arguments. Existing button assets, layout, action tokens, audio behavior, and hit maps remain. |
| Configuration | `config/`, `GameConfig/`, and `asset_config/`: replaced repeated scalar INI parsing/writing with the existing dataclass schema; retained mission labels, rendering aliases, conversions, validation, legacy sections, and file precedence. Shipped files and constructor values are unchanged. |
| Generation | All `generation/` modules: removed the disabled, unconfigured wall-transition-noise experiment and its test, along with post-processing inputs used only by that experiment. Worm carving, RNG order, morphology, roughness, process ownership, and artifact output remain. |
| Tools | `analyze_runtime_trace.py` now owns loading, normalization, mission summaries, and CLI; `trace_metrics.py` owns structured characterization calculations. Public metric imports remain available from the original analyzer. Updated the highway benchmark caller; retained tiled comparisons and the confidence-curve utility. |
| Tests and documentation | Reviewed the suite by subsystem, updated tests at moved/removed boundaries, and added configuration and pool-failure characterization. Updated README, CODEFLOW, TESTING, patch/validation records, and the historical SLAM plan. Bibliography and generated illustrations remain historical/reference assets. |

## Resulting simplifications

- The settings repository shrank from 803 to 235 lines. Dataclass fields now
  describe ordinary scalar settings once. Mission names and the two historical
  rendering key aliases remain explicit. Each malformed section still falls
  back as a whole; unrelated sections still load.
- Coverage-memory cells, undirected edges, exponential decay, crossing counters,
  and heading penalties have one owner in the existing exploration-policy
  module. Movement supplies local position, time, and map bounds. The formulas
  and order of traversal updates are unchanged.
- Structured trace metrics moved out of the 4,700-line analyzer into one
  cohesive module. The analyzer remains responsible for CLI and human-readable
  mission summaries; historical formats and existing public imports survive.
- Both drone process-pool route APIs share resource checks, submission,
  semaphore release, and error handling. They still submit their original
  worker functions and return their original fallback/result formats.
- Removed unused wrappers, imports, parameters, write-only bookkeeping, the
  inactive noise experiment, and obsolete tab drawing/loading. No framework,
  policy rewrite, or new runtime worker was introduced.
- Production Python decreased from 37,863 to 36,994 lines, including the new
  metrics module: a net reduction of 869 lines. Moving code is not counted as
  deletion in this total.

## Configuration and retained compatibility

HIGHWAY, INCIDENTAL_SCAN, and FOCUSED_FRONTIER_BATCH ship as `active` in
`GameConfig/simulation.default.ini`. Their constructors still use `off`.
The menu chooses the local simulation file if present, otherwise the shipped
file; it does not overlay a partial local file onto the shipped defaults.
Missing keys and invalid sections use the supplied typed fallback. Audio
loading separately merges shipped defaults and local overrides.

Retained deliberately:

- Legacy `[ENDGAME_BATCH]` input and historical `*_endgame_batch_*` trace
  normalization; new saves/events use Focused Frontier Batching terminology.
- Sector coordination, movement hooks, rendering, and characterization tests
  used for offline comparison; production composition uses component work.
- The tiled highway builder and historical snapshot/query representation,
  exercised by `benchmark_highway.py` and characterization tests.
- Historical waypoint/MCTS trace characterization and accepted old policy
  names, which still normalize to the supported random policy.
- Energy contracts, reserve policy, POI models, facade/dependency boundaries,
  and deliberate compatibility entry points with current consumers.
- Rover construction's apparently unused random heading draw: removing it
  would advance the shared RNG differently. Cleanup removed its unused empty
  direction log without changing random consumption.
- Independent collision, communication, and known-free route rasterization.
  Their sampling and corner rules differ; combining them could change behavior.

## Separate behavior finding

**D1 — Equal-confidence occupancy conflicts depend on drone order in the
combined display.** `SlamViewService._render_combined()` replaces a cell only
when incoming confidence is strictly greater. With two one-cell snapshots at
confidence 1.0, occupancies `(FREE, OCCUPIED)` display FREE, while reversing
the agent order displays OCCUPIED. This was reproduced directly with detached
`SlamSnapshot` inputs and a captured renderer call. `SlamMap` fusion instead
has an occupied-tie rule. This affects the combined visualization, not the
agents' local maps or mission authority. No fix is included; review a separate
display change and its desired conflict policy before implementing one.

## Verification and limits

- Complete suite: **612 tests passed** (four configuration characterizations
  and one pool-failure test added; one obsolete noise-only test removed).
- Focused suites passed after the meaningful cleanup batches.
- **721 differential configuration cases** matched committed HEAD, including
  invalid numeric values, modes, missing values, and non-finite inputs.
- **1,000 coverage transitions and 2,000 heading-pressure comparisons** matched
  the original methods exactly under a deterministic replay.
- Full summaries for `mission_trace_20261009_125123_464282.jsonl` and historical
  all-active seed-5 `mission_trace_20261008_115111_122353.jsonl` matched HEAD
  exactly. Direct script CLI execution also succeeded.
- `python tools/benchmark_highway.py --access-distance 80 --updates --background`
  passed initial publication and wall-correction checks for all three synthetic
  fixtures using a real background process. Builds completed in about 16–40 ms
  in this smoke check; this is not a mission-performance comparison.
- `python -m compileall -q .` passed. `git diff --check` passed with only
  LF/CRLF conversion warnings.
- All four generated map SHA-256 values remain unchanged:

| File under `Assets/Map/` | SHA-256 |
|---|---|
| `floor.png` | `8F150BC1B838F91EEC2B8E9132E1E9830ED4AF49E1897AAED1ECC08CA91BC45E` |
| `map.png` | `7CA1AF8F6B5F43FA8F6F9E8D6EF77EA1EE74880EE9F8856E811A245382A36CE1` |
| `map_matrix.txt` | `CDED49279DDC06CF3A1626213429F616116F6D3E8CE1CD510DF406775854BE6C` |
| `walls.png` | `E675B56895AA77E3E84A0AA7AE262334E8A4798F2B93E79A65EB5C10F6FCB875` |

No new full live mission was run. The supplied smooth all-active run remains
the latest live baseline: exploration 345.71 s, coverage 97.62%, shutdown
364.89 s, 48 claims/reports, 55 released docks, and no open batch claims/leases.
Native highway work can still exceed the 250 ms build-attempt allowance before
its next budget check. Main movement/coordinator modules remain substantial;
their distributed state machines were retained instead of rewriting ownership
or introducing forwarding layers. A visual/live smoke run can be performed by
the user before committing, following `TESTING.md`.

## Ready-to-paste battery handoff

Objective: design the battery implementation after reviewing the completed
behavior-preserving cleanup. Battery drain/charging was outside cleanup scope.
Workspace: `C:\Users\gianm\Documents\VisualStudioCodeProjects\PYTHON\Progetto_Distributed_Systems\Cave_Game`,
Windows/PowerShell. Cleanup started clean at `05a2d8a`; it was left uncommitted
for review. Inspect current HEAD/status and applicable AGENTS instructions.

Completed work: simplified dataclass-backed INI persistence, moved local
coverage accounting into `agents/exploration_policy.py`, extracted
`tools/trace_metrics.py`, consolidated pathfinding pool submission, removed
the inactive wall-noise experiment, obsolete UI icon code, and unused state,
arguments, imports, and wrappers. Updated corresponding tests and README/docs.
The change groups and exact verification evidence are recorded above.

Constraints: preserve physical communication, local-SLAM authority, component
and DFS ownership, tokened claims/reports, lineage, docking, report encounters,
quiescence, universal ACK, scan completion and route-suffix rules, routing
tie-breaks/limits/fallbacks, and focused-only batch activation. Shipped HIGHWAY,
INCIDENTAL_SCAN, and FOCUSED_FRONTIER_BATCH are active; constructors remain off.
Keep legacy ENDGAME_BATCH and historical telemetry compatibility. Preserve all
four generated map assets and unrelated worktree changes.

Verification: 612 tests, 721 config comparisons, 1,000 coverage transitions,
2,000 heading comparisons, exact summary replay of latest/historical traces,
and real synthetic background-worker publication/correction smoke checks.
No post-cleanup full live mission was run. Separate unresolved D1: combined
SLAM display conflicts at equal confidence depend on agent order; do not mix
that fix with battery work. Native highway calls can overrun attempt budgets.

Next steps: (1) inspect/review the cleanup diff and its current commit state;
(2) run the full suite if the baseline changed; (3) review `mission/energy.py`,
executor checkpoints/suspensions in `agents/drone_movement.py`, coordinator
acceptance in `mission/exploration_coordination.py`, and `DroneRuntimeState`;
(4) ask the user to select movement/rotation/scan/idle drain, reserve, charging,
and docking rules before implementing finite energy; (5) add focused behavior
tests and let the user run live simulations for trace validation.

Suggested opening prompt: “Review the cleanup baseline and
`docs/CLEANUP_AUDIT.md`, then help design battery behavior using the existing
energy and suspension contracts. Preserve distributed ownership and physical
communication. Agree on finite-energy rules before implementation, and keep
the separate combined-display defect out of scope.”
