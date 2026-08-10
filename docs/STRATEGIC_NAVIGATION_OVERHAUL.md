# Strategic Navigation and Exploration Overhaul

## Purpose

This document is the staged implementation plan for replacing Cave Game's
dense breadcrumb waypoint graph and per-tick frontier selection with:

- a strategic, spatially indexed navigation graph;
- stable frontier clusters and shared reservations;
- cached routes and persistent per-drone navigation intent;
- explicit `TRAVEL`, `SCAN`, `RECOVERY`, and `HOME` modes;
- belief-only planning and connectors; and
- a bounded, goal-conditioned local MCTS controller.

Complete the phases in order. Keep the full test suite green at every phase
boundary and do not reset or discard the existing uncommitted waypoint work.

## Baseline and Guardrails

The reference trace is `logs/mission_trace_20260714_000154.jsonl`, recorded
with seed 5, a SMALL map, and three drones for approximately 610 seconds.

Current baseline:

- 1,931 waypoint nodes, 2,304 active source-specific edges, and 2,253
  physical endpoint connections; the trace contains 2,322 successful edge
  add/shorten mutation events;
- 75.7% of nodes have a neighbor within 8 px;
- 80.1% of nodes are degree-two chain nodes;
- 1,551 route calls cover 233 unique drone-target pairs;
- 62.0% of decisions are zero-reward frontier fallbacks;
- 85.4% of MCTS searches perform at most one iteration;
- median MCTS decision time is 81.2 ms with a 40 ms budget;
- 42.1% of decisions following a waypoint segment switch targets; and
- d1's late A-B-A reversal rate reaches 80.3%.

Repository guardrails:

- The pre-migration baseline suite contains 280 passing tests.
- Preserve all tracked and untracked work already present in the worktree.
- Keep true cave-map data out of planning, frontier extraction, homing, route
  selection, and connector generation.
- Ground truth may remain only at simulator boundaries: sensor ray casting,
  physical collision enforcement, and physical communication line of sight.
- Preserve complete travelled and belief-corridor polylines when simplifying
  topology.
- Never merge nearby graph objects merely because they share a spatial bucket.

## Locked Architecture Decisions

1. Global cluster selection is deterministic. MCTS is retained only as a
   bounded, local, goal-conditioned controller.
2. Physically travelled routes are mission-global and trusted in the static
   cave. Belief-created corridors are validated for each requester and
   knowledge revision.
3. Frontier geometry remains per-drone knowledge. A shared registry provides
   stable identity and reservations without exposing clusters to drones that
   do not know them.
4. Normal travel follows stored edge polylines deterministically. Local MCTS is
   invoked only when competing deviation primitives exist; deterministic scan
   and single-choice recovery modes bypass it.
5. Route and cluster IDs are monotonic and never reused after retirement.
6. Rendering initially performs one full overlay rebuild per committed graph
   revision. Incremental dirty-edge rendering is deferred unless profiling
   proves it necessary.
7. Energy remains a planning cost proxy. This migration does not add battery
   depletion.

## Target Public Types and Contracts

### Navigation graph

- `WaypointId = int`
- `EdgeId = int`
- `WaypointRole`:
  - `HOME`
  - `JUNCTION`
  - `CHOKEPOINT`
  - `TURN`
  - `FRONTIER_GATEWAY`
  - `RECOVERY_ANCHOR`
- `WaypointNode`:
  - stable ID;
  - position;
  - one or more strategic roles; and
  - created/updated topology revisions.
- `WaypointEdge`:
  - stable ID and endpoint IDs;
  - exact oriented polyline and cost;
  - `TRAVELLED` or `BELIEF_CORRIDOR` evidence;
  - optional owner/requester scope; and
  - created/retired revisions.
- `WaypointRoute`:
  - status;
  - topology and requester-knowledge revisions;
  - node and edge IDs;
  - entry and exit connector polylines;
  - total/remaining cost; and
  - cache-hit status.
- `GraphDelta`:
  - one committed topology revision;
  - added, updated, and retired node/edge IDs; and
  - old-edge to split-edge replacement mappings.

During migration, retain the current route status strings and coordinate/source
aliases so trace analysis, rendering, and unaffected tests continue to work.

### Exploration and runtime state

- `FrontierClusterId = int`
- `MovementMode`: `TRAVEL`, `SCAN`, `RECOVERY`, `HOME`
- `NavigationIntent`:
  - stable goal cluster and gateway IDs;
  - assignment token;
  - topology and requester-knowledge revisions;
  - route edge IDs;
  - edge and polyline cursors;
  - remaining route cost; and
  - selection SLAM version.
- `MovementOutcome`:
  - travelled distance;
  - route-progress delta;
  - arrival and collision flags;
  - scan completion and actual information gain; and
  - explicit invalidation or transition reason.
- `FrontierCluster` and per-drone view:
  - stable ID, frontier cells/bounds, representative, expected gain, wall
    continuation gain, and wall-adjacent requester waypoints;
  - gateway ID and `known_by` drone IDs;
  - active/reserved/retired lifecycle;
  - first/last-seen revisions and missing-refresh count; and
  - revisit, stall, and zero-gain penalties.
- `SlamProgressSnapshot`:
  - SLAM version and completed-scan sequence;
  - cumulative sensor and shared newly-known cells; and
  - cumulative confidence gain.

`ExplorationContext` must no longer contain `cave_map`, truth-based segment
validation, or authoritative raw frontier coordinates. `DroneSnapshot.frontiers`
may temporarily expose cluster representatives as a compatibility view.

## Default Parameters

Use these defaults until trace evidence justifies tuning:

| Setting | Default |
|---|---:|
| Node spatial-hash cell | 32 px |
| Safe node merge radius | 8 px |
| Pose-to-graph connector radius | 64 px |
| Gateway/frontier connector cap | 192 px |
| Route-cache capacity | 64 roots |
| Sustained-turn threshold | 45 degrees |
| Minimum confirmed turn leg | 24 px |
| Chokepoint narrow clearance | 8 px |
| Chokepoint shoulder clearance | 16 px |
| Minimum chokepoint shoulder length | 24 px |
| Coarse recovery-anchor interval | 128 px |
| Cluster match distance | 32 px |
| Gateway minimum separation | 64 px |
| Missing-cluster hysteresis | 3 refreshes |
| Connector A* iteration cap | 4,000 |
| Connector A* time cap | 12 ms |
| Global/local planning budget | 40 ms |
| Watchdog no-progress time | 10 seconds |
| Watchdog no-progress distance | 64 px |
| Revisit window | 32 actions |
| Revisit trigger | 0.60 |

Legacy INI keys such as `spacing`, `direct_path_limit`, `stride`, and
`frontier_cluster_limit` should remain readable for one compatibility cycle,
but must no longer control breadcrumb sampling, omniscient direct A*, or
per-decision frontier preprocessing. Save the replacement schema on the next
settings write.

## Phase 0 - Characterization and Instrumentation

### Goal

Protect the failure modes before moving ownership or changing algorithms.

### Work

- [x] Refactor `tools/analyze_runtime_trace.py` around structured metrics while
      preserving `summarize()` as the CLI formatter.
- [x] Add metrics for waypoint density, target retention, route abandonment,
      A-B-A reversals, route-cache use, planner timing, and information gain
      per travelled pixel.
- [x] Add synthetic or distilled trace fixtures. Do not commit the 9.94 MB
      production trace.
- [x] Trace completed sensor sequence, newly-known cells, confidence gain, and
      travelled distance without changing movement behavior.
- [x] Add belief-only characterization tests showing that identical SLAM and
      terrain beliefs produce identical decisions under different cave maps.
- [x] Add failing/characterizing tests for target retention, route cursor
      persistence, zero-gain frontier regeneration, and budget overruns.
- [x] Run an instrumented seed-5/SMALL/three-drone legacy baseline so actual
      information gain per distance is available for the final comparison.

### Exit criteria

- [x] Existing trace metrics reproduce the audited density, switching,
      reversal, routing, and MCTS figures.
- [x] The analyzer handles both legacy and replacement event schemas.
- [x] All pre-existing tests plus the non-behavioral instrumentation tests pass.

Phase 0 complete (2026-07-15). All 308 tests pass with two intentional
expected-failure guards for the existing ground-truth cave-map leak. Compile
and diff checks pass. Reference replay uses an exclusive 8 px density radius,
distinct-neighbor topological degree, and distinguishes edge mutations from
active edges.

The canonical instrumented legacy baseline is
`mission_trace_20260715_212134_945240.jsonl`: a clean schema-v2 seed-5/SMALL/
three-drone capture spanning 650.229 simulation-event seconds. All acceptance
metrics use its fixed normalized 0-600 second window so later runs compare the
same duration:

```powershell
python tools/analyze_runtime_trace.py logs/mission_trace_20260715_212134_945240.jsonl --window-end 600 --reversal-window-start 540
```

Within that window, 7,167/7,167 scans have complete gain telemetry. Sensor
observations discover 1,021,831 newly-known cells over 64,556.467 px of exact
travel, establishing the primary baseline at **15.828484 cells/px**. Confidence
gain remains a separate signal at **15.749109/px**. The final two-times gate is
therefore **31.656968 newly-known sensor cells/px** over the same 0-600 window.

Late-window A-B-A is 0/41 for d0, N/A for d1, and 0/44 for d2. d1's N/A is a
stall signal, not success: from 540-600 seconds it travels 1,474.224 px and
completes 233 scans with zero newly-known cells and no frontier arrivals. The
same-seed behavior differs materially from the original failure trace, so keep
`mission_trace_20260714_000154.jsonl` as the audited failure reference and use
the new trace only as the instrumented efficiency baseline.

## Phase 1 - Strategic Graph Core

### Goal

Replace coordinate identity and route-local graph reconstruction while keeping
the current graph façade usable.

### Work

- [x] Replace coordinate-keyed storage with monotonic ID-keyed node and edge
      dictionaries.
- [x] Add persistent adjacency and a node spatial hash. Add an edge-polyline
      cell index for travelled-intersection discovery.
- [x] Treat spatial buckets only as candidate sources. Validate every merge by
      exact travelled intersection or requester-known-free LOS.
- [x] Add graph transactions so one logical mutation produces one topology
      revision and one `GraphDelta`.
- [x] Maintain connected components incrementally on additions and rebuild
      them after split/removal batches.
- [x] Implement atomic edge splitting at a real travelled junction. Preserve
      the original oriented polyline exactly in the replacement edges.
- [x] Collapse only unprotected, geometrically redundant degree-two nodes by
      concatenating their polylines. Never collapse active-route nodes.
- [x] Add reverse shortest-path trees rooted at gateways with a 64-entry LRU.
- [x] Key route trees by gateway, topology revision, requester, and scoped
      belief-edge validity. Unrelated SLAM revisions remain reusable. Do not
      use Floyd-Warshall.
- [x] Reuse cached routes optimistically across ordinary SLAM changes and
      revalidate only requester-scoped edges on the selected route.
- [x] Update graph snapshots and rendering to consume IDs and roles while
      retaining coordinate/source compatibility properties.

### Tests

- [x] IDs are stable, monotonic, and never reused.
- [x] Wall-separated nodes in one hash bucket never merge.
- [x] Travelled intersections split the correct edge in one revision.
- [x] Split/collapse operations preserve exact physical polylines and cost.
- [x] Component rejection happens before shortest-path search.
- [x] Repeated unchanged routes build one tree and then hit the cache.
- [x] Topology and requester-validity changes invalidate only the appropriate
      caches.
- [x] Concurrent insertion, routing, splitting, and snapshots remain coherent.

### Exit criteria

- [x] No route call rebuilds the full persistent adjacency.
- [x] Repeated route lookups at unchanged revisions are cache hits.
- [x] Renderer rebuilds occur once per committed graph revision.
- [x] The focused graph, renderer, and configuration suites pass.

Phase 1 complete (2026-07-15). The graph now stores stable monotonic node and
edge IDs behind coordinate/source compatibility properties. One logical
mutation commits one topology revision and `GraphDelta`; persistent adjacency,
node and edge-polyline spatial indexes, and connected components are maintained
under the graph lock. Transverse travelled intersections split and connect
exact polylines atomically, while protected-node-aware collapse restores the
same oriented geometry and cost.

Routing rejects disconnected components before search and uses requester- and
revision-keyed reverse shortest-path trees in a 64-entry LRU. Cache hits
revalidate only belief-scoped edges on selected paths, allowing unrelated SLAM
changes to reuse the tree. Breadcrumb ingestion remains in place for Phase 2.

## Phase 2 - Strategic Trail Ingestion

### Goal

Stop creating pose breadcrumbs and fixed-interval bridge nodes.

### Work

- [x] Replace `_waypoint_pending_path` sampling with a per-drone trail
      accumulator.
- [x] Keep the current pose and uncommitted tail ephemeral. Never force-flush a
      pose node before routing.
- [x] Promote a `TURN` only after a heading change of at least 45 degrees with
      at least 24 px of confirmed travel on both sides.
- [x] Promote a `JUNCTION` when a travelled trail actually intersects an
      existing edge, splitting that edge atomically.
- [x] Promote a `CHOKEPOINT` only at a confirmed local minimum of SLAM-known
      free-space clearance with the configured narrow/wide shoulders.
- [x] Add a `RECOVERY_ANCHOR` only after 128 px without another strategic node.
- [x] Store a complete travelled section between strategic endpoints as one
      edge polyline.
- [x] Store a complete bounded belief corridor as one edge polyline with no
      interior `known_free` nodes.
- [x] Consolidate overlapping trails through intersection/splitting rather than
      independent per-drone sampling.
- [x] Keep HOME permanently protected and allow nodes to acquire additional
      roles without changing ID.

### Tests

- [x] A straight path below the recovery interval creates only its strategic
      endpoints and one edge.
- [x] A long straight path creates only coarse recovery anchors.
- [x] A curved path preserves its full shape while promoting only confirmed
      turns.
- [x] Replaying the same or overlapping trail is topology-revision stable.
- [x] A near path across a wall remains separate.
- [x] A corridor bridge creates one scoped edge rather than a 32 px chain.
- [x] Far-route attempts do not add arbitrary current-pose nodes.

### Exit criteria

- [x] Node roles no longer include `travelled` or `known_free`.
- [x] Fixed-spacing and force-flush tests have been replaced by strategic-graph
      invariants.
- [x] The focused graph and movement suites pass.

Phase 2 complete (2026-07-15). Each movement controller now owns a strategic
trail accumulator. Current poses and uncommitted tails remain ephemeral;
routing no longer force-flushes arbitrary pose nodes. Confirmed turns,
SLAM-clearance chokepoints, real travelled intersections, and coarse recovery
anchors promote strategic nodes, while each committed section retains its
complete physical polyline.

Bounded known-free bridges now persist as one requester-revalidated belief
corridor with no fixed-interval interior nodes. Exact and overlapping trail
replays consolidate through stable endpoint reuse and atomic splitting. HOME
and all strategic roles remain protected, and existing nodes can acquire
additional roles without changing ID. The complete suite passes 326 tests with
the two intentional expected failures for the existing cave-map leaks.

## Phase 3 - Stable Frontier Clusters and Reservations

### Goal

Create one belief-only frontier pipeline with persistent identity and explicit
multi-drone coordination.

### Work

- [x] Build one canonical mask from SLAM occupancy/confidence: confidently
      free, confidently occupied, and unknown/low-confidence.
- [x] Use terrain confidence only when estimating value; never use it to infer
      traversability or true floor geometry.
- [x] Cache the frontier mask and update dirty regions plus the required halo.
      Use a full rebuild only initially or after a large shared-map update.
- [x] Extract connected frontier components once per relevant SLAM refresh.
      MCTS and persisted state must consume these same clusters.
- [x] Match a refreshed component to an existing ID when cell overlap is at
      least 25%, or its representative is within 32 px and expanded bounds
      overlap.
- [x] Retain a temporarily missing cluster for three refreshes.
- [x] Tombstone retired/zero-gain geometry so an unchanged pseudo-frontier does
      not receive a new ID on the next decision.
- [x] Add a shared registry that canonicalizes independently observed clusters
      while exposing each cluster only to drones in its `known_by` set.
- [x] Transfer cluster knowledge explicitly during proximity sharing instead
      of sharing raw coordinate lists as a side effect of terrain changes.
- [x] Add an atomic assignment registry. One cluster or consolidated gateway
      cannot be reserved by two drones simultaneously.
- [x] Protect a gateway only after a required belief corridor exists. A route
      using an ephemeral goal connector creates no speculative gateway.
- [x] Protect gateway IDs referenced by active intents. Retirement must produce
      an explicit lifecycle/topology event rather than silent disappearance.

### Tests

- [x] Different cave maps with identical beliefs produce identical masks,
      clusters, scores, and goals.
- [x] Dirty-region and full frontier extraction produce equivalent results.
- [x] Cluster IDs survive representative movement and brief disappearance.
- [x] Reached/retired zero-gain clusters do not regenerate unchanged.
- [x] Two independent observations of the same geometry receive one canonical
      ID without leaking the geometry to an uninformed drone.
- [x] Reservation races have exactly one winner and release deterministically.
- [x] Shared cluster knowledge follows the communication boundary.

### Exit criteria

- [x] No policy performs whole-map frontier extraction during `decide()`.
- [x] Runtime state no longer treats raw coordinates as authoritative frontier
      identity.
- [x] Root and persisted frontiers cannot diverge.
- [x] Frontier, sharing, and concurrency suites pass.

Phase 3 is implemented by `navigation/frontier_clusters.py`. Each drone owns a
version-cached belief extractor, while the mission owns the canonical registry,
assignment registry, and atomic gateway manager. Canonical IDs are shared, but
component geometry remains in per-drone views until an authorized proximity
exchange. `DroneSnapshot.frontiers` is retained as a representative-coordinate
compatibility view and now carries the corresponding stable cluster IDs.

## Phase 4 - Persistent Intent and Route Execution

### Goal

Latch goals and execute cached strategic routes without replanning each segment.

### Work

- [x] Add `MovementMode`, `NavigationIntent`, watchdog state, recent visits,
      and explicit transition reasons to synchronized drone runtime state.
- [x] Replace boolean `node_found` results with `MovementOutcome`.
- [x] Run global cluster selection only when no valid intent exists or after
      `REACHED`, `INVALIDATED`, `STALLED`, `RESERVATION_LOST`, or `HOME`.
- [x] Evaluate at most 32 candidates: the 16 highest-gain and 16 nearest/diverse
      unreserved candidates.
- [x] Exclude unreachable, revision-blacklisted, and other-drone-reserved
      clusters before scoring.
- [x] Normalize gain, route cost, revisit, and stall values, then score with:
      `3*gain - cost - 1.5*revisit - 2*stall - 2.5*premature_switch`.
- [x] Latch the chosen cluster, gateway, assignment token, route IDs, revisions,
      and cursors.
- [x] Execute at most `drone.step` pixels from the stored oriented edge
      polyline each movement tick.
- [x] Permit bounded belief-only A* only for pose-to-graph and
      gateway-to-frontier connectors.
- [x] On topology change, verify the remaining IDs. Continue across unrelated
      changes; explicitly invalidate and replan if a remaining edge retired.
- [x] On requester SLAM change, validate only remaining belief-corridor edges.
      Travelled edges remain mission-global trusted evidence.
- [x] Key unreachable blacklists by drone, cluster, SLAM revision, topology
      revision, reason, and affected bounds. Unrelated SLAM changes and time do
      not clear them.
- [x] Route `HOME` through the same graph and belief-connector machinery.
      Remove drone dependence on the ground-truth pathfinding service.

### Watchdog and modes

- [x] Count progress only from actual information gain or monotonic remaining
      route-cost reduction.
- [x] Track edge/coarse-cell revisits and immediate reverse-edge transitions in
      a 32-action window.
- [x] Enter `RECOVERY` after 10 seconds or 64 travelled pixels without progress,
      a revisit ratio of at least 0.60, or the second A-B-A reversal.
- [x] In `RECOVERY`, release/penalize the goal and follow the least-revisited
      safe adjacent edge or previous safe prefix before replanning.
- [x] Enter `SCAN` at the gateway and perform six 60-degree headings. Wait for
      the sensor scan sequence to advance after every rotation.
- [x] Retire/penalize a cluster after a completed zero-gain scan. If the scan
      reveals a continuing frontier, keep the stable cluster reservation and
      replan to its advanced gateway.

### Tests

- [x] Goal, assignment, route IDs, and cursors persist across every prefix.
- [x] Persistent edge execution performs no A* calls.
- [x] Only local connectors can invoke bounded A*.
- [x] An unrelated topology revision does not abandon the goal.
- [x] Split/retired route edges produce explicit invalidation and one replan.
- [x] Route progress prevents false stalls during information-neutral travel.
- [x] Movement success without gain/progress does not reset the watchdog.
- [x] The second A-B-A reversal causes recovery or retirement.
- [x] Scan completion, zero-gain retirement, blacklist expiry, and HOME routing
      are deterministic.

### Exit criteria

- [x] There are no unexplained target switches.
- [x] Route planning occurs only for a new/invalidated goal or route.
- [x] Stored edge paths are the normal travel mechanism.
- [x] Runtime-state, movement, routing, and mission interaction suites pass.

Phase 4 complete (2026-07-15). Strategic goals now persist in synchronized
runtime state with assignment/gateway identity, route revisions, exact oriented
segment paths, and durable cursors. Normal travel consumes at most one
`drone.step` prefix without invoking mission A*, while unrelated topology
changes are tolerated and retired remaining edges or invalid belief corridors
produce explicit invalidation outcomes.

Selection is bounded and deterministic, reservations and belief-scoped
unreachable records are filtered before normalized scoring, and HOME uses the
same belief-only graph machinery. The watchdog distinguishes route progress
from mere motion, tracks revisits and reversals, and falls back through a safe
travelled prefix in RECOVERY. Gateway arrival enters a sensor-sequence-gated
six-heading SCAN; zero-gain clusters retire, while continuing clusters retain
their reservation for the next route. Compatibility coordinate/frontier and
truth-tested movement APIs remain available at simulator boundaries.

## Phase 5 - Goal-Conditioned Local MCTS

### Goal

Turn MCTS into a small, deadline-safe controller instead of an expensive global
frontier policy.

### Work

- [x] Keep deterministic `FOLLOW_EDGE` as the normal valid-route fast path.
- [x] Restrict MCTS roots to:
  - `FOLLOW_EDGE`;
  - a one-step 15-degree left deviation;
  - a one-step 15-degree right deviation;
  - `ROTATE_SCAN`; and
  - `RECOVERY`.
- [x] Build only a bounded local belief window around the pose and active route.
- [x] Include stable goal ID, route/cursor progress, recent visits, previous
      primitive, and stall state in the search state.
- [x] Start the deadline before local preprocessing and reserve 30% of the
      configured budget for scheduler margin, diagnostics, and fallback.
- [x] Check the deadline before every root evaluation, expansion, simulation
      depth, and predicted ray cell.
- [x] Evaluate every generated root once before applying UCT.
- [x] If complete root coverage cannot finish, perform zero forced iterations
      and return the deterministic safe primitive with an overrun-stage trace.
- [x] Normalize reward components and use:
      `2*route_progress + 3*information_gain - 1.5*revisit`
      `- 2*oscillation - 2.5*target_switch - 0.25*turn`
      `- 0.25*time_energy - 4*collision_risk`.
- [x] Treat the first unknown/low-confidence cell as predicted gain and stop the
      sensor ray there. Confident occupied cells stop it without gain.
- [x] Keep `policy=frontier` as deterministic local control over the same
      cluster/intent substrate. `policy=mcts` enables the bounded controller.

### Tests

- [x] A fake clock proves preprocessing is included in the 40 ms budget.
- [x] No search iteration is forced after the deadline.
- [x] Every root primitive receives one evaluation before UCT.
- [x] Deadline checks occur inside simulation and ray loops.
- [x] Unknown cells occlude cells behind them.
- [x] Reward ordering favors goal progress and information while penalizing
      revisits, reversals, switches, turns, time, and risk.
- [x] Fixed seed and state produce deterministic decisions and diagnostics.

### Exit criteria

- [x] MCTS performs no full-map frontier work.
- [x] Root branching consists only of bounded goal-conditioned primitives.
- [x] The MCTS, exploration-policy, and movement suites pass.

Phase 5 complete (2026-07-15). Normal valid-route travel remains the exact
stored-polyline fast path. Genuine competing deviations use a goal-conditioned
controller whose fixed primitive vocabulary, route/cursor state, predicted
rays, and SLAM copy are bounded to the active local window. A scan has only one
legal `ROTATE_SCAN` action and takes the equivalent direct sensor-gated
60-degree fast path. Recovery likewise has one mode-safe action and follows the
exact polyline already stored in its intent; neither single-choice mode copies
SLAM or runs MCTS. The 40 ms deadline begins before any real MCTS copy, reserves
30% for scheduler margin, fallback, and diagnostics, and never backpropagates
an incomplete root or rollout. Incomplete root coverage returns the
deterministic mode-safe primitive.

Reward terms use the locked normalized weights and mirror runtime route-progress,
revisit, reversal, target-switch, energy, and collision semantics. Predicted
gain stops at the first unknown or low-confidence cell; confident occupancy
occludes without gain. Frontier mode uses the same stable intent and canonical
oriented route geometry without MCTS or global frontier work during `decide()`.
The complete suite passes 386 tests with one intentional expected failure; a
60-decision synthetic 40 ms workload completed root coverage on every search
with 36.061 ms median and 36.099 ms maximum. `compileall` and diff checks pass.
No production mission replay was run in this phase.

## Phase 6 - Telemetry, Rendering, and Legacy Removal

### Work

- [x] Emit stable cluster, gateway, node, edge, route, assignment, and intent
      IDs in all relevant trace events.
- [x] Emit route-cache hits, topology/knowledge revisions, cursor progress,
      remaining cost, replan reason, watchdog inputs, mode transitions, actual
      gain, and revisit ratio.
- [x] Preserve legacy trace fields until the new analyzer and one live trace
      validate the replacement schema.
- [x] Update the waypoint renderer for strategic roles and batched revisions.
- [x] Remove breadcrumb sampling, raw-frontier fallback, per-segment route
      reconstruction/A*, cave-map policy fields, and obsolete forwarding code.
- [x] Update typed configuration, INI persistence, documentation, and tests.
- [x] Confirm no silent fallback still invokes mission ground-truth drone A*.

### Exit criteria

- [x] The analyzer reports every final acceptance metric from one trace.
- [x] No runtime code path regenerates retired coordinate pseudo-frontiers.
- [x] No compatibility wrapper remains unless it is an intentional runtime API.
- [x] The complete automated suite and compile checks pass.

Phase 6 implementation audit (2026-07-16): the replacement schema and analyzer
are covered by synthetic schema-v3 traces and the 121.885-second live trace
`mission_trace_20260716_013949_346231.jsonl`. The analyzer reported every final
acceptance metric with `replacement_valid=True`, zero missing replacement
fields, zero planner cave-map fields, and zero stable-ID conflicts. Legacy trace
aliases remain intentionally dual-written through the post-fix smoke run.

The same full trace exposed a runtime integration defect outside the telemetry
schema: direct known-free connector execution did not extend the live strategic
trail. All three drones eventually moved beyond the 64 px pose connector while
the 128 px recovery anchor was still uncommitted, so global selection repeatedly
reduced 90-155 advertised clusters to zero `no_start_connector` candidates.
Executed ephemeral connectors now extend the trail, and route lookup can use the
exact reversed uncommitted travelled tail without persisting a pose node. The
complete suite now passes 374 tests; `compileall` and the diff check pass. A new
live smoke trace is still required to confirm the runtime stall is absent.

The 210.979-second follow-up trace
`mission_trace_20260716_015941_992163.jsonl` confirmed that the zero-candidate
stall was removed, but exposed a watchdog lifecycle defect. After the second
A-B-A trigger, the cumulative reversal count was never consumed by recovery;
drone 1 therefore produced 500 immediate reversal recoveries and 661
target-blacklist events, while drone 0 repeated reversal recovery during
homing. Those one-step route/recovery crossings also inflated the graph to 94
nodes, with 72.3% of nodes less than 8 px apart. Starting recovery now resets
the bounded watchdog epoch, and a recovery replacement is no longer classified
as a target-specific route failure. Two characterization tests lock both
behaviors. The complete suite now passes 376 tests; `compileall` and the diff
check pass. A post-fix live trace remains required to verify movement and graph
density under production scheduling.

The 3320.208-second trace `mission_trace_20260716_021113_278442.jsonl` then ran
without a fatal runtime error but did not formally complete: its final frame
still had active drones and hundreds of fragmented frontier records. It covered
72,549.90 px while growing the graph to 683 nodes and 854 active edges. Of
24,287 local searches, 24,167 selected the sole `ROTATE_SCAN` root; repeated UCT
rollouts consumed about 36 ms even though no competing action existed. Across
18,294 sensor scans, 60.2-67.2% produced zero newly-known cells, and identical
poses were scanned up to 25 times. Successful target selections had median
expected gain 1 while simultaneously visible clusters had median maximum gain
of 2,420-3,183. All 4,693 failed route candidates were blacklisted, including
candidates skipped only because the one-per-tick bridge-attempt budget had
already been consumed. These measurements explain the observed rotations,
short corrective motion, low-gain preference, and near-completion delay.

The live-trace follow-up fixes are characterization-tested. A single-root local
search now stops after its mandatory full root evaluation; a 100-decision
synthetic scan workload measured 0.433 ms median, 0.517 ms p99, and 0.618 ms
maximum with zero UCT iterations. Per-tick bridge-budget exhaustion is treated
as a transient skip and no longer creates target cooldowns or belief
blacklists. A valid graph route is simplified to one exact ephemeral
requester-belief LOS polyline only when it is strictly shorter than the graph
route; missing connectors and disconnected components retain their previous
failure semantics, and blocked/unknown LOS retains the stored graph route.
Executed LOS prefixes still extend the strategic trail without creating pose
breadcrumbs or changing stable graph IDs.

Gateway scans now retain the arrival heading instead of resetting to absolute
zero. When sensing advanced after cluster selection, arrival performs one fresh
belief-only frontier refresh and skips the six-heading rotation if that cluster
was already resolved en route. Selected-drone occupancy changes now invalidate
the cached SLAM view even when its version was rendered previously, and the
shared waypoint overlay is hidden in per-drone mode. SLAM sharing now mirrors
the exact merge rules rather than a stride/ratio prefilter, so isolated late
cells cannot be missed; compact pair events distinguish exchange, no-delta,
no-LOS, and cooldown outcomes. The focused suites and complete 385-test suite
pass, as do `compileall` and `git diff --check`. A new bounded production trace
is required to measure the resulting physical scan count, selection gain,
sharing outcomes, graph density, route latency, and mission completion time.

The 2739.106-second trace `mission_trace_20260716_162608_475035.jsonl`
confirmed that drone 2 could retire cluster 1657 and complete, while drones 0
and 1 retained runtime IDs for peer-retired clusters and never entered a
coherent exhaustion epoch. It also measured 816 nodes, 1029 active edges,
65.1% near-neighbor density, 178.26 ms route p95, zero repeated cache hits, and
a 385.74 px route for an 8.54 px target.

The acceptance follow-up is now characterization-tested. Exploration progress
is exposed-wall occupied-SLAM coverage (outer cave, pillars, and internal
walls), while terrain roughness is rover-routing/heatmap data only. A
team-level coordinator serializes frontier refresh/retirement with the
exhaustion handshake, reconciles every runtime snapshot, releases stale
assignments, resets reports when new work appears, and starts coordinated
homing only after all drones confirm an empty canonical registry. Stable
clusters retain their IDs and representatives while each drone selects a
locally known-free component cell.

Close non-LOS goals can use one 12 ms/4,000-expansion belief-only A* during
route construction; its exact polyline is stored and execution remains A*-free.
Route trees are keyed by requester-scoped belief-edge validity so unrelated
SLAM revisions can reuse them safely. Runtime routing no longer creates a
protected gateway for an ephemeral goal connector. Retired clusters shed
orphan corridors/gateways, and maintenance batch-collapses safe inactive
degree-two turn/junction nodes while protecting active route IDs, HOME,
CHOKEPOINT, RECOVERY_ANCHOR, and active FRONTIER_GATEWAY nodes. The complete
403-test suite passes, as do `compileall` and `git diff --check`. A bounded live
validation trace remains required.

## Phase 7 - Verification and Acceptance

### Automated verification

- [x] Run focused graph, frontier, runtime-state, movement, MCTS, sharing,
      rendering, configuration, and trace-analysis suites after their phases.
- [x] Run the complete suite (426 tests):

  ```powershell
  python -m unittest discover -s tests -v
  ```

- [x] Compile all Python modules:

  ```powershell
  python -m compileall -q .
  ```

- [x] Inspect `git diff --check` and confirm all pre-existing worktree changes
      remain present.

### Live comparison

The next validation run is bounded to 600 simulated seconds using the current
seed-19/MEDIUM/three-drone MCTS defaults, stopping early on team completion.
Compare it directly with `mission_trace_20260716_162608_475035.jsonl`. After
that regression is understood, retain the seed-5/SMALL normalized 0-600 second
run for the original legacy-baseline gates.

The first replacement run,
`mission_trace_20260716_193327_716793.jsonl`, shut down cleanly after 613.276
seconds and mapped 4,087/13,180 exposed wall pixels (31.0%). It exposed one
post-selection state-machine gap: drones 1 and 2 retained non-empty canonical
views while per-drone accessibility filtering produced no actionable waypoint.
They emitted 2,602 stalled frontier actions without entering the team wait
path. Canonical reconciliation itself was active (419 reconciliation events,
141 retirements), so this was not the stale-retired-ID failure from the prior
trace.

The follow-up patch characterizes and handles that state explicitly. A drone
performs one bounded local frontier refresh, re-scores the refreshed stable IDs,
and emits `no_actionable_frontier` plus
`drone_waiting_for_team_frontier` if no candidate remains. Unchanged local SLAM,
runtime frontier IDs, and canonical active IDs suppress subsequent worker-tick
selection until the 1.5-second retry bound; any shared-SLAM, local-view, team
registry, or coordinated-homing change wakes it immediately. A genuinely empty
local view uses the same suppression around the existing team-exhaustion
handshake, and canonical work still prevents premature homing.

The trace analyzer now clears its active-goal epoch on an explicit transition
to no current intent and falls back from a null `replan_reason` to the canonical
transition reason. Replaying the same trace therefore reports 523 explicit goal
changes and zero unexplained changes, instead of the prior false-positive
140/165 result.

Request another seed-19/MEDIUM/three-drone run bounded to 600 seconds. In
addition to the hard gates below, verify that a non-empty runtime frontier view
with zero actionable candidates produces one bounded refresh and a
`drone_waiting_for_team_frontier` event, not repeated per-tick
`drone_frontier_targets_exhausted`/`stalled` actions. Drones must wake on shared
SLAM or canonical-registry changes, and coordinated homing must still wait for
an empty canonical registry plus every participant's exhaustion report.

The post-fix replacement run,
`mission_trace_20260716_200849_857200.jsonl`, ran for 620.687 wall-clock seconds
with the normalized 0-600-second acceptance window intact and shut down cleanly.
The repaired behavior is validated: each drone emitted exactly one
`drone_frontier_targets_exhausted` event, compared with 1,759 and 843 for the two
stalled drones in the preceding run. No `no_actionable_frontier` wait was needed
in this run, all three drones continued translating or gaining SLAM cells near
shutdown, and there were no premature team-exhaustion or homing events. Exposed
wall mapping reached 6,384/13,180 pixels (48.4%).

Several gates improved and now pass: average graph density is 1.41 nodes per
occupied 32x32 bucket, the maximum bucket contains five nodes, degree-two nodes
are 10.4%, repeated-route cache hits are 100%, A-B-A reversals are 0%, all 13
second reversals recover or retire, persistent-route execution performs zero
A*, unchanged valid intents perform zero replans, sensor efficiency is
32.348832 newly-known cells/px, and the planner/schema audits pass. The corrected
goal audit reports 634 changes and zero unexplained changes.

Phase 7 remains open on three explicit hard gates:

- 46.2% of active nodes have a neighbor within 8 px; the limit is below 15%.
  The surviving pairs are mostly adjacent TURN/JUNCTION or JUNCTION/JUNCTION
  nodes, including nodes one pixel from HOME and RECOVERY_ANCHOR nodes.
- Route lookup p95 is 13.65 ms; the limit is 5 ms. The median is only 0.66 ms,
  but 80/882 calls exceed 5 ms and contention/outliers reach 188.88 ms.
- Local MCTS p99 passes at 32.93 ms, but the 64.09 ms maximum exceeds the 50 ms
  limit. Four searches exceed 50 ms, all incomplete safe fallbacks; their
  dominant overrun stage is preprocessing.

The three blockers now have characterization-tested structural fixes. Graph
maintenance contracts a sub-8-pixel pair only when the pair is joined by a
stored edge and one endpoint is an inactive TURN/JUNCTION. It preserves the
complete incident polylines through the retired position, refuses mixed-source
or mixed-owner rewrites, never joins spatially close disconnected nodes, and
protects active route IDs, active gateways, and dual strategic anchors.

Route telemetry now separates the time spent inside actual route lookup from
corridor repair and gateway lifecycle work. `route_lookup_elapsed_ms` and
`route_lookup_calls` drive the acceptance p95; the legacy total remains in
`route_elapsed_ms`, with the remainder reported as
`route_repair_elapsed_ms`. Local MCTS now requests its bounded SLAM window with
a non-blocking snapshot operation. Contention with a sensor writer returns the
existing safe fallback immediately with `overrun_stage=preprocessing_lock`
instead of waiting inside preprocessing.

Request one final seed-19/MEDIUM/three-drone run bounded to 600 seconds. The
new trace must retain the repaired frontier behavior and verify all three
remaining gates using the corrected telemetry: neighbor-within-8 below 15%,
lookup-only route p95 at most 5 ms, and local-MCTS p99/maximum at most 42/50 ms
with zero unsafe incomplete searches. Report route repair time separately so a
passing lookup gate cannot conceal expensive corridor construction. Phase 7
closes only after that trace and the original seed-5/SMALL normalized
comparison meet their applicable gates.

The next replacement trace,
`mission_trace_20260716_204417_822951.jsonl`, shut down cleanly after 856.435
seconds; acceptance was evaluated over its normalized first 600 seconds. The
connected-node lifecycle patch passes: 115 active nodes occupy 93 buckets,
average density is 1.24, only 10.4% have a neighbor within 8 px, degree-two
density is 11.3%, and the maximum bucket contains three nodes. The non-blocking
SLAM snapshot also passes: local-MCTS p99 is 34.30 ms, maximum is 44.53 ms, one
busy-writer snapshot produced the intended `preprocessing_lock` fallback, and
there are zero unsafe incomplete searches. One later safe preprocessing
fallback reached 59.04 ms at normalized 853.47 seconds, outside the declared
acceptance window; it remains a scheduling-risk observation rather than being
silently folded into the bounded result.

Lookup-only route p95 remains narrowly over the gate at 6.46 ms. Of 858 route
constructions, 50 exceed 5 ms. Failed close-target connector searches consume
the old 12 ms local-A* budget, and uncached persistent routes build one reverse
tree for every visible goal connector. The follow-up patch reduces the
construction-only connector budget to 4 ms, reports attempted searches even
when no connector is found, and replaces the repeated one-root builds with one
exact multi-source reverse Dijkstra seeded by all goal connectors. Three new
characterization tests cover the budget, failed-attempt telemetry, and
single-tree behavior; the complete 412-test suite passes.

Exploration remained productive but did not complete in the available run.
Exposed-wall coverage reached 6,235/13,180 (47.3%) at normalized 600 seconds
and 9,587/13,180 (72.7%) at shutdown. The normalized window travelled
21,271.92 px and gained 407,675 sensor cells, or 19.164937 cells/px, below the
previous medium trace's 32.348832 despite slightly higher wall coverage. The
formal legacy efficiency comparison still belongs to the pending
seed-5/SMALL normalized run, but this medium-map result is retained as an
exploration-policy warning for the next phase rather than being hidden by the
improved wall count. A final medium-19 trace is required for the revised route
lookup gate before Phase 7 closes.

The route follow-up trace,
`mission_trace_20260716_232629_239873.jsonl`, shut down cleanly after 612.405
seconds and was analyzed over the normalized first 600 seconds. The revised
route construction passes at 4.36 ms p95 with zero persistent-edge execution
A*. Average bucket density (1.37), maximum bucket size (4), degree-two density
(15.5%), goal reasons, reversals, watchdog transitions, and schema boundaries
also pass.

Two edge cases remain open. Raw near-node density is 16.5%. Seven of the eight
reconstructed sub-8-pixel pairs have no direct edge and are separated by
67-186 px of stored graph geometry, so merging them would violate the explicit
wall/parallel-corridor safety rule. Four pairs involve roleless inactive
degree-one travelled tails. Maintenance now retires only those non-transit
leaves and their incident travelled edge, while protecting every role-bearing
node and active node/edge/gateway identity. The sole short stored-edge pair in
the replay was protected by an active route.

Local-MCTS p99/max are 36.02/68.98 ms with zero unsafe incomplete searches.
All six searches over 50 ms are single-root scan decisions descheduled before
preprocessing (`preprocessing_cells=0`); the 34 actual recovery searches have
48.60 ms p99/max. Scan mode now uses its equivalent deterministic +60-degree
sensor-gated action without invoking MCTS or copying SLAM. Recovery MCTS keeps
root-first evaluation but stops optional work at 28 ms, reserving 12 ms of the
40 ms budget for observed scheduling delay, safe fallback, and diagnostics.
Two new graph tests and the revised scan/budget characterization bring the
complete suite to 414 passing tests. One bounded medium-19 validation trace is
still required before Phase 7 can close.

The next trace, `mission_trace_20260716_234914_443907.jsonl`, shut down cleanly
after 888.744 seconds and was again gated on its normalized first 600 seconds.
The graph lifecycle now passes every density gate: 75 nodes in 63 occupied
buckets (1.19 average), 10.7% with a neighbor within 8 px, 17.3% degree-two,
and four nodes in the densest bucket. Route lookup remains accepted at 4.33 ms
p95, with zero persistent-edge A*, zero unchanged-intent replans, 0.9% A-B-A
reversals, every second reversal handled, and no unexplained goal changes.

The scan fast path reduced actual MCTS searches from 6,573 to 21. All remaining
searches were recovery-mode searches. MCTS p99/max still failed at 67.57 ms
because one completed search spent eight optional UCT iterations comparing
`FOLLOW_EDGE` and `RECOVERY` roots with exactly the same target, heading, path,
visit distribution, and mean reward. Every one of the 21 searches showed the
same duplicate-root geometry. Recovery mode now exposes only its mode-safe
`RECOVERY` root, so it stops after the one mandatory evaluation just like the
controller's existing single-root rule. The new characterization brings the
complete suite to 415 passing tests.

Removing scan MCTS did not materially improve frame rate: normalized median
FPS was 2.37, with median render at 264.39 ms and sensors at 144.45 ms. Wall
coverage reached 4,702/13,180 (35.7%) at 600 seconds and 5,317/13,180 (40.3%)
at shutdown; sensor efficiency was 22.572214 cells/px. These results confirm
that the next FPS work belongs in rendering/sensing and that exploration-policy
efficiency remains a separate follow-up. One final bounded validation is still
required for the deduplicated recovery timing before Phase 7 can close.

The latest trace, `mission_trace_20260717_001209_898755.jsonl`, shut down
cleanly after 613.708 seconds. Graph and route gates remain healthy: 75 nodes
occupy 59 buckets, average density is 1.27, 9.3% have a neighbor within 8 px,
degree-two density is 14.7%, the maximum bucket contains three nodes, route
lookup p95 is 2.52 ms, persistent execution performs zero A*, and all 616 goal
changes have explicit reasons. All 25 performed MCTS calls were single-choice
recovery evaluations; one was descheduled before preprocessing and returned a
safe fallback after 55.38 ms. Recovery now bypasses belief preprocessing and
follows its already-stored exact intent, as scan mode already does. Actual
multi-choice deviation searches retain the bounded controller.

The trace also confirms the visual exploration-policy diagnosis. Across 738
frontier rebuilds, 62,207 of 96,446 cluster observations had `expected_gain=1`.
Drones 1 and 2 attempted 6,056 routes, of which 5,651 failed, producing 5,510
bridge-budget skips and 210 exhausted-target cycles. These were predominantly
unknown gaps between sparse vision samples, because canonical extraction still
treated every confident-free cell beside any unknown cell as a frontier and
the global score gave that raw unknown count triple weight. Normalized wall
coverage reached only 4,253/13,180 (32.3%), while sensor efficiency was
23.404855 newly-known cells/px.

The follow-up is characterization-tested at the frontier-definition boundary.
Generic discovery cells now require at least four unknown cells in their
clipped 3x3 neighborhood. An isolated unknown remains actionable only when it
touches both confident free space and an observed occupied surface, making it
a plausible missing wall pixel or wall continuation. Components carry separate
`wall_gain` and wall-adjacent free cells; each requester selects an accessible
wall cell without changing the stable cluster ID. Deterministic global scoring
uses a strict reachable/unreserved wall-continuation tier, with coherent open
unknown regions retained as the bounded fallback needed to discover another
wall. The true cave map remains telemetry-only. The threshold is typed and
persisted as `frontier.minimum_unknown_support=4`, and reservation traces now
include `wall_gain` and `wall_directed`.

The complete 421-test suite passes. One bounded seed-19/MEDIUM validation trace
is required before Phase 7 closes. It must preserve the graph/route/intent
gates, show a major reduction in fragmented clusters, failed routes, bridge
budget skips, and exhausted-target churn, and demonstrate wall-directed
reservations whenever an accessible wall-continuation tier exists. Report wall
coverage and newly-known cells/px against this trace. Performed MCTS should now
occur only for genuine competing deviations; deterministic scan/recovery
events must report `performed=false`.

The latest run also establishes the next performance baseline: median FPS is
2.85 and the last normalized frame is 1.95 FPS. Median frame time is 350.35 ms,
dominated by render (225.80 ms median) and sensors (121.81 ms median); display
is 4.64 ms and normal sharing is 0.32 ms median. Exploration-policy changes
should remain separate from these frame-stage costs so behavioral quality and
throughput can be measured independently. The original post-overhaul
seed-5/SMALL normalized 0-600 comparison also remains pending.

The wall-directed validation trace,
`mission_trace_20260717_003944_592580.jsonl`, preserves the graph, route,
intent, and MCTS safety gates but does not close Phase 7. In the normalized
0-600-second window, one-cell cluster observations fell from 60,259/93,444
(64.5%) to 51,109/102,998 (49.6%). Route calls fell from 6,391 to 959, failed
routes from 5,791 to 104, bridge-budget skips from 5,471 to 46, and exhausted
target cycles from 210 to 44. Lookup p95 remained bounded at 3.28 ms,
persistent execution performed zero A*, all goal changes were explicit, and
no local MCTS search was needed for deterministic stored intents.

Wall-directed selection accounted for 948/963 normalized reservations. Ten
opening fallback reservations occurred while both visible wall continuations
were owned by the other drones. The other five exposed a retained-target
exception: after a scan removed the retained cluster's wall gain, retention
could retry that generic cluster without re-entering the global strict tier.
Retained work now survives only if it remains in the current globally scored
strict candidate set, so it cannot bypass an accessible wall continuation.

The remaining acceptance failure is sensing productivity. Normalized exposed
wall coverage declined from 4,253/13,180 (32.3%) to 3,896/13,180 (29.6%), and
newly-known cells per travelled pixel declined from 23.404855 to 23.003179.
The prior sparse-ray SLAM input left angular holes even though Bresenham filled
the cells along each individual ray.

Vision is now decoupled from terrain sampling. `VisionSensor.scan_cone()`
returns a collision-bounded `VisionScan` with every visible cone cell plus the
existing sparse rays. `SlamMap.update_from_observations()` consumes only the
dense free/occupied cells, and those cells drive exploration gain and mission
completion. Roughness remains independent and samples every second cell on the
sparse rays. A six-heading radius-80 characterization covers all 20,081 cells
in the disk with zero gaps; the vectorized scan averages about 10.2 ms including
the existing 60 ray casts. Another seed-19/MEDIUM/three-drone 0-600 trace is
required before Phase 7 can close, with the same churn, graph, route, intent,
and MCTS gates plus materially improved wall coverage and sensor cells/px. The
complete 426-test suite passes before that live validation.

The dense-vision validation trace,
`mission_trace_20260717_012050_432287.jsonl`, confirms that sparse terrain
sampling no longer creates exploration gaps, but it exposes a separate
retained-wall policy loop. All 711 normalized reservations were wall-directed,
702/711 route calls succeeded, lookup p95 was 4.10 ms, persistent execution
performed zero A*, and deterministic stored intents performed zero MCTS
searches. Route, graph, goal-transition, and watchdog gates therefore remain
healthy. Productivity nevertheless regressed: exposed-wall coverage reached
only 3,401/13,180 (25.8%) and sensor cells/px fell to 12.699141.

The trace reconstructed 544 completed arrival scans. Each used the fixed
six-heading sweep, consuming approximately 55%, 61%, and 56% of the three
drones' normalized windows. Median displacement between consecutive frontier
arrivals was 2 px; 414/541 consecutive arrivals retained the same stable
cluster and 345 were both the same cluster and within 5 px. One-cell frontier
observations also rose to 52,964/85,893 (61.7%). Dense sensing was therefore
working, but each scan exposed a small occluded wall sliver, stable-ID
retention immediately selected it, and requester waypoint selection chose the
nearest wall cell before another full rotation.

Retained-wall control now batches instead of pixel-crawling. Scan baselines
use only `sensor_newly_known_cells` and `sensor_confidence_gain`, excluding
concurrent shared/collision progress. Initial frontier arrivals retain the
six-heading sweep. A productive retained wall must expose a locally known-free
continuation at least `frontier.continuation_min_distance=12.0` px away; the
selected cell maximizes nearby unknown support before distance. An unchanged
closer tip is suppressed for that drone until the canonical cluster geometry
changes, while another drone or genuinely expanded wall can reactivate it.
Accepted continuations center a
`frontier.continuation_scan_headings=3` sweep on locally unknown cells adjacent
to the observed wall. Reservation and continuation traces expose the directed
heading count, retention, suppression reason, local sensor gain, next waypoint,
and displacement. Phase 7 remains open for another normalized
seed-19/MEDIUM/three-drone trace demonstrating reduced scan dwell and same-ID
near-arrival chains plus recovered wall coverage and cells/px. The complete
433-test suite, bytecode compilation, and diff whitespace checks pass before
that live validation.

Hard completion gates:

- [ ] No `travelled` or `known_free` node roles exist.
- [ ] Average strategic nodes per occupied 32x32 bucket is at most 1.5.
- [ ] Fewer than 15% of nodes have another node within 8 px.
- [ ] Fewer than 30% of nodes are degree-two.
- [ ] No 32x32 bucket contains more than eight nodes.
- [ ] Repeated-route cache hit rate is at least 80%.
- [ ] Route lookup p95 is at most 5 ms.
- [ ] Every goal change has an allowed explicit reason.
- [ ] A-B-A reversal rate is below 5%.
- [ ] A second reversal causes recovery or goal retirement.
- [ ] MCTS p99 is at most 42 ms and maximum is at most 50 ms for the
      configured 40 ms budget.
- [ ] Every completed MCTS search evaluates all root primitives before UCT.
- [ ] Persistent-edge execution performs zero A* calls.
- [ ] An unchanged valid intent causes zero route replans.
- [ ] The watchdog changes mode by the next planning tick after its threshold.
- [ ] Newly-known sensor cells per travelled pixel are at least 31.656968 over
      the normalized 0-600 second window, twice the 15.828484 instrumented
      legacy baseline. Report confidence gain per pixel separately.
- [ ] None of the planner-facing types or calls contains the cave map.

## Risk Checklist

- [ ] Spatial proximity never bypasses travelled-intersection or known-free LOS
      validation.
- [ ] Edge splitting retires old IDs, invalidates route caches, and notifies
      every active route explicitly.
- [ ] Chain collapse preserves path orientation, geometry, and cost.
- [ ] Requester-scoped corridors are never accepted from another drone without
      current belief validation.
- [ ] A gateway referenced by an active route is protected from compaction.
- [ ] Cluster hysteresis prevents identity churn and tombstones prevent
      regeneration after zero-gain arrival.
- [ ] Assignment and graph locks are never held while calling into SLAM or each
      other; planning uses detached snapshots and short atomic commits.
- [ ] Long valid travel is not misclassified as stalled merely because SLAM is
      unchanged.
- [ ] No full-map preprocessing occurs inside the local MCTS deadline.

## Completion Definition

The overhaul is complete only when every phase exit criterion is satisfied,
the complete automated suite and compile checks pass, the dirty worktree has
been preserved, and the 600-second live trace meets every hard acceptance gate.
