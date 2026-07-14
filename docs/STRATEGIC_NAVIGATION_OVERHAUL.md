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

- 1,931 waypoint nodes and 2,322 edges;
- 75.7% of nodes have a neighbor within 8 px;
- 80.1% of nodes are degree-two chain nodes;
- 1,551 route calls cover 233 unique drone-target pairs;
- 62.0% of decisions are zero-reward frontier fallbacks;
- 85.4% of MCTS searches perform at most one iteration;
- median MCTS decision time is 81.2 ms with a 40 ms budget;
- 42.1% of decisions following a waypoint segment switch targets; and
- d1's late A-B-A reversal rate reaches 80.3%.

Repository guardrails:

- The current baseline suite contains 280 passing tests.
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
   invoked only for deviations, scanning, and recovery.
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
  - stable ID, frontier cells/bounds, representative, and expected gain;
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

- [ ] Refactor `tools/analyze_runtime_trace.py` around structured metrics while
      preserving `summarize()` as the CLI formatter.
- [ ] Add metrics for waypoint density, target retention, route abandonment,
      A-B-A reversals, route-cache use, planner timing, and information gain
      per travelled pixel.
- [ ] Add synthetic or distilled trace fixtures. Do not commit the 9.94 MB
      production trace.
- [ ] Trace completed sensor sequence, newly-known cells, confidence gain, and
      travelled distance without changing movement behavior.
- [ ] Add belief-only characterization tests showing that identical SLAM and
      terrain beliefs produce identical decisions under different cave maps.
- [ ] Add failing/characterizing tests for target retention, route cursor
      persistence, zero-gain frontier regeneration, and budget overruns.
- [ ] Run an instrumented seed-5/SMALL/three-drone legacy baseline so actual
      information gain per distance is available for the final comparison.

### Exit criteria

- [ ] Existing trace metrics reproduce the audited density, switching,
      reversal, routing, and MCTS figures.
- [ ] The analyzer handles both legacy and replacement event schemas.
- [ ] All pre-existing tests plus the non-behavioral instrumentation tests pass.

## Phase 1 - Strategic Graph Core

### Goal

Replace coordinate identity and route-local graph reconstruction while keeping
the current graph façade usable.

### Work

- [ ] Replace coordinate-keyed storage with monotonic ID-keyed node and edge
      dictionaries.
- [ ] Add persistent adjacency and a node spatial hash. Add an edge-polyline
      cell index for travelled-intersection discovery.
- [ ] Treat spatial buckets only as candidate sources. Validate every merge by
      exact travelled intersection or requester-known-free LOS.
- [ ] Add graph transactions so one logical mutation produces one topology
      revision and one `GraphDelta`.
- [ ] Maintain connected components incrementally on additions and rebuild
      them after split/removal batches.
- [ ] Implement atomic edge splitting at a real travelled junction. Preserve
      the original oriented polyline exactly in the replacement edges.
- [ ] Collapse only unprotected, geometrically redundant degree-two nodes by
      concatenating their polylines. Never collapse active-route nodes.
- [ ] Add reverse shortest-path trees rooted at gateways with a 64-entry LRU.
- [ ] Key route trees by gateway, topology revision, requester, and requester
      edge-knowledge revision. Do not use Floyd-Warshall.
- [ ] Reuse cached routes optimistically across ordinary SLAM changes and
      revalidate only requester-scoped edges on the selected route.
- [ ] Update graph snapshots and rendering to consume IDs and roles while
      retaining coordinate/source compatibility properties.

### Tests

- [ ] IDs are stable, monotonic, and never reused.
- [ ] Wall-separated nodes in one hash bucket never merge.
- [ ] Travelled intersections split the correct edge in one revision.
- [ ] Split/collapse operations preserve exact physical polylines and cost.
- [ ] Component rejection happens before shortest-path search.
- [ ] Repeated unchanged routes build one tree and then hit the cache.
- [ ] Topology and requester-validity changes invalidate only the appropriate
      caches.
- [ ] Concurrent insertion, routing, splitting, and snapshots remain coherent.

### Exit criteria

- [ ] No route call rebuilds the full persistent adjacency.
- [ ] Repeated route lookups at unchanged revisions are cache hits.
- [ ] Renderer rebuilds occur once per committed graph revision.
- [ ] The focused graph, renderer, and configuration suites pass.

## Phase 2 - Strategic Trail Ingestion

### Goal

Stop creating pose breadcrumbs and fixed-interval bridge nodes.

### Work

- [ ] Replace `_waypoint_pending_path` sampling with a per-drone trail
      accumulator.
- [ ] Keep the current pose and uncommitted tail ephemeral. Never force-flush a
      pose node before routing.
- [ ] Promote a `TURN` only after a heading change of at least 45 degrees with
      at least 24 px of confirmed travel on both sides.
- [ ] Promote a `JUNCTION` when a travelled trail actually intersects an
      existing edge, splitting that edge atomically.
- [ ] Promote a `CHOKEPOINT` only at a confirmed local minimum of SLAM-known
      free-space clearance with the configured narrow/wide shoulders.
- [ ] Add a `RECOVERY_ANCHOR` only after 128 px without another strategic node.
- [ ] Store a complete travelled section between strategic endpoints as one
      edge polyline.
- [ ] Store a complete bounded belief corridor as one edge polyline with no
      interior `known_free` nodes.
- [ ] Consolidate overlapping trails through intersection/splitting rather than
      independent per-drone sampling.
- [ ] Keep HOME permanently protected and allow nodes to acquire additional
      roles without changing ID.

### Tests

- [ ] A straight path below the recovery interval creates only its strategic
      endpoints and one edge.
- [ ] A long straight path creates only coarse recovery anchors.
- [ ] A curved path preserves its full shape while promoting only confirmed
      turns.
- [ ] Replaying the same or overlapping trail is topology-revision stable.
- [ ] A near path across a wall remains separate.
- [ ] A corridor bridge creates one scoped edge rather than a 32 px chain.
- [ ] Far-route attempts do not add arbitrary current-pose nodes.

### Exit criteria

- [ ] Node roles no longer include `travelled` or `known_free`.
- [ ] Fixed-spacing and force-flush tests have been replaced by strategic-graph
      invariants.
- [ ] The focused graph and movement suites pass.

## Phase 3 - Stable Frontier Clusters and Reservations

### Goal

Create one belief-only frontier pipeline with persistent identity and explicit
multi-drone coordination.

### Work

- [ ] Build one canonical mask from SLAM occupancy/confidence: confidently
      free, confidently occupied, and unknown/low-confidence.
- [ ] Use terrain confidence only when estimating value; never use it to infer
      traversability or true floor geometry.
- [ ] Cache the frontier mask and update dirty regions plus the required halo.
      Use a full rebuild only initially or after a large shared-map update.
- [ ] Extract connected frontier components once per relevant SLAM refresh.
      MCTS and persisted state must consume these same clusters.
- [ ] Match a refreshed component to an existing ID when cell overlap is at
      least 25%, or its representative is within 32 px and expanded bounds
      overlap.
- [ ] Retain a temporarily missing cluster for three refreshes.
- [ ] Tombstone retired/zero-gain geometry so an unchanged pseudo-frontier does
      not receive a new ID on the next decision.
- [ ] Add a shared registry that canonicalizes independently observed clusters
      while exposing each cluster only to drones in its `known_by` set.
- [ ] Transfer cluster knowledge explicitly during proximity sharing instead
      of sharing raw coordinate lists as a side effect of terrain changes.
- [ ] Add an atomic assignment registry. One cluster or consolidated gateway
      cannot be reserved by two drones simultaneously.
- [ ] Create a gateway lazily when a cluster is reserved. Reuse an existing
      gateway only for the same consolidated cluster with valid LOS and at
      least the configured separation policy.
- [ ] Protect gateway IDs referenced by active intents. Retirement must produce
      an explicit lifecycle/topology event rather than silent disappearance.

### Tests

- [ ] Different cave maps with identical beliefs produce identical masks,
      clusters, scores, and goals.
- [ ] Dirty-region and full frontier extraction produce equivalent results.
- [ ] Cluster IDs survive representative movement and brief disappearance.
- [ ] Reached/retired zero-gain clusters do not regenerate unchanged.
- [ ] Two independent observations of the same geometry receive one canonical
      ID without leaking the geometry to an uninformed drone.
- [ ] Reservation races have exactly one winner and release deterministically.
- [ ] Shared cluster knowledge follows the communication boundary.

### Exit criteria

- [ ] No policy performs whole-map frontier extraction during `decide()`.
- [ ] Runtime state no longer treats raw coordinates as authoritative frontier
      identity.
- [ ] Root and persisted frontiers cannot diverge.
- [ ] Frontier, sharing, and concurrency suites pass.

## Phase 4 - Persistent Intent and Route Execution

### Goal

Latch goals and execute cached strategic routes without replanning each segment.

### Work

- [ ] Add `MovementMode`, `NavigationIntent`, watchdog state, recent visits,
      and explicit transition reasons to synchronized drone runtime state.
- [ ] Replace boolean `node_found` results with `MovementOutcome`.
- [ ] Run global cluster selection only when no valid intent exists or after
      `REACHED`, `INVALIDATED`, `STALLED`, `RESERVATION_LOST`, or `HOME`.
- [ ] Evaluate at most 32 candidates: the 16 highest-gain and 16 nearest/diverse
      unreserved candidates.
- [ ] Exclude unreachable, revision-blacklisted, and other-drone-reserved
      clusters before scoring.
- [ ] Normalize gain, route cost, revisit, and stall values, then score with:
      `3*gain - cost - 1.5*revisit - 2*stall - 2.5*premature_switch`.
- [ ] Latch the chosen cluster, gateway, assignment token, route IDs, revisions,
      and cursors.
- [ ] Execute at most `drone.step` pixels from the stored oriented edge
      polyline each movement tick.
- [ ] Permit bounded belief-only A* only for pose-to-graph and
      gateway-to-frontier connectors.
- [ ] On topology change, verify the remaining IDs. Continue across unrelated
      changes; explicitly invalidate and replan if a remaining edge retired.
- [ ] On requester SLAM change, validate only remaining belief-corridor edges.
      Travelled edges remain mission-global trusted evidence.
- [ ] Key unreachable blacklists by drone, cluster, SLAM revision, topology
      revision, reason, and affected bounds. Unrelated SLAM changes and time do
      not clear them.
- [ ] Route `HOME` through the same graph and belief-connector machinery.
      Remove drone dependence on the ground-truth pathfinding service.

### Watchdog and modes

- [ ] Count progress only from actual information gain or monotonic remaining
      route-cost reduction.
- [ ] Track edge/coarse-cell revisits and immediate reverse-edge transitions in
      a 32-action window.
- [ ] Enter `RECOVERY` after 10 seconds or 64 travelled pixels without progress,
      a revisit ratio of at least 0.60, or the second A-B-A reversal.
- [ ] In `RECOVERY`, release/penalize the goal and follow the least-revisited
      safe adjacent edge or previous safe prefix before replanning.
- [ ] Enter `SCAN` at the gateway and perform six 60-degree headings. Wait for
      the sensor scan sequence to advance after every rotation.
- [ ] Retire/penalize a cluster after a completed zero-gain scan. If the scan
      reveals a continuing frontier, keep the stable cluster reservation and
      replan to its advanced gateway.

### Tests

- [ ] Goal, assignment, route IDs, and cursors persist across every prefix.
- [ ] Persistent edge execution performs no A* calls.
- [ ] Only local connectors can invoke bounded A*.
- [ ] An unrelated topology revision does not abandon the goal.
- [ ] Split/retired route edges produce explicit invalidation and one replan.
- [ ] Route progress prevents false stalls during information-neutral travel.
- [ ] Movement success without gain/progress does not reset the watchdog.
- [ ] The second A-B-A reversal causes recovery or retirement.
- [ ] Scan completion, zero-gain retirement, blacklist expiry, and HOME routing
      are deterministic.

### Exit criteria

- [ ] There are no unexplained target switches.
- [ ] Route planning occurs only for a new/invalidated goal or route.
- [ ] Stored edge paths are the normal travel mechanism.
- [ ] Runtime-state, movement, routing, and mission interaction suites pass.

## Phase 5 - Goal-Conditioned Local MCTS

### Goal

Turn MCTS into a small, deadline-safe controller instead of an expensive global
frontier policy.

### Work

- [ ] Keep deterministic `FOLLOW_EDGE` as the normal valid-route fast path.
- [ ] Restrict MCTS roots to:
  - `FOLLOW_EDGE`;
  - a one-step 15-degree left deviation;
  - a one-step 15-degree right deviation;
  - `ROTATE_SCAN`; and
  - `RECOVERY`.
- [ ] Build only a bounded local belief window around the pose and active route.
- [ ] Include stable goal ID, route/cursor progress, recent visits, previous
      primitive, and stall state in the search state.
- [ ] Start the deadline before local preprocessing and reserve 10% of the
      configured budget for diagnostics and fallback.
- [ ] Check the deadline before every root evaluation, expansion, simulation
      depth, and predicted ray cell.
- [ ] Evaluate every generated root once before applying UCT.
- [ ] If complete root coverage cannot finish, perform zero forced iterations
      and return the deterministic safe primitive with an overrun-stage trace.
- [ ] Normalize reward components and use:
      `2*route_progress + 3*information_gain - 1.5*revisit`
      `- 2*oscillation - 2.5*target_switch - 0.25*turn`
      `- 0.25*time_energy - 4*collision_risk`.
- [ ] Treat the first unknown/low-confidence cell as predicted gain and stop the
      sensor ray there. Confident occupied cells stop it without gain.
- [ ] Keep `policy=frontier` as deterministic local control over the same
      cluster/intent substrate. `policy=mcts` enables the bounded controller.

### Tests

- [ ] A fake clock proves preprocessing is included in the 40 ms budget.
- [ ] No search iteration is forced after the deadline.
- [ ] Every root primitive receives one evaluation before UCT.
- [ ] Deadline checks occur inside simulation and ray loops.
- [ ] Unknown cells occlude cells behind them.
- [ ] Reward ordering favors goal progress and information while penalizing
      revisits, reversals, switches, turns, time, and risk.
- [ ] Fixed seed and state produce deterministic decisions and diagnostics.

### Exit criteria

- [ ] MCTS performs no full-map frontier work.
- [ ] Root branching consists only of bounded goal-conditioned primitives.
- [ ] The MCTS, exploration-policy, and movement suites pass.

## Phase 6 - Telemetry, Rendering, and Legacy Removal

### Work

- [ ] Emit stable cluster, gateway, node, edge, route, assignment, and intent
      IDs in all relevant trace events.
- [ ] Emit route-cache hits, topology/knowledge revisions, cursor progress,
      remaining cost, replan reason, watchdog inputs, mode transitions, actual
      gain, and revisit ratio.
- [ ] Preserve legacy trace fields until the new analyzer and one live trace
      validate the replacement schema.
- [ ] Update the waypoint renderer for strategic roles and batched revisions.
- [ ] Remove breadcrumb sampling, raw-frontier fallback, per-segment route
      reconstruction/A*, cave-map policy fields, and obsolete forwarding code.
- [ ] Update typed configuration, INI persistence, documentation, and tests.
- [ ] Confirm no silent fallback still invokes mission ground-truth drone A*.

### Exit criteria

- [ ] The analyzer reports every final acceptance metric from one trace.
- [ ] No runtime code path regenerates retired coordinate pseudo-frontiers.
- [ ] No compatibility wrapper remains unless it is an intentional runtime API.
- [ ] The complete automated suite and compile checks pass.

## Phase 7 - Verification and Acceptance

### Automated verification

- [ ] Run focused graph, frontier, runtime-state, movement, MCTS, sharing,
      rendering, configuration, and trace-analysis suites after their phases.
- [ ] Run the complete suite:

  ```powershell
  python -m unittest discover -s tests -v
  ```

- [ ] Compile all Python modules:

  ```powershell
  python -m compileall -q .
  ```

- [ ] Inspect `git diff --check` and confirm all pre-existing worktree changes
      remain present.

### Live comparison

Run a 60-second smoke trace, then a 600-second seed-5/SMALL/three-drone trace.
Compare it with the reference and instrumented legacy baseline.

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
- [ ] Sensor information gain per travelled pixel is at least twice the
      instrumented legacy baseline.
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
