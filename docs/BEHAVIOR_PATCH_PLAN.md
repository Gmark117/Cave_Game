# Proposed behavior patches: contact, routing, scanning, and focused frontier batching

Status: Patch A is implemented and live-tested. Patch A2 is implemented and
unit-tested, with live trace validation pending. Patch C is implemented behind
the default-off `off`/`observe`/`active` mode, unit-tested, and paired on seeds
0 and 5; it remains default-off. Patch B and Patch D are implemented behind
separate committed-default-off rollouts and unit-tested; fresh matched live
validation of the highway-integrated behavior remains pending.

## Current evidence and boundaries

- In `mission_trace_20260914_140933_256983.jsonl`, drone 0 made verified LOS/proximity contact with the moving rover at about 357 s while returning with completed DFS work. It kept following an older endpoint. Its report was queued about 31 s later; sampled frames show roughly 1,000 px of additional travel after the contact.
- The same run completed at 608.41 s with 98.75% floor coverage and all 33 claimed tasks reported. It recorded 62 check-in A* movements over 26,583 px and 58 DFS-reposition A* movements over 13,738 px. The trace does not record the status, search expansions, or CPU time of each drone A* request, so the observed long-route local-minimum behavior needs route-level instrumentation.
- The lineage-aware low-gain rule recorded no live deferrals in that run. Keep its validation separate from these behavior changes.
- `mission_trace_20260915_153323_560163.jsonl` completed at 666.69 s with
  98.82% floor coverage and all 42 claimed tasks reported. Patch A handled all
  42 returning-report encounters at the moving or stationary rover. The run
  nevertheless recorded six rendezvous proposals, 24 endpoint fall-forwards,
  and about 26,408 px of check-in travel after 240 s.
- On the final rover leg from `(229, 124)` to `(460, 363)`, drones 1 and 2 each
  flew about 2,600 px of check-in routes. A physical pass by the moving rover
  reset each drone to the old committed endpoint, so both flew to the proposal,
  reversed to the old endpoint, and then flew to the proposal again. From
  540 s onward, six completed component tasks produced only five sensor-known
  cells while task travel was about 4,577 px and check-in travel was about
  10,593 px. Docking addresses the repeated check-in legs; highway routing and
  wall-pocket scans address different parts of the remaining endgame cost.

All patches preserve component-based exploration authority, local DFS claims, strict LOS/proximity communication, rover departure only after universal acknowledgements, and completion after physical report delivery and team quiescence. Display-only floor coverage remains diagnostic; about 98–99% is acceptable when meaningful component work is exhausted. A highway is navigation advice, never a component owner, assignment boundary, or reason to suppress a frontier. Do not add global peer-position or endpoint reads.

## Patch A — Hand in a completed report during a rover encounter

**Trigger.** A drone has entered the `returning` phase after finishing or suspending its assigned work, or already holds a completed report, and a movement step finds the rover within the existing proximity threshold with line of sight. Ordinary task transit and active DFS scanning do not trigger a mid-generation check-in.

**Behavior.** Stop the return route at that physical encounter. Capture the actual return path to the contact point, finalize the report if needed, and queue the normal `exploration_check_in` in the same drone-worker turn. Hold position while the rover worker accepts the report; its existing pending-check-in hold prevents rover departure. If the contact disappears before the check-in is accepted, retain the same report ID and resume rendezvous routing. Never mark the report delivered merely because a proximity share or ACK exchange occurred. A `Reporting` drone makes the same contact-first check before following its remembered endpoint.

**Code seam.** Add a boolean, LOS-checked contact predicate through the movement dependencies, backed by `TerrainSharingService.drone_at_rover`. Allow `_follow_path` to stop specifically for a returning/reporting encounter, then use `_finish_coordination_execution` and the existing queued check-in path. The movement controller receives only a verified contact result, not the rover's global position. Preserve one-time report acceptance and the existing acknowledgement protocol.

**Evidence and tests.** Trace `drone_report_encounter` with route source, report ID, distance, and queue/acceptance outcome. Test a crossing moving rover, blocked LOS, a rover that moves out of range before queueing, duplicate worker delivery, and no interruption during active DFS. Confirm universal ACK still gates rover departure.

This patch stands alone and should be reviewed and validated before changing routing.

USER NOTE: If the drone is trying to report and happens to make contact with a passing rover there should not be any chance for the rover to move away before queueing: the drone should request the rover to stop for a report before moving on to the rendezvous.

## Patch A2 — Dock checked-in drones to the rover

**Purpose and state.** Add an explicit `Docked` physical state for a drone that
has made verified LOS/proximity contact while returning a report, awaiting a
directive, or otherwise performing rover check-in. Docking means a mechanical
attachment to rover 0: the drone is carried at the rover's logical position
and does not plan or propel its own route until released. An active task,
branch follow, DFS reposition, or scan cannot dock merely because it passes the
rover.

**Dock acquisition.** The rover-side mission controller owns dock membership
under the same lock that serializes report stops, queued check-ins, assignment,
and rover movement. Initial acquisition must call the existing
`TerrainSharingService.drone_at_rover`; a remembered endpoint, proposal, relay,
or global agent position is not sufficient.

- A returning/reporting drone keeps Patch A's stop reservation until its
  completed report is finalized and queued. The physical `exploration_check_in`
  then docks it atomically. Once queued and docked, the rover may resume because
  the attachment preserves contact while the rover worker processes the report.
- An empty-handed `Rover check-in` drone tests the normal check-in callback at
  its current pose and after each check-in route step. If it meets the moving
  rover, the same atomic callback queues the no-report check-in, docks the
  drone, and stops its A* route. It does not send a second rover-stop signal:
  there is no unfinished report to protect, and docking prevents the rover from
  moving out of contact.

**Bound motion.** When the rover moves one path step, every docked drone is
moved to the same logical position before the rover-movement lock is released.
The movement is recorded as carried distance, not self-propelled drone travel.
The drone movement worker performs no A*, DFS, rotation, or independent move
while docked. Docked sensors remain inactive in this patch so rover carriage
does not become an unreviewed exploration or incidental-scan policy. Multiple
drones may occupy the same logical dock; any visual offset is rendering only
and cannot alter LOS, sharing, collision, or routing coordinates.

**Communication.** Mechanical attachment is persistent physical contact.
SLAM exchange and rendezvous acknowledgements may therefore continue through
the existing rover-contact functions while docked. Universal acknowledgement
still gates rover departure: docking can carry an ACK immediately because the
connection is physical, but it cannot synthesize an ACK for an undocked drone.
The drone never reads the rover's live position; the mission controller applies
the physical attachment during the rover step.

**Release.** A component task, branch follow, or radial probe is claimed only
while the drone is docked. Under the movement lock, perform the existing
departure share at the rover's current position and then remove dock membership
before delivering the directive. The first independent route therefore starts
from the actual release position. A `HOME` directive completes the drone while
it remains docked. Shutdown clears attachment bookkeeping. There is no
distance-based contact expiry for a docked drone; only explicit release can
separate it from the rover.

A bootstrap `ROVER_SCAN` is released in place before delivery, without a
departure share or route. This is required because that directive rotates and
senses at the rover while docked sensing and rotation are inactive.

**Quiescence and compatibility.** Coordinator waiting membership is still
created only by an accepted physical check-in. Team quiescence additionally
requires every waiting drone to be docked, which replaces the repeated
LOS-pruning loop for attached drones. Existing endpoint fall-forward remains
for undocked drones that genuinely reach an old confirmed endpoint and find
the rover absent. Patch A's report identity, exactly-once acceptance, and stop
reservation remain unchanged.

**Code seams.** Keep dock membership in `MissionControl`; expose only
`request_exploration_dock`/`is_exploration_docked` callbacks through movement
dependencies. Add a synchronized carried-position operation to
`DroneRuntimeState`. Extend component check-in path following to stop on a
successful empty check-in as well as a report encounter. The rover worker
updates docked positions in the same critical section as `Rover.move`, and
assignment performs departure sharing and undocking in that section. Display
and frame summaries report `Docked` with report-pending or awaiting-directive
detail.

**Telemetry and tests.** Trace dock acquisition source, queued report ID,
docked duration, carried path/distance, endpoint epochs learned while docked,
and release directive. Do not emit one trace event per carried pixel; summarize
the docking session when it ends. Test blocked LOS, threshold equality, contact
atomic with a rover step, a report encounter before queueing, an empty check-in
intercepting a moving rover, multiple attached drones, assignment release at
the current rover position, inactive docked sensing/movement, HOME while
docked, and universal ACK with one undocked drone. A fresh trace should show no
old↔new check-in oscillation after docking, no report loss, all task claims
reported, and physical quiescence.

Validate this patch in a fresh trace before enabling highway or incidental
rotation behavior.

## Patch B — Rover-computed highway from physically collated SLAM

**Implementation status.** The rover-SLAM graph, bounded queries, physical
snapshot publication, check-in-only drone following, configuration, telemetry,
and analyzer output are implemented behind `[HIGHWAY] mode = off|observe|active`.
The committed mode is `off`. `observe` publishes and evaluates the same advice
as `active` but leaves motion on the existing segmented A* path. The first
active scope is component return/check-in only; task transit, DFS reposition,
probe, and homing behavior are unchanged.

**Authority and input.** The rover builds a versioned highway from its own fused SLAM snapshot after drone data reaches it through verified contact. It does not derive highway edges from rover movement or planned rover paths. It cannot use the simulator cave map, display-wide floor map, a drone's unshared SLAM, or live peer positions. Unknown and insufficient-confidence cells are closed for highway construction. Each edge stores an exact traversable polyline and the rover SLAM version that justified it.

**Macro-areas.** Partition rover-known navigable free space into coarse tiles,
then distinguish disconnected free regions *within* a tile. Each reachable
region is a macro-area for routing coverage, not exploration ownership. Every
contiguous safe boundary run supplies a deterministic adjacent-tile portal.
Edges store exact strict-diagonal polylines from area anchors through those
portals. Query-time start and goal connectors are bounded to their macro-areas;
the sparse area route uses Dijkstra over the stored edges. A build either
publishes one complete immutable snapshot or retains the previous complete
snapshot. Builds are version-debounced and hard-bounded, as are connector
queries; there is no partially visible graph.

**Physical publication.** Give a drone a graph snapshot or versioned delta only at the same verified rover-contact exchange that carries SLAM. If that contact uploaded new SLAM, the graph computed from it may be delivered at a later physical contact; building it does not broadcast it remotely. An initial patch may use direct rover–drone delivery only. Drone–drone relay of tagged snapshots is a separate optional extension, still gated by physical contact. An older locally held graph remains usable as advice if its edges validate; no agent reads the rover's live graph remotely.

**Drone route choice.** For a long route, search a small set of reachable graph entries near the current pose and exits near the target. Estimate the *whole* route cost: local A* to entry + precomputed graph path + local A* from exit to target. Choose the lowest-cost valid pair, rather than the Euclidean-nearest exit alone. Follow the stored graph polyline directly. Bound each connector search to a local window and expansion budget, widening only when needed before falling back. The existing segmented A* remains available for short/direct routes, missing or stale graph coverage, failed connectors, and graph paths whose cost is clearly excessive. A drone may leave the suggestion when its own newer SLAM or collision validation shows a better or required deviation; it keeps local invalid-edge memory until sharing can inform the rover. Do not require Euclidean distance to the target to decrease on every step, since going around a wall can legitimately increase it.

The current drone A* adapter uses the simulator cave map. This patch keeps that adapter for connector and fallback searches while ensuring the *highway itself* comes only from physically collated rover SLAM. Replacing the adapter with belief-only drone A* would be a separate behavior change and should not be hidden inside this patch.

**Execution priority.** Physical report encounter and safety validation outrank highway following. A newly learned rendezvous endpoint or task change invalidates the remaining route and selects again from the drone's locally held knowledge. The highway can guide check-in, task transit, DFS reposition, and homing, but first enable it for check-in and compare that isolated behavior before widening its use. The route planner must not reassign component work.

The committed initial limits are a 32-pixel macro cell, four-version rebuild
debounce, 250 ms whole-build budget, 50 ms query budget, and 4,096 connector
expansions. Straight safe connectors avoid local A* work. On a synthetic
1,615-by-1,010 all-free snapshot, the optimized complete build takes about
145 ms on the development machine; live traces remain the authority for real
cave cost.

**Implemented sequence.**

1. Caller telemetry records every drone path request's goal, status, expansions,
   wall time, endpoint, remaining distance, and path length.
2. Rover graph construction, immutable versioning, bounded graph queries, stale
   snapshot retention, and verified-contact publication are complete.
3. A drone evaluates physically received advice for long check-in routes. Its
   newer local occupied evidence, endpoint shape, and circuity cap can reject
   the suggestion. `active` follows accepted polylines with physical collision
   checks and falls back to segmented A*; `observe` always falls back.

**Acceptance evidence.** Unit tests cover disconnected regions within one tile, narrow portals, safe diagonals, new obstacles, map-version changes, unreachable connectors, and physical publication. A route benchmark compares graph cost and build cost with rover-map A* over macro-area pairs; the graph should provide near-optimal routes or expose a fallback, not merely cover tiles. In paired traces, report highway-eligible share, A* calls/expansions/CPU time, graph build time, route distance and reversals, completion time, and report delivery. Require no communication or completion regression before enabling it by default.

## Patch C — Incidental rotation for small wall-adjacent pockets

**Implementation status.** The local-SLAM detector, persistent attempt memory,
resumable transit overlay, budgets, trace events, UI state, settings round-trip,
and analyzer aggregation are implemented. The committed default is `off`.
Matched seed 5 and seed 0 `observe`/`active` runs completed all five active
scans without timeout or cancellation, retained all interrupted routes, and
learned 2,333 cells with 1.53 s of scan waiting. Active was faster and shorter
in both asynchronous pairs, but one seed 0 pocket was revisited: the later
directive travelled 358.4 px and performed nine DFS scans for nine cells. This
supports a separately gated geographical-batching experiment, not enabling
Patch C by default.

The `[INCIDENTAL_SCAN]` defaults are two attempts, six cumulative simulated
seconds of waiting, a three-second attempt timeout, 180 cumulative requested
degrees, a one-sensor-range post-selection cooldown, and half-range sampling.
Only component-task transit, component-follow transit after branch DFS starts,
and `component_dfs_reposition_astar` are eligible. Initial branch following,
breadcrumb recovery, probes, rover return/check-in, reporting, homing, docking,
and waiting remain excluded.

**Purpose.** Clear small unknown pockets beside known walls while a drone already passes within sensor range, so DFS need not schedule a later return trip. Revealing a larger frontier is a possible side effect, not a reason to trigger the scan.

**Candidate.** On bounded sampling points of a task-transit or DFS-reposition route, use *that drone's current SLAM* to find an unknown connected pocket that touches a known occupied wall, fits within roughly one sensor footprint, and is visible from the current safe pose in a side-looking cone. Reject pockets connected to the edge of the local inspection window, since they may be the mouth of a large unknown region. Require the proposed heading to see unknown cells that the current heading is unlikely to see. Use the sensor's range/FOV and local occupancy for the visibility estimate; do not consult the true cave or the rover's undisclosed map. Keep tiny or migrated gateways eligible for normal future component work regardless of this scan.

**Behavior.** Pause the retained route, rotate in place toward the selected pocket, and wait for a fresh scan completion at the exact pose and heading, using the existing scan-sequence/timeout contract. Update local SLAM/frontiers, then resume the same route or replan only if that route became invalid. One pocket gets one attempt by stable geometric signature; reopen it only if materially new unknown support appears or its gateway moves. Enforce a configurable per-directive time/scan cap and distance cooldown. A timeout resumes transit without retry strikes. Initial scope excludes return/reporting/homing and rover-contact handling so incidental work cannot delay a pending report or universal ACK.

**Accounting.** Record a stable local pocket signature, predicted unknown support, requested heading, actual newly known cells, rotation/wait time, and whether later DFS work targeted the same vicinity. The incidental scan does not itself retire a rover work unit, claim another component, or invoke mid-generation assistance. Its observations enter normal physical report/sharing flow.

**Experiment.** First log candidate opportunities with the rotation disabled. Then compare flag-off and flag-on runs under the same seed, analyzer, and similar machine load. Measure wall-adjacent pocket cells closed, later travel back to those pockets, added scan time, total drone distance, accepted reports, coverage at quiescence, and mission time. Keep the feature only if saved revisits and useful coverage offset its stops without delaying completion; do not tune it to force 100% floor coverage.

## Patch D — Focused Frontier Batching

**Status and purpose.** Implemented behind a committed-default-off
`FOCUSED_FRONTIER_BATCH` rollout; focused unit and interaction coverage is in
place, while matched live validation remains pending. Focused Frontier
Batching reduces repeated drone-to-rover round trips during the registry's
focused phase by claiming nearby existing component tasks together and by
allowing bounded, provisional service of frontiers newly exposed inside an
exclusive spatial lease. It activates only when the existing registry-topology
predicate
`FrontierTaskCoordinator._focused_endgame_is_active()` is true. Display floor
coverage is not an activation input.

The first Patch-D traces exposed a scheduling defect rather than a useful
batching tradeoff: candidate economics repeatedly ran full-grid Python
shortest-path searches, including a 57.3-second rover check-in. Patch D now
uses bounded cached highway queries, caps the entire planning pass at 250 ms
and 128 route queries, reuses the selected exact connectors when constructing
the lease, and falls back to ordinary singular assignments when a planning or
route budget is unavailable. No partial batch claim is made on fallback.
The planning limit is a wall-clock limit covering cached-route evaluation and
lease preview construction as well as new graph queries. Lease rasterization
is restricted to the required route/component envelope and uses a precise
Euclidean distance margin; the planner reserves finalization time before it
starts that work. If a complete plan cannot be finalized inside the remaining
budget, it is discarded before any claim or lease mutation.

Focused Frontier Batching has its own
`[FOCUSED_FRONTIER_BATCH] mode = off|observe|active` rollout and is committed
`off`. `off` must preserve the current scheduler, directive shapes, claim path,
executor, report path, and trace output. `observe` computes the same seed
assignments, candidate batches, leases, costs, and rejection reasons that
`active` would compute, but it neither changes assignment ordering nor reserves
a cell, task, work unit, token, or random choice. `active` may issue a batch
only during the registry's focused phase. If that phase ends before dispatch,
fall back to ordinary component tasks; a batch already issued remains
authoritative until its physical report is accepted.

### Authority and invariants

- The rover coordinator remains the only authority for existing tasks and work
  units. A batch is a container of independent tasks, not a merged component,
  synthetic lineage parent, or territorial owner.
- Every member retains its original `task_id`, `component_id`, component
  revision, work-unit IDs, parent task, depth, and its own ordinary
  `ClaimLease` token. Exactly-once rules continue to be checked per claim.
- The spatial lease authorizes only bounded local observation. It cannot claim
  or complete an existing rover work unit, invent component identity, or make
  local SLAM globally visible before physical report delivery.
- All coordinator inputs come from rover-held SLAM and state learned through
  existing physical exchanges. The drone uses only its own SLAM, the directive,
  and its last physically learned rendezvous endpoint. Neither side reads a
  remote registry, the simulator truth map, or a peer's live position.
- An active lease has one owner and does not overlap another active Patch D
  lease. It has no wall-clock expiry: losing contact cannot make the rover
  reassign its work while the owner may still be servicing it. It is released
  only by accepted report, explicit directive cancellation during shutdown, or
  the existing terminal recovery path.
- Report encounter, stop reservation, docking, energy suspension, waiting,
  quiescence, and universal rendezvous acknowledgements keep their existing
  semantics. A batch returns one report through the same physical path.

### Directive, claim, lease, and report contracts

Add `DirectiveKind.COMPONENT_BATCH` instead of overloading the singular
`COMPONENT_TASK` fields with a fake root. The concrete contract should be
equivalent to:

```python
@dataclass(frozen=True)
class BatchMember:
    task: ExplorationTask
    work_units: tuple[ComponentWorkUnit, ...]
    claim: ClaimLease
    estimated_service_cost: float

@dataclass(frozen=True)
class SpatialLease:
    lease_id: int
    owner_drone_id: int
    directive_id: int
    issued_revision: int
    cell_spans: tuple[tuple[int, int, int], ...]  # y, inclusive x0, x1
    member_task_ids: tuple[int, ...]

@dataclass(frozen=True)
class BatchMemberReport:
    task_id: int
    component_id: int
    component_revision: int
    claim_token: ClaimToken
    disposition: str  # complete, suspended, unreachable, budget_exhausted
    work_unit_outcomes: tuple[WorkUnitOutcome, ...]
    causal_transitions: tuple[CausalTransition, ...]
    suspension: TaskSuspension | None

@dataclass(frozen=True)
class ProvisionalFrontierObservation:
    observation_id: int
    source_task_id: int
    cells: frozenset[Position]
    scan_position: Position
    scan_heading: int
    sensor_newly_known_cells: int
    sensor_confidence_gain: float
    local_route_distance: float
    causal_predecessor_id: int | None
```

`ExplorationDirective` gains `batch_members` and `spatial_lease`; its legacy
singular `task`, `work_units`, and `claim` stay empty for a batch. The batch
also carries the coordinator's separate/combined cost quote, avoided cost,
service budget, and the set or high-water mark of work units that existed at
`issued_revision`. `CoordinationReport` gains `lease_id`, one
`BatchMemberReport` for every member, and zero or more
`ProvisionalFrontierObservation` records. Legacy component-report fields stay
empty for `COMPONENT_BATCH`. The outer `report_id` remains the replay key.

Use run-length encoded map rows for `cell_spans`; both coordinator and drone
expand or query the same inclusive raster mask. This makes containment and
non-overlap exact and deterministic without sending a large set of pixels.
Coordinates are in the existing global SLAM frame, not screen or tile space.

Each existing task in a batch is claimed all-or-none under the coordinator
lock. Add a registry operation that first verifies that every requested work
unit exists, is `READY`, appears in only one member, and still belongs to that
member's component revision; only then change all units to `CLAIMED`. Allocate
and index one `ClaimLease` per member. If any validation or insertion fails,
issue no batch, leave every unit ready, and allow the scheduler to recompute.
Claiming the directive activates every member's units together. Do not replace
`_claims_by_task` or `_claims_by_token` with a batch-level token.

### Seed assignment and batch formation

Batching is a second pass after maximum-parallel seed assignment:

1. Build eligible ordinary tasks exactly as today, including lineage blockers,
   route-rejection memory, rover-known reachability, and energy eligibility.
2. Compute a maximum-cardinality matching from idle drones to seed tasks before
   considering any attachment. Preserve the focused-endgame cost tie-breaks.
   No task may be attached while an otherwise eligible idle drone lacks a seed.
   When fewer tasks than eligible idle drones exist, the unmatched drones wait
   normally.
3. Query the latest complete rover-SLAM highway snapshot for deterministic
   routes from the rover and between candidate task entries. Validate returned
   polylines against the current rover-known reachable mask. A missing, stale,
   disconnected, budget-exhausted, or invalid pair has infinite cost and cannot
   share a batch. Before the first graph exists, only a collision-free direct
   connector is allowed; Patch D never falls back to a whole-grid search.
4. For each leftover task, calculate its marginal saving against each seeded
   batch, choose the greatest positive eligible saving, attach it, and recompute
   until no attachment passes or a bound is reached. Break ties by seed drone
   ID and then task ID. A task appears in at most one proposed batch.
5. Build disjoint lease masks in the same deterministic order. A candidate that
   cannot receive a complete non-overlapping envelope remains an ordinary
   leftover task for a later scheduling opportunity. Only after all checks pass
   does `active` atomically claim and issue the member group.

For a member set `B`, let `T(B)` be the shortest estimated rover-to-members-to-
rover tour on the rover-known distance matrix, plus the unchanged sum of member
service estimates. With the small component cap below, enumerate member orders
rather than use a heuristic. Attaching task `t` is economic only when

```text
avoided_round_trip(B, t) = T(B) + T({t}) - T(B union {t})
```

is strictly positive and at least the configured minimum. Also require
`T(B union {t}) - T(B)` to fit the detour, energy, and service bounds. Service
estimates occur on both sides and normally cancel in the saving calculation,
but remain in the energy and elapsed-work projections. Record both rejected and
accepted quotes; never attach merely because centroids are close.

### Spatial lease construction and local admission

Construct the envelope only from the dispatch-time rover SLAM. Start with the
selected known-free tour corridors and member frontier geometry, dilate by the
configured lease margin, and include unknown-side cells visible within one
sensor range of an authorized known-free observation pose. Clip the result to
the map, the maximum-detour region, active leases, and a protection halo around
every non-member ready/claimed/active/blocked work unit. Reject an attachment if
clipping removes a member geometry or disconnects its rover-known service
route. Store the final mask and reserve it by `lease_id` until report acceptance.

After any authoritative or local scan, the drone may enqueue a newly exposed
frontier only if all of these checks pass:

- it passes the same significant-frontier size and unknown-support filters used
  by the registry;
- all of its sampled frontier cells and its selected safe observation pose are
  inside the lease, and the component does not touch the lease boundary;
- the observation pose has an exact route in the drone's current local SLAM;
- its stable geometry and anchor are absent from the directive's authoritative
  members, the local visited set, pending local work, and reserved/excluded
  geometry represented by the lease;
- inserting it into the current exact local route has positive avoided-round-
  trip value and every remaining bound still passes.

For this local decision, use the last physically learned rendezvous endpoint
`R`, the drone's current pose, and exact locally routable distances. Let
`C_base` be the best route through remaining batch work and back to `R`, and
`C_insert` the best route with candidate `f` inserted. The later standalone
travel estimate is `2 * d(R, f)`. Admit only when

```text
local_avoided(f) = C_base + 2 * d(R, f) - C_insert
```

is strictly positive and meets the configured minimum. If `R`, `f`, or any
required connector is locally unreachable, reject the candidate. Recompute the
order after each completed scan because local SLAM and exact routes may have
changed. This calculation is local advice and never reads the rover's current
position.

A lease-admitted node always has `work_unit_id=None`. It produces a provisional
observation, not a `WorkUnitOutcome`, and cannot use
`visited_successor_anchors` to retire registry work. At report reconciliation,
apply claimed member outcomes first, fuse the physically delivered SLAM, and
then match provisional observations. They may suppress only work units created
by that reconciliation from genuinely new frontier geometry. Any match to a
work unit that existed at `issued_revision` and was not claimed by this batch
remains actionable; keep the observation as evidence but do not mark that unit
visited. Preserve a member's causal transition only when the existing local
successor test establishes that relationship. Unrelated leased observations do
not acquire synthetic ancestry.

### Drone execution and hard bounds

The executor keeps one member context at a time: its task, claim, DFS stack,
outcomes, causal transitions, suspension, and actual path. It may dynamically
choose any unfinished member using its local exact routes and the incremental
cost of completing the remaining set and returning. Switching members never
moves outcomes or lineage between contexts. An unreachable or locally
invalidated route suspends that member and permits another safe member to run;
it does not discard already completed peers. When the final member or
provisional node is done, or a global bound requires return, freeze all
contexts, return once, and create one batch report. Patch A report encounter may
shorten that return exactly as it does now.

Proposed conservative initial `[FOCUSED_FRONTIER_BATCH]` defaults are:

```ini
[FOCUSED_FRONTIER_BATCH]
mode = off
maximum_claimed_components = 3
maximum_total_components = 4
lease_margin_sensor_ranges = 1.0
maximum_detour_sensor_ranges = 2.0
minimum_avoided_round_trip_sensor_ranges = 0.25
maximum_service_seconds = 45.0
maximum_total_dfs_nodes = 48
maximum_consecutive_low_gain_scans = 2
low_gain_maximum_new_cells = 1
low_gain_maximum_confidence_gain = 1.0
maximum_planning_ms = 250.0
maximum_route_queries = 128
```

`maximum_total_components` counts claimed members plus provisional frontier
components. Detour is the additional exact route length relative to servicing
the claimed seed alone. Service time starts when the drone is released and ends
when return begins; the return/report leg itself is never abandoned when the
cap fires. DFS nodes are cumulative across every member and provisional tree,
with the existing per-stack depth/node limits still acting as absolute guards.
After the configured consecutive low-gain scans, do not push another
non-authoritative successor; finish any current authoritative anchor and choose
another claimed member or return. The low-gain test uses both thresholds shown
above and resets only after a higher-gain scan.

At dispatch and before every member, provisional insertion, and DFS push,
project route-to-work, remaining service, route back to the last rendezvous, and
safety reserve through the existing `EnergyPolicy`. Capture the accepted
service-energy allowance in the directive and never exceed it. A finite-energy
failure suspends all unfinished member claims and starts one return. Under the
current unlimited policy the distance, elapsed-time, component, and DFS caps
remain effective; Patch D must not introduce battery drain as a side effect.

If Patch C is also active, its scan wait and route interruption count toward the
Patch D elapsed and energy budgets. An incidental pocket scan does not consume a
Patch D component slot or become provisional component service unless the
separate significant-frontier admission path later accepts it. Patch C keeps
its exclusions for return, reporting, homing, docking, and waiting.

### Atomic report acceptance and replay

Acceptance is a two-phase operation under the existing coordinator lock. Phase
one is read-only and validates the entire outer report before any task, work
unit, claim, lease, report ledger, scan evidence, or registry revision changes:

- the active directive ID, kind, drone owner, lease ID, and lease owner match;
- member reports form an exact one-to-one set with directive members, with no
  missing/extra task, duplicate token, or cross-member work-unit outcome;
- every task/component/revision/token tuple matches the immutable directive;
- each suspension matches its task, owner, token, remaining-work complement,
  and allowed disposition;
- causal predecessors belong to that member and all provisional observations
  have unique IDs, valid finite metrics, an authorized source member, and
  geometry/pose wholly contained by the lease;
- aggregate paths, distances, scan counts, and gains are internally
  consistent. They remain telemetry, not authority.

Malformed structure or unauthorized content rejects the whole report with no
mutation and leaves the directive, every claim, and the lease active. After
structural validation, classify each member against current registry state:

- **applicable:** its exact claim is still live and its claimed units may accept
  outcomes or suspension;
- **stale:** it was authorized by this active directive, but later accepted
  lineage/reconciliation made some or all units terminal or changed their
  revision;
- **suspended/failed:** it has a valid live claim and a valid suspension such as
  unreachable route, energy reserve, sensor timeout, or batch bound.

Staleness and a declared member failure are valid business outcomes, not report
malformation. Phase two builds a complete mutation plan: apply all applicable
member outcomes, release or suspend each live remainder, close stale claim
bookkeeping without rewriting terminal lineage, release the spatial lease,
record the outer `report_id`, and reconcile all member transitions and
provisional observations once against the delivered rover SLAM. Apply that plan
as one non-throwing commit while holding the lock; if a registry helper cannot
guarantee this after preflight, stage changes on detached task/registry state
and swap only on success. No observer may see half the members committed.

Thus one stale or failed member does not invalidate successfully serviced
members, while an invalid token or fabricated outcome cannot cause a partial
commit. Replaying an accepted outer `report_id` returns accepted without
reapplying any member, release, lease, evidence, or reconciliation mutation,
and preserves the existing already-published-directive signalling behavior.
The report is not accepted, and the drone is not waiting/quiescent, merely
because a subset of members was valid.

### Telemetry and analyzer output

Add bounded event payloads rather than per-pixel lease events:

- `rover_focused_frontier_batch_evaluated`: mode, focused revision, seed matching,
  candidate task IDs, separate/combined costs, avoided cost, detour, decision,
  and rejection reason;
- `rover_focused_frontier_batch_planning_completed`: whole-pass wall time, route-query
  count, cache hits, candidates, batch count, graph version, and completion or
  budget-fallback status;
- `rover_focused_frontier_batch_issued`: directive/lease IDs, drone, member task,
  component, work-unit and claim-token lists, lease span/cell counts, bounds,
  and projected energy/service cost;
- `drone_focused_frontier_batch_member_selected` and `_finished`: dynamic order, exact
  route quote, actual travel/gain, disposition, and remaining budgets;
- `drone_focused_frontier_batch_frontier_evaluated`: stable geometry signature,
  containment/reachability/reservation checks, baseline/insert/later costs,
  avoided value, and accept/reject reason;
- `drone_focused_frontier_batch_bound_reached`: bound name, usage, limit, current member,
  and number of suspended members;
- `rover_focused_frontier_batch_report_accepted` or `_rejected`: structural result,
  per-member applicable/stale/suspended classification, released claims/lease,
  provisional reconciliation counts, and replay status.

Extend `tools/analyze_runtime_trace.py` and its focused tests to report observe
opportunities, active batch count/size, seeds assigned before attachments,
estimated separate versus combined trips, estimated avoided distance, actual
member/detour/return distance, service duration, provisional gains, later
revisits, low-gain stops, bound stops, stale/failed members, report rejections
and replays, claim/lease leaks, and Patch C overlap. Keep existing mission time,
total/self-propelled/carried distance, scan wait, coverage, accepted reports,
route retention, quiescence, and ACK summaries so improvements are not inferred
from one metric.

### Tests and implementation seams

Coordinator and registry tests must cover:

- `off` equivalence, `observe` purity, focused-only activation, and deterministic
  mode/config round-trip;
- maximum-cardinality seeding before attachment, including three idle drones
  with four tasks, fewer tasks than drones, unreachable tasks, energy-ineligible
  drones, and deterministic ties;
- rover-known shortest-path cost rather than Euclidean proximity, positive and
  non-positive marginal savings, disconnected components, every hard bound,
  and disjoint lease construction/clipping;
- all-or-none multi-task claim creation/activation, separate tokens and lineage,
  failed second-member preflight with zero first-member mutation, and no token
  or lease leak after accepted reports;
- whole-report rejection for malformed membership, duplicate outcomes, invalid
  token, out-of-lease observation, or mismatched suspension; successful peers
  retained when another authorized member is stale, unreachable, or suspended;
- replay before and after a new directive, with no repeated outcomes, scan
  evidence, reconciliation, or release;
- provisional observations never completing a pre-existing unclaimed work unit,
  while report-created matching work may be reconciled as serviced; causal
  transitions remain attached to their actual predecessor only.

Executor tests must cover local exact-route reordering, one DFS context per
member, route failure followed by another member, each local-admission predicate,
lease-boundary contact rejection, duplicate geometry, positive local economics,
and return on component/detour/energy/time/DFS/low-gain limits. Assert that no
test dependency exposes rover live position, rover registry, peer state, or cave
truth. Retain characterization tests for report encounter during the one return,
docking and release, empty check-in, energy suspension, report retry, waiting,
team quiescence, and universal ACK gating. Add analyzer fixtures for observe,
active, replay, stale member, bounds, and Patch C interaction.

Implementation should be staged through these seams:

1. Add config/settings/UI round-trip, frozen contracts, snapshot visibility,
   atomic registry claim-group and pure cost/lease helpers in
   `mission/exploration_coordination.py` and `mapping/frontier_registry.py`.
   Keep mode `off` and executor behavior unchanged.
2. Add observe-only scheduling and analyzer telemetry in `mission/control.py`
   and `tools/analyze_runtime_trace.py`; prove it does not alter directives,
   claims, paths, or RNG consumption.
3. Add `COMPONENT_BATCH` execution in `agents/drone_movement.py`, keeping pure
   significant-frontier/lease/economic helpers in `agents/component_explorer.py`.
   Add atomic batch-report preflight/commit last, still default-off.
4. Update `docs/CODEFLOW.md` and `docs/TESTING.md` with the reviewed final
   contracts and validation commands. Do not run or overwrite generated map
   assets as part of implementation or tests.

### Staged live rollout and acceptance

First validate `[HIGHWAY] observe` versus `active` with `FOCUSED_FRONTIER_BATCH=off` and
`INCIDENTAL_SCAN=off`, checking graph/build budgets and check-in route behavior.
Then run `FOCUSED_FRONTIER_BATCH=observe` and matched `FOCUSED_FRONTIER_BATCH=active` missions with
`HIGHWAY=active` and Patch C still off. Start with seeds 0 and 5 so
the known late seed 0 revisit and the productive seed 5 pockets are represented;
add seeds only if the number of actual batch opportunities is too small. Compare
the proposal telemetry before interpreting asynchronous whole-run differences.

Patch D passes the isolated stage only if every claim is reported or safely
released, there are no overlapping/leaked leases, no unclaimed pre-existing
work is retired provisionally, replay is idempotent, physical quiescence and
universal ACK still hold, seed parallelism never decreases, and accepted batches
show positive estimated savings with fewer rover round trips or lower relevant
task/check-in distance without a material coverage or completion regression.
Timeouts, suspensions, stale members, and route retention must be reported, not
hidden by aggregate mission time.

Only after isolated acceptance compare both `FOCUSED_FRONTIER_BATCH=active` and
`INCIDENTAL_SCAN=active` against the same Patch D active / Patch C off baseline,
keeping `HIGHWAY=active` in both.
Require combined caps and accounting to remain correct and check whether Patch C
gains are duplicated by lease-local exploration. Keep both features default-off
until their isolated and combined evidence is reviewed. Do not infer causality
from faster asynchronous completion alone.

## Separate follow-ups

- **DFS repositioning:** Instrument per-directive reposition distance and useful scan gain before another DFS policy change. The latest run's 58 logged reposition A* paths all reached their targets; route failure is not the current demonstrated issue.
- **Low-gain suppression:** Its synthetic tests pass, but both complete runs
  had zero live deferrals. In the 2026-09-15 trace, 20 tasks completed after
  420 s for 111 sensor-known cells, and 17 of those tasks gained at most three
  cells. Validate whether exact anchor/heading matching is too brittle while
  preserving novel unknown support and moved/tiny gateway exemptions before
  changing thresholds.
- **Validation runs:** Do not start another full simulation as part of this draft. Review each patch and its trace scope first; after implementation, the user can run the paired simulations and provide traces for analysis.
