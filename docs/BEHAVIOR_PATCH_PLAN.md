# Proposed behavior patches: contact, SLAM highway, and wall-pocket scans

Status: Patch A is implemented and live-tested. Patch A2 is implemented and
unit-tested, with live trace validation pending. Patch B and Patch C remain
separate design drafts for review.

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

**Authority and input.** The rover builds a versioned highway from its own fused SLAM snapshot after drone data reaches it through verified contact. It does not derive highway edges from rover movement or planned rover paths. It cannot use the simulator cave map, display-wide floor map, a drone's unshared SLAM, or live peer positions. Unknown and insufficient-confidence cells are closed for highway construction. Each edge stores an exact traversable polyline and the rover SLAM version that justified it.

**Macro-areas.** Partition rover-known navigable free space into coarse tiles, then distinguish disconnected free regions *within* a tile. Each reachable region is a macro-area for routing coverage, not exploration ownership. Keep distinct safe doorway/adjacent-tile portals and junctions rather than replacing an entire region with one center. Connect portals using local shortest paths over rover-known free space and compress degree-two chains. Every reachable macro-area needs a graph entry; disconnected or newly unknown areas simply lack a highway until more SLAM is delivered. Add sparse shortcut edges where a sample of graph routes has excessive stretch relative to rover-map grid A*. Build or update affected portions on a bounded rover-side worker, and publish immutable graph snapshots so route readers never see half a rebuild.

**Physical publication.** Give a drone a graph snapshot or versioned delta only at the same verified rover-contact exchange that carries SLAM. If that contact uploaded new SLAM, the graph computed from it may be delivered at a later physical contact; building it does not broadcast it remotely. An initial patch may use direct rover–drone delivery only. Drone–drone relay of tagged snapshots is a separate optional extension, still gated by physical contact. An older locally held graph remains usable as advice if its edges validate; no agent reads the rover's live graph remotely.

**Drone route choice.** For a long route, search a small set of reachable graph entries near the current pose and exits near the target. Estimate the *whole* route cost: local A* to entry + precomputed graph path + local A* from exit to target. Choose the lowest-cost valid pair, rather than the Euclidean-nearest exit alone. Follow the stored graph polyline directly. Bound each connector search to a local window and expansion budget, widening only when needed before falling back. The existing segmented A* remains available for short/direct routes, missing or stale graph coverage, failed connectors, and graph paths whose cost is clearly excessive. A drone may leave the suggestion when its own newer SLAM or collision validation shows a better or required deviation; it keeps local invalid-edge memory until sharing can inform the rover. Do not require Euclidean distance to the target to decrease on every step, since going around a wall can legitimately increase it.

The current drone A* adapter uses the simulator cave map. This patch keeps that adapter for connector and fallback searches while ensuring the *highway itself* comes only from physically collated rover SLAM. Replacing the adapter with belief-only drone A* would be a separate behavior change and should not be hidden inside this patch.

**Execution priority.** Physical report encounter and safety validation outrank highway following. A newly learned rendezvous endpoint or task change invalidates the remaining route and selects again from the drone's locally held knowledge. The highway can guide check-in, task transit, DFS reposition, and homing, but first enable it for check-in and compare that isolated behavior before widening its use. The route planner must not reassign component work.

**Patch sequence.**

1. Add route-request telemetry to `PathfindingService.compute_path_segment` and its callers: goal, status, expansions, queue/search elapsed time, partial endpoint, and route length. Classify repeat endpoints and reversals without assuming that a valid wall detour is a failure. This is diagnostic only.
2. Implement and test rover SLAM macro-area graph construction, immutable versioning, graph shortest paths, and LOS-gated publication. Keep drone routing unchanged.
3. Add drone entry/exit selection and highway following for check-ins behind a feature flag, with per-edge validation and segmented-A* fallback. Expand to other route classes only after trace comparison.

**Acceptance evidence.** Unit tests cover disconnected regions within one tile, narrow portals, safe diagonals, new obstacles, map-version changes, unreachable connectors, and physical publication. A route benchmark compares graph cost and build cost with rover-map A* over macro-area pairs; the graph should provide near-optimal routes or expose a fallback, not merely cover tiles. In paired traces, report highway-eligible share, A* calls/expansions/CPU time, graph build time, route distance and reversals, completion time, and report delivery. Require no communication or completion regression before enabling it by default.

## Patch C — Incidental rotation for small wall-adjacent pockets

**Purpose.** Clear small unknown pockets beside known walls while a drone already passes within sensor range, so DFS need not schedule a later return trip. Revealing a larger frontier is a possible side effect, not a reason to trigger the scan.

**Candidate.** On bounded sampling points of a task-transit or DFS-reposition route, use *that drone's current SLAM* to find an unknown connected pocket that touches a known occupied wall, fits within roughly one sensor footprint, and is visible from the current safe pose in a side-looking cone. Reject pockets connected to the edge of the local inspection window, since they may be the mouth of a large unknown region. Require the proposed heading to see unknown cells that the current heading is unlikely to see. Use the sensor's range/FOV and local occupancy for the visibility estimate; do not consult the true cave or the rover's undisclosed map. Keep tiny or migrated gateways eligible for normal future component work regardless of this scan.

**Behavior.** Pause the retained route, rotate in place toward the selected pocket, and wait for a fresh scan completion at the exact pose and heading, using the existing scan-sequence/timeout contract. Update local SLAM/frontiers, then resume the same route or replan only if that route became invalid. One pocket gets one attempt by stable geometric signature; reopen it only if materially new unknown support appears or its gateway moves. Enforce a configurable per-directive time/scan cap and distance cooldown. A timeout resumes transit without retry strikes. Initial scope excludes return/reporting/homing and rover-contact handling so incidental work cannot delay a pending report or universal ACK.

**Accounting.** Record a stable local pocket signature, predicted unknown support, requested heading, actual newly known cells, rotation/wait time, and whether later DFS work targeted the same vicinity. The incidental scan does not itself retire a rover work unit, claim another component, or invoke mid-generation assistance. Its observations enter normal physical report/sharing flow.

**Experiment.** First log candidate opportunities with the rotation disabled. Then compare flag-off and flag-on runs under the same seed, analyzer, and similar machine load. Measure wall-adjacent pocket cells closed, later travel back to those pockets, added scan time, total drone distance, accepted reports, coverage at quiescence, and mission time. Keep the feature only if saved revisits and useful coverage offset its stops without delaying completion; do not tune it to force 100% floor coverage.

## Separate follow-ups

- **DFS repositioning:** Instrument per-directive reposition distance and useful scan gain before another DFS policy change. The latest run's 58 logged reposition A* paths all reached their targets; route failure is not the current demonstrated issue.
- **Low-gain suppression:** Its synthetic tests pass, but both complete runs
  had zero live deferrals. In the 2026-09-15 trace, 20 tasks completed after
  420 s for 111 sensor-known cells, and 17 of those tasks gained at most three
  cells. Validate whether exact anchor/heading matching is too brittle while
  preserving novel unknown support and moved/tiny gateway exemptions before
  changing thresholds.
- **Validation runs:** Do not start another full simulation as part of this draft. Review each patch and its trace scope first; after implementation, the user can run the paired simulations and provide traces for analysis.
