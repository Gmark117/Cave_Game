# Corridor backbone: frozen-map comparison

The 2026-10-08 change replaces tile-shaped routing with a medial-axis corridor
backbone, pruned terminal spurs, optional capillaries, and validated route
straightening. The shipped settings now enable Highway, Incidental Scan, and
Focused Frontier Batching; their configuration constructors retain `off`
fallbacks. Physical publication and local-SLAM authority are unchanged.

`maximum_access_distance_sensor_ranges = 2.0` bounds cardinal geodesic access
to the network in drone LiDAR ranges. Smaller values add more capillaries.
The old tile-size setting remains readable but no longer determines geometry.

## Reproduction

Synthetic fixtures require no simulation or generated cave:

```powershell
python tools/benchmark_highway.py --access-distance 80 --output generated-images/highway-comparison.svg
```

For the October 8 seed-5 run, the supplied cave matrix has matching generation
time and dimensions. Replaying the first 180 sensor scans and recorded physical
exchanges yields a frozen rover belief with 38,444 free cells and 31 rover
encounters. The replay is a reconstruction rather than an exact capture of the
original concurrent SLAM state. The map argument must match the trace's cave.
Neither the runtime builder nor its route queries read the cave matrix.

```powershell
python tools/benchmark_highway.py --trace logs/mission_trace_20261008_115111_122353.jsonl --map Assets/Map/map_matrix.txt --access-distance 320 --output generated-images/highway-comparison.svg
```

An exported frozen SLAM NPZ can instead be supplied with `--snapshot`. It needs
`occupancy` and `confidence`, with optional `origin` and `full_shape` arrays.
The access-distance CLI argument is in pixels; 320 px equals the configured
2.0 ranges for SMALL drones with 160 px LiDAR range.

## Results

Both builders received identical arrays and deterministic endpoint pairs.
Times below are one local sample and are not mission-performance evidence.

| Input | Matched routes | Tiled total distance | Backbone total distance | Tiled build | Backbone build |
|---|---:|---:|---:|---:|---:|
| Wall | 13 | 2452.09 px | 2069.25 px | 3.98 ms | 23.00 ms |
| Loop | 12 | 987.04 px | 784.11 px | 1.20 ms | 15.39 ms |
| Branches | 12 | 1183.97 px | 1105.07 px | 3.35 ms | 20.20 ms |
| Replayed rover | 12 | 2075.96 px | 1876.88 px | 24.63 ms | 34.00 ms |

The replayed graph shrank from 63 nodes/99 edges to 23 nodes/28 edges. All 12
sampled routes completed with both builders, with about 9.6% less aggregate
distance on the backbone. This sample does not establish global optimality or
a mission-duration improvement. The standalone SVG shows both networks and
one route per input.

Tests cover density/access bounds, exact costs, narrow elbows and bridges,
retained loops, disconnected and diagonal-only regions, immutable access
fields, query budgets, physical publication, and batching integration.

The later `mission_trace_20261008_123507_352406.jsonl` exposed stale rendering:
four builds completed through rover SLAM version 24; version 28 exhausted the
250 ms budget, and later versions hit the former 350,000-free-cell guard. The
graph therefore remained at version 24 even as the registry reached version 49.
Refresh requests now follow every newly received rover SLAM version on the rover
worker, sharing the input with frontier reconciliation. That observation-only
refresh cannot issue directives, release live claims, or finish the mission.
Drone highway delivery still requires a verified physical check-in.

The size guard has been replaced by conservative skeleton reduction: an entire
coarse block must be confidently free, every original free component must remain
represented and connected, and holes must survive. Otherwise finer resolution
is used. All lifted chords and geodesic coverage still use original pixels.
Reconstructing all 11,240 scans and 218 physical rover encounters from the new
trace produced a final frozen rover map with 585,669 free cells. Three builds
completed in 200-222 ms at skeleton scale 3, with 259 px maximum measured access
against the 320 px limit. All 24 sampled routes completed with exact costs and
safe cells/diagonal corners. Reconstruction is an offline approximation, not an
exact capture of the concurrent rover map. At that stage native-work limits declined
inputs above 750,000 free cells, 2,000,000 working pixels, or 500,000 skeleton
pixels after reduction; the current full-resolution fallback is described below. Over-budget or unsafe builds retain the previous
complete graph and ordinary fallback.

## October 9 live trace and simpler whole-map maintenance

`mission_trace_20261009_115726_982829.jsonl` completed exploration at 352.24 s.
It recorded 15 task claims/reports and three Focused Frontier Batches, with no
rejected reports, replays, open batch claims, or open leases. Five incidental
scans completed with retained suffixes. Shutdown was not captured. Whole-run
timing remains asynchronous evidence, not causal proof of improvement.

The regional experiment completed 27 builds, with 6 budget misses and 12
geometry-cache hits. It produced redundant branches in overlapping regions and
seam connectors. The renderer cleared its cached surface for each replacement;
it did not overlay the entire version history. On the same source version 17,
a regional graph had 271 nodes/475 edges, while a later full rebuild had 110/142.
The region implementation and its seam/query code have now been removed.

The worst smoothed frame sample was around 221.8 s at 0.55 FPS: 981 ms in sensing
and 815 ms in rendering. No substantial highway build ran during that dip; one
full attempt near 210.7 s declined its input after 14.5 ms. Two batch
assignment-to-selection gaps were about 16 s and 36 s. Inspection found repeated
uncached, unbounded Python Dijkstra calls in drone-side member selection. A
representative route on the saved large map took 1.69 s under profiling, with
397,854 heap removals and 3.55 million dictionary lookups. This is strong
mechanistic evidence of GIL contention, though the historical trace contains
no thread-stack samples proving every delayed frame came from that function.

Current highway jobs rebuild one complete map in a low-priority process. Only
one job can run, submissions are at least two wall-clock seconds apart, and new
uploads coalesce. Unchanged free geometry does not trigger periodic rebuilding.
Each completed graph replaces the last graph atomically. Free additions may
leave a coherent older input published while a newer build is queued; a
correction removing an input free cell discards an in-flight result. Physical
check-in remains the only delivery path to drones. The frontier registry
updates immediately and can lead the overlay during the throttle interval.

The reconstructed rover belief near 206 s contains 585,424 free cells. It
cannot use the conservative reduced skeleton and previously hit the 500,000
skeleton-pixel guard. Supported unreducible inputs now use native thinning at
original resolution. One prepared whole-map skeleton is retained for the
identical free mask, origin, and settings after a budget miss; it is never a
published partial graph. This resumes expensive access-field work on a later
attempt without regional stitching. One retry keeps that exact input even if
new free cells arrive, then releases it to the latest queued map. A correction
removing an input free cell cancels the retry. The 250 ms attempt budget, 750,000-free-cell
limit, and 2,000,000-working-pixel limit remain. A native call can overrun the
wall-clock budget before the next check rejects the attempt.

The actual process can be exercised without running a mission or writing cave
assets:

```powershell
python tools/benchmark_highway.py --snapshot logs/highway_20261009_pre_drop_snapshot.npz --access-distance 320 --updates --background
```

One sample completed after attempts of 275.14 ms (native-call budget overrun)
and 156.67 ms. Initial publication, including startup, transfer, and throttling,
took 2.19 s. A one-cell correction took 4.01 s across two attempts, demonstrating
the deliberate freshness tradeoff. Submission took 5.53 ms initially and 1.11 ms
for the correction. The initial full graph had 118 nodes/133 edges, versus the
live regional graph's 412/814 at the same free-cell count; maximum measured
access was 161 px against the 320 px bound. All 24 sampled routes completed,
with safe cells, diagonal corners, endpoints, and exact costs; the maximum query
time was 7.95 ms. Reconstruction approximates concurrent runtime state, and
these measurements do not prove a live FPS or mission-duration gain.

Drone-side local planning now uses versioned route caching and bounded A*.
Decision passes have a 250 ms deadline and 128-query cap; individual queries
have a 50 ms/4,096-expansion cap. Incomplete costs cannot justify provisional
work. Tour optimization retains the rover's order on a budget miss, and claimed
transit can use safe partial progress under the existing route retry policy.
The representative query above now returned 98 safe progress points after
4,096 expansions in 16.87 ms, rather than completing an unbounded full search.
It was correctly tagged partial and did not supply a complete economic cost.
`drone_local_route_planning_completed` records time, query/cache counts, and
budget status for the next live trace.

Return targets still advance only on physically carried confirmed arrival or
actual departure evidence. Proposals alone do not redirect ACK carriers;
universal ACK, claims, lineage, docking, and report acceptance retain their
contracts. Highway, Incidental Scan, and Focused Frontier Batching retain
`off` constructor fallbacks, while the shipped `simulation.default.ini` now
sets all three to `active`.

Future matched validation should compare frame-stage times, drone-local
planning bounds, graph publication latency, route fallbacks, accepted reports,
and ACK/quiescence behavior. Do not infer causal improvement from mission time.

Verification before the latest live run: 608 automated tests passed, `compileall`
passed, and `git diff --check` reported only LF/CRLF warnings. Historical
all-active seed-5 trace analysis still succeeds. SHA-256 hashes of all four
generated `Assets/Map` files were unchanged at that checkpoint.

## Latest all-active live baseline and cleanup

`mission_trace_20261009_125123_464282.jsonl` completed exploration at 345.71 s
with 97.62% floor coverage and captured shutdown at 364.89 s. All 48 claims
received reports. One Focused Frontier Batch had two applicable, complete
members; there were no rejected/replayed batch reports, open batch claims, or
open leases. All 55 docks released. Drone-local planning peaked at 76.74 ms.
Seven eligible returns used highway-derived routes; observed shortcuts match
intentional route straightening. The user reported a smooth run. This is
concurrent live evidence, not causal proof of a performance gain.

The subsequent [behavior-preserving cleanup](CLEANUP_AUDIT.md) retains these
contracts. Its analyzer produces exactly the same summary as commit `05a2d8a`
for this trace and the historical all-active seed-5 trace. Cleanup verification
includes the complete automated suite and a real background-worker smoke check
on synthetic maps; it does not regenerate cave assets or run a new mission.
