# SLAM Exploration Plan

## Summary

Implement the next SLAM/exploration phase in this order:

1. Extract exploration decision logic behind a policy interface.
2. Add a minimal localization layer that initially mirrors the simulator's true pose.
3. Route SLAM updates and exploration decisions through `PoseEstimate`.
4. Add landmark models and detection as the real localization upgrade path.
5. Add battery constraints after policy/localization boundaries are stable.

The first implementation should preserve current gameplay behavior while making the future SLAM localization and battery rules easy to add.

## Module Placement

Place `exploration_policy.py` under `agents/`, not `navigation/`.

Reasoning:

- `navigation/` owns route computation and pathfinding services.
- Exploration policy is agent behavior: choosing where to go, when to use frontiers, when to return home, and later when to conserve battery.
- `DroneMovementController` should continue to own movement execution and path traversal, while the policy owns target/action choice.

Expected high-level flow:

```text
runtime ground truth
    -> sensor observations
    -> localization + landmarks
    -> SLAM map update
    -> exploration policy
    -> movement/path command
```

## Key Changes

### Exploration Policy

Add `agents/exploration_policy.py`.

Define an `ExplorationContext` containing:

- current `PoseEstimate`;
- detached `DroneSnapshot`;
- local `SlamSnapshot`;
- local `TerrainSnapshot`;
- cave/floor constraints needed for validation;
- start position;
- frontier/exploration config values;
- battery fields reserved for later use.

Define an `ExplorationDecision` result that can represent:

- a direct next exploration target;
- a frontier target;
- a homing request;
- an exhausted/no-op state.

Move target choice and frontier prioritization out of `DroneMovementController`. Keep actual path traversal, runtime mutation, and movement timing in `DroneMovementController`.

### Localization

Add `mapping/localization.py`.

Define:

- `PoseEstimate(position, heading_deg, confidence, source, timestamp)`;
- `PerfectPoseLocalizer`, which returns the current simulated pose from a detached `DroneRuntimeState` snapshot.

Attach one localizer per drone in `Drone.__init__`.

Keep `DroneRuntimeState` as the owner of actual simulated position, heading, and path history. Localization owns the belief state only.

### Sensing And SLAM

Update `mapping/drone_sensor.py`.

On each sensor update:

1. Capture one runtime snapshot.
2. Ask the localizer for a `PoseEstimate`.
3. Build one gap-free `VisionScan` from the estimated pose.
4. Update `SlamMap` from every visible free/occupied cone cell.
5. Continue terrain sampling every second point on the scan's sparse ray hits;
   terrain samples do not contribute to exploration gain.
6. Store or expose the latest pose estimate for debugging/tests.

Do not move localization logic into `SlamMap`; `SlamMap` should remain a synchronized occupancy/confidence/point-cloud store.

### Landmarks

Add `mapping/landmarks.py` only when implementing real landmark localization.

Landmark state should be separate from occupancy storage. The later real localization flow should be:

```text
predict pose from movement/odometry
    -> observe landmarks from rays/point cloud
    -> match known landmarks
    -> correct pose estimate
    -> update landmark store
    -> update occupancy map from corrected pose
```

The first version does not need probabilistic SLAM. It needs the correct API boundary so landmark-based correction can replace the perfect-pose localizer later.

### Battery Constraints

Add battery behavior only after policy decisions are explicit and testable.

Initial work:

- keep battery state in `DroneRuntimeState`;
- include battery fields in `ExplorationContext`;
- let the first exploration policy ignore battery except for passing data through.

Later behavior:

- prevent decisions that leave insufficient reserve to return home;
- trigger return-to-base/recharge behavior;
- compare exploration value against movement energy cost.

## Tests

Add or update tests for these scenarios:

- `ExplorationPolicy` chooses a reachable frontier from local SLAM/terrain knowledge.
- `ExplorationPolicy` returns homing/exhausted when no useful frontier exists.
- Policy decisions do not read mission-global terrain telemetry.
- `PerfectPoseLocalizer` returns position and heading from a detached runtime snapshot.
- Localization does not mutate runtime state or SLAM arrays.
- `DroneSensorController.update()` uses the localizer pose as the SLAM origin.
- SLAM and terrain updates still come from one coherent captured snapshot.
- `DroneMovementController` delegates target selection to the policy while preserving current frontier rebuild and homing behavior.

## Assumptions

- Preserve current gameplay behavior during the first extraction.
- Localization v1 is an architectural boundary, not full probabilistic SLAM.
- `SlamMap` remains the map store and does not own localization.
- `DroneRuntimeState` remains the owner of mutable simulated runtime state.
- Battery rules come after exploration policy and localization boundaries are stable.
