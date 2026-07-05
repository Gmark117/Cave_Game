# Codebase Cleanup and Performance Audit

Date: 2026-07-04

This audit captures likely places where the codebase has accumulated extra
layers from repeated refactors. The goal is to keep each item independently
addressable, so we can improve clarity and performance without mixing unrelated
changes.

The initial audit made no code changes. Items are updated as they are addressed.

## Recommended Order

1. Optimize path rendering. Addressed.
2. Decide the rover runtime policy. Deferred by design.
3. Simplify drone direction selection. Partially addressed.
4. Consolidate ray line walking for sensing. Addressed.
5. Remove or mark compatibility/test-only API. Addressed.
6. Clean small unused imports and re-export leftovers. Addressed.

## 1. Path Rendering Redraws Full History Every Frame

Priority: High

Type: Performance and clarity

Status: Addressed

Files:
- `rendering/agent_renderer.py`

Current state:
- `DroneRenderer.draw_path` loops through every segment in
  `snapshot.path_history` every rendered frame.
- `RoverRenderer.draw_path` does the same for `rover.graph.pos`.
- Both renderers own persistent transparent `path_surface` objects, so old
  path segments do not need to be redrawn every frame.
- Drone path segments are redrawn even before checking `snapshot.show_path`.

Why this is roundabout:
- The surface already accumulates path pixels, but the code still treats path
  rendering as if it must rebuild the full history every frame.
- Work grows as the mission gets longer.

Suggested fix:
- Store the last rendered path index in each renderer.
- Draw only newly appended segments onto the persistent surface.
- Blit or skip the cached surface depending on the visibility flag.
- Reset the cache/index if a mission restarts or the path history is replaced.

Expected benefit:
- Lower render cost during longer missions.
- Simpler mental model: path drawing becomes incremental.

Validation:
- Unit test that repeated `draw_path` calls do not redraw old segments.
- Manual smoke test with path toggle on/off.

Resolution:
- `DroneRenderer` and `RoverRenderer` now track how many path points have
  already been painted to their persistent path surfaces.
- Repeated frame draws reuse the cached surface and draw only newly appended
  path segments.
- If a path history ever shrinks, the renderer clears the cached path surface
  and rebuilds from the new history.
- Added unit coverage for drone and rover incremental path drawing.

## 2. Rovers Are Disabled But Still Built, Rendered, and Shared With

Priority: High

Type: Architecture and some per-frame performance

Status: Deferred by design

Files:
- `mission/control.py`
- `mission/lifecycle.py`
- `agents/factory.py`
- `agents/rover.py`
- `mapping/terrain_sharing.py`
- `rendering/mission_renderer.py`

Current state:
- `MissionControl.rover_motion_enabled` is set to `False`.
- Rover worker threads are not started.
- Rovers are still constructed by `AgentFactory.build_rovers`.
- Rovers still have terrain knowledge, path surfaces, target services, and
  renderer state.
- The mission loop still calls `terrain_sharing.share_with_rovers()` each frame
  while unpaused.
- The renderer still iterates rovers and draws their paths/icons.

Why this is roundabout:
- The code is half-disabled: rover motion is off, but supporting runtime work
  still exists.
- This makes the code harder to reason about because "disabled" does not mean
  "absent from runtime".

Suggested options:
- Option A: fully disable rover runtime until policy is defined.
  - Do not build rovers.
  - Do not share with rovers.
  - Do not render rovers.
  - Keep rover classes/tests if future work needs them.
- Option B: make rovers first-class again.
  - Define rover-local knowledge policy.
  - Enable worker threads.
  - Make target selection use rover-local received knowledge rather than mission
    aggregate telemetry.

Expected benefit:
- Clearer mission runtime.
- Less per-frame work if choosing Option A.

Validation:
- If Option A: tests should confirm no rovers are built/rendered/shared while
  disabled.
- If Option B: add rover-local policy tests before enabling motion.

Decision:
- Keep rovers built, rendered, and available for sharing during this cleanup
  pass.
- The next project phase will define the joint drone/rover exploration policy,
  so removing rover runtime wiring now would likely create churn rather than
  lasting clarity.
- Rover motion remains disabled until that policy is defined.

## 3. Drone Direction Selection Samples All 360 Degrees

Priority: Medium

Type: Performance and algorithm clarity

Status: Partially addressed

Files:
- `agents/drone_movement.py`

Current state:
- `DroneMovementController.find_new_node` builds 360 candidate headings.
- It checks each heading against graph collision rules.
- It builds valid direction and frontier-target lists directly.
- If the randomly selected short movement step is invalid, it removes that
  aligned direction/target pair and retries.

Why this is roundabout:
- Checking every integer degree is probably more precision than the movement
  behavior needs.

Suggested fix:
- Introduce a configured angular resolution, for example 16, 24, or 36
  directions.
- Precompute direction offsets by map size or drone radius.
- Build valid candidates directly instead of building and subtracting a
  blacklist.

Expected benefit:
- Fewer collision checks per movement decision.
- Simpler movement selection code.

Validation:
- Existing movement tests should still pass.
- Add a test that candidates are generated from the configured resolution.
- Manual smoke test to verify drones still explore naturally.

Resolution:
- Removed the old blacklist and preallocated target matrix while preserving the
  current 360-integer-heading exploration behavior.
- Direction candidates and their frontier targets are now built directly in
  matching lists.
- Added tests that pin the current 360-heading behavior and the rejected-step
  retry path.

Deferred:
- Reducing the angular resolution is a simulation-model decision and should be
  handled during the exploration-policy phase.

## 4. Sensor Pipeline Walks Ray Lines Multiple Times

Priority: Medium

Type: Performance and duplication

Status: Addressed

Files:
- `mapping/vision_sensor.py`
- `mapping/drone_sensor.py`
- `mapping/slam_map.py`
- `mapping/roughness_sampler.py`

Current state:
- `VisionSensor` walks each ray to find the hit endpoint.
- `SlamMap.update_from_rays` walks the line again to mark free/occupied cells.
- `RoughnessSampler.sample_from_rays` walks sampled points along the same ray
  again for terrain roughness.

Why this is roundabout:
- One sensor observation is split into multiple line-walking passes.
- The same geometric path is rediscovered by separate helpers.

Suggested fix:
- Extend `RayHit` to include traversed cells, or add a shared ray-line helper
  used by both SLAM and roughness sampling.
- Keep the public sensor behavior the same while reducing duplicate work.

Expected benefit:
- Less repeated math per sensor update.
- Fewer subtly different line-walking implementations.

Validation:
- SLAM ray tests should still pass.
- Roughness sampler tests should still pass.
- Add a test that `RayHit` or the shared helper preserves endpoint behavior.

Resolution:
- Profiling after item 1 showed that sensing, not path rendering, was the
  largest frame-time slice.
- `RayHit` now carries the grid cells traversed by the sensor ray.
- `SlamMap` and `RoughnessSampler` reuse supplied ray points when present and
  fall back to their previous line-generation behavior for compatibility.
- A shared `mapping/ray_geometry.py` helper now owns Bresenham line generation.
- Drone ray range remains unchanged: live sensors still scan until they hit a
  wall or leave the map. `drone.radius` is not used as LiDAR range.
- `VisionSensor.max_range` remains an optional direct sensor setting for future
  experiments, but mission construction does not wire it to the legacy radius.
- Added tests for optional max-range raycasting, downstream reuse of supplied
  points, and live drone range staying uncapped by `drone.radius`.

Follow-up profiling:
- When ray range stayed unlimited, the ray-geometry reuse improved framerate by
  at most about 1 FPS. This cleanup is still useful for code clarity, but it is
  not a meaningful framerate fix unless paired with an explicit sensor-model
  decision such as shorter range, fewer rays, or lower sensing cadence.

## 5. Compatibility and Test-Only Surfaces Look Removable

Priority: Low to Medium

Type: Code clarity

Status: Addressed

Files:
- `config/sim_settings.py`
- `agents/drone_runtime_state.py`
- `mapping/slam_map.py`
- `ui/menu/facade.py`
- `ui/control_center/facade.py`

Current state:
- `config.SimSettings` is documented as a legacy flat adapter and is kept as an
  intentional compatibility API.
- `ControlCenter.percent_color` is kept as an intentional public facade helper.
- Test-only drone runtime setters, `SlamMap.is_known`, and private menu
  controller wrappers have been removed.

Why this is roundabout:
- These APIs may have been kept to avoid breaking tests during refactors.
- If no runtime or external caller needs them, they add surface area and make
  the intended ownership boundaries less crisp.

Suggested fix:
- Decide whether these are real public API or only transitional helpers.
- If transitional, remove them and update tests to target the owning class
  directly.
- If public, mark them clearly as compatibility API and keep them intentionally.

Expected benefit:
- Smaller public surface.
- Tests describe the current design rather than older class boundaries.

Validation:
- Unit suite after removing each wrapper independently.

Resolution:
- Removed `DroneRuntimeState.set_returning_home` and `set_battery`.
- Removed `SlamMap.is_known`.
- Removed `Menu._get_first_selectable` and `_get_next_selectable`.
- Updated tests to build display snapshots from current runtime behavior rather
  than relying on test-only runtime mutators.
- Updated `docs/CODEFLOW.md` so it no longer lists the removed SLAM helper.
- Left `SimSettings` and `ControlCenter.percent_color` in place because they are
  explicit compatibility/facade surfaces rather than accidental leftovers.

## 6. Small Unused Imports and Re-Export Leftovers

Priority: Low

Type: Cleanup

Status: Addressed

Files and examples:
- `agents/graph.py`: removed unused `List`.
- `generation/map_generator.py`: removed `_image_path_from_key` re-export.
- `mapping/terrain_fusion.py`: removed `fuse_terrain_samples` re-export.
- `rendering/slam_renderer.py`: removed unused `FREE`.
- `tools/generate_outlined_icons.py`: removed unused `ImageDraw` and unused
  image-size assignment.
- `ui/control_center/panels.py`: removed unused `pygame`.

Why this is roundabout:
- These are small remnants from earlier module shapes.
- They do not meaningfully affect frame rate, but they make imports noisier.

Suggested fix:
- Remove one module's unused imports at a time.
- If tests import through a compatibility path, either update the test or make
  the re-export explicit in an `__all__`.

Expected benefit:
- Cleaner modules and slightly clearer dependencies.

Validation:
- `python -m compileall -q .`
- `python -m unittest discover -s tests -v`

Resolution:
- Updated tests to import image-path resolution from
  `generation.map_artifact_writer` and terrain fusion from
  `mapping.terrain_knowledge`, matching the owning modules.
- Updated `docs/CODEFLOW.md` to describe `image_path_from_key` on
  `map_artifact_writer.py` instead of `map_generator.py`.

## Notes

- The cleanup pass removed several accumulated indirections and made the
  remaining compatibility surfaces explicit.
- Path-rendering and ray-geometry cleanup improved clarity, but profiling showed
  little framerate impact when LiDAR range remains unlimited.
- Further meaningful framerate gains probably require explicit model decisions:
  sensor range, ray count, sensing cadence, angular movement resolution, or the
  upcoming drone/rover exploration policy.
