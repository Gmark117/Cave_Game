# Codebase Cleanup and Performance Audit

Date: 2026-07-04

This audit captures likely places where the codebase has accumulated extra
layers from repeated refactors. The goal is to keep each item independently
addressable, so we can improve clarity and performance without mixing unrelated
changes.

No code changes were made as part of the audit.

## Recommended Order

1. Optimize path rendering.
2. Decide the rover runtime policy.
3. Simplify drone direction selection.
4. Consolidate ray line walking for sensing.
5. Remove or mark compatibility/test-only API.
6. Clean small unused imports and re-export leftovers.

## 1. Path Rendering Redraws Full History Every Frame

Priority: High

Type: Performance and clarity

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

## 2. Rovers Are Disabled But Still Built, Rendered, and Shared With

Priority: High

Type: Architecture and some per-frame performance

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

## 3. Drone Direction Selection Samples All 360 Degrees

Priority: Medium

Type: Performance and algorithm clarity

Files:
- `agents/drone_movement.py`

Current state:
- `DroneMovementController.find_new_node` builds 360 candidate headings.
- It checks each heading against graph collision rules.
- It builds a blacklist, then a valid direction list, then randomly removes
  invalid selected directions until a final movement target works.

Why this is roundabout:
- The code does several passes and list mutations to choose one valid movement
  direction.
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

## 4. Sensor Pipeline Walks Ray Lines Multiple Times

Priority: Medium

Type: Performance and duplication

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

## 5. Compatibility and Test-Only Surfaces Look Removable

Priority: Low to Medium

Type: Code clarity

Files:
- `config/sim_settings.py`
- `agents/drone_runtime_state.py`
- `mapping/slam_map.py`
- `ui/menu/facade.py`
- `ui/control_center/facade.py`

Current state:
- `config.SimSettings` is documented as a legacy flat adapter and appears to be
  used by tests, not current runtime construction.
- `DroneRuntimeState.set_returning_home` and `set_battery` appear test-only.
- `SlamMap.is_known` appears test-only.
- `Menu._get_first_selectable` and `_get_next_selectable` are compatibility
  wrappers around controller methods.
- `ControlCenter.percent_color` is a public helper retained as a facade wrapper.

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

## 6. Small Unused Imports and Re-Export Leftovers

Priority: Low

Type: Cleanup

Files and examples:
- `agents/graph.py`: likely unused `List`.
- `generation/map_generator.py`: `_image_path_from_key` appears to exist mainly
  for tests/compatibility.
- `mapping/terrain_fusion.py`: `fuse_terrain_samples` import appears to be a
  re-export for tests.
- `rendering/slam_renderer.py`: likely unused `FREE`.
- `tools/generate_outlined_icons.py`: likely unused `ImageDraw`.
- `ui/control_center/panels.py`: likely unused `pygame`.

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

## Notes

- The path-rendering issue is the most promising direct framerate improvement.
- The rover issue is the biggest architecture cleanup decision.
- The ray-walking issue is a good second performance pass, but it touches more
  behavior and should be done after the simpler rendering fix.
