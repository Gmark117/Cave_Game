"""Drone sensing orchestration.

The sensor controller owns ray casting, local SLAM updates, and terrain
sampling. It deliberately contains no drawing code, so sensing continues
regardless of whether the vision overlay is visible.
"""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from mapping.roughness_sampler import RoughnessSampler
from mapping.vision_sensor import RayHit, VisionSensor
from mapping.terrain_knowledge import TerrainSample
from contracts import DroneSensorDependencies


LIDAR_RANGE_RADIUS_MULTIPLIER = 4


@dataclass(frozen=True)
class SensorScanCompletion:
    """Sensor-local result for one fully applied scan pose."""

    pose: tuple[int, int, float]
    sequence: int
    newly_known_cells: int
    confidence_gain: float


class DroneSensorController:
    """Update one drone's local SLAM and terrain knowledge."""

    def __init__(
        self,
        drone: Any,
        dependencies: DroneSensorDependencies,
    ) -> None:
        """Create raycasting and terrain-sampling helpers for one drone."""
        self.drone = drone
        self.dependencies = dependencies
        settings = drone.settings
        scan_rays = settings.slam.scan_rays
        lidar_max_range = drone.radius * LIDAR_RANGE_RADIUS_MULTIPLIER

        self.scan_interval = settings.slam.scan_interval
        self.last_scan_time = 0.0
        self.vision_sensor = VisionSensor(
            drone.cave,
            fov_deg=60.0,
            num_rays=scan_rays,
            step=2,
            max_range=lidar_max_range,
        )
        self.roughness_sampler = RoughnessSampler(
            dependencies.terrain_roughness,
            drone.cave,
        )
        self.latest_pose_estimate = None
        self._last_scan_pose: tuple[int, int, float] | None = None
        self._last_completed_scan: SensorScanCompletion | None = None
        self._last_skip_pose_logged: tuple[int, int, float] | None = None

    @property
    def last_completed_scan(self) -> SensorScanCompletion | None:
        """Return the latest scan whose SLAM update has fully completed."""
        return self._last_completed_scan

    def update(self) -> None:
        """Cast rays and update the drone's SLAM and terrain knowledge."""
        drone = self.drone
        snapshot = drone.snapshot()
        now = self.dependencies.simulation_time()
        pose_estimate = drone.localizer.estimate(
            snapshot,
            timestamp=now,
        )
        self.latest_pose_estimate = pose_estimate
        pose_signature = self._pose_signature(pose_estimate)
        progress_before = drone.slam_map.progress_snapshot()
        if pose_signature == self._last_scan_pose:
            if self._last_skip_pose_logged != pose_signature:
                self._trace(
                    "sensor_pose_static_skip",
                    pose=pose_signature,
                    slam_version=drone.slam_map.version,
                )
                self._last_skip_pose_logged = pose_signature
                return
            return

        origin = pose_estimate.position
        vision_scan = self.vision_sensor.scan_cone(
            origin,
            pose_estimate.heading_deg,
        )
        ray_hits = vision_scan.ray_hits
        self._last_scan_pose = pose_signature
        self._last_skip_pose_logged = None
        # The renderer and terrain sampler consume sparse rays. SLAM consumes
        # the independent dense free/occupied visibility result below.
        drone.runtime_state.set_ray_points(hit.end for hit in ray_hits)

        slam_updated = drone.slam_map.update_from_observations(
            origin,
            free_cells=vision_scan.free_cells,
            occupied_cells=vision_scan.occupied_cells,
        )

        terrain_samples = self.scan_terrain(
            ray_hits,
            origin=origin,
            now=now,
        )
        progress_after = drone.slam_map.progress_snapshot()
        newly_known_cells = (
            progress_after.sensor_newly_known_cells
            - progress_before.sensor_newly_known_cells
        )
        confidence_gain = (
            progress_after.sensor_confidence_gain
            - progress_before.sensor_confidence_gain
        )
        self._last_completed_scan = SensorScanCompletion(
            pose=pose_signature,
            sequence=progress_after.completed_scan_sequence,
            newly_known_cells=newly_known_cells,
            confidence_gain=confidence_gain,
        )
        self._trace(
            "sensor_scan",
            pose=pose_signature,
            ray_count=len(ray_hits),
            visible_cell_count=(
                len(vision_scan.free_cells) + len(vision_scan.occupied_cells)
            ),
            visible_free_cells=len(vision_scan.free_cells),
            visible_occupied_cells=len(vision_scan.occupied_cells),
            slam_updated=slam_updated,
            slam_version_before=progress_before.version,
            slam_version_after=progress_after.version,
            completed_scan_sequence=(
                progress_after.completed_scan_sequence
            ),
            newly_known_cells=newly_known_cells,
            confidence_gain=confidence_gain,
            cumulative_sensor_newly_known_cells=(
                progress_after.sensor_newly_known_cells
            ),
            cumulative_sensor_confidence_gain=(
                progress_after.sensor_confidence_gain
            ),
            terrain_samples=terrain_samples,
        )

    def scan_terrain(
        self,
        ray_hits: Iterable[RayHit],
        origin: tuple[int, int] | None = None,
        now: float | None = None,
    ) -> int:
        """Sample visible roughness and update local and mission terrain maps."""
        drone = self.drone
        terrain = self.dependencies.terrain_roughness
        if terrain.shape != np.asarray(drone.cave).shape:
            return 0

        if now is None:
            now = self.dependencies.simulation_time()
        if (now - self.last_scan_time) < self.scan_interval:
            return 0
        self.last_scan_time = now

        # Terrain roughness is generated once with the cave. The sampler adds
        # small noise and confidence so repeated observations can be fused.
        self.roughness_sampler.terrain_roughness = terrain
        if origin is None:
            origin = drone.snapshot().position
        samples = self.roughness_sampler.sample_from_rays(
            origin,
            ray_hits,
            step=2,
        )
        self.record_local_scan(samples)
        self.dependencies.record_terrain_scan(samples)
        return len(samples)

    def record_local_scan(self, samples: Iterable[TerrainSample]) -> None:
        """Fuse terrain observations into this drone's local knowledge maps."""
        self.drone.terrain_knowledge.record_samples(samples)

    @staticmethod
    def _pose_signature(pose_estimate: Any) -> tuple[int, int, float]:
        """Return a stable key for sensor poses that produce identical rays."""
        position = pose_estimate.position
        return (
            int(round(position[0])),
            int(round(position[1])),
            round(float(pose_estimate.heading_deg) % 360.0, 3),
        )

    def _trace(self, event: str, **fields: Any) -> None:
        """Write one sensor trace event when runtime tracing is enabled."""
        trace = getattr(self.dependencies, "runtime_trace", None)
        if trace is None:
            return
        trace.record(
            event,
            sim_time=self.dependencies.simulation_time(),
            drone_id=self.drone.id,
            **fields,
        )
