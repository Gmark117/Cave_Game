"""Drone sensing orchestration.

The sensor controller owns ray casting, local SLAM updates, and terrain
sampling. It deliberately contains no drawing code, so sensing continues
regardless of whether the vision overlay is visible.
"""

from typing import Any, Iterable

import numpy as np

from mapping.roughness_sampler import RoughnessSampler
from mapping.vision_sensor import RayHit, VisionSensor
from mapping.terrain_knowledge import TerrainSample
from mission.service_dependencies import DroneSensorDependencies


class DroneSensorController:
    """Update one drone's local SLAM and terrain knowledge."""

    def __init__(
        self,
        drone: Any,
        dependencies: DroneSensorDependencies,
    ) -> None:
        self.drone = drone
        self.dependencies = dependencies
        settings = drone.settings
        scan_rays = settings.slam.scan_rays

        self.scan_interval = settings.slam.scan_interval
        self.last_scan_time = 0.0
        self.vision_sensor = VisionSensor(
            drone.cave,
            fov_deg=60.0,
            num_rays=scan_rays,
            step=2,
        )
        self.roughness_sampler = RoughnessSampler(
            dependencies.terrain_roughness,
            drone.cave,
        )

    def update(self) -> None:
        """Cast rays and update the drone's SLAM and terrain knowledge."""
        drone = self.drone
        snapshot = drone.snapshot()
        origin = snapshot.position
        ray_hits = self.vision_sensor.cast_cone(
            origin,
            snapshot.heading_deg,
        )
        drone.runtime_state.set_ray_points(hit.end for hit in ray_hits)

        drone.slam_map.update_from_rays(origin, ray_hits)

        self.scan_terrain(ray_hits, origin=origin)

    def scan_terrain(
        self,
        ray_hits: Iterable[RayHit],
        origin: tuple[int, int] | None = None,
    ) -> None:
        """Sample visible roughness and update local and mission terrain maps."""
        drone = self.drone
        terrain = self.dependencies.terrain_roughness
        if terrain.shape != np.asarray(drone.cave).shape:
            return

        now = self.dependencies.simulation_time()
        if (now - self.last_scan_time) < self.scan_interval:
            return
        self.last_scan_time = now

        self.roughness_sampler.terrain_roughness = terrain
        if origin is None:
            origin = drone.snapshot().position
        samples = self.roughness_sampler.sample_from_rays(
            origin,
            ray_hits,
        )
        self.record_local_scan(samples)
        self.dependencies.record_terrain_scan(samples)

    def record_local_scan(self, samples: Iterable[TerrainSample]) -> None:
        """Fuse terrain observations into this drone's local knowledge maps."""
        self.drone.terrain_knowledge.record_samples(samples)
