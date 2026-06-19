"""Drone sensing orchestration.

The sensor controller owns ray casting, local SLAM updates, and terrain
sampling. It deliberately contains no drawing code, so sensing continues
regardless of whether the vision overlay is visible.
"""

import time
from typing import Any, Iterable

import numpy as np

from RoughnessSampler import RoughnessSampler
from VisionSensor import RayHit, VisionSensor
from mapping.terrain_knowledge import TerrainSample


class DroneSensorController:
    """Update one drone's local SLAM and terrain knowledge."""

    def __init__(self, drone: Any) -> None:
        self.drone = drone
        settings = drone.settings
        scan_rays = int(getattr(settings, "slam_scan_rays", 60))

        self.scan_interval = float(
            getattr(settings, "slam_scan_interval", 0.25)
        )
        self.last_scan_time = 0.0
        self.vision_sensor = VisionSensor(
            drone.cave,
            fov_deg=60.0,
            num_rays=scan_rays,
            step=2,
        )
        self.roughness_sampler = RoughnessSampler(
            drone.control.terrain_roughness,
            drone.cave,
        )

    def update(self) -> None:
        """Cast rays and update the drone's SLAM and terrain knowledge."""
        drone = self.drone
        ray_hits = self.vision_sensor.cast_cone(drone.pos, drone.heading_deg)
        drone.ray_points = [hit.end for hit in ray_hits]

        with drone.slam_lock:
            drone.slam_map.update_from_rays(drone.pos, ray_hits)

        self.scan_terrain(ray_hits)

    def scan_terrain(self, ray_hits: Iterable[RayHit]) -> None:
        """Sample visible roughness and update local and mission terrain maps."""
        drone = self.drone
        control = drone.control
        if not hasattr(control, "terrain_roughness"):
            return

        terrain = control.terrain_roughness
        if terrain.shape != np.asarray(drone.cave).shape:
            return

        now = time.perf_counter()
        if (now - self.last_scan_time) < self.scan_interval:
            return
        self.last_scan_time = now

        self.roughness_sampler.terrain_roughness = terrain
        samples = self.roughness_sampler.sample_from_rays(
            drone.pos,
            ray_hits,
        )
        self.record_local_scan(samples)
        if hasattr(control, "record_terrain_scan"):
            control.record_terrain_scan(samples)

    def record_local_scan(self, samples: Iterable[TerrainSample]) -> None:
        """Fuse terrain observations into this drone's local knowledge maps."""
        self.drone.terrain_knowledge.record_samples(samples)
