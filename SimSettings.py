from dataclasses import dataclass


@dataclass
class SimSettings:
    mission: int = 0
    map_dim: str = "Medium"
    seed: int = 0
    num_drones: int = 3


    slam_scan_interval: float = 0.25
    slam_scan_rays: int = 60
    slam_point_cloud_max_points: int = 6000
    slam_render_point_tail: int = 400
    frontier_stride: int = 4
    frontier_confidence_threshold: float = 0.6
    frontier_rebuild_cooldown: float = 0.25
