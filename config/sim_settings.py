"""Legacy flat constructor for :mod:`SimulationConfig`.

New code should construct and consume ``SimulationConfig`` with its nested
sections. ``SimSettings`` remains as a read-only adapter for older callers.
"""

from __future__ import annotations

from config.simulation_config import (
    FrontierConfig,
    MissionConfig,
    RenderingConfig,
    SharingConfig,
    SimulationConfig,
    SlamConfig,
)


class SimSettings(SimulationConfig):
    """Build nested configuration from the former flat keyword arguments."""

    def __init__(
        self,
        mission: int = 0,
        map_dim: str = "Medium",
        seed: int = 0,
        num_drones: int = 3,
        slam_scan_interval: float = 0.25,
        slam_scan_rays: int = 60,
        slam_point_cloud_max_points: int = 6000,
        slam_render_point_tail: int = 400,
        slam_render_interval: float = 0.1,
        rover_share_interval: float = 0.5,
        frontier_stride: int = 4,
        frontier_confidence_threshold: float = 0.6,
        frontier_rebuild_cooldown: float = 0.25,
        drone_share_interval: float = 0.5,
        pair_share_cooldown: float = 1.2,
        share_compare_stride: int = 8,
        min_share_new_info_ratio: float = 0.04,
        min_share_overlap_diff_ratio: float = 0.18,
        min_share_roughness_delta: float = 0.12,
    ) -> None:
        super().__init__(
            mission_config=MissionConfig(
                objective=mission,
                map_dim=map_dim,
                seed=seed,
                num_drones=num_drones,
            ),
            slam=SlamConfig(
                scan_interval=slam_scan_interval,
                scan_rays=slam_scan_rays,
                point_cloud_max_points=slam_point_cloud_max_points,
            ),
            sharing=SharingConfig(
                drone_interval=drone_share_interval,
                pair_cooldown=pair_share_cooldown,
                rover_interval=rover_share_interval,
                compare_stride=share_compare_stride,
                min_new_info_ratio=min_share_new_info_ratio,
                min_overlap_diff_ratio=min_share_overlap_diff_ratio,
                min_roughness_delta=min_share_roughness_delta,
            ),
            frontier=FrontierConfig(
                stride=frontier_stride,
                confidence_threshold=frontier_confidence_threshold,
                rebuild_cooldown=frontier_rebuild_cooldown,
            ),
            rendering=RenderingConfig(
                point_tail=slam_render_point_tail,
                refresh_interval=slam_render_interval,
            ),
        )

    @property
    def mission(self) -> int:
        return self.mission_config.objective

    @property
    def map_dim(self) -> str:
        return self.mission_config.map_dim

    @property
    def seed(self) -> int:
        return self.mission_config.seed

    @property
    def num_drones(self) -> int:
        return self.mission_config.num_drones

    @property
    def slam_scan_interval(self) -> float:
        return self.slam.scan_interval

    @property
    def slam_scan_rays(self) -> int:
        return self.slam.scan_rays

    @property
    def slam_point_cloud_max_points(self) -> int:
        return self.slam.point_cloud_max_points

    @property
    def slam_render_point_tail(self) -> int:
        return self.rendering.point_tail

    @property
    def slam_render_interval(self) -> float:
        return self.rendering.refresh_interval

    @property
    def rover_share_interval(self) -> float:
        return self.sharing.rover_interval

    @property
    def frontier_stride(self) -> int:
        return self.frontier.stride

    @property
    def frontier_confidence_threshold(self) -> float:
        return self.frontier.confidence_threshold

    @property
    def frontier_rebuild_cooldown(self) -> float:
        return self.frontier.rebuild_cooldown
