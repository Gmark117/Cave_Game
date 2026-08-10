"""Belief-only strategic exploration policy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Tuple

from agents.drone_runtime_state import DroneSnapshot
from mapping.localization import PoseEstimate
from mapping.slam_map import SlamSnapshot
from navigation.waypoint_graph import bresenham_path


Position = Tuple[int, int]


class ExplorationDecisionKind(Enum):
    """Kinds of strategic or local decisions understood by movement."""

    STEP = "step"
    ROTATE = "rotate"
    FRONTIER = "frontier"
    HOMING = "homing"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class ExplorationDecision:
    """One detached strategic selection or bounded local primitive."""

    kind: ExplorationDecisionKind
    target: Position | None = None
    cluster_id: int | None = None
    direction: int | None = None
    frontier_targets: tuple[Position, ...] = ()
    frontier_cluster_ids: tuple[int, ...] = ()
    planned_path: tuple[Position, ...] = ()
    local_primitive: str | None = None


@dataclass(frozen=True)
class ExplorationContext:
    """Detached belief-only inputs for one policy evaluation."""

    pose_estimate: PoseEstimate
    runtime_snapshot: DroneSnapshot
    start_position: Position
    step: int
    radius: int
    frontier_confidence_threshold: float
    slam_snapshot: SlamSnapshot | None = None


class FrontierExplorationPolicy:
    """Deterministic control over stable clusters and active intents."""

    uses_strategic_control = True

    def decide(self, context: ExplorationContext) -> ExplorationDecision:
        """Select the lowest stable visible cluster ID, or home/exhaust."""
        snapshot = context.runtime_snapshot
        if snapshot.returning_home:
            return ExplorationDecision(
                kind=ExplorationDecisionKind.HOMING,
                target=context.start_position,
            )
        if snapshot.navigation_intent is not None:
            return self.decide_local(context)

        cluster_ids = snapshot.frontier_cluster_ids
        representatives = snapshot.frontiers
        if not cluster_ids or len(cluster_ids) != len(representatives):
            return ExplorationDecision(ExplorationDecisionKind.EXHAUSTED)
        ordered = tuple(sorted(zip(cluster_ids, representatives)))
        return ExplorationDecision(
            kind=ExplorationDecisionKind.FRONTIER,
            target=ordered[0][1],
            cluster_id=ordered[0][0],
            frontier_targets=tuple(position for _cluster_id, position in ordered),
            frontier_cluster_ids=tuple(cluster_id for cluster_id, _position in ordered),
        )

    def decide_local(self, context: ExplorationContext) -> ExplorationDecision:
        """Follow an active intent with deterministic bounded local control."""
        intent = context.runtime_snapshot.navigation_intent
        if intent is None:
            return ExplorationDecision(ExplorationDecisionKind.EXHAUSTED)
        if intent.mode.value == "scan":
            direction = int(round(
                float(intent.scan_base_heading)
                + (int(intent.scan_heading_cursor) + 1) * 60
            )) % 360
            return ExplorationDecision(
                kind=ExplorationDecisionKind.ROTATE,
                target=context.pose_estimate.position,
                cluster_id=intent.cluster_id,
                direction=direction,
                local_primitive="rotate_scan",
            )

        path = self._intent_prefix(
            context.pose_estimate.position,
            intent.route_paths,
            edge_cursor=intent.edge_cursor,
            polyline_cursor=intent.polyline_cursor,
            budget=float(context.step),
        )
        if len(path) <= 1:
            return ExplorationDecision(ExplorationDecisionKind.EXHAUSTED)
        start, target = path[0], path[-1]
        direction = int(round(
            math.degrees(math.atan2(target[0] - start[0], start[1] - target[1]))
        )) % 360
        primitive = (
            "recovery" if intent.mode.value == "recovery" else "follow_edge"
        )
        return ExplorationDecision(
            kind=ExplorationDecisionKind.STEP,
            target=target,
            cluster_id=intent.cluster_id,
            direction=direction,
            planned_path=path,
            local_primitive=primitive,
        )

    @staticmethod
    def _intent_prefix(
        position: Position,
        route_paths: tuple[tuple[Position, ...], ...],
        *,
        edge_cursor: int,
        polyline_cursor: int,
        budget: float,
    ) -> tuple[Position, ...]:
        """Take a deterministic at-most-budget prefix of oriented polylines."""
        selected = [position]
        travelled = 0.0
        cursor = max(0, int(edge_cursor))
        point_cursor = max(0, int(polyline_cursor))
        while cursor < len(route_paths):
            vertices = route_paths[cursor]
            raster: list[Position] = []
            for start, goal in zip(vertices, vertices[1:]):
                leg = [
                    (int(point[0]), int(point[1]))
                    for point in bresenham_path(start, goal)
                ]
                raster.extend(leg if not raster else leg[1:])
            if point_cursor >= len(raster) - 1:
                cursor += 1
                point_cursor = 0
                continue
            for index in range(point_cursor + 1, len(raster)):
                distance = math.dist(selected[-1], raster[index])
                if travelled + distance > max(0.0, budget) + 1e-9:
                    return tuple(selected)
                selected.append(raster[index])
                travelled += distance
            cursor += 1
            point_cursor = 0
        return tuple(selected)
