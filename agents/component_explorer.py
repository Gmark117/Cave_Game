"""Drone-local scan plans and depth-first frontier lineage state."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import cv2
import numpy as np

from mapping.frontiers import (
    eight_connected_components,
    significant_frontier_mask,
)
from mapping.slam_map import UNKNOWN, SlamSnapshot


Position = tuple[int, int]


@dataclass
class ScanPlanProgress:
    """Progress through headings serviced by the ordinary sensor scheduler."""

    position: Position
    headings: tuple[int, ...]
    started_at: float
    heading_index: int = 0
    requested_at: float | None = None
    minimum_scan_sequence: int = -1
    completed_headings: list[int] = field(default_factory=list)
    timed_out_headings: list[int] = field(default_factory=list)
    newly_known_cells: int = 0
    confidence_gain: float = 0.0

    @property
    def complete(self) -> bool:
        return self.heading_index >= len(self.headings)

    @property
    def current_heading(self) -> int | None:
        if self.complete:
            return None
        return int(self.headings[self.heading_index]) % 360


@dataclass(frozen=True)
class LocalComponentNode:
    """One provisional successor explored under an authoritative claim."""

    local_id: int
    cells: frozenset[Position]
    anchor_position: Position
    scan_heading: int
    work_unit_id: int | None = None


@dataclass
class LocalDFSFrame:
    node: LocalComponentNode
    parent_position: Position
    outbound_actual_path: tuple[Position, ...] = ()
    pending_children: list[LocalComponentNode] = field(default_factory=list)
    scanned: bool = False


class LocalDFSStack:
    """Bounded LIFO traversal of component successors."""

    def __init__(self, *, maximum_depth: int = 32, maximum_nodes: int = 128):
        if maximum_depth <= 0 or maximum_nodes <= 0:
            raise ValueError("DFS limits must be positive")
        self.maximum_depth = int(maximum_depth)
        self.maximum_nodes = int(maximum_nodes)
        self.frames: list[LocalDFSFrame] = []
        self.visited_geometry: set[frozenset[Position]] = set()
        self.visited_anchors: set[Position] = set()
        self.leaf_geometry: list[frozenset[Position]] = []
        self._next_local_id = 0
        self.created_nodes = 0
        self.limit_reached = False

    def new_node(
        self,
        cells: frozenset[Position],
        anchor_position: Position,
        scan_heading: int,
        work_unit_id: int | None = None,
    ) -> LocalComponentNode:
        node = LocalComponentNode(
            local_id=self._next_local_id,
            cells=cells,
            anchor_position=anchor_position,
            scan_heading=int(scan_heading) % 360,
            work_unit_id=work_unit_id,
        )
        self._next_local_id += 1
        return node

    def start(self, node: LocalComponentNode, *, parent_position: Position) -> None:
        if self.frames:
            raise RuntimeError("DFS stack is already active")
        self.frames.append(LocalDFSFrame(node, parent_position))
        self.created_nodes = 1

    @property
    def current(self) -> LocalDFSFrame | None:
        return self.frames[-1] if self.frames else None

    def record_outbound(self, path: Iterable[Position]) -> None:
        if self.current is None:
            return
        self.current.outbound_actual_path = tuple(path)

    def record_scan(
        self,
        successors: Iterable[LocalComponentNode],
    ) -> LocalComponentNode | None:
        frame = self.current
        if frame is None:
            return None
        frame.scanned = True
        self.visited_geometry.add(frame.node.cells)
        self.visited_anchors.add(frame.node.anchor_position)
        candidates_by_geometry = {
            item.cells: item for item in successors
            if item.cells not in self.visited_geometry
        }
        candidates = list(candidates_by_geometry.values())
        if len(self.frames) >= self.maximum_depth:
            self.limit_reached = bool(candidates)
            candidates = []
        else:
            capacity = max(0, self.maximum_nodes - self.created_nodes)
            if len(candidates) > capacity:
                self.limit_reached = True
                candidates = candidates[:capacity]
            self.created_nodes += len(candidates)
        frame.pending_children = list(candidates)
        if not frame.pending_children:
            self.leaf_geometry.append(frame.node.cells)
            return None
        child = frame.pending_children.pop(0)
        self.frames.append(LocalDFSFrame(
            child,
            parent_position=frame.node.anchor_position,
        ))
        return child

    def pop_completed(self) -> tuple[tuple[Position, ...], LocalComponentNode | None]:
        """Pop one leaf and return its exact reverse path and next sibling."""
        if not self.frames:
            return (), None
        completed = self.frames.pop()
        backtrack = tuple(reversed(completed.outbound_actual_path))
        if not self.frames:
            return backtrack, None
        parent = self.frames[-1]
        if parent.pending_children:
            sibling = parent.pending_children.pop(0)
            self.frames.append(LocalDFSFrame(
                sibling,
                parent_position=parent.node.anchor_position,
            ))
            return backtrack, sibling
        return backtrack, None


def related_local_successors(
    slam: SlamSnapshot,
    source_cells: frozenset[Position],
    *,
    confidence_threshold: float,
    minimum_component_cells: int,
    minimum_unknown_support_cells: int,
    lineage_radius: int,
    scan_origin: Position | None = None,
    scan_heading: float | None = None,
    sensor_range: float = 0.0,
    sensor_fov_deg: float = 0.0,
) -> tuple[frozenset[Position], ...]:
    """Return significant successors near lineage or inside the last scan."""
    frontier, _diagnostics = significant_frontier_mask(
        np.asarray(slam.occupancy),
        np.asarray(slam.confidence),
        confidence_threshold,
        minimum_component_cells=minimum_component_cells,
        minimum_unknown_support_cells=minimum_unknown_support_cells,
    )
    if not source_cells:
        return ()
    radius = max(0, int(lineage_radius))
    successors = []
    for component in eight_connected_components(frontier):
        cells = frozenset(component)
        lineage_near = _sets_near(
            source_cells,
            cells,
            radius=radius,
            shape=frontier.shape,
        )
        scan_visible = bool(
            scan_origin is not None
            and scan_heading is not None
            and sensor_range > 0.0
            and sensor_fov_deg > 0.0
            and any(
                _inside_scan_footprint(
                    scan_origin,
                    scan_heading,
                    point,
                    sensor_range=sensor_range,
                    sensor_fov_deg=sensor_fov_deg,
                )
                for point in cells
            )
        )
        if lineage_near or scan_visible:
            successors.append(cells)
    return tuple(successors)


def _sets_near(
    first: frozenset[Position],
    second: frozenset[Position],
    *,
    radius: int,
    shape: tuple[int, int],
) -> bool:
    if not first or not second:
        return False
    if first & second:
        return True
    first_x = tuple(point[0] for point in first)
    first_y = tuple(point[1] for point in first)
    second_x = tuple(point[0] for point in second)
    second_y = tuple(point[1] for point in second)
    first_box = (min(first_x), min(first_y), max(first_x), max(first_y))
    second_box = (min(second_x), min(second_y), max(second_x), max(second_y))
    if (
        first_box[2] + radius < second_box[0]
        or second_box[2] + radius < first_box[0]
        or first_box[3] + radius < second_box[1]
        or second_box[3] + radius < first_box[1]
    ):
        return False
    height, width = shape
    left = max(0, min(first_box[0], second_box[0]) - radius)
    top = max(0, min(first_box[1], second_box[1]) - radius)
    right = min(width - 1, max(first_box[2], second_box[2]) + radius)
    bottom = min(height - 1, max(first_box[3], second_box[3]) + radius)
    local = np.zeros((bottom - top + 1, right - left + 1), dtype=np.uint8)
    for x, y in first:
        local[y - top, x - left] = 1
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    envelope = cv2.dilate(local, kernel).astype(bool)
    return any(envelope[y - top, x - left] for x, y in second)


def _inside_scan_footprint(
    origin: Position,
    heading: float,
    point: Position,
    *,
    sensor_range: float,
    sensor_fov_deg: float,
) -> bool:
    distance = math.dist(origin, point)
    if distance > float(sensor_range) + 1e-9:
        return False
    if distance <= 1e-9:
        return True
    bearing = math.degrees(math.atan2(
        point[0] - origin[0],
        -(point[1] - origin[1]),
    )) % 360.0
    delta = abs((bearing - float(heading) + 180.0) % 360.0 - 180.0)
    return delta <= float(sensor_fov_deg) / 2.0 + 1e-9


def local_node_pose(
    cells: frozenset[Position],
    slam: SlamSnapshot,
    *,
    origin: Position,
    excluded_anchors: Iterable[Position] = (),
    minimum_anchor_spacing: float = 0.0,
    preferred_heading: float | None = None,
) -> tuple[Position, int] | None:
    """Choose a deterministic nearby anchor and heading into local unknown."""
    if not cells:
        return None
    excluded = tuple(excluded_anchors)
    spacing = max(0.0, float(minimum_anchor_spacing))
    candidates = tuple(
        point for point in cells
        if all(math.dist(point, anchor) + 1e-9 >= spacing
               for anchor in excluded)
    )
    if not candidates:
        return None
    if preferred_heading is None:
        anchor = min(
            candidates,
            key=lambda point: (
                math.dist(origin, point),
                point[1],
                point[0],
            ),
        )
    else:
        radians = math.radians(float(preferred_heading) % 360.0)
        direction_x = math.sin(radians)
        direction_y = -math.cos(radians)
        anchor = min(
            candidates,
            key=lambda point: (
                -(
                    (point[0] - origin[0]) * direction_x
                    + (point[1] - origin[1]) * direction_y
                ),
                math.dist(origin, point),
                point[1],
                point[0],
            ),
        )
    occupancy = np.asarray(slam.occupancy)
    confidence = np.asarray(slam.confidence)
    unknown = (
        (occupancy == UNKNOWN)
        | (confidence <= 0.0)
    )
    x, y = anchor
    height, width = unknown.shape
    neighbors = [
        (neighbor_x, neighbor_y)
        for neighbor_y in range(max(0, y - 1), min(height, y + 2))
        for neighbor_x in range(max(0, x - 1), min(width, x + 2))
        if unknown[neighbor_y, neighbor_x]
    ]
    if not neighbors:
        return anchor, 0
    target_x = sum(item[0] for item in neighbors) / len(neighbors)
    target_y = sum(item[1] for item in neighbors) / len(neighbors)
    heading = int(round(math.degrees(math.atan2(
        target_x - x,
        -(target_y - y),
    )))) % 360
    return anchor, heading
