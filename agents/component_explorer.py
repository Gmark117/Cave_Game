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
from mapping.ray_geometry import bresenham_line_points
from mapping.slam_map import FREE, OCCUPIED, UNKNOWN, SlamSnapshot


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
class FrontierObservationPose:
    """A reachable route prefix from which one small frontier can be scanned."""

    position: Position
    heading: int
    route_prefix: tuple[Position, ...]
    saved_route_distance: float


@dataclass(frozen=True)
class IncidentalPocketSignature:
    """Stable, trace-safe geometry used to suppress repeated side scans."""

    support_bins: tuple[Position, ...]
    gateway_bin: Position
    wall_bin: Position


@dataclass(frozen=True)
class IncidentalScanCandidate:
    """One enclosed wall-adjacent unknown pocket visible from a route pose."""

    signature: IncidentalPocketSignature
    cells: frozenset[Position]
    gateway: Position
    wall_contact: Position
    wall_cells: int
    heading: int
    heading_delta: float
    current_visibility: int
    proposed_visibility: int
    side_only_support: int


@dataclass(frozen=True)
class IncidentalScanDetection:
    """Bounded detector result and deterministic rejection accounting."""

    candidate: IncidentalScanCandidate | None
    component_count: int
    rejection_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class LocalComponentNode:
    """One provisional successor explored under an authoritative claim."""

    local_id: int
    cells: frozenset[Position]
    anchor_position: Position
    scan_heading: int
    work_unit_id: int | None = None
    allow_standoff: bool = False


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
        allow_standoff: bool = False,
    ) -> LocalComponentNode:
        node = LocalComponentNode(
            local_id=self._next_local_id,
            cells=cells,
            anchor_position=anchor_position,
            scan_heading=int(scan_heading) % 360,
            work_unit_id=work_unit_id,
            allow_standoff=bool(allow_standoff),
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
        """Pop one leaf; retain its reverse path for navigation fallback."""
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


def observation_pose_on_path(
    cells: frozenset[Position],
    slam: SlamSnapshot,
    path: Iterable[Position],
    *,
    origin: Position,
    sensor_range: float,
    sensor_fov_deg: float,
    confidence_threshold: float,
) -> FrontierObservationPose | None:
    """Choose the earliest known-clear scan pose on an existing route.

    The route remains the reachability authority.  This helper only removes a
    suffix when the complete small frontier and its adjacent unknown support
    fit inside one locally known-clear sensor cone.  It never consults the
    simulator's global cave map.
    """
    if not cells or sensor_range <= 0.0 or sensor_fov_deg <= 0.0:
        return None

    route: list[Position] = [
        (int(origin[0]), int(origin[1])),
    ]
    for point in path:
        normalized = (int(point[0]), int(point[1]))
        if normalized != route[-1]:
            route.append(normalized)

    occupancy = np.asarray(slam.occupancy)
    confidence = np.asarray(slam.confidence)
    offset_x, offset_y = (int(value) for value in slam.origin)
    height, width = occupancy.shape
    threshold = float(confidence_threshold)

    def local_index(point: Position) -> tuple[int, int] | None:
        local_x = int(point[0]) - offset_x
        local_y = int(point[1]) - offset_y
        if 0 <= local_x < width and 0 <= local_y < height:
            return local_x, local_y
        return None

    def known_free(point: Position) -> bool:
        index = local_index(point)
        if index is None:
            return False
        local_x, local_y = index
        return bool(
            occupancy[local_y, local_x] == FREE
            and confidence[local_y, local_x] >= threshold
        )

    unknown_support: set[Position] = set()
    for x, y in cells:
        for neighbor_y in range(y - 1, y + 2):
            for neighbor_x in range(x - 1, x + 2):
                point = (neighbor_x, neighbor_y)
                index = local_index(point)
                if index is None:
                    continue
                local_x, local_y = index
                if (
                    occupancy[local_y, local_x] == UNKNOWN
                    or confidence[local_y, local_x] < threshold
                ):
                    unknown_support.add(point)

    if not unknown_support:
        return None
    aim_points = unknown_support
    aim_x = sum(point[0] for point in aim_points) / len(aim_points)
    aim_y = sum(point[1] for point in aim_points) / len(aim_points)
    required_points = tuple(sorted(
        set(cells) | unknown_support,
        key=lambda point: (point[1], point[0]),
    ))

    suffix_distance = [0.0] * len(route)
    for index in range(len(route) - 2, -1, -1):
        suffix_distance[index] = (
            suffix_distance[index + 1]
            + math.dist(route[index], route[index + 1])
        )

    def line_is_known_clear(
        start: Position,
        end: Position,
        *,
        allow_unknown_endpoint: bool,
    ) -> bool:
        segment = bresenham_line_points(*start, *end)
        checked = segment[:-1] if allow_unknown_endpoint else segment
        return all(known_free(point) for point in checked)

    for index, candidate in enumerate(route):
        if not known_free(candidate):
            continue
        heading = int(round(math.degrees(math.atan2(
            aim_x - candidate[0],
            -(aim_y - candidate[1]),
        )))) % 360
        if not all(
            _inside_scan_footprint(
                candidate,
                heading,
                point,
                sensor_range=sensor_range,
                sensor_fov_deg=sensor_fov_deg,
            )
            for point in required_points
        ):
            continue
        if not all(
            line_is_known_clear(
                candidate,
                point,
                allow_unknown_endpoint=False,
            )
            for point in cells
        ):
            continue
        if not all(
            line_is_known_clear(
                candidate,
                point,
                allow_unknown_endpoint=True,
            )
            for point in unknown_support
        ):
            continue
        return FrontierObservationPose(
            position=candidate,
            heading=heading,
            route_prefix=tuple(route[:index + 1]),
            saved_route_distance=suffix_distance[index],
        )
    return None


def incidental_scan_candidate(
    slam: SlamSnapshot,
    *,
    position: Position,
    current_heading: float,
    sensor_range: float,
    sensor_fov_deg: float,
    confidence_threshold: float,
    frontier_stride: int,
    active_cells: Iterable[Position] = (),
) -> IncidentalScanDetection:
    """Select one small side-looking pocket using only local SLAM.

    Unknown components that reach either the bounded inspection window or the
    local snapshot boundary are deliberately excluded.  They may be the mouth
    of a larger unexplored basin and remain ordinary component work.
    """
    rejection_names = (
        "unsafe_pose",
        "window_boundary",
        "local_boundary",
        "no_wall",
        "no_gateway",
        "oversized_cone",
        "occluded",
        "active_target",
        "already_visible",
        "rear_facing",
    )
    rejected = {name: 0 for name in rejection_names}
    if sensor_range <= 0.0 or sensor_fov_deg <= 0.0:
        return IncidentalScanDetection(None, 0, tuple(rejected.items()))

    occupancy = np.asarray(slam.occupancy)
    confidence = np.asarray(slam.confidence)
    if occupancy.shape != confidence.shape or occupancy.ndim != 2:
        raise ValueError("SLAM occupancy and confidence must be matching 2D arrays")
    height, width = occupancy.shape
    offset_x, offset_y = (int(value) for value in slam.origin)
    center_x = int(position[0]) - offset_x
    center_y = int(position[1]) - offset_y
    radius = max(1, int(math.ceil(float(sensor_range))))
    left = max(0, center_x - radius)
    right = min(width - 1, center_x + radius)
    top = max(0, center_y - radius)
    bottom = min(height - 1, center_y + radius)
    if left > right or top > bottom:
        return IncidentalScanDetection(None, 0, tuple(rejected.items()))

    threshold = float(confidence_threshold)
    if not (
        0 <= center_x < width
        and 0 <= center_y < height
        and occupancy[center_y, center_x] == FREE
        and confidence[center_y, center_x] >= threshold
    ):
        rejected["unsafe_pose"] += 1
        return IncidentalScanDetection(None, 0, tuple(rejected.items()))
    unknown = (
        (occupancy == UNKNOWN)
        | (confidence < threshold)
    )
    local_unknown = unknown[top:bottom + 1, left:right + 1]
    components = eight_connected_components(local_unknown)
    stride = max(1, int(frontier_stride))
    active = tuple((int(x), int(y)) for x, y in active_cells)
    active_radius = float(2 * stride)

    def global_point(local_x: int, local_y: int) -> Position:
        return local_x + offset_x, local_y + offset_y

    def confident(label: int, local_x: int, local_y: int) -> bool:
        return bool(
            occupancy[local_y, local_x] == label
            and confidence[local_y, local_x] >= threshold
        )

    def known_wall_blocks(point: Position) -> bool:
        line = bresenham_line_points(*position, *point)
        for line_x, line_y in line[1:-1]:
            local_x = int(line_x) - offset_x
            local_y = int(line_y) - offset_y
            if (
                0 <= local_x < width
                and 0 <= local_y < height
                and confident(OCCUPIED, local_x, local_y)
            ):
                return True
        return False

    candidates: list[IncidentalScanCandidate] = []
    for component in components:
        local_cells = tuple((x + left, y + top) for x, y in component)
        if any(
            x in {0, width - 1} or y in {0, height - 1}
            for x, y in local_cells
        ):
            rejected["local_boundary"] += 1
            continue
        if any(
            x in {0, local_unknown.shape[1] - 1}
            or y in {0, local_unknown.shape[0] - 1}
            for x, y in component
        ):
            rejected["window_boundary"] += 1
            continue
        cells = frozenset(global_point(x, y) for x, y in local_cells)

        wall_contacts: set[Position] = set()
        gateways: set[Position] = set()
        for local_x, local_y in local_cells:
            for neighbor_y in range(local_y - 1, local_y + 2):
                for neighbor_x in range(local_x - 1, local_x + 2):
                    if neighbor_x == local_x and neighbor_y == local_y:
                        continue
                    if not (0 <= neighbor_x < width and 0 <= neighbor_y < height):
                        continue
                    if confident(OCCUPIED, neighbor_x, neighbor_y):
                        wall_contacts.add(global_point(neighbor_x, neighbor_y))
                    elif confident(FREE, neighbor_x, neighbor_y):
                        gateways.add(global_point(neighbor_x, neighbor_y))
        if not wall_contacts:
            rejected["no_wall"] += 1
            continue
        if not gateways:
            rejected["no_gateway"] += 1
            continue
        if active and any(
            math.dist(cell, target) <= active_radius + 1e-9
            for cell in cells
            for target in active
        ):
            rejected["active_target"] += 1
            continue

        aim_x = sum(point[0] for point in cells) / len(cells)
        aim_y = sum(point[1] for point in cells) / len(cells)
        heading = int(round(math.degrees(math.atan2(
            aim_x - position[0],
            -(aim_y - position[1]),
        )))) % 360
        if not all(
            _inside_scan_footprint(
                position,
                heading,
                point,
                sensor_range=sensor_range,
                sensor_fov_deg=sensor_fov_deg,
            )
            for point in cells
        ):
            rejected["oversized_cone"] += 1
            continue
        if any(known_wall_blocks(point) for point in cells):
            rejected["occluded"] += 1
            continue

        heading_delta = abs(
            (float(heading) - float(current_heading) + 180.0) % 360.0
            - 180.0
        )
        if heading_delta > 120.0 + 1e-9:
            rejected["rear_facing"] += 1
            continue
        current_visible = sum(
            _inside_scan_footprint(
                position,
                current_heading,
                point,
                sensor_range=sensor_range,
                sensor_fov_deg=sensor_fov_deg,
            )
            for point in cells
        )
        side_only = len(cells) - current_visible
        minimum_side_only = max(1, int(math.ceil(len(cells) * 0.25)))
        if (
            heading_delta + 1e-9 < float(sensor_fov_deg) / 2.0
            or side_only < minimum_side_only
        ):
            rejected["already_visible"] += 1
            continue

        gateway = min(
            gateways,
            key=lambda point: (math.dist(position, point), point[1], point[0]),
        )
        wall_contact = min(
            wall_contacts,
            key=lambda point: (math.dist(position, point), point[1], point[0]),
        )
        signature = IncidentalPocketSignature(
            support_bins=tuple(sorted(
                {(x // stride, y // stride) for x, y in cells},
                key=lambda point: (point[1], point[0]),
            )),
            gateway_bin=(gateway[0] // stride, gateway[1] // stride),
            wall_bin=(wall_contact[0] // stride, wall_contact[1] // stride),
        )
        candidates.append(IncidentalScanCandidate(
            signature=signature,
            cells=cells,
            gateway=gateway,
            wall_contact=wall_contact,
            wall_cells=len(wall_contacts),
            heading=heading,
            heading_delta=heading_delta,
            current_visibility=current_visible,
            proposed_visibility=len(cells),
            side_only_support=side_only,
        ))

    selected = min(
        candidates,
        key=lambda candidate: (
            len(candidate.cells),
            -candidate.side_only_support,
            candidate.heading_delta,
            candidate.signature.support_bins,
            candidate.signature.gateway_bin,
            candidate.signature.wall_bin,
        ),
        default=None,
    )
    return IncidentalScanDetection(
        selected,
        len(components),
        tuple(rejected.items()),
    )


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
