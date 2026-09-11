"""Stable frontier-component lineage and component-local work units."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import time
from typing import Iterable

import cv2
import numpy as np

from mapping.frontiers import (
    FrontierFilterDiagnostics,
    eight_connected_components,
    eight_neighbor_adjacency,
    significant_frontier_mask,
)
from mapping.slam_map import OCCUPIED, UNKNOWN, SlamSnapshot


Position = tuple[int, int]
ComponentId = int
WorkUnitId = int
RegistryRevision = int


class ComponentState(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    SPLIT = "split"
    MERGED = "merged"
    RESOLVED = "resolved"


class ExplorationMode(str, Enum):
    FOCUSED = "focused"
    WALL_FOLLOW = "wall_follow"
    SWEEP = "sweep"


class WorkUnitKind(str, Enum):
    FOCUSED_ANCHOR = "focused_anchor"
    SWEEP_ANCHOR = "sweep_anchor"
    WALL_SUBARC = "wall_subarc"


class WorkUnitState(str, Enum):
    READY = "ready"
    CLAIMED = "claimed"
    ACTIVE = "active"
    VISITED = "visited"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class LineageKind(str, Enum):
    CREATED = "created"
    CONTINUED = "continued"
    SPLIT = "split"
    MERGED = "merged"
    RESEGMENTED = "resegmented"
    DORMANT = "dormant"
    RESOLVED = "resolved"
    REACTIVATED = "reactivated"


@dataclass(frozen=True)
class SensorFootprint:
    """Geometry used to turn a frontier into bounded scan work."""

    max_range: float
    fov_deg: float
    frontier_stride: int = 4
    global_cell_size: int = 32
    minimum_wall_contact_cells: int = 4
    minimum_wall_contact_ratio: float = 0.25
    minimum_wall_run_ratio: float = 0.20

    def __post_init__(self) -> None:
        if self.max_range <= 0.0:
            raise ValueError("sensor max_range must be positive")
        if not 0.0 < self.fov_deg <= 360.0:
            raise ValueError("sensor fov_deg must be in (0, 360]")
        if self.frontier_stride <= 0 or self.global_cell_size <= 0:
            raise ValueError("frontier geometry scales must be positive")

    @property
    def lineage_radius(self) -> int:
        return max(
            2 * int(self.frontier_stride),
            min(
                int(self.global_cell_size),
                int(math.ceil(self.max_range / 2.0)),
            ),
        )

    @property
    def lateral_coverage_radius(self) -> float:
        """Return the frontier span covered by one outward-facing scan.

        Frontier cells normally lie across, rather than along, the scan
        heading.  Using the cone-angle test for those cells therefore reduces
        every sweep anchor to a single pixel even though the cone exposes a
        broad strip of unknown space ahead of the frontier.  The lateral
        projection of the sensor footprint is the appropriate anchor spacing.
        """
        half_angle = min(90.0, self.fov_deg / 2.0)
        return max(
            1.0,
            self.max_range * math.sin(math.radians(half_angle)),
        )


@dataclass(frozen=True)
class FrontierGeometry:
    cells: frozenset[Position]
    bounding_box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    wall_contact_cells: int
    wall_contact_ratio: float
    longest_wall_run_ratio: float
    unknown_support_cells: int
    slam_version: int


@dataclass
class FrontierComponentRecord:
    component_id: ComponentId
    state: ComponentState
    geometry: FrontierGeometry
    geometry_revision: int
    parent_ids: tuple[ComponentId, ...]
    child_ids: tuple[ComponentId, ...]
    first_seen_revision: RegistryRevision
    last_seen_revision: RegistryRevision
    missing_reconciliations: int
    dormant_reason: str | None
    exploration_mode: ExplorationMode
    work_unit_ids: tuple[WorkUnitId, ...]


@dataclass
class ComponentWorkUnit:
    work_unit_id: WorkUnitId
    component_id: ComponentId
    component_revision: int
    kind: WorkUnitKind
    cells: frozenset[Position]
    anchor_position: Position
    scan_headings: tuple[int, ...]
    estimated_effort: float
    state: WorkUnitState
    terminal_reason: str | None = None


@dataclass(frozen=True)
class CausalTransition:
    """A drone-observed predecessor/successor relationship."""

    predecessor_id: ComponentId
    successor_cells: tuple[frozenset[Position], ...]
    report_id: int
    visited_successor_anchors: tuple[Position, ...] = ()


@dataclass(frozen=True)
class LineageTransition:
    kind: LineageKind
    parent_ids: tuple[ComponentId, ...]
    child_ids: tuple[ComponentId, ...]
    evidence: str


@dataclass(frozen=True)
class FrontierRegistrySnapshot:
    revision: RegistryRevision
    components: tuple[FrontierComponentRecord, ...]
    work_units: tuple[ComponentWorkUnit, ...]


@dataclass(frozen=True)
class RegistryReconcileResult:
    revision: RegistryRevision
    elapsed_ms: float
    frontier_diagnostics: FrontierFilterDiagnostics
    transitions: tuple[LineageTransition, ...]
    active_component_ids: tuple[ComponentId, ...]
    ready_work_unit_ids: tuple[WorkUnitId, ...]


class FrontierComponentRegistry:
    """Reconcile significant frontiers without treating basins as identity."""

    def __init__(
        self,
        map_shape: tuple[int, int],
        footprint: SensorFootprint,
        *,
        confidence_threshold: float = 0.6,
        minimum_component_cells: int = 12,
        minimum_unknown_support_cells: int = 64,
        missing_reconciliations_before_resolved: int = 2,
    ) -> None:
        height, width = (int(value) for value in map_shape)
        if height <= 0 or width <= 0:
            raise ValueError("frontier registry map dimensions must be positive")
        if minimum_component_cells <= 0:
            raise ValueError("minimum_component_cells must be positive")
        if minimum_unknown_support_cells <= 0:
            raise ValueError("minimum_unknown_support_cells must be positive")
        if missing_reconciliations_before_resolved <= 0:
            raise ValueError(
                "missing_reconciliations_before_resolved must be positive"
            )
        self.map_shape = (height, width)
        self.footprint = footprint
        self.confidence_threshold = float(confidence_threshold)
        self.minimum_component_cells = int(minimum_component_cells)
        self.minimum_unknown_support_cells = int(
            minimum_unknown_support_cells
        )
        self.missing_reconciliations_before_resolved = int(
            missing_reconciliations_before_resolved
        )
        self.revision = 0
        self._next_component_id = 0
        self._next_work_unit_id = 0
        self.components: dict[ComponentId, FrontierComponentRecord] = {}
        self.work_units: dict[WorkUnitId, ComponentWorkUnit] = {}

    def snapshot(self) -> FrontierRegistrySnapshot:
        """Return detached records in deterministic ID order."""
        return FrontierRegistrySnapshot(
            revision=self.revision,
            components=tuple(
                replace(self.components[key])
                for key in sorted(self.components)
            ),
            work_units=tuple(
                replace(self.work_units[key])
                for key in sorted(self.work_units)
            ),
        )

    def ready_work_units(self) -> tuple[ComponentWorkUnit, ...]:
        return tuple(
            replace(unit)
            for _unit_id, unit in sorted(self.work_units.items())
            if unit.state == WorkUnitState.READY
        )

    def claim_work_units(self, work_unit_ids: Iterable[WorkUnitId]) -> bool:
        ids = tuple(dict.fromkeys(int(value) for value in work_unit_ids))
        if not ids or any(
            unit_id not in self.work_units
            or self.work_units[unit_id].state != WorkUnitState.READY
            for unit_id in ids
        ):
            return False
        for unit_id in ids:
            self.work_units[unit_id].state = WorkUnitState.CLAIMED
        return True

    def activate_work_units(self, work_unit_ids: Iterable[WorkUnitId]) -> None:
        for unit_id in work_unit_ids:
            unit = self.work_units.get(int(unit_id))
            if unit is not None and unit.state == WorkUnitState.CLAIMED:
                unit.state = WorkUnitState.ACTIVE

    def release_work_units(
        self,
        work_unit_ids: Iterable[WorkUnitId],
        *,
        blocked: bool = False,
    ) -> None:
        next_state = WorkUnitState.BLOCKED if blocked else WorkUnitState.READY
        for unit_id in work_unit_ids:
            unit = self.work_units.get(int(unit_id))
            if unit is not None and unit.state in {
                WorkUnitState.CLAIMED,
                WorkUnitState.ACTIVE,
            }:
                component = self.components.get(unit.component_id)
                if component is not None and component.state in {
                    ComponentState.SPLIT,
                    ComponentState.MERGED,
                    ComponentState.RESOLVED,
                }:
                    unit.state = WorkUnitState.CANCELLED
                    unit.terminal_reason = "component_lineage_closed"
                else:
                    unit.state = next_state

    def unblock_work_units(self, work_unit_ids: Iterable[WorkUnitId]) -> None:
        for unit_id in work_unit_ids:
            unit = self.work_units.get(int(unit_id))
            if unit is not None and unit.state == WorkUnitState.BLOCKED:
                unit.state = WorkUnitState.READY

    def block_work_units(self, work_unit_ids: Iterable[WorkUnitId]) -> None:
        """Prevent unclaimed units from racing a live lineage parent claim."""
        for unit_id in work_unit_ids:
            unit = self.work_units.get(int(unit_id))
            if unit is not None and unit.state == WorkUnitState.READY:
                unit.state = WorkUnitState.BLOCKED

    def complete_work_unit(
        self,
        work_unit_id: WorkUnitId,
        *,
        reason: str,
    ) -> bool:
        unit = self.work_units.get(int(work_unit_id))
        if unit is None or unit.state not in {
            WorkUnitState.CLAIMED,
            WorkUnitState.ACTIVE,
        }:
            return False
        unit.state = WorkUnitState.VISITED
        unit.terminal_reason = str(reason)
        component = self.components.get(unit.component_id)
        if component is not None:
            actionable = any(
                self.work_units[unit_id].state in {
                    WorkUnitState.READY,
                    WorkUnitState.CLAIMED,
                    WorkUnitState.ACTIVE,
                    WorkUnitState.BLOCKED,
                }
                for unit_id in component.work_unit_ids
            )
            if not actionable and component.state == ComponentState.ACTIVE:
                component.state = ComponentState.DORMANT
                component.dormant_reason = "current_geometry_visited"
        return True

    def reconcile(
        self,
        rover_slam: SlamSnapshot,
        *,
        causal_transitions: Iterable[CausalTransition] = (),
    ) -> RegistryReconcileResult:
        """Reconcile one authoritative rover snapshot into the lineage DAG."""
        started_at = time.perf_counter()
        causal = tuple(causal_transitions)
        occupancy = np.asarray(rover_slam.occupancy)
        confidence = np.asarray(rover_slam.confidence)
        if occupancy.shape != self.map_shape:
            raise ValueError("rover SLAM shape does not match frontier registry")
        frontier, diagnostics = significant_frontier_mask(
            occupancy,
            confidence,
            self.confidence_threshold,
            minimum_component_cells=self.minimum_component_cells,
            minimum_unknown_support_cells=self.minimum_unknown_support_cells,
        )
        unknown = (
            (occupancy == UNKNOWN)
            | (confidence < self.confidence_threshold)
        )
        known_occupied = (
            (occupancy == OCCUPIED)
            & (confidence >= self.confidence_threshold)
        )
        occupied_adjacency = eight_neighbor_adjacency(known_occupied)
        observations = tuple(
            self._geometry(
                cells,
                unknown,
                occupied_adjacency,
                rover_slam.version,
            )
            for cells in (
                frozenset(component)
                for component in eight_connected_components(frontier)
            )
        )
        self.revision += 1
        active_old = {
            component_id: record
            for component_id, record in self.components.items()
            if record.state in {ComponentState.ACTIVE, ComponentState.DORMANT}
        }
        edge_evidence = self._match_edges(
            active_old,
            observations,
            causal,
        )
        groups = self._lineage_groups(
            tuple(sorted(active_old)),
            tuple(range(len(observations))),
            edge_evidence,
        )
        matched_old: set[int] = set()
        matched_new: set[int] = set()
        transitions: list[LineageTransition] = []

        for old_ids, new_indices in groups:
            if not old_ids or not new_indices:
                continue
            matched_old.update(old_ids)
            matched_new.update(new_indices)
            evidence = self._group_evidence(
                old_ids, new_indices, edge_evidence,
            )
            if len(old_ids) == 1 and len(new_indices) == 1:
                component_id = old_ids[0]
                was_dormant = (
                    self.components[component_id].state
                    == ComponentState.DORMANT
                )
                self._continue_component(
                    self.components[component_id],
                    observations[new_indices[0]],
                    unknown,
                )
                transitions.append(LineageTransition(
                    kind=(
                        LineageKind.REACTIVATED
                        if was_dormant
                        and self.components[component_id].state
                        == ComponentState.ACTIVE
                        else LineageKind.CONTINUED
                    ),
                    parent_ids=(component_id,),
                    child_ids=(component_id,),
                    evidence=evidence,
                ))
                continue

            kind = (
                LineageKind.SPLIT
                if len(old_ids) == 1
                else LineageKind.MERGED
                if len(new_indices) == 1
                else LineageKind.RESEGMENTED
            )
            terminal_state = (
                ComponentState.SPLIT
                if kind == LineageKind.SPLIT
                else ComponentState.MERGED
            )
            for component_id in old_ids:
                record = self.components[component_id]
                record.state = terminal_state
                record.dormant_reason = None
                self._cancel_unclaimed_units(record)
            child_ids = tuple(
                self._create_component(
                    observations[index],
                    unknown,
                    parent_ids=tuple(old_ids),
                )
                for index in sorted(new_indices)
            )
            for component_id in old_ids:
                self.components[component_id].child_ids = child_ids
            transitions.append(LineageTransition(
                kind=kind,
                parent_ids=tuple(old_ids),
                child_ids=child_ids,
                evidence=evidence,
            ))

        for component_id, record in sorted(active_old.items()):
            if component_id in matched_old:
                continue
            record.missing_reconciliations += 1
            confidently_closed = not self._has_unknown_near(
                record.geometry.cells,
                unknown,
            )
            resolved = (
                confidently_closed
                or record.missing_reconciliations
                >= self.missing_reconciliations_before_resolved
            )
            if resolved:
                record.state = ComponentState.RESOLVED
                record.dormant_reason = None
                self._cancel_unclaimed_units(record)
                kind = LineageKind.RESOLVED
            else:
                record.state = ComponentState.DORMANT
                record.dormant_reason = "missing_from_authoritative_snapshot"
                kind = LineageKind.DORMANT
            transitions.append(LineageTransition(
                kind=kind,
                parent_ids=(component_id,),
                child_ids=(),
                evidence=(
                    "confident_closure" if confidently_closed else "missing"
                ),
            ))

        for index, geometry in enumerate(observations):
            if index in matched_new:
                continue
            component_id = self._create_component(
                geometry,
                unknown,
                parent_ids=(),
            )
            transitions.append(LineageTransition(
                kind=LineageKind.CREATED,
                parent_ids=(),
                child_ids=(component_id,),
                evidence="unmatched_significant_frontier",
            ))

        self._apply_causal_visits(
            causal,
            tuple(transitions),
        )

        return RegistryReconcileResult(
            revision=self.revision,
            elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
            frontier_diagnostics=diagnostics,
            transitions=tuple(transitions),
            active_component_ids=tuple(
                component_id
                for component_id, record in sorted(self.components.items())
                if record.state == ComponentState.ACTIVE
            ),
            ready_work_unit_ids=tuple(
                unit.work_unit_id for unit in self.ready_work_units()
            ),
        )

    def _apply_causal_visits(
        self,
        causal: tuple[CausalTransition, ...],
        transitions: tuple[LineageTransition, ...],
    ) -> None:
        """Retire only the anchor work observed by a claimed local DFS.

        A local successor can describe an entire wide frontier even though the
        drone scanned only one pose on it.  Matching an observed anchor to one
        work unit preserves that fact without suppressing the rest of the
        component.
        """
        children_by_parent: dict[int, set[int]] = {}
        for transition in transitions:
            for parent_id in transition.parent_ids:
                children_by_parent.setdefault(parent_id, set()).update(
                    transition.child_ids
                )
        for report in sorted(causal, key=lambda item: item.report_id):
            candidate_components = set(children_by_parent.get(
                report.predecessor_id,
                {report.predecessor_id},
            ))
            for component_id, component in self.components.items():
                if component.state != ComponentState.ACTIVE:
                    continue
                if any(
                    self._sets_near(
                        successor,
                        component.geometry.cells,
                    )
                    for successor in report.successor_cells
                ):
                    candidate_components.add(component_id)
            for anchor in report.visited_successor_anchors:
                candidates: list[tuple[int, float, int]] = []
                for component_id in sorted(candidate_components):
                    component = self.components.get(component_id)
                    if (
                        component is None
                        or component.state != ComponentState.ACTIVE
                    ):
                        continue
                    distance_to_component = min(
                        (math.dist(anchor, point)
                         for point in component.geometry.cells),
                        default=math.inf,
                    )
                    if distance_to_component > self.footprint.lineage_radius:
                        continue
                    for unit_id in component.work_unit_ids:
                        unit = self.work_units[unit_id]
                        if unit.state not in {
                            WorkUnitState.READY,
                            WorkUnitState.BLOCKED,
                        }:
                            continue
                        candidates.append((
                            0 if anchor in unit.cells else 1,
                            math.dist(anchor, unit.anchor_position),
                            unit_id,
                        ))
                if not candidates:
                    continue
                unit = self.work_units[min(candidates)[2]]
                unit.state = WorkUnitState.VISITED
                unit.terminal_reason = "visited_in_claimed_dfs"

        for component in self.components.values():
            if component.state != ComponentState.ACTIVE:
                continue
            actionable = any(
                self.work_units[unit_id].state in {
                    WorkUnitState.READY,
                    WorkUnitState.CLAIMED,
                    WorkUnitState.ACTIVE,
                    WorkUnitState.BLOCKED,
                }
                for unit_id in component.work_unit_ids
            )
            if not actionable:
                component.state = ComponentState.DORMANT
                component.dormant_reason = "current_geometry_visited"

    def _geometry(
        self,
        cells: frozenset[Position],
        unknown: np.ndarray,
        occupied_adjacency: np.ndarray,
        slam_version: int,
    ) -> FrontierGeometry:
        wall_contacts = frozenset(
            (x, y) for x, y in cells if occupied_adjacency[y, x]
        )
        wall_count = len(wall_contacts)
        longest_run = self._longest_eight_connected_run(wall_contacts)
        adjacent_unknown: set[Position] = set()
        height, width = unknown.shape
        for x, y in cells:
            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    if unknown[neighbor_y, neighbor_x]:
                        adjacent_unknown.add((neighbor_x, neighbor_y))
        xs = tuple(point[0] for point in cells)
        ys = tuple(point[1] for point in cells)
        size = max(1, len(cells))
        return FrontierGeometry(
            cells=cells,
            bounding_box=(min(xs), min(ys), max(xs), max(ys)),
            centroid=(sum(xs) / size, sum(ys) / size),
            wall_contact_cells=wall_count,
            wall_contact_ratio=wall_count / size,
            longest_wall_run_ratio=longest_run / size,
            unknown_support_cells=len(adjacent_unknown),
            slam_version=int(slam_version),
        )

    def _create_component(
        self,
        geometry: FrontierGeometry,
        unknown: np.ndarray,
        *,
        parent_ids: tuple[ComponentId, ...],
    ) -> ComponentId:
        component_id = self._next_component_id
        self._next_component_id += 1
        mode, units = self._new_work_units(
            component_id,
            0,
            geometry,
            unknown,
            geometry.cells,
        )
        record = FrontierComponentRecord(
            component_id=component_id,
            state=(ComponentState.ACTIVE if units else ComponentState.DORMANT),
            geometry=geometry,
            geometry_revision=0,
            parent_ids=tuple(sorted(parent_ids)),
            child_ids=(),
            first_seen_revision=self.revision,
            last_seen_revision=self.revision,
            missing_reconciliations=0,
            dormant_reason=None if units else "no_actionable_anchor",
            exploration_mode=mode,
            work_unit_ids=tuple(unit.work_unit_id for unit in units),
        )
        self.components[component_id] = record
        return component_id

    def _continue_component(
        self,
        record: FrontierComponentRecord,
        geometry: FrontierGeometry,
        unknown: np.ndarray,
    ) -> None:
        changed = geometry.cells != record.geometry.cells
        record.geometry = geometry
        record.last_seen_revision = self.revision
        record.missing_reconciliations = 0
        if not changed:
            actionable = any(
                self.work_units[unit_id].state in {
                    WorkUnitState.READY,
                    WorkUnitState.CLAIMED,
                    WorkUnitState.ACTIVE,
                    WorkUnitState.BLOCKED,
                }
                for unit_id in record.work_unit_ids
            )
            record.state = (
                ComponentState.ACTIVE if actionable else ComponentState.DORMANT
            )
            record.dormant_reason = (
                None if actionable else "current_geometry_visited"
            )
            return
        record.geometry_revision += 1
        covered: set[Position] = set()
        retained_ids: list[int] = []
        for unit_id in record.work_unit_ids:
            unit = self.work_units[unit_id]
            if unit.state in {
                WorkUnitState.VISITED,
                WorkUnitState.CLAIMED,
                WorkUnitState.ACTIVE,
            }:
                retained_ids.append(unit_id)
                covered.update(unit.cells)
            elif unit.state in {WorkUnitState.READY, WorkUnitState.BLOCKED}:
                unit.state = WorkUnitState.CANCELLED
                unit.terminal_reason = "component_geometry_changed"
                retained_ids.append(unit_id)
        remaining = frozenset(geometry.cells - covered)
        mode, units = self._new_work_units(
            record.component_id,
            record.geometry_revision,
            geometry,
            unknown,
            remaining,
        )
        record.exploration_mode = mode
        record.work_unit_ids = tuple(
            (*retained_ids, *(unit.work_unit_id for unit in units))
        )
        actionable = any(
            self.work_units[unit_id].state in {
                WorkUnitState.READY,
                WorkUnitState.CLAIMED,
                WorkUnitState.ACTIVE,
                WorkUnitState.BLOCKED,
            }
            for unit_id in record.work_unit_ids
        )
        record.state = (
            ComponentState.ACTIVE if actionable else ComponentState.DORMANT
        )
        record.dormant_reason = (
            None if actionable else "current_geometry_visited"
        )

    def _new_work_units(
        self,
        component_id: int,
        component_revision: int,
        geometry: FrontierGeometry,
        unknown: np.ndarray,
        eligible_cells: frozenset[Position],
    ) -> tuple[ExplorationMode, tuple[ComponentWorkUnit, ...]]:
        anchors = self._coverage_anchors(eligible_cells, unknown)
        wide = len(anchors) > 1
        wall_follow = bool(
            wide
            and geometry.wall_contact_cells
            >= self.footprint.minimum_wall_contact_cells
            and geometry.wall_contact_ratio
            >= self.footprint.minimum_wall_contact_ratio
            and geometry.longest_wall_run_ratio
            >= self.footprint.minimum_wall_run_ratio
        )
        if not wide:
            mode = ExplorationMode.FOCUSED
            kind = WorkUnitKind.FOCUSED_ANCHOR
        elif wall_follow:
            mode = ExplorationMode.WALL_FOLLOW
            kind = WorkUnitKind.WALL_SUBARC
        else:
            mode = ExplorationMode.SWEEP
            kind = WorkUnitKind.SWEEP_ANCHOR
        units: list[ComponentWorkUnit] = []
        unknown_support_share = (
            geometry.unknown_support_cells / max(1, len(anchors))
        )
        for position, heading, cells in anchors:
            unit_id = self._next_work_unit_id
            self._next_work_unit_id += 1
            unit = ComponentWorkUnit(
                work_unit_id=unit_id,
                component_id=component_id,
                component_revision=component_revision,
                kind=kind,
                cells=frozenset(cells),
                anchor_position=position,
                scan_headings=(heading,),
                estimated_effort=(
                    float(len(cells))
                    + unknown_support_share
                    / float(self.footprint.global_cell_size ** 2)
                ),
                state=WorkUnitState.READY,
            )
            self.work_units[unit_id] = unit
            units.append(unit)
        return mode, tuple(units)

    def _coverage_anchors(
        self,
        cells: frozenset[Position],
        unknown: np.ndarray,
    ) -> tuple[tuple[Position, int, frozenset[Position]], ...]:
        if not cells:
            return ()
        remaining = set(cells)
        anchors: list[tuple[Position, int, frozenset[Position]]] = []
        coverage_radius = self.footprint.lateral_coverage_radius
        radius_squared = coverage_radius * coverage_radius
        while remaining:
            position = min(remaining, key=lambda point: (point[1], point[0]))
            heading = self._unknown_heading(position, unknown)
            claimed = frozenset(
                point for point in remaining
                if (
                    (point[0] - position[0]) ** 2
                    + (point[1] - position[1]) ** 2
                    <= radius_squared + 1e-9
                )
            )
            anchors.append((position, heading, claimed))
            remaining.difference_update(claimed)
        return tuple(anchors)

    def _unknown_heading(self, point: Position, unknown: np.ndarray) -> int:
        x, y = point
        height, width = unknown.shape
        neighbors: list[Position] = []
        for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
            for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                if unknown[neighbor_y, neighbor_x]:
                    neighbors.append((neighbor_x, neighbor_y))
        if not neighbors:
            return 0
        target_x = sum(item[0] for item in neighbors) / len(neighbors)
        target_y = sum(item[1] for item in neighbors) / len(neighbors)
        return int(round(math.degrees(math.atan2(
            target_x - x,
            -(target_y - y),
        )))) % 360

    def _match_edges(
        self,
        old: dict[int, FrontierComponentRecord],
        new: tuple[FrontierGeometry, ...],
        causal: tuple[CausalTransition, ...],
    ) -> dict[tuple[int, int], str]:
        edges: dict[tuple[int, int], str] = {}
        for transition in sorted(causal, key=lambda item: item.report_id):
            if transition.predecessor_id not in old:
                continue
            for successor in transition.successor_cells:
                for index, geometry in enumerate(new):
                    if self._sets_near(successor, geometry.cells):
                        edges[(transition.predecessor_id, index)] = "causal_report"
        for component_id, record in old.items():
            for index, geometry in enumerate(new):
                if record.geometry.cells & geometry.cells:
                    edges.setdefault((component_id, index), "exact_overlap")

        near_by_old: dict[int, list[int]] = {key: [] for key in old}
        near_by_new: dict[int, list[int]] = {index: [] for index in range(len(new))}
        for component_id, record in old.items():
            for index, geometry in enumerate(new):
                if (component_id, index) in edges:
                    continue
                if self._sets_near(record.geometry.cells, geometry.cells):
                    near_by_old[component_id].append(index)
                    near_by_new[index].append(component_id)
        for component_id, indices in near_by_old.items():
            if len(indices) != 1:
                continue
            index = indices[0]
            if len(near_by_new[index]) == 1:
                edges[(component_id, index)] = "mutual_unique_near"
        return edges

    def _sets_near(
        self,
        first: frozenset[Position],
        second: frozenset[Position],
    ) -> bool:
        if not first or not second:
            return False
        if first & second:
            return True
        radius = self.footprint.lineage_radius
        first_box = self._bounds(first)
        second_box = self._bounds(second)
        if (
            first_box[2] + radius < second_box[0]
            or second_box[2] + radius < first_box[0]
            or first_box[3] + radius < second_box[1]
            or second_box[3] + radius < first_box[1]
        ):
            return False
        left = max(0, min(first_box[0], second_box[0]) - radius)
        top = max(0, min(first_box[1], second_box[1]) - radius)
        right = min(
            self.map_shape[1] - 1,
            max(first_box[2], second_box[2]) + radius,
        )
        bottom = min(
            self.map_shape[0] - 1,
            max(first_box[3], second_box[3]) + radius,
        )
        shape = (bottom - top + 1, right - left + 1)
        first_mask = np.zeros(shape, dtype=np.uint8)
        second_mask = np.zeros(shape, dtype=np.uint8)
        for x, y in first:
            first_mask[y - top, x - left] = 1
        for x, y in second:
            second_mask[y - top, x - left] = 1
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        first_dilated = cv2.dilate(first_mask, kernel).astype(bool)
        second_dilated = cv2.dilate(second_mask, kernel).astype(bool)
        first_covered = sum(
            second_dilated[y - top, x - left] for x, y in first
        ) / len(first)
        second_covered = sum(
            first_dilated[y - top, x - left] for x, y in second
        ) / len(second)
        return first_covered >= 0.5 and second_covered >= 0.5

    @staticmethod
    def _lineage_groups(
        old_ids: tuple[int, ...],
        new_indices: tuple[int, ...],
        edges: dict[tuple[int, int], str],
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        old_links = {old_id: set() for old_id in old_ids}
        new_links = {index: set() for index in new_indices}
        for old_id, new_index in edges:
            old_links[old_id].add(new_index)
            new_links[new_index].add(old_id)
        pending_old = {old_id for old_id in old_ids if old_links[old_id]}
        groups = []
        while pending_old:
            seed = min(pending_old)
            group_old: set[int] = set()
            group_new: set[int] = set()
            stack: list[tuple[str, int]] = [("old", seed)]
            while stack:
                kind, value = stack.pop()
                if kind == "old":
                    if value in group_old:
                        continue
                    group_old.add(value)
                    pending_old.discard(value)
                    stack.extend(("new", item) for item in old_links[value])
                else:
                    if value in group_new:
                        continue
                    group_new.add(value)
                    stack.extend(("old", item) for item in new_links[value])
            groups.append((tuple(sorted(group_old)), tuple(sorted(group_new))))
        return tuple(groups)

    @staticmethod
    def _group_evidence(
        old_ids: tuple[int, ...],
        new_indices: tuple[int, ...],
        edges: dict[tuple[int, int], str],
    ) -> str:
        evidence = {
            edges[(old_id, new_index)]
            for old_id in old_ids
            for new_index in new_indices
            if (old_id, new_index) in edges
        }
        return "+".join(sorted(evidence)) or "none"

    def _has_unknown_near(
        self,
        cells: frozenset[Position],
        unknown: np.ndarray,
    ) -> bool:
        radius = self.footprint.lineage_radius
        bounds = self._bounds(cells)
        left = max(0, bounds[0] - radius)
        top = max(0, bounds[1] - radius)
        right = min(self.map_shape[1] - 1, bounds[2] + radius)
        bottom = min(self.map_shape[0] - 1, bounds[3] + radius)
        mask = np.zeros(
            (bottom - top + 1, right - left + 1),
            dtype=np.uint8,
        )
        for x, y in cells:
            mask[y - top, x - left] = 1
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        envelope = cv2.dilate(mask, kernel).astype(bool)
        return bool(np.any(
            envelope & unknown[top:bottom + 1, left:right + 1]
        ))

    @staticmethod
    def _longest_eight_connected_run(
        cells: frozenset[Position],
    ) -> int:
        pending = set(cells)
        longest = 0
        while pending:
            seed = pending.pop()
            size = 1
            stack = [seed]
            while stack:
                x, y = stack.pop()
                for neighbor_y in range(y - 1, y + 2):
                    for neighbor_x in range(x - 1, x + 2):
                        neighbor = (neighbor_x, neighbor_y)
                        if neighbor in pending:
                            pending.remove(neighbor)
                            stack.append(neighbor)
                            size += 1
            longest = max(longest, size)
        return longest

    def _cancel_unclaimed_units(self, record: FrontierComponentRecord) -> None:
        for unit_id in record.work_unit_ids:
            unit = self.work_units[unit_id]
            if unit.state in {WorkUnitState.READY, WorkUnitState.BLOCKED}:
                unit.state = WorkUnitState.CANCELLED
                unit.terminal_reason = "component_lineage_closed"

    @staticmethod
    def _bounds(cells: frozenset[Position]) -> tuple[int, int, int, int]:
        xs = tuple(point[0] for point in cells)
        ys = tuple(point[1] for point in cells)
        return min(xs), min(ys), max(xs), max(ys)
