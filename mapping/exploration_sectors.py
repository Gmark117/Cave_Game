"""Dynamic, rover-coordinated exploration sectors."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import threading
from typing import Callable, Iterable

import numpy as np

from mapping.frontiers import (
    FrontierFilterDiagnostics,
    eight_connected_components,
    frontier_neighborhood_masks,
    significant_frontier_mask,
)
from mapping.slam_map import OCCUPIED, UNKNOWN, SlamSnapshot


Position = tuple[int, int]
SectorCell = tuple[int, int]


def _coarse_area_equivalents(pixel_count: int, cell_size: int) -> float:
    """Convert full-resolution pixels to coarse sector-cell work units."""
    return float(pixel_count) / float(cell_size * cell_size)


@dataclass(frozen=True)
class SectorFrontierComponent:
    """Stable rover identity and geometry for assigned frontier work."""

    component_id: int
    cells: frozenset[Position]
    estimated_effort: float = 0.0


@dataclass(frozen=True)
class SectorEffortEstimate:
    """Relative work units, with scan area deduplicated within the scope."""

    frontier_scan: float = 0.0
    approach: float = 0.0
    setup: float = 0.0
    dispersion: float = 0.0
    unknown_area: float = 0.0

    @property
    def total(self) -> float:
        return sum((self.frontier_scan, self.approach, self.setup,
                    self.dispersion, self.unknown_area))


@dataclass(frozen=True)
class SectorAssignment:
    """One drone's disjoint coarse-map territory for an exploration epoch."""

    sector_id: int
    generation: int
    owner_drone_id: int
    cell_size: int
    cells: frozenset[SectorCell]
    seed: Position
    gateway: Position
    frontier_cells: int
    rover_slam_version: int
    bootstrap: bool = False
    frontier_components: tuple[SectorFrontierComponent, ...] = ()
    estimated_effort: float = 0.0
    standby: bool = False
    exploration_mask: np.ndarray | None = field(
        default=None, compare=False, repr=False,
    )
    effort_breakdown: SectorEffortEstimate = SectorEffortEstimate()

    def contains(self, position: Position) -> bool:
        """Return whether a pixel coordinate belongs to this sector."""
        x, y = int(position[0]), int(position[1])
        return (x // self.cell_size, y // self.cell_size) in self.cells

    def permits_exploration(self, position: Position) -> bool:
        """Separate exploration targets from unrestricted transit geometry."""
        if self.standby or not self.contains(position):
            return False
        if self.exploration_mask is None:
            return True
        x, y = int(position[0]), int(position[1])
        height, width = self.exploration_mask.shape
        return 0 <= x < width and 0 <= y < height and bool(
            self.exploration_mask[y, x]
        )


@dataclass(frozen=True)
class SectorWorkloadDiagnostics:
    """Frontier-work balance before and after safe boundary transfers."""

    initial_frontier_workloads: tuple[int, ...]
    balanced_frontier_workloads: tuple[int, ...]
    moved_coarse_cell_count: int
    initial_estimated_efforts: tuple[float, ...] = ()
    balanced_estimated_efforts: tuple[float, ...] = ()


@dataclass(frozen=True)
class SectorSuppressionOutcome:
    """Drone-local suppression evidence reported during rover arrival."""

    component_id: int
    reasons: tuple[str, ...]
    sampled_target_count: int


@dataclass(frozen=True)
class SectorOutcomeReport:
    """One completed assignment's local frontier outcomes."""

    sector_id: int
    generation: int
    suppressions: tuple[SectorSuppressionOutcome, ...] = ()


@dataclass(frozen=True)
class SectorComponentOutcomeDiagnostic:
    """Rover evaluation of one component from the completed generation."""

    component_id: int
    previous_size: int
    current_size: int
    overlap_iou: float
    local_confident_cell_gain: int
    local_confident_occupied_gain: int
    local_confidence_gain: float
    suppression_reasons: tuple[str, ...]
    disposition: str


@dataclass(frozen=True)
class _TrackedFrontierComponent:
    """Internal component state retained until the next team checkpoint."""

    component_id: int
    cells: frozenset[Position]
    baseline_confident_occupied_cells: int
    baseline_confident_cells: int = 0
    baseline_confidence: float = 0.0


@dataclass(frozen=True)
class SectorFrontierOutcomeDiagnostics:
    """Rover-owned memory applied to previously unproductive frontiers."""

    previous_generation: int
    confident_occupied_gain: int | None
    remembered_component_count: int
    suppressed_component_count: int
    suppressed_frontier_pixels: int
    remaining_component_count: int
    remaining_frontier_pixels: int
    evaluated_component_count: int = 0
    locally_unchanged_component_count: int = 0
    reported_zero_gain_component_count: int = 0
    productive_component_count: int = 0
    resolved_component_count: int = 0
    components: tuple[SectorComponentOutcomeDiagnostic, ...] = ()


@dataclass(frozen=True)
class SectorCheckInResult:
    """Outcome of an attempted physical rover check-in."""

    arrived: bool
    assignment: SectorAssignment | None = None
    waiting_for_team: bool = False
    mission_exhausted: bool = False
    generation: int = -1
    waiting_generation: int = -1
    assignment_ready: threading.Event | None = None
    frontier_diagnostics: FrontierFilterDiagnostics | None = None
    workload_diagnostics: SectorWorkloadDiagnostics | None = None
    outcome_diagnostics: SectorFrontierOutcomeDiagnostics | None = None
    published_assignments: tuple[SectorAssignment, ...] = ()


@dataclass(frozen=True)
class ExplorationSectorSnapshot:
    """Detached coordinator state safe for rendering and diagnostics."""

    generation: int
    assignments: tuple[SectorAssignment, ...]
    waiting_drone_ids: frozenset[int]
    mission_exhausted: bool


class _ScopedFrontierWork:
    """Reuse snapshot geometry while evaluating prospective sector boundaries."""

    def __init__(
        self,
        components: tuple[_TrackedFrontierComponent, ...],
        neighborhoods: tuple[np.ndarray, ...],
        unknown: np.ndarray,
        cell_size: int,
        traversal_cost: np.ndarray,
        approach_distances: np.ndarray,
        aggregate: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.components = components
        self.neighborhoods = neighborhoods
        self.unknown = unknown
        self.cell_size = cell_size
        self.traversal_cost = traversal_cost
        self.approach_distances = approach_distances
        self.aggregate = aggregate
        self.counts: list[dict[SectorCell, int]] = []
        self._unknown_counts: dict[tuple[int, ...], np.ndarray] = {}
        for component in components:
            counts: dict[SectorCell, int] = {}
            for x, y in component.cells:
                cell = (x // cell_size, y // cell_size)
                counts[cell] = counts.get(cell, 0) + 1
            self.counts.append(counts)

    def component_indices(self, cells: set[SectorCell]) -> tuple[int, ...]:
        return tuple(i for i, counts in enumerate(self.counts)
                     if cells.intersection(counts))

    def _union(self, indices: tuple[int, ...]) -> np.ndarray:
        mask = np.zeros(self.unknown.shape, dtype=bool)
        for index in indices:
            mask |= self.neighborhoods[index]
        return mask

    def scope(self, cells: set[SectorCell]) -> np.ndarray:
        union = self._union(self.component_indices(cells))
        mask = np.zeros_like(union)
        size = self.cell_size
        for x, y in cells:
            window = (slice(y * size, (y + 1) * size),
                      slice(x * size, (x + 1) * size))
            mask[window] = union[window]
        mask.setflags(write=False)
        return mask

    def _unknown_counts_for(self, indices: tuple[int, ...]) -> np.ndarray:
        if indices not in self._unknown_counts:
            self._unknown_counts[indices] = self.aggregate(
                self._union(indices) & self.unknown,
            )
        return self._unknown_counts[indices]

    def preserves_transfer(
        self, donor: set[SectorCell], recipient: set[SectorCell], cell: SectorCell,
    ) -> bool:
        """Keep covered unknown work when ownership or component membership changes."""
        before = self.component_indices(donor)
        remaining = donor - {cell}
        after = self.component_indices(remaining)
        if before != after:
            lost = self._unknown_counts_for(before) - self._unknown_counts_for(after)
            if any(lost[y, x] > 0 for x, y in remaining):
                return False
        recipient_indices = self.component_indices(recipient | {cell})
        x, y = cell
        size = self.cell_size
        window = (slice(y * size, (y + 1) * size),
                  slice(x * size, (x + 1) * size))
        unknown = self.unknown[window]
        old_scope = np.zeros(unknown.shape, dtype=bool)
        new_scope = np.zeros_like(old_scope)
        for index in before:
            old_scope |= self.neighborhoods[index][window]
        for index in recipient_indices:
            new_scope |= self.neighborhoods[index][window]
        return not bool(np.any(unknown & old_scope & ~new_scope))

    def estimate(self, cells: set[SectorCell]) -> SectorEffortEstimate:
        indices = self.component_indices(cells)
        if not indices:
            return SectorEffortEstimate()
        unknown_counts = self._unknown_counts_for(indices)
        unknown_area = sum(int(unknown_counts[y, x]) for x, y in cells)
        scan = approach = dispersion = 0.0
        for index in indices:
            counts = self.counts[index]
            owned = cells.intersection(counts)
            approach += 2.0 * min(
                float(self.approach_distances[y, x]) for x, y in owned
            )
            dispersion += max(0, len(owned) - 1)
            scan += sum(
                counts[(x, y)] * (1.0 + 0.5 * max(
                    0.0, float(self.traversal_cost[y, x]) - 1.0,
                )) for x, y in owned
            )
        return SectorEffortEstimate(
            frontier_scan=scan,
            approach=approach,
            setup=float(len(indices)),
            dispersion=dispersion,
            unknown_area=_coarse_area_equivalents(
                unknown_area, self.cell_size,
            ),
        )


class ExplorationSectorCoordinator:
    """Build stable, disjoint sector epochs from the rover's team SLAM.

    The coordinator is a barrier: an epoch is repartitioned only after every
    drone has physically returned its completed assignment.  This makes the
    rover snapshot used for the next partition a complete team checkpoint.
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        drone_count: int,
        start_position: Position,
        *,
        cell_size: int = 32,
        confidence_threshold: float = 0.6,
        minimum_frontier_component_cells: int = 12,
        minimum_unknown_support_cells: int = 64,
    ) -> None:
        height, width = (int(value) for value in map_shape)
        if height <= 0 or width <= 0:
            raise ValueError("sector map dimensions must be positive")
        if drone_count <= 0:
            raise ValueError("sector drone_count must be positive")
        if cell_size <= 0:
            raise ValueError("sector cell_size must be positive")
        if minimum_frontier_component_cells <= 0:
            raise ValueError(
                "minimum_frontier_component_cells must be positive"
            )
        if minimum_unknown_support_cells <= 0:
            raise ValueError("minimum_unknown_support_cells must be positive")

        self.map_shape = (height, width)
        self.drone_count = int(drone_count)
        self.start_position = (
            int(start_position[0]),
            int(start_position[1]),
        )
        self.cell_size = int(cell_size)
        self.confidence_threshold = float(confidence_threshold)
        self.minimum_frontier_component_cells = int(
            minimum_frontier_component_cells
        )
        self.minimum_unknown_support_cells = int(
            minimum_unknown_support_cells
        )
        self._lock = threading.RLock()
        self._generation = -1
        self._assignments: dict[int, SectorAssignment] = {}
        self._pending_assignments: dict[int, SectorAssignment] = {}
        self._epoch_assignments: tuple[SectorAssignment, ...] = ()
        self._waiting: set[int] = set()
        self._mission_exhausted = False
        self._last_frontier_diagnostics: (
            FrontierFilterDiagnostics | None
        ) = None
        self._last_workload_diagnostics: SectorWorkloadDiagnostics | None = (
            None
        )
        self._last_outcome_diagnostics: (
            SectorFrontierOutcomeDiagnostics | None
        ) = None
        self._assignment_ready = {
            drone_id: threading.Event()
            for drone_id in range(self.drone_count)
        }
        self._active_frontier_components: tuple[
            _TrackedFrontierComponent, ...
        ] = ()
        self._unproductive_frontier_components: list[
            frozenset[Position]
        ] = []
        self._reported_suppression_reasons: dict[int, set[str]] = {}
        self._next_component_id = 0
        self._generation_start_occupied_cells: int | None = None

    @property
    def generation(self) -> int:
        """Return the most recently generated sector epoch."""
        with self._lock:
            return self._generation

    def snapshot(self) -> ExplorationSectorSnapshot:
        """Return one coherent copy of the current sector epoch."""
        with self._lock:
            return ExplorationSectorSnapshot(
                generation=self._generation,
                assignments=self._epoch_assignments,
                waiting_drone_ids=frozenset(self._waiting),
                mission_exhausted=self._mission_exhausted,
            )

    def check_in(
        self,
        drone_id: int,
        completed_sector_id: int | None,
        rover_slam: SlamSnapshot,
        outcome_report: SectorOutcomeReport | None = None,
    ) -> SectorCheckInResult:
        """Register one arrival and signal assignments when the barrier opens."""
        normalized_id = int(drone_id)
        if not 0 <= normalized_id < self.drone_count:
            raise ValueError("drone_id is outside the configured team")

        with self._lock:
            if self._mission_exhausted:
                return SectorCheckInResult(
                    arrived=True,
                    mission_exhausted=True,
                    generation=self._generation,
                    waiting_generation=self._generation,
                    assignment_ready=self._assignment_ready[normalized_id],
                )

            if normalized_id in self._waiting:
                return SectorCheckInResult(
                    arrived=True,
                    waiting_for_team=True,
                    generation=self._generation,
                    waiting_generation=self._generation,
                    assignment_ready=self._assignment_ready[normalized_id],
                )

            current = self._assignments.get(normalized_id)
            if current is not None:
                if completed_sector_id != current.sector_id:
                    return SectorCheckInResult(
                        arrived=True,
                        assignment=current,
                        generation=self._generation,
                    )
                self._register_outcome_report(current, outcome_report)
                self._assignments.pop(normalized_id, None)

            completed_generation = self._generation
            self._assignment_ready[normalized_id].clear()
            self._waiting.add(normalized_id)
            frontier_diagnostics = None
            workload_diagnostics = None
            outcome_diagnostics = None
            published_assignments = ()
            if len(self._waiting) == self.drone_count and not self._assignments:
                self._generation += 1
                if self._generation == 0:
                    assignments = self._build_bootstrap_assignments(
                        rover_slam.version
                    )
                    self._generation_start_occupied_cells = (
                        self._confident_occupied_cell_count(rover_slam)
                    )
                else:
                    assignments = self._build_dynamic_assignments(rover_slam)
                    frontier_diagnostics = self._last_frontier_diagnostics
                    workload_diagnostics = self._last_workload_diagnostics
                    outcome_diagnostics = self._last_outcome_diagnostics
                if not assignments:
                    self._mission_exhausted = True
                    self._pending_assignments.clear()
                    self._waiting.clear()
                    for ready in self._assignment_ready.values():
                        ready.set()
                    return SectorCheckInResult(
                        arrived=True,
                        mission_exhausted=True,
                        generation=self._generation,
                        waiting_generation=completed_generation,
                        assignment_ready=self._assignment_ready[normalized_id],
                        frontier_diagnostics=frontier_diagnostics,
                        workload_diagnostics=workload_diagnostics,
                        outcome_diagnostics=outcome_diagnostics,
                    )
                self._epoch_assignments = assignments
                published_assignments = assignments
                self._pending_assignments = {
                    assignment.owner_drone_id: assignment
                    for assignment in assignments
                }
                self._assignments = {
                    assignment.owner_drone_id: assignment
                    for assignment in assignments
                    if not assignment.standby
                }
                for ready in self._assignment_ready.values():
                    ready.set()

            return SectorCheckInResult(
                arrived=True,
                waiting_for_team=True,
                generation=self._generation,
                waiting_generation=completed_generation,
                assignment_ready=self._assignment_ready[normalized_id],
                frontier_diagnostics=frontier_diagnostics,
                workload_diagnostics=workload_diagnostics,
                outcome_diagnostics=outcome_diagnostics,
                published_assignments=published_assignments,
            )

    def claim_assignment(self, drone_id: int) -> SectorCheckInResult:
        """Consume an assignment after the rover signals the waiting drone."""
        normalized_id = int(drone_id)
        if not 0 <= normalized_id < self.drone_count:
            raise ValueError("drone_id is outside the configured team")
        with self._lock:
            ready = self._assignment_ready[normalized_id]
            if self._mission_exhausted:
                ready.clear()
                self._waiting.discard(normalized_id)
                return SectorCheckInResult(
                    arrived=True,
                    mission_exhausted=True,
                    generation=self._generation,
                )
            assignment = self._pending_assignments.get(normalized_id)
            if normalized_id not in self._waiting or assignment is None:
                ready.clear()
                return SectorCheckInResult(
                    arrived=True,
                    waiting_for_team=True,
                    generation=self._generation,
                    assignment_ready=ready,
                )
            ready.clear()
            self._pending_assignments.pop(normalized_id)
            if not assignment.standby:
                self._waiting.discard(normalized_id)
            return SectorCheckInResult(
                arrived=True,
                assignment=assignment,
                generation=assignment.generation,
                waiting_generation=assignment.generation,
                assignment_ready=ready if assignment.standby else None,
            )

    def stop(self) -> None:
        """Cancel future assignments and wake every drone standing by."""
        with self._lock:
            self._mission_exhausted = True
            self._assignments.clear()
            self._pending_assignments.clear()
            self._waiting.clear()
            for ready in self._assignment_ready.values():
                ready.set()

    def _grid_shape(self) -> tuple[int, int]:
        height, width = self.map_shape
        return (
            math.ceil(height / self.cell_size),
            math.ceil(width / self.cell_size),
        )

    def _build_bootstrap_assignments(
        self,
        rover_slam_version: int,
    ) -> tuple[SectorAssignment, ...]:
        """Split the initially unknown map into deterministic angular wedges."""
        rows, columns = self._grid_shape()
        cells_by_owner: list[set[SectorCell]] = [
            set() for _ in range(self.drone_count)
        ]
        start_x, start_y = self.start_position
        for cell_y in range(rows):
            for cell_x in range(columns):
                center = self._cell_center((cell_x, cell_y))
                delta_x = center[0] - start_x
                delta_y = center[1] - start_y
                bearing = math.degrees(math.atan2(delta_x, -delta_y)) % 360.0
                owner = int(round(
                    bearing / (360.0 / self.drone_count)
                )) % self.drone_count
                cells_by_owner[owner].add((cell_x, cell_y))

        # Narrow grids can leave an angular wedge empty. Move one cell from the
        # largest wedge so bootstrap ownership remains total and disjoint.
        for owner, cells in enumerate(cells_by_owner):
            if cells:
                continue
            donor = max(
                range(self.drone_count),
                key=lambda candidate: len(cells_by_owner[candidate]),
            )
            if len(cells_by_owner[donor]) <= 1:
                # Tiny mocked maps can contain fewer coarse cells than drones.
                # Sharing the sole gateway is the only representable fallback;
                # normal generated maps have hundreds of disjoint cells.
                cells.add((
                    start_x // self.cell_size,
                    start_y // self.cell_size,
                ))
                continue
            donated = max(
                cells_by_owner[donor],
                key=lambda cell: (
                    self._cell_distance_squared(cell, self.start_position),
                    cell[1],
                    cell[0],
                ),
            )
            cells_by_owner[donor].remove(donated)
            cells.add(donated)

        assignments = []
        for drone_id, cells in enumerate(cells_by_owner):
            seed_cell = max(
                cells,
                key=lambda cell: (
                    self._cell_distance_squared(cell, self.start_position),
                    -cell[1],
                    -cell[0],
                ),
            )
            assignments.append(self._assignment(
                drone_id,
                cells,
                seed_cell,
                frontier_cells=0,
                rover_slam_version=rover_slam_version,
                bootstrap=True,
            ))
        return tuple(assignments)

    def _build_dynamic_assignments(
        self,
        rover_slam: SlamSnapshot,
    ) -> tuple[SectorAssignment, ...]:
        """Partition component-local frontier work by estimated effort."""
        occupancy = np.asarray(rover_slam.occupancy)
        confidence = np.asarray(rover_slam.confidence)
        if occupancy.shape != self.map_shape:
            raise ValueError("rover SLAM shape does not match sector map")

        occupied_cells = self._confident_occupied_cell_count(rover_slam)
        occupied_gain = None
        if (
            self._generation > 1
            and self._generation_start_occupied_cells is not None
        ):
            occupied_gain = max(
                0,
                occupied_cells - self._generation_start_occupied_cells,
            )

        frontier, diagnostics = significant_frontier_mask(
            occupancy,
            confidence,
            self.confidence_threshold,
            minimum_component_cells=(
                self.minimum_frontier_component_cells
            ),
            minimum_unknown_support_cells=(
                self.minimum_unknown_support_cells
            ),
        )
        self._last_frontier_diagnostics = diagnostics
        self._last_workload_diagnostics = None
        components = tuple(
            frozenset(component)
            for component in eight_connected_components(frontier)
        )
        (
            tracked_components,
            suppressed_components,
            component_outcomes,
        ) = self._evaluate_component_outcomes(
            components,
            rover_slam,
        )
        if suppressed_components:
            for component in suppressed_components:
                for x, y in component:
                    frontier[y, x] = False
        self._active_frontier_components = tracked_components
        self._generation_start_occupied_cells = occupied_cells
        self._last_outcome_diagnostics = SectorFrontierOutcomeDiagnostics(
            previous_generation=self._generation - 1,
            confident_occupied_gain=occupied_gain,
            remembered_component_count=len(
                self._unproductive_frontier_components
            ),
            suppressed_component_count=len(suppressed_components),
            suppressed_frontier_pixels=sum(
                len(component) for component in suppressed_components
            ),
            remaining_component_count=len(tracked_components),
            remaining_frontier_pixels=sum(
                len(component.cells) for component in tracked_components
            ),
            evaluated_component_count=len(component_outcomes),
            locally_unchanged_component_count=sum(
                item.disposition == "unchanged_zero_local_gain"
                for item in component_outcomes
            ),
            reported_zero_gain_component_count=sum(
                item.disposition == "reported_zero_gain"
                for item in component_outcomes
            ),
            productive_component_count=sum(
                item.disposition == "productive"
                for item in component_outcomes
            ),
            resolved_component_count=sum(
                item.disposition == "resolved"
                for item in component_outcomes
            ),
            components=component_outcomes,
        )
        frontier_counts = self._aggregate_mask(frontier)
        candidates = tuple(
            (int(cell_x), int(cell_y))
            for cell_y, cell_x in np.argwhere(frontier_counts > 0)
        )
        if not candidates:
            self._last_workload_diagnostics = None
            return ()

        traversal_cost, blocked = self._coarse_traversal_cost(
            occupancy,
            confidence,
        )
        unknown = (occupancy == UNKNOWN) | (
            confidence < self.confidence_threshold
        )
        neighborhoods = frontier_neighborhood_masks(
            unknown, tuple(component.cells for component in tracked_components),
            halo=self.cell_size,
        )
        approach_distances = self._coarse_approach_distances(traversal_cost)
        scoped_work = _ScopedFrontierWork(
            tracked_components, neighborhoods, unknown, self.cell_size,
            traversal_cost, approach_distances, self._aggregate_mask,
        )
        estimated_effort, component_efforts = (
            self._estimated_frontier_effort(
                tracked_components,
                frontier_counts,
                traversal_cost,
                approach_distances=approach_distances,
                unknown_support=tuple(
                    int(np.count_nonzero(mask & unknown))
                    for mask in neighborhoods
                ),
            )
        )
        assigned_components = tuple(
            SectorFrontierComponent(
                component_id=component.component_id,
                cells=component.cells,
                estimated_effort=component_efforts[component.component_id],
            )
            for component in tracked_components
        )
        seeds = self._select_seeds(candidates, estimated_effort)
        owners = self._flood_assign(seeds, traversal_cost, blocked)
        owners, workload_diagnostics = self._rebalance_frontier_work(
            owners,
            frontier_counts,
            seeds,
            estimated_effort=(estimated_effort + self._aggregate_mask(
                np.logical_or.reduce(neighborhoods) & unknown,
            ) / self.cell_size),
            estimate_assignment=scoped_work.estimate,
            preserves_transfer=scoped_work.preserves_transfer,
        )
        self._last_workload_diagnostics = workload_diagnostics
        cells_by_owner: list[set[SectorCell]] = [
            set() for _ in range(self.drone_count)
        ]
        rows, columns = owners.shape
        for cell_y in range(rows):
            for cell_x in range(columns):
                owner = int(owners[cell_y, cell_x])
                if owner >= 0:
                    cells_by_owner[owner].add((cell_x, cell_y))

        assignments = []
        for drone_id, seed in enumerate(seeds):
            cells = cells_by_owner[drone_id]
            if not cells:
                cells.add(seed)
            work = sum(
                int(frontier_counts[cell_y, cell_x])
                for cell_x, cell_y in cells
            )
            effort = scoped_work.estimate(cells)
            components_for_owner = tuple(
                component
                for component in assigned_components
                if any(
                    (
                        point[0] // self.cell_size,
                        point[1] // self.cell_size,
                    ) in cells
                    for point in component.cells
                )
            )
            assignments.append(self._assignment(
                drone_id,
                cells,
                seed,
                frontier_cells=work,
                rover_slam_version=rover_slam.version,
                bootstrap=False,
                frontier_components=components_for_owner,
                estimated_effort=effort.total,
                exploration_mask=scoped_work.scope(cells),
                effort_breakdown=effort,
            ))
        for drone_id in range(len(seeds), self.drone_count):
            assignments.append(self._assignment(
                drone_id, set(),
                (self.start_position[0] // self.cell_size,
                 self.start_position[1] // self.cell_size),
                frontier_cells=0, rover_slam_version=rover_slam.version,
                bootstrap=False, standby=True,
            ))
        return tuple(assignments)

    def _register_outcome_report(
        self,
        assignment: SectorAssignment,
        report: SectorOutcomeReport | None,
    ) -> None:
        """Accept suppression evidence only for the assignment being retired."""
        if (
            report is None
            or report.sector_id != assignment.sector_id
            or report.generation != assignment.generation
        ):
            return
        assigned_ids = {
            component.component_id
            for component in assignment.frontier_components
        }
        for suppression in report.suppressions:
            if suppression.component_id not in assigned_ids:
                continue
            self._reported_suppression_reasons.setdefault(
                suppression.component_id,
                set(),
            ).update(str(reason) for reason in suppression.reasons)

    def _evaluate_component_outcomes(
        self,
        components: tuple[frozenset[Position], ...],
        rover_slam: SlamSnapshot,
    ) -> tuple[
        tuple[_TrackedFrontierComponent, ...],
        tuple[frozenset[Position], ...],
        tuple[SectorComponentOutcomeDiagnostic, ...],
    ]:
        """Retire only unchanged components with component-local zero gain."""
        matched_current: set[int] = set()
        retained_ids: dict[int, int] = {}
        suppressed_indices: set[int] = set()
        outcome_diagnostics: list[SectorComponentOutcomeDiagnostic] = []

        for previous in self._active_frontier_components:
            best_index = -1
            best_iou = 0.0
            for index, current in enumerate(components):
                if index in matched_current:
                    continue
                overlap = self._component_iou(previous.cells, current)
                if overlap > best_iou:
                    best_index = index
                    best_iou = overlap
            reasons = tuple(sorted(
                self._reported_suppression_reasons.get(
                    previous.component_id,
                    (),
                )
            ))
            if best_index < 0 or best_iou < 0.5:
                outcome_diagnostics.append(
                    SectorComponentOutcomeDiagnostic(
                        component_id=previous.component_id,
                        previous_size=len(previous.cells),
                        current_size=0,
                        overlap_iou=best_iou,
                        local_confident_cell_gain=0,
                        local_confident_occupied_gain=0,
                        local_confidence_gain=0.0,
                        suppression_reasons=reasons,
                        disposition="resolved",
                    )
                )
                continue

            matched_current.add(best_index)
            current = components[best_index]
            (
                current_confident,
                current_occupied,
                current_confidence,
            ) = self._local_evidence_near(
                rover_slam,
                previous.cells,
            )
            local_confident_gain = max(
                0,
                current_confident - previous.baseline_confident_cells,
            )
            local_gain = max(
                0,
                current_occupied
                - previous.baseline_confident_occupied_cells,
            )
            local_confidence_gain = max(
                0.0,
                current_confidence - previous.baseline_confidence,
            )
            reported_zero_gain = "zero_gain_directed_scan" in reasons
            has_local_gain = (
                local_confident_gain > 0
                or local_gain > 0
            )
            if reported_zero_gain or not has_local_gain:
                self._remember_unproductive_component(previous.cells)
                suppressed_indices.add(best_index)
                disposition = (
                    "reported_zero_gain"
                    if reported_zero_gain
                    else "unchanged_zero_local_gain"
                )
            else:
                retained_ids[best_index] = previous.component_id
                disposition = "productive"
            outcome_diagnostics.append(
                SectorComponentOutcomeDiagnostic(
                    component_id=previous.component_id,
                    previous_size=len(previous.cells),
                    current_size=len(current),
                    overlap_iou=best_iou,
                    local_confident_cell_gain=local_confident_gain,
                    local_confident_occupied_gain=local_gain,
                    local_confidence_gain=local_confidence_gain,
                    suppression_reasons=reasons,
                    disposition=disposition,
                )
            )

        tracked: list[_TrackedFrontierComponent] = []
        suppressed: list[frozenset[Position]] = []
        for index, component in enumerate(components):
            if (
                index in suppressed_indices
                or self._matches_unproductive_component(component)
            ):
                suppressed.append(component)
                continue
            component_id = retained_ids.get(index)
            if component_id is None:
                component_id = self._next_component_id
                self._next_component_id += 1
            baseline = self._local_evidence_near(
                rover_slam,
                component,
            )
            tracked.append(_TrackedFrontierComponent(
                component_id=component_id,
                cells=component,
                baseline_confident_cells=baseline[0],
                baseline_confident_occupied_cells=baseline[1],
                baseline_confidence=baseline[2],
            ))

        self._reported_suppression_reasons.clear()
        return (
            tuple(tracked),
            tuple(suppressed),
            tuple(outcome_diagnostics),
        )

    @staticmethod
    def _component_iou(
        first: frozenset[Position],
        second: frozenset[Position],
    ) -> float:
        """Return exact pixel intersection-over-union for two components."""
        intersection = len(first & second)
        if intersection == 0:
            return 0.0
        return intersection / len(first | second)

    def _local_evidence_near(
        self,
        rover_slam: SlamSnapshot,
        component: frozenset[Position],
    ) -> tuple[int, int, float]:
        """Measure confident cells, walls, and confidence in a local ROI."""
        if not component:
            return (0, 0, 0.0)
        occupancy = np.asarray(rover_slam.occupancy)
        confidence = np.asarray(rover_slam.confidence)
        padding = self.cell_size
        xs = tuple(point[0] for point in component)
        ys = tuple(point[1] for point in component)
        left = max(0, min(xs) - padding)
        right = min(self.map_shape[1], max(xs) + padding + 1)
        top = max(0, min(ys) - padding)
        bottom = min(self.map_shape[0], max(ys) + padding + 1)
        local_occupancy = occupancy[top:bottom, left:right]
        local_confidence = confidence[top:bottom, left:right]
        confident = local_confidence >= self.confidence_threshold
        return (
            int(np.count_nonzero(confident)),
            int(np.count_nonzero(
                (local_occupancy == OCCUPIED) & confident
            )),
            float(np.sum(local_confidence, dtype=np.float64)),
        )

    def _confident_occupied_cell_count(self, rover_slam: SlamSnapshot) -> int:
        """Count wall evidence available to the rover without ground truth."""
        occupancy = np.asarray(rover_slam.occupancy)
        confidence = np.asarray(rover_slam.confidence)
        return int(np.count_nonzero(
            (occupancy == OCCUPIED)
            & (confidence >= self.confidence_threshold)
        ))

    def _remember_unproductive_component(
        self,
        component: frozenset[Position],
    ) -> None:
        """Retain one zero-gain component unless it is already represented."""
        if not component:
            return
        if self._matches_unproductive_component(component):
            return
        self._unproductive_frontier_components.append(component)

    def _matches_unproductive_component(
        self,
        component: frozenset[Position],
    ) -> bool:
        """Match stable residual geometry while allowing material reshaping."""
        if not component:
            return False
        for remembered in self._unproductive_frontier_components:
            intersection = len(component & remembered)
            if intersection == 0:
                continue
            union = len(component | remembered)
            if union and intersection / union >= 0.5:
                return True
        return False

    def _aggregate_mask(self, mask: np.ndarray) -> np.ndarray:
        """Count full-resolution true pixels in each coarse sector cell."""
        rows, columns = self._grid_shape()
        counts = np.zeros((rows, columns), dtype=np.int32)
        pixel_y, pixel_x = np.nonzero(mask)
        if len(pixel_y):
            flat = (
                (pixel_y // self.cell_size) * columns
                + pixel_x // self.cell_size
            )
            counts.flat[:] = np.bincount(
                flat,
                minlength=rows * columns,
            )
        return counts

    def _estimated_frontier_effort(
        self,
        components: tuple[_TrackedFrontierComponent, ...],
        frontier_counts: np.ndarray,
        traversal_cost: np.ndarray,
        *,
        approach_distances: np.ndarray | None = None,
        unknown_support: tuple[int, ...] = (),
    ) -> tuple[np.ndarray, dict[int, float]]:
        """Estimate scan, approach, dispersion, and terrain effort per cell."""
        effort = np.zeros_like(frontier_counts, dtype=np.float64)
        component_efforts: dict[int, float] = {}
        distances = (
            self._coarse_approach_distances(traversal_cost)
            if approach_distances is None else approach_distances
        )
        for index, component in enumerate(components):
            counts_by_cell: dict[SectorCell, int] = {}
            for x, y in component.cells:
                cell = (x // self.cell_size, y // self.cell_size)
                counts_by_cell[cell] = counts_by_cell.get(cell, 0) + 1
            if not counts_by_cell:
                component_efforts[component.component_id] = 0.0
                continue

            component_size = len(component.cells)
            approach_cells = min(
                float(distances[y, x]) for x, y in counts_by_cell
            )
            area_cost = (
                _coarse_area_equivalents(
                    unknown_support[index], self.cell_size,
                )
                if unknown_support else 0.0
            )
            dispersion_cost = max(0, len(counts_by_cell) - 1)
            total = 0.0
            for (cell_x, cell_y), pixel_count in counts_by_cell.items():
                share = pixel_count / component_size
                terrain_multiplier = 1.0 + 0.5 * max(
                    0.0,
                    float(traversal_cost[cell_y, cell_x]) - 1.0,
                )
                cell_effort = (
                    pixel_count * terrain_multiplier
                    + share * (2.0 * approach_cells + dispersion_cost
                               + 1.0 + area_cost)
                )
                effort[cell_y, cell_x] += cell_effort
                total += cell_effort
            component_efforts[component.component_id] = total
        return effort, component_efforts

    def _coarse_approach_distances(self, costs: np.ndarray) -> np.ndarray:
        """Estimate round-trip-compatible travel on the coarse SLAM grid.

        Wall-heavy tiles remain expensive estimates, not hard reachability
        verdicts: a narrow traversable corridor can cross such a tile.
        """
        rows, columns = costs.shape
        x = min(columns - 1, max(0, self.start_position[0] // self.cell_size))
        y = min(rows - 1, max(0, self.start_position[1] // self.cell_size))
        distances = np.full(costs.shape, np.inf)
        distances[y, x] = 0.0
        heap = [(0.0, y, x)]
        while heap:
            distance, y, x = heapq.heappop(heap)
            if distance > distances[y, x]:
                continue
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if not (0 <= nx < columns and 0 <= ny < rows):
                    continue
                candidate = distance + 0.5 * (costs[y, x] + costs[ny, nx])
                if candidate < distances[ny, nx]:
                    distances[ny, nx] = candidate
                    heapq.heappush(heap, (float(candidate), ny, nx))
        return distances

    def _coarse_traversal_cost(
        self,
        occupancy: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate sector connectivity without consulting ground truth."""
        rows, columns = self._grid_shape()
        costs = np.ones((rows, columns), dtype=np.float64)
        blocked = np.zeros((rows, columns), dtype=bool)
        for cell_y in range(rows):
            top = cell_y * self.cell_size
            bottom = min(self.map_shape[0], top + self.cell_size)
            for cell_x in range(columns):
                left = cell_x * self.cell_size
                right = min(self.map_shape[1], left + self.cell_size)
                cell_occ = occupancy[top:bottom, left:right]
                cell_conf = confidence[top:bottom, left:right]
                known = cell_conf >= self.confidence_threshold
                occupied_ratio = float(np.mean(known & (cell_occ == OCCUPIED)))
                unknown_ratio = float(np.mean((~known) | (cell_occ == UNKNOWN)))
                blocked[cell_y, cell_x] = occupied_ratio >= 0.8
                costs[cell_y, cell_x] = (
                    1.0 + 2.0 * unknown_ratio + 4.0 * occupied_ratio
                )
        return costs, blocked

    def _select_seeds(
        self,
        candidates: Iterable[SectorCell],
        estimated_effort: np.ndarray,
    ) -> tuple[SectorCell, ...]:
        """Choose workload-aware, well-separated deterministic seed cells."""
        available = sorted(set(candidates), key=lambda cell: (cell[1], cell[0]))
        first = max(
            available,
            key=lambda cell: (
                float(estimated_effort[cell[1], cell[0]]),
                -cell[1],
                -cell[0],
            ),
        )
        seeds = [first]
        pool = available
        while len(seeds) < min(self.drone_count, len(available)):
            unused = [cell for cell in pool if cell not in seeds]
            selected = max(
                unused,
                key=lambda cell: (
                    min(self._cell_metric_squared(cell, seed) for seed in seeds)
                    * (1.0 + math.log1p(float(
                        estimated_effort[cell[1], cell[0]]
                    ))),
                    float(estimated_effort[cell[1], cell[0]]),
                    -cell[1],
                    -cell[0],
                ),
            )
            seeds.append(selected)
        return tuple(seeds)

    def _flood_assign(
        self,
        seeds: tuple[SectorCell, ...],
        traversal_cost: np.ndarray,
        blocked: np.ndarray,
    ) -> np.ndarray:
        """Run a deterministic multi-source Dijkstra partition."""
        rows, columns = blocked.shape
        owners = np.full((rows, columns), -1, dtype=np.int16)
        distance = np.full((rows, columns), np.inf, dtype=np.float64)
        heap: list[tuple[float, int, int, int]] = []
        for owner, (cell_x, cell_y) in enumerate(seeds):
            blocked[cell_y, cell_x] = False
            distance[cell_y, cell_x] = 0.0
            owners[cell_y, cell_x] = owner
            heapq.heappush(heap, (0.0, owner, cell_y, cell_x))

        while heap:
            current_distance, owner, cell_y, cell_x = heapq.heappop(heap)
            if current_distance > distance[cell_y, cell_x] + 1e-9:
                continue
            if owner != int(owners[cell_y, cell_x]):
                continue
            for offset_x, offset_y in ((0, -1), (-1, 0), (1, 0), (0, 1)):
                next_x = cell_x + offset_x
                next_y = cell_y + offset_y
                if not (0 <= next_x < columns and 0 <= next_y < rows):
                    continue
                if blocked[next_y, next_x]:
                    continue
                candidate_distance = current_distance + float(
                    traversal_cost[next_y, next_x]
                )
                old_distance = float(distance[next_y, next_x])
                old_owner = int(owners[next_y, next_x])
                if (
                    candidate_distance < old_distance - 1e-9
                    or (
                        abs(candidate_distance - old_distance) <= 1e-9
                        and (old_owner < 0 or owner < old_owner)
                    )
                ):
                    distance[next_y, next_x] = candidate_distance
                    owners[next_y, next_x] = owner
                    heapq.heappush(
                        heap,
                        (candidate_distance, owner, next_y, next_x),
                    )

        # A fully known wall component need not be explored, but assigning it
        # to the nearest seed keeps sector membership total and deterministic.
        unassigned_y, unassigned_x = np.where(owners < 0)
        for cell_y, cell_x in zip(unassigned_y, unassigned_x):
            owners[cell_y, cell_x] = min(
                range(len(seeds)),
                key=lambda owner: (
                    self._cell_metric_squared(
                        (int(cell_x), int(cell_y)),
                        seeds[owner],
                    ),
                    owner,
                ),
            )
        return owners

    def _rebalance_frontier_work(
        self,
        owners: np.ndarray,
        frontier_counts: np.ndarray,
        seeds: tuple[SectorCell, ...],
        *,
        estimated_effort: np.ndarray | None = None,
        estimate_assignment: Callable[
            [set[SectorCell]], SectorEffortEstimate
        ] | None = None,
        preserves_transfer: Callable[
            [set[SectorCell], set[SectorCell], SectorCell], bool
        ] | None = None,
    ) -> tuple[np.ndarray, SectorWorkloadDiagnostics]:
        """Move effort-bearing boundaries while sectors stay contiguous."""
        balanced = np.asarray(owners, dtype=np.int16).copy()
        effort = (
            np.asarray(frontier_counts, dtype=np.float64)
            if estimated_effort is None
            else np.asarray(estimated_effort, dtype=np.float64)
        )
        if effort.shape != frontier_counts.shape:
            raise ValueError("estimated effort must match frontier counts")
        cells_by_owner = [
            {
                (int(cell_x), int(cell_y))
                for cell_y, cell_x in zip(
                    *np.where(balanced == owner)
                )
            }
            for owner in range(self.drone_count)
        ]
        frontier_workloads = [
            sum(
                int(frontier_counts[cell_y, cell_x])
                for cell_x, cell_y in cells
            )
            for cells in cells_by_owner
        ]
        effort_workloads = [
            estimate_assignment(cells).total if estimate_assignment else sum(
                float(effort[cell_y, cell_x])
                for cell_x, cell_y in cells
            )
            for cells in cells_by_owner
        ]
        initial_frontier = tuple(frontier_workloads)
        initial_effort = tuple(effort_workloads)
        active_count = len(seeds)
        target_workload = sum(effort_workloads) / active_count
        seed_set = set(seeds)
        moved = 0

        while True:
            best: tuple[
                float,
                float,
                int,
                int,
                int,
                int,
                int,
            ] | None = None
            rows, columns = balanced.shape
            for cell_y in range(rows):
                for cell_x in range(columns):
                    cell = (cell_x, cell_y)
                    donor = int(balanced[cell_y, cell_x])
                    frontier_weight = int(frontier_counts[cell_y, cell_x])
                    effort_weight = float(effort[cell_y, cell_x])
                    if effort_weight <= 0 or cell in seed_set:
                        continue
                    recipients = {
                        int(balanced[next_y, next_x])
                        for next_x, next_y in (
                            (cell_x - 1, cell_y),
                            (cell_x + 1, cell_y),
                            (cell_x, cell_y - 1),
                            (cell_x, cell_y + 1),
                        )
                        if 0 <= next_x < columns
                        and 0 <= next_y < rows
                        and int(balanced[next_y, next_x]) != donor
                    }
                    if not recipients:
                        continue
                    if not self._removal_preserves_connectivity(
                        cells_by_owner[donor],
                        cell,
                    ):
                        continue
                    for recipient in recipients:
                        if preserves_transfer is not None and not preserves_transfer(
                            cells_by_owner[donor], cells_by_owner[recipient], cell,
                        ):
                            continue
                        if estimate_assignment is not None:
                            donor_effort = estimate_assignment(
                                cells_by_owner[donor] - {cell},
                            ).total
                            recipient_effort = estimate_assignment(
                                cells_by_owner[recipient] | {cell},
                            ).total
                            proposed = effort_workloads.copy()
                            proposed[donor] = donor_effort
                            proposed[recipient] = recipient_effort
                            # A strict global potential prevents cycling even
                            # when a transfer changes scan area or setup cost.
                            improvement = sum(v * v for v in effort_workloads) - sum(
                                v * v for v in proposed
                            )
                            if improvement > 1e-9:
                                candidate = (
                                    improvement, effort_weight, frontier_weight,
                                    -cell_y, -cell_x, -recipient, donor,
                                )
                                if best is None or candidate > best:
                                    best = candidate
                            continue
                        before = (
                            (effort_workloads[donor] - target_workload) ** 2
                            + (
                                effort_workloads[recipient]
                                - target_workload
                            ) ** 2
                        )
                        after = (
                            (
                                effort_workloads[donor]
                                - effort_weight
                                - target_workload
                            ) ** 2
                            + (
                                effort_workloads[recipient]
                                + effort_weight
                                - target_workload
                            ) ** 2
                        )
                        improvement = before - after
                        if improvement <= 1e-9:
                            continue
                        candidate = (
                            improvement,
                            effort_weight,
                            frontier_weight,
                            -cell_y,
                            -cell_x,
                            -recipient,
                            donor,
                        )
                        if best is None or candidate > best:
                            best = candidate
            if best is None:
                break
            (
                _improvement,
                effort_weight,
                frontier_weight,
                neg_y,
                neg_x,
                neg_recipient,
                donor,
            ) = best
            cell_x = -neg_x
            cell_y = -neg_y
            recipient = -neg_recipient
            cell = (cell_x, cell_y)
            balanced[cell_y, cell_x] = recipient
            cells_by_owner[donor].remove(cell)
            cells_by_owner[recipient].add(cell)
            frontier_workloads[donor] -= frontier_weight
            frontier_workloads[recipient] += frontier_weight
            effort_workloads[donor] -= effort_weight
            effort_workloads[recipient] += effort_weight
            if estimate_assignment is not None:
                effort_workloads[donor] = estimate_assignment(
                    cells_by_owner[donor],
                ).total
                effort_workloads[recipient] = estimate_assignment(
                    cells_by_owner[recipient],
                ).total
            moved += 1

        return balanced, SectorWorkloadDiagnostics(
            initial_frontier_workloads=initial_frontier,
            balanced_frontier_workloads=tuple(frontier_workloads),
            moved_coarse_cell_count=moved,
            initial_estimated_efforts=initial_effort,
            balanced_estimated_efforts=tuple(effort_workloads),
        )

    @staticmethod
    def _removal_preserves_connectivity(
        cells: set[SectorCell],
        removed: SectorCell,
    ) -> bool:
        """Return whether four-connected sector cells survive one removal."""
        if removed not in cells or len(cells) <= 1:
            return False
        remaining = cells - {removed}
        pending = [next(iter(remaining))]
        visited = {pending[0]}
        while pending:
            cell_x, cell_y = pending.pop()
            for neighbor in (
                (cell_x - 1, cell_y),
                (cell_x + 1, cell_y),
                (cell_x, cell_y - 1),
                (cell_x, cell_y + 1),
            ):
                if neighbor in remaining and neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        return len(visited) == len(remaining)

    def _assignment(
        self,
        drone_id: int,
        cells: set[SectorCell],
        seed_cell: SectorCell,
        *,
        frontier_cells: int,
        rover_slam_version: int,
        bootstrap: bool,
        frontier_components: tuple[SectorFrontierComponent, ...] = (),
        estimated_effort: float = 0.0,
        standby: bool = False,
        exploration_mask: np.ndarray | None = None,
        effort_breakdown: SectorEffortEstimate = SectorEffortEstimate(),
    ) -> SectorAssignment:
        return SectorAssignment(
            sector_id=self._generation * self.drone_count + drone_id,
            generation=self._generation,
            owner_drone_id=drone_id,
            cell_size=self.cell_size,
            cells=frozenset(cells),
            seed=self._cell_center(seed_cell),
            gateway=self.start_position,
            frontier_cells=int(frontier_cells),
            rover_slam_version=int(rover_slam_version),
            bootstrap=bootstrap,
            frontier_components=frontier_components,
            estimated_effort=float(estimated_effort),
            standby=standby,
            exploration_mask=exploration_mask,
            effort_breakdown=effort_breakdown,
        )

    def _cell_center(self, cell: SectorCell) -> Position:
        height, width = self.map_shape
        return (
            min(width - 1, cell[0] * self.cell_size + self.cell_size // 2),
            min(height - 1, cell[1] * self.cell_size + self.cell_size // 2),
        )

    def _cell_distance_squared(
        self,
        cell: SectorCell,
        position: Position,
    ) -> float:
        center = self._cell_center(cell)
        return float(
            (center[0] - position[0]) ** 2
            + (center[1] - position[1]) ** 2
        )

    @staticmethod
    def _cell_metric_squared(first: SectorCell, second: SectorCell) -> int:
        return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
