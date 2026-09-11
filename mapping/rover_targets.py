"""Frontier-driven rover staging target selection and reservation."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from contracts import RoverTargetDependencies


@dataclass(frozen=True)
class RoverFrontierTarget:
    """One rover-visible component staging point."""

    position: Tuple[int, int]
    component_id: int
    task_id: int
    claimed: bool
    depth: int
    estimated_effort: float
    parent_component_ids: tuple[int, ...] = ()
    service_cost: float = math.inf
    rover_route_cost: float = math.inf
    terrain_roughness: float = math.inf
    wall_clearance: float = 0.0


class RoverTargetService:
    """Move rovers toward active exploration while preserving rendezvous."""

    def __init__(self, dependencies: RoverTargetDependencies) -> None:
        """Store target-selection inputs and stable lineage reservations."""
        self.dependencies = dependencies
        self._targets_by_rover: dict[int, RoverFrontierTarget] = {}
        self._retarget_distance = 64.0

    def acquire(
        self, rover_id: int, current_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Reserve the best current frontier staging point for one rover."""
        dependencies = self.dependencies
        candidates = tuple(dependencies.get_frontier_candidates())
        with dependencies.assignment_lock:
            current = dependencies.assignments.get(rover_id)
            tracked = self._targets_by_rover.get(rover_id)
            continuation = self._lineage_continuation(
                tracked,
                candidates,
            )
            if (
                current is not None
                and continuation is not None
                and math.dist(current, continuation.position)
                <= self._retarget_distance
            ):
                self._targets_by_rover[rover_id] = continuation
                return current
            candidate_positions = {
                tuple(candidate.position) for candidate in candidates
            }
            if current in candidate_positions:
                self._targets_by_rover[rover_id] = next(
                    candidate for candidate in candidates
                    if tuple(candidate.position) == current
                )
                return current
            assigned_targets = {
                target
                for rid, target in dependencies.assignments.items()
                if rid != rover_id
            }
            available = tuple(
                candidate for candidate in candidates
                if tuple(candidate.position) not in assigned_targets
            )
            if not available:
                dependencies.assignments.pop(rover_id, None)
                self._targets_by_rover.pop(rover_id, None)
                return None
            selected = min(
                available,
                key=lambda candidate: (
                    -int(candidate.claimed),
                    float(candidate.service_cost),
                    float(candidate.terrain_roughness),
                    -float(candidate.wall_clearance),
                    -int(candidate.depth),
                    -float(candidate.estimated_effort),
                    float(candidate.rover_route_cost),
                    int(candidate.component_id),
                    int(candidate.task_id),
                ),
            )
            target = tuple(int(value) for value in selected.position)
            dependencies.assignments[rover_id] = target
            self._targets_by_rover[rover_id] = selected
            return target

    def target_is_current(self, rover_id: int) -> bool:
        """Return whether the assigned component staging point remains live."""
        dependencies = self.dependencies
        candidates = tuple(dependencies.get_frontier_candidates())
        with dependencies.assignment_lock:
            target = dependencies.assignments.get(int(rover_id))
            tracked = self._targets_by_rover.get(int(rover_id))
        if target is None:
            return False
        return any(
            tuple(candidate.position) == target
            for candidate in candidates
        )

    @staticmethod
    def _lineage_continuation(
        tracked: RoverFrontierTarget | None,
        candidates: tuple[RoverFrontierTarget, ...],
    ) -> RoverFrontierTarget | None:
        if tracked is None:
            return None
        descendants = tuple(
            candidate for candidate in candidates
            if (
                candidate.component_id == tracked.component_id
                or tracked.component_id in candidate.parent_component_ids
            )
        )
        if not descendants:
            return None
        return min(
            descendants,
            key=lambda candidate: (
                -int(candidate.claimed),
                float(candidate.service_cost),
                float(candidate.terrain_roughness),
                -float(candidate.wall_clearance),
                -int(candidate.depth),
                -float(candidate.estimated_effort),
                math.dist(tracked.position, candidate.position),
                int(candidate.component_id),
                int(candidate.task_id),
            ),
        )

    def should_hold(self, rover_id: int) -> bool:
        """Keep a rover stationary while a drone is closing to rendezvous."""
        return bool(self.dependencies.should_hold_position(int(rover_id)))

    def release(self, rover_id: int, completed: bool = False) -> None:
        """Release or mark complete a rover terrain target reservation."""
        dependencies = self.dependencies
        with dependencies.assignment_lock:
            target = dependencies.assignments.pop(rover_id, None)
            self._targets_by_rover.pop(rover_id, None)
            if completed and target is not None:
                dependencies.completed_targets.add(target)
