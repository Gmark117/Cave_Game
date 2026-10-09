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
    focused_endgame: bool = False
    remaining_max_distance: float = math.inf
    remaining_total_distance: float = math.inf
    current_max_distance: float = math.inf
    current_total_distance: float = math.inf


class RoverTargetService:
    """Move rovers toward active exploration while preserving rendezvous."""

    def __init__(self, dependencies: RoverTargetDependencies) -> None:
        """Store target-selection inputs and stable lineage reservations."""
        self.dependencies = dependencies
        self._targets_by_rover: dict[int, RoverFrontierTarget] = {}
        self._retarget_distance = 64.0
        self._failed_route_until: dict[
            tuple[int, Tuple[int, int], Tuple[int, int]], float
        ] = {}
        self._failed_route_retry_seconds = 60.0
        self._reported_failed_route_suppressions: set[
            tuple[int, Tuple[int, int], Tuple[int, int]]
        ] = set()

    def reject_failed_route(
        self,
        rover_id: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        *,
        sim_time: float,
    ) -> None:
        """Delay another full search for the same rover-local route."""
        key = (int(rover_id), tuple(start), tuple(goal))
        self._reported_failed_route_suppressions.discard(key)
        self._failed_route_until[key] = (
            float(sim_time) + self._failed_route_retry_seconds
        )
        self._trace(
            "rover_frontier_route_retry_deferred",
            rover_id=int(rover_id),
            start=tuple(start),
            target=tuple(goal),
            retry_after_seconds=self._failed_route_retry_seconds,
        )

    def acquire(
        self,
        rover_id: int,
        current_pos: Tuple[int, int],
        *,
        sim_time: float = 0.0,
    ) -> Optional[Tuple[int, int]]:
        """Reserve the best current frontier staging point for one rover."""
        dependencies = self.dependencies
        normalized_id = int(rover_id)
        current_pos = tuple(current_pos)
        self._failed_route_until = {
            key: deadline
            for key, deadline in self._failed_route_until.items()
            if deadline > float(sim_time)
            and (key[0] != normalized_id or key[1] == current_pos)
        }
        raw_candidates = tuple(dependencies.get_frontier_candidates())
        candidates = []
        for candidate in raw_candidates:
            failed_key = (
                normalized_id,
                current_pos,
                tuple(candidate.position),
            )
            deadline = self._failed_route_until.get(
                failed_key,
                -math.inf,
            )
            if deadline > float(sim_time):
                if failed_key not in self._reported_failed_route_suppressions:
                    self._reported_failed_route_suppressions.add(failed_key)
                    self._trace(
                        "rover_frontier_route_retry_suppressed",
                        rover_id=normalized_id,
                        start=current_pos,
                        target=tuple(candidate.position),
                        remaining_seconds=max(
                            0.0,
                            deadline - float(sim_time),
                        ),
                    )
                continue
            candidates.append(candidate)
        candidates = tuple(candidates)
        self._reported_failed_route_suppressions.intersection_update(
            self._failed_route_until
        )
        with dependencies.assignment_lock:
            current = dependencies.assignments.get(rover_id)
            tracked = self._targets_by_rover.get(rover_id)
            focused_endgame = any(
                candidate.focused_endgame for candidate in candidates
            )
            continuation = self._lineage_continuation(
                tracked,
                candidates,
            )
            if (
                current is not None
                and continuation is not None
                and not focused_endgame
                and math.dist(current, continuation.position)
                <= self._retarget_distance
            ):
                self._targets_by_rover[rover_id] = continuation
                return current
            candidate_positions = {
                tuple(candidate.position) for candidate in candidates
            }
            if current in candidate_positions and not focused_endgame:
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
            focused_endgame = any(
                candidate.focused_endgame for candidate in available
            )
            if focused_endgame:
                selected = min(
                    available,
                    key=lambda candidate: (
                        float(candidate.remaining_max_distance),
                        float(candidate.remaining_total_distance),
                        -int(candidate.claimed),
                        float(candidate.service_cost),
                        float(candidate.terrain_roughness),
                        -float(candidate.wall_clearance),
                        float(candidate.rover_route_cost),
                        int(candidate.component_id),
                        int(candidate.task_id),
                    ),
                )
            else:
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
            if selected.focused_endgame:
                self._trace(
                    "rover_focused_endgame_staging_selected",
                    rover_id=normalized_id,
                    target=target,
                    task_id=selected.task_id,
                    component_id=selected.component_id,
                    claimed=selected.claimed,
                    remaining_max_distance=(
                        selected.remaining_max_distance
                    ),
                    remaining_total_distance=(
                        selected.remaining_total_distance
                    ),
                    current_max_distance=selected.current_max_distance,
                    current_total_distance=selected.current_total_distance,
                    rover_route_cost=selected.rover_route_cost,
                    service_cost=selected.service_cost,
                )
            return target

    def target_is_current(self, rover_id: int) -> bool:
        """Return whether the assigned component staging point remains live."""
        dependencies = self.dependencies
        candidates = tuple(dependencies.get_frontier_candidates())
        with dependencies.assignment_lock:
            target = dependencies.assignments.get(int(rover_id))
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
        if any(candidate.focused_endgame for candidate in descendants):
            return min(
                descendants,
                key=lambda candidate: (
                    float(candidate.remaining_max_distance),
                    float(candidate.remaining_total_distance),
                    -int(candidate.claimed),
                    float(candidate.service_cost),
                    float(candidate.terrain_roughness),
                    -float(candidate.wall_clearance),
                    math.dist(tracked.position, candidate.position),
                    int(candidate.component_id),
                    int(candidate.task_id),
                ),
            )
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

    def _trace(self, event: str, **fields: object) -> None:
        trace = self.dependencies.runtime_trace
        if trace is not None:
            trace.record(
                event,
                sim_time=self.dependencies.simulation_time(),
                **fields,
            )
