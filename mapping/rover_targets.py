"""Provisional rover target selection and reservation logic.

Rover movement is disabled. This service currently reads mission telemetry and
must be changed to rover-local received knowledge when rover policy is defined.
"""

import math
from typing import Optional, Tuple

import numpy as np

from mission.service_dependencies import RoverTargetDependencies


class RoverTargetService:
    """Choose provisional terrain targets while rover movement is disabled."""

    def __init__(self, dependencies: RoverTargetDependencies) -> None:
        self.dependencies = dependencies

    def acquire(
        self, rover_id: int, current_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """Choose and reserve a discovered rough-terrain target for a rover."""
        dependencies = self.dependencies
        terrain = dependencies.terrain_knowledge.snapshot()
        with dependencies.assignment_lock:
            assigned_targets = {
                target
                for rid, target in dependencies.assignments.items()
                if rid != rover_id
            }
            candidate_mask = (
                (np.asarray(dependencies.cave_map) == 0)
                & (terrain.confidence >= 0.25)
                & (terrain.roughness >= 0.35)
            )

            if not np.any(candidate_mask):
                return None

            ys, xs = np.where(candidate_mask)
            best_target = None
            best_score = float("-inf")
            norm = max(
                1.0,
                math.hypot(
                    dependencies.norm_width,
                    dependencies.norm_height,
                ),
            )

            for x, y in zip(xs, ys):
                target = (int(x), int(y))
                if (
                    target in assigned_targets
                    or target in dependencies.completed_targets
                ):
                    continue

                distance_penalty = math.dist(current_pos, target) / norm
                score = (
                    (0.7 * float(terrain.roughness[y, x]))
                    + (0.3 * float(terrain.confidence[y, x]))
                    - distance_penalty
                )
                if score > best_score:
                    best_score = score
                    best_target = target

            if best_target is not None:
                dependencies.assignments[rover_id] = best_target
            return best_target

    def release(self, rover_id: int, completed: bool = False) -> None:
        """Release or mark complete a rover terrain target reservation."""
        dependencies = self.dependencies
        with dependencies.assignment_lock:
            target = dependencies.assignments.pop(rover_id, None)
            if completed and target is not None:
                dependencies.completed_targets.add(target)
