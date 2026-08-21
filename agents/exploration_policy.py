"""Small exploration-policy primitives for drone movement."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Iterable


class RandomDirectionPolicy:
    """Choose uniformly or by supplied weights with a private RNG."""

    def __init__(self, *, seed: int) -> None:
        self._random = random.Random(int(seed))

    def choose_direction(self, valid_directions: Iterable[int]) -> int:
        """Return one valid integer heading.

        A private seeded generator keeps each drone reproducible without
        coupling choices to thread scheduling or unrelated global randomness.
        """
        choices = tuple(int(direction) for direction in valid_directions)
        if not choices:
            raise ValueError("valid_directions must not be empty")
        return self._random.choice(choices)

    def choose_weighted_direction(
        self,
        direction_weights: Mapping[int, float],
    ) -> int:
        """Choose reproducibly from positive per-heading weights."""
        weighted = tuple(
            (int(direction), max(0.0, float(weight)))
            for direction, weight in direction_weights.items()
        )
        if not weighted:
            raise ValueError("direction_weights must not be empty")
        total = sum(weight for _direction, weight in weighted)
        if total <= 0.0:
            return self.choose_direction(
                direction for direction, _weight in weighted
            )

        sample = self._random.random() * total
        cumulative = 0.0
        for direction, weight in weighted:
            cumulative += weight
            if sample < cumulative:
                return direction
        return weighted[-1][0]
