"""Battery-independent energy contracts for exploration coordination.

The first component-exploration implementation intentionally uses the
unlimited policy below.  Keeping route, action, home, and reserve costs in the
contract makes later battery drain and charging an implementation change
rather than another exploration-policy rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EnergyState:
    """One detached energy reading used by a policy decision."""

    remaining_energy: float
    capacity: float
    unlimited: bool = False


@dataclass(frozen=True)
class EnergyRequirement:
    """Conservative costs required to accept one exploration action."""

    route_to_task_cost: float
    next_action_cost: float
    route_home_cost: float
    safety_reserve: float

    @property
    def total(self) -> float:
        return sum((
            self.route_to_task_cost,
            self.next_action_cost,
            self.route_home_cost,
            self.safety_reserve,
        ))


@dataclass(frozen=True)
class EnergyReturnDecision:
    """One executor-side reserve decision with all inputs preserved."""

    state: EnergyState
    route_home_cost: float
    next_action_cost: float
    safety_reserve: float
    must_return: bool


class EnergyPolicy(Protocol):
    """Decision surface consumed by the coordinator and drone executor."""

    def can_accept(
        self,
        state: EnergyState,
        requirement: EnergyRequirement,
    ) -> bool:
        """Return whether a drone may safely accept the quoted work."""
        ...

    def must_return(
        self,
        state: EnergyState,
        *,
        route_home_cost: float,
        next_action_cost: float,
        safety_reserve: float,
    ) -> bool:
        """Return whether the next action would consume the return reserve."""
        ...


class UnlimitedEnergyPolicy:
    """Initial policy: expose all hooks without draining or constraining work."""

    def state(self, *, remaining_energy: float = 100.0) -> EnergyState:
        return EnergyState(
            remaining_energy=float(remaining_energy),
            capacity=max(100.0, float(remaining_energy)),
            unlimited=True,
        )

    def can_accept(
        self,
        state: EnergyState,
        requirement: EnergyRequirement,
    ) -> bool:
        del requirement
        return bool(state.unlimited)

    def must_return(
        self,
        state: EnergyState,
        *,
        route_home_cost: float,
        next_action_cost: float,
        safety_reserve: float,
    ) -> bool:
        del route_home_cost, next_action_cost, safety_reserve
        return not state.unlimited and state.remaining_energy < 0.0


class ReserveEnergyPolicy:
    """Pure finite-energy policy ready for later drain/charging integration."""

    def can_accept(
        self,
        state: EnergyState,
        requirement: EnergyRequirement,
    ) -> bool:
        return bool(
            state.unlimited
            or state.remaining_energy >= requirement.total
        )

    def must_return(
        self,
        state: EnergyState,
        *,
        route_home_cost: float,
        next_action_cost: float,
        safety_reserve: float,
    ) -> bool:
        if state.unlimited:
            return False
        required = (
            float(route_home_cost)
            + float(next_action_cost)
            + float(safety_reserve)
        )
        return state.remaining_energy < required
