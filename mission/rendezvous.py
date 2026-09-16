"""Contact-carried rendezvous endpoint and acknowledgement protocol."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable


Position = tuple[int, int]


@dataclass(frozen=True)
class RendezvousEndpoint:
    """One rover endpoint announcement with a monotonic identity."""

    epoch: int
    position: Position


@dataclass(frozen=True)
class RendezvousSnapshot:
    current: RendezvousEndpoint
    proposal: RendezvousEndpoint | None
    rover_acknowledgements: frozenset[int]
    drone_endpoints: tuple[RendezvousEndpoint, ...]
    drone_announcements: tuple[RendezvousEndpoint, ...]
    drone_confirmed_endpoints: tuple[RendezvousEndpoint, ...]
    departure_ready: bool


class RendezvousProtocol:
    """Move endpoint knowledge only during verified physical contacts.

    The rover may announce a candidate at any time, but it cannot depart until
    every drone has acknowledged that epoch and those acknowledgements have
    reached the rover.  Drone-to-drone contacts relay both the newest endpoint
    and acknowledgement sets; no out-of-band broadcast exists.
    """

    def __init__(
        self,
        drone_count: int,
        initial_position: Position,
        *,
        trace: Callable[..., None] | None = None,
    ) -> None:
        if int(drone_count) <= 0:
            raise ValueError("rendezvous drone_count must be positive")
        self.drone_count = int(drone_count)
        initial = RendezvousEndpoint(
            0,
            (int(initial_position[0]), int(initial_position[1])),
        )
        self._lock = threading.RLock()
        self._current = initial
        self._proposal: RendezvousEndpoint | None = None
        # Announcements are knowledge; targets are the last physically
        # confirmed rendezvous point each drone will try first. Keeping them
        # separate prevents a proposal from pulling every acknowledgement
        # carrier away from the rover before the rover may depart.
        self._drone_announcements = {
            drone_id: initial for drone_id in range(self.drone_count)
        }
        self._drone_targets = {
            drone_id: initial for drone_id in range(self.drone_count)
        }
        # A rover visit is stronger evidence than a proposed destination.
        # Carry it separately so a drone that missed several stops can visit
        # the newest rover-confirmed stop before chasing a later proposal.
        self._drone_confirmed = {
            drone_id: initial for drone_id in range(self.drone_count)
        }
        self._drone_ack_ledgers = {
            drone_id: {0: {drone_id}}
            for drone_id in range(self.drone_count)
        }
        self._rover_ack_ledgers: dict[int, set[int]] = {
            0: set(range(self.drone_count)),
        }
        self._next_epoch = 1
        self._trace_callback = trace

    def propose(self, position: Position) -> RendezvousEndpoint:
        """Create or retain the rover's endpoint proposal."""
        target = int(position[0]), int(position[1])
        with self._lock:
            if self._proposal is not None and self._proposal.position == target:
                return self._proposal
            if self._proposal is None and self._current.position == target:
                return self._current
            proposal = RendezvousEndpoint(self._next_epoch, target)
            self._next_epoch += 1
            self._proposal = proposal
            self._rover_ack_ledgers[proposal.epoch] = set()
            self._trace(
                "rover_rendezvous_endpoint_proposed",
                rendezvous_epoch=proposal.epoch,
                endpoint=proposal.position,
            )
            return proposal

    def drone_rover_contact(self, drone_id: int) -> None:
        """Exchange endpoint and ack ledgers over one verified contact."""
        normalized = self._validate_drone(drone_id)
        with self._lock:
            # Physical contact confirms where the rover currently is. A newer
            # proposal remains known but does not become this drone's target
            # until it later observes that the rover left this endpoint.
            self._learn_endpoint(normalized, self._current)
            self._drone_targets[normalized] = self._current
            self._drone_confirmed[normalized] = self._current
            if self._proposal is not None:
                self._learn_endpoint(normalized, self._proposal)
            drone_ledger = self._drone_ack_ledgers[normalized]
            for epoch, acknowledgements in drone_ledger.items():
                self._rover_ack_ledgers.setdefault(epoch, set()).update(
                    acknowledgements
                )
            proposal = self._proposal
            if proposal is not None:
                acknowledgements = self._rover_ack_ledgers.setdefault(
                    proposal.epoch,
                    set(),
                )
                acknowledgements.add(normalized)
                drone_ledger.setdefault(proposal.epoch, set()).update(
                    acknowledgements
                )
                self._trace(
                    "drone_rover_rendezvous_ack_exchanged",
                    drone_id=normalized,
                    rendezvous_epoch=proposal.epoch,
                    acknowledgement_count=len(acknowledgements),
                    required_count=self.drone_count,
                )

    def drone_drone_contact(self, first_id: int, second_id: int) -> None:
        """Relay the newest endpoint and acknowledgements between two drones."""
        first = self._validate_drone(first_id)
        second = self._validate_drone(second_id)
        if first == second:
            return
        with self._lock:
            freshest = max(
                self._drone_announcements[first],
                self._drone_announcements[second],
                key=lambda endpoint: endpoint.epoch,
            )
            self._learn_endpoint(first, freshest)
            self._learn_endpoint(second, freshest)
            confirmed = max(
                self._drone_confirmed[first],
                self._drone_confirmed[second],
                key=lambda endpoint: endpoint.epoch,
            )
            self._drone_confirmed[first] = confirmed
            self._drone_confirmed[second] = confirmed
            first_ledger = self._drone_ack_ledgers[first]
            second_ledger = self._drone_ack_ledgers[second]
            epochs = set(first_ledger) | set(second_ledger)
            for epoch in epochs:
                combined = set(first_ledger.get(epoch, ()))
                combined.update(second_ledger.get(epoch, ()))
                first_ledger[epoch] = set(combined)
                second_ledger[epoch] = set(combined)
            self._trace(
                "drone_rendezvous_message_relayed",
                first_drone_id=first,
                second_drone_id=second,
                rendezvous_epoch=freshest.epoch,
                confirmed_epoch=confirmed.epoch,
                confirmed_endpoint=confirmed.position,
                acknowledgement_count=len(first_ledger.get(freshest.epoch, ())),
            )

    def drone_endpoint(self, drone_id: int) -> Position:
        """Return the rendezvous target selected from local contact history."""
        normalized = self._validate_drone(drone_id)
        with self._lock:
            return self._drone_targets[normalized].position

    def drone_missed_endpoint(
        self,
        drone_id: int,
        position: Position,
    ) -> Position:
        """Fall forward after a drone physically finds its old endpoint empty."""
        normalized = self._validate_drone(drone_id)
        observed = int(position[0]), int(position[1])
        with self._lock:
            current_target = self._drone_targets[normalized]
            announcement = self._drone_announcements[normalized]
            confirmed = self._drone_confirmed[normalized]
            if current_target.position != observed:
                return current_target.position
            if confirmed.epoch > current_target.epoch:
                next_target = confirmed
                source = "rover_confirmed_relay"
            elif announcement.epoch > current_target.epoch:
                next_target = announcement
                source = "proposal_after_absence"
            else:
                return current_target.position
            self._drone_targets[normalized] = next_target
            self._trace(
                "drone_rendezvous_endpoint_fallback",
                drone_id=normalized,
                previous_epoch=current_target.epoch,
                previous_endpoint=current_target.position,
                rendezvous_epoch=next_target.epoch,
                endpoint=next_target.position,
                target_source=source,
                reason="rover_absent_at_confirmed_endpoint",
            )
            return next_target.position

    def can_depart(self, position: Position) -> bool:
        """Return whether universal acknowledgement reached the rover."""
        target = int(position[0]), int(position[1])
        with self._lock:
            if target == self._current.position and self._proposal is None:
                return True
            proposal = self._proposal
            if proposal is None or proposal.position != target:
                return False
            return len(self._rover_ack_ledgers.get(proposal.epoch, ())) == (
                self.drone_count
            )

    def rover_arrived(self, position: Position) -> bool:
        """Commit a universally acknowledged endpoint after physical arrival."""
        target = int(position[0]), int(position[1])
        with self._lock:
            proposal = self._proposal
            if (
                proposal is None
                or proposal.position != target
                or not self.can_depart(target)
            ):
                return False
            self._current = proposal
            self._proposal = None
            self._trace(
                "rover_rendezvous_endpoint_reached",
                rendezvous_epoch=proposal.epoch,
                endpoint=proposal.position,
            )
            return True

    def snapshot(self) -> RendezvousSnapshot:
        with self._lock:
            proposal = self._proposal
            acknowledgements = frozenset(
                ()
                if proposal is None
                else self._rover_ack_ledgers.get(proposal.epoch, ())
            )
            return RendezvousSnapshot(
                current=self._current,
                proposal=proposal,
                rover_acknowledgements=acknowledgements,
                drone_endpoints=tuple(
                    self._drone_targets[drone_id]
                    for drone_id in range(self.drone_count)
                ),
                drone_announcements=tuple(
                    self._drone_announcements[drone_id]
                    for drone_id in range(self.drone_count)
                ),
                drone_confirmed_endpoints=tuple(
                    self._drone_confirmed[drone_id]
                    for drone_id in range(self.drone_count)
                ),
                departure_ready=(
                    proposal is not None
                    and len(acknowledgements) == self.drone_count
                ),
            )

    def _learn_endpoint(
        self,
        drone_id: int,
        endpoint: RendezvousEndpoint,
    ) -> None:
        if endpoint.epoch < self._drone_announcements[drone_id].epoch:
            return
        self._drone_announcements[drone_id] = endpoint
        self._drone_ack_ledgers[drone_id].setdefault(
            endpoint.epoch,
            set(),
        ).add(drone_id)

    def _validate_drone(self, drone_id: int) -> int:
        normalized = int(drone_id)
        if not 0 <= normalized < self.drone_count:
            raise ValueError("drone_id is outside the rendezvous team")
        return normalized

    def _trace(self, event: str, **fields: object) -> None:
        if self._trace_callback is not None:
            self._trace_callback(event, **fields)
