"""Coordinated mission pausing and pause-aware simulation time."""

import threading
import time
from typing import Callable, Hashable


class SimulationClock:
    """Return monotonic time with paused intervals removed."""

    def __init__(
        self,
        time_source: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._time_source = time_source
        self._lock = threading.Lock()
        self._paused_at: float | None = None
        self._paused_duration = 0.0

    def now(self) -> float:
        """Return the current pause-adjusted monotonic timestamp."""
        with self._lock:
            current = (
                self._paused_at
                if self._paused_at is not None
                else self._time_source()
            )
            return current - self._paused_duration

    def pause(self) -> None:
        """Freeze simulation time if it is currently running."""
        with self._lock:
            if self._paused_at is None:
                self._paused_at = self._time_source()

    def resume(self) -> None:
        """Resume simulation time without counting the paused interval."""
        with self._lock:
            if self._paused_at is None:
                return
            self._paused_duration += self._time_source() - self._paused_at
            self._paused_at = None


class PauseCoordinator:
    """Park registered workers at cooperative safe points."""

    ACTIVE = "active"
    PAUSED = "paused"

    def __init__(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event
        self._condition = threading.Condition()
        self._paused = False
        self._stopped = False
        self._workers: dict[int, tuple[Hashable, str]] = {}

    def register_current_worker(self, label: Hashable) -> bool:
        """Register the calling thread and honor an existing pause."""
        worker_id = threading.get_ident()
        with self._condition:
            self._workers[worker_id] = (label, self.ACTIVE)
            self._condition.notify_all()
        return self.checkpoint()

    def unregister_current_worker(self) -> None:
        """Remove the calling thread from the pause barrier."""
        worker_id = threading.get_ident()
        with self._condition:
            self._workers.pop(worker_id, None)
            self._condition.notify_all()

    def checkpoint(self) -> bool:
        """Block the calling worker while paused.

        Returns False when mission shutdown has begun.
        """
        worker_id = threading.get_ident()
        with self._condition:
            while True:
                if self._stopped or self._stop_event.is_set():
                    return False

                worker = self._workers.get(worker_id)
                if worker is None:
                    return not self._paused

                label, _ = worker
                if not self._paused:
                    self._workers[worker_id] = (label, self.ACTIVE)
                    return True

                self._workers[worker_id] = (label, self.PAUSED)
                self._condition.notify_all()
                self._condition.wait()

    def wait(self, duration: float) -> bool:
        """Wait for simulation time while reacting immediately to pause/stop."""
        remaining = max(0.0, float(duration))

        while remaining > 0.0:
            if not self.checkpoint():
                return False

            started = time.perf_counter()
            with self._condition:
                if self._stopped or self._stop_event.is_set():
                    return False
                self._condition.wait(timeout=remaining)
            elapsed = max(0.0, time.perf_counter() - started)
            remaining -= elapsed

        return self.checkpoint()

    def pause(self) -> None:
        """Close the gate and wait until all registered workers are parked."""
        with self._condition:
            self._paused = True
            self._condition.notify_all()
            self._condition.wait_for(
                lambda: self._stopped
                or all(
                    state == self.PAUSED
                    for _, state in self._workers.values()
                )
            )

    def resume(self) -> None:
        """Open the gate and wake all parked workers."""
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def stop(self) -> None:
        """Wake every worker so mission shutdown can complete."""
        with self._condition:
            self._stopped = True
            self._paused = False
            self._condition.notify_all()
