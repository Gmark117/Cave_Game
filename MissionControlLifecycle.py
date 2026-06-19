"""Mission lifecycle helpers for MissionControl.

This mixin owns the run loop, mission-completion check, and shutdown
sequence so MissionControl can focus on setup and agent coordination.
"""

import threading
import time
from typing import List

import pygame


class MissionControlLifecycleMixin:
    """Mixin that encapsulates mission execution and teardown."""

    def is_mission_over(self) -> bool:
        """Return True when all drones report mission completion."""
        if not self.drones:
            return False

        for drone in self.drones:
            if not drone.mission_completed():
                return False
        return True

    def _shutdown_mission(self, threads: List[threading.Thread]) -> None:
        """Stop workers, join threads, and release process/shared-memory resources."""
        self.mission_event.set()
        self.pause_event.set()

        for thread in threads:
            thread.join()

        self.pathfinding.shutdown()

        self.clock = None
        self._runtime_initialized = False

    def _start_agent_threads(self) -> List[threading.Thread]:
        """Create and start mission worker threads."""
        threads: List[threading.Thread] = []
        for i in range(self.num_drones):
            thread = threading.Thread(target=self.drone_thread, args=(i,))
            threads.append(thread)
            thread.start()

        if self.rover_motion_enabled:
            for i in range(self.num_rovers):
                thread = threading.Thread(target=self.rover_thread, args=(i,))
                threads.append(thread)
                thread.start()

        return threads

    def _run_mission_loop(self) -> None:
        """Process events, update simulation state, and render frames."""
        if self.clock is None:
            raise RuntimeError("Mission runtime clock is not initialized")

        fps = max(1, round(1 / self.delay))
        while not self.completed:
            frame_started = time.perf_counter()
            self.clock.tick(fps)
            wait_finished = time.perf_counter()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.completed = True
                    pygame.quit()
                    raise SystemExit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.stop_button_rect.collidepoint(event.pos):
                        self.completed = True
                        break
                    if self.restart_button_rect.collidepoint(event.pos):
                        self.restart_requested = True
                        self.completed = True
                        break
                    if self.pause_button_rect.collidepoint(event.pos):
                        self.toggle_pause()
                        continue
                    self.presentation.handle_click(
                        event.pos,
                        self.control_center,
                        self.drones,
                    )
            events_finished = time.perf_counter()

            if self.completed:
                break

            if not self.is_paused:
                self._share_terrain_with_rovers()
            sharing_finished = time.perf_counter()
            if not self.is_paused:
                self.completed = self.is_mission_over()
            status_finished = time.perf_counter()
            if not self.is_paused:
                self.update_sensors()
            sensors_finished = time.perf_counter()
            self.renderer.draw()
            render_finished = time.perf_counter()
            pygame.display.update()
            frame_finished = time.perf_counter()

            self.frame_profiler.record(
                frame_seconds=frame_finished - frame_started,
                wait_seconds=wait_finished - frame_started,
                stages={
                    "events": events_finished - wait_finished,
                    "sharing": sharing_finished - events_finished,
                    "status": status_finished - sharing_finished,
                    "sensors": sensors_finished - status_finished,
                    "render": render_finished - sensors_finished,
                    "display": frame_finished - render_finished,
                },
            )

    def run(self) -> None:
        """Initialize and run this mission exactly once."""
        if self._running:
            raise RuntimeError("Mission is already running")
        if self._has_run:
            raise RuntimeError("MissionControl instances are single-use")

        threads: List[threading.Thread] = []
        self._running = True
        try:
            self._initialize_runtime()
            if self.control_center is None:
                raise RuntimeError("Mission control center is not initialized")
            self.control_center.start_timer()
            threads = self._start_agent_threads()
            self._run_mission_loop()
        finally:
            self._shutdown_mission(threads)
            self._running = False
            self._has_run = True
            if pygame.get_init() and not self.restart_requested:
                self.game.display = self.game.to_windowed()

    def start_mission(self) -> None:
        """Compatibility alias for `run()`."""
        self.run()
