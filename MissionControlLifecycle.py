"""Mission lifecycle helpers for MissionControl.

This mixin owns the run loop, mission-completion check, and shutdown
sequence so MissionControl can focus on setup and agent coordination.
"""

import sys
import threading
from typing import List

import pygame


class MissionControlLifecycleMixin:
    """Mixin that encapsulates mission execution and teardown."""

    def is_mission_over(self) -> bool:
        """Return True when all drones report mission completion."""
        for drone in self.drones:
            if not drone.mission_completed():
                return False

        self.game.display = self.game.to_windowed()
        return True

    def _shutdown_mission(self, threads: List[threading.Thread]) -> None:
        """Stop workers, join threads, and release process/shared-memory resources."""
        self.mission_event.set()

        for thread in threads:
            thread.join()

        try:
            self.pool.shutdown(wait=True)
        except (RuntimeError, ValueError, OSError):
            pass

        if self.map_shm is not None:
            try:
                self.map_shm.close()
            except (BufferError, OSError):
                pass
            try:
                self.map_shm.unlink()
            except FileNotFoundError:
                pass
            except (BufferError, OSError):
                pass
            finally:
                self.map_shm = None

    def start_mission(self) -> None:
        """Run the mission loop until all drones complete or the user exits."""
        self.control_center.start_timer()
        fps = max(1, round(1 / self.delay))

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

        try:
            while not self.completed:
                self.clock.tick(fps)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._shutdown_mission(threads)
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.presentation.handle_click(event.pos, self.control_center, self.drones)

                self._share_terrain_with_rovers()
                self.completed = self.is_mission_over()
                self.draw()
                pygame.display.update()
        finally:
            self._shutdown_mission(threads)
