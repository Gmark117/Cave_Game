"""Rover agent for the Cave Explorer simulation.

This module defines rover movement and mission state. Pygame drawing is
delegated to `RoverRenderer`.
"""

import math
import random as rand
from dataclasses import dataclass
from typing import Tuple, List, Optional, TYPE_CHECKING

from agents.graph import Graph
from mapping.slam_map import SlamMap
from mapping.terrain_knowledge import TerrainKnowledge
from contracts import RoverNavigationDependencies
from rendering.agent_renderer import RoverRenderer

if TYPE_CHECKING:
    import pygame


@dataclass(frozen=True)
class RoverSnapshot:
    """Detached navigation state for rendering and the control center."""

    position: Tuple[int, int]
    target: Optional[Tuple[int, int]]
    status: str
    battery: int
    path_remaining: int
    show_path: bool
    heading_deg: float


class Rover:
    """Simple ground rover agent used for map exploration visualization.

    The rover stores runtime state and delegates Pygame drawing to its
    renderer. Types are intentionally permissive to avoid circular imports.
    """

    def __init__(self, game: object, control: object, id: int, start_pos: Tuple[int, int],
                 color: Tuple[int, int, int], icon: 'pygame.Surface', cave: list) -> None:
        """Initialize rover state and bind navigation callbacks."""
        self.game     = game
        self.settings = game.sim_settings
        self.cave     = cave
        # Navigation is injected through a small dependency object so rover
        # policy can be replaced without passing the whole MissionControl in.
        self.navigation = RoverNavigationDependencies(
            rover_targets=control.rover_targets,
            compute_rover_path=control.compute_rover_path,
            simulation_time=getattr(control, "simulation_time", lambda: 0.0),
            runtime_trace=getattr(control, "runtime_trace", None),
            announce_rendezvous=getattr(
                control,
                "announce_rendezvous",
                lambda _position: None,
            ),
            rendezvous_departure_ready=getattr(
                control,
                "rendezvous_departure_ready",
                lambda _position: True,
            ),
            rendezvous_arrived=getattr(
                control,
                "rendezvous_arrived",
                lambda _position: True,
            ),
        )
         
        self.id       = id # Unique identifier of the rover
        self.map_size = self.settings.mission_config.map_dim # Map dimension
        self.radius   = self.calculate_radius() # Radius that represent the field of view # 39
        self.step     = 10 # Step of the drone
        self.dir      = rand.randint(0,359)

        self.color = color
        self.alpha = 150
        self.icon  = icon

        self.battery  = 2400
        self.status = 'Ready'
        
        self.ray_points = []  # Initialize the list for rays
        self.delay      = control.delay

        self.show_path    = True
        self.speed_factor = 4
        self.current_path: List[Tuple[int, int]] = []
        self._announced_path: List[Tuple[int, int]] = []
        self.target: Optional[Tuple[int, int]] = None
         
        self.border    = []
        self.start_pos = start_pos
        self.pos       = start_pos
        # The source sprite faces north; headings increase clockwise.
        self.heading_deg = 0.0
        self.dir_log   = []
        self.graph     = Graph(*start_pos, cave)
        # Rovers maintain their own knowledge store; navigation and the rover
        # map view consume only this received local knowledge.
        self.terrain_knowledge = TerrainKnowledge(cave)
        map_h = len(cave)
        map_w = len(cave[0]) if map_h else 0
        max_points = self.settings.slam.point_cloud_max_points
        # The rover is the durable team checkpoint. Drones upload their local
        # occupancy belief here and download the accumulated team belief only
        # during a physical rendezvous.
        self.slam_map = SlamMap(map_h, map_w, max_points=max_points)
        self.renderer  = RoverRenderer(self)

    def snapshot(self) -> RoverSnapshot:
        """Return a detached copy of current rover navigation state."""
        return RoverSnapshot(
            position=(int(self.pos[0]), int(self.pos[1])),
            target=(
                None
                if self.target is None
                else (int(self.target[0]), int(self.target[1]))
            ),
            status=str(self.status),
            battery=int(self.battery),
            path_remaining=len(self.current_path),
            show_path=bool(self.show_path),
            heading_deg=float(self.heading_deg),
        )

    # Define the radius based on the map size
    def calculate_radius(self) -> int:
        """Return vision radius (pixels) based on chosen map size."""
        match self.map_size:
            case 'SMALL' : return 40
            case 'MEDIUM': return 20
            case 'LARGE'   : return 10
            case _       : return 20


    def move(self) -> None:
        """Move only after the next endpoint is universally acknowledged."""
        if self.navigation.rover_targets.should_hold(self.id):
            # A physical report encounter can happen halfway along a route.
            # Keep the remaining steps so movement resumes after acceptance.
            self.status = 'Rendezvous'
            return
        if self.current_path:
            self.status = 'Advancing'
            previous = self.pos
            self.pos = self.current_path.pop(0)
            delta_x = self.pos[0] - previous[0]
            delta_y = self.pos[1] - previous[1]
            if delta_x != 0 or delta_y != 0:
                self.heading_deg = math.degrees(
                    math.atan2(delta_x, -delta_y)
                ) % 360.0
            self.graph.add_node(self.pos)
            self.battery = max(0, self.battery - 1)

            if not self.current_path:
                self.status = 'Staging'
                self.navigation.rendezvous_arrived(tuple(self.pos))
                self._trace(
                    "rover_frontier_staging_reached",
                    target=self.target,
                    position=self.pos,
                )
            return

        if (
            self.target is not None
            and self.pos == self.target
            and not self.navigation.rover_targets.target_is_current(self.id)
        ):
            previous = self.target
            self.current_path.clear()
            self._announced_path.clear()
            self.navigation.rover_targets.release(self.id, completed=False)
            self.target = None
            self._trace("rover_frontier_target_invalidated", target=previous)

        if self.target is not None:
            if self.pos == self.target:
                self.status = 'Staging'
                return
            if not self.navigation.rendezvous_departure_ready(self.target):
                self.status = 'Announcing'
                return
            path = self._announced_path
            if not path:
                self.navigation.rover_targets.release(
                    self.id,
                    completed=False,
                )
                self.target = None
                self.status = 'Ready'
                return
            self.current_path = list(path)
            self._announced_path.clear()
            self.status = 'Advancing'
            self._trace(
                "rover_frontier_target_departure_authorized",
                target=self.target,
                position=self.pos,
                path_length=len(self.current_path) + 1,
            )
            return

        self.status = 'Updating'
        target = self.navigation.rover_targets.acquire(
            self.id,
            self.pos,
            sim_time=self.navigation.simulation_time(),
        )
        if target is None:
            self.status = 'Ready'
            return

        path = self.navigation.compute_rover_path(self.id, self.pos, target)
        if len(path) <= 1 and self.pos != target:
            self.navigation.rover_targets.reject_failed_route(
                self.id,
                self.pos,
                target,
                sim_time=self.navigation.simulation_time(),
            )
            self._trace(
                "rover_frontier_route_failed",
                start=self.pos,
                target=target,
            )
            self.navigation.rover_targets.release(self.id, completed=False)
            self.status = 'Ready'
            return
        self.target = target
        self._announced_path = list(path[1:])
        self.navigation.announce_rendezvous(target)
        if self.pos == target:
            self.status = 'Staging'
            return
        self.status = 'Announcing'
        self._trace(
            "rover_frontier_target_acquired",
            target=target,
            position=self.pos,
            path_length=len(path),
        )

    def _trace(self, event: str, **fields: object) -> None:
        trace = self.navigation.runtime_trace
        if trace is not None:
            trace.record(
                event,
                sim_time=self.navigation.simulation_time(),
                rover_id=int(self.id),
                **fields,
            )
