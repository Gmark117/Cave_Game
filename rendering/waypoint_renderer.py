"""Cached overlay rendering for the mission waypoint highway graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pygame

from navigation.waypoint_graph import (
    EDGE_KNOWN_FREE_CORRIDOR,
    EDGE_SLAM_LOS,
    EDGE_TRAVELLED,
)


Color = tuple[int, int, int, int]


@dataclass(frozen=True)
class _EdgeStyle:
    """Visual style for one persistent waypoint-edge source."""

    color: Color
    width: int


@dataclass(frozen=True)
class _NodeStyle:
    """Visual style for one persistent waypoint-node source."""

    color: Color
    radius: int


_DEFAULT_EDGE_STYLE = _EdgeStyle((225, 225, 225, 105), 1)
_EDGE_STYLES: Mapping[str, _EdgeStyle] = {
    EDGE_TRAVELLED: _EdgeStyle((55, 205, 255, 150), 2),
    EDGE_SLAM_LOS: _EdgeStyle((255, 196, 64, 135), 1),
    EDGE_KNOWN_FREE_CORRIDOR: _EdgeStyle((185, 120, 255, 170), 2),
}

_DEFAULT_NODE_STYLE = _NodeStyle((240, 240, 240, 220), 3)
_NODE_STYLES: Mapping[str, _NodeStyle] = {
    "home": _NodeStyle((80, 140, 255, 245), 5),
    EDGE_TRAVELLED: _NodeStyle((80, 235, 175, 225), 3),
    "known_free": _NodeStyle((185, 120, 255, 225), 3),
    "gateway": _NodeStyle((255, 126, 45, 240), 4),
}


class WaypointRenderer:
    """Render a versioned waypoint graph into one reusable map overlay.

    Graph changes happen on drone worker threads.  Reading ``graph.version``
    before requesting a detached snapshot keeps unchanged frames inexpensive;
    the cached transparent surface is simply blitted again.
    """

    def __init__(self, graph: Any, map_width: int, map_height: int) -> None:
        """Create a transparent map-sized cache for ``graph``."""
        self.graph = graph
        self.surface = pygame.Surface(
            (int(map_width), int(map_height)),
            pygame.SRCALPHA,
        )
        self.surface.fill((0, 0, 0, 0))
        self._rendered_version = -1

    @property
    def rendered_version(self) -> int:
        """Return the graph version currently represented by the cache."""
        return self._rendered_version

    def draw(self, target: pygame.Surface) -> None:
        """Refresh the cache when necessary and blit it onto ``target``."""
        graph_version = int(self.graph.version)
        if graph_version != self._rendered_version:
            snapshot = self.graph.snapshot()
            snapshot_version = int(snapshot.version)
            if snapshot_version != self._rendered_version:
                self._rebuild(snapshot)
                self._rendered_version = snapshot_version

        target.blit(self.surface, (0, 0))

    def _rebuild(self, snapshot: Any) -> None:
        """Draw one immutable graph snapshot in stable edge/node order."""
        self.surface.fill((0, 0, 0, 0))

        # Edges are intentionally drawn first so every node remains legible at
        # junctions and at the ends of overlapping travelled polylines.
        for edge in snapshot.edges:
            path = tuple(edge.path)
            if len(path) < 2:
                continue
            style = _EDGE_STYLES.get(edge.source, _DEFAULT_EDGE_STYLE)
            pygame.draw.lines(
                self.surface,
                style.color,
                False,
                path,
                style.width,
            )

        for node in snapshot.waypoints:
            style = _NODE_STYLES.get(node.source, _DEFAULT_NODE_STYLE)
            pygame.draw.circle(
                self.surface,
                style.color,
                node.position,
                style.radius,
            )
