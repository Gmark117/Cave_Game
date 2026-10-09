"""Cached display of exact polylines from the rover's highway graph."""

import pygame

from navigation.highway import HighwayGraphSnapshot


class HighwayRenderer:
    """Draw navigation advice without building or querying a highway."""

    COLOR = (160, 160, 160, 217)  # 85% opacity.
    WIDTH = 3

    def __init__(self, width: int, height: int) -> None:
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._snapshot: HighwayGraphSnapshot | None = None

    def draw(
        self,
        window: pygame.Surface,
        snapshot: HighwayGraphSnapshot | None,
    ) -> bool:
        """Blit each undirected edge once, rebuilding only for a new graph."""
        if snapshot is None:
            return False
        if snapshot is not self._snapshot:
            self.surface.fill((0, 0, 0, 0))
            for source_area, edges in enumerate(snapshot.adjacency):
                for edge in edges:
                    if source_area < edge.target_area and len(edge.path) >= 2:
                        pygame.draw.lines(
                            self.surface, self.COLOR, False, edge.path, self.WIDTH,
                        )
            self._snapshot = snapshot
        window.blit(self.surface, (0, 0))
        return True
