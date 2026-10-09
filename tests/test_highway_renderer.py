import os
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from navigation.highway import HighwayEdge, HighwayGraphSnapshot
from rendering.highway_renderer import HighwayRenderer


class HighwayRendererTests(unittest.TestCase):
    @staticmethod
    def graph():
        path = ((4, 4), (4, 24), (24, 24))
        return HighwayGraphSnapshot(
            version=1, origin=(0, 0), full_shape=(32, 32),
            macro_cell_size=16, confidence_threshold=0.6,
            area_labels=np.zeros((32, 32), dtype=np.int32),
            anchors=((0, 0), path[0], path[-1]),
            adjacency=((), (HighwayEdge(2, 40.0, path),),
                       (HighwayEdge(1, 40.0, tuple(reversed(path))),)),
            known_free_cells=41, build_elapsed_ms=0.0,
        )

    def test_exact_bent_polyline_is_translucent_and_cached(self):
        renderer = HighwayRenderer(32, 32)
        graph = self.graph()
        window = pygame.Surface((32, 32), pygame.SRCALPHA)
        with patch("pygame.draw.lines", wraps=pygame.draw.lines) as lines:
            renderer.draw(window, graph)
            renderer.draw(window, graph)
        lines.assert_called_once()
        self.assertEqual(renderer.surface.get_at((4, 12)), renderer.COLOR)
        self.assertEqual(renderer.surface.get_at((12, 24)), renderer.COLOR)
        self.assertEqual(renderer.surface.get_at((12, 12)).a, 0)
        self.assertEqual(renderer.COLOR[3], round(0.85 * 255))

    def test_missing_graph_draws_nothing_and_new_graph_clears_old_edges(self):
        renderer = HighwayRenderer(32, 32)
        window = pygame.Surface((32, 32), pygame.SRCALPHA)
        window.fill((10, 20, 30, 255))
        self.assertFalse(renderer.draw(window, None))
        self.assertEqual(window.get_at((4, 12)), (10, 20, 30, 255))
        graph = self.graph()
        renderer.draw(window, graph)
        renderer.draw(window, replace(graph, version=2, adjacency=((), (), ())))
        self.assertEqual(renderer.surface.get_at((4, 12)).a, 0)


if __name__ == "__main__":
    unittest.main()
