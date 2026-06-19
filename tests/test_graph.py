import unittest

import numpy as np

from Graph import Graph


class GraphTests(unittest.TestCase):
    def test_accepts_clear_floor_route(self) -> None:
        cave = np.zeros((5, 5), dtype=np.uint8)
        graph = Graph(0, 0, cave)

        self.assertTrue(graph.is_valid((0, 0), (4, 4)))

    def test_rejects_route_crossing_wall(self) -> None:
        cave = np.zeros((5, 5), dtype=np.uint8)
        cave[2, 2] = 1
        graph = Graph(0, 0, cave)

        self.assertFalse(graph.is_valid((0, 0), (4, 4)))


if __name__ == "__main__":
    unittest.main()

