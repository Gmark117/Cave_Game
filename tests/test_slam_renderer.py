import os
import unittest

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from SlamMap import FREE, OCCUPIED, UNKNOWN
from SlamRenderer import SlamRenderer


class SlamRendererTests(unittest.TestCase):
    def test_occupancy_render_distinguishes_unknown_free_and_walls(self) -> None:
        renderer = SlamRenderer(3, 1)
        occupancy = np.array([[UNKNOWN, FREE, OCCUPIED]], dtype=np.int8)
        confidence = np.array([[0.0, 1.0, 1.0]], dtype=np.float32)

        surface = renderer.render(occupancy, confidence, draw_points=False)

        self.assertEqual(surface.get_at((0, 0)).a, 0)
        self.assertEqual(surface.get_at((1, 0))[:3], (255, 255, 255))
        self.assertGreater(surface.get_at((2, 0)).r, surface.get_at((2, 0)).g)

    def test_roughness_render_uses_confidence_for_alpha(self) -> None:
        renderer = SlamRenderer(2, 1)
        roughness = np.array([[0.0, 1.0]], dtype=np.float32)
        confidence = np.array([[0.0, 1.0]], dtype=np.float32)

        surface = renderer.render(
            None,
            None,
            draw_points=False,
            roughness=roughness,
            roughness_conf=confidence,
        )

        self.assertEqual(surface.get_at((0, 0)).a, 0)
        self.assertEqual(surface.get_at((1, 0)).a, 255)
        self.assertGreater(surface.get_at((1, 0)).r, surface.get_at((1, 0)).g)


if __name__ == "__main__":
    unittest.main()
