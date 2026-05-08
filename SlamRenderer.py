"""Render SLAM occupancy grids and sparse point clouds."""

from typing import Iterable, Tuple

import numpy as np
import pygame

from SlamMap import FREE, OCCUPIED


class SlamRenderer:
    """Renders SLAM occupancy and point data into a pygame surface."""

    def __init__(self, map_w: int, map_h: int) -> None:
        self.surface = pygame.Surface((map_w, map_h), pygame.SRCALPHA)

    def render(
        self,
        occupancy: np.ndarray,
        confidence: np.ndarray,
        point_cloud: Iterable[Tuple[int, int]] | None = None,
        draw_points: bool = True
    ) -> pygame.Surface:
        """Render occupancy/confidence maps and return the surface."""
        h, w = occupancy.shape
        if w <= 0 or h <= 0:
            self.surface.fill((0, 0, 0, 0))
            return self.surface

        known_mask = confidence > 0.0
        occ_mask = occupancy == OCCUPIED
        free_mask = known_mask & (~occ_mask)

        red = np.zeros((h, w), dtype=np.float32)
        green = np.zeros((h, w), dtype=np.float32)
        blue = np.zeros((h, w), dtype=np.float32)
        alpha = np.zeros((h, w), dtype=np.float32)

        # Known free cells: white with confidence-based alpha
        red[free_mask] = 255.0
        green[free_mask] = 255.0
        blue[free_mask] = 255.0
        alpha[free_mask] = 80.0 + (confidence[free_mask] * 140.0)

        # Known occupied cells: black with confidence-based alpha
        red[occ_mask] = 0.0
        green[occ_mask] = 0.0
        blue[occ_mask] = 0.0
        alpha[occ_mask] = 120.0 + (confidence[occ_mask] * 120.0)

        red = np.clip(red, 0.0, 255.0).astype(np.uint8)
        green = np.clip(green, 0.0, 255.0).astype(np.uint8)
        blue = np.clip(blue, 0.0, 255.0).astype(np.uint8)
        alpha = np.clip(alpha, 0.0, 200.0).astype(np.uint8)

        rgb_view = pygame.surfarray.pixels3d(self.surface)
        alpha_view = pygame.surfarray.pixels_alpha(self.surface)
        rgb_view[:, :, 0] = red.T
        rgb_view[:, :, 1] = green.T
        rgb_view[:, :, 2] = blue.T
        alpha_view[:, :] = alpha.T
        del rgb_view
        del alpha_view

        if draw_points and point_cloud:
            for x, y in list(point_cloud)[-1200:]:
                if 0 <= x < w and 0 <= y < h:
                    self.surface.set_at((x, y), (255, 180, 60, 190))

        return self.surface
