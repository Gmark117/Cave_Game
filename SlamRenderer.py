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
        occupancy: np.ndarray | None,
        confidence: np.ndarray | None,
        point_cloud: Iterable[Tuple[int, int]] | None = None,
        draw_points: bool = True,
        roughness: np.ndarray | None = None,
        roughness_conf: np.ndarray | None = None
    ) -> pygame.Surface:
        """Render occupancy/confidence maps or a terrain roughness heatmap and return the surface.

        If `roughness` is provided, the renderer draws a colored heatmap driven by
        the roughness values (0.0..1.0) with alpha modulated by `roughness_conf`.
        Otherwise it falls back to occupancy/confidence-based rendering.
        """
        # Prefer shape from provided arrays; try occupancy then roughness
        if occupancy is not None:
            h, w = occupancy.shape
        elif roughness is not None:
            h, w = roughness.shape
        else:
            # Nothing to render
            self.surface.fill((0, 0, 0, 0))
            return self.surface
        if w <= 0 or h <= 0:
            self.surface.fill((0, 0, 0, 0))
            return self.surface


        # If roughness data is provided, render a color heatmap based on roughness
        if roughness is not None and roughness_conf is not None:
            # Only render cells with known data (confidence > 0)
            known = roughness_conf > 0.0
            r = np.clip(roughness, 0.0, 1.0)
            c = np.clip(roughness_conf, 0.0, 1.0)

            # Map roughness to color: low -> green, mid -> yellow, high -> red
            red = (r * 255.0).astype(np.float32)
            green = ((1.0 - r) * 200.0 + (r * 55.0)).astype(np.float32)
            blue = np.full((h, w), 40.0, dtype=np.float32)
            alpha = np.where(known, 100.0 + (c * 155.0), 0.0).astype(np.float32)

            red = np.clip(red, 0.0, 255.0).astype(np.uint8)
            green = np.clip(green, 0.0, 255.0).astype(np.uint8)
            blue = np.clip(blue, 0.0, 255.0).astype(np.uint8)
            alpha = np.clip(alpha, 0.0, 255.0).astype(np.uint8)
        else:
            # Occupancy-based rendering (fallback)
            if confidence is None or occupancy is None:
                self.surface.fill((0, 0, 0, 0))
                return self.surface

            known_mask = confidence > 0.0
            occ_mask = occupancy == OCCUPIED
            free_mask = known_mask & (~occ_mask)
            confidence_curve = np.power(np.clip(confidence, 0.0, 1.0), 6.0)

            red = np.zeros((h, w), dtype=np.float32)
            green = np.zeros((h, w), dtype=np.float32)
            blue = np.zeros((h, w), dtype=np.float32)
            alpha = np.zeros((h, w), dtype=np.float32)

            # Known free cells: white, with confidence encoded by a gamma curve
            free_brightness = 30.0 + (confidence_curve[free_mask] * 225.0)
            red[free_mask] = free_brightness
            green[free_mask] = free_brightness
            blue[free_mask] = free_brightness
            alpha[free_mask] = 255.0

            # Known occupied cells: warm red, with confidence encoded by the same gamma curve
            occ_brightness = 25.0 + (confidence_curve[occ_mask] * 230.0)
            red[occ_mask] = occ_brightness
            green[occ_mask] = 45.0 + (confidence_curve[occ_mask] * 60.0)
            blue[occ_mask] = 40.0
            alpha[occ_mask] = 255.0

        red = np.clip(red, 0.0, 255.0).astype(np.uint8)
        green = np.clip(green, 0.0, 255.0).astype(np.uint8)
        blue = np.clip(blue, 0.0, 255.0).astype(np.uint8)
        alpha = np.clip(alpha, 0.0, 255.0).astype(np.uint8)

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
