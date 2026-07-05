"""Runtime facade for procedural cave map generation.

`MapGenerator` keeps the legacy game-facing API while focused collaborators
own cave generation, worker-process management, post-processing, roughness
synthesis, and artifact output.
"""

import logging

import pygame

from asset_config.gameplay import Display
from asset_config.mapgen import MapGen
from generation.cave_generator import (
    CaveGenerationProgress,
    CaveGenerator,
)
from generation.map_artifact_writer import MapArtifactWriter


logger = logging.getLogger(__name__)


class MapGenerator:
    """Build the cave artifacts consumed by mission setup."""

    NUM_PROCESSES = getattr(MapGen, "DEFAULT_NUM_PROCESSES", 8)

    def __init__(self, game, settings) -> None:
        """Generate a cave map and expose legacy attributes for MissionControl."""
        self.game = game
        self.settings = settings
        self.width = Display.FULL_W - Display.LEGEND_WIDTH
        self.height = Display.FULL_H

        mission_config = self.settings.mission_config
        # The pure generator returns arrays and metadata; this facade handles UI
        # progress callbacks and writing runtime map artifacts afterward.
        generator = CaveGenerator(
            self.width,
            self.height,
            mission_config.seed,
            mission_config.map_dim,
            self.NUM_PROCESSES,
        )
        result = generator.generate(
            CaveGenerationProgress(
                on_digging=self._show_digging_progress,
                on_post_processing=self._show_post_processing_progress,
            )
        )

        self.bin_map = result.bin_map
        self.terrain_roughness = result.terrain_roughness
        self.worm_x = result.worm_x
        self.worm_y = result.worm_y
        self.worm_inputs = result.worm_inputs
        self.proc_counter = result.completed_workers
        self.worker_crashed = result.worker_crashed

        MapArtifactWriter().write(self.bin_map)

    def _show_digging_progress(self) -> None:
        """Show the digging loading state if the menu is available."""

        try:
            self.game.menu.blit_loading(["Digging..."])
        except (AttributeError, pygame.error) as exc:
            logger.debug("Loading overlay unavailable during dig phase: %s", exc)
            pass

    def _show_post_processing_progress(self) -> None:
        """Show the post-processing loading state."""

        self.game.menu.blit_loading(["Breeding bats..."])
