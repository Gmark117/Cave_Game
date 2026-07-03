"""Generated-map file and image-layer output."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pygame

from asset_config.media import GAME_DIR, Images
from asset_config.rendering import Colors
from MapGenHelpers import with_surfarrays


def image_path_from_key(key: str) -> Path:
    """Resolve an Images enum member by key and return its path."""

    normalized = key.upper()
    try:
        return Images[normalized].value
    except KeyError as exc:
        valid = ", ".join(item.name for item in Images)
        raise KeyError(f"Unknown image key '{key}'. Valid keys: {valid}") from exc


class MapArtifactWriter:
    """Persist generated cave images and matrix artifacts."""

    def __init__(self, base_dir: Path = GAME_DIR) -> None:
        self.base_dir = Path(base_dir)

    def write(self, bin_map: np.ndarray) -> None:
        """Write the cave image, transparent layers, and matrix."""

        self.save_map(bin_map)
        self.extract_cave_layer(Colors.WHITE.value, "CAVE_WALLS")
        self.extract_cave_layer(Colors.BLACK.value, "CAVE_FLOOR")

    def save_map(self, bin_map: np.ndarray) -> None:
        """Persist `map.png` and `map_matrix.txt` under `Assets/Map`."""

        map_dir = self.base_dir / "Assets" / "Map"
        map_dir.mkdir(parents=True, exist_ok=True)
        byte_map = np.where(bin_map == 1, 0, 255).astype(np.uint8)
        cv2.imwrite(str(map_dir / "map.png"), byte_map)
        np.savetxt(map_dir / "map_matrix.txt", bin_map)

    def extract_cave_layer(
        self,
        color_to_remove: Tuple[int, int, int],
        output_key: str,
    ) -> None:
        """Make `color_to_remove` transparent in the saved cave image."""

        cave_map = pygame.image.load(self._image_path("CAVE_MAP")).convert_alpha()
        with with_surfarrays(cave_map) as (rgb_arr, alpha_arr):
            mask = (rgb_arr == list(color_to_remove)).all(axis=2)
            alpha_arr[mask] = 0
        pygame.image.save(cave_map, self._image_path(output_key))

    def _image_path(self, key: str) -> Path:
        path = image_path_from_key(key)
        try:
            return self.base_dir / path.relative_to(GAME_DIR)
        except ValueError:
            return path
