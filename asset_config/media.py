"""Media resource paths."""

from enum import Enum
from pathlib import Path


GAME_DIR = Path(__file__).resolve().parent.parent


class Audio(Enum):
    """Audio resources."""

    AMBIENT = GAME_DIR / "Assets" / "Audio" / "Menu.wav"
    BUTTON = GAME_DIR / "Assets" / "Audio" / "Button.wav"


class Images(Enum):
    """Image resources."""

    CAVE = GAME_DIR / "Assets" / "Images" / "cave.jpg"
    DARK_CAVE = GAME_DIR / "Assets" / "Images" / "cave_black.jpg"
    GAME_ICON = GAME_DIR / "Assets" / "Images" / "drone.png"
    GAME_ICON_BG = GAME_DIR / "Assets" / "Images" / "drone_BG.jpg"
    ROVER = GAME_DIR / "Assets" / "Images" / "rover_top.png"
    DRONE = GAME_DIR / "Assets" / "Images" / "drone_top.png"

    CAVE_MAP = GAME_DIR / "Assets" / "Map" / "map.png"
    CAVE_MATRIX = GAME_DIR / "Assets" / "Map" / "map_matrix.txt"
    CAVE_WALLS = GAME_DIR / "Assets" / "Map" / "walls.png"
    CAVE_FLOOR = GAME_DIR / "Assets" / "Map" / "floor.png"
