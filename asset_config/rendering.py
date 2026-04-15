"""Rendering enums and color/font resources."""

from enum import Enum
from pathlib import Path


GAME_DIR = Path(__file__).resolve().parent.parent


class Colors(Enum):
    """Color palette for UI and game elements."""

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    EUCALYPTUS = (95, 133, 117)
    GREENDARK = (117, 132, 104)
    YELLOW = (255, 255, 51)
    RED = (255, 0, 0)
    GREEN = (51, 255, 51)
    GREY = (100, 100, 100)
    BLUE = (0, 0, 153)


class DroneColors(Enum):
    """Colors for drones."""

    RED = (255, 0, 0)
    PINK = (255, 184, 255)
    L_BLUE = (0, 255, 255)
    ORANGE = (255, 184, 82)
    PURPLE = (148, 0, 221)
    BROWN = (160, 82, 45)
    GREEN = (34, 139, 34)
    GOLD = (255, 215, 0)


class RoverColors(Enum):
    """Colors for rovers."""

    RED = (220, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 128, 0)


class Fonts(Enum):
    """Font resources."""

    SMALL = GAME_DIR / "Assets" / "Fonts" / "Cave-Stone.ttf"
    BIG = GAME_DIR / "Assets" / "Fonts" / "CroMagnum.ttf"


class RectHandle(Enum):
    """Pygame rect anchor points."""

    CENTER = "center"
    MIDTOP = "midtop"
    MIDRIGHT = "midright"
    MIDLEFT = "midleft"
