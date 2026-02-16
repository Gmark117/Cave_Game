"""
Assets module for Cave Explorer game.

This module centralizes all game constants, enums, and resource paths.
Organized into logical sections for maintainability.
"""

from pathlib import Path
from enum import Enum
from typing import Tuple


# =============================================================================
# GAME DIRECTORY
# =============================================================================

GAME_DIR = Path(__file__).parent


# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

class Display:
    """Display dimensions and positions."""
    W = 1200
    H = 750
    FULL_W = 1915  # Hardcoded; adjust for your screen
    FULL_H = 1010
    LEGEND_WIDTH = 300
    
    # Calculated positions
    CENTER_W = W / 2
    CENTER_H = H / 2
    ALIGN_L = CENTER_W - 550


# =============================================================================
# MENU AND GAME STATES
# =============================================================================

class GameOptions:
    """Game configuration options."""
    MISSION = ["Exploration", "Search and Rescue"]
    MAP_SIZE = ["Small", "Medium", "Big"]
    PREFAB = ['No', 'Yes']
    VISION = [39, 19, 4]  # Vision ranges for map sizes
    DRONE_ICON = [(30, 30), (20, 20), (10, 10)]
    ROVER_ICON = [(40, 40), (25, 25), (10, 10)]
    SEED_DEFAULTS = [5, 19, 837]  # Default seeds for map sizes


# =============================================================================
# MAP GENERATION
# =============================================================================

class MapGen:
    """Map generation parameters."""
    STEP = 10
    STRENGTH = 16
    LIFE = 75

    # Chaotic brush factor range (fraction of strength)
    CHAOTIC_FACTOR_LOW = 0.2
    CHAOTIC_FACTOR_HIGH = 0.4

    # Wall transition noise defaults
    WALL_NOISE_BASE_DEPTH = 3
    WALL_NOISE_BASE_PROB = 0.6
    WALL_NOISE_MIN_SPIKES = 24
    WALL_NOISE_SPIKE_DIVISOR = 25
    WALL_NOISE_SPIKE_EXTRA = 6
    WALL_NOISE_DILATE_KSIZE = 3
    WALL_NOISE_DILATE_ITER = 1
    
    # Additional map-processing defaults
    BLUR_KERNEL_FINAL = 5
    BLUR_KERNEL_MULTIPLIER_STRENGTH = 1.5
    MEDIAN_FILTER_REDUCTION = 1
    BORDER_THICKNESS = 50
    DEFAULT_NUM_PROCESSES = 8


# =============================================================================
# COLORS
# =============================================================================

class Colors(Enum):
    """Color palette for UI and game elements."""
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    EUCALYPTUS = (95, 133, 117)  # Primary accent color
    GREENDARK = (117, 132, 104)
    YELLOW = (255, 255, 51)
    RED = (255, 0, 0)
    GREEN = (51, 255, 51)
    GREY = (100, 100, 100)
    BLUE = (0, 0, 153)

class DroneColors(Enum):
    """Colors for different drones."""
    RED = (255, 0, 0)    # Blinky
    PINK = (255, 184, 255)  # Pinky
    L_BLUE = (0, 255, 255)  # Inky
    ORANGE = (255, 184, 82)  # Clyde
    PURPLE = (148, 0, 221)  # Sue
    BROWN = (160, 82, 45)   # Tim
    GREEN = (34, 139, 34)   # Funky
    GOLD = (255, 215, 0)    # Kinky

class RoverColors(Enum):
    """Colors for rovers."""
    RED = (220, 0, 0)    # Huey
    BLUE = (0, 0, 255)   # Dewey
    GREEN = (0, 128, 0)  # Louie


# =============================================================================
# FONTS
# =============================================================================

class Fonts(Enum):
    """Font resources."""
    SMALL = GAME_DIR / 'Assets' / 'Fonts' / 'Cave-Stone.ttf'
    BIG = GAME_DIR / 'Assets' / 'Fonts' / 'CroMagnum.ttf'


# =============================================================================
# AUDIO
# =============================================================================

class Audio(Enum):
    """Audio resources."""
    AMBIENT = GAME_DIR / 'Assets' / 'Audio' / 'Menu.wav'
    BUTTON = GAME_DIR / 'Assets' / 'Audio' / 'Button.wav'


# =============================================================================
# IMAGES
# =============================================================================

class Images(Enum):
    """Image resources."""
    CAVE = GAME_DIR / 'Assets' / 'Images' / 'cave.jpg'
    DARK_CAVE = GAME_DIR / 'Assets' / 'Images' / 'cave_black.jpg'
    GAME_ICON = GAME_DIR / 'Assets' / 'Images' / 'drone.png'
    GAME_ICON_BG = GAME_DIR / 'Assets' / 'Images' / 'drone_BG.jpg'
    ROVER = GAME_DIR / 'Assets' / 'Images' / 'rover_top.png'
    DRONE = GAME_DIR / 'Assets' / 'Images' / 'drone_top.png'
    
    CAVE_MAP = GAME_DIR / 'Assets' / 'Map' / 'map.png'
    CAVE_MATRIX = GAME_DIR / 'Assets' / 'Map' / 'map_matrix.txt'
    CAVE_WALLS = GAME_DIR / 'Assets' / 'Map' / 'walls.png'
    CAVE_FLOOR = GAME_DIR / 'Assets' / 'Map' / 'floor.png'


# =============================================================================
# RECT HANDLE AND BRUSH
# =============================================================================

class RectHandle(Enum):
    """Pygame rect anchor points."""
    CENTER = 'center'
    MIDTOP = 'midtop'
    MIDRIGHT = 'midright'
    MIDLEFT = 'midleft'

class Brush(Enum):
    """Brush types for drawing."""
    ROUND = 0
    ELLIPSE = 1
    CHAOTIC = 2
    DIAMOND = 3
    OCTAGON = 4
    RECTANGULAR = 5


# =============================================================================
# MAP GENERATOR INPUTS
# =============================================================================

class WormInputs(Enum):
    """Map generation parameters for different cave sizes."""
    SMALL = [MapGen.STEP * 4, MapGen.STRENGTH * 4, MapGen.LIFE]
    MEDIUM = [MapGen.STEP * 2, MapGen.STRENGTH * 2, MapGen.LIFE * 4]
    BIG = [MapGen.STEP, MapGen.STRENGTH, MapGen.LIFE * 15]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def next_cell_coords(x: int, y: int, step_len: int, dir: int) -> Tuple[int, int]:
    """Calculate next cell coordinates based on position, step, and direction."""
    import math
    rad = math.radians(dir)
    dx = round(step_len * math.sin(rad))
    dy = -round(step_len * math.cos(rad))
    return x + dx, y + dy

def wall_hit(map_matrix: list, pos: Tuple[int, int]) -> bool:
    """Return True if position corresponds to a wall in map_matrix."""
    return map_matrix[pos[1]][pos[0]] == 1

def check_pixel_color(surface, pixel: Tuple[int, int], color: Tuple[int, int, int], is_not: bool = False) -> bool:
    """Check if pixel color matches (or doesn't match) the given color."""
    pixel_color = surface.get_at(pixel)[:3]
    return pixel_color != color if is_not else pixel_color == color
