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
    DRONE_ICON = [(30, 30), (10, 10), (1, 1)]
    ROVER_ICON = [(40, 40), (15, 15), (5, 5)]
    SEED_DEFAULTS = [5, 19, 837]  # Default seeds for map sizes

# =============================================================================
# MAP GENERATION
# =============================================================================

class MapGen:
    """Map generation parameters."""
    STEP = 10
    STRENGTH = 16
    LIFE = 75

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
    GREY = (112, 128, 144)
    BLUE = (0, 0, 153)

# =============================================================================
# DRONE AND ROVER COLORS
# =============================================================================

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
    BIG = GAME_DIR / 'Assets' / 'Fonts' / 'Cave-Stone.ttf'
    SMALL = GAME_DIR / 'Assets' / 'Fonts' / '8-BIT.TTF'

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

def sqr(x: float) -> float:
    """Calculate the square of the passed argument."""
    return x ** 2

def next_cell_coords(x: int, y: int, step_len: int, dir: int) -> Tuple[int, int]:
    """Calculate next cell coordinates based on position, step, and direction."""
    assert step_len > 0
    
    # Normalize direction to 0-360 range
    dir = dir % 360
    
    # Determine the octant (8 main directions)
    octant = round(dir / 45) % 8
    
    # Map octant to coordinate changes
    directions = [
        (0, -step_len),      # 0° - up
        (step_len, -step_len),  # 45° - up-right
        (step_len, 0),       # 90° - right
        (step_len, step_len),   # 135° - down-right
        (0, step_len),       # 180° - down
        (-step_len, step_len),  # 225° - down-left
        (-step_len, 0),      # 270° - left
        (-step_len, -step_len)  # 315° - up-left
    ]
    
    dx, dy = directions[octant]
    return x + dx, y + dy

def wall_hit(map_matrix: list, pos: Tuple[int, int]) -> bool:
    """Return True if position corresponds to a wall in map_matrix."""
    return map_matrix[pos[1]][pos[0]] == 1

def check_pixel_color(surface, pixel: Tuple[int, int], color: Tuple[int, int, int], is_not: bool = False) -> bool:
    """Check if pixel color matches (or doesn't match) the given color."""
    pixel_color = surface.get_at(pixel)[:3]
    return pixel_color != color if is_not else pixel_color == color

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Display
FULLSCREEN_W = Display.FULL_W
FULLSCREEN_H = Display.FULL_H
LEGEND_WIDTH = Display.LEGEND_WIDTH

# Game options
vision_options = GameOptions.VISION
drone_icon_options = GameOptions.DRONE_ICON
rover_icon_options = GameOptions.ROVER_ICON

# Map gen
step = MapGen.STEP
strength = MapGen.STRENGTH
life = MapGen.LIFE