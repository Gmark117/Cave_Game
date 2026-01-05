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
    ALIGN_CL = CENTER_W - 50
    ALIGN_CR = CENTER_W + 50

# =============================================================================
# MENU AND GAME STATES
# =============================================================================

class MenuStates:
    """Menu navigation states."""
    MAIN = ['Start', 'Options', 'Credits', 'Exit']
    OPTIONS = ['Game Volume', 'Music Volume', 'Button Sound', 'Back']
    SIMULATION = ['Mode', 'Map Dimension', 'Seed', 'Drones', 'Prefab', 'Back', 'Start Simulation']

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
# AXES CLASS
# =============================================================================

class Axes:
    """Manages directional axes for movement calculations."""
    def __init__(self, step_len: int):
        self.up = 0
        self.diag_q1 = step_len
        self.right = 2 * step_len
        self.diag_q4 = 3 * step_len
        self.down = 4 * step_len
        self.diag_q3 = 5 * step_len
        self.left = 6 * step_len
        self.diag_q2 = 7 * step_len

        self.list = [self.up, self.diag_q1, self.right, self.diag_q4,
                     self.down, self.diag_q3, self.left, self.diag_q2]

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

def map_direction(step_len: int, dir: int) -> Tuple[int, int]:
    """Map the given direction to possible pixels for a given step length."""
    import math
    targets = step_len * 8
    sector_len = 360 / targets
    sector_offset = math.floor(sector_len / 2)
    corrected_dir = dir + sector_offset
    target_cell = math.floor((corrected_dir % 360) / sector_len)
    return target_cell, targets

def next_cell_coords(x: int, y: int, step_len: int, dir: int) -> Tuple[int, int]:
    """Calculate next cell coordinates based on position, step, and direction."""
    import math
    assert step_len > 0
    target_cell, targets = map_direction(step_len, dir)
    axes = Axes(step_len)
    
    # Direct matches
    if target_cell == axes.up:
        return x, y - step_len
    elif target_cell == axes.diag_q1:
        return x + step_len, y - step_len
    elif target_cell == axes.right:
        return x + step_len, y
    elif target_cell == axes.diag_q4:
        return x + step_len, y + step_len
    elif target_cell == axes.down:
        return x, y + step_len
    elif target_cell == axes.diag_q3:
        return x - step_len, y + step_len
    elif target_cell == axes.left:
        return x - step_len, y
    elif target_cell == axes.diag_q2:
        return x - step_len, y - step_len
    
    # Intermediate positions (simplified logic)
    for i in axes.list:
        check_range = range(axes.list[axes.list.index(i) - 1] + 1, i) if i != 0 else range(axes.list[-1] + 1, targets)
        for j in check_range:
            if target_cell == j:
                # Calculate offsets (original logic preserved but could be optimized)
                if i == axes.up:
                    return x - (targets - j), y - step_len
                elif i == axes.diag_q1:
                    return x + j, y - step_len
                elif i == axes.right:
                    return x + step_len, y - (i - j)
                elif i == axes.diag_q4:
                    return x + step_len, y + (i - j)
                elif i == axes.down:
                    return x + (i - j), y + step_len
                elif i == axes.diag_q3:
                    return x - (j - i + step_len), y + step_len
                elif i == axes.left:
                    return x - step_len, y + (i - j)
                elif i == axes.diag_q2:
                    return x - step_len, y - (j - i + step_len)
    return x, y  # Fallback

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
DISPLAY_W = Display.W
DISPLAY_H = Display.H
FULLSCREEN_W = Display.FULL_W
FULLSCREEN_H = Display.FULL_H
CENTER_W = Display.CENTER_W
CENTER_H = Display.CENTER_H
ALIGN_L = Display.ALIGN_L
LEGEND_WIDTH = Display.LEGEND_WIDTH

# Menu states
main_menu_states = MenuStates.MAIN
options_menu_states = MenuStates.OPTIONS
sim_menu_states = MenuStates.SIMULATION

# Game options
mission_options = GameOptions.MISSION
map_options = GameOptions.MAP_SIZE
prefab_options = GameOptions.PREFAB
vision_options = GameOptions.VISION
drone_icon_options = GameOptions.DRONE_ICON
rover_icon_options = GameOptions.ROVER_ICON
seed = GameOptions.SEED_DEFAULTS

# Map gen
step = MapGen.STEP
strength = MapGen.STRENGTH
life = MapGen.LIFE