import os
import tkinter
import math
import pygame
from enum import Enum

root = tkinter.Tk()

#  ____   _____  _____  _____  ___  _   _   ____  ____  
# / ___| | ____||_   _||_   _||_ _|| \ | | / ___|/ ___|
# \___ \ |  _|    | |    | |   | | |  \| || |  _ \___ \
#  ___) || |___   | |    | |   | | | |\  || |_| | ___) |
# |____/ |_____|  |_|    |_|  |___||_| \_| \____||____/

# Game directory: ..\CaveGame
GAME_DIR = os.path.dirname(os.path.abspath(__file__))

# Display dimensions
DISPLAY_W    = 1200
DISPLAY_H    = 750
FULLSCREEN_W = 1915 #root.winfo_screenwidth() - 5
FULLSCREEN_H = 1010 #root.winfo_screenheight() - 70
LEGEND_WIDTH = 300

# Display positions
CENTER_W    = DISPLAY_W/2
CENTER_H    = DISPLAY_H/2
ALIGN_L     = CENTER_W - 550

# Lists for menu voices and settings
main_menu_states    = ['Start', 'Options', 'Credits', 'Exit']
options_menu_states = ['Game Volume', 'Music Volume', 'Button Sound', 'Back']
sim_menu_states     = ['Mode', 'Map Dimension', 'Seed', 'Drones', 'Prefab', 'Back', 'Start Simulation']
mission_options     = ["Exploration", "Search and Rescue"]
map_options         = ["Small", "Medium", "Big"]
prefab_options      = ['No', 'Yes']
vision_options      = [     39,       19,     4]
drone_icon_options  = [(30,30),  (10,10), (1,1)]
rover_icon_options  = [(40,40),  (15,15), (5,5)]
seed                = [      5,       19,   837]

# Map Generator Inputs
step     = 10
strength = 16
life     = 75

#   ____  _         _     ____   ____   _____  ____  
#  / ___|| |       / \   / ___| / ___| | ____|/ ___| 
# | |    | |      / _ \  \___ \ \___ \ |  _|  \___ \ 
# | |___ | |___  / ___ \  ___) | ___) || |___  ___) |
#  \____||_____|/_/   \_\|____/ |____/ |_____||____/ 

class WormInputs(Enum):
        SMALL        = [4*step, 4*strength,    life]
        MEDIUM       = [2*step, 2*strength,  4*life]
        BIG          = [  step,   strength, 15*life]

class Colors(Enum):
        BLACK        = (  0,   0,   0)
        WHITE        = (255, 255, 255)
        EUCALYPTUS   = ( 95, 133, 117)
        GREENDARK    = (117, 132, 104)
        YELLOW       = (255, 255,  51)
        RED          = (255,   0,   0)
        GREEN        = ( 51, 255,  51)
        GREY         = (112, 128, 144)
        BLUE         = (  0,   0, 153)

class DroneColors(Enum):
        RED          = (255,   0,   0)   # Blinky
        PINK         = (255, 184, 255)   # Pinky
        L_BLUE       = (  0, 255, 255)   # Inky
        ORANGE       = (255, 184,  82)   # Clyde
        PURPLE       = (148,   0, 221)   # Sue
        BROWN        = (160,  82,  45)   # Tim
        GREEN        = ( 34, 139,  34)   # Funky
        GOLD         = (255, 215,   0)   # Kinky

class RoverColors(Enum):
        RED          = (220,   0,   0)   # Huey
        BLUE         = (  0,   0, 255)   # Dewey
        GREEN        = (  0, 128,   0)   # Louie
        
class Fonts(Enum):
        BIG          = os.path.join(GAME_DIR, 'Assets', 'Fonts', 'Cave-Stone.ttf')  
        SMALL        = os.path.join(GAME_DIR, 'Assets', 'Fonts', '8-BIT.TTF') 

class Audio(Enum):
        AMBIENT      = os.path.join(GAME_DIR, 'Assets', 'Audio', 'Menu.wav')
        BUTTON       = os.path.join(GAME_DIR, 'Assets', 'Audio', 'Button.wav')

class Images(Enum):
        CAVE         = os.path.join(GAME_DIR, 'Assets', 'Images', 'cave.jpg')
        DARK_CAVE    = os.path.join(GAME_DIR, 'Assets', 'Images', 'cave_black.jpg')
        GAME_ICON    = os.path.join(GAME_DIR, 'Assets', 'Images', 'drone.png')
        GAME_ICON_BG = os.path.join(GAME_DIR, 'Assets', 'Images', 'drone_BG.jpg')
        ROVER        = os.path.join(GAME_DIR, 'Assets', 'Images', 'rover_top.png')
        DRONE        = os.path.join(GAME_DIR, 'Assets', 'Images', 'drone_top.png')
        
        CAVE_MAP     = os.path.join(GAME_DIR, 'Assets',    'Map', 'map.png')
        CAVE_MATRIX  = os.path.join(GAME_DIR, 'Assets',    'Map', 'map_matrix.txt')
        CAVE_WALLS   = os.path.join(GAME_DIR, 'Assets',    'Map', 'walls.png')
        CAVE_FLOOR   = os.path.join(GAME_DIR, 'Assets',    'Map', 'floor.png')

class RectHandle(Enum):
        CENTER       = 'center'
        MIDTOP       = 'midtop'
        MIDRIGHT     = 'midright'
        MIDLEFT      = 'midleft'

class Brush(Enum):
        ROUND        = 0
        ELLIPSE      = 1
        CHAOTIC      = 2
        DIAMOND      = 3
        OCTAGON      = 4
        RECTANGULAR  = 5

class Axes():
        def __init__(self, step_len):
                self.up      = 0
                self.diag_q1 = step_len
                self.right   = 2*step_len
                self.diag_q4 = 3*step_len
                self.down    = 4*step_len
                self.diag_q3 = 5*step_len
                self.left    = 6*step_len
                self.diag_q2 = 7*step_len

                self.list  = [self.up, self.diag_q1, self.right, self.diag_q4,
                              self.down, self.diag_q3, self.left, self.diag_q2]


#  _____  _   _  _   _   ____  _____  ___   ___   _   _  ____  
# |  ___|| | | || \ | | / ___||_   _||_ _| / _ \ | \ | |/ ___|
# | |_   | | | ||  \| || |      | |   | | | | | ||  \| |\___ \
# |  _|  | |_| || |\  || |___   | |   | | | |_| || |\  | ___) |
# |_|     \___/ |_| \_| \____|  |_|  |___| \___/ |_| \_||____/

# Calculate the square of the passed argument
def sqr(x):
        return x**2

# Map the given direction to the possible pixels,
# given the length of the step
def map_direction(step_len, dir):
        # Number of possible cells for a given step length
        targets = step_len * 8
        # The circle is divided into N sectors based on the number of targets
        sector_len = 360 / targets
        # The sectors are shifted backwards to align with the positions of the cells
        sector_offset = math.floor(sector_len / 2)
        # Sectors must be aligned with pixels positions and shifted back
        # Therefore the second half of a sector ends up in the next one
        corrected_dir = dir + sector_offset
        # Sector numbering starts at 0
        target_cell = math.floor((corrected_dir % 360)/ sector_len)
        return target_cell, targets

# Calculates the coordinates of the next pixel/cell based on the current position
# the step length, and the direction
def next_cell_coords(x, y, step_len, dir):
        # Ensure the step length is positive
        assert step_len>0
        # Map the step length and direction to the target cell 
        target_cell, targets = map_direction(step_len, dir)
        # Create an Axes object to manage the various directions and diagonals
        axes = Axes(step_len)
        # Match the 'target_cell' value to check if the movement 
        # is along an axis or a diagonal
        match target_cell:
                case axes.up:
                        y -= step_len
                        return x, y
                case axes.diag_q1:
                        x += step_len
                        y -= step_len
                        return x, y
                case axes.right:
                        x += step_len
                        return x, y
                case axes.diag_q4:
                        x += step_len
                        y += step_len
                        return x, y
                case axes.down:
                        y += step_len
                        return x, y
                case axes.diag_q3:
                        x -= step_len
                        y += step_len
                        return x, y
                case axes.left:
                        x -= step_len
                        return x, y
                case axes.diag_q2:
                        x -= step_len
                        y -= step_len
                        return x, y

        # If the direction doesn't fall directly on an axis or diagonal, check intermediate pixel values
        # Loop through the values of 'axes.list' to find the range between axis/diagonal values
        for i in axes.list:
                if i==0:
                        # Check the range between the last item in the list and the current target
                        check = range(axes.list[-1] + 1, targets)
                else:
                        # Check the range between the previous item and the current axis/diagonal value
                        check = range(axes.list[axes.list.index(i)-1]+1, i)
                        
                # Loop through the 'check' range to find where 'target_cell' lies between axes or diagonals
                for j in check:
                        if target_cell==j:
                                # Match the current axis/diagonal and adjust the coordinates accordingly
                                match i:
                                        case axes.up:
                                                x -= targets - j
                                                y -= step_len
                                                return x, y
                                        case axes.diag_q1:
                                                x += j
                                                y -= step_len
                                                return x, y
                                        case axes.right:
                                                x += step_len
                                                y -= i - j
                                                return x, y
                                        case axes.diag_q4:
                                                x += step_len
                                                y += i - j
                                                return x, y
                                        case axes.down:
                                                x += i - j
                                                y += step_len
                                                return x, y
                                        case axes.diag_q3:
                                                x -= j - i + step_len
                                                y += step_len
                                                return x, y
                                        case axes.left:
                                                x -= step_len
                                                y += i - j
                                                return x, y
                                        case axes.diag_q2:
                                                x -= step_len
                                                y -= j - i + step_len
                                                return x, y

# Return true if the position (y,x) corresponds to a wall in the map_matrix
def wall_hit(map_matrix, pos):
        if map_matrix[pos[1]][pos[0]]==1: # 1 = black
                return True
        return False

# Check if the pixel color matches or not the 'color'
def check_pixel_color(surface, pixel, color, is_not=False):
        if is_not:
                return pygame.Surface.get_at(surface, pixel)[:3] != color
        else:
                return pygame.Surface.get_at(surface, pixel)[:3] == color 