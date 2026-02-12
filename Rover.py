import pygame
import random as rand
import time
import math
from Assets import next_cell_coords, check_pixel_color, Colors
from Graph import Graph


class Rover():
    def __init__(self, game, control, id, start_pos, color, icon, cave, strategy="random"):
        self.game     = game
        self.settings = game.sim_settings
        self.cave     = cave
        self.control  = control
        self.strategy = strategy
         
        self.id       = id # Unique identifier of the drone
        self.map_size = self.settings.map_dim # Map dimension
        self.radius   = self.calculate_radius() # Radius that represent the field of view # 39
        self.step     = 10 # Step of the drone
        self.dir      = rand.randint(0,359)

        self.color = color
        self.alpha = 150
        self.icon  = icon

        self.battery  = 2400
        self.statuses = ['Ready', 'Updating', 'Advancing', 'Done']
        
        # Transparent surface used to track the explored path
        self.floor_surf = pygame.Surface((self.game.width,self.game.height), pygame.SRCALPHA)
        self.floor_surf.fill((*Colors.WHITE.value, 0))
        self.ray_points = []  # Initialize the list for rays
        self.delay      = self.control.delay

        self.show_path    = True
        self.speed_factor = 4
         
        self.border    = []
        self.start_pos = start_pos
        self.pos       = start_pos
        self.dir_log   = []
        self.graph     = Graph(*start_pos, cave)

    # Define the radius based on the map size
    def calculate_radius(self):
        match self.map_size:
            case 'SMALL' : return 40
            case 'MEDIUM': return 20
            case 'BIG'   : return 10
            case _       : return 20


#  ____   ____      _  __        __ ___  _   _   ____ 
# |  _ \ |  _ \    / \ \ \      / /|_ _|| \ | | / ___|
# | | | || |_) |  / _ \ \ \ /\ / /  | | |  \| || |  _
# | |_| ||  _ <  / ___ \ \ V  V /   | | | |\  || |_| |
# |____/ |_| \_\/_/   \_\ \_/\_/   |___||_| \_| \____|

    # Draw the rover icon
    def draw_icon(self):
        icon_width, icon_height = self.icon.get_size()  # Get dimensions of the icon
        icon_position = (int(self.pos[0] - icon_width // 2), int(self.pos[1] - icon_height // 2))  # Center the icon

        # Blit the drone icon at the calculated position
        self.game.window.blit(self.icon, icon_position)