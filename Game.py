import pygame
import Assets
from MainMenu import MainMenu
from OptionsMenu import OptionsMenu
from SimulationMenu import SimulationMenu
from CreditsMenu import CreditsMenu
from MapGenerator import MapGenerator
from Drone import Drone

class Game():
    def __init__(self): 
        # Initialise pygame features
        pygame.init()
        
        # If we run the game we are not necessary playing
        self.running, self.playing = True, False
        
        # Initialise key flags to navigate in the menu
        self.UP_KEY,   self.DOWN_KEY, self.START_KEY = False, False, False
        self.BACK_KEY, self.LEFT_KEY, self.RIGHT_KEY = False, False, False
        
        # Choose and set window dimensions
        self.width = Assets.DISPLAY_W
        self.height = Assets.DISPLAY_H
        
        # Initialise game window
        self.display = pygame.Surface((self.width,self.height))
        self.window  = pygame.display.set_mode((self.width,self.height), pygame.RESIZABLE)

        # Set window title
        pygame.display.set_caption('Cave Game')

        # Set game icon
        pygame.display.set_icon(pygame.image.load(Assets.Backgrounds['DRONE'].value))
        # pygame.display.set_icon(pygame.image.load(Assets.Backgrounds['DRONE_BG'].value))

        # Initialise each menu and set the current one
        self.options         = OptionsMenu(self)
        self.main_menu       = MainMenu(self)
        self.credits         = CreditsMenu(self)
        self.simulation      = SimulationMenu(self)
        self.curr_menu       = self.main_menu

    # Game loop function
    def game_loop(self):
        if self.playing:
            self.run_map_generator()
            self.drones = Drone(self, 3)
    
    # Create the cave
    def run_map_generator(self):
        settings  = self.simulation.get_sim_settings()
        self.cave = MapGenerator(self, settings)

   # Check player inputs
    def check_events(self):
        # Get the input
        for event in pygame.event.get():
            match event.type:
                # If the player clicks the x on top of the window exit the game
                case pygame.QUIT:
                    self.running, self.playing = False, False
                    self.curr_menu.run_display = False
                
                # If the player clicks something on the keyboard
                # they can go up or down with the arrows or
                # they can select with ENTER and go back with BACKSPACE
                case pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_RETURN:
                            self.START_KEY = True
                        case pygame.K_BACKSPACE:
                            self.BACK_KEY  = True
                        case pygame.K_DOWN:
                            self.DOWN_KEY  = True
                        case pygame.K_UP:
                            self.UP_KEY    = True
                        case pygame.K_LEFT:
                            self.LEFT_KEY  = True
                        case pygame.K_RIGHT:
                            self.RIGHT_KEY = True
                    
    # Reset pushed key flags
    def reset_keys(self):
        self.UP_KEY,   self.DOWN_KEY, self.START_KEY = False, False, False
        self.BACK_KEY, self.LEFT_KEY, self.RIGHT_KEY = False, False, False
    
    # Update the display
    def blit_screen(self):
        self.window.blit(self.display, (0, 0))
        pygame.display.update()
        self.reset_keys()
