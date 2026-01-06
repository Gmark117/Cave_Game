import os
import sys
import pygame
import Assets
from typing import NoReturn
from MapGenerator import MapGenerator
from MissionControl import MissionControl
from Menu import Menu

class Game:
    """Main game class handling initialization, menus, and simulation."""

    def __init__(self) -> None:
        """Initialize the game with pygame, window, and menus."""
        # Center the game window on the screen
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        
        # Initialize all Pygame modules
        try:
            pygame.init()
        except pygame.error as e:
            print(f"Failed to initialize Pygame: {e}")
            sys.exit(1)
        
        # Set game state variables: running
        self.running: bool = True
        
        # Initialize key flags to handle menu navigation
        self.UP_KEY: bool = False
        self.DOWN_KEY: bool = False
        self.START_KEY: bool = False
        self.BACK_KEY: bool = False
        self.LEFT_KEY: bool = False
        self.RIGHT_KEY: bool = False
        
        # Set the window to windowed mode
        self.to_windowed()

        # Initialize the menus
        self.menu = Menu(self)


    def run(self) -> NoReturn:
        """Main menu loop - displays current menu until game exits."""
        while self.running:
            self.menu.display()

    def start_mission(self) -> None:
        """Start the mission with current settings.

        Retrieves settings from simulation menu, generates the cave map,
        initializes mission control, runs the simulation (blocking until completion),
        then returns to the main menu.
        """
        match self.menu.simulation[2].value:
            case 0: map_dim = 'SMALL'
            case 1: map_dim = 'MEDIUM'
            case 2: map_dim = 'BIG'
        self.sim_settings = [
            self.menu.simulation[1].value,
            map_dim,
            int(self.menu.simulation[3].text_input),
            [3,4,5,6][self.menu.simulation[4].value],
            self.menu.simulation[5].value
        ]
        self.cartographer = MapGenerator(self)
        self.mission_control = MissionControl(self)
        # Simulation runs here (blocks until completion)

    def check_events(self) -> None:
        """Check player inputs and update key flags."""
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                case pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_RETURN:
                            self.START_KEY = True
                        case pygame.K_BACKSPACE:
                            self.BACK_KEY = True
                        case pygame.K_DOWN:
                            self.DOWN_KEY = True
                        case pygame.K_UP:
                            self.UP_KEY = True
                        case pygame.K_LEFT:
                            self.LEFT_KEY = True
                        case pygame.K_RIGHT:
                            self.RIGHT_KEY = True
                    
    def reset_keys(self) -> None:
        """Reset pushed key flags to prevent multiple triggers."""
        self.UP_KEY = False
        self.DOWN_KEY = False
        self.START_KEY = False
        self.BACK_KEY = False
        self.LEFT_KEY = False
        self.RIGHT_KEY = False


    def blit_screen(self) -> None:
        """Update the display by blitting the current surface to the window."""
        self.window.blit(self.display, (0, 0))
        pygame.display.update()
        self.reset_keys()  # Reset key flags for the next frame
    
    def _setup_window(self, width: int, height: int) -> pygame.Surface:
        """Set up the window with given dimensions."""
        self.width = width
        self.height = height
        self.display = pygame.Surface((self.width, self.height))
        try:
            self.window = pygame.display.set_mode((self.width, self.height), pygame.SCALED)
        except pygame.error as e:
            print(f"Failed to set display mode: {e}")
            sys.exit(1)
        pygame.display.set_caption('Cave Game')
        try:
            pygame.display.set_icon(pygame.image.load(Assets.Images['GAME_ICON'].value))
        except pygame.error as e:
            print(f"Failed to load game icon: {e}")
        return self.display

    def to_maximised(self) -> pygame.Surface:
        """Maximize the game window to full screen."""
        return self._setup_window(Assets.Display.FULL_W, Assets.Display.FULL_H)

    def to_windowed(self) -> pygame.Surface:
        """Return to the original window dimensions."""
        return self._setup_window(Assets.Display.W, Assets.Display.H)
