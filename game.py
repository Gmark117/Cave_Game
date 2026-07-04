"""High-level application object for menu and mission transitions.

``Game`` owns the Pygame window, keyboard flags, and the current menu/mission
objects. The cleanup renamed this module from ``Game.py`` to ``game.py`` so it
follows normal Python lowercase module naming.
"""

import os
import logging
import pygame
from config.simulation_config import SimulationConfig
from asset_config.gameplay import Display
from asset_config.media import Images
from generation.map_generator import MapGenerator
from mission.control import MissionControl
from mission.objectives import MissionObjective, build_mission_objective
from ui.menu.facade import Menu


logger = logging.getLogger(__name__)


class Game:
    """Main game class handling initialization, menus, and simulation."""

    def __init__(self) -> None:
        """Initialize the game with pygame, window, and menus."""
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        try:
            pygame.init()
        except pygame.error as e:
            raise RuntimeError("Failed to initialize Pygame") from e

        self.running: bool = True
        # The menu reads these one-frame flags instead of handling raw Pygame
        # events directly. ``blit_screen`` resets them after each rendered frame.
        self.UP_KEY: bool = False
        self.DOWN_KEY: bool = False
        self.START_KEY: bool = False
        self.BACK_KEY: bool = False
        self.LEFT_KEY: bool = False
        self.RIGHT_KEY: bool = False

        self.sim_settings: SimulationConfig | None = None
        self.mission_objective: MissionObjective | None = None
        # These are created lazily once the user chooses "Start Mission".
        self.cartographer: MapGenerator | None = None
        self.mission_control: MissionControl | None = None

        self.to_windowed()
        self.menu = Menu(self)


    def run(self) -> None:
        """Main menu loop - displays current menu until game exits."""
        while self.running:
            self.menu.display()

    def start_mission(self) -> None:
        """Start the mission with current settings.

        Retrieves settings from simulation menu, generates the cave map,
        initializes mission control, runs the simulation (blocking until completion),
        then returns to the main menu.
        """
        self.sim_settings = self.menu.build_sim_settings()
        # Objective construction is explicit so planned mission types fail early
        # instead of silently running the Exploration rules.
        self.mission_objective = build_mission_objective(
            self.sim_settings.mission_config.objective
        )
        self.cartographer = MapGenerator(self, self.sim_settings)

        while True:
            self.mission_control = MissionControl(self)
            self.mission_control.run()
            # Restart creates a fresh MissionControl instance, which prevents
            # stale threads, events, or process-pool state from leaking forward.
            if self.mission_control.restart_requested is not True:
                break

    def check_events(self) -> None:
        """Check player inputs and update key flags."""
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    self.running = False
                    menu = getattr(self, "menu", None)
                    if menu is not None:
                        # Closing the window should break out of a nested menu
                        # loop as well as the outer ``Game.run`` loop.
                        menu.show_menu = False
                    return
                
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
        self.reset_keys()
    
    def _setup_window(self, width: int, height: int) -> pygame.Surface:
        """Set up the window with given dimensions."""
        self.width = width
        self.height = height
        self.display = pygame.Surface((self.width, self.height))
        try:
            self.window = pygame.display.set_mode((self.width, self.height), pygame.SCALED)
        except pygame.error as e:
            raise RuntimeError("Failed to set display mode") from e
        pygame.display.set_caption('Cave Game')
        try:
            pygame.display.set_icon(pygame.image.load(Images.GAME_ICON.value))
        except pygame.error as e:
            logger.warning("Failed to load game icon: %s", e)
        return self.display

    def to_maximised(self) -> pygame.Surface:
        """Maximize the game window to full screen."""
        return self._setup_window(Display.FULL_W, Display.FULL_H)

    def to_windowed(self) -> pygame.Surface:
        """Return to the original window dimensions."""
        return self._setup_window(Display.W, Display.H)

