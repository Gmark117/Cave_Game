import os
import configparser
from enum import Enum
from typing import List, Optional, Callable, Any, Tuple
import pygame
import pygame.mixer as mix
import Assets

class MenuItemType(Enum):
    TITLE = "title"
    BUTTON = "button"
    SELECTOR = "selector"
    TEXT_INPUT = "text_input"
    SLIDER = "slider"

class MenuItem:
    """
    Represents a single item in a menu, such as a button, selector, or text input.
    
    Attributes:
        game: Reference to the game instance.
        label: Display text for the item.
        position: (x, y) coordinates for rendering.
        item_type: Type of menu item.
        action: Callable to execute on selection.
        value: Current value (e.g., for selectors or sliders).
        options: List of options for selectors/sliders.
        size: Font size.
        font_big: Whether to use big font.
        alignment: Text alignment.
        is_selected: Whether the item is currently selected.
        text_input: Input text for TEXT_INPUT items.
        selectable: Whether the item can be selected.
    """
    
    def __init__(self, game: Any, label: str, position: Tuple[int, int], item_type: MenuItemType, 
                 action: Optional[Callable] = None, value: Optional[Any] = None, 
                 options: Optional[List[Any]] = None, size: int = 25, font_big: bool = False, 
                 alignment: str = 'midleft', selectable: Optional[bool] = None, 
                 text_input: Optional[str] = None):
        self.game = game
        self.label = label
        self.position = position
        self.item_type = item_type
        self.action = action
        self.value = value
        self.options = options or []
        self.size = size
        self.font_big = font_big
        self.alignment = alignment
        self.is_selected = False
        self.text_input = text_input if text_input is not None else ("" if item_type == MenuItemType.TEXT_INPUT else None)
        # Titles are not selectable by default, other items are
        self.selectable = selectable if selectable is not None else (item_type != MenuItemType.TITLE)

    def draw(self) -> None:
        """Render the menu item on the screen."""
        x, y = self.position
        # Titles are green, selected items red, others white
        color = Assets.Colors.EUCALYPTUS.value if self.item_type == MenuItemType.TITLE else (Assets.Colors.RED.value if self.is_selected else Assets.Colors.WHITE.value)
        # Offset for displaying values to the right of labels
        value_offset = 350

        # Draw label
        self._draw_text(self.label, x, y, self.size, Assets.Fonts.BIG.value if self.font_big else Assets.Fonts.SMALL.value, color, self.alignment)

        # Draw value based on type
        if self.item_type == MenuItemType.SELECTOR and self.options:
            value_text = str(self.options[self.value]) if isinstance(self.value, int) else str(self.value)
            self._draw_text(value_text, x + value_offset, y, self.size, Assets.Fonts.SMALL.value, Assets.Colors.GREENDARK.value, self.alignment)
        elif self.item_type == MenuItemType.TEXT_INPUT:
            # Blinking text effect on empty input
            if not self.text_input:
                display_text = "Enter value" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
                color = Assets.Colors.RED.value
            else:
                display_text = self.text_input
                color = Assets.Colors.GREENDARK.value
            self._draw_text(display_text, x + value_offset, y, self.size, Assets.Fonts.SMALL.value, color, self.alignment)
        elif self.item_type == MenuItemType.SLIDER:
            # Draw 5 bars representing the slider value
            slider_x = x + value_offset
            max_width = 200
            num_bars = 5
            bar_width = max_width / num_bars
            min_val, max_val, step = self.options
            # Calculate how many bars should be filled based on current value
            filled_bars = round(self.value / (max_val / num_bars))
            for i in range(num_bars):
                color = Assets.Colors.GREEN.value if i < filled_bars else Assets.Colors.WHITE.value
                pygame.draw.rect(self.game.display, color, (slider_x + i * bar_width, y - 8, bar_width - 2, 20))
        elif self.item_type == MenuItemType.TITLE or self.item_type == MenuItemType.BUTTON:
            # Buttons and Titles just show their label, already drawn above
            pass

    def _draw_text(self, text: str, x: int, y: int, size: int, font: str, color: Tuple[int, int, int], align: str) -> None:
        """Draw text on the screen with specified parameters."""
        style = pygame.font.Font(font, size)
        text_surface = style.render(text, True, color)
        rect = text_surface.get_rect()

        # Position text based on alignment: midright, midleft, or center
        if align == 'midright':
            rect.midright = (x, y)
        elif align == 'midleft':
            rect.midleft = (x, y)
        elif align == 'center':
            rect.center = (x, y)

        self.game.display.blit(text_surface, rect)

    def handle_input(self, game: Any, demo_cave: bool = True) -> bool:
        """Handle input for the menu item. Returns True if value changed."""
        if self.item_type == MenuItemType.SELECTOR:
            # Wrap around options using modulo
            if game.LEFT_KEY:
                self.value = (self.value - 1) % len(self.options)
                return True
            elif game.RIGHT_KEY:
                self.value = (self.value + 1) % len(self.options)
                return True
        elif self.item_type == MenuItemType.SLIDER:
            min_val, max_val, step = self.options
            if game.LEFT_KEY:
                self.value = max(min_val, self.value - step)
                self.game.menu._play_button()
                return True
            elif game.RIGHT_KEY:
                self.value = min(max_val, self.value + step)
                self.game.menu._play_button()
                return True
        elif self.item_type == MenuItemType.TEXT_INPUT and self.is_selected:
            # Pass demo_cave parameter to control input acceptance
            return self._handle_text_input(demo_cave)
        return False

    def _handle_text_input(self, demo_cave: bool = True) -> bool:
        """Handle text input for TEXT_INPUT items. Returns True if text changed."""
        keys = pygame.key.get_pressed()
        modified = False

        # Only accept input when not using the demo cave
        if not demo_cave:
            # Handle number keys 0-9
            for key in range(pygame.K_0, pygame.K_9 + 1):
                if keys[key]:
                    self.text_input += chr(key)
                    modified = True

            # Handle backspace for deletion
            if keys[pygame.K_BACKSPACE] and self.text_input:
                self.text_input = self.text_input[:-1]
                modified = True

        return modified

class Menu:
    """
    Manages the game's menu system, including main menu, simulation settings, options, and credits.
    
    Attributes:
        game: Reference to the game instance.
        background: Background image for the menu.
        button: Sound effect for button presses.
        volume: Current volume level (0-100).
        sound_on_off: Music on/off state.
        button_on_off: Button sound on/off state.
        main: List of menu items for the main menu.
        simulation: List of menu items for simulation settings.
        options: List of menu items for audio options.
        credits: List of menu items for credits.
        current_menu: Currently displayed menu.
        current_index: Index of the currently selected item.
        show_menu: Whether the menu is being displayed.
        seed_input: Current seed input value.
    """
    
    def __init__(self, game: Any) -> None:
        self.game = game

        self.background = pygame.image.load(Assets.Images['CAVE'].value)
        #self.dark_background = pygame.image.load(Assets.Images['DARK_CAVE'].value)

        # Initialize pygame mixer for audio
        mix.init()
        mix.music.load(Assets.Audio['AMBIENT'].value)
        self.button = mix.Sound(Assets.Audio['BUTTON'].value)
        self.button.set_volume(0.5)
        self.load_options()
        # Start background music if enabled and not already playing
        if self.sound_on_off == 'on' and not mix.music.get_busy():
            mix.music.play(-1)

        self.create_main_menu()
        self.create_simulation_menu()
        self.create_options_menu()
        self.create_credits_menu()

        self.current_menu = self.main
        self.current_index = self._get_first_selectable()

    def create_main_menu(self) -> None:
        """Create the main menu items."""
        self.main = [
            MenuItem(self.game, "CAVE EXPLORER", (Assets.ALIGN_L, Assets.CENTER_H - 250), MenuItemType.TITLE, size=110, font_big=True),
            MenuItem(self.game, "Simulation Settings", (Assets.ALIGN_L, Assets.CENTER_H - 50), MenuItemType.BUTTON, action=lambda: setattr(self, 'current_menu', self.simulation)),
            MenuItem(self.game, "Audio Settings", (Assets.ALIGN_L, Assets.CENTER_H + 10), MenuItemType.BUTTON, action=lambda: setattr(self, 'current_menu', self.options)),
            MenuItem(self.game, "Credits", (Assets.ALIGN_L, Assets.CENTER_H + 70), MenuItemType.BUTTON, action=lambda: setattr(self, 'current_menu', self.credits)),
            MenuItem(self.game, "Exit", (Assets.ALIGN_L, Assets.CENTER_H + 130), MenuItemType.BUTTON, action=lambda: [setattr(self.game, 'running', False), setattr(self, 'show_menu', False)])
        ]

    def create_simulation_menu(self) -> None:
        """Create the simulation settings menu items."""
        self.simulation = [
            MenuItem(self.game, "SIMULATION SETTINGS", (Assets.ALIGN_L, Assets.CENTER_H - 170), MenuItemType.TITLE, size=50, font_big=True),
            MenuItem(self.game, "Objective", (Assets.ALIGN_L, Assets.CENTER_H - 90), MenuItemType.SELECTOR, value=0, options=Assets.mission_options),
            MenuItem(self.game, "Cave Size", (Assets.ALIGN_L, Assets.CENTER_H - 50), MenuItemType.SELECTOR, value=0, options=Assets.map_options),
            MenuItem(self.game, "Seed", (Assets.ALIGN_L, Assets.CENTER_H - 10), MenuItemType.TEXT_INPUT, text_input=""),
            MenuItem(self.game, "Drones", (Assets.ALIGN_L, Assets.CENTER_H + 30), MenuItemType.SELECTOR, value=0, options=[3,4,5,6]),
            MenuItem(self.game, "Demo Cave", (Assets.ALIGN_L, Assets.CENTER_H + 70), MenuItemType.SELECTOR, value=1, options=Assets.prefab_options),
            MenuItem(self.game, "Back", (Assets.ALIGN_L, Assets.CENTER_H + 120), MenuItemType.BUTTON, action=lambda: setattr(self, 'current_menu', self.main)),
            MenuItem(self.game, "Start Mission", (Assets.ALIGN_L, Assets.CENTER_H + 220), MenuItemType.BUTTON, action=self.start_mission, size=100, font_big=True)
        ]
        self.set_default_seed()

    def create_options_menu(self) -> None:
        """Create the options menu items."""
        self.options = [
            MenuItem(self.game, "AUDIO SETTINGS", (Assets.ALIGN_L, Assets.CENTER_H - 150), MenuItemType.TITLE, size=50, font_big=True),
            MenuItem(self.game, "Game Volume", (Assets.ALIGN_L, Assets.CENTER_H - 40), MenuItemType.SLIDER, value=self.volume, options=[0, 100, 20]),
            MenuItem(self.game, "Music", (Assets.ALIGN_L, Assets.CENTER_H), MenuItemType.SELECTOR, value=0 if self.sound_on_off == 'on' else 1, options=['on', 'off']),
            MenuItem(self.game, "Button", (Assets.ALIGN_L, Assets.CENTER_H + 40), MenuItemType.SELECTOR, value=0 if self.button_on_off == 'on' else 1, options=['on', 'off']),
            MenuItem(self.game, "Back", (Assets.ALIGN_L, Assets.CENTER_H + 120), MenuItemType.BUTTON, action=lambda: (self.save_options(), setattr(self, 'current_menu', self.main)))
        ]

    def create_credits_menu(self) -> None:
        """Create the credits menu items."""
        self.credits = [
            MenuItem(self.game, "CREDITS", (Assets.ALIGN_L, Assets.CENTER_H - 150), MenuItemType.TITLE, size=70, font_big=True),
            MenuItem(self.game, "Daniela Argeri - 219892", (Assets.ALIGN_L, Assets.CENTER_H - 60), MenuItemType.BUTTON, selectable=False),
            MenuItem(self.game, "Gianmarco Lavacca - 224558", (Assets.ALIGN_L, Assets.CENTER_H - 10), MenuItemType.BUTTON, selectable=False),
            MenuItem(self.game, "Stefania Zaninotto - 220952", (Assets.ALIGN_L, Assets.CENTER_H + 40), MenuItemType.BUTTON, selectable=False),
            MenuItem(self.game, "Back", (Assets.ALIGN_L, Assets.CENTER_H + 140), MenuItemType.BUTTON, action=lambda: setattr(self, 'current_menu', self.main))
        ]

    def display(self) -> None:
        """Display the menu and handle the menu loop."""
        self.show_menu = True
        while self.show_menu:
            self.game.check_events()
            self._handle_global_input()
            self._draw()
            self.game.blit_screen()
            pygame.time.wait(125)  # ~8 FPS

    def _draw(self) -> None:
        """Draw the menu background and all menu items."""
        # Background
        self.game.display.blit(self.background, (0, 0))

        # Menu items
        for i, item in enumerate(self.current_menu):
            item.is_selected = (i == self.current_index)
            item.draw()

    def _get_first_selectable(self) -> int:
        """Get the index of the first selectable menu item."""
        for i, item in enumerate(self.current_menu):
            if item.selectable:
                return i
        return 0

    def _get_next_selectable(self, direction: str) -> int:
        """Get the index of the next selectable menu item in the given direction."""
        step = 1 if direction == 'down' else -1
        index = self.current_index
        for _ in range(len(self.current_menu)):
            index = (index + step) % len(self.current_menu)
            if self.current_menu[index].selectable:
                return index
        return self.current_index  # if none, stay

    def _handle_global_input(self) -> None:
        """Handle global menu navigation and selection input."""
        # Navigate up/down through selectable items
        if self.game.UP_KEY:
            self.current_index = self._get_next_selectable('up')
        elif self.game.DOWN_KEY:
            self.current_index = self._get_next_selectable('down')

        # Execute action on selected item
        elif self.game.START_KEY:
            current_item = self.current_menu[self.current_index]
            if current_item.action:
                self._play_button()
                current_item.action()
                # Reset to first selectable after action (e.g., menu change)
                self.current_index = self._get_first_selectable()
            elif current_item.item_type == MenuItemType.BUTTON:
                # Default action for buttons without explicit action
                self._play_button()
                pass

        # Handle item-specific input (selectors, sliders, text input)
        else:
            current_item = self.current_menu[self.current_index]
            # Item was modified
            if current_item.handle_input(self.game, self.simulation[5].value):
                # Update dependent settings when demo cave selection changes
                if self.current_menu == self.simulation and self.simulation[5].value == 1 and (not self.simulation[3].text_input or int(self.simulation[3].text_input) != Assets.seed[self.simulation[2].value]):
                    self.set_default_seed()
                # Update options if in options menu
                elif self.current_menu == self.options:
                    self._update_options()

    def _play_button(self) -> None:
        """Play the button sound effect if enabled."""
        if self.button_on_off == 'on':
            self.button.play()

    def load_options(self) -> None:
        """Load audio options from the configuration file."""
        config_path = os.path.join(Assets.GAME_DIR, 'GameConfig', 'options.ini')
        # Set defaults if config file doesn't exist
        if not os.path.exists(config_path):
            self.volume = 100
            self.sound_on_off = 'on'
            self.button_on_off = 'on'
        config = configparser.ConfigParser()
        config.read(config_path)
        # Load with fallback defaults
        self.volume = config.getint('Options', 'volume', fallback=100)
        self.sound_on_off = config.get('Options', 'music', fallback='on')
        self.button_on_off = config.get('Options', 'button', fallback='on')
        
        # Apply loaded volume settings
        mix.music.set_volume(self.volume / 100)
        self.button.set_volume(self.volume / 100)

    def _update_options(self) -> None:
        """Update audio options based on menu selections."""
        if hasattr(self, 'options') and self.options and len(self.options) > 1:
            # Update volume from slider
            self.volume = self.options[1].value
            mix.music.set_volume(self.volume / 100)
            self.button.set_volume(self.volume / 100)
            
            # Handle music toggle (0=on, 1=off)
            music_val = self.options[2].value
            if music_val == 0 and self.sound_on_off == 'off':
                mix.music.play(-1)  # Loop indefinitely
                self.sound_on_off = 'on'
            elif music_val == 1 and self.sound_on_off == 'on':
                mix.music.stop()
                self.sound_on_off = 'off'
            
            # Handle button sound toggle
            button_val = self.options[3].value
            self.button_on_off = 'on' if button_val == 0 else 'off'

    def save_options(self) -> None:
        """Save audio options to the configuration file."""
        config_path = os.path.join(Assets.GAME_DIR, 'GameConfig', 'options.ini')
        config = configparser.ConfigParser()
        config['Options'] = {
            'volume': self.volume,
            'music': self.sound_on_off,
            'button': self.button_on_off
        }
        with open(config_path, 'w') as f:
            config.write(f)

    def set_default_seed(self) -> None:
        """Set the default seed value based on current map selection."""
        self.seed_input = str(Assets.seed[self.simulation[2].value])
        self.simulation[3].text_input = self.seed_input

    def save_symSettings(self) -> None:
        """Save simulation settings to the configuration file."""
        config_path = os.path.join(Assets.GAME_DIR, 'GameConfig', 'symSettings.ini')
        config = configparser.ConfigParser()
        config['symSettings'] = {
            'Mode': Assets.mission_options[self.simulation[1].value],  # Index into mission options
            'Map_dimension': Assets.map_options[self.simulation[2].value],  # Index into map options
            'Seed': self.simulation[3].text_input,
            'Drones': [3,4,5,6][self.simulation[4].value],  # Index into drone count options
            'Prefab': Assets.prefab_options[self.simulation[5].value]  # Index into prefab options
        }
        with open(config_path, 'w') as f:
            config.write(f)

    def start_mission(self) -> None:
        """Start the mission with current settings."""
        if len(self.simulation[3].text_input) != 0:
            self.save_symSettings()
            self.game.start_mission()
            self.show_menu = False