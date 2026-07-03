"""Menu façade for Cave Explorer.

The façade preserves the API used by ``Game`` while delegating menu state,
rendering, persistence, and audio to focused collaborators.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Sequence, cast

import pygame

from MenuAudioService import MenuAudioService
from MenuController import MenuController
from MenuModels import (
    ButtonItem,
    MenuAction,
    MenuRow,
    MenuScreen,
    SelectorItem,
    SliderItem,
    TextInputItem,
    TitleItem,
)
from MenuRenderer import MenuRenderer
from MenuSettingsRepository import (
    AudioSettings,
    MenuSettingsRepository,
)
from SimulationConfig import SimulationConfig
from asset_config.gameplay import Display, GameOptions


GAME_DIR = Path(__file__).parent
DRONE_OPTIONS = (3, 4, 5, 6, 7, 8)


class Menu:
    """Coordinate the menu controller, renderer, settings, and audio service."""

    def __init__(self, game: object) -> None:
        self.game = game
        self.config = SimulationConfig()

        self.settings_repository = MenuSettingsRepository(GAME_DIR)
        audio_settings = self.settings_repository.load_audio()
        self.volume = audio_settings.volume
        self.sound_on_off = audio_settings.music
        self.button_on_off = audio_settings.button

        self.renderer = MenuRenderer(game)
        self.audio = MenuAudioService(audio_settings)

        self.create_main_menu()
        self.create_simulation_menu()
        self.load_simulation_settings()
        self.create_options_menu()
        self.create_credits_menu()

        screens = {
            MenuScreen.MAIN: self.main,
            MenuScreen.SIMULATION: self.simulation,
            MenuScreen.AUDIO: self.options,
            MenuScreen.CREDITS: self.credits,
        }
        self.controller = MenuController(
            screens,
            self._handle_action,
            self._play_button,
            self._update_options,
        )
        self.show_menu = False

    @property
    def current_menu(self) -> List[MenuRow]:
        return self.controller.current_items

    @property
    def current_index(self) -> int:
        return self.controller.current_index

    @current_index.setter
    def current_index(self, value: int) -> None:
        self.controller.current_index = value

    def create_main_menu(self) -> None:
        """Create typed definitions for the main menu."""
        self.main: List[MenuRow] = [
            TitleItem(
                "CAVE EXPLORER",
                (Display.ALIGN_L, Display.CENTER_H - 250),
                size=110,
                font_big=True,
            ),
            ButtonItem(
                "Simulation Settings",
                (Display.ALIGN_L, Display.CENTER_H - 50),
                action=MenuAction.OPEN_SIMULATION,
            ),
            ButtonItem(
                "Audio Settings",
                (Display.ALIGN_L, Display.CENTER_H + 10),
                action=MenuAction.OPEN_AUDIO,
            ),
            ButtonItem(
                "Credits",
                (Display.ALIGN_L, Display.CENTER_H + 70),
                action=MenuAction.OPEN_CREDITS,
            ),
            ButtonItem(
                "Exit",
                (Display.ALIGN_L, Display.CENTER_H + 130),
                action=MenuAction.EXIT,
            ),
        ]

    def create_simulation_menu(self) -> None:
        """Create typed definitions for simulation settings."""
        self.simulation: List[MenuRow] = [
            TitleItem(
                "SIMULATION SETTINGS",
                (Display.ALIGN_L, Display.CENTER_H - 170),
                size=50,
                font_big=True,
            ),
            SelectorItem(
                "Objective",
                (Display.ALIGN_L, Display.CENTER_H - 90),
                options=GameOptions.MISSION,
            ),
            SelectorItem(
                "Cave Size",
                (Display.ALIGN_L, Display.CENTER_H - 50),
                options=GameOptions.MAP_SIZE,
            ),
            TextInputItem(
                "Seed",
                (Display.ALIGN_L, Display.CENTER_H - 10),
            ),
            SelectorItem(
                "Drones",
                (Display.ALIGN_L, Display.CENTER_H + 30),
                options=DRONE_OPTIONS,
            ),
            ButtonItem(
                "Back",
                (Display.ALIGN_L, Display.CENTER_H + 120),
                action=MenuAction.BACK_TO_MAIN,
            ),
            ButtonItem(
                "Start Mission",
                (Display.ALIGN_L, Display.CENTER_H + 220),
                size=100,
                font_big=True,
                action=MenuAction.START_MISSION,
            ),
        ]
        self.set_default_seed()

    def create_options_menu(self) -> None:
        """Create typed definitions for audio settings."""
        self.options: List[MenuRow] = [
            TitleItem(
                "AUDIO SETTINGS",
                (Display.ALIGN_L, Display.CENTER_H - 150),
                size=50,
                font_big=True,
            ),
            SliderItem(
                "Game Volume",
                (Display.ALIGN_L, Display.CENTER_H - 40),
                value=self.volume,
                minimum=0,
                maximum=100,
                step=20,
            ),
            SelectorItem(
                "Music",
                (Display.ALIGN_L, Display.CENTER_H),
                value=0 if self.sound_on_off == "on" else 1,
                options=("on", "off"),
            ),
            SelectorItem(
                "Button",
                (Display.ALIGN_L, Display.CENTER_H + 40),
                value=0 if self.button_on_off == "on" else 1,
                options=("on", "off"),
            ),
            ButtonItem(
                "Back",
                (Display.ALIGN_L, Display.CENTER_H + 120),
                action=MenuAction.SAVE_AUDIO_AND_BACK,
            ),
        ]

    def create_credits_menu(self) -> None:
        """Create typed definitions for credits."""
        self.credits: List[MenuRow] = [
            TitleItem(
                "CREDITS",
                (Display.ALIGN_L, Display.CENTER_H - 150),
                size=70,
                font_big=True,
            ),
            ButtonItem(
                "Daniela Argeri ~~~ 219892",
                (Display.ALIGN_L, Display.CENTER_H - 60),
                selectable=False,
            ),
            ButtonItem(
                "Gianmarco Lavacca ~~~ 224558",
                (Display.ALIGN_L, Display.CENTER_H - 10),
                selectable=False,
            ),
            ButtonItem(
                "Back",
                (Display.ALIGN_L, Display.CENTER_H + 90),
                action=MenuAction.BACK_TO_MAIN,
            ),
        ]

    def display(self) -> None:
        """Run the menu loop until an action closes it."""
        self.show_menu = True
        while self.show_menu:
            self.game.check_events()
            self._handle_global_input()
            self._draw()
            self.game.blit_screen()

            if self._number_key_pressed():
                while self._number_key_pressed():
                    self.game.check_events()
                    pygame.time.wait(10)

    def _draw(self) -> None:
        self.renderer.draw(self.current_menu, self.current_index)

    def _handle_global_input(self) -> None:
        self.controller.handle_input(self.game)

    def _number_key_pressed(self) -> bool:
        return self.controller.number_key_pressed()

    def _get_first_selectable(self) -> int:
        return self.controller.first_selectable()

    def _get_next_selectable(self, direction: str) -> int:
        return self.controller.next_selectable(direction)

    def _handle_action(self, action: MenuAction) -> None:
        if action is MenuAction.OPEN_SIMULATION:
            self.controller.open_screen(MenuScreen.SIMULATION, select_last=True)
        elif action is MenuAction.OPEN_AUDIO:
            self.controller.open_screen(MenuScreen.AUDIO)
        elif action is MenuAction.OPEN_CREDITS:
            self.controller.open_screen(MenuScreen.CREDITS)
        elif action is MenuAction.BACK_TO_MAIN:
            self.controller.open_screen(MenuScreen.MAIN)
        elif action is MenuAction.SAVE_AUDIO_AND_BACK:
            self.save_options()
            self.controller.open_screen(MenuScreen.MAIN)
        elif action is MenuAction.START_MISSION:
            self.start_mission()
        elif action is MenuAction.EXIT:
            self.game.running = False
            self.show_menu = False

    def _play_button(self) -> None:
        self.audio.play_button(self.button_on_off == "on")

    def load_options(self) -> None:
        """Reload audio values from the existing options file."""
        settings = self.settings_repository.load_audio()
        self.volume = settings.volume
        self.sound_on_off = settings.music
        self.button_on_off = settings.button
        self.audio.apply_volume(self.volume)

    def _update_options(self) -> None:
        """Apply values from the audio menu to the mixer."""
        volume_item = cast(SliderItem, self.options[1])
        music_item = cast(SelectorItem[str], self.options[2])
        button_item = cast(SelectorItem[str], self.options[3])

        self.volume = volume_item.value
        self.audio.apply_volume(self.volume)

        music_setting = str(music_item.options[music_item.value])
        if music_setting == "on" and self.sound_on_off == "off":
            self.audio.set_music(True)
        elif music_setting == "off" and self.sound_on_off == "on":
            self.audio.set_music(False)
        self.sound_on_off = music_setting
        self.button_on_off = str(button_item.options[button_item.value])

    def save_options(self) -> None:
        """Persist the current audio options using the existing schema."""
        self.settings_repository.save_audio(
            AudioSettings(
                volume=self.volume,
                music=self.sound_on_off,
                button=self.button_on_off,
            )
        )

    def set_default_seed(self) -> None:
        """Set the seed associated with the currently selected map size."""
        map_item = cast(SelectorItem[str], self.simulation[2])
        seed_item = cast(TextInputItem, self.simulation[3])
        seed_item.text = str(GameOptions.SEED_DEFAULTS[map_item.value])

    def save_simulation_settings(self) -> None:
        """Persist the current typed configuration."""
        self.config = self._config_from_menu()
        self.settings_repository.save_simulation(self.config)

    def load_simulation_settings(self) -> None:
        """Load typed configuration from the current or legacy file."""
        mission_item = cast(SelectorItem[str], self.simulation[1])
        map_item = cast(SelectorItem[str], self.simulation[2])
        seed_item = cast(TextInputItem, self.simulation[3])
        drones_item = cast(SelectorItem[int], self.simulation[4])
        loaded = self.settings_repository.load_simulation(self.config)
        if loaded is None:
            return

        self.config = loaded
        mission = loaded.mission_config
        if 0 <= mission.objective < len(mission_item.options):
            mission_item.value = mission.objective
        map_names = [str(option).upper() for option in map_item.options]
        if mission.map_dim.upper() in map_names:
            map_item.value = map_names.index(mission.map_dim.upper())
        if mission.seed:
            seed_item.text = str(mission.seed)
        if mission.num_drones in drones_item.options:
            drones_item.value = list(drones_item.options).index(
                mission.num_drones
            )

        if not mission.seed:
            self.set_default_seed()

    def build_sim_settings(self) -> SimulationConfig:
        """Assemble current menu values into runtime simulation settings."""
        self.config = self._config_from_menu()
        return self.config

    def blit_loading(self, text: Sequence[str] = ("Loading...",)) -> None:
        """Render the existing loading screen."""
        self.renderer.draw_loading(text)

    def start_mission(self) -> None:
        """Save valid settings and start the mission."""
        seed_item = cast(TextInputItem, self.simulation[3])
        if seed_item.text:
            self.save_simulation_settings()
            self.game.start_mission()
            self.show_menu = False

    def _config_from_menu(self) -> SimulationConfig:
        mission_item = cast(SelectorItem[str], self.simulation[1])
        map_item = cast(SelectorItem[str], self.simulation[2])
        seed_item = cast(TextInputItem, self.simulation[3])
        drones_item = cast(SelectorItem[int], self.simulation[4])
        return replace(
            self.config,
            mission_config=replace(
                self.config.mission_config,
                objective=mission_item.value,
                map_dim=str(map_item.options[map_item.value]).upper(),
                seed=int(seed_item.text) if seed_item.text else 0,
                num_drones=int(drones_item.options[drones_item.value]),
            ),
        )
