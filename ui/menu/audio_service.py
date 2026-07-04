"""Pygame mixer ownership for menu audio."""

from __future__ import annotations

import pygame.mixer as mix

from ui.menu.settings_repository import AudioSettings
from asset_config.media import Audio


class MenuAudioService:
    """Own mixer resources and apply menu audio preferences."""

    def __init__(self, settings: AudioSettings) -> None:
        """Initialize the mixer and apply persisted audio settings."""
        mix.init()
        mix.music.load(Audio.AMBIENT.value)
        self.button = mix.Sound(Audio.BUTTON.value)
        self.button.set_volume(0.5)
        self.apply_volume(settings.volume)
        if settings.music == "on" and not mix.music.get_busy():
            mix.music.play(-1)

    def apply_volume(self, volume: int) -> None:
        """Apply a 0..100 volume value to music and button sound."""
        mix.music.set_volume(volume / 100)
        self.button.set_volume(volume / 100)

    def set_music(self, enabled: bool) -> None:
        """Start or stop the looping menu background track."""
        if enabled:
            mix.music.play(-1)
        else:
            mix.music.stop()

    def play_button(self, enabled: bool) -> None:
        """Play the click sound when button audio is enabled."""
        if enabled:
            self.button.play()
