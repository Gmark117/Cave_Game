import unittest
from unittest.mock import Mock, patch

from ui.menu.audio_service import MenuAudioService
from ui.menu.settings_repository import AudioSettings


class MenuAudioServiceTests(unittest.TestCase):
    @patch("ui.menu.audio_service.mix")
    def test_initializes_resources_and_applies_preferences(self, mixer) -> None:
        mixer.Sound.return_value = Mock()
        mixer.music.get_busy.return_value = False

        service = MenuAudioService(AudioSettings(60, "on", "on"))

        mixer.init.assert_called_once_with()
        mixer.music.set_volume.assert_called_once_with(0.6)
        service.button.set_volume.assert_called_with(0.6)
        mixer.music.play.assert_called_once_with(-1)

    @patch("ui.menu.audio_service.mix")
    def test_button_play_respects_enabled_state(self, mixer) -> None:
        mixer.Sound.return_value = Mock()
        mixer.music.get_busy.return_value = True
        service = MenuAudioService(AudioSettings())

        service.play_button(False)
        service.play_button(True)

        service.button.play.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
