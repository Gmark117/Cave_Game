"""Typed state models for the Cave Explorer menu."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, Sequence, Tuple, TypeVar


Position = Tuple[int, int]
T = TypeVar("T")


class MenuScreen(Enum):
    """Logical menu screens controlled by the menu controller."""

    MAIN = "main"
    SIMULATION = "simulation"
    AUDIO = "audio"
    CREDITS = "credits"


class MenuAction(Enum):
    """Actions that can be triggered by selectable menu rows."""

    OPEN_SIMULATION = "open_simulation"
    OPEN_AUDIO = "open_audio"
    OPEN_CREDITS = "open_credits"
    EXIT = "exit"
    BACK_TO_MAIN = "back_to_main"
    SAVE_AUDIO_AND_BACK = "save_audio_and_back"
    START_MISSION = "start_mission"


@dataclass
class MenuItem:
    """Shared presentation state for one menu row."""

    label: str
    position: Position
    size: int = 35
    font_big: bool = False
    alignment: str = "midleft"
    selectable: bool = True


@dataclass
class TitleItem(MenuItem):
    """Non-selectable title row."""

    selectable: bool = field(default=False, init=False)


@dataclass
class ButtonItem(MenuItem):
    """Selectable row that triggers a ``MenuAction``."""

    action: MenuAction = MenuAction.BACK_TO_MAIN


@dataclass
class SelectorItem(MenuItem, Generic[T]):
    """Left/right row for choosing one value from a sequence."""

    options: Sequence[T] = field(default_factory=tuple)
    value: int = 0


@dataclass
class TextInputItem(MenuItem):
    """Row that stores simple numeric text input."""

    text: str = ""


@dataclass
class SliderItem(MenuItem):
    """Left/right row for bounded numeric values such as volume."""

    value: int = 0
    minimum: int = 0
    maximum: int = 100
    step: int = 20


MenuRow = TitleItem | ButtonItem | SelectorItem[object] | TextInputItem | SliderItem
MenuActionHandler = Callable[[MenuAction], None]
