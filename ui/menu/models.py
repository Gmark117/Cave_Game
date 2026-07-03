"""Typed state models for the Cave Explorer menu."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, Sequence, Tuple, TypeVar


Position = Tuple[int, int]
T = TypeVar("T")


class MenuScreen(Enum):
    MAIN = "main"
    SIMULATION = "simulation"
    AUDIO = "audio"
    CREDITS = "credits"


class MenuAction(Enum):
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
    selectable: bool = field(default=False, init=False)


@dataclass
class ButtonItem(MenuItem):
    action: MenuAction = MenuAction.BACK_TO_MAIN


@dataclass
class SelectorItem(MenuItem, Generic[T]):
    options: Sequence[T] = field(default_factory=tuple)
    value: int = 0


@dataclass
class TextInputItem(MenuItem):
    text: str = ""


@dataclass
class SliderItem(MenuItem):
    value: int = 0
    minimum: int = 0
    maximum: int = 100
    step: int = 20


MenuRow = TitleItem | ButtonItem | SelectorItem[object] | TextInputItem | SliderItem
MenuActionHandler = Callable[[MenuAction], None]
