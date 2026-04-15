"""Gameplay and display configuration values."""


class Display:
    """Display dimensions and positions."""

    W = 1200
    H = 750
    FULL_W = 1915
    FULL_H = 1010
    LEGEND_WIDTH = 300

    CENTER_W = W / 2
    CENTER_H = H / 2
    ALIGN_L = CENTER_W - 550


class GameOptions:
    """Game configuration options."""

    MISSION = ["Exploration", "Search and Rescue"]
    MAP_SIZE = ["Small", "Medium", "Big"]
    PREFAB = ["No", "Yes"]
    VISION = [39, 19, 4]
    DRONE_ICON = [(30, 30), (20, 20), (10, 10)]
    ROVER_ICON = [(40, 40), (25, 25), (10, 10)]
    SEED_DEFAULTS = [5, 19, 837]
