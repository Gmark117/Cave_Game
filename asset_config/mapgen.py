"""Map generation constants and presets."""

from enum import Enum


class MapGen:
    """Map generation parameters."""

    STEP = 10
    STRENGTH = 16
    LIFE = 75

    CHAOTIC_FACTOR_LOW = 0.2
    CHAOTIC_FACTOR_HIGH = 0.4

    WALL_NOISE_BASE_DEPTH = 3
    WALL_NOISE_BASE_PROB = 0.6
    WALL_NOISE_MIN_SPIKES = 24
    WALL_NOISE_SPIKE_DIVISOR = 25
    WALL_NOISE_SPIKE_EXTRA = 6
    WALL_NOISE_DILATE_KSIZE = 3
    WALL_NOISE_DILATE_ITER = 1

    BLUR_KERNEL_FINAL = 5
    BLUR_KERNEL_MULTIPLIER_STRENGTH = 1.5
    MEDIAN_FILTER_REDUCTION = 1
    BORDER_THICKNESS = 50
    DEFAULT_NUM_PROCESSES = 8


class Brush(Enum):
    """Brush types for drawing."""

    ROUND = 0
    ELLIPSE = 1
    CHAOTIC = 2
    DIAMOND = 3
    OCTAGON = 4
    RECTANGULAR = 5


class WormInputs(Enum):
    """Map generation parameters for different cave sizes."""

    SMALL = [MapGen.STEP * 4, MapGen.STRENGTH * 4, MapGen.LIFE]
    MEDIUM = [MapGen.STEP * 2, MapGen.STRENGTH * 2, MapGen.LIFE * 4]
    BIG = [MapGen.STEP, MapGen.STRENGTH, MapGen.LIFE * 15]
