# sygnals/core/audio/effects/__init__.py

"""
Audio Effects Subpackage.

Contains implementations of various audio effects like reverb, delay, EQ,
modulation effects (chorus, flanger, tremolo), compression, and utilities
like gain adjustment, noise reduction, transient shaping, and stereo widening.

Note: Time/pitch manipulation and noise addition for augmentation purposes
      are located in the `sygnals.core.augment` package.
"""

# Import specific effects functions or classes
from .compression import simple_dynamic_range_compression
from .reverb import apply_reverb
from .delay import apply_delay
from .equalizer import apply_graphic_eq, apply_parametric_eq # Experimental EQ
from .chorus import apply_chorus
from .flanger import apply_flanger
from .tremolo import apply_tremolo
# Import utility effects
from .utility import (
    adjust_gain,
    noise_reduction_spectral,
    transient_shaping_hpss,
    stereo_widening_midside
)


__all__ = [
    # Core Effects
    "simple_dynamic_range_compression",
    "apply_reverb",
    "apply_delay",
    "apply_graphic_eq", # Experimental
    "apply_parametric_eq", # Experimental
    # Modulation Effects
    "apply_chorus",
    "apply_flanger",
    "apply_tremolo",
    # Utility Effects
    "adjust_gain",
    "noise_reduction_spectral",
    "transient_shaping_hpss",
    "stereo_widening_midside",
    # Add other effect functions/classes as they are implemented
]
