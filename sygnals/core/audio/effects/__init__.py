# sygnals/core/audio/effects/__init__.py

"""
Audio Effects Subpackage.

Contains implementations of various audio effects like reverb, delay, EQ,
modulation effects (chorus, flanger, tremolo), compression, pitch/time manipulation,
and utility effects (gain, noise).
"""

# Import specific effects functions or classes here as they are implemented/updated
from .pitch_shift import pitch_shift
from .time_stretch import time_stretch
from .compression import simple_dynamic_range_compression
from .reverb import apply_reverb
from .delay import apply_delay
from .equalizer import apply_graphic_eq, apply_parametric_eq
from .chorus import apply_chorus # Placeholder added
from .flanger import apply_flanger # Placeholder added
from .tremolo import apply_tremolo # Placeholder added
from .utility import adjust_gain, add_noise # Utility effects added

__all__ = [
    # Core Effects
    "pitch_shift",
    "time_stretch",
    "simple_dynamic_range_compression",
    "apply_reverb",
    "apply_delay",
    "apply_graphic_eq",
    "apply_parametric_eq",
    # Modulation Effects (Placeholders)
    "apply_chorus",
    "apply_flanger",
    "apply_tremolo",
    # Utility/Augmentation Effects
    "adjust_gain",
    "add_noise",
    # Add other effect functions/classes as they are implemented
]
