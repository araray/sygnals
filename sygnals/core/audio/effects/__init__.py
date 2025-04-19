# sygnals/core/audio/effects/__init__.py

"""
Audio Effects Subpackage.

Contains implementations of various audio effects like reverb, delay, EQ,
modulation effects (chorus, flanger, tremolo), compression, and basic utilities.

Note: Time/pitch manipulation and noise addition for augmentation purposes
      have been moved to the `sygnals.core.augment` package.
"""

# Import specific effects functions or classes here as they are implemented/updated
from .compression import simple_dynamic_range_compression
from .reverb import apply_reverb
from .delay import apply_delay
from .equalizer import apply_graphic_eq, apply_parametric_eq
from .chorus import apply_chorus # Placeholder
from .flanger import apply_flanger # Placeholder
from .tremolo import apply_tremolo # Placeholder
from .utility import adjust_gain # Keep utility effects like gain adjust here

__all__ = [
    # Core Effects
    "simple_dynamic_range_compression",
    "apply_reverb",
    "apply_delay",
    "apply_graphic_eq",
    "apply_parametric_eq",
    # Modulation Effects (Placeholders)
    "apply_chorus",
    "apply_flanger",
    "apply_tremolo",
    # Utility Effects
    "adjust_gain",
    # Add other effect functions/classes as they are implemented
    # Removed: pitch_shift, time_stretch, add_noise (moved to augment)
]
