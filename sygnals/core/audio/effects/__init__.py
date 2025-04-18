# sygnals/core/audio/effects/__init__.py

"""
Audio Effects Subpackage.

Contains implementations of various audio effects like reverb, delay, pitch shifting, etc.
"""

# Import specific effects functions or classes here as they are created
from .pitch_shift import pitch_shift
from .time_stretch import time_stretch
from .compression import simple_dynamic_range_compression
from .reverb import apply_reverb # Added reverb
from .delay import apply_delay   # Added delay
from .equalizer import apply_graphic_eq, apply_parametric_eq # Added EQ placeholders

__all__ = [
    "pitch_shift",
    "time_stretch",
    "simple_dynamic_range_compression",
    "apply_reverb",
    "apply_delay",
    "apply_graphic_eq",
    "apply_parametric_eq",
    # Add other effect functions/classes as they are implemented
]
