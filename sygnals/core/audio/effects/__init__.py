# sygnals/core/audio/effects/__init__.py

"""
Audio Effects Subpackage.

Contains implementations of various audio effects like reverb, delay, pitch shifting, etc.
"""

# Import specific effects functions or classes here as they are created
from .pitch_shift import pitch_shift
from .time_stretch import time_stretch
from .compression import simple_dynamic_range_compression

__all__ = [
    "pitch_shift",
    "time_stretch",
    "simple_dynamic_range_compression",
    # Add other effect functions/classes as they are implemented
]
