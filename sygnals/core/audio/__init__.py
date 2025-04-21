# sygnals/core/audio/__init__.py

"""
Core Audio Processing Package.

Contains modules for audio-specific input/output, effects, and feature extraction.
"""

from . import io
from . import effects
from . import features

__all__ = [
    "io",
    "effects",
    "features",
]
