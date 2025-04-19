# sygnals/core/augment/__init__.py

"""
Core Data Augmentation Package.

Contains modules for applying various augmentation techniques to signals,
often used to increase the diversity of training data for machine learning models.
Includes noise addition, time/pitch manipulation, etc.
"""

from .noise import add_noise
from .effects_based import pitch_shift, time_stretch

__all__ = [
    "add_noise",
    "pitch_shift",
    "time_stretch",
    # Add other augmentation functions/classes as they are implemented
]
