# sygnals/core/features/__init__.py

"""
Core Feature Extraction Package.

Contains modules for extracting various types of features from signals:
- Time Domain
- Frequency Domain
- Cepstral (e.g., MFCC)
- Audio Specific (handled in sygnals.core.audio.features)

Includes a manager to orchestrate feature extraction.
"""

from . import time_domain
from . import frequency_domain
from . import cepstral
from . import manager

__all__ = [
    "time_domain",
    "frequency_domain",
    "cepstral",
    "manager",
]
