# sygnals/core/__init__.py

"""
Core Processing Package for Sygnals.

Contains modules for:
- Data Handling (I/O, basic manipulation)
- Digital Signal Processing (DSP) algorithms
- Signal Transforms
- Digital Filters
- Audio Processing (I/O, Effects, Features)
- Feature Extraction (Time, Frequency, Cepstral)
- Plugin Management Base
- Batch Processing Logic
- Custom Code Execution Utilities
- Storage Utilities (e.g., Database)
"""

# Import key modules or namespaces
from . import data_handler
from . import dsp
from . import transforms
from . import filters
from . import audio # New audio subpackage
from . import features # New features subpackage
from . import plugin_manager # Keep for now, may evolve in Phase 3
from . import batch_processor # Keep for now, may evolve
from . import custom_exec # Keep for now
from . import storage # Keep for now

__all__ = [
    "data_handler",
    "dsp",
    "transforms",
    "filters",
    "audio", # Expose the audio subpackage
    "features", # Expose the features subpackage
    "plugin_manager",
    "batch_processor",
    "custom_exec",
    "storage",
]
