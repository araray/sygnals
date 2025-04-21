# sygnals/core/audio/effects/compression.py

"""
Implementation of basic audio compression effects.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Union, Literal, Dict

logger = logging.getLogger(__name__)

def simple_dynamic_range_compression(
    y: NDArray[np.float64],
    threshold: float = 0.8,
    ratio: float = 4.0
) -> NDArray[np.float64]:
    """
    Applies a very simple dynamic range compression (downward compression).

    This is a basic example and not a full-featured compressor.
    It reduces the volume of parts of the signal above the threshold.

    Args:
        y: Input audio time series (float64). Assumed to be normalized [-1, 1].
        threshold: Amplitude threshold above which compression starts (0 to 1).
        ratio: Compression ratio (e.g., 4.0 means 4:1 compression). Must be >= 1.0.

    Returns:
        Compressed audio time series (float64).
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")
    if ratio < 1.0:
        raise ValueError("Compression ratio must be >= 1.0.")

    logger.debug(f"Applying simple compression: threshold={threshold}, ratio={ratio}")

    # Work with absolute amplitude
    abs_y = np.abs(y)
    y_compressed = y.copy() # Start with original signal

    # Find samples above the threshold
    above_threshold_indices = np.where(abs_y > threshold)[0]

    if above_threshold_indices.size > 0:
        # Calculate gain reduction only for samples above threshold
        # Amount over threshold (in linear amplitude)
        over_amount = abs_y[above_threshold_indices] - threshold
        # Apply compression ratio to the amount over threshold
        reduced_over_amount = over_amount / ratio
        # New amplitude is threshold + reduced amount over threshold
        new_amplitude = threshold + reduced_over_amount

        # Apply the gain reduction to the original signal (preserving sign)
        # Gain factor = new_amplitude / original_amplitude
        gain_factor = new_amplitude / abs_y[above_threshold_indices]
        y_compressed[above_threshold_indices] *= gain_factor

    # Ensure output is float64
    return y_compressed.astype(np.float64, copy=False)

# --- Add other compression types later (e.g., upward, multi-band) ---
