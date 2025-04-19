# sygnals/core/audio/effects/utility.py

"""
Utility audio effects, primarily gain adjustment.
Noise addition moved to sygnals.core.augment.noise.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Literal
import warnings # Import the warnings module

logger = logging.getLogger(__name__)

def adjust_gain(
    y: NDArray[np.float64],
    gain_db: float
) -> NDArray[np.float64]:
    """
    Adjusts the gain (amplitude) of an audio signal by a specified amount in decibels (dB).

    Args:
        y: Input audio time series (1D or multi-channel float64).
        gain_db: Gain adjustment in decibels. Positive values amplify, negative values attenuate.

    Returns:
        Audio time series with gain adjusted (float64).

    Raises:
        ValueError: If input `y` is not a NumPy array.

    Example:
        >>> signal = np.array([0.1, 0.2, -0.1])
        >>> amplified_signal = adjust_gain(signal, 6.0) # Approx double amplitude
        >>> attenuated_signal = adjust_gain(signal, -6.0) # Approx half amplitude
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("Input audio data must be a NumPy array.")

    logger.info(f"Adjusting gain by {gain_db:.2f} dB.")

    # Convert dB gain to linear amplitude multiplier
    # gain_db = 20 * log10(multiplier) => multiplier = 10**(gain_db / 20)
    multiplier = 10.0**(gain_db / 20.0)

    # Apply gain
    y_adjusted = y * multiplier

    # Optional: Check for clipping after gain adjustment
    # max_abs_out = np.max(np.abs(y_adjusted))
    # if max_abs_out > 1.0:
    #     logger.warning(f"Gain adjustment resulted in peak amplitude > 1.0 ({max_abs_out:.2f}). Clipping may occur if saving to PCM.")

    return y_adjusted.astype(np.float64, copy=False)

# Removed add_noise function (moved to sygnals.core.augment.noise)
