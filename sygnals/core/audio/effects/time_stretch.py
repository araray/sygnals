# sygnals/core/audio/effects/time_stretch.py

"""
Implementation of the time stretching audio effect.
"""

import logging
import librosa
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

def time_stretch(
    y: NDArray[np.float64],
    rate: float
) -> NDArray[np.float64]:
    """
    Time-stretches an audio signal without changing pitch using librosa.

    Args:
        y: Input audio time series (float64).
        rate: Factor by which to stretch the audio.
              rate > 1.0 speeds up the audio.
              rate < 1.0 slows down the audio.

    Returns:
        Time-stretched audio time series (float64).
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    if rate <= 0:
        raise ValueError("Time stretch rate must be positive.")

    logger.debug(f"Applying time stretch: rate={rate}")

    try:
        # Note: librosa.effects.time_stretch uses phase vocoder
        y_stretched = librosa.effects.time_stretch(
            y=y,
            rate=rate
        )
        # Ensure output is float64
        return y_stretched.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error during time stretching: {e}")
        raise
