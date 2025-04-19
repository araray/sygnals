# sygnals/core/augment/effects_based.py

"""
Data augmentation techniques based on applying audio effects like
pitch shifting and time stretching.
"""

import logging
import librosa
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Union, Literal, Dict

logger = logging.getLogger(__name__)

def pitch_shift(
    y: NDArray[np.float64],
    sr: int,
    n_steps: float,
    bins_per_octave: int = 12
) -> NDArray[np.float64]:
    """
    Shifts the pitch of an audio signal using librosa.

    Commonly used for data augmentation to simulate variations in pitch.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate of `y`.
        n_steps: Number of semitones to shift. Can be fractional.
                 Positive values shift pitch up, negative values shift down.
        bins_per_octave: Number of steps per octave (default: 12 for standard semitones).

    Returns:
        Pitch-shifted audio time series (float64).

    Raises:
        ValueError: If input `y` is not 1D.
        Exception: For errors during librosa pitch shifting.
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")

    logger.info(f"Applying pitch shift augmentation: n_steps={n_steps}, bins_per_octave={bins_per_octave}")

    try:
        y_shifted = librosa.effects.pitch_shift(
            y=y,
            sr=sr,
            n_steps=n_steps,
            bins_per_octave=bins_per_octave,
            res_type='kaiser_best' # Default librosa quality
        )
        # Ensure output is float64
        return y_shifted.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error during pitch shifting augmentation: {e}")
        raise

def time_stretch(
    y: NDArray[np.float64],
    rate: float
) -> NDArray[np.float64]:
    """
    Time-stretches an audio signal without changing pitch using librosa.

    Commonly used for data augmentation to simulate variations in tempo or duration.

    Args:
        y: Input audio time series (1D float64).
        rate: Factor by which to stretch the audio.
              rate > 1.0 speeds up the audio (makes it shorter).
              rate < 1.0 slows down the audio (makes it longer).

    Returns:
        Time-stretched audio time series (float64).

    Raises:
        ValueError: If input `y` is not 1D or rate is not positive.
        Exception: For errors during librosa time stretching.
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    if rate <= 0:
        raise ValueError("Time stretch rate must be positive.")

    logger.info(f"Applying time stretch augmentation: rate={rate}")

    try:
        # Note: librosa.effects.time_stretch uses phase vocoder
        y_stretched = librosa.effects.time_stretch(
            y=y,
            rate=rate
        )
        # Ensure output is float64
        return y_stretched.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error during time stretching augmentation: {e}")
        raise
