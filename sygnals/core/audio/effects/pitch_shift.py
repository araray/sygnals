# sygnals/core/audio/effects/pitch_shift.py

"""
Implementation of the pitch shifting audio effect.
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

    Args:
        y: Input audio time series (float64).
        sr: Sampling rate of `y`.
        n_steps: Number of semitones to shift. Can be fractional.
        bins_per_octave: Number of steps per octave (default: 12 for standard semitones).

    Returns:
        Pitch-shifted audio time series (float64).
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")

    logger.debug(f"Applying pitch shift: n_steps={n_steps}, bins_per_octave={bins_per_octave}")

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
        logger.error(f"Error during pitch shifting: {e}")
        raise
