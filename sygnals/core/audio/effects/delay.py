# sygnals/core/audio/effects/delay.py

"""
Implementation of a basic audio delay effect.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Union, Literal, Dict

logger = logging.getLogger(__name__)

def apply_delay(
    y: NDArray[np.float64],
    sr: int,
    delay_time: float = 0.5,
    feedback: float = 0.4,
    wet_level: float = 0.5,
    dry_level: float = 1.0
) -> NDArray[np.float64]:
    """
    Applies a simple feedback delay effect to an audio signal.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate of `y`.
        delay_time: Delay time in seconds (default: 0.5).
        feedback: Feedback gain (0 to < 1). Controls how much of the delayed
                  signal is fed back into the delay line, creating echoes.
                  (default: 0.4). Values >= 1 can lead to infinite gain.
        wet_level: Gain level for the delayed (wet) signal (0 to 1, default: 0.5).
        dry_level: Gain level for the original (dry) signal (0 to 1, default: 1.0).

    Returns:
        Audio time series with delay applied (float64). Length matches input.
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    if delay_time < 0:
        raise ValueError("delay_time must be non-negative.")
    if not 0.0 <= feedback < 1.0:
        # Feedback >= 1 can cause instability / infinite gain
        raise ValueError("feedback gain must be between 0.0 and < 1.0.")
    if not 0.0 <= wet_level <= 1.0:
        raise ValueError("wet_level must be between 0.0 and 1.0.")
    if not 0.0 <= dry_level <= 1.0:
        raise ValueError("dry_level must be between 0.0 and 1.0.")

    logger.debug(f"Applying delay: time={delay_time}s, feedback={feedback}, wet={wet_level}, dry={dry_level}")

    # Calculate delay in samples
    delay_samples = int(delay_time * sr)

    # If delay is zero or negligible, return dry signal scaled
    if delay_samples <= 0:
        logger.warning("Delay time is zero or negative. Returning dry signal only.")
        return (dry_level * y).astype(np.float64, copy=False)

    # Create output array (same size as input)
    output_signal = np.zeros_like(y, dtype=np.float64)
    # Create delay buffer (can be slightly larger for safety, but delay_samples should suffice)
    delay_buffer = np.zeros(delay_samples, dtype=np.float64)
    buffer_write_idx = 0

    # Process sample by sample (simple but less efficient than vectorized approaches)
    for i in range(len(y)):
        # Get delayed sample from buffer
        buffer_read_idx = (buffer_write_idx - delay_samples + len(delay_buffer)) % len(delay_buffer)
        delayed_sample = delay_buffer[buffer_read_idx]

        # Calculate output sample (mix dry and wet)
        output_signal[i] = (dry_level * y[i]) + (wet_level * delayed_sample)

        # Calculate feedback signal and write to buffer
        feedback_sample = y[i] + feedback * delayed_sample
        delay_buffer[buffer_write_idx] = feedback_sample

        # Advance buffer write index
        buffer_write_idx = (buffer_write_idx + 1) % len(delay_buffer)

    # Normalize output (optional)
    # max_abs_out = np.max(np.abs(output_signal))
    # if max_abs_out > 1.0:
    #     logger.warning(f"Delay output peak ({max_abs_out:.2f}) exceeds 1.0. Consider reducing levels/feedback.")
        # output_signal /= max_abs_out

    return output_signal.astype(np.float64, copy=False)
