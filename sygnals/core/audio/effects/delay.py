# sygnals/core/audio/effects/delay.py

"""
Implementation of a basic audio delay effect with feedback.
"""

import logging
import numpy as np
from numpy.typing import NDArray
# Import necessary types
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

    Creates echoes by mixing the original signal with delayed and attenuated
    versions of itself.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate of `y` (Hz).
        delay_time: Delay time in seconds (must be non-negative, default: 0.5).
        feedback: Feedback gain (0.0 to < 1.0). Controls how much of the delayed
                  signal is fed back into the delay line, creating echoes.
                  Values >= 1.0 can lead to instability (infinite gain). (default: 0.4).
        wet_level: Gain level for the delayed (wet) signal (0.0 to 1.0, default: 0.5).
        dry_level: Gain level for the original (dry) signal (0.0 to 1.0, default: 1.0).

    Returns:
        Audio time series with delay applied (float64). Length matches input `y`.

    Raises:
        ValueError: If input `y` is not 1D or parameters are invalid.

    Example:
        >>> sr = 22050
        >>> signal = np.zeros(sr)
        >>> signal[1000:1500] = 1.0 # A short pulse
        >>> delayed_signal = apply_delay(signal, sr, delay_time=0.2, feedback=0.6, wet_level=0.7)
        >>> # delayed_signal will contain the original pulse plus decaying echoes every 0.2 seconds.
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

    logger.info(f"Applying delay: time={delay_time}s, feedback={feedback}, wet={wet_level}, dry={dry_level}")

    # Calculate delay in samples
    delay_samples = int(delay_time * sr)

    # If delay is zero or negligible, just scale the input
    if delay_samples <= 0:
        logger.warning("Delay time is zero or negative. Returning dry signal scaled by (dry + wet).")
        # If delay is 0, wet signal is effectively the input signal
        return ((dry_level + wet_level) * y).astype(np.float64, copy=False)

    # Create output array (same size as input) initialized to zeros
    output_signal = np.zeros_like(y, dtype=np.float64)
    # Create delay buffer (needs to hold 'delay_samples')
    # Using a buffer larger than necessary can simplify indexing if needed, but exact size is efficient
    delay_buffer = np.zeros(delay_samples, dtype=np.float64)

    # --- More efficient processing using array operations ---
    buffer_write_idx = 0
    n_samples = len(y)

    for i in range(n_samples):
        # Calculate read index (wrapping around the buffer)
        buffer_read_idx = (buffer_write_idx - delay_samples + len(delay_buffer)) % len(delay_buffer)

        # Get delayed sample from buffer
        delayed_sample = delay_buffer[buffer_read_idx]

        # Calculate output sample (mix dry and wet)
        output_signal[i] = (dry_level * y[i]) + (wet_level * delayed_sample)

        # Calculate feedback signal to write into the buffer
        # Input to buffer = current sample + feedback * delayed sample
        feedback_sample = y[i] + feedback * delayed_sample

        # Write to buffer
        delay_buffer[buffer_write_idx] = feedback_sample

        # Advance buffer write index (circular buffer)
        buffer_write_idx = (buffer_write_idx + 1) % len(delay_buffer)

    # Optional: Normalize output to prevent clipping if gain/feedback cause levels > 1
    # max_abs_out = np.max(np.abs(output_signal))
    # if max_abs_out > 1.0:
    #     logger.warning(f"Delay output peak ({max_abs_out:.2f}) exceeds 1.0. Consider reducing levels/feedback.")
    #     # output_signal /= max_abs_out # Simple peak normalization

    return output_signal.astype(np.float64, copy=False)
