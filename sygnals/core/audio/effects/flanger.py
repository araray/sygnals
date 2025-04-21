# sygnals/core/audio/effects/flanger.py

"""
Implementation of the audio flanger effect.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from scipy.signal import sawtooth # For LFO generation
# Import necessary types
from typing import List, Tuple, Dict, Optional, Union, Literal, Any

# Import LFO generator from tremolo (or move to a shared utility)
# For now, copy the LFO generator here for self-containment
# TODO: Refactor LFO generation into a shared utility module if more effects use it.
from .tremolo import _generate_lfo

logger = logging.getLogger(__name__)

# Define a small epsilon for safe division and log calculations
_EPSILON = np.finfo(np.float64).eps

def apply_flanger(
    y: NDArray[np.float64],
    sr: int,
    rate: float = 0.5, # LFO rate in Hz (typically 0.1 Hz to 5 Hz)
    depth: float = 0.005, # Modulation depth in seconds (e.g., 5ms)
    delay: float = 0.001, # Base delay in seconds (typically 1-5ms)
    feedback: float = 0.5, # Feedback gain (-1 to < 1). Negative feedback inverts phase.
    wet_level: float = 0.5, # Gain for the wet (flanged) signal
    dry_level: float = 0.5, # Gain for the dry (original) signal (often 50/50 mix)
    lfo_shape: Literal['sine', 'triangle'] = 'sine' # LFO shape (square is less common for flanger)
) -> NDArray[np.float64]:
    """
    Applies a flanger audio effect using a modulated delay line with feedback.

    Flanging mixes the original signal with a copy delayed by a very short,
    time-varying amount (modulated by an LFO). Feedback is often applied to
    create characteristic resonant "sweeping" sounds.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate of `y`.
        rate: Rate of the LFO modulating the delay time (Hz, default: 0.5). Must be positive.
        depth: Depth of the delay time modulation (seconds, default: 0.005). Must be non-negative.
               Max depth is limited by base delay.
        delay: Average delay time offset (seconds, default: 0.001). Must be > depth and very short.
        feedback: Feedback gain for the delayed signal (-1.0 to < 1.0, default: 0.5).
                  Negative values invert the phase of the feedback. Values >= 1.0 can lead to instability.
        wet_level: Gain level for the flanged (wet) signal (0 to 1, default: 0.5).
        dry_level: Gain level for the original (dry) signal (0 to 1, default: 0.5).
        lfo_shape: Waveform shape of the LFO ('sine' or 'triangle', default: 'sine').

    Returns:
        The processed audio time series with flanger applied (float64).

    Raises:
        ValueError: If input `y` is not 1D or parameters are invalid.
    """
    # --- Parameter Validation ---
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    if rate <= 0:
        raise ValueError("LFO rate must be positive.")
    if depth < 0:
        raise ValueError("Modulation depth must be non-negative.")
    if delay <= 0:
        raise ValueError("Base delay must be positive.")
    if depth >= delay:
        raise ValueError(f"Modulation depth ({depth}s) must be less than the base delay ({delay}s).")
    # Allow negative feedback for phase inversion effects
    if not -1.0 <= feedback < 1.0:
        raise ValueError("Feedback gain must be between -1.0 and < 1.0.")
    if not 0.0 <= wet_level <= 1.0:
        raise ValueError("Wet level must be between 0.0 and 1.0.")
    if not 0.0 <= dry_level <= 1.0:
        raise ValueError("Dry level must be between 0.0 and 1.0.")
    if lfo_shape not in ['sine', 'triangle']: # Square LFO less common for flanger
        raise ValueError("LFO shape must be 'sine' or 'triangle'.")

    logger.info(f"Applying Flanger: rate={rate} Hz, depth={depth}s, delay={delay}s, feedback={feedback}, wet={wet_level}, shape={lfo_shape}")

    # --- Initialization ---
    num_samples = len(y)
    output_signal = np.zeros_like(y, dtype=np.float64)

    # Calculate maximum delay needed in samples
    max_delay_samples = int(np.ceil((delay + depth) * sr)) + 2 # Buffer for interpolation
    delay_buffer = np.zeros(max_delay_samples, dtype=np.float64)
    buffer_write_idx = 0

    # Generate LFO signal (0 to 1)
    lfo = _generate_lfo(sr, num_samples, rate, lfo_shape)

    # Calculate modulated delay time in samples
    modulated_delay_samples = (delay + (lfo * 2.0 - 1.0) * depth) * sr
    if np.min(modulated_delay_samples) <= 0:
         logger.warning(f"Calculated minimum modulated delay ({np.min(modulated_delay_samples)/sr:.4f}s) is near zero. Clamping.")
         modulated_delay_samples = np.maximum(modulated_delay_samples, _EPSILON * sr)

    # --- Processing Loop ---
    buffer_indices = np.arange(max_delay_samples)

    for i in range(num_samples):
        # Calculate the fractional read position relative to the write position
        current_delay = modulated_delay_samples[i]
        target_relative_read_pos = -current_delay # Target position in the past

        # Perform linear interpolation to get the delayed sample
        relative_indices = (buffer_indices - buffer_write_idx + max_delay_samples) % max_delay_samples
        sort_indices = np.argsort(relative_indices)
        xp_sorted = relative_indices[sort_indices]
        fp_sorted = delay_buffer[sort_indices]
        delayed_sample = np.interp(target_relative_read_pos, xp_sorted, fp_sorted)

        # Calculate the output signal (mix dry and wet)
        output_signal[i] = (dry_level * y[i]) + (wet_level * delayed_sample)

        # Calculate the value to write into the delay buffer (input + feedback * delayed_output)
        # Flanger feedback often uses the delayed sample directly
        buffer_input = y[i] + feedback * delayed_sample

        # Write to buffer, wrapping index
        delay_buffer[buffer_write_idx] = buffer_input

        # Advance buffer write index
        buffer_write_idx = (buffer_write_idx + 1) % max_delay_samples

    return output_signal.astype(np.float64, copy=False)
