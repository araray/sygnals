# sygnals/core/audio/effects/chorus.py

"""
Implementation of the audio chorus effect.
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

def apply_chorus(
    y: NDArray[np.float64],
    sr: int,
    rate: float = 1.5, # LFO rate in Hz
    depth: float = 0.002, # Modulation depth in seconds (e.g., 2ms)
    delay: float = 0.025, # Base delay in seconds (e.g., 25ms)
    feedback: float = 0.2, # Feedback gain (0 to < 1)
    wet_level: float = 0.5, # Gain for the wet (chorused) signal
    dry_level: float = 1.0, # Gain for the dry (original) signal
    lfo_shape: Literal['sine', 'triangle', 'square'] = 'sine' # LFO shape
) -> NDArray[np.float64]:
    """
    Applies a chorus audio effect using a modulated delay line.

    Chorus creates a thicker sound by mixing the original signal with copies
    that are slightly delayed and pitch-modulated using a Low-Frequency Oscillator (LFO).

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate of `y`.
        rate: Rate of the LFO modulating the delay time (Hz, default: 1.5). Must be positive.
        depth: Depth of the delay time modulation (seconds, default: 0.002). Must be non-negative.
               Determines the intensity of the pitch modulation effect. Max depth is limited by base delay.
        delay: Average delay time offset (seconds, default: 0.025). Must be > depth.
        feedback: Feedback gain for the delayed signal (0 to < 1, default: 0.2).
                  Values >= 1.0 can lead to instability.
        wet_level: Gain level for the chorused (wet) signal (0 to 1, default: 0.5).
        dry_level: Gain level for the original (dry) signal (0 to 1, default: 1.0).
        lfo_shape: Waveform shape of the LFO ('sine', 'triangle', 'square', default: 'sine').

    Returns:
        The processed audio time series with chorus applied (float64).

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
        # If depth >= delay, modulated delay can become zero or negative, causing issues.
        raise ValueError(f"Modulation depth ({depth}s) must be less than the base delay ({delay}s).")
    if not 0.0 <= feedback < 1.0:
        raise ValueError("Feedback gain must be between 0.0 and < 1.0.")
    if not 0.0 <= wet_level <= 1.0:
        raise ValueError("Wet level must be between 0.0 and 1.0.")
    if not 0.0 <= dry_level <= 1.0:
        raise ValueError("Dry level must be between 0.0 and 1.0.")
    if lfo_shape not in ['sine', 'triangle', 'square']:
        raise ValueError("LFO shape must be 'sine', 'triangle', or 'square'.")

    logger.info(f"Applying Chorus: rate={rate} Hz, depth={depth}s, delay={delay}s, feedback={feedback}, wet={wet_level}, shape={lfo_shape}")

    # --- Initialization ---
    num_samples = len(y)
    output_signal = np.zeros_like(y, dtype=np.float64)

    # Calculate maximum delay needed in samples to size the buffer adequately
    max_delay_samples = int(np.ceil((delay + depth) * sr)) + 2 # Add buffer for interpolation
    delay_buffer = np.zeros(max_delay_samples, dtype=np.float64)
    buffer_write_idx = 0 # Current position to write into the buffer

    # Generate LFO signal (ranging 0 to 1)
    lfo = _generate_lfo(sr, num_samples, rate, lfo_shape)

    # Calculate modulated delay time in samples for each output sample
    # LFO (0..1) -> Modulated Delay Offset (-depth..+depth)
    # Modulated Delay = delay + (lfo * 2 - 1) * depth
    # Delay in Samples = (delay + (lfo * 2 - 1) * depth) * sr
    modulated_delay_samples = (delay + (lfo * 2.0 - 1.0) * depth) * sr

    # Ensure delay is always positive (should be guaranteed by depth < delay check)
    if np.min(modulated_delay_samples) <= 0:
         logger.warning(f"Calculated minimum modulated delay ({np.min(modulated_delay_samples)/sr:.4f}s) is near zero. Clamping.")
         modulated_delay_samples = np.maximum(modulated_delay_samples, _EPSILON * sr) # Clamp to small positive

    # --- Processing Loop ---
    buffer_indices = np.arange(max_delay_samples) # Indices for interpolation

    for i in range(num_samples):
        # Calculate the read position (fractional index) in the past
        current_delay = modulated_delay_samples[i]
        read_pos_fractional = buffer_write_idx - current_delay

        # Wrap the fractional read position around the buffer
        # We need integer indices for np.interp's xp argument
        # The value we want is at read_pos_fractional relative to buffer_write_idx

        # Use np.interp for linear interpolation
        # xp: The x-coordinates of the data points (buffer indices relative to current write pos)
        # fp: The y-coordinates of the data points (values in the buffer)
        # x: The x-coordinate where to interpolate (the fractional read position)

        # Create relative indices for interpolation based on current write position
        relative_indices = (buffer_indices - buffer_write_idx + max_delay_samples) % max_delay_samples
        # Sort buffer values based on these relative indices for np.interp
        # This ensures xp is monotonically increasing as required by np.interp
        sort_indices = np.argsort(relative_indices)
        xp_sorted = relative_indices[sort_indices]
        fp_sorted = delay_buffer[sort_indices]

        # The target relative read position (negative because it's in the past)
        target_relative_read_pos = -current_delay

        # Interpolate using the sorted relative indices and buffer values
        delayed_sample = np.interp(target_relative_read_pos, xp_sorted, fp_sorted)


        # Calculate output sample (mix dry and wet)
        output_signal[i] = (dry_level * y[i]) + (wet_level * delayed_sample)

        # Calculate feedback signal to write into the buffer
        # Input to buffer = current sample + feedback * delayed sample (or output?)
        # Common feedback is from the delayed sample before mixing
        feedback_sample = y[i] + feedback * delayed_sample

        # Write to buffer, wrapping index
        delay_buffer[buffer_write_idx] = feedback_sample

        # Advance buffer write index (circular buffer)
        buffer_write_idx = (buffer_write_idx + 1) % max_delay_samples

    # Optional: Normalize output if needed
    # max_abs_out = np.max(np.abs(output_signal))
    # if max_abs_out > 1.0: logger.warning(...)

    return output_signal.astype(np.float64, copy=False)
