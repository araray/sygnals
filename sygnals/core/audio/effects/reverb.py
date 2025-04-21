# sygnals/core/audio/effects/reverb.py

"""
Implementation of a basic audio reverb effect using convolution with a generated impulse response.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve # Use FFT-based convolution for efficiency
# Import necessary types
from typing import Tuple, Optional, Union, Literal, Dict

logger = logging.getLogger(__name__)

def _generate_basic_ir(
    sr: int,
    decay_time: float = 0.5,
    seed: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Generates a very basic exponentially decaying noise impulse response (IR).

    This creates a simple synthetic IR suitable for basic reverb effects.
    More realistic reverbs often use measured IRs or complex algorithms.

    Args:
        sr: Sampling rate (Hz).
        decay_time: Approximate time (in seconds) for the reverb tail to decay
                    significantly (e.g., by 60 dB, RT60). Must be non-negative.
        seed: Optional random seed for noise generation reproducibility.

    Returns:
        Impulse response (1D NumPy array, float64). Returns a Dirac delta ([1.0])
        if decay_time is effectively zero.
    """
    if decay_time < 0:
        raise ValueError("decay_time must be non-negative.")

    # Calculate IR length based on decay time. Add buffer for tail.
    # Ensure length is at least 1 sample.
    ir_length_samples = max(1, int(sr * decay_time * 1.5))

    # If decay time is negligible, return a simple impulse (no reverb)
    if decay_time < 1e-6 or ir_length_samples <= 1:
        logger.debug("Decay time is near zero, returning Dirac delta impulse response.")
        return np.array([1.0], dtype=np.float64)

    logger.debug(f"Generating basic IR: sr={sr}, decay_time={decay_time:.2f}s, length={ir_length_samples} samples")

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Generate white noise
    noise = rng.standard_normal(ir_length_samples).astype(np.float64)

    # Create exponential decay envelope
    # Calculate decay factor such that amplitude reaches -60dB (0.001) at decay_time
    # exp(-decay_factor * decay_time * sr) = 0.001
    # -decay_factor * decay_time * sr = ln(0.001)
    # decay_factor = -ln(0.001) / (decay_time * sr)
    # Use epsilon to avoid division by zero if decay_time is extremely small but > 0
    epsilon = 1e-9
    decay_factor = -np.log(0.001) / (decay_time * sr + epsilon)
    time_indices = np.arange(ir_length_samples)
    envelope = np.exp(-decay_factor * time_indices)

    # Apply envelope to noise
    ir = noise * envelope

    # Normalize IR peak to 1.0 (optional, affects overall loudness of reverb)
    max_abs_val = np.max(np.abs(ir))
    if max_abs_val > epsilon:
        ir /= max_abs_val
    else:
        # If IR is effectively zero, return Dirac delta
        logger.warning("Generated IR is near zero, returning Dirac delta.")
        return np.array([1.0], dtype=np.float64)

    # Optional: Ensure first sample represents direct sound (can be debated)
    # ir[0] = 1.0 # Uncomment if direct sound should always be max amplitude in IR

    return ir.astype(np.float64, copy=False)

def apply_reverb(
    y: NDArray[np.float64],
    sr: int,
    decay_time: float = 0.5,
    wet_level: float = 0.3,
    dry_level: float = 0.7,
    ir_seed: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Applies a simple convolution reverb to an audio signal using a generated IR.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate of `y`.
        decay_time: Approximate decay time (RT60) of the reverb tail in seconds (default: 0.5).
        wet_level: Gain level for the reverberated (wet) signal (0 to 1, default: 0.3).
        dry_level: Gain level for the original (dry) signal (0 to 1, default: 0.7).
        ir_seed: Optional random seed for generating the impulse response, ensuring reproducibility.

    Returns:
        Audio time series with reverb applied (float64). Length will be `len(y) + len(ir) - 1`.

    Raises:
        ValueError: If input `y` is not 1D or if level/decay parameters are invalid.
        Exception: For errors during IR generation or convolution.

    Example:
        >>> sr = 22050
        >>> signal = np.random.randn(sr) # 1 second of noise
        >>> reverberated_signal = apply_reverb(signal, sr, decay_time=0.8, wet_level=0.4)
        >>> print(len(reverberated_signal) > len(signal)) # Output is longer
        True
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    if not 0.0 <= wet_level <= 1.0:
        raise ValueError("wet_level must be between 0.0 and 1.0.")
    if not 0.0 <= dry_level <= 1.0:
        raise ValueError("dry_level must be between 0.0 and 1.0.")
    # decay_time validation happens in _generate_basic_ir

    logger.info(f"Applying reverb: decay={decay_time}s, wet={wet_level}, dry={dry_level}")

    try:
        # Generate impulse response
        ir = _generate_basic_ir(sr, decay_time, seed=ir_seed)
    except Exception as e:
        logger.error(f"Failed to generate impulse response: {e}")
        raise

    # If IR is just a Dirac delta (decay_time=0), reverb has no effect other than scaling
    if len(ir) == 1 and np.isclose(ir[0], 1.0):
        logger.debug("IR is Dirac delta, applying only dry/wet scaling.")
        # In this specific case, wet_signal = y, so output = dry*y + wet*y
        return ((dry_level + wet_level) * y).astype(np.float64, copy=False)

    try:
        # Apply convolution using FFT for efficiency
        # 'full' mode gives the complete convolution result, including the tail
        logger.debug(f"Convolving signal (len {len(y)}) with IR (len {len(ir)})...")
        wet_signal = fftconvolve(y, ir, mode='full')
        logger.debug(f"Convolution complete. Wet signal length: {len(wet_signal)}")
    except Exception as e:
        logger.error(f"Error during FFT convolution for reverb: {e}")
        raise

    # Mix dry and wet signals
    # Pad dry signal to match the length of the wet signal
    output_len = len(wet_signal)
    # Create padding tuple: (before, after)
    padding = (0, output_len - len(y))
    dry_signal_padded = np.pad(y, padding, mode='constant', constant_values=0)

    output_signal = (dry_level * dry_signal_padded) + (wet_level * wet_signal)

    # Optional: Normalize output to prevent clipping if levels sum > 1 or convolution increases peak
    # max_abs_out = np.max(np.abs(output_signal))
    # if max_abs_out > 1.0:
    #     logger.warning(f"Reverb output peak ({max_abs_out:.2f}) exceeds 1.0. Consider reducing levels or normalizing.")
    #     # output_signal /= max_abs_out # Simple peak normalization

    return output_signal.astype(np.float64, copy=False)
