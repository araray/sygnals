# sygnals/core/audio/effects/reverb.py

"""
Implementation of a basic audio reverb effect.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve # Use FFT-based convolution for efficiency
from typing import Tuple, Optional, Union, Literal, Dict

logger = logging.getLogger(__name__)

def _generate_basic_ir(sr: int, decay_time: float = 0.5, seed: Optional[int] = None) -> NDArray[np.float64]:
    """
    Generates a very basic impulse response (IR) for reverb simulation.
    Uses exponentially decaying white noise.

    Args:
        sr: Sampling rate (Hz).
        decay_time: Approximate time (in seconds) for the reverb tail to decay significantly (e.g., by 60 dB).
        seed: Optional random seed for noise generation reproducibility.

    Returns:
        Impulse response (1D NumPy array, float64).
    """
    rng = np.random.default_rng(seed)
    ir_length_samples = int(sr * decay_time * 1.5) # Make IR slightly longer than decay time
    if ir_length_samples <= 0:
        return np.array([1.0], dtype=np.float64) # Return dirac delta if decay_time is non-positive

    # Generate white noise
    noise = rng.standard_normal(ir_length_samples).astype(np.float64)

    # Create exponential decay envelope
    # decay_factor determines how quickly it decays. e.g., decay to 1/1000 (-60dB) in decay_time
    decay_factor = -np.log(0.001) / (decay_time * sr) if decay_time > 0 else 0
    envelope = np.exp(-decay_factor * np.arange(ir_length_samples))

    # Apply envelope to noise
    ir = noise * envelope

    # Normalize IR (optional, affects loudness) - simple peak normalization here
    max_abs = np.max(np.abs(ir))
    if max_abs > 1e-6:
        ir /= max_abs

    # Ensure first sample is 1.0 for the direct sound (optional, depends on desired effect)
    # ir[0] = 1.0 # Uncomment to force direct sound

    return ir

def apply_reverb(
    y: NDArray[np.float64],
    sr: int,
    decay_time: float = 0.5,
    wet_level: float = 0.3,
    dry_level: float = 0.7,
    ir_seed: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Applies a simple convolution reverb to an audio signal.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate of `y`.
        decay_time: Approximate decay time of the reverb tail in seconds (default: 0.5).
        wet_level: Gain level for the reverberated (wet) signal (0 to 1, default: 0.3).
        dry_level: Gain level for the original (dry) signal (0 to 1, default: 0.7).
        ir_seed: Optional random seed for generating the impulse response.

    Returns:
        Audio time series with reverb applied (float64). Length will be longer than input
        due to convolution with the IR tail.
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    if not 0.0 <= wet_level <= 1.0:
        raise ValueError("wet_level must be between 0.0 and 1.0.")
    if not 0.0 <= dry_level <= 1.0:
        raise ValueError("dry_level must be between 0.0 and 1.0.")

    logger.debug(f"Applying reverb: decay={decay_time}s, wet={wet_level}, dry={dry_level}")

    # Generate impulse response
    ir = _generate_basic_ir(sr, decay_time, seed=ir_seed)

    if ir.size <= 1 and wet_level > 0:
         logger.warning("Generated impulse response is too short. Reverb effect might be minimal.")
         # If IR is just [1.], wet signal is same as dry signal
         return (dry_level * y + wet_level * y).astype(np.float64, copy=False)

    # Apply convolution using FFT for efficiency
    # 'full' mode gives the complete convolution result, including the tail
    wet_signal = fftconvolve(y, ir, mode='full')

    # Mix dry and wet signals
    # Pad dry signal to match the length of the wet signal
    output_len = len(wet_signal)
    dry_signal_padded = np.pad(y, (0, output_len - len(y)), mode='constant')

    output_signal = (dry_level * dry_signal_padded) + (wet_level * wet_signal)

    # Normalize output (optional, prevents potential clipping if levels sum > 1)
    # max_abs_out = np.max(np.abs(output_signal))
    # if max_abs_out > 1.0:
    #     logger.warning(f"Reverb output peak ({max_abs_out:.2f}) exceeds 1.0. Consider reducing levels.")
        # output_signal /= max_abs_out # Simple peak normalization

    return output_signal.astype(np.float64, copy=False)
