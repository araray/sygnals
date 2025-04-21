# sygnals/core/audio/effects/tremolo.py

"""
Implementation of the audio tremolo effect.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from scipy.signal import sawtooth # Used for triangle wave generation
# Import necessary types
from typing import List, Tuple, Dict, Optional, Union, Literal, Any

logger = logging.getLogger(__name__)

def _generate_lfo(
    sr: int,
    num_samples: int,
    rate: float,
    shape: Literal['sine', 'triangle', 'square']
) -> NDArray[np.float64]:
    """
    Generates a Low-Frequency Oscillator (LFO) signal.

    Args:
        sr: Sampling rate (Hz).
        num_samples: Number of samples for the LFO signal.
        rate: LFO frequency in Hz.
        shape: LFO waveform shape ('sine', 'triangle', 'square').

    Returns:
        LFO signal ranging from 0 to 1.
    """
    t = np.arange(num_samples) / sr # Time vector
    lfo_phase = 2 * np.pi * rate * t

    if shape == 'sine':
        # Generate sine wave from -1 to 1, then scale to 0 to 1
        lfo = (np.sin(lfo_phase) + 1.0) / 2.0
    elif shape == 'triangle':
        # sawtooth generates -1 to 1, width=0.5 gives triangle. Scale to 0 to 1.
        lfo = (sawtooth(lfo_phase, width=0.5) + 1.0) / 2.0
    elif shape == 'square':
        # Generate square wave (-1 or 1), then scale to 0 to 1
        lfo = (np.sign(np.sin(lfo_phase)) + 1.0) / 2.0
        # Handle potential exact zeros from sign function if needed, though unlikely with float phase
        lfo[lfo == 0.5] = 0.0 # Map potential intermediate value to 0
    else:
        # Should not happen due to prior validation, but handle defensively
        raise ValueError(f"Internal error: Unexpected LFO shape '{shape}'")

    return lfo.astype(np.float64, copy=False)


def apply_tremolo(
    y: NDArray[np.float64],
    sr: int,
    rate: float = 5.0, # LFO rate in Hz
    depth: float = 0.5, # Modulation depth (0 to 1)
    shape: Literal['sine', 'triangle', 'square'] = 'sine' # LFO waveform shape
) -> NDArray[np.float64]:
    """
    Applies a tremolo audio effect.

    Tremolo modulates the amplitude (volume) of the signal using a Low-Frequency
    Oscillator (LFO).

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate of `y`.
        rate: Rate of the LFO modulating the amplitude (Hz, default: 5.0). Must be positive.
        depth: Depth of the amplitude modulation (0.0 to 1.0, default: 0.5).
               0.0 means no effect, 1.0 means amplitude goes from 0 to original peak.
        shape: Waveform shape of the LFO ('sine', 'triangle', 'square', default: 'sine').

    Returns:
        The processed audio time series with tremolo applied (float64).

    Raises:
        ValueError: If input `y` is not 1D or parameters are invalid.
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("Tremolo depth must be between 0.0 and 1.0.")
    if rate <= 0:
        raise ValueError("Tremolo rate must be positive.")
    if shape not in ['sine', 'triangle', 'square']:
        raise ValueError("LFO shape must be 'sine', 'triangle', or 'square'.")

    logger.info(f"Applying Tremolo: rate={rate} Hz, depth={depth}, shape={shape}")

    # 1. Generate LFO signal ranging from 0 to 1
    num_samples = len(y)
    lfo = _generate_lfo(sr, num_samples, rate, shape)

    # 2. Scale LFO based on depth
    # Modulation signal should range from (1 - depth) to 1
    # lfo_scaled = 1.0 - depth * (1.0 - lfo) # Incorrect: This scales 0..1 to depth..1
    # Correct scaling: LFO (0..1) -> Modulator (1-depth..1)
    # modulator = lfo * depth + (1.0 - depth) # Incorrect: scales 0..1 to (1-depth)..(1)
    # Correct: Modulator = 1.0 - depth + lfo * depth
    # Alternative: Modulator = 1.0 - depth * (1.0 - lfo) # This is correct if LFO was 0..1
    # Let's use: Modulator = (1.0 - depth) + lfo * depth
    modulator = (1.0 - depth) + (lfo * depth)

    # 3. Apply amplitude modulation
    y_tremolo = y * modulator

    # Ensure output is float64
    return y_tremolo.astype(np.float64, copy=False)
