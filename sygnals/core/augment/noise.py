# sygnals/core/augment/noise.py

"""
Noise-based data augmentation techniques.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Literal
import warnings

logger = logging.getLogger(__name__)

# Define a small epsilon for safe division and log calculations
_EPSILON = np.finfo(np.float64).eps # Use machine epsilon for float64


def add_noise(
    y: NDArray[np.float64],
    snr_db: float,
    noise_type: Literal['gaussian', 'white', 'pink', 'brown'] = 'gaussian',
    seed: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Adds noise to an audio signal at a specified Signal-to-Noise Ratio (SNR) in dB.

    This is a common data augmentation technique for improving model robustness.

    Args:
        y: Input clean audio time series (1D float64).
        snr_db: Desired Signal-to-Noise Ratio in decibels. Lower values mean more noise.
        noise_type: Type of noise to add ('gaussian'/'white', 'pink', 'brown').
                    Note: 'pink' and 'brown' noise generation are placeholders.
        seed: Optional random seed for noise generation reproducibility.

    Returns:
        Audio time series with noise added (float64).

    Raises:
        ValueError: If input `y` is not 1D or noise_type is invalid.
        NotImplementedError: If 'pink' or 'brown' noise is requested (currently placeholders).

    Example:
        >>> sr = 22050
        >>> signal = librosa.tone(440, sr=sr, duration=1)
        >>> noisy_signal = add_noise(signal, snr_db=10.0, noise_type='gaussian')
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array for noise addition.")

    logger.info(f"Applying noise augmentation: type={noise_type}, SNR={snr_db:.2f} dB.")

    rng = np.random.default_rng(seed)
    n_samples = len(y)

    # Generate noise based on type
    if noise_type in ['gaussian', 'white']:
        noise = rng.standard_normal(n_samples).astype(np.float64)
    elif noise_type == 'pink':
        # Placeholder for pink noise generation
        warnings.warn("Pink noise generation is currently a placeholder (using white noise).", UserWarning, stacklevel=2)
        logger.warning("Pink noise generation is currently a placeholder (using white noise).")
        noise = rng.standard_normal(n_samples).astype(np.float64) # Using white noise as placeholder
        # raise NotImplementedError("Pink noise generation is not yet implemented.")
    elif noise_type == 'brown':
        # Placeholder for brown noise (Brownian/red noise) generation
        warnings.warn("Brown noise generation is currently a placeholder (using white noise).", UserWarning, stacklevel=2)
        logger.warning("Brown noise generation is currently a placeholder (using white noise).")
        noise = rng.standard_normal(n_samples).astype(np.float64) # Using white noise as placeholder
        # raise NotImplementedError("Brown noise generation is not yet implemented.")
    else:
        raise ValueError(f"Invalid noise_type: '{noise_type}'. Choose 'gaussian', 'white', 'pink', or 'brown'.")

    # Calculate signal power and noise power
    signal_power = np.mean(y**2)
    noise_power = np.mean(noise**2)

    # Avoid division by zero if signal or noise power is zero
    if signal_power < _EPSILON:
        logger.warning("Signal power is near zero. Cannot reliably scale noise based on SNR. Returning original signal.")
        return y.astype(np.float64, copy=False)
    if noise_power < _EPSILON:
        logger.warning("Generated noise power is near zero. Cannot scale noise. Returning original signal.")
        return y.astype(np.float64, copy=False)


    # Calculate required noise scaling factor based on SNR
    # SNR_db = 10 * log10(signal_power / noise_power_scaled)
    # noise_power_scaled = signal_power / (10**(SNR_db / 10))
    # scaling_factor^2 * noise_power = noise_power_scaled
    # scaling_factor = sqrt(noise_power_scaled / noise_power)
    snr_linear = 10.0**(snr_db / 10.0)
    required_noise_power_scaled = signal_power / snr_linear
    scaling_factor = np.sqrt(required_noise_power_scaled / noise_power)

    # Scale noise and add to signal
    noise_scaled = noise * scaling_factor
    y_noisy = y + noise_scaled

    return y_noisy.astype(np.float64, copy=False)
