# sygnals/core/audio/effects/utility.py

"""
Utility audio effects, including gain adjustment, noise reduction,
transient shaping, and stereo widening.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Literal, Tuple
import warnings # Import the warnings module
import librosa # Used for HPSS, STFT, ISTFT

logger = logging.getLogger(__name__)

# Define a small epsilon for safe division and log calculations
_EPSILON = np.finfo(np.float64).eps

# --- Gain Adjustment ---

def adjust_gain(
    y: NDArray[np.float64],
    gain_db: float
) -> NDArray[np.float64]:
    """
    Adjusts the gain (amplitude) of an audio signal by a specified amount in decibels (dB).

    Args:
        y: Input audio time series (1D or multi-channel float64).
        gain_db: Gain adjustment in decibels. Positive values amplify, negative values attenuate.

    Returns:
        Audio time series with gain adjusted (float64).

    Raises:
        ValueError: If input `y` is not a NumPy array.
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("Input audio data must be a NumPy array.")

    logger.info(f"Adjusting gain by {gain_db:.2f} dB.")

    # Convert dB gain to linear amplitude multiplier
    multiplier = 10.0**(gain_db / 20.0)

    # Apply gain
    y_adjusted = y * multiplier

    # Optional: Check for clipping (handled elsewhere, e.g., during save)
    # max_abs_out = np.max(np.abs(y_adjusted))
    # if max_abs_out > 1.0:
    #     logger.warning(...)

    return y_adjusted.astype(np.float64, copy=False)

# --- Noise Reduction (Spectral Subtraction - Basic) ---

def noise_reduction_spectral(
    y: NDArray[np.float64],
    sr: int,
    noise_profile_duration: float = 0.5,
    reduction_amount: float = 1.0,
    n_fft: int = 2048,
    hop_length: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Applies basic noise reduction using spectral subtraction.

    NOTE: This is a very basic implementation and may introduce audible artifacts
          ("musical noise"). More advanced techniques exist. It assumes the
          initial part of the signal contains only noise.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate (Hz).
        noise_profile_duration: Duration (in seconds) at the beginning of the signal
                                assumed to be noise, used to build the noise profile. (Default: 0.5)
        reduction_amount: Factor controlling how much noise to subtract (0.0 to ~2.0).
                          1.0 is standard subtraction. > 1.0 is aggressive. (Default: 1.0)
        n_fft: FFT window size for STFT. (Default: 2048)
        hop_length: Hop length for STFT. Defaults to n_fft // 4.

    Returns:
        Noise-reduced audio time series (float64).

    Raises:
        ValueError: If input `y` is not 1D or parameters are invalid.
    """
    if y.ndim != 1:
        raise ValueError("Input audio 'y' must be 1D for spectral noise reduction.")
    if noise_profile_duration <= 0 or noise_profile_duration * sr > len(y):
        raise ValueError("Invalid noise_profile_duration.")
    if reduction_amount < 0:
        raise ValueError("reduction_amount must be non-negative.")

    logger.info(f"Applying spectral noise reduction: profile_dur={noise_profile_duration}s, reduction={reduction_amount}")

    hop_length_calc = hop_length if hop_length is not None else n_fft // 4

    # 1. Estimate Noise Profile from initial segment
    noise_samples = int(noise_profile_duration * sr)
    noise_segment = y[:noise_samples]
    # Compute STFT of the noise segment
    noise_stft = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_length_calc)
    # Calculate average noise power spectrum across frames
    noise_power_spectrum = np.mean(np.abs(noise_stft)**2, axis=1)

    # 2. Compute STFT of the full signal
    signal_stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length_calc)
    signal_magnitude = np.abs(signal_stft)
    signal_phase = np.angle(signal_stft)
    signal_power_spectrum = signal_magnitude**2

    # 3. Subtract noise power (spectral subtraction)
    # Ensure noise profile matches dimensions for broadcasting
    noise_power_spectrum_broadcast = noise_power_spectrum[:, np.newaxis]
    # Subtract scaled noise power, floor at zero (power cannot be negative)
    denoised_power_spectrum = np.maximum(
        0,
        signal_power_spectrum - reduction_amount * noise_power_spectrum_broadcast
    )

    # 4. Reconstruct magnitude and combine with original phase
    denoised_magnitude = np.sqrt(denoised_power_spectrum)
    denoised_stft = denoised_magnitude * np.exp(1j * signal_phase)

    # 5. Inverse STFT to get time-domain signal
    y_denoised = librosa.istft(denoised_stft, hop_length=hop_length_calc, length=len(y))

    return y_denoised.astype(np.float64, copy=False)


# --- Transient Shaping (HPSS - Basic) ---

def transient_shaping_hpss(
    y: NDArray[np.float64],
    sr: int,
    percussive_scale: float = 1.0,
    harmonic_margin: Union[float, Tuple[float, ...]] = 1.0,
    percussive_margin: Union[float, Tuple[float, ...]] = 1.0
) -> NDArray[np.float64]:
    """
    Performs basic transient shaping using Harmonic-Percussive Source Separation (HPSS).

    Separates the signal into harmonic and percussive components using librosa.hpss,
    allows scaling the percussive component, and then recombines them.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate (Hz).
        percussive_scale: Factor to scale the percussive component before recombination.
                          > 1.0 enhances transients, < 1.0 reduces transients. (Default: 1.0)
        harmonic_margin: Margin parameter for HPSS separation (controls harmonic smoothness).
                         See librosa.effects.hpss documentation. (Default: 1.0)
        percussive_margin: Margin parameter for HPSS separation (controls percussive sparsity).
                           See librosa.effects.hpss documentation. (Default: 1.0)

    Returns:
        Audio time series with transients shaped (float64).

    Raises:
        ValueError: If input `y` is not 1D.
    """
    if y.ndim != 1:
        raise ValueError("Input audio 'y' must be 1D for HPSS transient shaping.")

    logger.info(f"Applying transient shaping (HPSS): percussive_scale={percussive_scale}")

    # 1. Perform HPSS
    # Note: HPSS can be computationally intensive
    y_harmonic, y_percussive = librosa.effects.hpss(
        y,
        margin=(harmonic_margin, percussive_margin)
    )

    # 2. Scale the percussive component
    y_percussive_scaled = y_percussive * percussive_scale

    # 3. Recombine
    y_shaped = y_harmonic + y_percussive_scaled

    return y_shaped.astype(np.float64, copy=False)


# --- Stereo Widening (Mid/Side - Basic) ---

def stereo_widening_midside(
    y: NDArray[np.float64],
    width_factor: float = 1.5
) -> NDArray[np.float64]:
    """
    Applies basic stereo widening using Mid/Side processing.

    Enhances stereo width by amplifying the 'Side' channel (difference signal)
    relative to the 'Mid' channel (sum signal).

    Args:
        y: Input stereo audio time series (float64 NumPy array).
           MUST have shape (2, n_samples) - i.e., 2 channels.
        width_factor: Factor controlling the width.
                      1.0 = original width.
                      > 1.0 increases width (amplifies Side channel).
                      < 1.0 decreases width (attenuates Side channel).
                      0.0 results in mono (Mid channel only).
                      (Default: 1.5)

    Returns:
        Stereo audio time series with adjusted width (float64, shape (2, n_samples)).

    Raises:
        ValueError: If input `y` is not a 2-channel NumPy array or width_factor is negative.
    """
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[0] != 2:
        raise ValueError("Input audio 'y' must be a 2-channel NumPy array with shape (2, n_samples) for stereo widening.")
    if width_factor < 0:
        raise ValueError("width_factor must be non-negative.")

    logger.info(f"Applying Mid/Side stereo widening: width_factor={width_factor}")

    # Extract Left and Right channels
    left = y[0, :]
    right = y[1, :]

    # 1. Convert to Mid/Side
    # Mid = (Left + Right) / 2  (Average to avoid potential clipping)
    # Side = (Left - Right) / 2 (Difference, also scaled)
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    # 2. Adjust Side channel gain based on width_factor
    # Scaling factor for Side: sqrt(2 * width_factor) is sometimes used,
    # but simpler to just scale Side directly by width_factor for basic effect.
    # Let's scale the difference component.
    # If width_factor = 1, side_scaled = side.
    # If width_factor = 0, side_scaled = 0 (mono).
    # If width_factor = 2, side_scaled = 2 * side (wider).
    side_scaled = side * width_factor

    # 3. Convert back to Left/Right
    # Left = Mid + Side_scaled
    # Right = Mid - Side_scaled
    left_new = mid + side_scaled
    right_new = mid - side_scaled

    # Combine back into (2, n_samples) array
    y_widened = np.stack([left_new, right_new], axis=0)

    # Optional: Clipping or normalization if needed, but usually handled later
    # max_abs = np.max(np.abs(y_widened))
    # if max_abs > 1.0: logger.warning(...)

    return y_widened.astype(np.float64, copy=False)
