# sygnals/core/features/frequency_domain.py

"""
Functions for extracting frequency-domain features from the spectrum of signal frames.

These features describe the distribution and characteristics of energy across frequencies.
Most functions here operate on the magnitude spectrum of a single time frame,
except where noted (e.g., spectral_contrast operates on a spectrogram).
"""

import logging
import numpy as np
import librosa # Used for spectral_contrast and potentially reference calculations
from numpy.typing import NDArray
# Import necessary types
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Define a small epsilon for safe division and log calculations
_EPSILON = np.finfo(np.float64).eps # Use machine epsilon for float64


def spectral_centroid(
    magnitude_spectrum: NDArray[np.float64],
    frequencies: NDArray[np.float64]
) -> np.float64:
    """
    Calculates the spectral centroid for a single frame's magnitude spectrum.

    The spectral centroid is the weighted mean of the frequencies present in the
    signal, weighted by their magnitude. It indicates where the 'center of mass'
    of the spectrum is located and is related to the perception of brightness of a sound.

    Formula: centroid = sum(frequencies[k] * magnitude_spectrum[k]) / sum(magnitude_spectrum[k])

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).
                            Shape: (n_fft // 2 + 1,).
        frequencies: Frequencies corresponding to the spectrum bins (Hz).
                     Shape: (n_fft // 2 + 1,).

    Returns:
        Spectral centroid frequency (float64, Hz). Returns 0.0 if the spectrum sum is negligible.

    Raises:
        ValueError: If input shapes mismatch.

    Example:
        >>> freqs = np.array([0, 100, 200, 300], dtype=float)
        >>> mags = np.array([0, 0.5, 1.0, 0.5], dtype=float) # Peak at 200 Hz
        >>> spectral_centroid(mags, freqs)
        200.0
    """
    if magnitude_spectrum.shape != frequencies.shape:
        raise ValueError(f"Spectrum shape {magnitude_spectrum.shape} and frequencies shape {frequencies.shape} must match.")
    if magnitude_spectrum.size == 0:
        return np.float64(0.0) # Handle empty input

    # Ensure magnitudes are non-negative
    if np.any(magnitude_spectrum < 0):
        logger.warning("Input magnitude_spectrum contains negative values. Using absolute values.")
        magnitude_spectrum = np.abs(magnitude_spectrum)

    # Use epsilon to avoid division by zero if spectrum is all zeros
    spectrum_sum = np.sum(magnitude_spectrum)

    if spectrum_sum < _EPSILON:
        logger.debug("Spectrum sum is near zero, returning centroid 0.0")
        return np.float64(0.0)

    # centroid = sum(freq * mag) / sum(mag)
    weighted_sum = np.sum(frequencies * magnitude_spectrum)
    return np.float64(weighted_sum / spectrum_sum) # Ensure float64 output

def spectral_bandwidth(
    magnitude_spectrum: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    centroid: Optional[np.float64] = None,
    p: int = 2 # Order of the p-norm (2 corresponds to standard deviation)
) -> np.float64:
    """
    Calculates the spectral bandwidth for a single frame's magnitude spectrum.

    The p-th order spectral bandwidth measures how the spectrum is spread around
    its centroid. For p=2, it represents the frequency standard deviation, weighted
    by magnitude.

    Formula (p=2): bandwidth = sqrt( sum(magnitude[k] * (frequencies[k] - centroid)^2) / sum(magnitude[k]) )

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).
                            Shape: (n_fft // 2 + 1,).
        frequencies: Frequencies corresponding to the spectrum bins (Hz).
                     Shape: (n_fft // 2 + 1,).
        centroid: Pre-calculated spectral centroid (Hz). If None, it will be calculated internally.
                  Providing it avoids redundant calculation.
        p: Order of the norm (default: 2). p=1 gives mean absolute deviation.

    Returns:
        Spectral bandwidth (float64, Hz). Returns 0.0 if the spectrum sum is negligible.

    Raises:
        ValueError: If input shapes mismatch or p <= 0.

    Example:
        >>> freqs = np.array([100, 200, 300], dtype=float)
        >>> mags = np.array([1.0, 0.1, 1.0], dtype=float) # Energy at 100 and 300 Hz
        >>> spectral_bandwidth(mags, freqs, p=2)
        # Expected bandwidth around 100 Hz (std dev)
        99.999...
    """
    if magnitude_spectrum.shape != frequencies.shape:
        raise ValueError(f"Spectrum shape {magnitude_spectrum.shape} and frequencies shape {frequencies.shape} must match.")
    if p <= 0:
        raise ValueError("Order 'p' for spectral bandwidth must be positive.")
    if magnitude_spectrum.size == 0:
        return np.float64(0.0) # Handle empty input

    # Ensure magnitudes are non-negative
    if np.any(magnitude_spectrum < 0):
        logger.warning("Input magnitude_spectrum contains negative values. Using absolute values.")
        magnitude_spectrum = np.abs(magnitude_spectrum)

    spectrum_sum = np.sum(magnitude_spectrum)

    if spectrum_sum < _EPSILON:
        logger.debug("Spectrum sum is near zero, returning bandwidth 0.0")
        return np.float64(0.0)

    # Calculate centroid if not provided
    if centroid is None:
        centroid = spectral_centroid(magnitude_spectrum, frequencies)

    # bandwidth = [ sum( mag * |freq - centroid|^p ) / sum(mag) ] ^ (1/p)
    deviation = np.abs(frequencies - centroid)**p
    weighted_deviation_sum = np.sum(magnitude_spectrum * deviation)

    # Handle potential precision issues if weighted_deviation_sum is negative (shouldn't happen)
    if weighted_deviation_sum < 0:
        logger.warning(f"Weighted deviation sum is negative ({weighted_deviation_sum:.2e}), possibly due to precision. Clamping to 0.")
        weighted_deviation_sum = 0.0

    bandwidth = (weighted_deviation_sum / spectrum_sum)**(1.0 / p)
    return np.float64(bandwidth) # Ensure float64 output

def spectral_contrast(
    S: NDArray[np.float64], # Note: Operates on Spectrogram, not single frame spectrum
    sr: int,
    n_bands: int = 6,
    fmin: float = 200.0,
    freqs: Optional[NDArray[np.float64]] = None, # Frequencies for S rows
    **kwargs: Any # Additional args for librosa.feature.spectral_contrast
) -> NDArray[np.float64]:
    """
    Calculates spectral contrast using librosa (operates on a full spectrogram).

    Spectral contrast considers the spectral peak-to-valley ratio within several
    frequency bands (typically octave-scaled). Higher contrast values often
    correlate with clearer tonal content or presence of distinct spectral peaks.

    Args:
        S: Magnitude or power spectrogram (frequency x time). Shape (n_freq_bins, n_frames).
           Should be non-negative.
        sr: Sampling rate (Hz).
        n_bands: Number of frequency bands to compute contrast for (default: 6).
        fmin: Minimum frequency (Hz) for the lowest band edge (default: 200.0 Hz).
        freqs: Optional array of frequencies corresponding to the rows of S.
               If None, they are estimated using `librosa.fft_frequencies` based on
               the number of rows in S (assuming standard FFT binning).
        **kwargs: Additional keyword arguments passed to `librosa.feature.spectral_contrast`
                  (e.g., `linear`).

    Returns:
        Spectral contrast for each band and time frame (shape: (n_bands + 1, n_frames)).
        The last row (`contrast[n_bands, :]`) represents the difference between the
        mean of the peaks and valleys across all bands. Values are float64.

    Raises:
        ValueError: If input S is not 2D.
        Exception: For errors during librosa calculation.

    Example:
        >>> sr = 22050
        >>> y = librosa.chirp(fmin=100, fmax=5000, sr=sr, duration=2)
        >>> S_mag = np.abs(librosa.stft(y))
        >>> contrast = spectral_contrast(S=S_mag, sr=sr, n_bands=6)
        >>> print(contrast.shape)
        (7, 173) # (6 bands + delta, num_frames)
    """
    if S.ndim != 2:
        raise ValueError("Input S must be a 2D spectrogram (frequency x time).")
    if np.any(S < 0):
        logger.warning("Input spectrogram S contains negative values. Using absolute values.")
        S = np.abs(S)

    logger.debug(f"Calculating Spectral Contrast: n_bands={n_bands}, fmin={fmin}")
    try:
        # Librosa's spectral_contrast function handles the calculation
        contrast = librosa.feature.spectral_contrast(
            S=S,
            sr=sr,
            n_bands=n_bands,
            fmin=fmin,
            freq=freqs, # Pass frequencies if available
            **kwargs
        )
        # Ensure output type
        return contrast.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error calculating Spectral Contrast: {e}")
        raise

def spectral_flatness(
    magnitude_spectrum: NDArray[np.float64]
) -> np.float64:
    """
    Calculates the spectral flatness (Wiener entropy) for a single frame's spectrum.

    Spectral flatness is the ratio of the geometric mean to the arithmetic mean
    of the spectrum. It indicates how flat (noise-like) or peaky (tonal) the
    spectrum is.
    - Value close to 1.0: Flat spectrum (like white noise).
    - Value close to 0.0: Peaky spectrum (like a sine wave).

    Formula: flatness = geometric_mean(spectrum) / arithmetic_mean(spectrum)

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).
                            Shape: (n_fft // 2 + 1,).

    Returns:
        Spectral flatness value (float64, between 0.0 and 1.0). Returns 0.0 if frame is empty.

    Example:
        >>> flat_spectrum = np.ones(100)
        >>> peak_spectrum = np.zeros(100); peak_spectrum[10] = 1.0
        >>> spectral_flatness(flat_spectrum)
        1.0
        >>> spectral_flatness(peak_spectrum)
        # Expected close to 0.0
    """
    if magnitude_spectrum.size == 0:
        return np.float64(0.0) # Handle empty input

    # Ensure magnitudes are non-negative
    if np.any(magnitude_spectrum < 0):
         logger.warning("Magnitude spectrum contains negative values. Using absolute values.")
         magnitude_spectrum = np.abs(magnitude_spectrum)

    # Add small epsilon to avoid log(0) and division by zero issues
    mag_spec_eps = magnitude_spectrum + _EPSILON

    # Geometric Mean = exp( mean( log(spectrum) ) )
    # Use log of the epsilon-added spectrum
    log_spectrum = np.log(mag_spec_eps)
    geometric_mean = np.exp(np.mean(log_spectrum))

    # Arithmetic Mean = mean( spectrum )
    # Use the original spectrum for arithmetic mean if possible, fallback to eps version
    arithmetic_mean = np.mean(magnitude_spectrum)

    # Handle case where arithmetic mean is close to zero
    if arithmetic_mean < _EPSILON:
        logger.debug("Arithmetic mean of spectrum is near zero, returning flatness 0.0")
        return np.float64(0.0)

    # Flatness = Geometric Mean / Arithmetic Mean
    flatness = geometric_mean / arithmetic_mean
    # Ensure the result is within the expected [0, 1] range due to potential precision issues
    return np.clip(np.float64(flatness), 0.0, 1.0)


def spectral_rolloff(
    magnitude_spectrum: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    roll_percent: float = 0.85 # Percentage of total spectral energy
) -> np.float64:
    """
    Calculates the spectral rolloff frequency for a single frame's spectrum.

    The spectral rolloff is the frequency below which a specified percentage
    (`roll_percent`) of the total spectral *energy* (based on power spectrum)
    is contained. It's often used as a measure of the spectral shape's skewness
    towards lower or higher frequencies.

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).
                            Shape: (n_fft // 2 + 1,).
        frequencies: Frequencies corresponding to the spectrum bins (Hz).
                     Shape: (n_fft // 2 + 1,).
        roll_percent: Rolloff percentage (0.0 to 1.0). Default: 0.85 (85%).

    Returns:
        Spectral rolloff frequency (float64, Hz). Returns the maximum frequency if
        the spectrum sum is negligible or frame is empty.

    Raises:
        ValueError: If input shapes mismatch or roll_percent is invalid.

    Example:
        >>> freqs = np.array([0, 100, 200, 300, 400], dtype=float)
        >>> mags = np.array([0, 1, 1, 0, 0], dtype=float) # Energy concentrated at 100, 200 Hz
        >>> spectral_rolloff(mags, freqs, roll_percent=0.85)
        # Should be 200 Hz, as 100% of energy is below or at 200 Hz
        200.0
        >>> spectral_rolloff(mags, freqs, roll_percent=0.40)
        # Should be 100 Hz, as 50% of energy is at or below 100 Hz
        100.0
    """
    if magnitude_spectrum.shape != frequencies.shape:
        raise ValueError(f"Spectrum shape {magnitude_spectrum.shape} and frequencies shape {frequencies.shape} must match.")
    if not 0.0 <= roll_percent <= 1.0:
        raise ValueError("roll_percent must be between 0.0 and 1.0.")
    if magnitude_spectrum.size == 0:
        return np.float64(0.0) # Handle empty input, return 0 Hz? Or max freq? Max makes more sense.
        # return frequencies[-1] if frequencies.size > 0 else np.float64(0.0) # Return max freq if possible

    # Ensure magnitudes are non-negative
    if np.any(magnitude_spectrum < 0):
        logger.warning("Input magnitude_spectrum contains negative values. Using absolute values.")
        magnitude_spectrum = np.abs(magnitude_spectrum)

    # Use power spectrum (magnitude squared) for energy calculation
    power_spectrum = magnitude_spectrum**2
    total_energy = np.sum(power_spectrum)

    if total_energy < _EPSILON:
        logger.debug("Total energy is near zero, returning max frequency as rolloff.")
        # Return the highest frequency bin available
        return frequencies[-1] if frequencies.size > 0 else np.float64(0.0)

    # Cumulative energy
    cumulative_energy = np.cumsum(power_spectrum)
    threshold = roll_percent * total_energy

    # Find the frequency bin where cumulative energy meets or exceeds the threshold
    # np.searchsorted finds the insertion point, which corresponds to the first index >= threshold
    rolloff_indices = np.where(cumulative_energy >= threshold)[0]

    if rolloff_indices.size == 0:
        # This case should be unlikely if total_energy > 0, but handle defensively
        logger.warning("Could not find rolloff point, returning max frequency.")
        return frequencies[-1]

    rolloff_index = rolloff_indices[0]

    # Ensure index is within bounds (should be guaranteed by logic above)
    # rolloff_index = min(rolloff_index, len(frequencies) - 1)

    return np.float64(frequencies[rolloff_index]) # Ensure float64 output


def dominant_frequency(
     magnitude_spectrum: NDArray[np.float64],
     frequencies: NDArray[np.float64]
) -> np.float64:
    """
    Finds the frequency with the maximum magnitude in the spectrum for a single frame.

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame.
                            Shape: (n_fft // 2 + 1,).
        frequencies: Frequencies corresponding to the spectrum bins (Hz).
                     Shape: (n_fft // 2 + 1,).

    Returns:
        The dominant frequency (float64, Hz). Returns 0.0 if spectrum is empty.

    Raises:
        ValueError: If input shapes mismatch.

    Example:
        >>> freqs = np.array([0, 100, 200, 300], dtype=float)
        >>> mags = np.array([0, 0.5, 1.0, 0.5], dtype=float) # Peak at 200 Hz
        >>> dominant_frequency(mags, freqs)
        200.0
    """
    if magnitude_spectrum.shape != frequencies.shape:
        raise ValueError(f"Spectrum shape {magnitude_spectrum.shape} and frequencies shape {frequencies.shape} must match.")
    if magnitude_spectrum.size == 0:
        logger.debug("Spectrum is empty, returning dominant frequency 0.0")
        return np.float64(0.0)

    dominant_index = np.argmax(magnitude_spectrum)
    return np.float64(frequencies[dominant_index]) # Ensure float64 output


# Dictionary mapping feature names to functions for the manager
# These functions require magnitude_spectrum and frequencies as input (per frame)
# Note: spectral_contrast is excluded here as it requires the full spectrogram S
FREQUENCY_DOMAIN_FEATURES: Dict[str, Any] = {
    "spectral_centroid": spectral_centroid,
    "spectral_bandwidth": spectral_bandwidth,
    "spectral_flatness": spectral_flatness,
    "spectral_rolloff": spectral_rolloff,
    "dominant_frequency": dominant_frequency,
    # Add other frequency-domain features here if needed
}
