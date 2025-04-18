# sygnals/core/features/frequency_domain.py

"""
Functions for extracting frequency-domain features from the spectrum of signal frames.
These features describe the distribution and characteristics of energy across frequencies.
Most functions here operate on the magnitude spectrum of a single time frame,
except where noted (e.g., spectral_contrast operates on a spectrogram).
"""

import logging
import numpy as np
import librosa # Used for spectral_contrast
from numpy.typing import NDArray
from scipy.stats import skew, kurtosis # Although not used here, kept for potential future additions
# Import necessary types
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Note: These functions typically operate on the magnitude spectrum of a single frame

def spectral_centroid(
    magnitude_spectrum: NDArray[np.float64],
    frequencies: NDArray[np.float64]
) -> np.float64:
    """
    Calculates the spectral centroid for a single frame.

    The spectral centroid is the weighted mean of the frequencies present in the
    signal, weighted by their magnitude. It indicates where the 'center of mass'
    of the spectrum is located and is related to the perception of brightness of a sound.

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).
                            Shape: (n_fft // 2 + 1,).
        frequencies: Frequencies corresponding to the spectrum bins (Hz).
                     Shape: (n_fft // 2 + 1,).

    Returns:
        Spectral centroid frequency (float64, Hz). Returns 0.0 if the spectrum sum is negligible.
    """
    if magnitude_spectrum.shape != frequencies.shape:
        raise ValueError(f"Spectrum shape {magnitude_spectrum.shape} and frequencies shape {frequencies.shape} must match.")
    if np.any(magnitude_spectrum < 0):
        logger.warning("Input magnitude_spectrum contains negative values. Using absolute values.")
        magnitude_spectrum = np.abs(magnitude_spectrum)

    # Use epsilon to avoid division by zero if spectrum is all zeros
    epsilon = 1e-10
    spectrum_sum = np.sum(magnitude_spectrum)

    if spectrum_sum < epsilon:
        logger.debug("Spectrum sum is near zero, returning centroid 0.0")
        return 0.0

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
    Calculates the spectral bandwidth for a single frame.

    The p-th order spectral bandwidth measures how the spectrum is spread around
    its centroid. For p=2, it represents the spectral standard deviation.

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).
                            Shape: (n_fft // 2 + 1,).
        frequencies: Frequencies corresponding to the spectrum bins (Hz).
                     Shape: (n_fft // 2 + 1,).
        centroid: Pre-calculated spectral centroid (Hz). If None, it will be calculated internally.
        p: Order of the norm (default: 2). p=1 gives mean absolute deviation.

    Returns:
        Spectral bandwidth (float64, Hz). Returns 0.0 if the spectrum sum is negligible.
    """
    if magnitude_spectrum.shape != frequencies.shape:
        raise ValueError(f"Spectrum shape {magnitude_spectrum.shape} and frequencies shape {frequencies.shape} must match.")
    if np.any(magnitude_spectrum < 0):
        logger.warning("Input magnitude_spectrum contains negative values. Using absolute values.")
        magnitude_spectrum = np.abs(magnitude_spectrum)

    epsilon = 1e-10
    spectrum_sum = np.sum(magnitude_spectrum)

    if spectrum_sum < epsilon:
        logger.debug("Spectrum sum is near zero, returning bandwidth 0.0")
        return 0.0

    # Calculate centroid if not provided
    if centroid is None:
        centroid = spectral_centroid(magnitude_spectrum, frequencies)

    # bandwidth = [ sum( mag * |freq - centroid|^p ) / sum(mag) ] ^ (1/p)
    deviation = np.abs(frequencies - centroid)**p
    weighted_deviation_sum = np.sum(magnitude_spectrum * deviation)

    # Handle potential precision issues if weighted_deviation_sum is extremely small
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
    freqs: Optional[NDArray[np.float64]] = None # Frequencies for S rows
) -> NDArray[np.float64]:
    """
    Calculates spectral contrast using librosa (operates on a full spectrogram).

    Spectral contrast considers the spectral peak-to-valley ratio within several
    frequency bands. Higher contrast values often correlate with clearer tonal content.

    Args:
        S: Magnitude or power spectrogram (frequency x time). Shape (n_freq_bins, n_frames).
        sr: Sampling rate (Hz).
        n_bands: Number of frequency bands to compute contrast for (default: 6).
        fmin: Minimum frequency (Hz) for the lowest band edge (default: 200.0 Hz).
        freqs: Optional array of frequencies corresponding to the rows of S.
               If None, they are estimated using `librosa.fft_frequencies`.

    Returns:
        Spectral contrast for each band and time frame (shape: (n_bands + 1, n_frames)).
        The last row represents the difference between the mean of the peaks and valleys
        across all bands. Values are float64.
    """
    if S.ndim != 2:
        raise ValueError("Input S must be a 2D spectrogram (frequency x time).")

    logger.debug(f"Calculating Spectral Contrast: n_bands={n_bands}, fmin={fmin}")
    try:
        # Librosa's spectral_contrast function handles the calculation
        contrast = librosa.feature.spectral_contrast(
            S=S,
            sr=sr,
            n_bands=n_bands,
            fmin=fmin,
            freq=freqs # Pass frequencies if available
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
    Calculates the spectral flatness (Wiener entropy) for a single frame.

    Spectral flatness is the ratio of the geometric mean to the arithmetic mean
    of the spectrum. It indicates how flat (noise-like) or peaky (tonal) the
    spectrum is.
    - Value close to 1.0: Flat spectrum (like white noise).
    - Value close to 0.0: Peaky spectrum (like a sine wave).

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).
                            Shape: (n_fft // 2 + 1,).

    Returns:
        Spectral flatness value (float64, between 0.0 and 1.0).
    """
    if np.any(magnitude_spectrum < 0):
         logger.warning("Magnitude spectrum contains negative values. Using absolute values.")
         magnitude_spectrum = np.abs(magnitude_spectrum)

    # Add small epsilon to avoid log(0) and division by zero issues
    epsilon = 1e-10
    mag_spec_eps = magnitude_spectrum + epsilon

    # Geometric Mean = exp( mean( log(spectrum) ) )
    geometric_mean = np.exp(np.mean(np.log(mag_spec_eps)))
    # Arithmetic Mean = mean( spectrum )
    arithmetic_mean = np.mean(mag_spec_eps)

    # Handle case where arithmetic mean is close to zero
    if arithmetic_mean < epsilon:
        logger.debug("Arithmetic mean of spectrum is near zero, returning flatness 0.0")
        return 0.0

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
    Calculates the spectral rolloff frequency for a single frame.

    The spectral rolloff is the frequency below which a specified percentage
    (roll_percent) of the total spectral energy is contained. It's often used
    as a measure of the spectral shape's skewness towards lower or higher frequencies.

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).
                            Shape: (n_fft // 2 + 1,).
        frequencies: Frequencies corresponding to the spectrum bins (Hz).
                     Shape: (n_fft // 2 + 1,).
        roll_percent: Rolloff percentage (0.0 to 1.0). Default: 0.85 (85%).

    Returns:
        Spectral rolloff frequency (float64, Hz). Returns the maximum frequency if
        the spectrum sum is negligible.
    """
    if magnitude_spectrum.shape != frequencies.shape:
        raise ValueError(f"Spectrum shape {magnitude_spectrum.shape} and frequencies shape {frequencies.shape} must match.")
    if not 0.0 <= roll_percent <= 1.0:
        raise ValueError("roll_percent must be between 0.0 and 1.0.")
    if np.any(magnitude_spectrum < 0):
        logger.warning("Input magnitude_spectrum contains negative values. Using absolute values.")
        magnitude_spectrum = np.abs(magnitude_spectrum)

    # Use power spectrum (magnitude squared) for energy calculation
    power_spectrum = magnitude_spectrum**2
    total_energy = np.sum(power_spectrum)
    epsilon = 1e-10

    if total_energy < epsilon:
        logger.debug("Total energy is near zero, returning max frequency as rolloff.")
        return frequencies[-1] if frequencies.size > 0 else 0.0

    # Cumulative energy
    cumulative_energy = np.cumsum(power_spectrum)
    threshold = roll_percent * total_energy

    # Find the frequency bin where cumulative energy meets or exceeds the threshold
    # np.searchsorted finds the insertion point, which corresponds to the first index >= threshold
    rolloff_index = np.searchsorted(cumulative_energy, threshold, side='left')

    # Ensure index is within bounds
    rolloff_index = min(rolloff_index, len(frequencies) - 1)

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
        The dominant frequency (float64, Hz). Returns 0.0 if spectrum is empty or all zero.
    """
    if magnitude_spectrum.shape != frequencies.shape:
        raise ValueError(f"Spectrum shape {magnitude_spectrum.shape} and frequencies shape {frequencies.shape} must match.")
    if magnitude_spectrum.size == 0:
        logger.debug("Spectrum is empty, returning dominant frequency 0.0")
        return 0.0

    dominant_index = np.argmax(magnitude_spectrum)
    return np.float64(frequencies[dominant_index]) # Ensure float64 output


# Dictionary mapping feature names to functions for the manager
# These functions require magnitude_spectrum and frequencies as input (per frame)
FREQUENCY_DOMAIN_FEATURES: Dict[str, Any] = {
    "spectral_centroid": spectral_centroid,
    "spectral_bandwidth": spectral_bandwidth,
    # "spectral_contrast": spectral_contrast, # Needs full spectrogram S, handled separately by manager
    "spectral_flatness": spectral_flatness,
    "spectral_rolloff": spectral_rolloff,
    "dominant_frequency": dominant_frequency,
    # Add other frequency-domain features here if needed
}
