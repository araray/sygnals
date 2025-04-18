# sygnals/core/features/frequency_domain.py

"""
Functions for extracting frequency-domain features from the spectrum of signal frames.
Requires FFT computation results as input.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)

# Note: These functions typically operate on the magnitude spectrum of a single frame

def spectral_centroid(
    magnitude_spectrum: NDArray[np.float64],
    frequencies: NDArray[np.float64]
) -> np.float64:
    """
    Calculates the spectral centroid.
    The weighted mean of the frequencies present in the signal, weighted by their magnitude.
    Indicates where the 'center of mass' of the spectrum is.

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).
        frequencies: Frequencies corresponding to the spectrum bins.

    Returns:
        Spectral centroid frequency.
    """
    if magnitude_spectrum.size != frequencies.size:
        raise ValueError("Spectrum and frequencies must have the same size.")
    if np.sum(magnitude_spectrum) < 1e-10: # Avoid division by zero if spectrum is all zeros
        return 0.0
    # Ensure spectrum is non-negative
    mag_spec = np.maximum(0, magnitude_spectrum)
    # centroid = sum(freq * mag) / sum(mag)
    return np.sum(frequencies * mag_spec) / np.sum(mag_spec)

def spectral_bandwidth(
    magnitude_spectrum: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    centroid: Optional[np.float64] = None,
    p: int = 2 # Order of the p-norm (2 corresponds to standard deviation)
) -> np.float64:
    """
    Calculates the spectral bandwidth.
    The p-th order spectral bandwidth measures how the spectrum is spread around its centroid.
    p=2 gives the spectral standard deviation.

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).
        frequencies: Frequencies corresponding to the spectrum bins.
        centroid: Pre-calculated spectral centroid. If None, it will be calculated.
        p: Order of the norm (default: 2).

    Returns:
        Spectral bandwidth.
    """
    if magnitude_spectrum.size != frequencies.size:
        raise ValueError("Spectrum and frequencies must have the same size.")
    norm_const = np.sum(magnitude_spectrum)
    if norm_const < 1e-10:
        return 0.0

    if centroid is None:
        centroid = spectral_centroid(magnitude_spectrum, frequencies)

    # Ensure spectrum is non-negative
    mag_spec = np.maximum(0, magnitude_spectrum)

    # bandwidth = [ sum( mag * (freq - centroid)^p ) / sum(mag) ] ^ (1/p)
    deviation = np.abs(frequencies - centroid)**p
    weighted_deviation = np.sum(mag_spec * deviation)
    bandwidth = (weighted_deviation / norm_const)**(1.0 / p)
    return bandwidth

def spectral_contrast(
    S: NDArray[np.float64], # Magnitude or Power Spectrogram (freq x time)
    sr: int,
    n_bands: int = 6,
    fmin: float = 200.0,
    freqs: Optional[NDArray[np.float64]] = None # Frequencies for S rows
) -> NDArray[np.float64]:
    """
    Calculates spectral contrast using librosa.
    Considers the spectral peak-to-valley ratio within frequency bands.

    Args:
        S: Magnitude or power spectrogram (frequency x time).
        sr: Sampling rate.
        n_bands: Number of frequency bands to compute contrast for.
        fmin: Minimum frequency for the lowest band edge.
        freqs: Frequencies corresponding to rows of S. Calculated if None.

    Returns:
        Spectral contrast for each band (shape: (n_bands + 1, time)).
        The "+1" corresponds to the difference between max and min across all bands.
    """
    logger.debug(f"Calculating Spectral Contrast: n_bands={n_bands}, fmin={fmin}")
    try:
        # Librosa expects magnitude spectrogram S
        contrast = librosa.feature.spectral_contrast(
            S=S,
            sr=sr,
            n_bands=n_bands,
            fmin=fmin,
            freq=freqs
        )
        return contrast.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error calculating Spectral Contrast: {e}")
        raise

def spectral_flatness(
    magnitude_spectrum: NDArray[np.float64]
) -> np.float64:
    """
    Calculates the spectral flatness (Wiener entropy).
    Ratio of the geometric mean to the arithmetic mean of the spectrum.
    Indicates how flat (noise-like) or peaky (tonal) the spectrum is.
    Value close to 1 means flat (white noise), close to 0 means peaky (sine wave).

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).

    Returns:
        Spectral flatness value.
    """
    if np.any(magnitude_spectrum < 0):
         logger.warning("Magnitude spectrum contains negative values. Taking absolute value.")
         magnitude_spectrum = np.abs(magnitude_spectrum)

    # Add small epsilon to avoid log(0) and division by zero
    epsilon = 1e-10
    mag_spec_eps = magnitude_spectrum + epsilon

    geometric_mean = np.exp(np.mean(np.log(mag_spec_eps)))
    arithmetic_mean = np.mean(mag_spec_eps)

    # Flatness should be <= 1
    flatness = geometric_mean / arithmetic_mean
    return np.clip(flatness, 0.0, 1.0) # Ensure bounds


def spectral_rolloff(
    magnitude_spectrum: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    roll_percent: float = 0.85 # Percentage of total spectral energy
) -> np.float64:
    """
    Calculates the spectral rolloff frequency.
    The frequency below which a specified percentage (roll_percent) of the total
    spectral energy is contained.

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame (non-negative values).
        frequencies: Frequencies corresponding to the spectrum bins.
        roll_percent: Rolloff percentage (0 to 1). Default: 0.85.

    Returns:
        Spectral rolloff frequency.
    """
    if magnitude_spectrum.size != frequencies.size:
        raise ValueError("Spectrum and frequencies must have the same size.")
    if not 0.0 <= roll_percent <= 1.0:
        raise ValueError("roll_percent must be between 0.0 and 1.0.")

    # Ensure spectrum is non-negative
    mag_spec = np.maximum(0, magnitude_spectrum)
    total_energy = np.sum(mag_spec)

    if total_energy < 1e-10:
        return frequencies[-1] # Return max frequency if no energy

    # Cumulative energy
    cumulative_energy = np.cumsum(mag_spec)
    threshold = roll_percent * total_energy

    # Find the frequency bin where cumulative energy exceeds the threshold
    rolloff_indices = np.where(cumulative_energy >= threshold)[0]

    if rolloff_indices.size > 0:
        # Get the first index where the threshold is met or exceeded
        rolloff_index = rolloff_indices[0]
        return frequencies[rolloff_index]
    else:
        # Should not happen if total_energy > 0, but return max freq as fallback
        return frequencies[-1]


def dominant_frequency(
     magnitude_spectrum: NDArray[np.float64],
     frequencies: NDArray[np.float64]
) -> np.float64:
    """
    Finds the frequency with the maximum magnitude in the spectrum.

    Args:
        magnitude_spectrum: Magnitude spectrum of a single frame.
        frequencies: Frequencies corresponding to the spectrum bins.

    Returns:
        The dominant frequency. Returns 0.0 if spectrum is all zero.
    """
    if magnitude_spectrum.size != frequencies.size:
        raise ValueError("Spectrum and frequencies must have the same size.")
    if magnitude_spectrum.size == 0:
        return 0.0

    dominant_index = np.argmax(magnitude_spectrum)
    return frequencies[dominant_index]


# Dictionary mapping feature names to functions (requires spectrum and freqs)
# Note: Spectral contrast needs the full spectrogram, handled differently in manager
FREQUENCY_DOMAIN_FEATURES = {
    "spectral_centroid": spectral_centroid,
    "spectral_bandwidth": spectral_bandwidth,
    # "spectral_contrast": spectral_contrast, # Needs spectrogram S
    "spectral_flatness": spectral_flatness,
    "spectral_rolloff": spectral_rolloff,
    "dominant_frequency": dominant_frequency,
}
