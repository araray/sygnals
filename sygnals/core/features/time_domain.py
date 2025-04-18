# sygnals/core/features/time_domain.py

"""
Functions for extracting time-domain features from signals.
Operates on frames (windows) of the signal.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from scipy.stats import skew, kurtosis, entropy

logger = logging.getLogger(__name__)

# Note: RMS and ZCR are often considered audio features, implemented in audio.features

def mean_amplitude(frame: NDArray[np.float64]) -> np.float64:
    """Calculates the mean of the absolute amplitude within a frame."""
    return np.mean(np.abs(frame))

def std_dev_amplitude(frame: NDArray[np.float64]) -> np.float64:
    """Calculates the standard deviation of the amplitude within a frame."""
    return np.std(frame)

def skewness(frame: NDArray[np.float64]) -> np.float64:
    """Calculates the skewness of the amplitude distribution within a frame."""
    # Handle potential zero variance case which results in NaN skewness
    if np.var(frame) < 1e-10:
        return 0.0
    return skew(frame)

def kurtosis_val(frame: NDArray[np.float64]) -> np.float64:
    """Calculates the kurtosis (Fisher's definition) of the amplitude distribution."""
    # Handle potential zero variance case
    if np.var(frame) < 1e-10:
        return 0.0 # Or perhaps -3 depending on definition preference
    return kurtosis(frame, fisher=True) # Fisher=True gives excess kurtosis (0 for normal dist)

def peak_amplitude(frame: NDArray[np.float64]) -> np.float64:
    """Calculates the maximum absolute amplitude within a frame."""
    return np.max(np.abs(frame))

def crest_factor(frame: NDArray[np.float64]) -> np.float64:
    """
    Calculates the crest factor (peak amplitude / RMS amplitude).
    Indicates the ratio of peak values to the effective value.
    """
    peak = peak_amplitude(frame)
    rms = np.sqrt(np.mean(frame**2))
    if rms < 1e-10: # Avoid division by zero
        return 0.0 if peak < 1e-10 else np.inf
    return peak / rms

def signal_entropy(frame: NDArray[np.float64], num_bins: int = 10) -> np.float64:
    """
    Calculates the entropy of the signal amplitude distribution within the frame.
    Discretizes the signal into bins first.
    """
    if frame.size == 0:
        return 0.0
    # Create histogram
    hist, bin_edges = np.histogram(frame, bins=num_bins, density=True)
    # Calculate entropy using scipy.stats.entropy
    # Requires probabilities, density=True gives PDF estimate
    # Need to multiply by bin width to approximate probability mass if bins are uniform
    bin_width = bin_edges[1] - bin_edges[0]
    pk = hist * bin_width
    # Ensure probabilities sum close to 1 (adjust if needed due to edge effects)
    pk = pk[pk > 0] # Remove zero probabilities for entropy calculation
    if pk.size == 0:
        return 0.0
    # Normalize if sum is not close to 1 (can happen with density=True)
    if not np.isclose(np.sum(pk), 1.0):
         pk = pk / np.sum(pk)

    return entropy(pk)

# Dictionary mapping feature names to functions for the manager
TIME_DOMAIN_FEATURES = {
    "mean_amplitude": mean_amplitude,
    "std_dev_amplitude": std_dev_amplitude,
    "skewness": skewness,
    "kurtosis": kurtosis_val,
    "peak_amplitude": peak_amplitude,
    "crest_factor": crest_factor,
    "signal_entropy": signal_entropy,
    # RMS and ZCR are in audio.features but could be registered here if needed generically
}
