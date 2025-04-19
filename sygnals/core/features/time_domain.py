# sygnals/core/features/time_domain.py

"""
Functions for extracting time-domain features from signals.

These features are typically calculated on short frames (windows) of the signal
and describe characteristics of the amplitude distribution and signal shape
within that frame.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from scipy.stats import skew, kurtosis, entropy
# Import necessary types
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Define a small epsilon for safe division and log calculations
_EPSILON = np.finfo(np.float64).eps # Use machine epsilon for float64

def mean_amplitude(frame: NDArray[np.float64]) -> np.float64:
    """
    Calculates the mean of the absolute amplitude values within a frame.

    This provides a measure of the average signal level within the frame,
    ignoring the sign.

    Args:
        frame: Input signal frame (1D NumPy array, float64).

    Returns:
        Mean absolute amplitude (float64). Returns 0.0 if the frame is empty.

    Example:
        >>> frame = np.array([-0.5, 0.0, 0.5, 1.0])
        >>> mean_amplitude(frame)
        0.5
    """
    if frame.size == 0:
        return np.float64(0.0)
    return np.mean(np.abs(frame))

def std_dev_amplitude(frame: NDArray[np.float64]) -> np.float64:
    """
    Calculates the standard deviation of the amplitude values within a frame.

    This measures the spread or variation of the signal's amplitude around its mean
    within the frame.

    Args:
        frame: Input signal frame (1D NumPy array, float64).

    Returns:
        Standard deviation of amplitude (float64). Returns 0.0 if the frame is empty.

    Example:
        >>> frame = np.array([1.0, 1.0, -1.0, -1.0]) # Mean is 0
        >>> std_dev_amplitude(frame)
        1.0
    """
    if frame.size == 0:
        return np.float64(0.0)
    return np.std(frame)

def skewness(frame: NDArray[np.float64]) -> np.float64:
    """
    Calculates the skewness of the amplitude distribution within a frame.

    Skewness measures the asymmetry of the probability distribution of amplitude values.
    - Positive skew: Tail on the right side is longer/fatter.
    - Negative skew: Tail on the left side is longer/fatter.
    - Zero skew: Symmetric distribution (like Gaussian).

    Uses `scipy.stats.skew`.

    Args:
        frame: Input signal frame (1D NumPy array, float64).

    Returns:
        Skewness value (float64). Returns 0.0 if the frame has zero variance or is empty.

    Example:
        >>> frame = np.array([1, 1, 1, 5]) # Positively skewed
        >>> skewness(frame)
        1.7320508103043323
    """
    if frame.size < 2: # Skewness requires at least 2 points
        return np.float64(0.0)
    # Handle potential zero variance case which results in NaN skewness
    if np.var(frame) < _EPSILON:
        return np.float64(0.0)
    # bias=False provides the unbiased estimator
    return np.float64(skew(frame, bias=False))

def kurtosis_val(frame: NDArray[np.float64]) -> np.float64:
    """
    Calculates the kurtosis (Fisher's definition) of the amplitude distribution.

    Kurtosis measures the "tailedness" or "peakedness" of the distribution.
    Fisher's definition (excess kurtosis) is relative to a normal distribution:
    - Positive kurtosis: Heavier tails, more peaked than normal.
    - Negative kurtosis: Lighter tails, less peaked than normal.
    - Zero kurtosis: Normal distribution kurtosis.

    Uses `scipy.stats.kurtosis`.

    Args:
        frame: Input signal frame (1D NumPy array, float64).

    Returns:
        Excess kurtosis value (float64). Returns 0.0 if the frame has zero variance or is empty.

    Example:
        >>> frame = np.random.normal(0, 1, 1000) # Normal distribution
        >>> kurtosis_val(frame)
        # Should be close to 0.0
    """
    if frame.size < 4: # Kurtosis requires at least 4 points for unbiased estimate
        return np.float64(0.0)
    # Handle potential zero variance case
    if np.var(frame) < _EPSILON:
        return np.float64(0.0) # Kurtosis is ill-defined for constant data
    # fisher=True gives excess kurtosis, bias=False for unbiased estimator
    return np.float64(kurtosis(frame, fisher=True, bias=False))

def peak_amplitude(frame: NDArray[np.float64]) -> np.float64:
    """
    Calculates the maximum absolute amplitude value within a frame.

    Indicates the highest signal excursion from zero within the frame.

    Args:
        frame: Input signal frame (1D NumPy array, float64).

    Returns:
        Peak absolute amplitude (float64). Returns 0.0 if the frame is empty.

    Example:
        >>> frame = np.array([-1.2, 0.5, 1.0, -0.8])
        >>> peak_amplitude(frame)
        1.2
    """
    if frame.size == 0:
        return np.float64(0.0)
    return np.max(np.abs(frame))

def crest_factor(frame: NDArray[np.float64]) -> np.float64:
    """
    Calculates the crest factor (peak amplitude / RMS amplitude).

    Indicates the ratio of peak values to the effective (RMS) value of the signal
    within the frame. Higher values indicate more prominent peaks relative to the
    average power.

    Args:
        frame: Input signal frame (1D NumPy array, float64).

    Returns:
        Crest factor (float64). Returns 0.0 if the frame is empty or all zeros.
        Returns positive infinity if RMS is zero but peak is non-zero (unlikely for real signals).

    Example:
        >>> sine_frame = np.sin(np.linspace(0, 2*np.pi, 100)) # Amplitude 1
        >>> crest_factor(sine_frame)
        # Should be close to sqrt(2) (~1.414)
        >>> square_wave = np.concatenate([np.ones(50), -np.ones(50)])
        >>> crest_factor(square_wave)
        # Should be close to 1.0
    """
    if frame.size == 0:
        return np.float64(0.0)

    peak = peak_amplitude(frame)
    # Calculate RMS energy safely
    rms = np.sqrt(np.mean(frame**2))

    if rms < _EPSILON:
        # If RMS is effectively zero, crest factor is 0 if peak is also zero,
        # otherwise it's infinite (or very large). Return 0 for practical purposes.
        return np.float64(0.0)
    else:
        return np.float64(peak / rms)

def signal_entropy(frame: NDArray[np.float64], num_bins: int = 10) -> np.float64:
    """
    Calculates the entropy of the signal amplitude distribution within the frame.

    Discretizes the signal amplitude into `num_bins` and computes the Shannon entropy
    of the resulting probability distribution. Higher entropy indicates a more
    uniform or unpredictable distribution of amplitude values.

    Uses `scipy.stats.entropy`.

    Args:
        frame: Input signal frame (1D NumPy array, float64).
        num_bins: Number of bins to use for discretizing the amplitude distribution. (Default: 10)

    Returns:
        Entropy value (float64, non-negative). Returns 0.0 if the frame is empty or constant.

    Example:
        >>> constant_frame = np.ones(100)
        >>> noise_frame = np.random.rand(100)
        >>> signal_entropy(constant_frame)
        0.0
        >>> signal_entropy(noise_frame)
        # Should be > 0, likely close to log(num_bins) for uniform noise
    """
    if frame.size < 2 or num_bins < 1: # Need at least 2 points for histogram, 1 bin
        return np.float64(0.0)

    # Handle constant frame case directly (entropy is 0)
    if np.all(frame == frame[0]):
        return np.float64(0.0)

    # Create histogram to estimate probability distribution
    # Use density=False to get counts, then normalize to get probabilities
    counts, bin_edges = np.histogram(frame, bins=num_bins, density=False)

    # Calculate probabilities (normalize counts)
    # Filter out zero counts before calculating entropy
    pk = counts[counts > 0] / frame.size

    # Calculate entropy using scipy.stats.entropy (base e, natural logarithm)
    return np.float64(entropy(pk))

# Dictionary mapping feature names to functions for the manager
# These functions expect a single frame (1D NumPy array) as input.
TIME_DOMAIN_FEATURES: Dict[str, Any] = {
    "mean_amplitude": mean_amplitude,
    "std_dev_amplitude": std_dev_amplitude,
    "skewness": skewness,
    "kurtosis": kurtosis_val,
    "peak_amplitude": peak_amplitude,
    "crest_factor": crest_factor,
    "signal_entropy": signal_entropy,
    # Note: RMS and ZCR are typically handled in audio.features as they often
    # use librosa's frame-based calculation directly on the whole signal.
}
