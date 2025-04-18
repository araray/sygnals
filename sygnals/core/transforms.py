# sygnals/core/transforms.py

"""
Implementations of various signal transforms like Discrete Wavelet Transform (DWT)
and Hilbert Transform.
FFT/STFT/CQT are primarily handled in dsp.py.
MFCC is handled in features/cepstral.py.
"""

import logging
import numpy as np
import pywt # For Wavelet Transforms
from numpy.typing import NDArray
# Import necessary types
from typing import List, Tuple, Optional
from scipy.signal import hilbert # For Hilbert transform

logger = logging.getLogger(__name__)

# --- Wavelet Transform (Discrete) ---

def discrete_wavelet_transform(
    data: NDArray[np.float64],
    wavelet: str = "db4", # Daubechies 4 is a common default
    level: Optional[int] = None,
    mode: str = 'symmetric'
) -> List[NDArray[np.float64]]:
    """
    Performs multi-level 1D Discrete Wavelet Transform (DWT) using PyWavelets.

    The DWT decomposes a signal into approximation (cA) and detail (cD) coefficients
    at different levels (scales).

    Args:
        data: Input signal (1D NumPy array, float64).
        wavelet: Name of the wavelet to use (e.g., 'db4', 'haar', 'sym5', 'bior3.7').
                 See `pywt.wavelist()` for available discrete wavelets.
        level: Decomposition level (integer >= 1). If None, computes the maximum
               possible level based on signal length and wavelet filter length
               using `pywt.dwt_max_level`.
        mode: Signal extension mode used for convolution at the boundaries.
              See `pywt.Modes`. Default: 'symmetric'.

    Returns:
        List of coefficient arrays ordered from coarsest approximation (cA_n)
        to finest detail (cD1): [cA_n, cD_n, cD_n-1, ..., cD1].
        All arrays in the list are float64.

    Raises:
        ValueError: If level < 1 or data is not 1D.
        Exception: For errors during PyWavelets processing.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    # Determine decomposition level if not specified
    if level is None:
        try:
            # Calculate max level based on data length and wavelet filter length
            filter_len = pywt.Wavelet(wavelet).dec_len
            level = pywt.dwt_max_level(data_len=len(data), filter_len=filter_len)
            logger.debug(f"Calculated max DWT level for wavelet '{wavelet}': {level}")
            # Ensure level is at least 1, even if max_level calculates 0 for very short signals
            level = max(1, level)
        except Exception as e:
            logger.error(f"Could not determine max DWT level for wavelet '{wavelet}': {e}")
            raise ValueError(f"Invalid wavelet name '{wavelet}' or error calculating max level.") from e
    elif not isinstance(level, int) or level < 1:
        raise ValueError(f"Decomposition level must be an integer >= 1, got {level}.")

    logger.debug(f"Performing DWT: wavelet={wavelet}, level={level}, mode={mode}")
    try:
        # Perform the multi-level decomposition
        coeffs = pywt.wavedec(data, wavelet, level=level, mode=mode)
        # Ensure all coefficient arrays are float64
        return [c.astype(np.float64, copy=False) for c in coeffs]
    except Exception as e:
        logger.error(f"Error during Discrete Wavelet Transform: {e}")
        raise

def inverse_discrete_wavelet_transform(
    coeffs: List[NDArray[np.float64]],
    wavelet: str,
    mode: str = 'symmetric'
) -> NDArray[np.float64]:
    """
    Reconstructs a signal from its DWT coefficients using PyWavelets (`waverec`).

    Args:
        coeffs: List of DWT coefficient arrays in the format returned by
                `discrete_wavelet_transform`: [cA_n, cD_n, ..., cD1].
        wavelet: Name of the wavelet used for the original decomposition. Must match.
        mode: Signal extension mode used during reconstruction. Should match the
              mode used during decomposition. Default: 'symmetric'.

    Returns:
        Reconstructed signal (1D NumPy array, float64). Length might differ slightly
        from the original signal depending on the wavelet, level, and mode.

    Raises:
        ValueError: If the coefficients format is incorrect.
        Exception: For errors during PyWavelets processing.
    """
    if not isinstance(coeffs, list) or len(coeffs) < 2:
         raise ValueError("Input 'coeffs' must be a list containing at least cA and cD coefficients.")

    logger.debug(f"Performing inverse DWT: wavelet={wavelet}, mode={mode}, levels={len(coeffs)-1}")
    try:
        # Perform multi-level reconstruction
        reconstructed_signal = pywt.waverec(coeffs, wavelet, mode=mode)
        # Ensure output is float64
        return reconstructed_signal.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error during Inverse Discrete Wavelet Transform: {e}")
        raise

# --- Hilbert Transform ---

def hilbert_transform(
    data: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """
    Computes the analytic signal using the Hilbert transform (scipy.signal.hilbert).

    The analytic signal `xa(t)` of a real signal `x(t)` is defined as:
    `xa(t) = x(t) + j * xh(t)`
    where `xh(t)` is the Hilbert transform of `x(t)` (essentially, `x(t)` with a -90 degree phase shift).

    The magnitude `abs(xa(t))` gives the instantaneous amplitude (envelope).
    The angle `angle(xa(t))` gives the instantaneous phase.

    Args:
        data: Input signal (1D NumPy array, float64).

    Returns:
        The complex analytic signal (1D NumPy array, complex128).

    Raises:
        ValueError: If input data is not 1D.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    logger.debug("Computing Hilbert Transform to get analytic signal.")
    try:
        # Compute the analytic signal using scipy.signal.hilbert
        analytic_signal = hilbert(data)
        # Ensure output type is complex128
        return analytic_signal.astype(np.complex128, copy=False)
    except Exception as e:
        logger.error(f"Error computing Hilbert Transform: {e}")
        raise


# --- Laplace Transform (Numerical - Basic Implementation) ---
# Note: This is a very basic numerical approximation and may have limitations
# in accuracy and convergence depending on the signal and 's' values.
# Inverse Laplace transform is significantly more complex and is omitted here.

def laplace_transform_numerical(
    data: NDArray[np.float64],
    s_values: NDArray[np.complex128], # s is complex: sigma + j*omega
    t_step: float = 1.0 # Time step between data samples
) -> NDArray[np.complex128]:
    """
    Compute Laplace Transform numerically using simple rectangular integration.

    Definition: L{f(t)}(s) = integral from 0 to inf of [ f(t) * exp(-st) ] dt
    Approximation: sum_{n=0}^{N-1} [ data[n] * exp(-s * n * t_step) ] * t_step

    Args:
        data: Input time-domain signal samples (1D float64). Assumed f(t=0) = data[0].
        s_values: Array of complex 's' values (sigma + j*omega) at which to compute the transform.
        t_step: Time step between samples in `data` (e.g., 1/fs).

    Returns:
        Numerical Laplace Transform result for each s in s_values (1D NumPy array, complex128).

    Raises:
        ValueError: If inputs are not 1D.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if s_values.ndim != 1: raise ValueError("s_values must be 1D.")
    logger.debug(f"Computing numerical Laplace Transform for {len(s_values)} s-values using {len(data)} data points with t_step={t_step}.")

    n = np.arange(len(data)) # Sample indices
    t = n * t_step          # Time corresponding to each sample
    transform_results = np.zeros(len(s_values), dtype=np.complex128)

    try:
        # Iterate through each complex 's' value
        for i, s in enumerate(s_values):
            # Calculate the integrand f(t) * exp(-st) for all time points t
            integrand = data * np.exp(-s * t)
            # Approximate the integral using sum * dt (rectangular rule)
            transform_results[i] = np.sum(integrand) * t_step
        return transform_results
    except Exception as e:
        logger.error(f"Error computing numerical Laplace Transform: {e}")
        raise

# __all__ can be defined if specific functions need to be exported when using 'from .transforms import *'
# __all__ = [
#     "discrete_wavelet_transform",
#     "inverse_discrete_wavelet_transform",
#     "hilbert_transform",
#     "laplace_transform_numerical",
# ]
