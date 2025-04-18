# sygnals/core/transforms.py

"""
Implementations of various signal transforms like Wavelet and Hilbert.
FFT/STFT/CQT are primarily handled in dsp.py.
MFCC is handled in features/cepstral.py.
"""

import logging
import numpy as np
import pywt
from numpy.typing import NDArray
from typing import List, Tuple, Optional
from scipy.signal import hilbert # For Hilbert transform

logger = logging.getLogger(__name__)

# --- Wavelet Transform (Discrete) ---

def discrete_wavelet_transform(
    data: NDArray[np.float64],
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = 'symmetric'
) -> List[NDArray[np.float64]]:
    """
    Performs multi-level Discrete Wavelet Transform (DWT) using PyWavelets.

    Args:
        data: Input signal (1D float64).
        wavelet: Name of the wavelet to use (e.g., 'db4', 'haar', 'sym5').
                 See pywt.wavelist() for available wavelets.
        level: Decomposition level. If None, computes the maximum possible level
               (pywt.dwt_max_level).
        mode: Signal extension mode used for convolution (e.g., 'symmetric', 'zero', 'periodic').
              Default: 'symmetric'.

    Returns:
        List of coefficient arrays ordered from coarsest approximation (cA)
        to finest detail (cD1): [cA_n, cD_n, cD_n-1, ..., cD1].
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    if level is None:
        level = pywt.dwt_max_level(data_len=len(data), filter_len=pywt.Wavelet(wavelet).dec_len)
        logger.debug(f"Calculated max DWT level: {level}")
    elif level < 1:
        raise ValueError("Decomposition level must be >= 1.")

    logger.debug(f"Performing DWT: wavelet={wavelet}, level={level}, mode={mode}")
    try:
        coeffs = pywt.wavedec(data, wavelet, level=level, mode=mode)
        # Ensure all coefficients are float64
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
    Reconstructs a signal from its DWT coefficients using PyWavelets.

    Args:
        coeffs: List of DWT coefficient arrays ([cA_n, cD_n, ..., cD1]).
        wavelet: Name of the wavelet used for decomposition.
        mode: Signal extension mode used for reconstruction. Must match decomposition mode.

    Returns:
        Reconstructed signal (1D float64).
    """
    logger.debug(f"Performing inverse DWT: wavelet={wavelet}, mode={mode}")
    try:
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

    The analytic signal `xa(t)` of signal `x(t)` is `xa(t) = x(t) + j * xh(t)`,
    where `xh(t)` is the Hilbert transform of `x(t)`.

    Useful for obtaining instantaneous amplitude (envelope) and phase.

    Args:
        data: Input signal (1D float64).

    Returns:
        The complex analytic signal (complex128).
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    logger.debug("Computing Hilbert Transform to get analytic signal.")
    try:
        analytic_signal = hilbert(data)
        return analytic_signal.astype(np.complex128, copy=False)
    except Exception as e:
        logger.error(f"Error computing Hilbert Transform: {e}")
        raise


# --- Laplace Transform (Numerical - Keep basic version or remove?) ---
# This is less standard for typical audio/signal DSP toolkits compared to FFT/Wavelet.
# Keeping the basic numerical version for now, but mark as potentially less robust.

def laplace_transform_numerical(
    data: NDArray[np.float64],
    s_values: NDArray[np.complex128], # s is complex: sigma + j*omega
    t_step: float = 1.0 # Time step between data samples
) -> NDArray[np.complex128]:
    """
    Compute Laplace Transform numerically using simple summation.
    Note: This is a basic numerical approximation and may have limitations.

    L{f(t)}(s) = integral from 0 to inf of [ f(t) * exp(-st) ] dt
    Approximation: sum [ data[n] * exp(-s * n * t_step) ] * t_step

    Args:
        data: Input time-domain signal samples (1D float64).
        s_values: Array of complex 's' values at which to compute the transform.
        t_step: Time step between samples in `data`. Assumes data starts at t=0.

    Returns:
        Numerical Laplace Transform result for each s in s_values (complex128).
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if s_values.ndim != 1: raise ValueError("s_values must be 1D.")
    logger.debug(f"Computing numerical Laplace Transform for {len(s_values)} s-values.")

    n = np.arange(len(data))
    t = n * t_step
    transform = np.zeros(len(s_values), dtype=np.complex128)

    try:
        for i, s in enumerate(s_values):
            integrand = data * np.exp(-s * t)
            transform[i] = np.sum(integrand) * t_step # Simple rectangular integration
        return transform
    except Exception as e:
        logger.error(f"Error computing numerical Laplace Transform: {e}")
        raise

# Inverse Laplace is generally much harder numerically and often requires specific methods
# (e.g., Bromwich integral, Post's inversion formula, Talbot's method).
# The previous implementation was likely incorrect/oversimplified.
# Removing inverse_laplace_transform for now as a robust implementation is complex.

# def inverse_laplace_transform(...):
#    pass # Requires advanced numerical methods


# Ensure __all__ reflects available functions if needed later
# __all__ = [
#     "discrete_wavelet_transform",
#     "inverse_discrete_wavelet_transform",
#     "hilbert_transform",
#     "laplace_transform_numerical",
# ]
