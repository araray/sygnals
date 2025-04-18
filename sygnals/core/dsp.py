# sygnals/core/dsp.py

"""
Core Digital Signal Processing (DSP) functions, excluding specific filter implementations.
Includes FFT, convolution, windowing, etc.
"""

import logging
from typing import Tuple, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, ifft, fftfreq # Use scipy.fft
from scipy.signal import fftconvolve, get_window

logger = logging.getLogger(__name__)

# --- FFT-related functions ---

def compute_fft(
    data: NDArray[np.float_],
    fs: Union[int, float] = 1.0,
    n: Optional[int] = None,
    window: Optional[str] = "hann"
) -> Tuple[NDArray[np.float_], NDArray[np.complex_]]:
    """
    Computes the Fast Fourier Transform (FFT) of a real-valued signal.

    Args:
        data: Input time-domain signal (1D NumPy array).
        fs: Sampling frequency of the signal (default: 1.0 Hz).
        n: Length of the FFT. If None, uses the length of the data.
           If n > len(data), the data is zero-padded.
           If n < len(data), the data is truncated.
        window: Name of the window function to apply before FFT (e.g., 'hann', 'hamming').
                If None, no window is applied. See scipy.signal.get_window for options.

    Returns:
        A tuple containing:
        - freqs (NDArray[np.float_]): Array of frequencies corresponding to the FFT bins.
        - spectrum (NDArray[np.complex_]): Complex-valued FFT result.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    if window:
        logger.debug(f"Applying '{window}' window before FFT.")
        try:
            win = get_window(window, data.shape[0])
            data = data * win
        except ValueError as e:
            logger.warning(f"Could not apply window '{window}': {e}. Proceeding without window.")

    if n is None:
        n = data.shape[0]

    logger.debug(f"Computing FFT with N={n}, Fs={fs}")
    spectrum = fft(data, n=n)
    freqs = fftfreq(n, d=1/fs)

    return freqs, spectrum

def compute_ifft(
    spectrum: NDArray[np.complex_],
    n: Optional[int] = None
) -> NDArray[np.float_]:
    """
    Computes the Inverse Fast Fourier Transform (IFFT).

    Assumes the input spectrum corresponds to a real-valued time-domain signal.

    Args:
        spectrum: Complex-valued frequency spectrum.
        n: Length of the inverse FFT. If None, uses the length of the spectrum.
           Should typically match the original FFT length.

    Returns:
        Real-valued time-domain signal.
    """
    if spectrum.ndim != 1:
        raise ValueError("Input spectrum must be a 1D array.")

    if n is None:
        n = spectrum.shape[0]

    logger.debug(f"Computing IFFT with N={n}")
    # Use np.real to discard negligible imaginary parts due to numerical precision
    time_domain_signal = np.real(ifft(spectrum, n=n))

    return time_domain_signal

# --- Convolution-related functions ---

def apply_convolution(
    data: NDArray[np.float_],
    kernel: NDArray[np.float_],
    mode: str = "same"
) -> NDArray[np.float_]:
    """
    Applies 1D convolution using the Fast Fourier Transform method.

    Args:
        data: Input signal (1D NumPy array).
        kernel: The convolution kernel (1D NumPy array).
        mode: Convolution mode ('full', 'valid', 'same'). See scipy.signal.fftconvolve.
              Default is 'same', returning output of the same size as 'data'.

    Returns:
        The result of the convolution.
    """
    if data.ndim != 1 or kernel.ndim != 1:
        raise ValueError("Input data and kernel must be 1D arrays.")

    logger.debug(f"Applying convolution with kernel size {kernel.shape[0]}, mode='{mode}'")
    return fftconvolve(data, kernel, mode=mode)

# --- Window functions ---

def apply_window(
    data: NDArray[np.float_],
    window_type: str = "hann"
) -> NDArray[np.float_]:
    """
    Applies a specified window function to the data.

    Args:
        data: Input signal (1D NumPy array).
        window_type: Name of the window function (e.g., 'hann', 'hamming', 'blackman').
                     See scipy.signal.get_window for available types.

    Returns:
        Windowed data.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    logger.debug(f"Applying '{window_type}' window.")
    try:
        window = get_window(window_type, data.shape[0])
        return data * window
    except ValueError as e:
        logger.error(f"Invalid window type '{window_type}': {e}")
        raise # Re-raise the exception after logging

# Note: Butterworth filters moved to filters.py
