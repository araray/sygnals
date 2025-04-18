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
    data: NDArray[np.float64], # Changed from np.float_
    fs: Union[int, float] = 1.0,
    n: Optional[int] = None,
    window: Optional[str] = "hann"
) -> Tuple[NDArray[np.float64], NDArray[np.complex_]]: # Changed from np.float_
    """
    Computes the Fast Fourier Transform (FFT) of a real-valued signal.

    Args:
        data: Input time-domain signal (1D NumPy array of float64).
        fs: Sampling frequency of the signal (default: 1.0 Hz).
        n: Length of the FFT. If None, uses the length of the data.
           If n > len(data), the data is zero-padded.
           If n < len(data), the data is truncated.
        window: Name of the window function to apply before FFT (e.g., 'hann', 'hamming').
                If None, no window is applied. See scipy.signal.get_window for options.

    Returns:
        A tuple containing:
        - freqs (NDArray[np.float64]): Array of frequencies corresponding to the FFT bins.
        - spectrum (NDArray[np.complex_]): Complex-valued FFT result.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    if window:
        logger.debug(f"Applying '{window}' window before FFT.")
        try:
            # Ensure window length matches data length precisely
            win = get_window(window, data.shape[0], fftbins=False) # fftbins=False for general use
            if len(win) != len(data):
                 # Handle potential length mismatch if fftbins=True was intended for specific cases
                 logger.warning(f"Window length {len(win)} differs from data length {len(data)}. Adjusting window.")
                 # Simple truncation/padding - consider more sophisticated handling if needed
                 if len(win) > len(data):
                     win = win[:len(data)]
                 else: # len(win) < len(data)
                     win = np.pad(win, (0, len(data) - len(win)), mode='constant')

            data = data * win
        except ValueError as e:
            logger.warning(f"Could not apply window '{window}': {e}. Proceeding without window.")
        except Exception as e: # Catch broader exceptions during windowing
             logger.warning(f"Unexpected error applying window '{window}': {e}. Proceeding without window.")


    if n is None:
        n = data.shape[0]

    logger.debug(f"Computing FFT with N={n}, Fs={fs}")
    spectrum = fft(data, n=n)
    freqs = fftfreq(n, d=1/fs)

    return freqs, spectrum

def compute_ifft(
    spectrum: NDArray[np.complex_],
    n: Optional[int] = None
) -> NDArray[np.float64]: # Changed from np.float_
    """
    Computes the Inverse Fast Fourier Transform (IFFT).

    Assumes the input spectrum corresponds to a real-valued time-domain signal.

    Args:
        spectrum: Complex-valued frequency spectrum.
        n: Length of the inverse FFT. If None, uses the length of the spectrum.
           Should typically match the original FFT length.

    Returns:
        Real-valued time-domain signal (float64).
    """
    if spectrum.ndim != 1:
        raise ValueError("Input spectrum must be a 1D array.")

    if n is None:
        n = spectrum.shape[0]

    logger.debug(f"Computing IFFT with N={n}")
    # Use np.real to discard negligible imaginary parts due to numerical precision
    time_domain_signal = np.real(ifft(spectrum, n=n))

    # Ensure output is float64
    return time_domain_signal.astype(np.float64, copy=False)

# --- Convolution-related functions ---

def apply_convolution(
    data: NDArray[np.float64], # Changed from np.float_
    kernel: NDArray[np.float64], # Changed from np.float_
    mode: str = "same"
) -> NDArray[np.float64]: # Changed from np.float_
    """
    Applies 1D convolution using the Fast Fourier Transform method.

    Args:
        data: Input signal (1D NumPy array of float64).
        kernel: The convolution kernel (1D NumPy array of float64).
        mode: Convolution mode ('full', 'valid', 'same'). See scipy.signal.fftconvolve.
              Default is 'same', returning output of the same size as 'data'.

    Returns:
        The result of the convolution (float64).
    """
    if data.ndim != 1 or kernel.ndim != 1:
        raise ValueError("Input data and kernel must be 1D arrays.")

    logger.debug(f"Applying convolution with kernel size {kernel.shape[0]}, mode='{mode}'")
    # Ensure output is float64
    result = fftconvolve(data, kernel, mode=mode)
    return result.astype(np.float64, copy=False)

# --- Window functions ---

def apply_window(
    data: NDArray[np.float64], # Changed from np.float_
    window_type: str = "hann"
) -> NDArray[np.float64]: # Changed from np.float_
    """
    Applies a specified window function to the data.

    Args:
        data: Input signal (1D NumPy array of float64).
        window_type: Name of the window function (e.g., 'hann', 'hamming', 'blackman').
                     See scipy.signal.get_window for available types.

    Returns:
        Windowed data (float64).
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    logger.debug(f"Applying '{window_type}' window.")
    try:
        # Ensure window length matches data length precisely
        window = get_window(window_type, data.shape[0], fftbins=False)
        if len(window) != len(data):
             logger.warning(f"Window length {len(window)} differs from data length {len(data)}. Adjusting window.")
             if len(window) > len(data):
                 window = window[:len(data)]
             else:
                 window = np.pad(window, (0, len(data) - len(window)), mode='constant')

        # Ensure output is float64
        return (data * window).astype(np.float64, copy=False)
    except ValueError as e:
        logger.error(f"Invalid window type '{window_type}': {e}")
        raise # Re-raise the exception after logging
    except Exception as e: # Catch broader exceptions during windowing
        logger.warning(f"Unexpected error applying window '{window_type}': {e}. Proceeding without window.")
        # Return original data if windowing fails unexpectedly
        return data.astype(np.float64, copy=False)


# Note: Butterworth filters moved to filters.py
