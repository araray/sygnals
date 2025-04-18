# sygnals/core/filters.py

"""
Implementations of various digital filters (Butterworth, Chebyshev, FIR, etc.).
Focuses on designing filters using scipy.signal and applying them, typically
using zero-phase filtering via Second-Order Sections (SOS) for stability.
"""

import logging
# Import necessary types
from typing import Union, Tuple

import numpy as np
from numpy.typing import NDArray
# Import filter design and application functions from scipy.signal
from scipy.signal import butter, cheby1, firwin, lfilter, sosfiltfilt, iirdesign, zpk2sos, get_window

logger = logging.getLogger(__name__)

# --- Filter Design & Application (using SOS) ---

def design_butterworth_sos(
    cutoff: Union[float, Tuple[float, float]],
    fs: float,
    order: int,
    filter_type: str # Use Literal for stricter type checking if desired: Literal['lowpass', 'highpass', 'bandpass', 'bandstop']
) -> NDArray[np.float64]:
    """
    Designs a Butterworth IIR filter and returns it in Second-Order Sections (SOS) format.

    SOS format is generally preferred over transfer function (ba) format for
    numerical stability, especially for higher-order filters.

    Args:
        cutoff: The critical frequency or frequencies (in Hz).
                - For 'lowpass' or 'highpass', provide a single float cutoff frequency.
                - For 'bandpass' or 'bandstop', provide a tuple (low_cutoff, high_cutoff).
        fs: The sampling frequency of the signal (in Hz).
        order: The order of the filter. Higher orders provide steeper rolloff but
               can have phase distortion (mitigated by zero-phase filtering) and
               potential instability if not using SOS.
        filter_type: Type of filter: 'lowpass', 'highpass', 'bandpass', or 'bandstop'.

    Returns:
        SOS representation of the filter coefficients (NumPy array, float64).
        Shape is (n_sections, 6), where n_sections is typically order/2 for Butterworth.

    Raises:
        ValueError: If cutoff frequencies are invalid (e.g., >= Nyquist, low >= high).
    """
    nyquist = 0.5 * fs
    # Validate cutoff frequency/frequencies relative to Nyquist frequency
    if isinstance(cutoff, (int, float)):
        # Lowpass or Highpass
        if not 0 < cutoff < nyquist:
             raise ValueError(f"Cutoff frequency ({cutoff} Hz) must be strictly between 0 and Nyquist ({nyquist} Hz).")
        # Normalize cutoff frequency by Nyquist frequency for scipy functions
        normal_cutoff = cutoff / nyquist
    elif isinstance(cutoff, tuple) and len(cutoff) == 2:
        # Bandpass or Bandstop
        low, high = cutoff
        if not (0 < low < nyquist and 0 < high < nyquist):
             raise ValueError(f"Both low ({low} Hz) and high ({high} Hz) cutoff frequencies must be strictly between 0 and Nyquist ({nyquist} Hz).")
        if low >= high:
             raise ValueError(f"Low cutoff ({low} Hz) must be less than high cutoff ({high} Hz).")
        # Normalize both cutoff frequencies
        normal_cutoff = (low / nyquist, high / nyquist)
    else:
        raise TypeError("cutoff must be a float (for low/high pass) or a tuple of two floats (for band pass/stop).")


    logger.debug(f"Designing {order}-order Butterworth {filter_type} filter. "
                 f"Cutoff(s): {cutoff} Hz, Fs: {fs} Hz, Normalized Cutoff: {normal_cutoff}")

    try:
        # Design the filter using scipy.signal.butter, requesting SOS output
        sos = butter(order, normal_cutoff, btype=filter_type, analog=False, output='sos')
        # Ensure output array is float64
        return sos.astype(np.float64, copy=False)
    except ValueError as e:
        # Catch potential errors from butter function (e.g., invalid order)
        logger.error(f"Butterworth filter design failed: {e}. Check parameters (order, cutoff, fs).")
        raise

def apply_sos_filter(sos: NDArray[np.float64], data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Applies a filter in SOS format using zero-phase filtering (scipy.signal.sosfiltfilt).

    Zero-phase filtering processes the signal forwards and then backwards,
    resulting in no phase distortion but doubling the filter order's effect on
    magnitude response. It's suitable for offline processing where causality is
    not a strict requirement.

    Args:
        sos: Filter coefficients in Second-Order Sections format (float64 NumPy array).
             Obtained from functions like `design_butterworth_sos`.
        data: The input signal (1D NumPy array of float64).

    Returns:
        The filtered signal (1D NumPy array, float64).

    Raises:
        ValueError: If input data is not 1D.
    """
    if data.ndim != 1:
        raise ValueError("Input data for filtering must be a 1D array.")
    if sos.ndim != 2 or sos.shape[1] != 6:
         raise ValueError("Input sos must be a 2D array with shape (n_sections, 6).")

    logger.debug(f"Applying SOS filter (zero-phase) with {sos.shape[0]} sections.")
    # Use sosfiltfilt for zero-phase filtering
    # This function handles edge effects more gracefully than applying sosfilt twice manually.
    filtered_data = sosfiltfilt(sos, data)
    # Ensure output type is float64
    return filtered_data.astype(np.float64, copy=False)

# --- Convenience Filter Functions (using Butterworth SOS and Zero-Phase Filtering) ---

def low_pass_filter(
    data: NDArray[np.float64],
    cutoff: float,
    fs: float,
    order: int = 5 # Default order
) -> NDArray[np.float64]:
    """
    Applies a low-pass Butterworth filter using zero-phase filtering.

    Args:
        data: Input signal (1D NumPy array, float64).
        cutoff: Cutoff frequency (Hz). Frequencies above this will be attenuated.
        fs: Sampling frequency (Hz).
        order: Filter order (default: 5).

    Returns:
        Filtered signal (1D NumPy array, float64).
    """
    logger.info(f"Applying low-pass filter: cutoff={cutoff} Hz, order={order}")
    sos = design_butterworth_sos(cutoff, fs, order, 'lowpass')
    return apply_sos_filter(sos, data)

def high_pass_filter(
    data: NDArray[np.float64],
    cutoff: float,
    fs: float,
    order: int = 5
) -> NDArray[np.float64]:
    """
    Applies a high-pass Butterworth filter using zero-phase filtering.

    Args:
        data: Input signal (1D NumPy array, float64).
        cutoff: Cutoff frequency (Hz). Frequencies below this will be attenuated.
        fs: Sampling frequency (Hz).
        order: Filter order (default: 5).

    Returns:
        Filtered signal (1D NumPy array, float64).
    """
    logger.info(f"Applying high-pass filter: cutoff={cutoff} Hz, order={order}")
    sos = design_butterworth_sos(cutoff, fs, order, 'highpass')
    return apply_sos_filter(sos, data)

def band_pass_filter(
    data: NDArray[np.float64],
    low_cutoff: float,
    high_cutoff: float,
    fs: float,
    order: int = 5
) -> NDArray[np.float64]:
    """
    Applies a band-pass Butterworth filter using zero-phase filtering.

    Args:
        data: Input signal (1D NumPy array, float64).
        low_cutoff: Lower cutoff frequency (Hz) of the passband.
        high_cutoff: Higher cutoff frequency (Hz) of the passband.
        fs: Sampling frequency (Hz).
        order: Filter order (default: 5). Note: Bandpass order is effectively 2*order.

    Returns:
        Filtered signal (1D NumPy array, float64).
    """
    logger.info(f"Applying band-pass filter: low={low_cutoff} Hz, high={high_cutoff} Hz, order={order}")
    # Parameter validation (low < high, < Nyquist) happens inside design_butterworth_sos
    sos = design_butterworth_sos((low_cutoff, high_cutoff), fs, order, 'bandpass')
    return apply_sos_filter(sos, data)

def band_stop_filter(
    data: NDArray[np.float64],
    low_cutoff: float,
    high_cutoff: float,
    fs: float,
    order: int = 5
) -> NDArray[np.float64]:
    """
    Applies a band-stop (notch) Butterworth filter using zero-phase filtering.

    Args:
        data: Input signal (1D NumPy array, float64).
        low_cutoff: Lower cutoff frequency (Hz) of the stop band.
        high_cutoff: Higher cutoff frequency (Hz) of the stop band.
        fs: Sampling frequency (Hz).
        order: Filter order (default: 5). Note: Bandstop order is effectively 2*order.

    Returns:
        Filtered signal (1D NumPy array, float64).
    """
    logger.info(f"Applying band-stop filter: low={low_cutoff} Hz, high={high_cutoff} Hz, order={order}")
    # Parameter validation happens inside design_butterworth_sos
    sos = design_butterworth_sos((low_cutoff, high_cutoff), fs, order, 'bandstop')
    return apply_sos_filter(sos, data)


# --- Other Filter Types (Could be added later) ---

# Example structure for a Chebyshev Type I filter
# def design_cheby1_sos(...):
#     # Use scipy.signal.cheby1(..., output='sos')
#     pass

# Example structure for an FIR filter using firwin
# def design_fir_taps(num_taps, cutoff, fs, window='hamming', filter_type='lowpass'):
#     nyquist = 0.5 * fs
#     # Normalize cutoff(s)
#     # Determine pass_zero based on filter_type
#     taps = firwin(num_taps, normalized_cutoff, window=window, pass_zero=pass_zero, fs=fs) # fs arg available in newer scipy
#     return taps.astype(np.float64, copy=False)

# def apply_fir_filter(taps, data):
#     # Apply using lfilter (causal) or fftconvolve
#     # For zero-phase FIR, one common method is to apply lfilter, flip, apply again, flip
#     logger.debug(f"Applying FIR filter (causal) with {len(taps)} taps.")
#     return lfilter(taps, 1.0, data).astype(np.float64, copy=False)
