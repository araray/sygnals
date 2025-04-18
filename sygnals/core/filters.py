# sygnals/core/filters.py

"""
Implementations of various digital filters (Butterworth, Chebyshev, FIR, etc.).
"""

import logging
from typing import Union, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, cheby1, firwin, lfilter, sosfiltfilt, iirdesign, zpk2sos

logger = logging.getLogger(__name__)

# --- Filter Design & Application ---

def design_butterworth_sos(
    cutoff: Union[float, Tuple[float, float]],
    fs: float,
    order: int,
    filter_type: str
) -> NDArray[np.float_]:
    """
    Designs a Butterworth filter and returns it in Second-Order Sections (SOS) format.
    SOS format is generally preferred for numerical stability.

    Args:
        cutoff: The critical frequency or frequencies (in Hz). For lowpass/highpass,
                a scalar. For bandpass/bandstop, a tuple (low, high).
        fs: The sampling frequency (in Hz).
        order: The order of the filter.
        filter_type: Type of filter {'lowpass', 'highpass', 'bandpass', 'bandstop'}.

    Returns:
        SOS representation of the filter coefficients.
    """
    nyquist = 0.5 * fs
    if isinstance(cutoff, (int, float)):
        # Lowpass or Highpass
        normal_cutoff = cutoff / nyquist
    else:
        # Bandpass or Bandstop
        normal_cutoff = (cutoff[0] / nyquist, cutoff[1] / nyquist)

    logger.debug(f"Designing {order}-order Butterworth {filter_type} filter. "
                 f"Cutoff(s): {cutoff} Hz, Fs: {fs} Hz, Normalized: {normal_cutoff}")

    try:
        # Output 'sos' for better numerical stability with higher orders
        sos = butter(order, normal_cutoff, btype=filter_type, analog=False, output='sos')
        return sos
    except ValueError as e:
        logger.error(f"Filter design failed: {e}. Check cutoff frequencies relative to Nyquist ({nyquist} Hz).")
        raise

def apply_sos_filter(sos: NDArray[np.float_], data: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Applies a filter in SOS format using zero-phase filtering (filtfilt).

    Args:
        sos: Filter coefficients in Second-Order Sections format.
        data: The input signal (1D NumPy array).

    Returns:
        The filtered signal.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    logger.debug(f"Applying SOS filter (zero-phase) with {sos.shape[0]} sections.")
    # sosfiltfilt provides zero-phase filtering, which is often desirable
    return sosfiltfilt(sos, data)

# --- Convenience Filter Functions ---

def low_pass_filter(
    data: NDArray[np.float_],
    cutoff: float,
    fs: float,
    order: int = 5
) -> NDArray[np.float_]:
    """
    Applies a low-pass Butterworth filter using zero-phase filtering.

    Args:
        data: Input signal.
        cutoff: Cutoff frequency (Hz).
        fs: Sampling frequency (Hz).
        order: Filter order (default: 5).

    Returns:
        Filtered signal.
    """
    sos = design_butterworth_sos(cutoff, fs, order, 'lowpass')
    return apply_sos_filter(sos, data)

def high_pass_filter(
    data: NDArray[np.float_],
    cutoff: float,
    fs: float,
    order: int = 5
) -> NDArray[np.float_]:
    """
    Applies a high-pass Butterworth filter using zero-phase filtering.

    Args:
        data: Input signal.
        cutoff: Cutoff frequency (Hz).
        fs: Sampling frequency (Hz).
        order: Filter order (default: 5).

    Returns:
        Filtered signal.
    """
    sos = design_butterworth_sos(cutoff, fs, order, 'highpass')
    return apply_sos_filter(sos, data)

def band_pass_filter(
    data: NDArray[np.float_],
    low_cutoff: float,
    high_cutoff: float,
    fs: float,
    order: int = 5
) -> NDArray[np.float_]:
    """
    Applies a band-pass Butterworth filter using zero-phase filtering.

    Args:
        data: Input signal.
        low_cutoff: Lower cutoff frequency (Hz).
        high_cutoff: Higher cutoff frequency (Hz).
        fs: Sampling frequency (Hz).
        order: Filter order (default: 5).

    Returns:
        Filtered signal.
    """
    if low_cutoff >= high_cutoff:
        raise ValueError(f"Low cutoff ({low_cutoff}) must be less than high cutoff ({high_cutoff}).")
    sos = design_butterworth_sos((low_cutoff, high_cutoff), fs, order, 'bandpass')
    return apply_sos_filter(sos, data)

def band_stop_filter(
    data: NDArray[np.float_],
    low_cutoff: float,
    high_cutoff: float,
    fs: float,
    order: int = 5
) -> NDArray[np.float_]:
    """
    Applies a band-stop (notch) Butterworth filter using zero-phase filtering.

    Args:
        data: Input signal.
        low_cutoff: Lower cutoff frequency of the stop band (Hz).
        high_cutoff: Higher cutoff frequency of the stop band (Hz).
        fs: Sampling frequency (Hz).
        order: Filter order (default: 5).

    Returns:
        Filtered signal.
    """
    if low_cutoff >= high_cutoff:
        raise ValueError(f"Low cutoff ({low_cutoff}) must be less than high cutoff ({high_cutoff}).")
    sos = design_butterworth_sos((low_cutoff, high_cutoff), fs, order, 'bandstop')
    return apply_sos_filter(sos, data)


# --- Other Filter Types (Placeholder Examples - Implement as needed) ---

# def chebyshev1_filter(...):
#     # Design using cheby1(..., output='sos')
#     # Apply using apply_sos_filter
#     pass

# def fir_filter(data, num_taps, cutoff, fs, window='hamming', filter_type='lowpass'):
#     """Designs and applies an FIR filter using the window method."""
#     nyquist = 0.5 * fs
#     if isinstance(cutoff, (int, float)):
#         normal_cutoff = cutoff / nyquist
#         pass_zero = (filter_type == 'lowpass' or filter_type == 'bandstop')
#     else: # bandpass/bandstop
#         normal_cutoff = (cutoff[0] / nyquist, cutoff[1] / nyquist)
#         if filter_type == 'bandpass':
#             pass_zero = False
#         elif filter_type == 'bandstop':
#             pass_zero = True
#         else:
#              raise ValueError("Invalid filter_type for band filter")

#     taps = firwin(num_taps, normal_cutoff, window=window, pass_zero=pass_zero)
#     # FIR filters are often applied with lfilter (causal) or fftconvolve
#     # For zero-phase, could potentially use filtfilt but less common for FIR
#     logger.debug(f"Applying FIR filter ({num_taps} taps, type: {filter_type})")
#     return lfilter(taps, 1.0, data) # Causal filtering
