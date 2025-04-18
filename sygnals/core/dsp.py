# sygnals/core/dsp.py

"""
Core Digital Signal Processing (DSP) functions.
Includes FFT, STFT, CQT, Correlation, PSD, Convolution, Windowing, Envelope Detection etc.
Excludes specific filter implementations (see filters.py).
"""

import logging
from typing import Tuple, Optional, Union, Literal

import numpy as np
import librosa # Use librosa for STFT, CQT etc. for consistency and features
from numpy.typing import NDArray
from scipy.fft import fft, ifft, fftfreq # Use scipy.fft for basic FFT/IFFT
from scipy.signal import fftconvolve, get_window, hilbert, correlate, periodogram, welch

logger = logging.getLogger(__name__)

# --- FFT-related functions ---

def compute_fft(
    data: NDArray[np.float64],
    fs: Union[int, float] = 1.0,
    n: Optional[int] = None,
    window: Optional[str] = "hann"
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
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
        - spectrum (NDArray[np.complex128]): Complex-valued FFT result.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    if window:
        logger.debug(f"Applying '{window}' window before FFT.")
        try:
            win = get_window(window, data.shape[0], fftbins=False)
            if len(win) != len(data):
                 logger.warning(f"Window length {len(win)} differs from data length {len(data)}. Adjusting window.")
                 if len(win) > len(data): win = win[:len(data)]
                 else: win = np.pad(win, (0, len(data) - len(win)), mode='constant')
            data = data * win
        except ValueError as e:
            logger.warning(f"Could not apply window '{window}': {e}. Proceeding without window.")
        except Exception as e:
             logger.warning(f"Unexpected error applying window '{window}': {e}. Proceeding without window.")

    if n is None:
        n = data.shape[0]

    logger.debug(f"Computing FFT with N={n}, Fs={fs}")
    spectrum = fft(data, n=n)
    freqs = fftfreq(n, d=1/fs)

    return freqs.astype(np.float64, copy=False), spectrum.astype(np.complex128, copy=False)

def compute_ifft(
    spectrum: NDArray[np.complex128],
    n: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Computes the Inverse Fast Fourier Transform (IFFT).
    Assumes the input spectrum corresponds to a real-valued time-domain signal.

    Args:
        spectrum: Complex-valued frequency spectrum (complex128).
        n: Length of the inverse FFT. If None, uses the length of the spectrum.

    Returns:
        Real-valued time-domain signal (float64).
    """
    if spectrum.ndim != 1:
        raise ValueError("Input spectrum must be a 1D array.")
    if n is None:
        n = spectrum.shape[0]
    logger.debug(f"Computing IFFT with N={n}")
    time_domain_signal = np.real(ifft(spectrum, n=n))
    return time_domain_signal.astype(np.float64, copy=False)


# --- Time-Frequency Transforms ---

def compute_stft(
    y: NDArray[np.float64],
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = 'hann',
    center: bool = True,
    pad_mode: str = 'constant', # Changed from 'reflect' in librosa 0.10+
) -> NDArray[np.complex128]:
    """
    Computes the Short-Time Fourier Transform (STFT) using librosa.

    Args:
        y: Input time-domain signal (1D float64).
        n_fft: Length of the FFT window.
        hop_length: Number of samples between successive frames (default: win_length // 4).
        win_length: Each frame of audio is windowed by `window` of length `win_length`.
                    Defaults to `n_fft`.
        window: Window function name (see scipy.signal.get_window) or window array.
        center: If True, pad `y` so that frames are centered at `t * hop_length`.
        pad_mode: Padding mode used if `center=True`. Default 'constant' with 0s.

    Returns:
        Complex-valued STFT matrix (shape: (1 + n_fft/2, num_frames)).
    """
    if y.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    logger.debug(f"Computing STFT: n_fft={n_fft}, hop={hop_length}, win={win_length}, window={window}, center={center}")
    try:
        stft_matrix = librosa.stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        return stft_matrix.astype(np.complex128, copy=False)
    except Exception as e:
        logger.error(f"Error computing STFT: {e}")
        raise

def compute_cqt(
    y: NDArray[np.float64],
    sr: int,
    hop_length: Optional[int] = 512,
    fmin: Optional[float] = None,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    **kwargs # Other librosa.cqt args (tuning, filter_scale, norm, etc.)
) -> NDArray[np.complex128]:
    """
    Computes the Constant-Q Transform (CQT) using librosa.

    Args:
        y: Input time-domain signal (1D float64).
        sr: Sampling rate.
        hop_length: Number of samples between successive CQT columns.
        fmin: Minimum frequency (Hz). Defaults to C1 (~32.7 Hz).
        n_bins: Number of frequency bins. Spans `n_bins / bins_per_octave` octaves.
        bins_per_octave: Number of bins per octave.
        **kwargs: Additional arguments passed to `librosa.cqt`.

    Returns:
        Complex-valued CQT matrix (shape: (n_bins, num_frames)).
    """
    if y.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    if fmin is None:
        fmin = librosa.note_to_hz('C1')
    logger.debug(f"Computing CQT: sr={sr}, hop={hop_length}, fmin={fmin}, n_bins={n_bins}, bins_per_octave={bins_per_octave}")
    try:
        cqt_matrix = librosa.cqt(
            y=y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            **kwargs
        )
        return cqt_matrix.astype(np.complex128, copy=False)
    except Exception as e:
        logger.error(f"Error computing CQT: {e}")
        raise


# --- Convolution-related functions ---

def apply_convolution(
    data: NDArray[np.float64],
    kernel: NDArray[np.float64],
    mode: str = "same"
) -> NDArray[np.float64]:
    """
    Applies 1D convolution using the Fast Fourier Transform method (fftconvolve).

    Args:
        data: Input signal (1D NumPy array of float64).
        kernel: The convolution kernel (1D NumPy array of float64).
        mode: Convolution mode ('full', 'valid', 'same'). Default is 'same'.

    Returns:
        The result of the convolution (float64).
    """
    if data.ndim != 1 or kernel.ndim != 1:
        raise ValueError("Input data and kernel must be 1D arrays.")
    logger.debug(f"Applying convolution with kernel size {kernel.shape[0]}, mode='{mode}'")
    result = fftconvolve(data, kernel, mode=mode)
    return result.astype(np.float64, copy=False)


# --- Correlation ---

def compute_correlation(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    mode: Literal['full', 'valid', 'same'] = 'full',
    method: Literal['auto', 'direct', 'fft'] = 'auto'
) -> NDArray[np.float64]:
    """
    Computes the cross-correlation of two 1-dimensional sequences using scipy.signal.correlate.

    Args:
        x: First input sequence (1D float64).
        y: Second input sequence (1D float64).
        mode: Correlation mode ('full', 'valid', 'same'). Default: 'full'.
        method: Computation method ('auto', 'direct', 'fft'). Default: 'auto'.

    Returns:
        Cross-correlation result (float64). Lags are implicitly defined by the mode.
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input sequences for correlation must be 1D arrays.")
    logger.debug(f"Computing correlation: mode='{mode}', method='{method}'")
    try:
        correlation = correlate(x, y, mode=mode, method=method)
        return correlation.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error computing correlation: {e}")
        raise

def compute_autocorrelation(
    x: NDArray[np.float64],
    mode: Literal['full', 'valid', 'same'] = 'full',
    method: Literal['auto', 'direct', 'fft'] = 'auto'
) -> NDArray[np.float64]:
    """
    Computes the auto-correlation of a 1-dimensional sequence.

    Args:
        x: Input sequence (1D float64).
        mode: Correlation mode ('full', 'valid', 'same'). Default: 'full'.
        method: Computation method ('auto', 'direct', 'fft'). Default: 'auto'.

    Returns:
        Auto-correlation result (float64). The center of the 'full' mode result
        corresponds to zero lag.
    """
    logger.debug(f"Computing auto-correlation: mode='{mode}', method='{method}'")
    # Autocorrelation is correlation with itself
    return compute_correlation(x, x, mode=mode, method=method)


# --- Power Spectral Density (PSD) ---

def compute_psd_periodogram(
    x: NDArray[np.float64],
    fs: float = 1.0,
    window: str = 'hann',
    nfft: Optional[int] = None,
    detrend: Union[str, bool] = 'constant',
    scaling: Literal['density', 'spectrum'] = 'density'
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimates Power Spectral Density using Periodogram method (scipy.signal.periodogram).

    Args:
        x: Input time series (1D float64).
        fs: Sampling frequency (Hz).
        window: Window function name or array. Default: 'hann'.
        nfft: Length of the FFT used. If None, defaults to len(x).
        detrend: Specifies how to detrend each segment ('constant', 'linear', False). Default: 'constant'.
        scaling: 'density' (V**2/Hz) or 'spectrum' (V**2). Default: 'density'.

    Returns:
        Tuple containing:
        - frequencies (NDArray[np.float64]): Frequencies of the PSD estimate.
        - Pxx (NDArray[np.float64]): Power Spectral Density or Power Spectrum.
    """
    if x.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    logger.debug(f"Computing PSD (Periodogram): fs={fs}, window={window}, nfft={nfft}, detrend={detrend}, scaling={scaling}")
    try:
        frequencies, Pxx = periodogram(
            x,
            fs=fs,
            window=window,
            nfft=nfft,
            detrend=detrend,
            return_onesided=True, # Usually want one-sided for real signals
            scaling=scaling
        )
        return frequencies.astype(np.float64, copy=False), Pxx.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error computing Periodogram PSD: {e}")
        raise

def compute_psd_welch(
    x: NDArray[np.float64],
    fs: float = 1.0,
    window: str = 'hann',
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: Union[str, bool] = 'constant',
    scaling: Literal['density', 'spectrum'] = 'density'
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimates Power Spectral Density using Welch's method (scipy.signal.welch).
    Averages periodograms of overlapping segments.

    Args:
        x: Input time series (1D float64).
        fs: Sampling frequency (Hz).
        window: Window function name or array. Default: 'hann'.
        nperseg: Length of each segment. Defaults to 256.
        noverlap: Number of points to overlap between segments. Defaults to nperseg // 2.
        nfft: Length of the FFT used for each segment. Defaults to nperseg.
        detrend: Specifies how to detrend each segment ('constant', 'linear', False). Default: 'constant'.
        scaling: 'density' (V**2/Hz) or 'spectrum' (V**2). Default: 'density'.

    Returns:
        Tuple containing:
        - frequencies (NDArray[np.float64]): Frequencies of the PSD estimate.
        - Pxx (NDArray[np.float64]): Power Spectral Density or Power Spectrum.
    """
    if x.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    logger.debug(f"Computing PSD (Welch): fs={fs}, window={window}, nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}, detrend={detrend}, scaling={scaling}")
    try:
        frequencies, Pxx = welch(
            x,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend,
            return_onesided=True,
            scaling=scaling
        )
        return frequencies.astype(np.float64, copy=False), Pxx.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error computing Welch PSD: {e}")
        raise


# --- Envelope Detection ---

def amplitude_envelope(
    y: NDArray[np.float64],
    method: Literal['hilbert', 'rms'] = 'hilbert',
    frame_length: Optional[int] = None, # Required for RMS method
    hop_length: Optional[int] = None    # Required for RMS method
) -> NDArray[np.float64]:
    """
    Computes the amplitude envelope of a signal.

    Args:
        y: Input time series (1D float64).
        method: 'hilbert' (using Hilbert transform) or 'rms' (using frame-based RMS).
        frame_length: Frame length in samples (required for 'rms').
        hop_length: Hop length in samples (required for 'rms').

    Returns:
        Amplitude envelope (1D float64). Length matches `y` for 'hilbert',
        length is num_frames for 'rms'.
    """
    if y.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    logger.debug(f"Computing Amplitude Envelope using method: {method}")

    if method == 'hilbert':
        try:
            # Hilbert transform gives the analytic signal (y + j*hilbert(y))
            # The magnitude is the instantaneous amplitude (envelope)
            analytic_signal = hilbert(y)
            envelope = np.abs(analytic_signal)
            return envelope.astype(np.float64, copy=False)
        except Exception as e:
            logger.error(f"Error computing Hilbert envelope: {e}")
            raise
    elif method == 'rms':
        if frame_length is None or hop_length is None:
            raise ValueError("frame_length and hop_length are required for 'rms' envelope method.")
        try:
            # Use the RMS energy function from audio features
            from ..audio.features import rms_energy # Local import to avoid circular dependency if moved
            # Note: RMS is related to envelope but not exactly the same as Hilbert envelope
            # It gives energy per frame, which acts as a smoothed envelope
            rms_env = rms_energy(y, frame_length=frame_length, hop_length=hop_length)
            return rms_env # Already float64
        except Exception as e:
            logger.error(f"Error computing RMS envelope: {e}")
            raise
    else:
        raise ValueError(f"Unsupported envelope method: {method}. Choose 'hilbert' or 'rms'.")


# --- Window functions ---

def apply_window(
    data: NDArray[np.float64],
    window_type: str = "hann"
) -> NDArray[np.float64]:
    """
    Applies a specified window function to the data.

    Args:
        data: Input signal (1D NumPy array of float64).
        window_type: Name of the window function (e.g., 'hann', 'hamming', 'blackman').

    Returns:
        Windowed data (float64).
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    logger.debug(f"Applying '{window_type}' window.")
    try:
        window = get_window(window_type, data.shape[0], fftbins=False)
        if len(window) != len(data):
             logger.warning(f"Window length {len(window)} differs from data length {len(data)}. Adjusting window.")
             if len(window) > len(data): window = window[:len(data)]
             else: window = np.pad(window, (0, len(data) - len(window)), mode='constant')
        return (data * window).astype(np.float64, copy=False)
    except ValueError as e:
        logger.error(f"Invalid window type '{window_type}': {e}")
        raise
    except Exception as e:
        logger.warning(f"Unexpected error applying window '{window_type}': {e}. Proceeding without window.")
        return data.astype(np.float64, copy=False)
