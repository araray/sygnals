# sygnals/utils/visualizations.py

"""
Functions for generating various visualizations of signal data.
"""

import logging
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pywt # For wavelet scalogram
from numpy.typing import NDArray
from scipy.signal import spectrogram

# Import DSP functions
from sygnals.core.dsp import compute_fft
from sygnals.core.transforms import wavelet_transform # Assuming wavelet_transform is in transforms.py

logger = logging.getLogger(__name__)


# --- Spectrogram Plot ---

def plot_spectrogram(
    data: NDArray[np.float64],
    sr: float,
    output_file: str,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    window: str = "hann",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    db_scale: bool = True,
    cmap: str = 'viridis'
):
    """
    Generate and save a spectrogram plot using scipy.signal.spectrogram.

    Args:
        data: Input audio signal (1D NumPy array, float64).
        sr: Sampling rate of the signal (Hz).
        output_file: Path where to save the plot image.
        f_min: Minimum frequency to display (Hz).
        f_max: Maximum frequency to display (Hz).
        window: Window function for STFT.
        nperseg: Length of each segment for STFT.
        noverlap: Overlap between segments.
        db_scale: Display power in dB (log scale).
        cmap: Colormap.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0: logger.warning("Input data empty. Skipping spectrogram."); return

    logger.info(f"Generating spectrogram: sr={sr}, output={output_file}")
    logger.debug(f"Params: window={window}, nperseg={nperseg}, noverlap={noverlap}, f_min={f_min}, f_max={f_max}, db={db_scale}")

    nperseg = nperseg or min(data.size, int(sr * 0.025)) # Default ~25ms window
    noverlap = noverlap or nperseg // 2
    if nperseg > data.size: nperseg, noverlap = data.size, 0

    try:
        f, t, Sxx = spectrogram(data, fs=sr, window=window, nperseg=nperseg, noverlap=noverlap)
    except ValueError as e: logger.error(f"Spectrogram calculation error: {e}"); raise

    plt.figure(figsize=(10, 6))
    plot_data = 10 * np.log10(Sxx + 1e-10) if db_scale else Sxx
    plt.pcolormesh(t, f, plot_data, shading="gouraud", cmap=cmap)
    plt.colorbar(label="Power [dB]" if db_scale else "Power")

    f_nyquist = sr / 2
    plt.ylim(f_min or 0, f_max or f_nyquist)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.title("Spectrogram")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    try: plt.savefig(output_file, dpi=300, bbox_inches="tight"); logger.info(f"Spectrogram saved.")
    except Exception as e: logger.error(f"Failed to save spectrogram: {e}")
    finally: plt.close()


# --- FFT Magnitude Plot ---

def plot_fft_magnitude( # Renamed for clarity
    data: NDArray[np.float64],
    sr: float,
    output_file: str,
    window: Optional[str] = "hann",
    **fft_kwargs
):
    """
    Generate and save an FFT magnitude spectrum plot.

    Args:
        data: Input time-domain signal (1D float64).
        sr: Sampling rate (Hz).
        output_file: Path to save the plot image.
        window: Window function applied before FFT.
        **fft_kwargs: Additional arguments passed to `compute_fft`.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0: logger.warning("Input data empty. Skipping FFT magnitude plot."); return

    logger.info(f"Generating FFT magnitude plot: sr={sr}, output={output_file}")
    try:
        freqs, spectrum = compute_fft(data, fs=sr, window=window, **fft_kwargs)
    except Exception as e: logger.error(f"FFT computation error: {e}"); raise

    magnitude = np.abs(spectrum)
    n = spectrum.size # Use spectrum size (accounts for padding via n in fft_kwargs)
    positive_freqs = freqs[:n // 2]
    positive_magnitude = magnitude[:n // 2]

    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Magnitude Spectrum")
    plt.grid(True, alpha=0.5)
    plt.xlim(0, sr / 2)
    plt.tight_layout()

    try: plt.savefig(output_file, dpi=300, bbox_inches="tight"); logger.info(f"FFT magnitude plot saved.")
    except Exception as e: logger.error(f"Failed to save FFT magnitude plot: {e}")
    finally: plt.close()


# --- FFT Phase Plot (New) ---

def plot_fft_phase(
    data: NDArray[np.float64],
    sr: float,
    output_file: str,
    window: Optional[str] = "hann",
    unwrap: bool = True, # Option to unwrap phase
    **fft_kwargs
):
    """
    Generate and save an FFT phase spectrum plot.

    Args:
        data: Input time-domain signal (1D float64).
        sr: Sampling rate (Hz).
        output_file: Path to save the plot image.
        window: Window function applied before FFT.
        unwrap: If True, unwrap the phase angle (default: True).
        **fft_kwargs: Additional arguments passed to `compute_fft`.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0: logger.warning("Input data empty. Skipping FFT phase plot."); return

    logger.info(f"Generating FFT phase plot: sr={sr}, unwrap={unwrap}, output={output_file}")
    try:
        freqs, spectrum = compute_fft(data, fs=sr, window=window, **fft_kwargs)
    except Exception as e: logger.error(f"FFT computation error: {e}"); raise

    phase = np.angle(spectrum)
    if unwrap:
        phase = np.unwrap(phase)

    n = spectrum.size
    positive_freqs = freqs[:n // 2]
    positive_phase = phase[:n // 2]

    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_phase)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.title(f"FFT Phase Spectrum {'(Unwrapped)' if unwrap else ''}")
    plt.grid(True, alpha=0.5)
    plt.xlim(0, sr / 2)
    plt.tight_layout()

    try: plt.savefig(output_file, dpi=300, bbox_inches="tight"); logger.info(f"FFT phase plot saved.")
    except Exception as e: logger.error(f"Failed to save FFT phase plot: {e}")
    finally: plt.close()


# --- Waveform Plot ---

def plot_waveform(
    data: NDArray[np.float64],
    sr: float,
    output_file: str,
    max_samples: Optional[int] = None
):
    """
    Generate and save a waveform plot (amplitude vs. time).

    Args:
        data: Input time-domain signal (1D float64).
        sr: Sampling rate (Hz).
        output_file: Path to save the plot image.
        max_samples: If set, only plot the first `max_samples`.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0: logger.warning("Input data empty. Skipping waveform plot."); return

    logger.info(f"Generating waveform plot: sr={sr}, output={output_file}")

    plot_data = data[:max_samples] if max_samples and data.size > max_samples else data
    duration = plot_data.size / sr
    time = np.linspace(0, duration, num=plot_data.size, endpoint=False)

    plt.figure(figsize=(12, 4))
    plt.plot(time, plot_data, linewidth=0.8)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.grid(True, alpha=0.5)
    plt.xlim(0, duration)
    plt.tight_layout()

    try: plt.savefig(output_file, dpi=300, bbox_inches="tight"); logger.info(f"Waveform plot saved.")
    except Exception as e: logger.error(f"Failed to save waveform plot: {e}")
    finally: plt.close()


# --- Wavelet Scalogram Plot (New) ---

def plot_scalogram(
    data: NDArray[np.float64],
    output_file: str,
    scales: Union[NDArray[np.float64], int] = 64, # Scales to use or number of scales
    wavelet: str = 'morl', # Common choice for CWT scalograms
    sr: Optional[float] = None, # Optional: for y-axis labeling in Hz
    cmap: str = 'viridis'
):
    """
    Generate and save a Wavelet Transform Scalogram (magnitude plot).
    Uses Continuous Wavelet Transform (CWT).

    Args:
        data: Input time-domain signal (1D float64).
        output_file: Path to save the plot image.
        scales: Array of scales to use for CWT, or an integer number of scales
                (logarithmically spaced from 1 up to a reasonable max).
        wavelet: Wavelet to use (e.g., 'morl', 'cmor', 'gaus1'). See pywt.wavelist(kind='continuous').
        sr: Optional sampling rate (Hz) for frequency axis calculation.
        cmap: Colormap.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0: logger.warning("Input data empty. Skipping scalogram plot."); return

    logger.info(f"Generating scalogram: wavelet={wavelet}, output={output_file}")

    # Determine scales if integer provided
    if isinstance(scales, int):
        num_scales = scales
        # Define scales logarithmically, e.g., from 1 up to data.size / 8
        max_scale = max(2, data.size // 8) # Heuristic max scale
        scales = np.logspace(0, np.log10(max_scale), num=num_scales)
        logger.debug(f"Generated {num_scales} scales from 1 to {max_scale:.2f}")

    try:
        # Perform Continuous Wavelet Transform (CWT)
        coeffs, freqs_cwt = pywt.cwt(data, scales, wavelet, sampling_period=1.0/sr if sr else 1.0)
        # coeffs shape: (num_scales, data_length)
    except Exception as e:
        logger.error(f"CWT calculation error: {e}")
        raise

    plt.figure(figsize=(10, 6))
    # Plot magnitude of coefficients
    plt.imshow(np.abs(coeffs), extent=[0, data.size, scales[-1], scales[0]], aspect='auto', cmap=cmap)
    plt.colorbar(label="Magnitude")

    # Configure axes
    plt.xlabel("Time (samples)")
    if sr and freqs_cwt is not None:
        # Use frequencies calculated by pywt.cwt if available and sr provided
        # Need to map scales to frequencies appropriately
        # Often plotted with log scale for frequency/scale
        plt.ylabel("Frequency (Hz)")
        # Set y-ticks based on calculated frequencies (might need formatting)
        tick_indices = np.linspace(0, len(scales)-1, num=8, dtype=int) # Example: 8 ticks
        plt.yticks(scales[tick_indices], [f"{freqs_cwt[i]:.2f}" for i in tick_indices])
        plt.yscale('log') # Log scale is common for scales/frequencies
    else:
        plt.ylabel("Scale")
        # Optionally use log scale for scales too
        # plt.yscale('log')

    plt.title(f"Scalogram (Wavelet: {wavelet})")
    plt.tight_layout()

    try: plt.savefig(output_file, dpi=300, bbox_inches="tight"); logger.info(f"Scalogram plot saved.")
    except Exception as e: logger.error(f"Failed to save scalogram plot: {e}")
    finally: plt.close()


# --- Placeholder for other plots ---
# def plot_pole_zero(...): pass
# def plot_lissajous(...): pass
# def plot_constellation(...): pass
# def plot_heatmap(...): pass
# def plot_feature_histogram(...): pass
# def plot_feature_scatter(...): pass
