# sygnals/utils/visualizations.py

"""
Functions for generating various visualizations of signal data using Matplotlib.
"""

import logging
# Import necessary types
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pywt # For Continuous Wavelet Transform (CWT) used in scalogram
from numpy.typing import NDArray
# Import specific signal processing functions
from scipy.signal import spectrogram # For spectrogram calculation
from sygnals.core.dsp import compute_fft # For FFT calculation

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
    cmap: str = 'viridis',
    title: str = "Spectrogram"
):
    """
    Generate and save a spectrogram plot using scipy.signal.spectrogram.

    A spectrogram visualizes the time-varying frequency content of a signal.

    Args:
        data: Input audio signal (1D NumPy array, float64).
        sr: Sampling rate of the signal (Hz).
        output_file: Path where to save the plot image (e.g., 'spectrogram.png').
        f_min: Minimum frequency to display (Hz). Defaults to 0.
        f_max: Maximum frequency to display (Hz). Defaults to Nyquist frequency (sr/2).
        window: Window function name for STFT (e.g., 'hann', 'hamming').
        nperseg: Length of each segment for STFT. Controls frequency resolution.
                 Defaults to a value yielding reasonable resolution (e.g., 25ms).
        noverlap: Overlap between segments in samples. Controls time resolution.
                  Defaults to nperseg // 2.
        db_scale: If True, display power in decibels (log scale). Otherwise, linear power.
        cmap: Matplotlib colormap name (e.g., 'viridis', 'magma', 'gray').
        title: Title for the plot.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0:
        logger.warning(f"Input data empty. Skipping spectrogram plot for {output_file}.")
        return

    logger.info(f"Generating spectrogram: sr={sr}, output={output_file}")
    logger.debug(f"Params: window={window}, nperseg={nperseg}, noverlap={noverlap}, f_min={f_min}, f_max={f_max}, db={db_scale}")

    # Sensible defaults for segment length and overlap
    default_nperseg = min(data.size, int(sr * 0.025)) # Default ~25ms window, capped by data length
    nperseg = nperseg if nperseg is not None else default_nperseg
    noverlap = noverlap if noverlap is not None else nperseg // 2

    # Ensure nperseg and noverlap are valid given data size
    if nperseg > data.size:
        logger.warning(f"nperseg ({nperseg}) > data size ({data.size}). Setting nperseg=data size, noverlap=0.")
        nperseg = data.size
        noverlap = 0
    if noverlap >= nperseg:
        logger.warning(f"noverlap ({noverlap}) >= nperseg ({nperseg}). Setting noverlap = nperseg // 2.")
        noverlap = nperseg // 2

    try:
        # Calculate spectrogram using scipy.signal
        f, t, Sxx = spectrogram(data, fs=sr, window=window, nperseg=nperseg, noverlap=noverlap, scaling='density')
        # Sxx has units of V**2/Hz (power spectral density)
    except ValueError as e:
        logger.error(f"Spectrogram calculation error: {e}")
        raise

    plt.figure(figsize=(10, 6))

    # Convert to dB if requested
    plot_data = Sxx
    colorbar_label = "Power Spectral Density [V^2/Hz]"
    if db_scale:
        # Add epsilon to avoid log10(0)
        epsilon = 1e-10
        plot_data = 10 * np.log10(Sxx + epsilon)
        colorbar_label = "Power Spectral Density [dB/Hz]"

    # Plot using pcolormesh
    # shading='gouraud' provides smoother interpolation, 'auto' or 'flat' are alternatives
    mesh = plt.pcolormesh(t, f, plot_data, shading="gouraud", cmap=cmap, vmin=np.percentile(plot_data, 5) if db_scale else None) # Adjust vmin for dB scale contrast
    plt.colorbar(mesh, label=colorbar_label)

    # Set frequency limits
    f_nyquist = sr / 2
    plt.ylim(f_min if f_min is not None else 0, f_max if f_max is not None else f_nyquist)

    # Labels and title
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3, linestyle="--") # Grid lines for frequency axis
    plt.tight_layout()

    # Save the figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Spectrogram saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save spectrogram plot to {output_file}: {e}")
    finally:
        # Close the plot to free memory, important when generating many plots
        plt.close()


# --- FFT Magnitude Plot ---

def plot_fft_magnitude(
    data: NDArray[np.float64],
    sr: float,
    output_file: str,
    window: Optional[str] = "hann",
    title: str = "FFT Magnitude Spectrum",
    **fft_kwargs: Any
):
    """
    Generate and save an FFT magnitude spectrum plot.

    Shows the magnitude (amplitude) of different frequency components in the signal.

    Args:
        data: Input time-domain signal (1D NumPy array, float64).
        sr: Sampling rate (Hz).
        output_file: Path to save the plot image (e.g., 'fft_magnitude.png').
        window: Window function applied before FFT (e.g., 'hann', None).
        title: Title for the plot.
        **fft_kwargs: Additional arguments passed to `sygnals.core.dsp.compute_fft`
                      (e.g., `n` for FFT length/padding).
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0:
        logger.warning(f"Input data empty. Skipping FFT magnitude plot for {output_file}.")
        return

    logger.info(f"Generating FFT magnitude plot: sr={sr}, output={output_file}")
    try:
        # Compute FFT using the function from dsp module
        freqs, spectrum = compute_fft(data, fs=sr, window=window, **fft_kwargs)
    except Exception as e:
        logger.error(f"FFT computation error: {e}")
        raise

    # Calculate magnitude and select positive frequencies
    magnitude = np.abs(spectrum)
    n = spectrum.size # Use spectrum size (accounts for padding via n in fft_kwargs)
    # Only plot the first half (positive frequencies) for real signals
    positive_freqs_indices = np.where(freqs >= 0)[0]
    positive_freqs = freqs[positive_freqs_indices]
    positive_magnitude = magnitude[positive_freqs_indices]
    # Often only plot up to Nyquist
    nyquist = sr / 2
    plot_indices = np.where(positive_freqs <= nyquist)[0]


    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs[plot_indices], positive_magnitude[plot_indices])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.grid(True, alpha=0.5)
    plt.xlim(0, nyquist) # Limit x-axis to Nyquist frequency
    plt.tight_layout()

    # Save the figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"FFT magnitude plot saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save FFT magnitude plot to {output_file}: {e}")
    finally:
        plt.close()


# --- FFT Phase Plot (New) ---

def plot_fft_phase(
    data: NDArray[np.float64],
    sr: float,
    output_file: str,
    window: Optional[str] = "hann",
    unwrap: bool = True, # Option to unwrap phase
    title: str = "FFT Phase Spectrum",
    **fft_kwargs: Any
):
    """
    Generate and save an FFT phase spectrum plot.

    Shows the phase angle of different frequency components in the signal.

    Args:
        data: Input time-domain signal (1D NumPy array, float64).
        sr: Sampling rate (Hz).
        output_file: Path to save the plot image (e.g., 'fft_phase.png').
        window: Window function applied before FFT (e.g., 'hann', None).
        unwrap: If True (default), unwrap the phase angle to avoid jumps > pi.
        title: Title for the plot.
        **fft_kwargs: Additional arguments passed to `sygnals.core.dsp.compute_fft`
                      (e.g., `n` for FFT length/padding).
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0:
        logger.warning(f"Input data empty. Skipping FFT phase plot for {output_file}.")
        return

    logger.info(f"Generating FFT phase plot: sr={sr}, unwrap={unwrap}, output={output_file}")
    try:
        # Compute FFT using the function from dsp module
        freqs, spectrum = compute_fft(data, fs=sr, window=window, **fft_kwargs)
    except Exception as e:
        logger.error(f"FFT computation error: {e}")
        raise

    # Calculate phase angle
    phase = np.angle(spectrum)
    if unwrap:
        # Unwrap phase to make it continuous (avoids +/- pi jumps)
        phase = np.unwrap(phase)

    # Select positive frequencies
    n = spectrum.size
    positive_freqs_indices = np.where(freqs >= 0)[0]
    positive_freqs = freqs[positive_freqs_indices]
    positive_phase = phase[positive_freqs_indices]
    # Often only plot up to Nyquist
    nyquist = sr / 2
    plot_indices = np.where(positive_freqs <= nyquist)[0]

    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs[plot_indices], positive_phase[plot_indices])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.title(f"{title}{' (Unwrapped)' if unwrap else ''}")
    plt.grid(True, alpha=0.5)
    plt.xlim(0, nyquist) # Limit x-axis to Nyquist frequency
    plt.tight_layout()

    # Save the figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"FFT phase plot saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save FFT phase plot to {output_file}: {e}")
    finally:
        plt.close()


# --- Waveform Plot ---

def plot_waveform(
    data: NDArray[np.float64],
    sr: float,
    output_file: str,
    max_samples: Optional[int] = None,
    title: str = "Waveform"
):
    """
    Generate and save a waveform plot (amplitude vs. time).

    Args:
        data: Input time-domain signal (1D NumPy array, float64).
        sr: Sampling rate (Hz). Used to calculate the time axis.
        output_file: Path to save the plot image (e.g., 'waveform.png').
        max_samples: If set, only plot the first `max_samples` of the data.
                     Useful for very long signals to avoid overly dense plots.
        title: Title for the plot.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0:
        logger.warning(f"Input data empty. Skipping waveform plot for {output_file}.")
        return

    logger.info(f"Generating waveform plot: sr={sr}, output={output_file}")

    # Select data to plot (potentially truncated)
    plot_data = data[:max_samples] if max_samples and data.size > max_samples else data
    num_plot_samples = plot_data.size

    # Create time axis based on sampling rate
    duration = num_plot_samples / sr
    # Use endpoint=False for time axis if representing sample start times
    time_axis = np.linspace(0, duration, num=num_plot_samples, endpoint=False)

    plt.figure(figsize=(12, 4)) # Typically wider than tall
    plt.plot(time_axis, plot_data, linewidth=0.8) # Thinner line often looks better
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.5)
    plt.xlim(0, duration) # Set x-axis limits to the plotted duration
    # Optionally set y-limits based on data range or fixed values (e.g., -1 to 1)
    # y_min, y_max = np.min(plot_data), np.max(plot_data)
    # margin = (y_max - y_min) * 0.1
    # plt.ylim(y_min - margin, y_max + margin)
    plt.tight_layout()

    # Save the figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Waveform plot saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save waveform plot to {output_file}: {e}")
    finally:
        plt.close()


# --- Wavelet Scalogram Plot (New) ---

def plot_scalogram(
    data: NDArray[np.float64],
    output_file: str,
    scales: Union[NDArray[np.float64], int] = 64, # Scales to use or number of scales
    wavelet: str = 'morl', # Common choice for CWT scalograms ('cmorB-C', 'gausN')
    sr: Optional[float] = None, # Optional: for y-axis labeling in Hz
    cmap: str = 'viridis',
    title: str = "Wavelet Scalogram"
):
    """
    Generate and save a Wavelet Transform Scalogram using Continuous Wavelet Transform (CWT).

    A scalogram visualizes the time-varying energy or magnitude of the signal across
    different scales (related to frequency).

    Args:
        data: Input time-domain signal (1D NumPy array, float64).
        output_file: Path to save the plot image (e.g., 'scalogram.png').
        scales: Array of scales to use for CWT, or an integer number of scales.
                If int, logarithmically spaced scales are generated. Smaller scales
                correspond to higher frequencies. (Default: 64 scales).
        wavelet: Name of the continuous wavelet to use (e.g., 'morl', 'cmor1.5-1.0', 'gaus1').
                 See `pywt.wavelist(kind='continuous')`. (Default: 'morl').
        sr: Optional sampling rate (Hz). If provided, the y-axis can be labeled with
            approximate frequencies corresponding to the scales.
        cmap: Matplotlib colormap name (e.g., 'viridis', 'magma', 'jet').
        title: Title for the plot.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0:
        logger.warning(f"Input data empty. Skipping scalogram plot for {output_file}.")
        return

    logger.info(f"Generating scalogram: wavelet={wavelet}, output={output_file}")

    # Determine scales array if an integer number is provided
    if isinstance(scales, int):
        num_scales = scales
        # Define scales logarithmically, e.g., from 1 up to a fraction of data length
        # Max scale choice is heuristic, depends on expected signal features
        max_scale = max(2, data.size // 8)
        scales_arr = np.logspace(np.log10(1), np.log10(max_scale), num=num_scales)
        logger.debug(f"Generated {num_scales} scales logarithmically from 1 to {max_scale:.2f}")
    elif isinstance(scales, (np.ndarray, list)):
         scales_arr = np.asarray(scales)
         logger.debug(f"Using provided scales array (length {len(scales_arr)}).")
    else:
        raise TypeError("scales must be an integer or a NumPy array/list.")

    try:
        # Perform Continuous Wavelet Transform (CWT) using pywt.cwt
        # sampling_period = 1.0/sr if sr else 1.0 # Needed for frequency calculation
        sampling_period = 1.0 / sr if sr is not None and sr > 0 else 1.0
        coeffs, freqs_cwt = pywt.cwt(data, scales_arr, wavelet, sampling_period=sampling_period)
        # coeffs shape: (num_scales, data_length)
    except Exception as e:
        logger.error(f"CWT calculation error: {e}")
        raise

    plt.figure(figsize=(10, 6))
    # Plot magnitude of coefficients |CWT(scale, time)|
    magnitude = np.abs(coeffs)

    # Use imshow for efficient plotting of the 2D array
    # extent defines the coordinates: [time_min, time_max, scale_max, scale_min] (imshow origin is top-left)
    # Use time in samples for x-axis
    time_axis = np.arange(data.size)
    # extent = [time_axis[0], time_axis[-1], scales_arr[-1], scales_arr[0]] # Scales descending
    # Alternative: use pcolormesh for potentially better axis handling with frequencies
    time_mesh, scale_mesh = np.meshgrid(time_axis, scales_arr)

    mesh = plt.pcolormesh(time_mesh, scale_mesh, magnitude, cmap=cmap, shading='gouraud')
    plt.colorbar(mesh, label="Magnitude")

    # Configure axes
    plt.xlabel("Time (samples)")
    if sr and freqs_cwt is not None:
        # Use frequencies calculated by pywt.cwt for y-axis labeling
        # Plotting scale vs time, but label y-axis with corresponding frequencies
        # Often use log scale for frequency/scale axis
        plt.ylabel("Approx. Frequency (Hz)")
        plt.yscale('log') # Log scale is common for frequencies/scales
        # Adjust y-ticks to show frequencies - may need refinement for clarity
        min_freq, max_freq = np.min(freqs_cwt), np.max(freqs_cwt)
        plt.ylim(np.min(scales_arr), np.max(scales_arr)) # Keep scale limits for plotting
        # Add secondary axis or manually set tick labels? Manual ticks might be best.
        num_ticks = 8
        tick_indices = np.logspace(np.log10(0), np.log10(len(scales_arr)-1), num=num_ticks, dtype=int)
        tick_indices = np.clip(tick_indices, 0, len(scales_arr)-1) # Ensure indices are valid
        tick_indices = np.unique(tick_indices) # Remove duplicates
        plt.yticks(scales_arr[tick_indices], [f"{freqs_cwt[i]:.1f}" for i in tick_indices])

    else:
        plt.ylabel("Scale")
        plt.yscale('log') # Log scale is common for scales
        plt.ylim(np.min(scales_arr), np.max(scales_arr))

    plt.title(f"{title} (Wavelet: {wavelet})")
    plt.tight_layout()

    # Save the figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Scalogram plot saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save scalogram plot to {output_file}: {e}")
    finally:
        plt.close()


# --- Placeholder for other plots ---
# def plot_pole_zero(...): pass
# def plot_lissajous(...): pass
# def plot_constellation(...): pass
# def plot_heatmap(...): pass
# def plot_feature_histogram(...): pass
# def plot_feature_scatter(...): pass
