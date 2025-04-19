# sygnals/utils/visualizations.py

"""
Functions for generating various visualizations of signal data using Matplotlib.

Provides functions to plot waveforms, spectrograms, FFT magnitude/phase spectra,
and wavelet scalograms. Ensures plots are saved correctly and memory is managed.
"""

import logging
# Import necessary types
from typing import Any, Optional, Tuple, Union, Literal

import matplotlib.pyplot as plt
import numpy as np
import pywt # For Continuous Wavelet Transform (CWT) used in scalogram
from numpy.typing import NDArray
# Import specific signal processing functions
from scipy.signal import spectrogram # For spectrogram calculation
# Use compute_fft from the dsp module for consistency
from ..core.dsp import compute_fft # Relative import assumes visualizations is called from within sygnals package context

logger = logging.getLogger(__name__)

# Define a small epsilon for log calculations
_EPSILON = np.finfo(np.float64).eps

# --- Spectrogram Plot ---

def plot_spectrogram(
    data: NDArray[np.float64],
    sr: float,
    output_file: str,
    f_min: Optional[float] = 0.0, # Default f_min to 0
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

    Raises:
        ValueError: If input data is not 1D.
        Exception: For errors during spectrogram calculation or plotting/saving.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0:
        logger.warning(f"Input data empty. Skipping spectrogram plot for {output_file}.")
        return

    fig = None # Initialize fig to None
    try:
        logger.info(f"Generating spectrogram: sr={sr}, output={output_file}")
        logger.debug(f"Params: window={window}, nperseg={nperseg}, noverlap={noverlap}, f_min={f_min}, f_max={f_max}, db={db_scale}")

        # Sensible defaults for segment length and overlap
        default_nperseg = min(data.size, int(sr * 0.025)) # Default ~25ms window, capped by data length
        nperseg_calc = nperseg if nperseg is not None else default_nperseg
        # Ensure nperseg is at least 1
        nperseg_calc = max(1, nperseg_calc)
        # Default overlap based on calculated nperseg
        noverlap_calc = noverlap if noverlap is not None else nperseg_calc // 2

        # Ensure nperseg and noverlap are valid given data size
        if nperseg_calc > data.size:
            logger.warning(f"nperseg ({nperseg_calc}) > data size ({data.size}). Setting nperseg=data size, noverlap=0.")
            nperseg_calc = data.size
            noverlap_calc = 0
        # Ensure overlap is less than segment length
        noverlap_calc = min(noverlap_calc, nperseg_calc - 1) if nperseg_calc > 0 else 0


        logger.debug(f"Calculated spectrogram params: nperseg={nperseg_calc}, noverlap={noverlap_calc}")

        # Calculate spectrogram using scipy.signal
        f, t, Sxx = spectrogram(
            data, fs=sr, window=window, nperseg=nperseg_calc, noverlap=noverlap_calc, scaling='density'
        )
        # Sxx has units of V**2/Hz (power spectral density)

        fig = plt.figure(figsize=(10, 6))

        # Convert to dB if requested
        plot_data = Sxx
        colorbar_label = "Power Spectral Density [V^2/Hz]"
        if db_scale:
            # Add epsilon to avoid log10(0)
            plot_data = 10 * np.log10(Sxx + _EPSILON)
            colorbar_label = "Power Spectral Density [dB/Hz]"
            # Adjust vmin for better contrast on dB scale
            vmin = np.percentile(plot_data, 5)
            vmax = np.percentile(plot_data, 99) # Also set vmax for better range
        else:
             vmin=None
             vmax=None

        # Plot using pcolormesh
        mesh = plt.pcolormesh(t, f, plot_data, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(mesh, label=colorbar_label)

        # Set frequency limits
        f_nyquist = sr / 2.0
        f_min_plot = f_min if f_min is not None else 0.0
        f_max_plot = f_max if f_max is not None else f_nyquist
        plt.ylim(max(0.0, f_min_plot), min(f_nyquist, f_max_plot)) # Ensure limits are valid

        # Labels and title
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.title(title)
        plt.grid(True, axis='y', alpha=0.3, linestyle="--") # Grid lines for frequency axis
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Spectrogram saved to {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate/save spectrogram plot to {output_file}: {e}")
        # Re-raise exception if needed, or just log
        # raise # Uncomment to propagate error
    finally:
        # Close the plot to free memory, important when generating many plots
        if fig is not None:
             plt.close(fig)


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
    Raises:
        ValueError: If input data is not 1D.
        Exception: For errors during FFT calculation or plotting/saving.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0:
        logger.warning(f"Input data empty. Skipping FFT magnitude plot for {output_file}.")
        return

    fig = None
    try:
        logger.info(f"Generating FFT magnitude plot: sr={sr}, output={output_file}")
        # Compute FFT using the function from dsp module
        freqs, spectrum = compute_fft(data, fs=sr, window=window, **fft_kwargs)

        # Calculate magnitude and select positive frequencies up to Nyquist
        magnitude = np.abs(spectrum)
        nyquist = sr / 2.0
        # Find indices for frequencies >= 0 and <= Nyquist
        plot_indices = np.where((freqs >= 0) & (freqs <= nyquist))[0]
        plot_freqs = freqs[plot_indices]
        plot_magnitude = magnitude[plot_indices]

        if plot_freqs.size == 0:
            logger.warning(f"No valid positive frequencies to plot for {output_file}. Skipping plot.")
            return

        fig = plt.figure(figsize=(10, 6))
        plt.plot(plot_freqs, plot_magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(title)
        plt.grid(True, alpha=0.5)
        plt.xlim(0, nyquist) # Limit x-axis to Nyquist frequency
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"FFT magnitude plot saved to {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate/save FFT magnitude plot to {output_file}: {e}")
    finally:
        if fig is not None:
            plt.close(fig)


# --- FFT Phase Plot ---

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

    Shows the phase angle (in radians) of different frequency components in the signal.

    Args:
        data: Input time-domain signal (1D NumPy array, float64).
        sr: Sampling rate (Hz).
        output_file: Path to save the plot image (e.g., 'fft_phase.png').
        window: Window function applied before FFT (e.g., 'hann', None).
        unwrap: If True (default), unwrap the phase angle using `np.unwrap`
                to make it continuous (avoids +/- pi jumps).
        title: Title for the plot.
        **fft_kwargs: Additional arguments passed to `sygnals.core.dsp.compute_fft`
                      (e.g., `n` for FFT length/padding).
    Raises:
        ValueError: If input data is not 1D.
        Exception: For errors during FFT calculation or plotting/saving.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0:
        logger.warning(f"Input data empty. Skipping FFT phase plot for {output_file}.")
        return

    fig = None
    try:
        logger.info(f"Generating FFT phase plot: sr={sr}, unwrap={unwrap}, output={output_file}")
        # Compute FFT using the function from dsp module
        freqs, spectrum = compute_fft(data, fs=sr, window=window, **fft_kwargs)

        # Calculate phase angle
        phase = np.angle(spectrum)
        if unwrap:
            # Unwrap phase to make it continuous
            phase = np.unwrap(phase)

        # Select positive frequencies up to Nyquist
        nyquist = sr / 2.0
        plot_indices = np.where((freqs >= 0) & (freqs <= nyquist))[0]
        plot_freqs = freqs[plot_indices]
        plot_phase = phase[plot_indices]

        if plot_freqs.size == 0:
            logger.warning(f"No valid positive frequencies to plot for {output_file}. Skipping plot.")
            return

        fig = plt.figure(figsize=(10, 6))
        plt.plot(plot_freqs, plot_phase)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (radians)")
        plt.title(f"{title}{' (Unwrapped)' if unwrap else ''}")
        plt.grid(True, alpha=0.5)
        plt.xlim(0, nyquist) # Limit x-axis to Nyquist frequency
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"FFT phase plot saved to {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate/save FFT phase plot to {output_file}: {e}")
    finally:
        if fig is not None:
            plt.close(fig)


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

    Raises:
        ValueError: If input data is not 1D.
        Exception: For errors during plotting or saving.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0:
        logger.warning(f"Input data empty. Skipping waveform plot for {output_file}.")
        return

    fig = None
    try:
        logger.info(f"Generating waveform plot: sr={sr}, output={output_file}")

        # Select data to plot (potentially truncated)
        plot_data = data[:max_samples] if max_samples and data.size > max_samples else data
        num_plot_samples = plot_data.size

        if num_plot_samples == 0: # Check again after potential slicing
            logger.warning(f"No samples to plot for waveform {output_file}. Skipping.")
            return

        # Create time axis based on sampling rate
        # Duration calculation needs care: num_plot_samples / sr is total duration
        # endpoint=False means the last time point is just before the next sample would start
        time_axis = np.arange(num_plot_samples) / sr

        fig = plt.figure(figsize=(12, 4)) # Typically wider than tall
        plt.plot(time_axis, plot_data, linewidth=0.8) # Thinner line often looks better
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True, alpha=0.5)
        plt.xlim(0, time_axis[-1] if time_axis.size > 0 else 0) # Set x-axis limits to the plotted duration
        # Optionally set y-limits based on data range or fixed values (e.g., -1 to 1)
        y_min, y_max = np.min(plot_data), np.max(plot_data)
        margin = (y_max - y_min) * 0.1 + _EPSILON # Add epsilon for constant signals
        plt.ylim(y_min - margin, y_max + margin)
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Waveform plot saved to {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate/save waveform plot to {output_file}: {e}")
    finally:
        if fig is not None:
            plt.close(fig)


# --- Wavelet Scalogram Plot ---

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
    different scales (related to frequency). Lower scales correspond to higher frequencies.

    Args:
        data: Input time-domain signal (1D NumPy array, float64).
        output_file: Path to save the plot image (e.g., 'scalogram.png').
        scales: Array of scales to use for CWT, or an integer number of scales.
                If int, logarithmically spaced scales are generated. Smaller scales
                correspond to higher frequencies. (Default: 64 scales).
        wavelet: Name of the continuous wavelet to use (e.g., 'morl', 'cmor1.5-1.0', 'gaus1').
                 See `pywt.wavelist(kind='continuous')`. (Default: 'morl').
        sr: Optional sampling rate (Hz). If provided, the y-axis can be labeled with
            approximate frequencies corresponding to the scales using `pywt.scale2frequency`.
        cmap: Matplotlib colormap name (e.g., 'viridis', 'magma', 'jet').
        title: Title for the plot.

    Raises:
        ValueError: If input data is not 1D or scales/wavelet are invalid.
        Exception: For errors during CWT calculation or plotting/saving.
    """
    if data.ndim != 1: raise ValueError("Input data must be 1D.")
    if data.size == 0:
        logger.warning(f"Input data empty. Skipping scalogram plot for {output_file}.")
        return

    fig = None
    try:
        logger.info(f"Generating scalogram: wavelet={wavelet}, output={output_file}")

        # Determine scales array if an integer number is provided
        if isinstance(scales, int):
            num_scales = max(1, scales) # Ensure at least 1 scale
            # Define scales logarithmically, e.g., from 1 up to a fraction of data length
            # Max scale choice is heuristic, depends on expected signal features
            # Ensure max_scale > min_scale (which is 1)
            max_scale = max(2.0, data.size / 8.0)
            # Use np.geomspace for log spacing (more direct than logspace(log10(...)))
            scales_arr = np.geomspace(1.0, max_scale, num=num_scales)
            logger.debug(f"Generated {num_scales} scales geometrically from 1 to {max_scale:.2f}")
        elif isinstance(scales, (np.ndarray, list)):
             scales_arr = np.asarray(scales)
             if scales_arr.size == 0 or np.any(scales_arr <= 0):
                 raise ValueError("Scales array must not be empty and contain only positive values.")
             logger.debug(f"Using provided scales array (length {len(scales_arr)}).")
        else:
            raise TypeError("scales must be an integer or a NumPy array/list.")

        # Perform Continuous Wavelet Transform (CWT) using pywt.cwt
        sampling_period = 1.0 / sr if sr is not None and sr > 0 else 1.0
        coeffs, freqs_cwt = pywt.cwt(data, scales_arr, wavelet, sampling_period=sampling_period)
        # coeffs shape: (num_scales, data_length)

        fig = plt.figure(figsize=(10, 6))
        # Plot magnitude of coefficients |CWT(scale, time)|
        magnitude = np.abs(coeffs)

        # Use imshow or pcolormesh. pcolormesh allows non-uniform axes.
        time_axis = np.arange(data.size) / sr if sr else np.arange(data.size)
        xlabel = "Time (seconds)" if sr else "Time (samples)"

        # Choose y-axis: scales or frequencies
        if sr and freqs_cwt is not None:
            # Use frequencies on y-axis (log scale often preferred)
            y_axis = freqs_cwt
            ylabel = "Frequency (Hz)"
            yscale = 'log'
            # Ensure frequencies are positive for log scale
            valid_freq_indices = np.where(y_axis > 0)[0]
            if len(valid_freq_indices) < len(y_axis):
                logger.warning("Some CWT frequencies are zero or negative, cannot plot on log scale. Using linear scale.")
                yscale = 'linear'
                # Or filter data? For now, just switch scale.
                # y_axis = y_axis[valid_freq_indices]
                # magnitude = magnitude[valid_freq_indices, :]
        else:
            # Use scales on y-axis (log scale often preferred)
            y_axis = scales_arr
            ylabel = "Scale"
            yscale = 'log'

        # Need meshgrid for pcolormesh if using time/frequency axes directly
        time_mesh, y_mesh = np.meshgrid(time_axis, y_axis)

        # Handle potential empty magnitude after frequency filtering
        if magnitude.size == 0:
             logger.warning(f"No valid data points to plot for scalogram {output_file}. Skipping plot.")
             return

        mesh = plt.pcolormesh(time_mesh, y_mesh, magnitude, cmap=cmap, shading='gouraud')
        plt.colorbar(mesh, label="Magnitude")

        # Configure axes
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale(yscale)
        # Adjust limits slightly to avoid issues with log scale if min freq/scale is very small
        plt.ylim(np.min(y_axis) * 0.9, np.max(y_axis) * 1.1) if y_axis.size > 0 else None
        plt.xlim(time_axis[0], time_axis[-1]) if time_axis.size > 0 else None
        plt.title(f"{title} (Wavelet: {wavelet})")
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Scalogram plot saved to {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate/save scalogram plot to {output_file}: {e}")
    finally:
        if fig is not None:
            plt.close(fig)


# --- Placeholder for other plots ---
# def plot_pole_zero(...): pass
# def plot_lissajous(...): pass
# def plot_constellation(...): pass
# def plot_heatmap(...): pass
# def plot_feature_histogram(...): pass
# def plot_feature_scatter(...): pass
