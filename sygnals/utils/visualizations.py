# sygnals/utils/visualizations.py

"""
Functions for generating various visualizations of signal data.
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import spectrogram

# Import the compute_fft function
from sygnals.core.dsp import compute_fft

logger = logging.getLogger(__name__)


# --- Spectrogram Plot ---

def plot_spectrogram(
    data: NDArray[np.float64],
    sr: float,
    output_file: str,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    window: str = "hann",
    nperseg: Optional[int] = None, # Allow default based on data length
    noverlap: Optional[int] = None,
    db_scale: bool = True,
    cmap: str = 'viridis'
):
    """
    Generate and save a spectrogram plot with configurable parameters.

    Args:
        data: Input audio signal (1D NumPy array, float64).
        sr: Sampling rate of the signal (Hz).
        output_file: Path where to save the plot image (e.g., 'spectrogram.png').
        f_min: Minimum frequency to display (Hz). If None, starts from 0 Hz.
        f_max: Maximum frequency to display (Hz). If None, shows up to Nyquist frequency.
        window: Window function to use for the STFT (default: 'hann').
        nperseg: Length of each segment for the STFT. Defaults to a reasonable value if None.
                 Adjusts time-frequency resolution.
        noverlap: Number of points to overlap between segments. If None, defaults to nperseg // 2.
        db_scale: If True, display power in dB (log scale). If False, display linear power.
        cmap: Colormap for the plot (default: 'viridis').
    """
    if data.ndim != 1:
        raise ValueError("Input data for spectrogram must be a 1D array.")
    if data.size == 0:
        logger.warning("Input data for spectrogram is empty. Skipping plot.")
        return

    logger.info(f"Generating spectrogram for data of length {data.size}, sr={sr}")
    logger.debug(f"Spectrogram parameters: window={window}, nperseg={nperseg}, noverlap={noverlap}, "
                 f"f_min={f_min}, f_max={f_max}, db_scale={db_scale}")

    # Sensible defaults if not provided
    if nperseg is None:
        # Default nperseg based on signal length, aiming for ~20-40ms windows often used in audio
        # Ensure it doesn't exceed data length
        nperseg = min(data.size, int(sr * 0.025)) # ~25ms window if possible
    if noverlap is None:
        noverlap = nperseg // 2

    # Ensure nperseg is not larger than data length after defaults
    if nperseg > data.size:
        logger.warning(f"nperseg ({nperseg}) is greater than input length ({data.size}), "
                       f"using nperseg = {data.size}")
        nperseg = data.size
        noverlap = 0 # No overlap if window is full data length

    try:
        # Calculate spectrogram with specified parameters
        f, t, Sxx = spectrogram(data, fs=sr, window=window, nperseg=nperseg, noverlap=noverlap)
    except ValueError as e:
        logger.error(f"Error calculating spectrogram: {e}")
        # Provide more context if possible (e.g., parameter values)
        logger.error(f"Parameters used: fs={sr}, nperseg={nperseg}, noverlap={noverlap}, window='{window}'")
        raise

    # Create the figure with a specific size
    plt.figure(figsize=(10, 6))

    # Plot the spectrogram
    if db_scale:
        # Convert power to dB, add small epsilon for log stability
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        plt.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap=cmap)
        plt.colorbar(label="Power [dB]")
    else:
        plt.pcolormesh(t, f, Sxx, shading="gouraud", cmap=cmap)
        plt.colorbar(label="Power")


    # Set the frequency limits if specified
    f_nyquist = sr / 2
    actual_f_min = 0 if f_min is None else f_min
    actual_f_max = f_nyquist if f_max is None else f_max
    plt.ylim(actual_f_min, actual_f_max)

    # Customize the plot appearance
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.title("Spectrogram")

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle="--")

    # Tight layout to prevent label clipping
    plt.tight_layout()

    # Save and close
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Spectrogram saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save spectrogram plot to {output_file}: {e}")
    finally:
        plt.close()


# --- FFT Plot ---

def plot_fft(
    data: NDArray[np.float64],
    sr: float,
    output_file: str,
    window: Optional[str] = "hann",
    **fft_kwargs # Allow passing extra args to compute_fft if needed (e.g., n)
):
    """
    Generate and save an FFT magnitude spectrum plot from a raw signal.

    Args:
        data: Input time-domain signal (1D NumPy array, float64).
        sr: Sampling rate of the signal (Hz).
        output_file: Path where to save the plot image (e.g., 'fft_plot.png').
        window: Window function to apply before FFT (default: 'hann'). None for no window.
        **fft_kwargs: Additional keyword arguments passed to `dsp.compute_fft`.
    """
    if data.ndim != 1:
        raise ValueError("Input data for FFT plot must be a 1D array.")
    if data.size == 0:
        logger.warning("Input data for FFT plot is empty. Skipping plot.")
        return

    logger.info(f"Generating FFT plot for data of length {data.size}, sr={sr}")

    # Compute FFT using the core DSP function
    try:
        freqs, spectrum = compute_fft(data, fs=sr, window=window, **fft_kwargs)
    except Exception as e:
         logger.error(f"Error computing FFT for plotting: {e}")
         raise

    # Calculate magnitude
    magnitude = np.abs(spectrum)

    # Only plot the positive frequencies (up to Nyquist)
    n = data.size
    positive_freqs = freqs[:n // 2]
    positive_magnitude = magnitude[:n // 2]

    # Plot the FFT magnitude spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Magnitude Spectrum")
    plt.grid(True, alpha=0.5)
    plt.xlim(0, sr / 2) # Set x-axis limit to Nyquist frequency
    plt.tight_layout()

    # Save and close
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"FFT plot saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save FFT plot to {output_file}: {e}")
    finally:
        plt.close()


# --- Waveform Plot ---

def plot_waveform(
    data: NDArray[np.float64],
    sr: float,
    output_file: str,
    max_samples: Optional[int] = None # Option to limit plotted samples for long signals
):
    """
    Generate and save a waveform plot (amplitude vs. time).

    Args:
        data: Input time-domain signal (1D NumPy array, float64).
        sr: Sampling rate of the signal (Hz).
        output_file: Path where to save the plot image (e.g., 'waveform.png').
        max_samples: If set, only plot the first `max_samples` of the waveform.
                     Useful for very long signals to speed up plotting.
    """
    if data.ndim != 1:
        raise ValueError("Input data for waveform plot must be a 1D array.")
    if data.size == 0:
        logger.warning("Input data for waveform plot is empty. Skipping plot.")
        return

    logger.info(f"Generating waveform plot for data of length {data.size}, sr={sr}")

    # Limit samples if requested
    if max_samples is not None and data.size > max_samples:
        logger.debug(f"Plotting only the first {max_samples} samples of the waveform.")
        plot_data = data[:max_samples]
        duration = max_samples / sr
    else:
        plot_data = data
        duration = data.size / sr

    # Create time axis
    time = np.linspace(0, duration, num=plot_data.size, endpoint=False)

    # Plot the waveform
    plt.figure(figsize=(12, 4)) # Wider aspect ratio often suitable for waveforms
    plt.plot(time, plot_data, linewidth=0.8) # Thinner line can look better
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.grid(True, alpha=0.5)
    plt.xlim(0, duration)
    plt.tight_layout()

    # Save and close
    try:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Waveform plot saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save waveform plot to {output_file}: {e}")
    finally:
        plt.close()
