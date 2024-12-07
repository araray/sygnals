import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram


def plot_spectrogram_old(data, sr, output_file):
    """Generate and save a spectrogram plot."""
    f, t, Sxx = spectrogram(data, sr)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Power [dB]")
    plt.title("Spectrogram")
    plt.savefig(output_file)
    plt.close()


def plot_spectrogram(
    data, sr, output_file, f_min=None, f_max=None, window="hann", nperseg=1024
):
    """
    Generate and save a spectrogram plot with configurable frequency range and window parameters.

    Args:
        data: Input audio signal
        sr: Sampling rate of the signal
        output_file: Path where to save the plot
        f_min: Minimum frequency to display (Hz). If None, starts from 0 Hz
        f_max: Maximum frequency to display (Hz). If None, shows up to Nyquist frequency
        window: Window function to use for the STFT ('hann' by default)
        nperseg: Length of each segment for the STFT (affects time-frequency resolution)
    """
    # Calculate spectrogram with specified parameters
    f, t, Sxx = spectrogram(data, sr, window=window, nperseg=nperseg)

    # Create the figure with a specific size
    plt.figure(figsize=(10, 6))

    # Plot the spectrogram with log scale
    plt.pcolormesh(
        t,
        f,
        10 * np.log10(Sxx + 1e-10),  # Add small constant to prevent log(0)
        shading="gouraud",
    )

    # Set the frequency limits if specified
    if f_min is not None or f_max is not None:
        f_min = 0 if f_min is None else f_min
        f_max = sr / 2 if f_max is None else f_max
        plt.ylim(f_min, f_max)

    # Customize the plot appearance
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Power [dB]")
    plt.title("Spectrogram")

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle="--")

    # Tight layout to prevent label clipping
    plt.tight_layout()

    # Save and close
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fft(data, sr, output_file):
    """Generate and save an FFT plot."""
    n = len(data)
    freqs = np.fft.fftfreq(n, d=1 / sr)
    spectrum = np.fft.fft(data)
    magnitude = np.abs(spectrum)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs[: n // 2], magnitude[: n // 2])  # Only plot positive frequencies
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("FFT Spectrum")
    plt.grid()
    plt.savefig(output_file)
    plt.close()


def plot_waveform(data, sr, output_file):
    """Generate and save a waveform plot."""
    time = np.arange(len(data)) / sr
    plt.figure(figsize=(10, 6))
    plt.plot(time, data)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.grid()
    plt.savefig(output_file)
    plt.close()
