import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

def plot_spectrogram(data, sr, output_file):
    """Generate and save a spectrogram plot."""
    f, t, Sxx = spectrogram(data, sr)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Power [dB]')
    plt.title('Spectrogram')
    plt.savefig(output_file)
    plt.close()

def plot_fft(data, sr, output_file):
    """Generate and save an FFT plot."""
    n = len(data)
    freqs = np.fft.fftfreq(n, d=1/sr)
    spectrum = np.fft.fft(data)
    magnitude = np.abs(spectrum)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs[:n // 2], magnitude[:n // 2])  # Only plot positive frequencies
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Spectrum')
    plt.grid()
    plt.savefig(output_file)
    plt.close()

def plot_waveform(data, sr, output_file):
    """Generate and save a waveform plot."""
    time = np.arange(len(data)) / sr
    plt.figure(figsize=(10, 6))
    plt.plot(time, data)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.grid()
    plt.savefig(output_file)
    plt.close()
