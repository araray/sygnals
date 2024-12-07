import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import butter, fftconvolve, get_window, lfilter


# FFT-related functions
def compute_fft(data, fs=1):
    """Compute FFT and return frequency and magnitude."""
    n = len(data)
    freqs = np.fft.fftfreq(n, d=1/fs)
    spectrum = fft(data)
    return freqs, spectrum


def compute_ifft(spectrum):
    """Compute inverse FFT from a spectrum."""
    return ifft(spectrum).real


# Filtering-related functions
def butterworth_filter(data, cutoff, fs, order=5, filter_type="low"):
    """Apply a Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return lfilter(b, a, data)


def low_pass_filter(data, cutoff, fs, order=5):
    """Apply a low-pass Butterworth filter."""
    return butterworth_filter(data, cutoff, fs, order, "low")


def high_pass_filter(data, cutoff, fs, order=5):
    """Apply a high-pass Butterworth filter."""
    return butterworth_filter(data, cutoff, fs, order, "high")


def band_pass_filter(data, low_cutoff, high_cutoff, fs, order=5):
    """Apply a band-pass Butterworth filter."""
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype="band", analog=False)
    return lfilter(b, a, data)


# Convolution-related functions
def apply_convolution(data, kernel):
    """Apply convolution using a custom kernel."""
    return fftconvolve(data, kernel, mode="same")


# Window functions
def apply_window(data, window_type="hamming"):
    """Apply a window function to the data."""
    window = get_window(window_type, len(data))
    return data * window
