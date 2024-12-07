import numpy as np
from scipy.signal import butter, cheby1, firwin, lfilter


# Butterworth Filter
def butterworth_filter(data, cutoff, fs, order=5, filter_type="low"):
    """Apply a Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return lfilter(b, a, data)


# Chebyshev Filter
def chebyshev_filter(data, cutoff, fs, order=5, ripple=0.05, filter_type="low"):
    """Apply a Chebyshev type I filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = cheby1(order, ripple, normal_cutoff, btype=filter_type, analog=False)
    return lfilter(b, a, data)


# FIR Filter
def fir_filter(data, num_taps, cutoff, fs, filter_type="low"):
    """Design and apply an FIR filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    taps = firwin(num_taps, normal_cutoff, pass_zero=(filter_type == "low"))
    return lfilter(taps, 1.0, data)


def low_pass_filter(data, cutoff, fs, order=5):
    """Apply a low-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, data)


def high_pass_filter(data, cutoff, fs, order=5):
    """Apply a high-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return lfilter(b, a, data)
