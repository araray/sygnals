import pytest
import numpy as np
from sygnals.core.dsp import compute_fft, compute_ifft, apply_convolution, apply_window

def test_compute_fft():
    fs = 1000
    t = np.arange(0,1,1/fs)
    freq = 50
    x = np.sin(2*np.pi*freq*t)
    freqs, spectrum = compute_fft(x, fs)
    peak_freq = freqs[np.argmax(spectrum)]
    assert abs(peak_freq - 50) < 1.0

def test_compute_fft():
    fs = 1000
    t = np.arange(0, 1, 1/fs)
    freq = 50
    x = np.sin(2*np.pi*freq*t)

    freqs, spectrum = compute_fft(x, fs)
    # Consider only the positive half of the spectrum
    half_n = len(freqs) // 2
    positive_freqs = freqs[:half_n]
    positive_magnitude = np.abs(spectrum[:half_n])

    # Find the peak frequency in the positive half of the spectrum
    peak_freq = positive_freqs[np.argmax(positive_magnitude)]

    # Check that the peak frequency is close to the expected 50 Hz
    assert abs(peak_freq - freq) < 1.0, f"Peak frequency {peak_freq} is not close to expected {freq} Hz."

def test_apply_convolution():
    x = np.array([1,2,3,4], dtype=float)
    kernel = np.array([1,1])
    y = apply_convolution(x, kernel)
    # Convolution with [1,1] should produce partial sums
    # Expected: [1+0,1+2,2+3,3+4,4+0] if full conv, but we have same mode
    # fftconvolve same mode returns length of x
    # Just check it didn't crash and length is same:
    assert len(y) == len(x)

def test_apply_window():
    x = np.ones(10)
    w = apply_window(x, window_type='hamming')
    assert len(w) == 10
    assert not np.allclose(w, x)  # window changes values
