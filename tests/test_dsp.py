# tests/test_dsp.py

import pytest
import numpy as np
from numpy.testing import assert_allclose

# Functions to test are now FFT, IFFT, convolution, windowing
from sygnals.core.dsp import compute_fft, compute_ifft, apply_convolution, apply_window

# --- Test compute_fft ---

def test_compute_fft_sine_wave():
    """Test FFT on a simple sine wave."""
    fs = 1000.0
    duration = 1.0
    freq = 50.0
    t = np.arange(0, duration, 1/fs)
    x = np.sin(2 * np.pi * freq * t)

    freqs, spectrum = compute_fft(x, fs=fs, window=None) # No window for clean sine

    # Find the peak frequency in the positive half of the spectrum
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = np.abs(spectrum[:len(spectrum)//2])
    peak_index = np.argmax(positive_magnitude)
    detected_freq = positive_freqs[peak_index]

    # Check that the peak frequency is close to the expected 50 Hz
    assert abs(detected_freq - freq) < (fs / len(t)), \
        f"Detected peak frequency {detected_freq} Hz is not close to expected {freq} Hz."

def test_compute_fft_padding():
    """Test FFT with zero-padding."""
    fs = 100
    x = np.ones(50) # 50 samples
    n_fft = 128 # Pad to 128 points

    freqs, spectrum = compute_fft(x, fs=fs, n=n_fft, window=None)

    assert len(freqs) == n_fft
    assert len(spectrum) == n_fft
    # Check DC component (should be sum of x)
    assert_allclose(np.abs(spectrum[0]), np.sum(x), atol=1e-7)

def test_compute_fft_windowing():
    """Test that applying a window changes the spectrum."""
    fs = 100
    x = np.ones(50)
    freqs_nowin, spec_nowin = compute_fft(x, fs=fs, window=None)
    freqs_win, spec_win = compute_fft(x, fs=fs, window='hann')

    assert not np.allclose(spec_nowin, spec_win) # Spectra should differ

# --- Test compute_ifft ---

def test_compute_ifft_reconstruction():
    """Test that IFFT reconstructs the original signal."""
    fs = 100
    x = np.random.randn(128) # Random signal
    freqs, spectrum = compute_fft(x, fs=fs, window=None)
    x_reconstructed = compute_ifft(spectrum)

    assert x_reconstructed.shape == x.shape
    assert_allclose(x, x_reconstructed, atol=1e-9, rtol=1e-7) # Allow for numerical precision

# --- Test apply_convolution ---

def test_apply_convolution_simple():
    """Test convolution with a simple kernel."""
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    kernel = np.array([1, 0, -1], dtype=float) # Simple derivative-like kernel
    # Expected output (mode='same'):
    # y[0] = x[0]*0 + x[1]*1 + x[2]*0 = 2 (assuming zero padding outside) - fftconvolve handles boundaries
    # y[1] = x[0]*-1 + x[1]*0 + x[2]*1 = 3 - 1 = 2
    # y[2] = x[1]*-1 + x[2]*0 + x[3]*1 = 4 - 2 = 2
    # y[3] = x[2]*-1 + x[3]*0 + x[4]*1 = 5 - 3 = 2
    # y[4] = x[3]*-1 + x[4]*0 + x[5?]*1 = ? - 4 = ? -> boundary effect
    # Let's check the known middle values
    expected_middle = np.array([2., 2., 2.])
    y = apply_convolution(x, kernel, mode='same')

    assert y.shape == x.shape
    # Check the middle part where boundaries don't affect 'same' mode as much
    # Note: fftconvolve might handle boundaries differently than manual calculation
    # A simpler check might be against np.convolve
    y_np = np.convolve(x, kernel, mode='same')
    assert_allclose(y, y_np, atol=1e-9)


def test_apply_convolution_modes():
    """Test different convolution modes."""
    x = np.array([1, 2, 3])
    kernel = np.array([1, 1])
    y_full = apply_convolution(x, kernel, mode='full') # Expected length 3 + 2 - 1 = 4
    y_same = apply_convolution(x, kernel, mode='same') # Expected length 3
    y_valid = apply_convolution(x, kernel, mode='valid') # Expected length 3 - 2 + 1 = 2

    assert len(y_full) == 4
    assert len(y_same) == 3
    assert len(y_valid) == 2

# --- Test apply_window ---

def test_apply_window_types():
    """Test applying different window types."""
    x = np.ones(100)
    x_hann = apply_window(x, window_type='hann')
    x_hamming = apply_window(x, window_type='hamming')

    assert x_hann.shape == x.shape
    assert x_hamming.shape == x.shape
    assert not np.allclose(x_hann, x) # Window should modify the signal
    assert not np.allclose(x_hann, x_hamming) # Different windows should differ

def test_apply_window_invalid():
    """Test applying an invalid window type."""
    x = np.ones(100)
    with pytest.raises(ValueError):
        apply_window(x, window_type='invalid_window_name')

# --- Test Butterworth Filters (Now in test_filters.py) ---
# These tests should be moved/created in tests/test_filters.py
# def test_low_pass_filter(): ...
# def test_high_pass_filter(): ...
# def test_band_pass_filter(): ...
