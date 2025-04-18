# tests/test_filters.py

import pytest
import numpy as np
from numpy.testing import assert_allclose

# Functions to test
from sygnals.core.filters import (
    design_butterworth_sos, apply_sos_filter,
    low_pass_filter, high_pass_filter, band_pass_filter, band_stop_filter
)
from scipy.signal import freqz # For checking filter frequency response

# --- Test Fixtures ---
@pytest.fixture
def sample_signal():
    """Generate a signal with multiple frequency components."""
    fs = 1000.0
    duration = 2.0
    t = np.arange(0, duration, 1/fs)
    # Components at 50 Hz, 150 Hz, 300 Hz
    x = (
        1.0 * np.sin(2 * np.pi * 50 * t) +
        0.5 * np.sin(2 * np.pi * 150 * t) +
        0.2 * np.sin(2 * np.pi * 300 * t)
    ).astype(np.float64)
    return x, fs

# --- Test Filter Design ---
def test_design_butterworth_sos_lowpass():
    """Test designing a lowpass Butterworth filter."""
    fs = 1000.0
    cutoff = 100.0
    order = 4
    sos = design_butterworth_sos(cutoff, fs, order, 'lowpass')
    assert isinstance(sos, np.ndarray)
    assert sos.shape[1] == 6 # SOS format has 6 coefficients per section
    assert sos.shape[0] == order // 2 # Number of sections for Butterworth

    # Check frequency response (optional but good)
    w, h = freqz(sos, worN=2000, fs=fs, output='sos')
    response_db = 20 * np.log10(np.abs(h) + 1e-10)
    # Check passband (e.g., at cutoff/2) - should be close to 0 dB
    passband_freq_idx = np.argmin(np.abs(w - cutoff / 2))
    assert -1.0 < response_db[passband_freq_idx] < 0.1 # Allow slight ripple/attenuation
    # Check cutoff (-3 dB point)
    cutoff_freq_idx = np.argmin(np.abs(w - cutoff))
    assert -4.0 < response_db[cutoff_freq_idx] < -2.0 # Should be close to -3dB
    # Check stopband (e.g., at cutoff*2) - should be significantly attenuated
    stopband_freq_idx = np.argmin(np.abs(w - cutoff * 2))
    assert response_db[stopband_freq_idx] < -20.0 # Example attenuation check

def test_design_butterworth_sos_invalid_cutoff():
    """Test design with invalid cutoff frequencies."""
    fs = 1000.0
    order = 4
    # Cutoff >= Nyquist
    with pytest.raises(ValueError, match="must be between 0 and Nyquist"):
        design_butterworth_sos(600.0, fs, order, 'lowpass')
    # Low cutoff >= high cutoff
    with pytest.raises(ValueError, match="Low cutoff .* must be less than high cutoff"):
        design_butterworth_sos((200.0, 100.0), fs, order, 'bandpass')

# --- Test Filter Application ---
def test_apply_sos_filter(sample_signal):
    """Test applying a designed SOS filter."""
    x, fs = sample_signal
    # Design a lowpass filter
    sos = design_butterworth_sos(cutoff=100.0, fs=fs, order=6, filter_type='lowpass')
    y = apply_sos_filter(sos, x)

    assert y.shape == x.shape
    assert y.dtype == np.float64
    # Check that output is different from input
    assert not np.allclose(x, y)
    # Check basic properties: output power should be less than input power
    # because higher frequencies are attenuated
    assert np.sum(y**2) < np.sum(x**2)

# --- Test Convenience Functions ---
def test_low_pass_filter(sample_signal):
    """Test the low_pass_filter convenience function."""
    x, fs = sample_signal
    cutoff = 100.0
    y = low_pass_filter(x, cutoff=cutoff, fs=fs, order=6)

    # Verify high frequencies are attenuated
    # FFT of input vs output
    freqs_x, spec_x = compute_fft(x, fs=fs, window=None)
    freqs_y, spec_y = compute_fft(y, fs=fs, window=None)
    mag_x = np.abs(spec_x[:len(spec_x)//2])
    mag_y = np.abs(spec_y[:len(spec_y)//2])
    freqs = freqs_x[:len(freqs_x)//2]

    # Find indices for frequencies above cutoff (e.g., > 120 Hz)
    high_freq_indices = np.where(freqs > cutoff + 20)[0]
    # Average magnitude in high frequencies should be much lower in output
    avg_mag_x_high = np.mean(mag_x[high_freq_indices])
    avg_mag_y_high = np.mean(mag_y[high_freq_indices])
    assert avg_mag_y_high < 0.1 * avg_mag_x_high # Expect significant attenuation

def test_high_pass_filter(sample_signal):
    """Test the high_pass_filter convenience function."""
    x, fs = sample_signal
    cutoff = 200.0
    y = high_pass_filter(x, cutoff=cutoff, fs=fs, order=6)

    # Verify low frequencies are attenuated
    freqs_x, spec_x = compute_fft(x, fs=fs, window=None)
    freqs_y, spec_y = compute_fft(y, fs=fs, window=None)
    mag_x = np.abs(spec_x[:len(spec_x)//2])
    mag_y = np.abs(spec_y[:len(spec_y)//2])
    freqs = freqs_x[:len(freqs_x)//2]

    # Find indices for frequencies below cutoff (e.g., < 180 Hz)
    low_freq_indices = np.where(freqs < cutoff - 20)[0]
    avg_mag_x_low = np.mean(mag_x[low_freq_indices])
    avg_mag_y_low = np.mean(mag_y[low_freq_indices])
    assert avg_mag_y_low < 0.1 * avg_mag_x_low

def test_band_pass_filter(sample_signal):
    """Test the band_pass_filter convenience function."""
    x, fs = sample_signal # Components at 50, 150, 300 Hz
    low_cutoff = 100.0
    high_cutoff = 200.0
    y = band_pass_filter(x, low_cutoff=low_cutoff, high_cutoff=high_cutoff, fs=fs, order=6)

    # Verify frequencies outside the passband are attenuated
    freqs_x, spec_x = compute_fft(x, fs=fs, window=None)
    freqs_y, spec_y = compute_fft(y, fs=fs, window=None)
    mag_x = np.abs(spec_x[:len(spec_x)//2])
    mag_y = np.abs(spec_y[:len(spec_y)//2])
    freqs = freqs_x[:len(freqs_x)//2]

    # Check 50 Hz component (should be attenuated)
    idx_50 = np.argmin(np.abs(freqs - 50))
    assert mag_y[idx_50] < 0.2 * mag_x[idx_50]
    # Check 150 Hz component (should pass)
    idx_150 = np.argmin(np.abs(freqs - 150))
    assert mag_y[idx_150] > 0.7 * mag_x[idx_150] # Allow some passband ripple/attenuation
    # Check 300 Hz component (should be attenuated)
    idx_300 = np.argmin(np.abs(freqs - 300))
    assert mag_y[idx_300] < 0.2 * mag_x[idx_300]

def test_band_stop_filter(sample_signal):
    """Test the band_stop_filter convenience function."""
    x, fs = sample_signal # Components at 50, 150, 300 Hz
    low_cutoff = 100.0
    high_cutoff = 200.0
    y = band_stop_filter(x, low_cutoff=low_cutoff, high_cutoff=high_cutoff, fs=fs, order=6)

    # Verify frequencies inside the stopband are attenuated
    freqs_x, spec_x = compute_fft(x, fs=fs, window=None)
    freqs_y, spec_y = compute_fft(y, fs=fs, window=None)
    mag_x = np.abs(spec_x[:len(spec_x)//2])
    mag_y = np.abs(spec_y[:len(spec_y)//2])
    freqs = freqs_x[:len(freqs_x)//2]

    # Check 50 Hz component (should pass)
    idx_50 = np.argmin(np.abs(freqs - 50))
    assert mag_y[idx_50] > 0.7 * mag_x[idx_50]
    # Check 150 Hz component (should be attenuated)
    idx_150 = np.argmin(np.abs(freqs - 150))
    assert mag_y[idx_150] < 0.2 * mag_x[idx_150]
    # Check 300 Hz component (should pass)
    idx_300 = np.argmin(np.abs(freqs - 300))
    assert mag_y[idx_300] > 0.7 * mag_x[idx_300]


# Helper function (used in tests above)
def compute_fft(data, fs, window=None):
    """Simplified FFT for testing filters."""
    if window:
        win = get_window(window, len(data))
        data = data * win
    spectrum = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1/fs)
    return freqs, spectrum
