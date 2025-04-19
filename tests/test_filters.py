# tests/test_filters.py

import pytest
import numpy as np
from numpy.testing import assert_allclose

# Functions to test
from sygnals.core.filters import (
    design_butterworth_sos, apply_sos_filter,
    low_pass_filter, high_pass_filter, band_pass_filter, band_stop_filter
)
# Import compute_fft from dsp for testing filter effects
from sygnals.core.dsp import compute_fft
# Import sosfreqz for analyzing SOS filter response
from scipy.signal import sosfreqz, get_window

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
# test_design_butterworth_sos_lowpass remains the same as the previous version
def test_design_butterworth_sos_lowpass():
    """Test designing a lowpass Butterworth filter and check its frequency response."""
    fs = 1000.0
    cutoff = 100.0
    order = 4 # Use a standard order for design check
    sos = design_butterworth_sos(cutoff, fs, order, 'lowpass')
    assert isinstance(sos, np.ndarray), "SOS output should be a NumPy array."
    assert sos.shape[1] == 6, "SOS format must have 6 coefficients per section."
    assert sos.shape[0] == order // 2, "Number of sections should match filter order."

    # Check frequency response using sosfreqz (correct function for SOS)
    worN = 2000 # Number of frequency points
    w, h = sosfreqz(sos, worN=worN, fs=fs) # Use sosfreqz

    # Ensure w and h are numpy arrays and have expected shape
    assert isinstance(w, np.ndarray), "Frequency vector w should be a NumPy array."
    assert isinstance(h, np.ndarray), "Response vector h should be a NumPy array."
    assert w.shape == (worN,), f"Frequency vector w has incorrect shape: {w.shape}"
    assert h.shape == (worN,), f"Response vector h has incorrect shape: {h.shape}"

    response_db = 20 * np.log10(np.abs(h) + 1e-10) # Add epsilon for stability

    # Check passband (e.g., at cutoff/2) - should be close to 0 dB
    passband_freq_idx = np.argmin(np.abs(w - cutoff / 2))
    assert -1.0 < response_db[passband_freq_idx] < 0.1, \
        f"Passband response at {w[passband_freq_idx]:.1f} Hz is {response_db[passband_freq_idx]:.2f} dB, expected close to 0 dB."

    # Check cutoff (-3 dB point)
    cutoff_freq_idx = np.argmin(np.abs(w - cutoff))
    assert -4.0 < response_db[cutoff_freq_idx] < -2.0, \
        f"Cutoff response at {w[cutoff_freq_idx]:.1f} Hz is {response_db[cutoff_freq_idx]:.2f} dB, expected approx -3 dB."

    # Check stopband (e.g., at cutoff*2) - should be significantly attenuated
    stopband_freq_idx = np.argmin(np.abs(w - cutoff * 2))
    # For order 4, expect reasonable attenuation, e.g., > 15dB
    assert response_db[stopband_freq_idx] < -15.0, \
        f"Stopband response at {w[stopband_freq_idx]:.1f} Hz is {response_db[stopband_freq_idx]:.2f} dB, expected < -15 dB."

# test_design_butterworth_sos_invalid_cutoff remains the same
def test_design_butterworth_sos_invalid_cutoff():
    """Test filter design with invalid cutoff frequencies."""
    fs = 1000.0
    order = 4
    # Cutoff >= Nyquist
    with pytest.raises(ValueError, match="strictly between 0 and Nyquist"):
        design_butterworth_sos(500.0, fs, order, 'lowpass') # Nyquist is 500
    with pytest.raises(ValueError, match="strictly between 0 and Nyquist"):
        design_butterworth_sos(600.0, fs, order, 'lowpass')
    # Low cutoff >= high cutoff
    with pytest.raises(ValueError, match="Low cutoff .* must be less than high cutoff"):
        design_butterworth_sos((200.0, 100.0), fs, order, 'bandpass')
    # Invalid cutoff type
    with pytest.raises(TypeError, match="cutoff must be a float .* or a tuple"):
        design_butterworth_sos([100.0], fs, order, 'lowpass')

# --- Test Filter Application ---
# test_apply_sos_filter remains the same
def test_apply_sos_filter(sample_signal):
    """Test applying a designed SOS filter using zero-phase filtering."""
    x, fs = sample_signal
    # Design a lowpass filter
    sos = design_butterworth_sos(cutoff=100.0, fs=fs, order=6, filter_type='lowpass')
    y = apply_sos_filter(sos, x)

    assert y.shape == x.shape, "Output shape must match input shape."
    assert y.dtype == np.float64, "Output dtype should be float64."
    # Check that output is different from input (filter had an effect)
    assert not np.allclose(x, y), "Filtered output should not be identical to input."
    # Check basic properties: output power should be less than input power
    # because higher frequencies are attenuated by the lowpass filter.
    assert np.sum(y**2) < np.sum(x**2), "Output power should be less than input power for lowpass filter."

# --- Test Convenience Functions ---
FILTER_TEST_ORDER = 8 # Keep reduced order

def test_low_pass_filter(sample_signal):
    """Test the low_pass_filter convenience function for attenuation."""
    x, fs = sample_signal # Components at 50, 150, 300 Hz
    cutoff = 100.0
    filter_order = FILTER_TEST_ORDER # Use reduced order
    y = low_pass_filter(x, cutoff=cutoff, fs=fs, order=filter_order)

    # --- Primary Check: Overall RMS Reduction ---
    rms_x = np.sqrt(np.mean(x**2))
    rms_y = np.sqrt(np.mean(y**2))
    assert rms_y < rms_x, f"Lowpass filter failed: Output RMS ({rms_y:.4f}) >= Input RMS ({rms_x:.4f})"

    # --- Secondary Check: Specific Component Attenuation ---
    freqs_x, spec_x = compute_fft(x, fs=fs, window=None)
    freqs_y, spec_y = compute_fft(y, fs=fs, window=None)
    mag_x = np.abs(spec_x[:len(spec_x)//2])
    mag_y = np.abs(spec_y[:len(spec_y)//2])
    freqs = freqs_x[:len(freqs_x)//2]

    idx_50 = np.argmin(np.abs(freqs - 50)) # Should pass
    idx_150 = np.argmin(np.abs(freqs - 150)) # Should be attenuated
    idx_300 = np.argmin(np.abs(freqs - 300)) # Should be strongly attenuated

    assert mag_y[idx_50] > 0.8 * mag_x[idx_50], "50 Hz component significantly attenuated by lowpass filter."
    # Relaxed attenuation checks for order 8
    assert mag_y[idx_150] < 0.5 * mag_x[idx_150], f"150 Hz component not sufficiently attenuated ({mag_y[idx_150]:.4f} vs {mag_x[idx_150]:.4f}). Expected < 0.5x"
    assert mag_y[idx_300] < 0.2 * mag_x[idx_300], f"300 Hz component not sufficiently attenuated ({mag_y[idx_300]:.4f} vs {mag_x[idx_300]:.4f}). Expected < 0.2x"

def test_high_pass_filter(sample_signal):
    """Test the high_pass_filter convenience function for attenuation."""
    x, fs = sample_signal # Components at 50, 150, 300 Hz
    cutoff = 100.0 # Cutoff frequency
    filter_order = FILTER_TEST_ORDER # Use reduced order
    y = high_pass_filter(x, cutoff=cutoff, fs=fs, order=filter_order)

    # --- Primary Check: Overall RMS Reduction ---
    rms_x = np.sqrt(np.mean(x**2))
    rms_y = np.sqrt(np.mean(y**2))
    assert rms_y < rms_x, f"Highpass filter failed: Output RMS ({rms_y:.4f}) >= Input RMS ({rms_x:.4f})"

    # --- Secondary Check: Specific Component Attenuation ---
    freqs_x, spec_x = compute_fft(x, fs=fs, window=None)
    freqs_y, spec_y = compute_fft(y, fs=fs, window=None)
    mag_x = np.abs(spec_x[:len(spec_x)//2])
    mag_y = np.abs(spec_y[:len(spec_y)//2])
    freqs = freqs_x[:len(freqs_x)//2]

    idx_50 = np.argmin(np.abs(freqs - 50)) # Should be attenuated
    idx_150 = np.argmin(np.abs(freqs - 150)) # Should pass
    idx_300 = np.argmin(np.abs(freqs - 300)) # Should pass

    # Relaxed attenuation checks for order 8
    assert mag_y[idx_50] < 0.2 * mag_x[idx_50], f"50 Hz component not sufficiently attenuated ({mag_y[idx_50]:.4f} vs {mag_x[idx_50]:.4f}). Expected < 0.2x"
    assert mag_y[idx_150] > 0.8 * mag_x[idx_150], "150 Hz component significantly attenuated by highpass filter."
    assert mag_y[idx_300] > 0.8 * mag_x[idx_300], "300 Hz component significantly attenuated by highpass filter."


def test_band_pass_filter(sample_signal):
    """Test the band_pass_filter convenience function."""
    x, fs = sample_signal # Components at 50, 150, 300 Hz
    low_cutoff = 100.0
    high_cutoff = 200.0
    filter_order = FILTER_TEST_ORDER // 2 # Effective order 8
    y = band_pass_filter(x, low_cutoff=low_cutoff, high_cutoff=high_cutoff, fs=fs, order=filter_order)

    # --- Primary Check: Overall RMS Reduction (compared to input) ---
    # Bandpass will remove energy, so output RMS should be lower
    rms_x = np.sqrt(np.mean(x**2))
    rms_y = np.sqrt(np.mean(y**2))
    assert rms_y < rms_x, f"Bandpass filter failed: Output RMS ({rms_y:.4f}) >= Input RMS ({rms_x:.4f})"

    # --- Secondary Check: Specific Component Attenuation/Passing ---
    freqs_x, spec_x = compute_fft(x, fs=fs, window=None)
    freqs_y, spec_y = compute_fft(y, fs=fs, window=None)
    mag_x = np.abs(spec_x[:len(spec_x)//2])
    mag_y = np.abs(spec_y[:len(spec_y)//2])
    freqs = freqs_x[:len(freqs_x)//2]

    idx_50 = np.argmin(np.abs(freqs - 50)) # Should be attenuated
    idx_150 = np.argmin(np.abs(freqs - 150)) # Should pass
    idx_300 = np.argmin(np.abs(freqs - 300)) # Should be attenuated

    # Relaxed attenuation checks for effective order 8
    assert mag_y[idx_50] < 0.2 * mag_x[idx_50], f"50 Hz component not sufficiently attenuated ({mag_y[idx_50]:.4f} vs {mag_x[idx_50]:.4f}). Expected < 0.2x"
    assert mag_y[idx_150] > 0.7 * mag_x[idx_150], "150 Hz component significantly attenuated by bandpass filter."
    assert mag_y[idx_300] < 0.2 * mag_x[idx_300], f"300 Hz component not sufficiently attenuated ({mag_y[idx_300]:.4f} vs {mag_x[idx_300]:.4f}). Expected < 0.2x"

def test_band_stop_filter(sample_signal):
    """Test the band_stop_filter convenience function."""
    x, fs = sample_signal # Components at 50, 150, 300 Hz
    low_cutoff = 100.0
    high_cutoff = 200.0
    filter_order = FILTER_TEST_ORDER // 2 # Effective order 8
    y = band_stop_filter(x, low_cutoff=low_cutoff, high_cutoff=high_cutoff, fs=fs, order=filter_order)

    # --- Primary Check: Overall RMS Reduction (compared to input) ---
    # Bandstop will remove energy (the 150Hz component), so output RMS should be lower
    rms_x = np.sqrt(np.mean(x**2))
    rms_y = np.sqrt(np.mean(y**2))
    assert rms_y < rms_x, f"Bandstop filter failed: Output RMS ({rms_y:.4f}) >= Input RMS ({rms_x:.4f})"

    # --- Secondary Check: Specific Component Attenuation/Passing ---
    freqs_x, spec_x = compute_fft(x, fs=fs, window=None)
    freqs_y, spec_y = compute_fft(y, fs=fs, window=None)
    mag_x = np.abs(spec_x[:len(spec_x)//2])
    mag_y = np.abs(spec_y[:len(spec_y)//2])
    freqs = freqs_x[:len(freqs_x)//2]

    idx_50 = np.argmin(np.abs(freqs - 50)) # Should pass
    idx_150 = np.argmin(np.abs(freqs - 150)) # Should be attenuated
    idx_300 = np.argmin(np.abs(freqs - 300)) # Should pass

    # Relaxed attenuation checks for effective order 8
    assert mag_y[idx_50] > 0.7 * mag_x[idx_50], "50 Hz component significantly attenuated by bandstop filter."
    assert mag_y[idx_150] < 0.2 * mag_x[idx_150], f"150 Hz component not sufficiently attenuated ({mag_y[idx_150]:.4f} vs {mag_x[idx_150]:.4f}). Expected < 0.2x"
    assert mag_y[idx_300] > 0.7 * mag_x[idx_300], "300 Hz component significantly attenuated by bandstop filter."
