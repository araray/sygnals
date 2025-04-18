# tests/test_features_freq.py

"""
Tests for frequency-domain feature extraction functions in
sygnals.core.features.frequency_domain. These typically operate
on the magnitude spectrum of a single analysis frame.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less

# Import feature functions to test
from sygnals.core.features.frequency_domain import (
    spectral_centroid,
    spectral_bandwidth,
    spectral_flatness,
    spectral_rolloff,
    dominant_frequency,
    # spectral_contrast # Note: spectral_contrast operates on spectrogram, test elsewhere or add fixture
)

# --- Test Fixtures ---

@pytest.fixture
def sample_spectrum_sine():
    """Generate a sample magnitude spectrum resembling a sine wave peak."""
    sr = 1000.0
    n_fft = 1024
    freq_target = 100.0 # Target frequency peak
    # Generate frequencies for one-sided spectrum
    frequencies = np.fft.rfftfreq(n_fft, d=1.0/sr) # Frequencies from 0 to Nyquist
    magnitude_spectrum = np.zeros_like(frequencies, dtype=np.float64)
    # Create a peak around the target frequency
    peak_idx = np.argmin(np.abs(frequencies - freq_target))
    # Simple triangular peak for testing
    peak_width = 5 # Bins
    start_idx = max(0, peak_idx - peak_width)
    end_idx = min(len(frequencies), peak_idx + peak_width + 1)
    magnitude_spectrum[start_idx:end_idx] = np.linspace(0, 1, end_idx - start_idx)**2 # Example shape
    magnitude_spectrum[start_idx:peak_idx+1] = np.linspace(0, 1, peak_idx - start_idx + 1)**2
    magnitude_spectrum[peak_idx+1:end_idx] = np.linspace(1, 0, end_idx - (peak_idx+1))**2 *0.9 # Asymmetric peak

    return magnitude_spectrum, frequencies, freq_target

@pytest.fixture
def sample_spectrum_flat():
    """Generate a relatively flat magnitude spectrum (like noise)."""
    sr = 1000.0
    n_fft = 1024
    frequencies = np.fft.rfftfreq(n_fft, d=1.0/sr)
    # Constant magnitude, add small noise
    magnitude_spectrum = (np.ones_like(frequencies) + np.random.rand(len(frequencies))*0.1).astype(np.float64)
    # Zero out DC component for some tests if needed
    # magnitude_spectrum[0] = 0
    return magnitude_spectrum, frequencies

# --- Test Cases ---

def test_spectral_centroid(sample_spectrum_sine, sample_spectrum_flat):
    """Test the spectral_centroid calculation."""
    mag_sine, freqs_sine, target_freq_sine = sample_spectrum_sine
    mag_flat, freqs_flat = sample_spectrum_flat

    # Centroid for sine-like spectrum should be close to the peak frequency
    centroid_sine = spectral_centroid(mag_sine, freqs_sine)
    assert isinstance(centroid_sine, np.float64)
    assert pytest.approx(centroid_sine, rel=0.1) == target_freq_sine # Allow 10% relative error

    # Centroid for flat spectrum should be roughly in the middle of the frequency range
    centroid_flat = spectral_centroid(mag_flat, freqs_flat)
    nyquist = freqs_flat[-1]
    assert nyquist * 0.4 < centroid_flat < nyquist * 0.6 # Expect centroid around center

    # Test zero spectrum
    centroid_zero = spectral_centroid(np.zeros_like(freqs_sine), freqs_sine)
    assert centroid_zero == 0.0

def test_spectral_bandwidth(sample_spectrum_sine, sample_spectrum_flat):
    """Test the spectral_bandwidth calculation (using p=2, std dev)."""
    mag_sine, freqs_sine, target_freq_sine = sample_spectrum_sine
    mag_flat, freqs_flat = sample_spectrum_flat

    # Bandwidth for sine-like spectrum should be relatively small
    bw_sine = spectral_bandwidth(mag_sine, freqs_sine, p=2)
    assert isinstance(bw_sine, np.float64)
    assert bw_sine > 0 # Should not be zero unless single peak
    assert bw_sine < target_freq_sine * 0.5 # Expect bandwidth smaller than peak freq

    # Bandwidth for flat spectrum should be relatively large
    bw_flat = spectral_bandwidth(mag_flat, freqs_flat, p=2)
    nyquist = freqs_flat[-1]
    assert bw_flat > nyquist * 0.2 # Expect significant portion of Nyquist range

    # Test zero spectrum
    bw_zero = spectral_bandwidth(np.zeros_like(freqs_sine), freqs_sine)
    assert bw_zero == 0.0

def test_spectral_flatness(sample_spectrum_sine, sample_spectrum_flat):
    """Test the spectral_flatness calculation."""
    mag_sine, freqs_sine, target_freq_sine = sample_spectrum_sine
    mag_flat, freqs_flat = sample_spectrum_flat

    # Flatness for sine-like (peaky) spectrum should be low (closer to 0)
    flatness_sine = spectral_flatness(mag_sine)
    assert isinstance(flatness_sine, np.float64)
    assert 0.0 <= flatness_sine <= 1.0
    assert flatness_sine < 0.1 # Expect low flatness for peaky spectrum

    # Flatness for flat spectrum should be high (closer to 1)
    flatness_flat = spectral_flatness(mag_flat)
    assert 0.0 <= flatness_flat <= 1.0
    assert flatness_flat > 0.8 # Expect high flatness

    # Test zero spectrum (handle potential division by zero)
    flatness_zero = spectral_flatness(np.zeros_like(freqs_sine))
    assert flatness_zero == 0.0

def test_spectral_rolloff(sample_spectrum_sine, sample_spectrum_flat):
    """Test the spectral_rolloff calculation."""
    mag_sine, freqs_sine, target_freq_sine = sample_spectrum_sine
    mag_flat, freqs_flat = sample_spectrum_flat

    # Rolloff for sine-like spectrum (85%) should be slightly above the peak frequency
    rolloff_sine_85 = spectral_rolloff(mag_sine, freqs_sine, roll_percent=0.85)
    assert isinstance(rolloff_sine_85, np.float64)
    assert rolloff_sine_85 >= target_freq_sine # Should be at or above peak

    # Rolloff for sine-like spectrum (10%) should be near the start of the peak
    rolloff_sine_10 = spectral_rolloff(mag_sine, freqs_sine, roll_percent=0.10)
    assert rolloff_sine_10 <= target_freq_sine

    # Rolloff for flat spectrum (85%) should be near 85% of the Nyquist frequency
    rolloff_flat_85 = spectral_rolloff(mag_flat, freqs_flat, roll_percent=0.85)
    nyquist = freqs_flat[-1]
    assert pytest.approx(rolloff_flat_85, rel=0.15) == nyquist * 0.85

    # Test zero spectrum
    rolloff_zero = spectral_rolloff(np.zeros_like(freqs_sine), freqs_sine)
    assert rolloff_zero == freqs_sine[-1] # Should return max frequency

def test_dominant_frequency(sample_spectrum_sine, sample_spectrum_flat):
    """Test the dominant_frequency calculation."""
    mag_sine, freqs_sine, target_freq_sine = sample_spectrum_sine
    mag_flat, freqs_flat = sample_spectrum_flat

    # Dominant frequency for sine-like spectrum should be the peak frequency
    dom_freq_sine = dominant_frequency(mag_sine, freqs_sine)
    assert isinstance(dom_freq_sine, np.float64)
    assert dom_freq_sine == freqs_sine[np.argmax(mag_sine)]
    # Check if it's close to the intended target frequency
    assert pytest.approx(dom_freq_sine) == target_freq_sine

    # Dominant frequency for flat spectrum might be anywhere (due to noise)
    dom_freq_flat = dominant_frequency(mag_flat, freqs_flat)
    assert dom_freq_flat in freqs_flat # Should be one of the frequencies

    # Test zero spectrum
    dom_freq_zero = dominant_frequency(np.zeros_like(freqs_sine), freqs_sine)
    # argmax returns 0 for all zeros
    assert dom_freq_zero == freqs_sine[0]


# --- Test for spectral_contrast (requires spectrogram) ---
# Placeholder: This feature needs a spectrogram input, testing might fit better
# in test_features_manager.py where spectrograms are generated, or requires
# a dedicated fixture here.
# def test_spectral_contrast():
#     # TODO: Implement test for spectral_contrast
#     # Needs a fixture that provides a sample spectrogram (S) and sr
#     pass
