# tests/test_dsp.py

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import librosa # For generating test signals and comparing results

# Functions to test
from sygnals.core.dsp import (
    compute_fft, compute_ifft, apply_convolution, apply_window,
    compute_stft, compute_cqt, compute_correlation, compute_autocorrelation,
    compute_psd_periodogram, compute_psd_welch, amplitude_envelope
)

# --- Test Fixtures ---
@pytest.fixture
def sine_wave():
    """Generate a simple sine wave test signal."""
    fs = 1000.0
    duration = 1.0
    freq = 50.0
    t = np.arange(0, duration, 1/fs)
    # Ensure data is float64
    x = np.sin(2 * np.pi * freq * t).astype(np.float64)
    return x, fs, freq

@pytest.fixture
def chirp_signal():
    """Generate a chirp signal."""
    fs = 8000.0
    duration = 1.0
    f0 = 100.0 # Start frequency
    f1 = 1000.0 # End frequency
    t = np.arange(0, duration, 1/fs)
    # Use fmin instead of f0 for newer librosa versions
    x = librosa.chirp(fmin=f0, f1=f1, sr=fs, duration=duration).astype(np.float64)
    return x, fs

@pytest.fixture
def random_signal():
    """Generate a random noise signal."""
    fs = 1000.0
    duration = 1.0
    # Ensure data is float64
    x = np.random.randn(int(fs * duration)).astype(np.float64)
    return x, fs

# --- Test compute_fft ---
def test_compute_fft_sine_wave(sine_wave):
    """Test FFT computation on a sine wave."""
    x, fs, freq = sine_wave
    freqs, spectrum = compute_fft(x, fs=fs, window=None)
    # Get the positive frequency part of the spectrum
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = np.abs(spectrum[:len(spectrum)//2])
    # Find the frequency bin with the maximum magnitude
    peak_index = np.argmax(positive_magnitude)
    detected_freq = positive_freqs[peak_index]
    # Assert that the detected frequency is close to the expected frequency
    # Tolerance depends on FFT resolution (fs / len(x))
    assert abs(detected_freq - freq) < (fs / len(x)), f"Detected peak {detected_freq} Hz != expected {freq} Hz."

# --- Test compute_ifft ---
def test_compute_ifft_reconstruction(random_signal):
    """Test if IFFT correctly reconstructs the original signal."""
    x, fs = random_signal
    freqs, spectrum = compute_fft(x, fs=fs, window=None)
    x_reconstructed = compute_ifft(spectrum)
    # Check shape and numerical closeness
    assert x_reconstructed.shape == x.shape
    assert_allclose(x, x_reconstructed, atol=1e-9, rtol=1e-7) # High precision expected

# --- Test compute_stft ---
def test_compute_stft(sine_wave):
    """Test STFT computation."""
    x, fs, freq = sine_wave
    n_fft = 512
    stft_matrix = compute_stft(x, n_fft=n_fft, window='hann')
    # Check output shape and type
    assert stft_matrix.shape[0] == 1 + n_fft // 2 # Number of frequency bins
    assert stft_matrix.dtype == np.complex128
    # Check energy concentration around the expected frequency bin
    freq_bins = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    expected_bin = np.argmin(np.abs(freq_bins - freq)) # Find bin closest to sine frequency
    mean_energy_per_frame = np.mean(np.abs(stft_matrix)**2, axis=1) # Average energy across time frames
    assert np.argmax(mean_energy_per_frame) == expected_bin # Max energy should be at the expected bin

# --- Test compute_cqt ---
def test_compute_cqt(chirp_signal):
    """Test CQT computation on a chirp signal."""
    x, fs = chirp_signal
    n_bins = 48 # 4 octaves * 12 bins/octave
    cqt_matrix = compute_cqt(x, sr=fs, n_bins=n_bins, bins_per_octave=12)
    # Check output shape and type
    assert cqt_matrix.shape[0] == n_bins
    assert cqt_matrix.dtype == np.complex128
    # Check that energy follows the chirp frequency over time (qualitative)
    energy_profile = np.argmax(np.abs(cqt_matrix), axis=0) # Index of max energy bin per frame
    # Expect indices to generally increase over time for upward chirp, ignoring edges
    # Allow for small dips (-1) due to algorithm specifics
    assert np.all(np.diff(energy_profile[len(energy_profile)//4:-len(energy_profile)//4]) >= -1)

# --- Test apply_convolution ---
def test_apply_convolution_simple():
    """Test simple 1D convolution."""
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    kernel = np.array([1, 0, -1], dtype=float) # Simple derivative kernel
    y = apply_convolution(x, kernel, mode='same')
    y_np = np.convolve(x, kernel, mode='same') # Compare with numpy's convolution
    assert_allclose(y, y_np, atol=1e-9)

# --- Test Correlation ---
def test_compute_correlation():
    """Test cross-correlation calculation."""
    x = np.array([1, 2, 3, 2, 1], dtype=float)
    y = np.array([0, 1, 2, 1, 0], dtype=float) # y peaks at the same index as x
    corr_full = compute_correlation(x, y, mode='full')
    # Peak should be at lag corresponding to the shift (relative to center)
    # Center index for 'full' mode: len(x) + len(y) - 1 = 9 -> center index 4
    # Since x and y align, peak should be at zero lag (index 4)
    assert np.argmax(corr_full) == 4, f"Peak expected at index 4, found at {np.argmax(corr_full)}"

def test_compute_autocorrelation(sine_wave):
    """Test auto-correlation calculation."""
    x, fs, freq = sine_wave
    autocorr_full = compute_autocorrelation(x, mode='full')
    center_index = len(x) - 1 # Center index for autocorrelation 'full' mode (zero lag)
    # Peak should be at zero lag (center index)
    assert np.argmax(autocorr_full) == center_index
    # Should be periodic with period corresponding to freq
    expected_period_samples = fs / freq
    # Check peak at one period away (approximate due to discretization)
    lag1_index = center_index + int(round(expected_period_samples))
    lag_minus1_index = center_index - int(round(expected_period_samples))
    # Ensure indices are within bounds
    lag1_index = min(lag1_index, len(autocorr_full) - 1)
    lag_minus1_index = max(0, lag_minus1_index)
    # Check if value at lag1 is close to max (allow tolerance)
    assert autocorr_full[lag1_index] > 0.8 * autocorr_full[center_index]
    assert autocorr_full[lag_minus1_index] > 0.8 * autocorr_full[center_index]


# --- Test PSD ---
def test_compute_psd_periodogram(sine_wave):
    """Test PSD calculation using periodogram."""
    x, fs, freq = sine_wave
    f, Pxx = compute_psd_periodogram(x, fs=fs, window='hann')
    assert f.shape == Pxx.shape
    assert f.dtype == np.float64
    assert Pxx.dtype == np.float64
    # Peak frequency should match sine wave frequency
    peak_freq_idx = np.argmax(Pxx)
    assert abs(f[peak_freq_idx] - freq) < 1.0 # Allow some spectral leakage

def test_compute_psd_welch(sine_wave):
    """Test PSD calculation using Welch's method."""
    x, fs, freq = sine_wave
    f, Pxx = compute_psd_welch(x, fs=fs, nperseg=256)
    assert f.shape == Pxx.shape
    assert f.dtype == np.float64
    assert Pxx.dtype == np.float64
    # Peak frequency should match sine wave frequency (Welch is smoother)
    peak_freq_idx = np.argmax(Pxx)
    assert abs(f[peak_freq_idx] - freq) < 1.0

# --- Test Envelope ---
def test_amplitude_envelope_hilbert(sine_wave):
    """Test Hilbert amplitude envelope calculation."""
    # Fixture sine wave has amplitude 1.0
    x, fs, freq = sine_wave
    envelope = amplitude_envelope(x, method='hilbert')
    assert envelope.shape == x.shape
    assert envelope.dtype == np.float64
    # Envelope of sine wave A*sin(wt) should be close to A
    assert_allclose(np.mean(envelope), 1.0, atol=0.05) # Mean should be close to 1.0
    assert np.all(envelope >= 0) # Envelope should be non-negative

def test_amplitude_envelope_rms(sine_wave):
    """Test RMS amplitude envelope calculation."""
    # Fixture sine wave has amplitude 1.0
    x, fs, freq = sine_wave
    frame_length = 256
    hop_length = 128
    envelope = amplitude_envelope(x, method='rms', frame_length=frame_length, hop_length=hop_length)
    assert envelope.dtype == np.float64
    # Check length corresponds to number of frames
    # Use librosa's frame calculation for consistency (center=True is default in rms_energy)
    num_frames = len(librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length, center=True)[0])
    assert len(envelope) == num_frames
    # RMS envelope of sine wave A*sin(wt) should be approx A/sqrt(2)
    expected_rms = 1.0 / np.sqrt(2)
    assert_allclose(np.mean(envelope), expected_rms, atol=0.05)
    assert np.all(envelope >= 0) # RMS should be non-negative

# --- Test apply_window ---
def test_apply_window_types(random_signal):
    """Test applying different window types."""
    x, fs = random_signal
    x_hann = apply_window(x, window_type='hann')
    x_hamming = apply_window(x, window_type='hamming')
    # Check shapes and ensure windowing modified the signal
    assert x_hann.shape == x.shape
    assert x_hamming.shape == x.shape
    assert not np.allclose(x_hann, x) # Windowed signal should differ from original
    assert not np.allclose(x_hann, x_hamming) # Different windows should produce different results
