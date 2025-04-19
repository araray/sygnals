# tests/test_audio_features.py

"""
Tests for audio feature extraction functions in sygnals.core.audio.features.
"""

import pytest
import numpy as np
import librosa # For generating test signals and reference calculations
from numpy.testing import assert_allclose, assert_equal, assert_array_less

# Import feature functions to test
from sygnals.core.audio.features import (
    zero_crossing_rate,
    rms_energy,
    fundamental_frequency,
    get_basic_audio_metrics,
    detect_onsets,
    harmonic_to_noise_ratio, # Placeholder
    jitter,                  # Placeholder
    shimmer                  # Placeholder
)

# --- Test Fixtures ---

@pytest.fixture
def sine_wave_audio():
    """Generate a sine wave test signal."""
    sr = 22050
    duration = 1.0
    freq = 440.0 # A4 note
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (0.7 * np.sin(2 * np.pi * freq * t)).astype(np.float64)
    # Return all three values
    return signal, sr, freq

@pytest.fixture
def silent_audio():
    """Generate a silent audio signal."""
    sr = 22050
    duration = 1.0
    signal = np.zeros(int(sr * duration), dtype=np.float64)
    return signal, sr

@pytest.fixture
def noise_audio():
    """Generate white noise."""
    sr = 22050
    duration = 1.0
    signal = (np.random.rand(int(sr*duration)) * 0.2 - 0.1).astype(np.float64)
    return signal, sr

# --- Test Cases ---

def test_zero_crossing_rate(sine_wave_audio, silent_audio, noise_audio):
    """Test the zero_crossing_rate feature."""
    # Unpack fixture correctly
    signal_sine, sr, freq = sine_wave_audio
    signal_silent, _ = silent_audio
    signal_noise, _ = noise_audio

    # Parameters for feature extraction
    frame_length = 1024
    hop_length = 512

    # ZCR for sine wave (should be low and consistent)
    zcr_sine = zero_crossing_rate(signal_sine, frame_length=frame_length, hop_length=hop_length)
    assert zcr_sine.ndim == 1
    assert zcr_sine.dtype == np.float64
    assert np.all(zcr_sine >= 0)
    expected_zcr_sine = 2 * freq / sr # Theoretical ZCR for pure sine
    # Check if average ZCR is close to theoretical (allow tolerance for framing)
    assert np.abs(np.mean(zcr_sine) - expected_zcr_sine) < 0.05

    # ZCR for silence (should be zero)
    zcr_silent = zero_crossing_rate(signal_silent, frame_length=frame_length, hop_length=hop_length)
    assert_allclose(zcr_silent, 0.0, atol=1e-7)

    # ZCR for noise (should be relatively high)
    zcr_noise = zero_crossing_rate(signal_noise, frame_length=frame_length, hop_length=hop_length)
    assert np.mean(zcr_noise) > 0.1 # Expect higher ZCR for noise compared to sine


def test_rms_energy(sine_wave_audio, silent_audio):
    """Test the rms_energy feature."""
    # Unpack fixture correctly
    signal_sine, sr, freq = sine_wave_audio
    signal_silent, _ = silent_audio

    frame_length = 1024
    hop_length = 512

    # RMS for sine wave (should be related to amplitude)
    rms_sine = rms_energy(signal_sine, frame_length=frame_length, hop_length=hop_length)
    assert rms_sine.ndim == 1
    assert rms_sine.dtype == np.float64
    assert np.all(rms_sine >= 0)
    # Expected RMS for A*sin(wt) is A/sqrt(2)
    amplitude = 0.7 # From fixture
    expected_rms_sine = amplitude / np.sqrt(2)
    assert_allclose(np.mean(rms_sine), expected_rms_sine, atol=0.05) # Check average RMS

    # RMS for silence (should be zero)
    rms_silent = rms_energy(signal_silent, frame_length=frame_length, hop_length=hop_length)
    assert_allclose(rms_silent, 0.0, atol=1e-7)


def test_fundamental_frequency(sine_wave_audio, silent_audio):
    """Test the fundamental_frequency (pitch) estimation."""
    # Unpack fixture correctly
    signal_sine, sr, freq = sine_wave_audio
    signal_silent, _ = silent_audio

    # Test with pyin
    times_p, f0_p, vf_p, vp_p = fundamental_frequency(signal_sine, sr=sr, method='pyin')
    assert times_p.ndim == 1 and f0_p.ndim == 1 and vf_p.ndim == 1 and vp_p.ndim == 1
    assert f0_p.dtype == np.float64 # Check output type
    # Check if estimated pitch is close to the actual frequency for voiced frames
    voiced_indices = np.where(vf_p > 0.5)[0] # Get indices where voiced flag is 1
    assert len(voiced_indices) > 0 # Sine wave should be mostly voiced
    assert_allclose(f0_p[voiced_indices], freq, rtol=0.1) # Allow 10% tolerance for pitch estimation

    # Test with yin
    times_y, f0_y, vf_y, vp_y = fundamental_frequency(signal_sine, sr=sr, method='yin')
    voiced_indices_y = np.where(vf_y > 0.5)[0]
    assert len(voiced_indices_y) > 0
    assert_allclose(f0_y[voiced_indices_y], freq, rtol=0.1)

    # Test on silence (should be mostly unvoiced, f0 near zero)
    times_s, f0_s, vf_s, vp_s = fundamental_frequency(signal_silent, sr=sr, method='pyin')
    assert np.sum(vf_s) < 0.1 * len(vf_s) # Expect very few voiced frames
    assert_allclose(f0_s, 0.0, atol=1.0) # F0 should be near zero (or NaN replaced by 0)


def test_get_basic_audio_metrics(sine_wave_audio, silent_audio):
    """Test the get_basic_audio_metrics function."""
    # Unpack fixture correctly
    signal_sine, sr, freq = sine_wave_audio
    signal_silent, _ = silent_audio
    amplitude = 0.7 # From fixture

    metrics_sine = get_basic_audio_metrics(signal_sine, sr)
    assert isinstance(metrics_sine, dict)
    assert "duration_seconds" in metrics_sine
    assert "rms_global" in metrics_sine
    assert "peak_amplitude" in metrics_sine
    assert np.isclose(metrics_sine["duration_seconds"], 1.0)
    assert np.isclose(metrics_sine["rms_global"], amplitude / np.sqrt(2), atol=1e-3)
    assert np.isclose(metrics_sine["peak_amplitude"], amplitude, atol=1e-3)

    metrics_silent = get_basic_audio_metrics(signal_silent, sr)
    assert np.isclose(metrics_silent["duration_seconds"], 1.0)
    assert np.isclose(metrics_silent["rms_global"], 0.0)
    assert np.isclose(metrics_silent["peak_amplitude"], 0.0)


def test_detect_onsets(sine_wave_audio):
    """Test the onset detection feature."""
    # Create a signal with clear onsets
    # Unpack sr from fixture
    _, sr, _ = sine_wave_audio
    clicks = librosa.clicks(times=[0.2, 0.5, 0.8], sr=sr, length=int(sr*1.2))
    signal = (clicks * 0.5).astype(np.float64)

    onset_frames = detect_onsets(signal, sr=sr, backtrack=False) # Use backtrack=False for simpler check
    assert onset_frames.ndim == 1
    assert onset_frames.dtype == np.int64

    # Convert frames to times
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Check if detected times are close to the actual click times
    expected_times = np.array([0.2, 0.5, 0.8])
    assert len(onset_times) == len(expected_times)
    assert_allclose(onset_times, expected_times, atol=0.05) # Allow 50ms tolerance


# --- Tests for Placeholder Features ---

def test_hnr_placeholder(sine_wave_audio):
    """Test the harmonic_to_noise_ratio placeholder."""
    # Unpack fixture correctly, ignore frequency
    signal, sr, _ = sine_wave_audio
    frame_length = 1024
    hop_length = 512
    hnr_result = harmonic_to_noise_ratio(signal, sr, frame_length=frame_length, hop_length=hop_length)
    assert hnr_result.ndim == 1
    assert hnr_result.dtype == np.float64
    # Check that all values are NaN
    assert np.all(np.isnan(hnr_result))
    # Check length matches expected number of frames
    expected_num_frames = len(librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length, center=True)[0])
    assert len(hnr_result) == expected_num_frames


def test_jitter_placeholder(sine_wave_audio):
    """Test the jitter placeholder."""
    # Unpack fixture correctly, ignore frequency
    signal, sr, _ = sine_wave_audio
    frame_length = 1024
    hop_length = 512
    jitter_result = jitter(signal, sr, frame_length=frame_length, hop_length=hop_length)
    assert jitter_result.ndim == 1
    assert jitter_result.dtype == np.float64
    # Check that all values are NaN
    assert np.all(np.isnan(jitter_result))
    # Check length matches expected number of frames
    expected_num_frames = len(librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length, center=True)[0])
    assert len(jitter_result) == expected_num_frames


def test_shimmer_placeholder(sine_wave_audio):
    """Test the shimmer placeholder."""
    # Unpack fixture correctly, ignore frequency
    signal, sr, _ = sine_wave_audio
    frame_length = 1024
    hop_length = 512
    shimmer_result = shimmer(signal, sr, frame_length=frame_length, hop_length=hop_length)
    assert shimmer_result.ndim == 1
    assert shimmer_result.dtype == np.float64
    # Check that all values are NaN
    assert np.all(np.isnan(shimmer_result))
    # Check length matches expected number of frames
    expected_num_frames = len(librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length, center=True)[0])
    assert len(shimmer_result) == expected_num_frames


# --- Add tests for other audio features as implemented ---
