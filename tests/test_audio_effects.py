# tests/test_audio_effects.py

"""
Tests for audio effect functions in sygnals.core.audio.effects.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

# Import effect functions to test
from sygnals.core.audio.effects import (
    simple_dynamic_range_compression,
    pitch_shift,
    time_stretch,
    # Import other effects as they are added (e.g., reverb, delay)
)

# --- Test Fixtures ---

@pytest.fixture
def sample_audio_short():
    """Generate a short sample audio signal."""
    sr = 22050
    duration = 0.5
    freq = 440.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Signal with varying amplitude
    signal = (np.sin(2 * np.pi * freq * t) * np.linspace(0.1, 0.9, len(t))).astype(np.float64)
    return signal, sr

# --- Test Cases ---

def test_simple_dynamic_range_compression(sample_audio_short):
    """Test the simple dynamic range compression effect."""
    signal, sr = sample_audio_short
    threshold = 0.4
    ratio = 3.0
    compressed_signal = simple_dynamic_range_compression(signal, threshold=threshold, ratio=ratio)

    assert compressed_signal.shape == signal.shape
    assert compressed_signal.dtype == np.float64
    # Check that peak amplitude is reduced if original peak was > threshold
    original_peak = np.max(np.abs(signal))
    compressed_peak = np.max(np.abs(compressed_signal))
    if original_peak > threshold:
        assert compressed_peak < original_peak
    # Check that values below threshold are mostly unchanged (allow for float precision)
    below_thresh_indices = np.where(np.abs(signal) <= threshold)[0]
    if below_thresh_indices.size > 0:
        assert_allclose(signal[below_thresh_indices], compressed_signal[below_thresh_indices], atol=1e-7)

def test_pitch_shift(sample_audio_short):
    """Test the pitch shifting effect."""
    signal, sr = sample_audio_short
    n_steps = -3.0 # Shift down by 3 semitones
    shifted_signal = pitch_shift(signal, sr, n_steps=n_steps)

    assert shifted_signal.shape == signal.shape # Pitch shift shouldn't change length significantly
    assert shifted_signal.dtype == np.float64
    # More detailed tests could involve analyzing the frequency content before/after shift

def test_time_stretch(sample_audio_short):
    """Test the time stretching effect."""
    signal, sr = sample_audio_short
    rate = 0.75 # Slow down
    stretched_signal = time_stretch(signal, rate=rate)

    assert stretched_signal.dtype == np.float64
    # Check if the length changed approximately according to the rate
    expected_len = len(signal) / rate
    # Allow some tolerance due to framing/algorithm details
    assert abs(len(stretched_signal) - expected_len) < 0.1 * len(signal)

# --- Add tests for other effects (Reverb, Delay, EQ, etc.) as they are implemented ---

# Example placeholder:
# def test_reverb_effect():
#     # TODO: Implement test for reverb effect
#     pass

# def test_delay_effect():
#     # TODO: Implement test for delay effect
#     pass
