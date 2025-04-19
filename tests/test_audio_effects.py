# tests/test_audio_effects.py

"""
Tests for audio effect functions in sygnals.core.audio.effects.
Tests for augmentation effects (noise, pitch, time) are in test_augment.py.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less

# Import effect functions to test (excluding those moved to augment)
from sygnals.core.audio.effects import (
    simple_dynamic_range_compression,
    apply_reverb,
    apply_delay,
    apply_graphic_eq,
    apply_parametric_eq,
    apply_chorus,    # Placeholder
    apply_flanger,   # Placeholder
    apply_tremolo,   # Placeholder
    adjust_gain,
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

@pytest.fixture
def sample_pulse():
    """Generate a short pulse signal."""
    sr = 22050
    duration = 1.0
    signal = np.zeros(int(sr * duration), dtype=np.float64)
    signal[sr//10 : sr//10 + 100] = 0.8 # Short pulse
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

# --- Tests for Reverb and Delay ---

def test_apply_reverb(sample_pulse):
    """Test the basic reverb effect."""
    signal, sr = sample_pulse
    decay_time = 0.3
    wet_level = 0.4
    reverberated_signal = apply_reverb(signal, sr, decay_time=decay_time, wet_level=wet_level)

    assert reverberated_signal.dtype == np.float64
    # Output length should be len(signal) + len(ir) - 1
    # Estimate IR length (can vary slightly)
    expected_ir_len = int(sr * decay_time * 1.5)
    expected_output_len = len(signal) + expected_ir_len - 1
    # Allow some slack in length check due to IR generation details
    assert abs(len(reverberated_signal) - expected_output_len) < sr * 0.1

    # Check if energy persists after the original pulse ends (qualitative)
    pulse_end_idx = sr//10 + 100
    tail_energy = np.mean(reverberated_signal[pulse_end_idx + sr//10:]**2) # Energy later in signal
    original_tail_energy = np.mean(signal[pulse_end_idx + sr//10:]**2)
    assert tail_energy > original_tail_energy # Expect reverb tail energy

def test_apply_delay(sample_pulse):
    """Test the basic delay effect."""
    signal, sr = sample_pulse
    delay_time = 0.15
    feedback = 0.5
    wet_level = 0.6
    delayed_signal = apply_delay(signal, sr, delay_time=delay_time, feedback=feedback, wet_level=wet_level)

    assert delayed_signal.dtype == np.float64
    assert delayed_signal.shape == signal.shape # Delay output length matches input

    # Check for echo peak at the delay time after the original pulse peak
    pulse_peak_idx = sr//10 + 50
    delay_samples = int(delay_time * sr)
    echo_peak_idx = pulse_peak_idx + delay_samples

    # Ensure index is within bounds
    if echo_peak_idx < len(delayed_signal):
         # Find peak around the expected echo time
         search_win = 50 # Samples around expected echo peak
         start = max(0, echo_peak_idx - search_win)
         end = min(len(delayed_signal), echo_peak_idx + search_win)
         # Check that there's significant energy (echo) in the expected region
         assert np.max(np.abs(delayed_signal[start:end])) > 0.1 # Echo should have noticeable amplitude

# --- Tests for EQ (Placeholders / Basic Functionality) ---

@pytest.mark.skip(reason="EQ implementation using iirdesign is experimental/incomplete")
def test_apply_graphic_eq(sample_pulse):
    """Test the graphic EQ (placeholder)."""
    signal, sr = sample_pulse
    band_gains = [(100, -10.0), (1000, 6.0)]
    processed_signal = apply_graphic_eq(signal, sr, band_gains)
    # Placeholder currently returns original signal
    assert_allclose(signal, processed_signal)
    # TODO: Add real checks when EQ is fully implemented (e.g., check frequency response)

@pytest.mark.skip(reason="EQ implementation using iirdesign is experimental/incomplete")
def test_apply_parametric_eq(sample_pulse):
    """Test the parametric EQ (placeholder)."""
    signal, sr = sample_pulse
    params = [
        {'type': 'low_shelf', 'freq': 200, 'gain_db': -6.0},
        # {'type': 'peak', 'freq': 1000, 'gain_db': 3.0, 'q': 1.5}, # Peak skipped
        {'type': 'high_shelf', 'freq': 3000, 'gain_db': 4.0}
    ]
    processed_signal = apply_parametric_eq(signal, sr, params)
    # Placeholder currently returns original signal (or applies only shelf filters if implemented)
    # assert_allclose(signal, processed_signal) # This will fail if shelves are applied
    assert processed_signal.shape == signal.shape # Shape should match
    # TODO: Add real checks when EQ is fully implemented

# --- Tests for Modulation Effects (Placeholders) ---

def test_apply_chorus_placeholder(sample_audio_short):
    """Test the chorus placeholder."""
    signal, sr = sample_audio_short
    processed_signal = apply_chorus(signal, sr)
    assert_allclose(signal, processed_signal) # Placeholder returns original

def test_apply_flanger_placeholder(sample_audio_short):
    """Test the flanger placeholder."""
    signal, sr = sample_audio_short
    processed_signal = apply_flanger(signal, sr)
    assert_allclose(signal, processed_signal) # Placeholder returns original

def test_apply_tremolo_placeholder(sample_audio_short):
    """Test the tremolo placeholder."""
    signal, sr = sample_audio_short
    processed_signal = apply_tremolo(signal, sr)
    assert_allclose(signal, processed_signal) # Placeholder returns original


# --- Tests for Utility Effects ---

def test_adjust_gain(sample_audio_short):
    """Test the adjust_gain utility."""
    signal, sr = sample_audio_short
    original_rms = np.sqrt(np.mean(signal**2))

    # Test amplification (+6 dB approx doubles amplitude/RMS)
    gain_db_amp = 6.0
    amplified_signal = adjust_gain(signal, gain_db=gain_db_amp)
    amplified_rms = np.sqrt(np.mean(amplified_signal**2))
    expected_multiplier_amp = 10**(gain_db_amp / 20.0)
    assert amplified_signal.dtype == np.float64
    assert amplified_signal.shape == signal.shape
    assert np.isclose(amplified_rms, original_rms * expected_multiplier_amp, rtol=1e-3)

    # Test attenuation (-6 dB approx halves amplitude/RMS)
    gain_db_att = -6.0
    attenuated_signal = adjust_gain(signal, gain_db=gain_db_att)
    attenuated_rms = np.sqrt(np.mean(attenuated_signal**2))
    expected_multiplier_att = 10**(gain_db_att / 20.0)
    assert attenuated_signal.dtype == np.float64
    assert attenuated_signal.shape == signal.shape
    assert np.isclose(attenuated_rms, original_rms * expected_multiplier_att, rtol=1e-3)

    # Test zero gain
    zero_gain_signal = adjust_gain(signal, gain_db=0.0)
    assert_allclose(signal, zero_gain_signal, atol=1e-9)

# Removed test_add_noise as it's now in test_augment.py
# Removed test_pitch_shift as it's now in test_augment.py
# Removed test_time_stretch as it's now in test_augment.py
