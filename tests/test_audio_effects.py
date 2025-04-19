# tests/test_audio_effects.py

"""
Tests for audio effect functions in sygnals.core.audio.effects.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less

# Import effect functions to test
from sygnals.core.audio.effects import (
    simple_dynamic_range_compression,
    pitch_shift,
    time_stretch,
    apply_reverb,
    apply_delay,
    apply_graphic_eq,
    apply_parametric_eq,
    apply_chorus,
    apply_flanger,
    apply_tremolo,
    adjust_gain,
    add_noise,
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


def test_add_noise(sample_audio_short):
    """Test the add_noise utility."""
    signal, sr = sample_audio_short
    signal_power = np.mean(signal**2)

    # Test adding noise with specific SNR
    snr_db = 10.0
    noisy_signal_g = add_noise(signal, snr_db=snr_db, noise_type='gaussian', seed=42)
    assert noisy_signal_g.dtype == np.float64
    assert noisy_signal_g.shape == signal.shape
    # Verify SNR approximately matches target
    noise_added_g = noisy_signal_g - signal
    noise_power_g = np.mean(noise_added_g**2)
    # Handle potential zero signal power case although fixture avoids it
    actual_snr_db_g = 10 * np.log10(signal_power / noise_power_g) if signal_power > 1e-15 and noise_power_g > 1e-15 else -np.inf
    assert np.isclose(actual_snr_db_g, snr_db, atol=1.0) # Allow 1 dB tolerance

    # Test reproducibility with seed
    noisy_signal_g_seed1 = add_noise(signal, snr_db=snr_db, noise_type='gaussian', seed=42)
    noisy_signal_g_seed2 = add_noise(signal, snr_db=snr_db, noise_type='gaussian', seed=123)
    assert_allclose(noisy_signal_g, noisy_signal_g_seed1)
    assert not np.allclose(noisy_signal_g, noisy_signal_g_seed2)

    # Test placeholder noise types (should currently add white noise and warn)
    with pytest.warns(UserWarning, match="Pink noise generation is currently a placeholder"):
         noisy_signal_p = add_noise(signal, snr_db=snr_db, noise_type='pink')
         assert noisy_signal_p.shape == signal.shape # Check basic execution

    with pytest.warns(UserWarning, match="Brown noise generation is currently a placeholder"):
         noisy_signal_b = add_noise(signal, snr_db=snr_db, noise_type='brown')
         assert noisy_signal_b.shape == signal.shape # Check basic execution

    # Test invalid noise type
    with pytest.raises(ValueError, match="Invalid noise_type"):
        add_noise(signal, snr_db=snr_db, noise_type='invalid')
