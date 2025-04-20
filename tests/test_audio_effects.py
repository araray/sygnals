# tests/test_audio_effects.py

"""
Tests for audio effect functions in sygnals.core.audio.effects.
Tests for augmentation effects (noise, pitch, time) are in test_augment.py.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less, assert_raises
import librosa # For HPSS in tests

# Import effect functions to test
from sygnals.core.audio.effects import (
    simple_dynamic_range_compression,
    apply_reverb,
    apply_delay,
    apply_graphic_eq,      # Still marked as experimental
    apply_parametric_eq,   # Still marked as experimental
    apply_chorus,          # Implemented
    apply_flanger,         # Implemented
    apply_tremolo,         # Implemented
    adjust_gain,
    noise_reduction_spectral, # Added
    transient_shaping_hpss,   # Added
    stereo_widening_midside   # Added
)
# Import compute_fft for spectral checks if needed
from sygnals.core.dsp import compute_fft

# --- Test Fixtures ---

@pytest.fixture
def sample_audio_short():
    """Generate a short sample audio signal (mono)."""
    sr = 22050
    duration = 0.5
    freq = 440.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Signal with varying amplitude
    signal = (np.sin(2 * np.pi * freq * t) * np.linspace(0.1, 0.9, len(t))).astype(np.float64)
    return signal, sr

@pytest.fixture
def sample_pulse():
    """Generate a short pulse signal (mono)."""
    sr = 22050
    duration = 1.0
    signal = np.zeros(int(sr * duration), dtype=np.float64)
    signal[sr//10 : sr//10 + 100] = 0.8 # Short pulse
    return signal, sr

@pytest.fixture
def sample_audio_stereo():
    """Generate a simple stereo audio signal."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Create slightly different left and right channels
    left = (0.6 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64)
    right = (0.5 * np.sin(2 * np.pi * 445.0 * t + np.pi/4)).astype(np.float64) # Different freq/phase
    signal_stereo = np.stack([left, right], axis=0) # Shape (2, n_samples)
    return signal_stereo, sr

@pytest.fixture
def sample_noisy_signal():
    """Generate a signal with initial noise followed by a sine wave."""
    sr = 22050
    noise_dur = 0.5
    signal_dur = 1.0
    total_dur = noise_dur + signal_dur
    rng = np.random.default_rng(123)
    # Noise segment
    noise = rng.normal(0, 0.05, int(sr * noise_dur)).astype(np.float64)
    # Signal segment
    t_signal = np.linspace(0, signal_dur, int(sr * signal_dur), endpoint=False)
    signal_part = (0.6 * np.sin(2 * np.pi * 660.0 * t_signal)).astype(np.float64)
    # Combine
    full_signal = np.concatenate([noise, signal_part])
    return full_signal, sr, noise_dur

@pytest.fixture
def sample_transient_signal():
    """Generate a signal with transients (clicks)."""
    sr = 22050
    duration = 1.5
    clicks = librosa.clicks(times=[0.3, 0.7, 1.1], sr=sr, length=int(sr*duration), click_freq=1000, click_duration=0.05)
    # Add some background sine wave
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    background = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float64)
    signal = (clicks + background).astype(np.float64)
    return signal, sr


# --- Test Cases ---

# Existing tests for Compression, Reverb, Delay, Modulation, Gain...
# (Keep previous tests here, omitted for brevity)
def test_simple_dynamic_range_compression(sample_audio_short):
    signal, sr = sample_audio_short
    threshold = 0.4; ratio = 3.0
    compressed_signal = simple_dynamic_range_compression(signal, threshold=threshold, ratio=ratio)
    assert compressed_signal.shape == signal.shape and compressed_signal.dtype == np.float64
    original_peak = np.max(np.abs(signal)); compressed_peak = np.max(np.abs(compressed_signal))
    if original_peak > threshold: assert compressed_peak < original_peak

def test_apply_reverb(sample_pulse):
    signal, sr = sample_pulse; decay_time = 0.3; wet_level = 0.4
    reverberated_signal = apply_reverb(signal, sr, decay_time=decay_time, wet_level=wet_level)
    assert reverberated_signal.dtype == np.float64
    expected_ir_len = int(sr * decay_time * 1.5); expected_output_len = len(signal) + expected_ir_len - 1
    assert abs(len(reverberated_signal) - expected_output_len) < sr * 0.1
    pulse_end_idx = sr//10 + 100; tail_energy = np.mean(reverberated_signal[pulse_end_idx + sr//10:]**2)
    original_tail_energy = np.mean(signal[pulse_end_idx + sr//10:]**2)
    assert tail_energy > original_tail_energy

def test_apply_delay(sample_pulse):
    signal, sr = sample_pulse; delay_time = 0.15; feedback = 0.5; wet_level = 0.6
    delayed_signal = apply_delay(signal, sr, delay_time=delay_time, feedback=feedback, wet_level=wet_level)
    assert delayed_signal.dtype == np.float64 and delayed_signal.shape == signal.shape
    pulse_peak_idx = sr//10 + 50; delay_samples = int(delay_time * sr); echo_peak_idx = pulse_peak_idx + delay_samples
    if echo_peak_idx < len(delayed_signal):
        search_win = 50; start = max(0, echo_peak_idx - search_win); end = min(len(delayed_signal), echo_peak_idx + search_win)
        assert np.max(np.abs(delayed_signal[start:end])) > 0.1

@pytest.mark.skip(reason="EQ implementation experimental")
def test_apply_graphic_eq(sample_pulse): pass
@pytest.mark.skip(reason="EQ implementation experimental")
def test_apply_parametric_eq(sample_pulse): pass

def test_apply_tremolo(sample_audio_short):
    signal, sr = sample_audio_short; rate = 4.0; depth = 0.7; shape = 'sine'
    processed_signal = apply_tremolo(signal, sr, rate=rate, depth=depth, shape=shape)
    assert processed_signal.shape == signal.shape and processed_signal.dtype == np.float64
    assert not np.allclose(signal, processed_signal, atol=1e-6)
    processed_signal_no_depth = apply_tremolo(signal, sr, rate=rate, depth=0.0, shape=shape)
    assert_allclose(signal, processed_signal_no_depth, atol=1e-9)
    with assert_raises(ValueError): apply_tremolo(signal, sr, rate=0)
    with assert_raises(ValueError): apply_tremolo(signal, sr, depth=1.1)
    with assert_raises(ValueError): apply_tremolo(signal, sr, shape='invalid') # type: ignore

def test_apply_chorus(sample_audio_short):
    signal, sr = sample_audio_short; rate = 1.0; depth = 0.003; delay = 0.030; feedback = 0.1; wet_level = 0.6
    processed_signal = apply_chorus(signal, sr, rate=rate, depth=depth, delay=delay, feedback=feedback, wet_level=wet_level)
    assert processed_signal.shape == signal.shape and processed_signal.dtype == np.float64
    assert not np.allclose(signal, processed_signal, atol=1e-6)
    processed_signal_dry = apply_chorus(signal, sr, rate=rate, depth=depth, delay=delay, feedback=feedback, wet_level=0.0, dry_level=1.0)
    assert_allclose(signal, processed_signal_dry, atol=1e-9)
    with assert_raises(ValueError): apply_chorus(signal, sr, depth=0.030, delay=0.030)
    with assert_raises(ValueError): apply_chorus(signal, sr, feedback=1.0)

def test_apply_flanger(sample_audio_short):
    signal, sr = sample_audio_short; rate = 0.3; depth = 0.002; delay = 0.003; feedback = 0.6; wet_level = 0.5; dry_level = 0.5
    processed_signal = apply_flanger(signal, sr, rate=rate, depth=depth, delay=delay, feedback=feedback, wet_level=wet_level, dry_level=dry_level)
    assert processed_signal.shape == signal.shape and processed_signal.dtype == np.float64
    assert not np.allclose(signal, processed_signal, atol=1e-6)
    processed_signal_dry = apply_flanger(signal, sr, rate=rate, depth=depth, delay=delay, feedback=feedback, wet_level=0.0, dry_level=0.5)
    assert_allclose(signal * 0.5, processed_signal_dry, atol=1e-9)
    with assert_raises(ValueError): apply_flanger(signal, sr, depth=0.003, delay=0.003)
    with assert_raises(ValueError): apply_flanger(signal, sr, feedback=1.1)

def test_adjust_gain(sample_audio_short):
    signal, sr = sample_audio_short; original_rms = np.sqrt(np.mean(signal**2))
    gain_db_amp = 6.0; amplified_signal = adjust_gain(signal, gain_db=gain_db_amp)
    amplified_rms = np.sqrt(np.mean(amplified_signal**2)); expected_multiplier_amp = 10**(gain_db_amp / 20.0)
    assert amplified_signal.dtype == np.float64 and amplified_signal.shape == signal.shape
    assert np.isclose(amplified_rms, original_rms * expected_multiplier_amp, rtol=1e-3)
    gain_db_att = -6.0; attenuated_signal = adjust_gain(signal, gain_db=gain_db_att)
    attenuated_rms = np.sqrt(np.mean(attenuated_signal**2)); expected_multiplier_att = 10**(gain_db_att / 20.0)
    assert attenuated_signal.dtype == np.float64 and attenuated_signal.shape == signal.shape
    assert np.isclose(attenuated_rms, original_rms * expected_multiplier_att, rtol=1e-3)
    zero_gain_signal = adjust_gain(signal, gain_db=0.0)
    assert_allclose(signal, zero_gain_signal, atol=1e-9)

# --- Tests for New Utility Effects ---

def test_noise_reduction_spectral(sample_noisy_signal):
    """Test the basic spectral noise reduction."""
    signal, sr, noise_dur = sample_noisy_signal
    noise_samples = int(noise_dur * sr)
    signal_part = signal[noise_samples:]
    noise_part = signal[:noise_samples]

    # Calculate RMS before reduction
    rms_original_noise = np.sqrt(np.mean(noise_part**2))
    rms_original_signal = np.sqrt(np.mean(signal_part**2))
    rms_original_total = np.sqrt(np.mean(signal**2))

    # Apply reduction
    reduced_signal = noise_reduction_spectral(signal, sr, noise_profile_duration=noise_dur, reduction_amount=1.0)

    assert reduced_signal.shape == signal.shape
    assert reduced_signal.dtype == np.float64
    assert not np.allclose(signal, reduced_signal)

    # Check if RMS of the total signal decreased (noise should be reduced)
    rms_reduced_total = np.sqrt(np.mean(reduced_signal**2))
    assert rms_reduced_total < rms_original_total

    # Check RMS of the initial noise part (should be significantly lower)
    reduced_noise_part = reduced_signal[:noise_samples]
    rms_reduced_noise = np.sqrt(np.mean(reduced_noise_part**2))
    assert rms_reduced_noise < rms_original_noise * 0.5 # Expect significant reduction

    # Check RMS of the signal part (should be less affected, but might decrease slightly)
    reduced_signal_part = reduced_signal[noise_samples:]
    rms_reduced_signal = np.sqrt(np.mean(reduced_signal_part**2))
    assert rms_reduced_signal <= rms_original_signal # Allow slight reduction
    assert rms_reduced_signal > rms_original_signal * 0.7 # But not too much reduction

    # Test invalid parameters
    with assert_raises(ValueError): noise_reduction_spectral(signal, sr, noise_profile_duration=0)
    with assert_raises(ValueError): noise_reduction_spectral(signal, sr, noise_profile_duration=10.0) # Longer than signal
    with assert_raises(ValueError): noise_reduction_spectral(signal, sr, reduction_amount=-0.1)
    with assert_raises(ValueError): noise_reduction_spectral(np.stack([signal,signal]), sr) # Stereo input


def test_transient_shaping_hpss(sample_transient_signal):
    """Test transient shaping using HPSS."""
    signal, sr = sample_transient_signal

    # Enhance transients
    scale_enhance = 2.0
    enhanced_signal = transient_shaping_hpss(signal, sr, percussive_scale=scale_enhance)
    assert enhanced_signal.shape == signal.shape
    assert enhanced_signal.dtype == np.float64
    assert not np.allclose(signal, enhanced_signal)
    # Check if peak amplitude increased (likely due to enhanced transients)
    assert np.max(np.abs(enhanced_signal)) > np.max(np.abs(signal))

    # Reduce transients
    scale_reduce = 0.2
    reduced_signal = transient_shaping_hpss(signal, sr, percussive_scale=scale_reduce)
    assert reduced_signal.shape == signal.shape
    assert reduced_signal.dtype == np.float64
    assert not np.allclose(signal, reduced_signal)
    # Check if peak amplitude decreased (likely due to reduced transients)
    assert np.max(np.abs(reduced_signal)) < np.max(np.abs(signal))

    # No change
    scale_one = 1.0
    no_change_signal = transient_shaping_hpss(signal, sr, percussive_scale=scale_one)
    # Allow small tolerance for HPSS reconstruction
    assert_allclose(signal, no_change_signal, atol=1e-5)

    # Test invalid input
    with assert_raises(ValueError): transient_shaping_hpss(np.stack([signal,signal]), sr) # Stereo


def test_stereo_widening_midside(sample_audio_stereo):
    """Test stereo widening using Mid/Side processing."""
    signal_stereo, sr = sample_audio_stereo
    left_orig = signal_stereo[0, :]
    right_orig = signal_stereo[1, :]

    # Increase width
    width_increase = 2.0
    widened_signal = stereo_widening_midside(signal_stereo, width_factor=width_increase)
    assert widened_signal.shape == signal_stereo.shape
    assert widened_signal.dtype == np.float64
    assert not np.allclose(signal_stereo, widened_signal)
    # Check if difference between channels increased
    diff_orig = np.mean((left_orig - right_orig)**2)
    diff_widened = np.mean((widened_signal[0, :] - widened_signal[1, :])**2)
    assert diff_widened > diff_orig * 1.1 # Expect difference to increase significantly

    # Decrease width (approach mono)
    width_decrease = 0.2
    narrowed_signal = stereo_widening_midside(signal_stereo, width_factor=width_decrease)
    assert narrowed_signal.shape == signal_stereo.shape
    diff_narrowed = np.mean((narrowed_signal[0, :] - narrowed_signal[1, :])**2)
    assert diff_narrowed < diff_orig * 0.9 # Expect difference to decrease significantly

    # Make mono
    width_mono = 0.0
    mono_signal = stereo_widening_midside(signal_stereo, width_factor=width_mono)
    assert mono_signal.shape == signal_stereo.shape
    # Left and Right channels should be identical (Mid signal)
    assert_allclose(mono_signal[0, :], mono_signal[1, :], atol=1e-7)
    # Verify it's the Mid signal
    mid = (left_orig + right_orig) / 2.0
    assert_allclose(mono_signal[0, :], mid, atol=1e-7)

    # No change
    width_one = 1.0
    no_change_signal = stereo_widening_midside(signal_stereo, width_factor=width_one)
    assert_allclose(signal_stereo, no_change_signal, atol=1e-7)

    # Test invalid input
    with assert_raises(ValueError, match="Input audio 'y' must be a 2-channel NumPy array"):
        stereo_widening_midside(signal_stereo[0,:]) # Mono input
    with assert_raises(ValueError, match="width_factor must be non-negative"):
        stereo_widening_midside(signal_stereo, width_factor=-0.5)
