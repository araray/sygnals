# tests/test_augment.py

"""
Tests for data augmentation functions in sygnals.core.augment.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_less
import warnings # To check for placeholder warnings

# Import augmentation functions to test
from sygnals.core.augment import (
    add_noise,
    pitch_shift,
    time_stretch,
)

# --- Test Fixtures ---

@pytest.fixture
def sample_audio_mono():
    """Generate a simple mono audio signal."""
    sr = 22050
    duration = 1.0
    freq = 440.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (0.7 * np.sin(2 * np.pi * freq * t)).astype(np.float64)
    return signal, sr

# --- Test Cases ---

def test_add_noise(sample_audio_mono):
    """Test the add_noise augmentation function."""
    signal, sr = sample_audio_mono
    signal_power = np.mean(signal**2)

    # Test adding noise with specific SNR
    snr_db = 15.0
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

    # Test on silent signal (should return original signal)
    silent_signal = np.zeros_like(signal)
    noisy_silent = add_noise(silent_signal, snr_db=snr_db)
    assert_allclose(noisy_silent, silent_signal, atol=1e-9)


def test_pitch_shift(sample_audio_mono):
    """Test the pitch_shift augmentation function."""
    signal, sr = sample_audio_mono

    # Test shifting up
    steps_up = 2.5
    shifted_up = pitch_shift(signal, sr, n_steps=steps_up)
    assert shifted_up.dtype == np.float64
    assert shifted_up.shape == signal.shape # Librosa pitch shift preserves length

    # Test shifting down
    steps_down = -3.0
    shifted_down = pitch_shift(signal, sr, n_steps=steps_down)
    assert shifted_down.dtype == np.float64
    assert shifted_down.shape == signal.shape

    # Test zero shift (should be very close to original)
    shifted_zero = pitch_shift(signal, sr, n_steps=0.0)
    assert_allclose(signal, shifted_zero, atol=1e-5) # Allow small tolerance for processing

    # Test invalid input dimension
    signal_2d = np.stack([signal, signal])
    with pytest.raises(ValueError, match="Input audio data must be a 1D array"):
        pitch_shift(signal_2d, sr, n_steps=1.0)


def test_time_stretch(sample_audio_mono):
    """Test the time_stretch augmentation function."""
    signal, sr = sample_audio_mono
    original_len = len(signal)

    # Test slowing down (rate < 1.0) -> longer signal
    rate_slow = 0.8
    stretched_slow = time_stretch(signal, rate=rate_slow)
    assert stretched_slow.dtype == np.float64
    expected_len_slow = original_len / rate_slow
    # Allow tolerance due to framing/algorithm details
    assert abs(len(stretched_slow) - expected_len_slow) < 0.1 * original_len

    # Test speeding up (rate > 1.0) -> shorter signal
    rate_fast = 1.25
    stretched_fast = time_stretch(signal, rate=rate_fast)
    assert stretched_fast.dtype == np.float64
    expected_len_fast = original_len / rate_fast
    assert abs(len(stretched_fast) - expected_len_fast) < 0.1 * original_len

    # Test rate = 1.0 (should be very close to original)
    stretched_one = time_stretch(signal, rate=1.0)
    # Length might still differ slightly due to processing
    assert abs(len(stretched_one) - original_len) < 10 # Allow small difference
    # Content should be very similar
    # assert_allclose(signal, stretched_one[:original_len], atol=1e-4) # Compare overlapping part

    # Test invalid rate
    with pytest.raises(ValueError, match="Time stretch rate must be positive"):
        time_stretch(signal, rate=0.0)
    with pytest.raises(ValueError, match="Time stretch rate must be positive"):
        time_stretch(signal, rate=-0.5)

    # Test invalid input dimension
    signal_2d = np.stack([signal, signal])
    with pytest.raises(ValueError, match="Input audio data must be a 1D array"):
        time_stretch(signal_2d, rate=1.1)
