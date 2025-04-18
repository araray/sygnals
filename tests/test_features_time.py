# tests/test_features_time.py

"""
Tests for time-domain feature extraction functions in
sygnals.core.features.time_domain. These operate on single frames (windows)
of a signal.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats import skew, kurtosis # For reference calculations

# Import feature functions to test
from sygnals.core.features.time_domain import (
    mean_amplitude,
    std_dev_amplitude,
    skewness as skewness_feature, # Rename to avoid clash with scipy.stats.skew
    kurtosis_val,
    peak_amplitude,
    crest_factor,
    signal_entropy,
)

# --- Test Fixtures ---

@pytest.fixture
def sample_frame_zeros():
    """Generate a frame of all zeros."""
    return np.zeros(512, dtype=np.float64)

@pytest.fixture
def sample_frame_constant():
    """Generate a frame with a constant non-zero value."""
    return np.full(512, 0.5, dtype=np.float64)

@pytest.fixture
def sample_frame_sine():
    """Generate a frame containing part of a sine wave."""
    sr = 1000.0
    frame_len = 512
    freq = 50.0
    t = np.arange(frame_len) / sr
    return (0.8 * np.sin(2 * np.pi * freq * t)).astype(np.float64)

@pytest.fixture
def sample_frame_gaussian():
    """Generate a frame with Gaussian noise."""
    return np.random.normal(loc=0.0, scale=0.2, size=512).astype(np.float64)

# --- Test Cases ---

def test_mean_amplitude(sample_frame_zeros, sample_frame_constant, sample_frame_sine):
    """Test the mean_amplitude feature."""
    assert mean_amplitude(sample_frame_zeros) == 0.0
    assert mean_amplitude(sample_frame_constant) == 0.5
    # Mean absolute value of sine A*sin(x) over full cycle is 2A/pi
    # This frame isn't a full cycle, but should be > 0 and < peak (0.8)
    mean_amp_sine = mean_amplitude(sample_frame_sine)
    assert 0 < mean_amp_sine < 0.8

def test_std_dev_amplitude(sample_frame_zeros, sample_frame_constant, sample_frame_sine):
    """Test the std_dev_amplitude feature."""
    assert std_dev_amplitude(sample_frame_zeros) == 0.0
    assert std_dev_amplitude(sample_frame_constant) == 0.0
    # Std dev of sine A*sin(x) is A/sqrt(2)
    std_dev_sine = std_dev_amplitude(sample_frame_sine)
    assert pytest.approx(std_dev_sine, rel=0.1) == 0.8 / np.sqrt(2)

def test_skewness_feature(sample_frame_zeros, sample_frame_constant, sample_frame_gaussian):
    """Test the skewness feature."""
    # Skewness is undefined (or 0) for constant data
    assert skewness_feature(sample_frame_zeros) == 0.0
    assert skewness_feature(sample_frame_constant) == 0.0
    # Skewness for Gaussian noise should be close to 0
    skew_gauss = skewness_feature(sample_frame_gaussian)
    assert pytest.approx(skew_gauss, abs=0.3) == 0.0 # Allow some deviation for random sample

def test_kurtosis_val(sample_frame_zeros, sample_frame_constant, sample_frame_gaussian):
    """Test the kurtosis_val feature (Fisher's definition)."""
    # Kurtosis is undefined (or -3 / 0 depending on definition) for constant data
    assert kurtosis_val(sample_frame_zeros) == 0.0 # Function handles zero variance
    assert kurtosis_val(sample_frame_constant) == 0.0 # Function handles zero variance
    # Kurtosis for Gaussian noise should be close to 0 (Fisher's definition)
    kurt_gauss = kurtosis_val(sample_frame_gaussian)
    assert pytest.approx(kurt_gauss, abs=0.5) == 0.0 # Allow larger deviation

def test_peak_amplitude(sample_frame_zeros, sample_frame_constant, sample_frame_sine):
    """Test the peak_amplitude feature."""
    assert peak_amplitude(sample_frame_zeros) == 0.0
    assert peak_amplitude(sample_frame_constant) == 0.5
    # Peak for sine A*sin(x) should be close to A
    assert pytest.approx(peak_amplitude(sample_frame_sine), abs=1e-3) == 0.8

def test_crest_factor(sample_frame_zeros, sample_frame_constant, sample_frame_sine):
    """Test the crest_factor feature."""
    # Crest factor for zeros or constant should be handled (e.g., return 0 or 1)
    assert crest_factor(sample_frame_zeros) == 0.0 # Peak=0, RMS=0 -> 0
    assert crest_factor(sample_frame_constant) == 1.0 # Peak=0.5, RMS=0.5 -> 1
    # Crest factor for sine wave is Peak / RMS = A / (A/sqrt(2)) = sqrt(2)
    cf_sine = crest_factor(sample_frame_sine)
    assert pytest.approx(cf_sine) == np.sqrt(2)

def test_signal_entropy(sample_frame_zeros, sample_frame_constant, sample_frame_gaussian):
    """Test the signal_entropy feature."""
    # Entropy of constant signal should be low (ideally 0, but depends on binning)
    entropy_zeros = signal_entropy(sample_frame_zeros)
    entropy_const = signal_entropy(sample_frame_constant)
    assert entropy_zeros == 0.0 # All values in one bin
    assert entropy_const == 0.0 # All values in one bin

    # Entropy of Gaussian noise should be relatively high
    entropy_gauss = signal_entropy(sample_frame_gaussian)
    assert entropy_gauss > 1.0 # Expect higher entropy for more random distribution


# TODO: Add tests for edge cases (e.g., very short frames, frames with NaNs/Infs if applicable)
#       and validate against reference calculations where possible.
