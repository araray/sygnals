# tests/test_features_manager.py

"""
Tests for the feature extraction manager in sygnals.core.features.manager.
"""

import pytest
import numpy as np
import pandas as pd
import librosa # For generating test signals
from numpy.testing import assert_allclose, assert_equal

# Import the main function to test
from sygnals.core.features.manager import extract_features, FeatureExtractionError

# --- Test Fixtures ---

@pytest.fixture
def sample_audio_long():
    """Generate a slightly longer audio signal for robust framing."""
    sr = 22050
    duration = 2.5 # seconds
    # Combine sine and chirp for varied content
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal_sine = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    signal_chirp = 0.4 * librosa.chirp(fmin=400, fmax=1000, sr=sr, duration=duration)
    signal = (signal_sine + signal_chirp).astype(np.float64)
    return signal, sr

# --- Test Cases ---

def test_extract_single_time_feature(sample_audio_long):
    """Test extracting a single time-domain feature."""
    signal, sr = sample_audio_long
    features_to_extract = ["mean_amplitude"]
    frame_length=1024
    hop_length=512

    result_df = extract_features(signal, sr, features_to_extract,
                                 frame_length=frame_length, hop_length=hop_length,
                                 output_format='dataframe')

    assert isinstance(result_df, pd.DataFrame)
    assert "mean_amplitude" in result_df.columns
    assert result_df.index.name == 'time'
    assert result_df["mean_amplitude"].dtype == np.float64
    # Check number of frames based on signal length, frame, hop
    expected_num_frames = int(np.floor(len(signal) / hop_length)) + 1
    assert len(result_df) == expected_num_frames

def test_extract_single_freq_feature(sample_audio_long):
    """Test extracting a single frequency-domain feature."""
    signal, sr = sample_audio_long
    features_to_extract = ["spectral_centroid"]
    frame_length=2048
    hop_length=512

    result_dict = extract_features(signal, sr, features_to_extract,
                                   frame_length=frame_length, hop_length=hop_length,
                                   output_format='dict_of_arrays')

    assert isinstance(result_dict, dict)
    assert "spectral_centroid" in result_dict
    assert "time" in result_dict
    assert result_dict["spectral_centroid"].dtype == np.float64
    expected_num_frames = int(np.floor(len(signal) / hop_length)) + 1
    assert len(result_dict["spectral_centroid"]) == expected_num_frames
    assert len(result_dict["time"]) == expected_num_frames

def test_extract_mfcc(sample_audio_long):
    """Test extracting MFCC features."""
    signal, sr = sample_audio_long
    features_to_extract = ["mfcc"]
    n_mfcc = 20
    feature_params = {'mfcc': {'n_mfcc': n_mfcc}}
    frame_length=2048
    hop_length=512

    result_df = extract_features(signal, sr, features_to_extract,
                                 frame_length=frame_length, hop_length=hop_length,
                                 feature_params=feature_params, output_format='dataframe')

    assert isinstance(result_df, pd.DataFrame)
    # Check if all MFCC columns are present
    expected_mfcc_cols = [f"mfcc_{i}" for i in range(n_mfcc)]
    for col in expected_mfcc_cols:
        assert col in result_df.columns
        assert result_df[col].dtype == np.float64
    expected_num_frames = int(np.floor(len(signal) / hop_length)) + 1
    assert len(result_df) == expected_num_frames

def test_extract_spectral_contrast(sample_audio_long):
    """Test extracting spectral contrast features."""
    signal, sr = sample_audio_long
    features_to_extract = ["spectral_contrast"]
    n_bands = 6
    feature_params = {'spectral_contrast': {'n_bands': n_bands}}
    frame_length=2048
    hop_length=512

    result_dict = extract_features(signal, sr, features_to_extract,
                                   frame_length=frame_length, hop_length=hop_length,
                                   feature_params=feature_params, output_format='dict_of_arrays')

    assert isinstance(result_dict, dict)
    # Check if all contrast band columns and delta are present
    expected_contrast_cols = [f"contrast_band_{i}" for i in range(n_bands)] + ["contrast_delta"]
    for col in expected_contrast_cols:
        assert col in result_dict
        assert result_dict[col].dtype == np.float64
    assert "time" in result_dict
    expected_num_frames = int(np.floor(len(signal) / hop_length)) + 1
    assert len(result_dict["time"]) == expected_num_frames


def test_extract_multiple_features(sample_audio_long):
    """Test extracting a mix of feature types."""
    signal, sr = sample_audio_long
    features_to_extract = ["rms_energy", "spectral_centroid", "mfcc"]
    n_mfcc = 5 # Use fewer MFCCs for simplicity
    feature_params = {'mfcc': {'n_mfcc': n_mfcc}}
    frame_length=2048
    hop_length=1024 # Use different hop

    result_df = extract_features(signal, sr, features_to_extract,
                                 frame_length=frame_length, hop_length=hop_length,
                                 feature_params=feature_params, output_format='dataframe')

    assert isinstance(result_df, pd.DataFrame)
    assert "rms_energy" in result_df.columns
    assert "spectral_centroid" in result_df.columns
    expected_mfcc_cols = [f"mfcc_{i}" for i in range(n_mfcc)]
    for col in expected_mfcc_cols:
        assert col in result_df.columns

    expected_num_frames = int(np.floor(len(signal) / hop_length)) + 1
    assert len(result_df) == expected_num_frames


def test_extract_unknown_feature(sample_audio_long):
    """Test requesting an unknown feature."""
    signal, sr = sample_audio_long
    features_to_extract = ["rms_energy", "unknown_feature"]

    with pytest.raises(ValueError, match="Unknown feature\\(s\\) requested: \\['unknown_feature'\\]"):
        extract_features(signal, sr, features_to_extract)

def test_extract_feature_with_params(sample_audio_long):
    """Test passing specific parameters to a feature."""
    signal, sr = sample_audio_long
    features_to_extract = ["spectral_rolloff"]
    # Default roll_percent is 0.85
    result_85 = extract_features(signal, sr, features_to_extract, output_format='dict_of_arrays')
    # Change roll_percent
    feature_params = {"spectral_rolloff": {"roll_percent": 0.95}}
    result_95 = extract_features(signal, sr, features_to_extract,
                                 feature_params=feature_params, output_format='dict_of_arrays')

    assert "spectral_rolloff" in result_85
    assert "spectral_rolloff" in result_95
    # Expect rolloff frequency to be higher for 95%
    # Use mean rolloff across frames for comparison
    assert np.mean(result_95["spectral_rolloff"]) > np.mean(result_85["spectral_rolloff"])


def test_extract_short_signal(sample_audio_long):
    """Test extracting features from a signal shorter than frame length."""
    signal_short = sample_audio_long[0][:512] # Shorter than default frame_length
    sr = sample_audio_long[1]
    features_to_extract = ["rms_energy", "spectral_centroid"]
    frame_length=1024
    hop_length=512

    result_df = extract_features(signal_short, sr, features_to_extract,
                                 frame_length=frame_length, hop_length=hop_length)

    # Expect one frame of output due to padding (if center=True)
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 1
    assert "rms_energy" in result_df.columns
    assert "spectral_centroid" in result_df.columns


def test_extract_no_features(sample_audio_long):
    """Test calling extract_features with an empty feature list."""
    signal, sr = sample_audio_long
    features_to_extract = []
    result_df = extract_features(signal, sr, features_to_extract, output_format='dataframe')
    result_dict = extract_features(signal, sr, features_to_extract, output_format='dict_of_arrays')

    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty

    assert isinstance(result_dict, dict)
    assert 'time' in result_dict # Should still contain time
    assert len(result_dict) == 1


# TODO: Add more tests for edge cases, different parameter combinations,
#       validation of specific feature values against known signals/reference implementations.
