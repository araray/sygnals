# tests/test_features_cepstral.py

"""
Tests for cepstral feature extraction functions in sygnals.core.features.cepstral.
Primarily focuses on MFCCs.
"""

import pytest
import numpy as np
import librosa # For generating test signals and reference calculations
from numpy.testing import assert_allclose, assert_equal

# Import feature functions to test
from sygnals.core.features.cepstral import mfcc

# --- Test Fixtures ---

@pytest.fixture
def sample_audio_for_mfcc():
    """Generate a sample audio signal suitable for MFCC testing."""
    sr = 22050
    duration = 1.5
    # Use a chirp signal as it has varying frequency content
    signal = librosa.chirp(fmin=100, fmax=sr/3, sr=sr, duration=duration).astype(np.float64) * 0.6
    return signal, sr

@pytest.fixture
def precomputed_melspec(sample_audio_for_mfcc):
    """Generate a pre-computed log-power Mel spectrogram."""
    signal, sr = sample_audio_for_mfcc
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    # Calculate power spectrogram
    S_power = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))**2
    # Calculate Mel spectrogram
    S_mel = librosa.feature.melspectrogram(S=S_power, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Convert to log scale (dB)
    S_mel_log = librosa.power_to_db(S_mel, ref=np.max)
    return S_mel_log, sr, hop_length

# --- Test Cases ---

def test_mfcc_from_audio(sample_audio_for_mfcc):
    """Test calculating MFCCs directly from an audio time series."""
    signal, sr = sample_audio_for_mfcc
    n_mfcc = 13
    n_fft = 2048 # Use standard parameters
    hop_length = 512

    mfccs = mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Check output shape: (n_mfcc, num_frames)
    expected_num_frames = int(np.floor(len(signal) / hop_length)) + 1
    assert mfccs.shape[0] == n_mfcc
    assert mfccs.shape[1] == expected_num_frames
    assert mfccs.dtype == np.float64

def test_mfcc_from_spectrogram(precomputed_melspec):
    """Test calculating MFCCs from a pre-computed log-power Mel spectrogram."""
    S_mel_log, sr, hop_length = precomputed_melspec
    n_mfcc = 20 # Use a different number of MFCCs

    # Note: When passing S, y is ignored. sr might still be needed if hop_length isn't implicitly known or passed.
    # Our mfcc function wrapper passes sr, so it should be fine.
    mfccs = mfcc(S=S_mel_log, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length) # Pass hop_length if needed by internal DCT

    # Check output shape
    expected_num_frames = S_mel_log.shape[1]
    assert mfccs.shape[0] == n_mfcc
    assert mfccs.shape[1] == expected_num_frames
    assert mfccs.dtype == np.float64

def test_mfcc_parameters(sample_audio_for_mfcc):
    """Test the effect of different MFCC parameters (n_mfcc, lifter)."""
    signal, sr = sample_audio_for_mfcc
    hop_length = 512

    # Calculate with default n_mfcc=13, no liftering
    mfccs_13 = mfcc(y=signal, sr=sr, n_mfcc=13, hop_length=hop_length, lifter=0)
    # Calculate with n_mfcc=20, no liftering
    mfccs_20 = mfcc(y=signal, sr=sr, n_mfcc=20, hop_length=hop_length, lifter=0)
    # Calculate with n_mfcc=13, with liftering
    mfccs_13_lifted = mfcc(y=signal, sr=sr, n_mfcc=13, hop_length=hop_length, lifter=22) # Common lifter value

    assert mfccs_13.shape[0] == 13
    assert mfccs_20.shape[0] == 20
    assert mfccs_13_lifted.shape[0] == 13
    assert mfccs_13.shape[1] == mfccs_20.shape[1] == mfccs_13_lifted.shape[1] # Number of frames should match

    # Check that liftering changes the values
    assert not np.allclose(mfccs_13, mfccs_13_lifted)

    # Check that the first 13 coeffs of mfccs_20 are similar (but not identical due to DCT normalization)
    # to mfccs_13. This is not strictly guaranteed but often holds approximately.
    # assert_allclose(mfccs_13, mfccs_20[:13, :], atol=1.0) # Allow some tolerance

def test_mfcc_missing_input():
    """Test calling mfcc without providing 'y' or 'S'."""
    with pytest.raises(ValueError, match="Either audio time series 'y' or Mel spectrogram 'S' must be provided."):
        mfcc(sr=22050, n_mfcc=13)

# --- Add tests for other cepstral features (LPC, etc.) as implemented ---
