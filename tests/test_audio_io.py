# tests/test_audio_io.py

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path # Use pathlib
from numpy.testing import assert_allclose # Import assert_allclose

# Import functions from the new structure
from sygnals.core.audio.io import load_audio, save_audio
from sygnals.core.audio.effects import time_stretch, pitch_shift, simple_dynamic_range_compression
# Removed slice_audio, save_audio_as_csv, save_audio_as_json imports

# Ensure test data exists (or create it if needed)
# Assuming test_audio_001.wav exists in tests/data/
TEST_AUDIO_FILE = Path("tests/data/test_audio_001.wav")
if not TEST_AUDIO_FILE.exists():
     pytest.skip(f"Test audio file not found: {TEST_AUDIO_FILE}", allow_module_level=True)


@pytest.mark.parametrize("audio_file_path", [TEST_AUDIO_FILE])
def test_load_audio(audio_file_path: Path):
    """Test loading an audio file."""
    # Test loading with native SR
    data_native, sr_native = load_audio(audio_file_path, sr=None)
    assert isinstance(data_native, np.ndarray)
    assert data_native.dtype == np.float64 # Check for float64 conversion
    assert sr_native > 0

    # Test loading with resampling
    target_sr = sr_native // 2 # Example target SR
    if target_sr > 0: # Ensure target SR is valid
        data_resampled, sr_resampled = load_audio(audio_file_path, sr=target_sr)
        assert sr_resampled == target_sr
        # Check if length changed due to resampling
        assert abs(len(data_resampled) / target_sr - len(data_native) / sr_native) < 0.1 # Allow slight duration difference

    # Test loading mono (assuming test file might be stereo)
    data_mono, sr_mono = load_audio(audio_file_path, sr=None, mono=True)
    assert data_mono.ndim == 1

def test_save_audio(tmp_path: Path):
    """Test saving an audio file."""
    # Generate a simple sine wave
    sr = 22050
    duration = 1.0
    freq = 440.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Ensure data is float64
    data = np.sin(2 * np.pi * freq * t).astype(np.float64)
    out_file = tmp_path / "out_test.wav"

    # Save the audio
    save_audio(data, sr, out_file, subtype='PCM_16') # Use default subtype
    assert out_file.exists()

    # Load it back and verify
    loaded_data, loaded_sr = sf.read(str(out_file), dtype='float64') # Read as float64
    assert loaded_sr == sr
    # Allow for small differences due to PCM quantization if saved as PCM
    assert_allclose(data, loaded_data, atol=1e-4) # Increased tolerance for PCM_16

def test_save_audio_float(tmp_path: Path):
    """Test saving audio with FLOAT subtype."""
    sr = 16000
    data = (np.random.rand(sr) - 0.5).astype(np.float64) # Random float data
    out_file = tmp_path / "out_float.wav"
    save_audio(data, sr, out_file, subtype='FLOAT')
    assert out_file.exists()
    loaded_data, loaded_sr = sf.read(str(out_file), dtype='float64')
    assert loaded_sr == sr
    # Use the imported assert_allclose
    assert_allclose(data, loaded_data, atol=1e-7) # Should be very close for FLOAT

# Test slice_audio removed as function is not in io or effects

def test_time_stretch():
    """Test the time_stretch effect."""
    sr = 22050
    data = np.ones(sr, dtype=np.float64) # 1 second of ones
    stretch_rate = 2.0 # Speed up
    stretched = time_stretch(data, rate=stretch_rate)
    assert stretched.dtype == np.float64
    # time_stretch should produce ~ 0.5s of data if rate=2.0
    expected_len = sr / stretch_rate
    # Allow some tolerance for the stretching algorithm's framing
    assert abs(len(stretched) - expected_len) < 0.1 * sr # e.g., within 10% of expected length

def test_pitch_shift():
    """Test the pitch_shift effect."""
    sr = 22050
    data = np.ones(sr, dtype=np.float64)
    n_steps = 2.0 # Shift up by 2 semitones
    shifted = pitch_shift(data, sr, n_steps=n_steps)
    assert shifted.dtype == np.float64
    # pitch shift using librosa's default method shouldn't significantly change length
    assert abs(len(shifted) - len(data)) < 10 # Allow minor length difference

def test_dynamic_range_compression():
    """Test the simple_dynamic_range_compression effect."""
    # Test data with values above and below threshold
    data = np.array([0.2, 0.6, 0.9, -0.7, -1.0], dtype=np.float64)
    threshold = 0.5
    ratio = 4.0
    compressed = simple_dynamic_range_compression(data, threshold=threshold, ratio=ratio)
    assert compressed.dtype == np.float64
    assert len(compressed) == len(data)

    # Check that values below threshold are unchanged
    assert np.isclose(compressed[0], data[0])

    # Check that values above threshold are reduced
    # For 0.6: abs=0.6. over = 0.6 - 0.5 = 0.1. reduced_over = 0.1 / 4 = 0.025. new_amp = 0.5 + 0.025 = 0.525
    assert np.isclose(abs(compressed[1]), 0.525)
    assert abs(compressed[1]) < abs(data[1]) # Magnitude should decrease
    # Expected for 0.9: over = 0.9 - 0.5 = 0.4. reduced_over = 0.4 / 4 = 0.1. new_amp = 0.5 + 0.1 = 0.6
    assert np.isclose(abs(compressed[2]), 0.6)
    # Expected for -0.7: abs=0.7. over = 0.7 - 0.5 = 0.2. reduced_over = 0.2 / 4 = 0.05. new_amp = 0.5 + 0.05 = 0.55
    assert np.isclose(abs(compressed[3]), 0.55)
    # Expected for -1.0: abs=1.0. over = 1.0 - 0.5 = 0.5. reduced_over = 0.5 / 4 = 0.125. new_amp = 0.5 + 0.125 = 0.625
    assert np.isclose(abs(compressed[4]), 0.625)

# Tests for save_audio_as_csv and save_audio_as_json removed
