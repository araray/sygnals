# tests/test_batch_processor.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path # Use pathlib
from typing import Tuple, Optional, Union, Literal, Dict

# Import the function to test
from sygnals.core.batch_processor import process_batch
# Import read_data to verify output (optional, could just check file existence/size)
from sygnals.core.data_handler import read_data

# --- Test Fixture ---

@pytest.fixture
def setup_batch_dirs(tmp_path: Path) -> Tuple[Path, Path]:
    """Creates temporary input and output directories for batch processing tests."""
    input_dir = tmp_path / "batch_input"
    output_dir = tmp_path / "batch_output"
    input_dir.mkdir()
    # output_dir is created by process_batch, no need to mkdir here

    # Create some dummy CSV data files in the input directory
    sr = 100 # Sample rate for dummy data
    t = np.linspace(0, 1, sr, endpoint=False)
    # File 1: Sine wave
    df1 = pd.DataFrame({"time": t, "value": np.sin(2 * np.pi * 5 * t)}) # Added time column
    df1.to_csv(input_dir / "signal_sine.csv", index=False)
    # File 2: Constant value
    df2 = pd.DataFrame({"time": t, "value": np.ones(sr) * 0.5}) # Added time column
    df2.to_csv(input_dir / "signal_const.csv", index=False)
    # File 3: Empty value column (to test potential errors)
    # Ensure 'value' column exists but is empty, also add 'time'
    df3 = pd.DataFrame({"time": pd.Series(dtype='float64'), "value": pd.Series(dtype='float64')})
    df3.to_csv(input_dir / "signal_empty.csv", index=False)
    # File 4: Non-csv file (should be ignored by current simple implementation)
    (input_dir / "other_file.txt").touch()
    # File 5: Add a dummy WAV file
    import soundfile as sf
    dummy_audio = np.random.randn(sr * 2).astype(np.float64) # 2 seconds audio
    sf.write(input_dir / "audio_noise.wav", dummy_audio, sr)

    return input_dir, output_dir

# --- Test Cases ---

def test_process_batch_fft(setup_batch_dirs):
    """Test batch processing with the 'fft' transform."""
    input_dir, output_dir = setup_batch_dirs

    # Run the batch processor
    process_batch(str(input_dir), str(output_dir), "fft")

    # Check that output directory was created
    assert output_dir.exists()
    assert output_dir.is_dir()

    # Check that output files were created for the valid inputs (CSV and WAV)
    out_file_csv1 = output_dir / "signal_sine_processed.csv"
    out_file_csv2 = output_dir / "signal_const_processed.csv"
    out_file_wav = output_dir / "audio_noise_processed.csv" # FFT output is CSV
    out_file_empty = output_dir / "signal_empty_processed.csv"
    ignored_file_txt = output_dir / "other_file_processed.csv"

    assert out_file_csv1.exists(), "FFT output for signal_sine.csv missing"
    assert out_file_csv1.stat().st_size > 0

    assert out_file_csv2.exists(), "FFT output for signal_const.csv missing"
    assert out_file_csv2.stat().st_size > 0

    assert out_file_wav.exists(), "FFT output for audio_noise.wav missing"
    assert out_file_wav.stat().st_size > 0

    # Check how empty file was handled (should be skipped, no output file)
    assert not out_file_empty.exists(), f"Output file for empty CSV {out_file_empty} should not exist."

    # Check ignored file
    assert not ignored_file_txt.exists(), "Output file for ignored .txt file should not exist."

    # Optional: Verify content of one output file
    fft_output = read_data(out_file_csv1)
    assert isinstance(fft_output, pd.DataFrame)
    assert "Frequency (Hz)" in fft_output.columns
    assert "Magnitude" in fft_output.columns
    assert len(fft_output) > 0 # Should have frequency bins


def test_process_batch_wavelet(setup_batch_dirs):
    """Test batch processing with the 'wavelet' transform."""
    input_dir, output_dir = setup_batch_dirs

    # Run the batch processor
    process_batch(str(input_dir), str(output_dir), "wavelet")

    # Check that output directory was created
    assert output_dir.exists()

    # Check that output files were created (expecting .npz for wavelet)
    out_file_csv1 = output_dir / "signal_sine_processed.npz" # Expect NPZ
    out_file_csv2 = output_dir / "signal_const_processed.npz" # Expect NPZ
    out_file_wav = output_dir / "audio_noise_processed.npz"   # Expect NPZ
    out_file_empty = output_dir / "signal_empty_processed.npz"
    ignored_file_txt = output_dir / "other_file_processed.npz"

    assert out_file_csv1.exists(), "Wavelet output for signal_sine.csv missing"
    assert out_file_csv1.stat().st_size > 0

    assert out_file_csv2.exists(), "Wavelet output for signal_const.csv missing"
    assert out_file_csv2.stat().st_size > 0

    assert out_file_wav.exists(), "Wavelet output for audio_noise.wav missing"
    assert out_file_wav.stat().st_size > 0

    # Check empty/ignored files
    assert not out_file_empty.exists(), f"Output file for empty CSV {out_file_empty} should not exist."
    assert not ignored_file_txt.exists(), "Output file for ignored .txt file should not exist."

    # Optional: Verify content of one output file
    wavelet_output = read_data(out_file_csv1) # read_data handles NPZ
    assert isinstance(wavelet_output, dict)
    # Check if keys look like wavelet levels (e.g., "Level_0", "Level_1", ...)
    assert any(key.startswith("Level_") for key in wavelet_output.keys())
    assert len(wavelet_output) > 1 # Should have multiple levels


def test_process_batch_invalid_transform(setup_batch_dirs):
    """Test batch processing with an invalid transform name."""
    input_dir, output_dir = setup_batch_dirs

    # Run with an unsupported transform - expect ValueError
    with pytest.raises(ValueError, match="Invalid transform specified: 'invalid_transform'"):
        process_batch(str(input_dir), str(output_dir), "invalid_transform")

    # Check that output directory might be created, but no processed files exist
    # (as the function should exit early)
    processed_files = list(output_dir.glob("*_processed.*")) # Check any processed extension
    assert len(processed_files) == 0 # No files should be processed
