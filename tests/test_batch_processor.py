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
    df1 = pd.DataFrame({"value": np.sin(2 * np.pi * 5 * t)})
    df1.to_csv(input_dir / "signal_sine.csv", index=False)
    # File 2: Constant value
    df2 = pd.DataFrame({"value": np.ones(sr) * 0.5})
    df2.to_csv(input_dir / "signal_const.csv", index=False)
    # File 3: Empty value column (to test potential errors)
    df3 = pd.DataFrame({"value": []})
    df3.to_csv(input_dir / "signal_empty.csv", index=False)
    # File 4: Non-csv file (should be ignored by current simple implementation)
    (input_dir / "other_file.txt").touch()

    return input_dir, output_dir

# --- Test Cases ---

def test_process_batch_fft(setup_batch_dirs):
    """Test batch processing with the 'fft' transform."""
    input_dir, output_dir = setup_batch_dirs

    # Run the batch processor
    # Note: Current implementation reads only 'value' column and flattens.
    process_batch(str(input_dir), str(output_dir), "fft")

    # Check that output directory was created
    assert output_dir.exists()
    assert output_dir.is_dir()

    # Check that output files were created for the CSV inputs
    out_file1 = output_dir / "signal_sine.csv_processed.csv"
    out_file2 = output_dir / "signal_const.csv_processed.csv"
    out_file_empty = output_dir / "signal_empty.csv_processed.csv" # Check if empty file processed
    ignored_file = output_dir / "other_file.txt_processed.csv"

    assert out_file1.exists()
    assert out_file1.stat().st_size > 0 # FFT output should not be empty

    assert out_file2.exists()
    assert out_file2.stat().st_size > 0 # FFT of constant should exist

    # Check how empty file was handled (depends on implementation details in batch_processor)
    # Current batch_processor might raise error or produce empty/minimal output for empty input
    # For now, just check existence
    assert out_file_empty.exists()

    assert not ignored_file.exists() # Should ignore non-csv

    # Optional: Verify content of one output file
    fft_output = read_data(out_file1)
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

    # Check that output files were created
    out_file1 = output_dir / "signal_sine.csv_processed.csv"
    out_file2 = output_dir / "signal_const.csv_processed.csv"
    out_file_empty = output_dir / "signal_empty.csv_processed.csv"

    assert out_file1.exists()
    assert out_file1.stat().st_size > 0

    assert out_file2.exists()
    assert out_file2.stat().st_size > 0

    assert out_file_empty.exists() # Check existence for empty input case

    # Optional: Verify content of one output file
    wavelet_output = read_data(out_file1)
    assert isinstance(wavelet_output, pd.DataFrame)
    # Check if columns look like wavelet levels (e.g., "Level 1", "Level 2", ...)
    assert any(col.startswith("Level ") for col in wavelet_output.columns)
    assert len(wavelet_output) > 0 # Wavelet coeffs should exist


def test_process_batch_invalid_transform(setup_batch_dirs):
    """Test batch processing with an invalid transform name."""
    input_dir, output_dir = setup_batch_dirs

    # Run with an unsupported transform
    # The current implementation doesn't explicitly handle errors for invalid transforms,
    # it might just not produce output or raise an error depending on internal logic.
    # We'll assume it should run without error but produce no output files for this case.
    try:
        process_batch(str(input_dir), str(output_dir), "invalid_transform")
    except Exception as e:
        pytest.fail(f"process_batch raised unexpected exception for invalid transform: {e}")

    # Check that output directory might be created, but no processed files exist
    assert output_dir.exists()
    processed_files = list(output_dir.glob("*_processed.csv"))
    assert len(processed_files) == 0 # No files should be processed
