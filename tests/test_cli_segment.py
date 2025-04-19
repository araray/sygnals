# tests/test_cli_segment.py

"""
Tests for the 'sygnals segment' CLI command group.
"""

import pytest
import numpy as np
import pandas as pd # FIX: Import pandas
from numpy.testing import assert_equal # Import assert_equal for numpy array comparison
from pathlib import Path
from click.testing import CliRunner
import click # Import click to check for exceptions

# Import the main CLI entry point and the command group/functions to test/mock
from sygnals.cli.main import cli
# Import module to patch, but patch where used
# from sygnals.core import segmentation
# from sygnals.cli import segment_cmd # Import to patch helper

# --- Test Fixtures ---

@pytest.fixture
def runner() -> CliRunner:
    """Provides a Click CliRunner instance."""
    # Capture stderr for checking error messages
    return CliRunner(mix_stderr=False)

@pytest.fixture
def mock_audio_read(mocker):
    """Mocks read_data to return dummy audio data."""
    sr = 16000
    dummy_audio = np.random.randn(sr * 5).astype(np.float64) # 5 seconds
    # Mock read_data within the CLI module where it's called
    mock = mocker.patch("sygnals.cli.segment_cmd.read_data", return_value=(dummy_audio, sr))
    return mock, dummy_audio, sr

@pytest.fixture
def mock_segment_save(mocker):
    """Mocks the _save_segments helper function."""
    # Mock the helper function within the segment_cmd module where it's defined/used
    mock = mocker.patch("sygnals.cli.segment_cmd._save_segments")
    return mock

# --- Test Cases for 'segment fixed-length' ---

# FIX: Add mocker fixture to signature
def test_segment_fixed_length_cmd_success(runner: CliRunner, mock_audio_read, mock_segment_save, tmp_path: Path, mocker):
    """Test successful execution of 'segment fixed-length'."""
    mock_read, dummy_audio, sr = mock_audio_read
    mock_save = mock_segment_save
    input_file = tmp_path / "input.wav"
    input_file.touch() # Needs to exist for click.Path validation
    output_dir = tmp_path / "output_segments"

    seg_len = 1.5
    overlap = 0.5
    min_len = 0.1

    # Mock the core segmentation function where it's called
    # FIX: Correct patch target
    mock_core_segment = mocker.patch("sygnals.cli.segment_cmd.segment_fixed_length", return_value=[np.zeros(10), np.zeros(10)]) # Return dummy segments

    args = [
        "segment", "fixed-length", str(input_file),
        "--output", str(output_dir),
        "--length", str(seg_len),
        "--overlap", str(overlap),
        "--min-length", str(min_len),
        "--pad" # Explicitly enable padding (default)
    ]

    result = runner.invoke(cli, args)

    print("CLI Output:\n", result.output) # For debugging stdout
    print("CLI Stderr:\n", result.stderr) # For debugging stderr
    if result.exception: print("Exception:\n", result.exception)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}"
    assert "Successfully segmented" in result.output

    # Verify read_data was called
    mock_read.assert_called_once_with(input_file, sr=None)

    # Verify the core segmentation function was called with correct args
    mock_core_segment.assert_called_once()
    call_args, call_kwargs = mock_core_segment.call_args
    # Use assert_equal for numpy arrays
    assert_equal(call_kwargs.get('y'), dummy_audio) # Check signal data
    assert call_kwargs.get('sr') == sr
    assert call_kwargs.get('segment_length_sec') == seg_len
    assert call_kwargs.get('overlap_ratio') == overlap
    assert call_kwargs.get('pad') is True
    assert call_kwargs.get('min_segment_length_sec') == min_len

    # Verify the save helper was called
    mock_save.assert_called_once()
    save_call_args, save_call_kwargs = mock_save.call_args
    assert len(save_call_args[0]) == 2 # Check number of dummy segments passed
    assert save_call_args[1] == sr # Check sample rate
    assert save_call_args[2] == output_dir # Check output path
    assert save_call_args[3] == input_file.stem # Check base filename


def test_segment_fixed_length_cmd_no_pad(runner: CliRunner, mock_audio_read, mock_segment_save, tmp_path: Path, mocker):
    """Test 'segment fixed-length' with --no-pad."""
    mock_read, _, sr = mock_audio_read
    mock_save = mock_segment_save
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_dir = tmp_path / "output_segments"
    # FIX: Correct patch target
    mock_core_segment = mocker.patch("sygnals.cli.segment_cmd.segment_fixed_length", return_value=[np.zeros(10)])

    args = [
        "segment", "fixed-length", str(input_file),
        "--output", str(output_dir),
        "--length", "1.0",
        "--no-pad" # Disable padding
    ]

    result = runner.invoke(cli, args)

    print("CLI Output:\n", result.output) # For debugging stdout
    print("CLI Stderr:\n", result.stderr) # For debugging stderr
    if result.exception: print("Exception:\n", result.exception)

    assert result.exit_code == 0
    # Verify pad=False was passed to core function
    mock_core_segment.assert_called_once()
    call_args, call_kwargs = mock_core_segment.call_args
    assert call_kwargs.get('pad') is False


def test_segment_fixed_length_cmd_invalid_input(runner: CliRunner, tmp_path: Path, mocker):
    """Test 'segment fixed-length' with non-audio input."""
    # Mock read_data to return a DataFrame (non-audio)
    # FIX: Correct patch target
    mock_read = mocker.patch("sygnals.cli.segment_cmd.read_data", return_value=pd.DataFrame({'a': [1]}))
    input_file = tmp_path / "input.csv" # Use CSV extension
    input_file.touch()
    output_dir = tmp_path / "output_segments"

    args = [
        "segment", "fixed-length", str(input_file),
        "--output", str(output_dir),
        "--length", "1.0",
    ]

    result = runner.invoke(cli, args)

    print("CLI Output:\n", result.output) # For debugging stdout
    print("CLI Stderr:\n", result.stderr) # For debugging stderr
    if result.exception: print("Exception:\n", result.exception)

    assert result.exit_code != 0
    # Check stderr for the error message
    assert "Input file" in result.stderr and "not recognized as audio" in result.stderr
    mock_read.assert_called_once()


def test_segment_fixed_length_cmd_missing_output(runner: CliRunner, tmp_path: Path):
    """Test 'segment fixed-length' without required --output."""
    input_file = tmp_path / "input.wav"
    input_file.touch()
    args = [
        "segment", "fixed-length", str(input_file),
        "--length", "1.0",
    ]
    result = runner.invoke(cli, args)
    assert result.exit_code != 0
    # FIX: Check stderr for Click's missing parameter message
    assert "Missing parameter: output" in result.stderr


def test_segment_fixed_length_cmd_zero_segments(runner: CliRunner, mock_audio_read, tmp_path: Path, mocker):
    """Test 'segment fixed-length' when core function returns zero segments."""
    mock_read, _, sr = mock_audio_read
    # Mock save helper (should not be called)
    # FIX: Correct patch target
    mock_save = mocker.patch("sygnals.cli.segment_cmd._save_segments")
    # Mock core function to return empty list
    # FIX: Correct patch target
    mock_core_segment = mocker.patch("sygnals.cli.segment_cmd.segment_fixed_length", return_value=[])
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_dir = tmp_path / "output_segments"

    args = [
        "segment", "fixed-length", str(input_file),
        "--output", str(output_dir),
        "--length", "10.0", # Use long length to ensure it might return zero
    ]

    result = runner.invoke(cli, args)

    print("CLI Output:\n", result.output) # For debugging stdout
    print("CLI Stderr:\n", result.stderr) # For debugging stderr
    if result.exception: print("Exception:\n", result.exception)

    assert result.exit_code == 0
    # FIX: Check stderr for the warning message (click.echo goes to stdout by default)
    assert "Warning: No segments generated" in result.output
    mock_core_segment.assert_called_once()
    mock_save.assert_not_called() # Save should not be called

# TODO: Add tests for other segmentation methods (silence, event) when implemented.
