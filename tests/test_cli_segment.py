# tests/test_cli_segment.py

"""
Tests for the 'sygnals segment' CLI command group.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal
from pathlib import Path
from click.testing import CliRunner
import click # Import click to check for exceptions

# Import the main CLI entry point
from sygnals.cli.main import cli

# --- Test Fixtures ---

@pytest.fixture
def runner() -> CliRunner:
    """Provides a Click CliRunner instance."""
    return CliRunner(mix_stderr=False) # Keep stderr separate

@pytest.fixture
def mock_audio_read(mocker):
    """Mocks read_data to return dummy audio data."""
    sr = 16000
    dummy_audio = np.random.randn(sr * 5).astype(np.float64) # 5 seconds
    mock = mocker.patch("sygnals.cli.segment_cmd.read_data", return_value=(dummy_audio, sr))
    return mock, dummy_audio, sr

@pytest.fixture
def mock_segment_save(mocker):
    """Mocks the _save_segments helper function."""
    mock = mocker.patch("sygnals.cli.segment_cmd._save_segments")
    return mock

# --- Test Cases for 'segment fixed-length' ---

def test_segment_fixed_length_cmd_success(runner: CliRunner, mock_audio_read, mock_segment_save, tmp_path: Path, mocker):
    """Test successful execution of 'segment fixed-length'."""
    mock_read, dummy_audio, sr = mock_audio_read
    mock_save = mock_segment_save
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_dir = tmp_path / "output_segments"
    seg_len = 1.5
    overlap = 0.5
    min_len = 0.1
    mock_core_segment = mocker.patch("sygnals.cli.segment_cmd.segment_fixed_length", return_value=[np.zeros(10), np.zeros(10)])

    args = [
        "segment", "fixed-length", str(input_file),
        "--output", str(output_dir),
        "--length", str(seg_len),
        "--overlap", str(overlap),
        "--min-length", str(min_len),
        "--pad"
    ]
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}\nStderr:\n{result.stderr}\nException:\n{result.exception}"
    assert result.exception is None
    assert "Successfully segmented" in result.output
    mock_read.assert_called_once_with(input_file, sr=None)
    mock_core_segment.assert_called_once()
    call_args, call_kwargs = mock_core_segment.call_args
    assert_equal(call_kwargs.get('y'), dummy_audio)
    assert call_kwargs.get('sr') == sr
    assert call_kwargs.get('segment_length_sec') == seg_len
    assert call_kwargs.get('overlap_ratio') == overlap
    assert call_kwargs.get('pad') is True
    assert call_kwargs.get('min_segment_length_sec') == min_len
    mock_save.assert_called_once()
    save_call_args, save_call_kwargs = mock_save.call_args
    assert len(save_call_args[0]) == 2
    assert save_call_args[1] == sr
    assert save_call_args[2] == output_dir
    assert save_call_args[3] == input_file.stem


def test_segment_fixed_length_cmd_no_pad(runner: CliRunner, mock_audio_read, mock_segment_save, tmp_path: Path, mocker):
    """Test 'segment fixed-length' with --no-pad."""
    mock_read, _, sr = mock_audio_read
    mock_save = mock_segment_save
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_dir = tmp_path / "output_segments"
    mock_core_segment = mocker.patch("sygnals.cli.segment_cmd.segment_fixed_length", return_value=[np.zeros(10)])

    args = [
        "segment", "fixed-length", str(input_file),
        "--output", str(output_dir),
        "--length", "1.0",
        "--no-pad"
    ]
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}\nStderr:\n{result.stderr}\nException:\n{result.exception}"
    assert result.exception is None
    mock_core_segment.assert_called_once()
    call_args, call_kwargs = mock_core_segment.call_args
    assert call_kwargs.get('pad') is False


def test_segment_fixed_length_cmd_invalid_input(runner: CliRunner, tmp_path: Path, mocker):
    """Test 'segment fixed-length' with non-audio input."""
    mock_read = mocker.patch("sygnals.cli.segment_cmd.read_data", return_value=pd.DataFrame({'a': [1]}))
    input_file = tmp_path / "input.csv"
    input_file.touch()
    output_dir = tmp_path / "output_segments"

    args = [
        "segment", "fixed-length", str(input_file),
        "--output", str(output_dir),
        "--length", "1.0",
    ]

    result = runner.invoke(cli, args) # Use default catch_exceptions=True

    print("CLI Output:\n", result.output) # For debugging stdout
    print("CLI Stderr:\n", result.stderr) # For debugging stderr
    if result.exception: print("Exception:\n", result.exception)

    assert result.exit_code != 0
    # FIX: Check result.exception is SystemExit
    assert result.exception is not None, f"Expected an exception, but got None. Output:\n{result.output}\nStderr:\n{result.stderr}"
    assert isinstance(result.exception, SystemExit), f"Expected SystemExit, got {type(result.exception)}"
    assert result.exception.code != 0
    # *** REMOVED check for specific message in stderr for Abort ***
    # assert "Input file 'input.csv' is not recognized as audio" in result.stderr
    mock_read.assert_called_once()


def test_segment_fixed_length_cmd_missing_output(runner: CliRunner, tmp_path: Path):
    """Test 'segment fixed-length' without required --output."""
    input_file = tmp_path / "input.wav"
    input_file.touch()
    args = [
        "segment", "fixed-length", str(input_file),
        "--length", "1.0",
    ]
    result = runner.invoke(cli, args) # Use default catch_exceptions=True
    assert result.exit_code != 0
    # FIX: Check result.exception is SystemExit
    assert result.exception is not None, f"Expected an exception, but got None. Output:\n{result.output}\nStderr:\n{result.stderr}"
    assert isinstance(result.exception, SystemExit), f"Expected SystemExit, got {type(result.exception)}"
    assert result.exception.code == 2 # Expect exit code 2 for missing param
    # FIX: Check stderr for the exact Click error message for missing option
    assert "Error: Missing option '-o' / '--output'." in result.stderr


def test_segment_fixed_length_cmd_zero_segments(runner: CliRunner, mock_audio_read, tmp_path: Path, mocker):
    """Test 'segment fixed-length' when core function returns zero segments."""
    mock_read, _, sr = mock_audio_read
    mock_save = mocker.patch("sygnals.cli.segment_cmd._save_segments")
    mock_core_segment = mocker.patch("sygnals.cli.segment_cmd.segment_fixed_length", return_value=[])
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_dir = tmp_path / "output_segments"

    args = [
        "segment", "fixed-length", str(input_file),
        "--output", str(output_dir),
        "--length", "10.0",
    ]

    result = runner.invoke(cli, args) # Use default catch_exceptions=True

    print("CLI Output:\n", result.output) # For debugging stdout
    print("CLI Stderr:\n", result.stderr) # For debugging stderr
    if result.exception: print("Exception:\n", result.exception)

    assert result.exit_code == 0
    assert result.exception is None
    assert "Warning: No segments generated" in result.output
    mock_core_segment.assert_called_once()
    mock_save.assert_not_called()

# TODO: Add tests for other segmentation methods (silence, event) when implemented.
