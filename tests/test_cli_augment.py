# tests/test_cli_augment.py

"""
Tests for the 'sygnals augment' CLI command group.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal
from pathlib import Path
from click.testing import CliRunner
import click # Import click to check for exceptions
import warnings # To check for placeholder warnings

# Import the main CLI entry point
from sygnals.cli.main import cli

# --- Test Fixtures ---

@pytest.fixture
def runner() -> CliRunner:
    """Provides a Click CliRunner instance."""
    return CliRunner(mix_stderr=False) # Keep stderr separate

@pytest.fixture
def mock_audio_read_save(mocker):
    """Mocks read_data and save_data."""
    sr = 16000
    dummy_audio = np.random.randn(sr * 3).astype(np.float64) # 3 seconds
    mock_read = mocker.patch("sygnals.cli.augment_cmd.read_data", return_value=(dummy_audio, sr))
    mock_save = mocker.patch("sygnals.cli.augment_cmd.save_data")
    return mock_read, mock_save, dummy_audio, sr

# --- Test Cases ---

def test_augment_add_noise_cmd(runner: CliRunner, mock_audio_read_save, tmp_path: Path, mocker):
    """Test successful execution of 'augment add-noise'."""
    mock_read, mock_save, dummy_audio, sr = mock_audio_read_save
    mock_core_noise = mocker.patch("sygnals.cli.augment_cmd.add_noise", return_value=dummy_audio * 0.5)
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_file = tmp_path / "output_noisy.wav"
    snr = 12.5
    noise_type = 'gaussian'
    seed = 42
    args = [
        "augment", "add-noise", str(input_file),
        "--output", str(output_file),
        "--snr", str(snr),
        "--noise-type", noise_type,
        "--seed", str(seed)
    ]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}\nStderr:\n{result.stderr}\nException:\n{result.exception}"
    assert result.exception is None
    assert "Successfully applied" in result.output and "noise" in result.output
    mock_read.assert_called_once_with(input_file)
    mock_core_noise.assert_called_once()
    call_args, call_kwargs = mock_core_noise.call_args
    assert_equal(call_kwargs.get('y'), dummy_audio)
    assert call_kwargs.get('snr_db') == snr
    assert call_kwargs.get('noise_type') == noise_type
    assert call_kwargs.get('seed') == seed
    mock_save.assert_called_once()
    save_call_args, save_call_kwargs = mock_save.call_args
    assert_equal(save_call_args[0][0], dummy_audio * 0.5)
    assert save_call_args[0][1] == sr
    assert save_call_args[1] == output_file

def test_augment_pitch_shift_cmd(runner: CliRunner, mock_audio_read_save, tmp_path: Path, mocker):
    """Test successful execution of 'augment pitch-shift'."""
    mock_read, mock_save, dummy_audio, sr = mock_audio_read_save
    mock_core_pitch = mocker.patch("sygnals.cli.augment_cmd.pitch_shift", return_value=dummy_audio * 0.8)
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_file = tmp_path / "output_shifted.wav"
    steps = -2.0
    bins = 24
    args = [
        "augment", "pitch-shift", str(input_file),
        "--output", str(output_file),
        "--steps", str(steps),
        "--bins-per-octave", str(bins)
    ]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}\nStderr:\n{result.stderr}\nException:\n{result.exception}"
    assert result.exception is None
    assert "Successfully applied pitch shift" in result.output
    mock_read.assert_called_once_with(input_file)
    mock_core_pitch.assert_called_once()
    call_args, call_kwargs = mock_core_pitch.call_args
    assert_equal(call_kwargs.get('y'), dummy_audio)
    assert call_kwargs.get('sr') == sr
    assert call_kwargs.get('n_steps') == steps
    assert call_kwargs.get('bins_per_octave') == bins
    mock_save.assert_called_once()
    save_call_args, save_call_kwargs = mock_save.call_args
    assert_equal(save_call_args[0][0], dummy_audio * 0.8)
    assert save_call_args[0][1] == sr
    assert save_call_args[1] == output_file

def test_augment_time_stretch_cmd(runner: CliRunner, mock_audio_read_save, tmp_path: Path, mocker):
    """Test successful execution of 'augment time-stretch'."""
    mock_read, mock_save, dummy_audio, sr = mock_audio_read_save
    stretched_audio = np.random.randn(len(dummy_audio) // 2).astype(np.float64)
    mock_core_stretch = mocker.patch("sygnals.cli.augment_cmd.time_stretch", return_value=stretched_audio)
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_file = tmp_path / "output_stretched.wav"
    rate = 1.5
    args = [
        "augment", "time-stretch", str(input_file),
        "--output", str(output_file),
        "--rate", str(rate)
    ]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}\nStderr:\n{result.stderr}\nException:\n{result.exception}"
    assert result.exception is None
    assert "Successfully applied time stretch" in result.output
    mock_read.assert_called_once_with(input_file)
    mock_core_stretch.assert_called_once()
    call_args, call_kwargs = mock_core_stretch.call_args
    assert_equal(call_kwargs.get('y'), dummy_audio)
    assert call_kwargs.get('rate') == rate
    mock_save.assert_called_once()
    save_call_args, save_call_kwargs = mock_save.call_args
    assert_equal(save_call_args[0][0], stretched_audio)
    assert save_call_args[0][1] == sr
    assert save_call_args[1] == output_file

# Test error handling
def test_augment_cmd_invalid_input_type(runner: CliRunner, tmp_path: Path, mocker):
    """Test augment commands with non-audio input."""
    # Mock read_data to simulate reading non-audio (e.g., DataFrame) which causes UsageError in command
    mock_read = mocker.patch("sygnals.cli.augment_cmd.read_data", side_effect=click.UsageError("Input file 'input.csv' is not recognized as audio."))
    input_file = tmp_path / "input.csv"
    input_file.touch()
    output_file = tmp_path / "output.wav"

    # Try add-noise
    args_noise = ["augment", "add-noise", str(input_file), "-o", str(output_file), "--snr", "10"]
    result_noise = runner.invoke(cli, args_noise) # Use default catch_exceptions=True

    # Assertions for error handling
    assert result_noise.exit_code != 0, f"Expected non-zero exit code for invalid input, got {result_noise.exit_code}"
    assert result_noise.exception is not None, f"Expected an exception, but got None. Output:\n{result_noise.output}\nStderr:\n{result_noise.stderr}"
    assert isinstance(result_noise.exception, SystemExit), f"Expected SystemExit, got {type(result_noise.exception)}"
    # Check the exit code from SystemExit (usually 1 for Abort/UsageError handled by ConfigGroup)
    assert result_noise.exception.code != 0

    # Try pitch-shift (expect similar failure)
    args_pitch = ["augment", "pitch-shift", str(input_file), "-o", str(output_file), "--steps", "1"]
    result_pitch = runner.invoke(cli, args_pitch)
    assert result_pitch.exit_code != 0
    assert result_pitch.exception is not None
    assert isinstance(result_pitch.exception, SystemExit)
    assert result_pitch.exception.code != 0


def test_augment_cmd_missing_option(runner: CliRunner, tmp_path: Path):
    """Test augment commands with missing required options."""
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_file = tmp_path / "output.wav"

    # Missing --snr for add-noise
    args_noise = ["augment", "add-noise", str(input_file), "-o", str(output_file)]
    result_noise = runner.invoke(cli, args_noise)
    assert result_noise.exit_code != 0, f"Expected non-zero exit code for missing --snr, got {result_noise.exit_code}"
    assert result_noise.exception is not None, "Expected an exception for missing --snr"
    assert isinstance(result_noise.exception, SystemExit), f"Expected SystemExit, got {type(result_noise.exception)}"
    # Click usually exits with 2 for missing parameters/options
    assert result_noise.exception.code == 2

    # Missing --steps for pitch-shift
    args_pitch = ["augment", "pitch-shift", str(input_file), "-o", str(output_file)]
    result_pitch = runner.invoke(cli, args_pitch)
    assert result_pitch.exit_code != 0, f"Expected non-zero exit code for missing --steps, got {result_pitch.exit_code}"
    assert result_pitch.exception is not None, "Expected an exception for missing --steps"
    assert isinstance(result_pitch.exception, SystemExit), f"Expected SystemExit, got {type(result_pitch.exception)}"
    assert result_pitch.exception.code == 2

    # Missing --rate for time-stretch
    args_stretch = ["augment", "time-stretch", str(input_file), "-o", str(output_file)]
    result_stretch = runner.invoke(cli, args_stretch)
    assert result_stretch.exit_code != 0, f"Expected non-zero exit code for missing --rate, got {result_stretch.exit_code}"
    assert result_stretch.exception is not None, "Expected an exception for missing --rate"
    assert isinstance(result_stretch.exception, SystemExit), f"Expected SystemExit, got {type(result_stretch.exception)}"
    assert result_stretch.exception.code == 2
