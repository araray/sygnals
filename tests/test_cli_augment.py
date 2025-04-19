# tests/test_cli_augment.py

"""
Tests for the 'sygnals augment' CLI command group.
"""

import pytest
import numpy as np
from pathlib import Path
from click.testing import CliRunner

# Import the main CLI entry point and the command group/functions to test/mock
from sygnals.cli.main import cli
from sygnals.core import augment # To mock functions inside

# --- Test Fixtures ---

@pytest.fixture
def runner() -> CliRunner:
    """Provides a Click CliRunner instance."""
    return CliRunner()

@pytest.fixture
def mock_audio_read_save(mocker):
    """Mocks read_data and save_data."""
    sr = 16000
    dummy_audio = np.random.randn(sr * 3).astype(np.float64) # 3 seconds
    mock_read = mocker.patch("sygnals.cli.augment_cmd.read_data", return_value=(dummy_audio, sr))
    mock_save = mocker.patch("sygnals.cli.augment_cmd.save_data")
    return mock_read, mock_save, dummy_audio, sr

# --- Test Cases ---

# Test 'augment add-noise'
def test_augment_add_noise_cmd(runner: CliRunner, mock_audio_read_save, tmp_path: Path, mocker):
    """Test successful execution of 'augment add-noise'."""
    mock_read, mock_save, dummy_audio, sr = mock_audio_read_save
    mock_core_noise = mocker.patch.object(augment, 'add_noise', return_value=dummy_audio * 0.5) # Return modified audio
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

    print("CLI Output:\n", result.output) # For debugging
    if result.exception: print("Exception:\n", result.exception)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}"
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
    assert_equal(save_call_args[0][0], dummy_audio * 0.5) # Check augmented data
    assert save_call_args[0][1] == sr # Check sample rate
    assert save_call_args[1] == output_file # Check output path


# Test 'augment pitch-shift'
def test_augment_pitch_shift_cmd(runner: CliRunner, mock_audio_read_save, tmp_path: Path, mocker):
    """Test successful execution of 'augment pitch-shift'."""
    mock_read, mock_save, dummy_audio, sr = mock_audio_read_save
    mock_core_pitch = mocker.patch.object(augment, 'pitch_shift', return_value=dummy_audio * 0.8)
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

    assert result.exit_code == 0
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
    assert_equal(save_call_args[0][0], dummy_audio * 0.8) # Check augmented data
    assert save_call_args[0][1] == sr
    assert save_call_args[1] == output_file


# Test 'augment time-stretch'
def test_augment_time_stretch_cmd(runner: CliRunner, mock_audio_read_save, tmp_path: Path, mocker):
    """Test successful execution of 'augment time-stretch'."""
    mock_read, mock_save, dummy_audio, sr = mock_audio_read_save
    # Simulate stretching changing the length
    stretched_audio = np.random.randn(len(dummy_audio) // 2).astype(np.float64)
    mock_core_stretch = mocker.patch.object(augment, 'time_stretch', return_value=stretched_audio)
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_file = tmp_path / "output_stretched.wav"

    rate = 1.5 # Speed up

    args = [
        "augment", "time-stretch", str(input_file),
        "--output", str(output_file),
        "--rate", str(rate)
    ]

    result = runner.invoke(cli, args)

    assert result.exit_code == 0
    assert "Successfully applied time stretch" in result.output

    mock_read.assert_called_once_with(input_file)
    mock_core_stretch.assert_called_once()
    call_args, call_kwargs = mock_core_stretch.call_args
    assert_equal(call_kwargs.get('y'), dummy_audio)
    assert call_kwargs.get('rate') == rate

    mock_save.assert_called_once()
    save_call_args, save_call_kwargs = mock_save.call_args
    assert_equal(save_call_args[0][0], stretched_audio) # Check augmented data
    assert save_call_args[0][1] == sr # SR remains the same
    assert save_call_args[1] == output_file


# Test error handling
def test_augment_cmd_invalid_input_type(runner: CliRunner, tmp_path: Path, mocker):
    """Test augment commands with non-audio input."""
    # Mock read_data to return a DataFrame
    mock_read = mocker.patch("sygnals.cli.augment_cmd.read_data", return_value=pd.DataFrame({'a': [1]}))
    input_file = tmp_path / "input.csv"
    input_file.touch()
    output_file = tmp_path / "output.wav"

    # Try add-noise
    args_noise = ["augment", "add-noise", str(input_file), "-o", str(output_file), "--snr", "10"]
    result_noise = runner.invoke(cli, args_noise)
    assert result_noise.exit_code != 0
    assert "Input file" in result_noise.output and "not recognized as audio" in result_noise.output

    # Try pitch-shift
    args_pitch = ["augment", "pitch-shift", str(input_file), "-o", str(output_file), "--steps", "1"]
    result_pitch = runner.invoke(cli, args_pitch)
    assert result_pitch.exit_code != 0
    assert "Input file" in result_pitch.output and "not recognized as audio" in result_pitch.output


def test_augment_cmd_missing_option(runner: CliRunner, tmp_path: Path):
    """Test augment commands with missing required options."""
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_file = tmp_path / "output.wav"

    # Missing --snr for add-noise
    args_noise = ["augment", "add-noise", str(input_file), "-o", str(output_file)]
    result_noise = runner.invoke(cli, args_noise)
    assert result_noise.exit_code != 0
    assert "Missing option" in result_noise.output and "--snr" in result_noise.output

    # Missing --steps for pitch-shift
    args_pitch = ["augment", "pitch-shift", str(input_file), "-o", str(output_file)]
    result_pitch = runner.invoke(cli, args_pitch)
    assert result_pitch.exit_code != 0
    assert "Missing option" in result_pitch.output and "--steps" in result_pitch.output

    # Missing --rate for time-stretch
    args_stretch = ["augment", "time-stretch", str(input_file), "-o", str(output_file)]
    result_stretch = runner.invoke(cli, args_stretch)
    assert result_stretch.exit_code != 0
    assert "Missing option" in result_stretch.output and "--rate" in result_stretch.output
