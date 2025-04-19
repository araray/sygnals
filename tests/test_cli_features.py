# tests/test_cli_features.py

"""
Tests for the 'sygnals features' CLI command group, focusing on 'transform scale'.
Tests for 'features extract' can be added here later.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from click.testing import CliRunner
import click # Import click to check for exceptions
from numpy.testing import assert_allclose # Import assert_allclose

# Import the main CLI entry point and the command group/functions to test/mock
from sygnals.cli.main import cli
# Import the module to patch, but patch where it's *used*
# from sygnals.core.ml_utils import scaling # Don't need this for patching target
from sygnals.core.ml_utils.scaling import _SKLEARN_AVAILABLE # To conditionally skip

# --- Test Fixtures ---

@pytest.fixture
def runner() -> CliRunner:
    """Provides a Click CliRunner instance."""
    # Capture stderr for checking error messages
    return CliRunner(mix_stderr=False)

@pytest.fixture
def sample_features_csv(tmp_path: Path) -> Path:
    """Creates a dummy CSV feature file."""
    df = pd.DataFrame({
        'time': np.linspace(0, 1, 10),
        'feat1': np.random.rand(10) * 5 + 10, # Scale 10-15
        'feat2': np.random.rand(10) * 0.1 - 0.05 # Scale -0.05 to 0.05
    })
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def sample_features_npz(tmp_path: Path) -> Path:
    """Creates a dummy NPZ feature file."""
    data_dict = {
        'data': np.random.rand(20, 3) * np.array([[10, 0.1, 100]]), # Different scales
        'sr': np.array(16000) # Example metadata
    }
    npz_path = tmp_path / "features.npz"
    np.savez(npz_path, **data_dict)
    return npz_path

@pytest.fixture
def mock_save_data(mocker):
    """Mocks the save_data function used by the CLI command."""
    # Mock save_data where it's called in the CLI module
    mock = mocker.patch("sygnals.cli.features_cmd.save_data")
    return mock

# --- Test Cases for 'features transform scale' ---

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_features_transform_scale_csv_success(runner: CliRunner, sample_features_csv: Path, mock_save_data, mocker):
    """Test successful scaling of features from a CSV file."""
    input_file = sample_features_csv
    output_file = input_file.parent / "scaled_features.csv"
    scaler_type = 'standard'

    # Mock the core scaling function where it's called in the CLI module
    # FIX: Correct patch target
    dummy_scaled_data = np.random.rand(10, 2) # Shape matches numeric columns in CSV
    mock_apply_scaling = mocker.patch("sygnals.cli.features_cmd.apply_scaling", return_value=(dummy_scaled_data, None)) # Don't care about scaler instance here

    args = [
        "features", "transform", "scale", str(input_file),
        "--output", str(output_file),
        "--scaler", scaler_type
    ]

    result = runner.invoke(cli, args)

    print("CLI Output:\n", result.output) # For debugging stdout
    print("CLI Stderr:\n", result.stderr) # For debugging stderr
    if result.exception: print("Exception:\n", result.exception)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}"
    assert "Successfully scaled features" in result.output

    # Verify apply_scaling was called correctly
    mock_apply_scaling.assert_called_once()
    call_args, call_kwargs = mock_apply_scaling.call_args
    assert call_kwargs.get('scaler_type') == scaler_type
    # FIX: Check that fit=True was passed (now explicitly passed by CLI command)
    assert call_kwargs.get('fit') is True
    # Check input features shape (should be 10 samples, 2 numeric features)
    assert call_kwargs.get('features').shape == (10, 2)

    # Verify save_data was called correctly
    mock_save_data.assert_called_once()
    save_call_args, save_call_kwargs = mock_save_data.call_args
    saved_data = save_call_args[0]
    assert isinstance(saved_data, pd.DataFrame) # Should save DataFrame for CSV output
    assert_allclose(saved_data[['feat1', 'feat2']].values, dummy_scaled_data) # Check scaled data content
    assert 'time' in saved_data.columns # Check time column preserved
    assert save_call_args[1] == output_file


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_features_transform_scale_npz_success(runner: CliRunner, sample_features_npz: Path, mock_save_data, mocker):
    """Test successful scaling of features from an NPZ file."""
    input_file = sample_features_npz
    output_file = input_file.parent / "scaled_features.npz"
    scaler_type = 'minmax'

    # Mock the core scaling function where it's called
    # FIX: Correct patch target
    dummy_scaled_data = np.random.rand(20, 3) # Shape matches 'data' array in NPZ
    mock_apply_scaling = mocker.patch("sygnals.cli.features_cmd.apply_scaling", return_value=(dummy_scaled_data, None))

    args = [
        "features", "transform", "scale", str(input_file),
        "--output", str(output_file),
        "--scaler", scaler_type
    ]

    result = runner.invoke(cli, args)

    print("CLI Output:\n", result.output) # For debugging stdout
    print("CLI Stderr:\n", result.stderr) # For debugging stderr
    if result.exception: print("Exception:\n", result.exception)

    assert result.exit_code == 0
    assert "Successfully scaled features" in result.output

    # Verify apply_scaling was called correctly
    mock_apply_scaling.assert_called_once()
    call_args, call_kwargs = mock_apply_scaling.call_args
    assert call_kwargs.get('scaler_type') == scaler_type
    # FIX: Check that fit=True was passed
    assert call_kwargs.get('fit') is True
    assert call_kwargs.get('features').shape == (20, 3) # Check input shape

    # Verify save_data was called correctly
    mock_save_data.assert_called_once()
    save_call_args, save_call_kwargs = mock_save_data.call_args
    saved_data = save_call_args[0]
    assert isinstance(saved_data, dict) # Should save dict for NPZ output
    assert 'data' in saved_data # Check original keys preserved/updated
    assert 'sr' in saved_data
    assert_allclose(saved_data['data'], dummy_scaled_data) # Check scaled data content
    assert save_call_args[1] == output_file


def test_features_transform_scale_invalid_input(runner: CliRunner, tmp_path: Path, mocker):
    """Test scaling with input that cannot be read as features."""
    # Mock read_data where it's called in the CLI module
    mock_read = mocker.patch("sygnals.cli.features_cmd.read_data", return_value=(np.zeros(100), 16000))
    input_file = tmp_path / "input.wav" # Use WAV extension
    input_file.touch()
    output_file = tmp_path / "output.csv"

    args = [
        "features", "transform", "scale", str(input_file),
        "--output", str(output_file),
    ]

    result = runner.invoke(cli, args, catch_exceptions=False) # Catch exceptions

    assert result.exit_code != 0
    # FIX: Check the original exception type and message using exc_info
    assert result.exc_info is not None
    exc_type, exc_value, _ = result.exc_info
    assert issubclass(exc_type, click.exceptions.Abort) # Check if it aborted
    assert "Input file" in str(exc_value) and "not suitable for feature scaling" in str(exc_value)


def test_features_transform_scale_no_numeric(runner: CliRunner, tmp_path: Path, mocker):
    """Test scaling with CSV containing no numeric columns."""
    df = pd.DataFrame({'text': ['a', 'b', 'c'], 'category': ['X', 'Y', 'X']})
    input_file = tmp_path / "no_numeric.csv"
    df.to_csv(input_file, index=False)
    output_file = tmp_path / "output.csv"

    # Mock read_data where it's called in the CLI module
    mocker.patch("sygnals.cli.features_cmd.read_data", return_value=df)

    args = [
        "features", "transform", "scale", str(input_file),
        "--output", str(output_file),
    ]

    result = runner.invoke(cli, args, catch_exceptions=False) # Catch exceptions

    assert result.exit_code != 0
    # FIX: Check the original exception type and message using exc_info
    assert result.exc_info is not None
    exc_type, exc_value, _ = result.exc_info
    # The original error is ValueError, wrapped in UsageError, then Abort
    assert issubclass(exc_type, click.exceptions.Abort)
    assert "Error during feature scaling" in str(exc_value)
    assert "No numeric columns found" in str(exc_value)


def test_features_transform_scale_missing_sklearn(runner: CliRunner, sample_features_csv: Path, mocker):
    """Test scaling command when scikit-learn is not installed."""
    if _SKLEARN_AVAILABLE:
        pytest.skip("scikit-learn is installed, skipping this test.")

    input_file = sample_features_csv
    output_file = input_file.parent / "output.csv"
    # Mock read_data to avoid file system access if needed, but it should fail later
    mocker.patch("sygnals.cli.features_cmd.read_data", return_value=pd.read_csv(input_file))

    args = [
        "features", "transform", "scale", str(input_file),
        "--output", str(output_file),
    ]

    result = runner.invoke(cli, args, catch_exceptions=False) # Catch exceptions

    assert result.exit_code != 0
    # Check stderr for the error message (UsageError might print to stderr)
    # Or check exception message
    assert result.exc_info is not None
    exc_type, exc_value, _ = result.exc_info
    assert issubclass(exc_type, click.exceptions.Abort) # Abort wraps UsageError which wraps ImportError
    assert "Missing dependency for scaling" in str(exc_value)
    assert "scikit-learn" in str(exc_value)

# TODO: Add tests for 'features extract' when implemented.
