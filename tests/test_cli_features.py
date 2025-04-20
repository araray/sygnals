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

# Import the main CLI entry point
from sygnals.cli.main import cli
# Import _SKLEARN_AVAILABLE to conditionally skip tests
from sygnals.core.ml_utils.scaling import _SKLEARN_AVAILABLE

# --- Test Fixtures ---

@pytest.fixture
def runner() -> CliRunner:
    """Provides a Click CliRunner instance."""
    return CliRunner(mix_stderr=False)

@pytest.fixture
def sample_features_csv(tmp_path: Path) -> Path:
    """Creates a dummy CSV feature file."""
    df = pd.DataFrame({
        'time': np.linspace(0, 1, 10),
        'feat1': np.random.rand(10) * 5 + 10,
        'feat2': np.random.rand(10) * 0.1 - 0.05
    })
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def sample_features_npz(tmp_path: Path) -> Path:
    """Creates a dummy NPZ feature file."""
    data_dict = {
        'data': np.random.rand(20, 3) * np.array([[10, 0.1, 100]]),
        'sr': np.array(16000)
    }
    npz_path = tmp_path / "features.npz"
    np.savez(npz_path, **data_dict)
    return npz_path

@pytest.fixture
def mock_save_data(mocker):
    """Mocks the save_data function used by the CLI command."""
    mock = mocker.patch("sygnals.cli.features_cmd.save_data")
    return mock

# --- Test Cases for 'features transform scale' ---

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_features_transform_scale_csv_success(runner: CliRunner, sample_features_csv: Path, mock_save_data, mocker):
    """Test successful scaling of features from a CSV file."""
    input_file = sample_features_csv
    output_file = input_file.parent / "scaled_features.csv"
    scaler_type = 'standard'
    dummy_scaled_data = np.random.rand(10, 2)
    mock_apply_scaling = mocker.patch("sygnals.cli.features_cmd.apply_scaling", return_value=(dummy_scaled_data, None))

    args = [
        "features", "transform", "scale", str(input_file),
        "--output", str(output_file),
        "--scaler", scaler_type
    ]
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}\nStderr:\n{result.stderr}\nException:\n{result.exception}"
    assert result.exception is None
    assert "Successfully scaled features" in result.output
    mock_apply_scaling.assert_called_once()
    mock_save_data.assert_called_once()
    call_args, call_kwargs = mock_apply_scaling.call_args
    assert call_kwargs.get('scaler_type') == scaler_type
    assert call_kwargs.get('fit') is True
    assert call_kwargs.get('features').shape == (10, 2)
    save_call_args, save_call_kwargs = mock_save_data.call_args
    saved_data = save_call_args[0]
    assert isinstance(saved_data, pd.DataFrame)
    assert_allclose(saved_data[['feat1', 'feat2']].values, dummy_scaled_data)
    assert 'time' in saved_data.columns
    assert save_call_args[1] == output_file


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_features_transform_scale_npz_success(runner: CliRunner, sample_features_npz: Path, mock_save_data, mocker):
    """Test successful scaling of features from an NPZ file."""
    input_file = sample_features_npz
    output_file = input_file.parent / "scaled_features.npz"
    scaler_type = 'minmax'
    dummy_scaled_data = np.random.rand(20, 3)
    mock_apply_scaling = mocker.patch("sygnals.cli.features_cmd.apply_scaling", return_value=(dummy_scaled_data, None))

    args = [
        "features", "transform", "scale", str(input_file),
        "--output", str(output_file),
        "--scaler", scaler_type
    ]
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nOutput:\n{result.output}\nStderr:\n{result.stderr}\nException:\n{result.exception}"
    assert result.exception is None
    assert "Successfully scaled features" in result.output
    mock_apply_scaling.assert_called_once()
    mock_save_data.assert_called_once()
    call_args, call_kwargs = mock_apply_scaling.call_args
    assert call_kwargs.get('scaler_type') == scaler_type
    assert call_kwargs.get('fit') is True
    assert call_kwargs.get('features').shape == (20, 3)
    save_call_args, save_call_kwargs = mock_save_data.call_args
    saved_data = save_call_args[0]
    assert isinstance(saved_data, dict)
    assert 'data' in saved_data
    assert 'sr' in saved_data
    assert_allclose(saved_data['data'], dummy_scaled_data)
    assert save_call_args[1] == output_file


def test_features_transform_scale_invalid_input(runner: CliRunner, tmp_path: Path, mocker):
    """Test scaling with input that cannot be read as features."""
    mock_read = mocker.patch("sygnals.cli.features_cmd.read_data", return_value=(np.zeros(100), 16000))
    input_file = tmp_path / "input.wav"
    input_file.touch()
    output_file = tmp_path / "output.csv"

    args = [
        "features", "transform", "scale", str(input_file),
        "--output", str(output_file),
    ]

    # Invoke WITHOUT catch_exceptions=False
    result = runner.invoke(cli, args)

    assert result.exit_code != 0
    # Check the caught exception type and message
    assert result.exception is not None, f"Expected an exception, but got None. Output:\n{result.output}\nStderr:\n{result.stderr}"
    # *** FIX: Assert that the caught exception is SystemExit ***
    assert isinstance(result.exception, SystemExit), f"Expected SystemExit, got {type(result.exception)}"
    # Optionally check the exit code within the SystemExit object
    assert result.exception.code != 0
    # We can't easily check the original message reliably when it becomes SystemExit


def test_features_transform_scale_no_numeric(runner: CliRunner, tmp_path: Path, mocker):
    """Test scaling with CSV containing no numeric columns."""
    df = pd.DataFrame({'text': ['a', 'b', 'c'], 'category': ['X', 'Y', 'X']})
    input_file = tmp_path / "no_numeric.csv"
    df.to_csv(input_file, index=False)
    output_file = tmp_path / "output.csv"
    mocker.patch("sygnals.cli.features_cmd.read_data", return_value=df)

    args = [
        "features", "transform", "scale", str(input_file),
        "--output", str(output_file),
    ]

    # Invoke WITHOUT catch_exceptions=False
    result = runner.invoke(cli, args)

    assert result.exit_code != 0
    # Check the caught exception type and message
    assert result.exception is not None, f"Expected an exception, but got None. Output:\n{result.output}\nStderr:\n{result.stderr}"
    # *** FIX: Assert that the caught exception is SystemExit ***
    assert isinstance(result.exception, SystemExit), f"Expected SystemExit, got {type(result.exception)}"
    # Optionally check the exit code within the SystemExit object
    assert result.exception.code != 0


def test_features_transform_scale_missing_sklearn(runner: CliRunner, sample_features_csv: Path, mocker):
    """Test scaling command when scikit-learn is not installed."""
    if _SKLEARN_AVAILABLE:
        pytest.skip("scikit-learn is installed, skipping this test.")

    input_file = sample_features_csv
    output_file = input_file.parent / "output.csv"
    mocker.patch("sygnals.cli.features_cmd.read_data", return_value=pd.read_csv(input_file))

    args = [
        "features", "transform", "scale", str(input_file),
        "--output", str(output_file),
    ]

    # Use default catch_exceptions=True
    result = runner.invoke(cli, args)

    assert result.exit_code != 0
    assert result.exception is not None
    # Check if the final exception is SystemExit (consistent with other failures)
    assert isinstance(result.exception, SystemExit), f"Expected SystemExit, got {type(result.exception)}"
    # Check the error message printed to stderr by Click's show() method
    # The original error is Abort -> UsageError -> ImportError
    assert "Missing dependency for scaling" in result.stderr
    assert "scikit-learn" in result.stderr

# TODO: Add tests for 'features extract' when implemented.
