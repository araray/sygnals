# tests/test_cli_save.py

"""
Tests for the 'sygnals save' CLI command group, focusing on 'save dataset'.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Literal, Dict, Any, Tuple
from numpy.testing import assert_allclose, assert_equal
import logging # Import logging for caplog
import click # Import click for exception checking
import json # For creating JSON string for aggregation test

from click.testing import CliRunner

# Import the main CLI entry point and the command group/functions to test/mock
from sygnals.cli.main import cli
# Import formatters to mock later if implemented
# We will mock them within the test functions using mocker

# --- Test Fixtures ---

@pytest.fixture
def runner() -> CliRunner:
    """Provides a Click CliRunner instance."""
    # Capture stderr for checking error messages
    return CliRunner(mix_stderr=False)

# Fixture for input data suitable for 'vectors' or 'sequences' (dict of 1D arrays)
@pytest.fixture
def input_features_dict_npz(tmp_path: Path) -> Path:
    """Creates dummy NPZ feature file (dict of 1D arrays)."""
    data_dict = {
        'rms': np.random.rand(50).astype(np.float64),
        'zcr': np.random.rand(50).astype(np.float64) * 0.1,
        'time': np.linspace(0, 5, 50) # Include a non-feature array
    }
    fpath = tmp_path / "input_dict_features.npz"
    np.savez(fpath, **data_dict)
    return fpath

# Fixture for input data suitable for 'image' (2D array)
@pytest.fixture
def input_image_npz(tmp_path: Path) -> Path:
    """Creates dummy NPZ feature file (2D array)."""
    data_dict = {
        'spectrogram': np.random.rand(64, 80).astype(np.float64),
        'other_info': np.array([1, 2, 3])
    }
    fpath = tmp_path / "input_image_features.npz"
    np.savez(fpath, **data_dict)
    return fpath

# Fixture for segment info file
@pytest.fixture
def segment_info_csv(tmp_path: Path) -> Path:
    """Creates a dummy segment info CSV file."""
    df = pd.DataFrame({
        'start_frame': [0, 15, 30],
        'end_frame': [15, 30, 50],
        'label': ['A', 'B', 'A']
    })
    fpath = tmp_path / "segment_info.csv"
    df.to_csv(fpath, index=False)
    return fpath


@pytest.fixture
def mock_save_data(mocker):
    """Mocks the save_data function used by the CLI command."""
    # Mock save_data where it's called in the CLI module
    mock = mocker.patch("sygnals.cli.save_cmd.save_data")
    return mock

# --- Test Cases for 'save dataset' ---

def test_save_dataset_assembly_none(runner: CliRunner, input_features_dict_npz: Path, mock_save_data, mocker):
    """Test 'save dataset' with assembly_method='none'."""
    input_file = input_features_dict_npz
    # Load expected data for comparison
    expected_data = dict(np.load(input_file))
    output_file = input_file.parent / f"output_dataset{input_file.suffix}"
    assembly = 'none'

    # Mock read_data to return the expected dictionary
    mock_read = mocker.patch("sygnals.cli.save_cmd.read_data", return_value=expected_data)

    args = [
        "save", "dataset", str(input_file),
        "--output", str(output_file),
        "--assembly-method", assembly
    ]
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nStderr:\n{result.stderr}"
    assert result.exception is None
    assert "Successfully saved dataset" in result.output
    assert f"(assembly: {assembly})" in result.output
    mock_read.assert_called_once_with(input_file)
    mock_save_data.assert_called_once()
    save_call_args, _ = mock_save_data.call_args
    saved_data = save_call_args[0]
    saved_path = save_call_args[1]
    assert isinstance(saved_data, dict)
    assert saved_data.keys() == expected_data.keys()
    for key in expected_data: assert_allclose(saved_data[key], expected_data[key])
    assert saved_path == output_file


def test_save_dataset_format_override(runner: CliRunner, input_features_dict_npz: Path, mock_save_data, mocker):
    """Test 'save dataset' with --format override."""
    input_file = input_features_dict_npz
    expected_data = dict(np.load(input_file))
    output_base = input_file.parent / "output_override"
    override_format = "csv" # Override to csv
    expected_output_file = output_base.with_suffix(f".{override_format}")
    mock_read = mocker.patch("sygnals.cli.save_cmd.read_data", return_value=expected_data)

    args = [ "save", "dataset", str(input_file), "--output", str(output_base), "--format", override_format ]
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nStderr:\n{result.stderr}"
    assert result.exception is None
    mock_save_data.assert_called_once()
    save_call_args, _ = mock_save_data.call_args
    assert save_call_args[1] == expected_output_file # Check path uses overridden format


def test_save_dataset_assembly_vectors(runner: CliRunner, input_features_dict_npz: Path, segment_info_csv: Path, mock_save_data, mocker):
    """Test 'save dataset' with assembly_method='vectors'."""
    input_file = input_features_dict_npz
    input_data_dict = dict(np.load(input_file))
    segment_file = segment_info_csv
    output_file = input_file.parent / "output_vectors.csv" # Output as CSV
    assembly = 'vectors'
    aggregation = 'std'
    # Mock read_data: first call reads features, second reads segment info
    mock_read = mocker.patch("sygnals.cli.save_cmd.read_data")
    mock_read.side_effect = [input_data_dict, pd.read_csv(segment_file)]
    # Mock the formatter
    dummy_formatted_data = pd.DataFrame({'rms': [0.1, 0.2], 'zcr': [0.3, 0.4]})
    mock_formatter = mocker.patch("sygnals.cli.save_cmd.format_feature_vectors_per_segment", return_value=dummy_formatted_data)

    args = [
        "save", "dataset", str(input_file),
        "--output", str(output_file),
        "--assembly-method", assembly,
        "--segment-info", str(segment_file),
        "--aggregation", aggregation
    ]
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nStderr:\n{result.stderr}"
    assert result.exception is None
    assert mock_read.call_count == 2
    mock_formatter.assert_called_once()
    # Verify formatter arguments
    fmt_call_args, fmt_call_kwargs = mock_formatter.call_args
    assert 'features_dict' in fmt_call_kwargs
    assert 'segment_indices' in fmt_call_kwargs
    assert fmt_call_kwargs.get('aggregation') == aggregation
    assert list(fmt_call_kwargs['features_dict'].keys()) == ['rms', 'zcr'] # Check only feature keys passed
    assert len(fmt_call_kwargs['segment_indices']) == 3 # From fixture
    # Verify save_data call
    mock_save_data.assert_called_once_with(dummy_formatted_data, output_file)


def test_save_dataset_assembly_vectors_json_agg(runner: CliRunner, input_features_dict_npz: Path, segment_info_csv: Path, mock_save_data, mocker):
    """Test 'vectors' assembly with JSON aggregation string."""
    input_file = input_features_dict_npz
    input_data_dict = dict(np.load(input_file))
    segment_file = segment_info_csv
    output_file = input_file.parent / "output_vectors.npz"
    assembly = 'vectors'
    agg_dict = {'rms': 'max', 'zcr': 'mean'}
    agg_json = json.dumps(agg_dict)
    mock_read = mocker.patch("sygnals.cli.save_cmd.read_data")
    mock_read.side_effect = [input_data_dict, pd.read_csv(segment_file)]
    dummy_formatted_data = np.random.rand(3, 2)
    mock_formatter = mocker.patch("sygnals.cli.save_cmd.format_feature_vectors_per_segment", return_value=dummy_formatted_data)

    args = [ "save", "dataset", str(input_file), "-o", str(output_file), "--assembly-method", assembly, "--segment-info", str(segment_file), "--aggregation", agg_json ]
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nStderr:\n{result.stderr}"
    mock_formatter.assert_called_once()
    _, fmt_call_kwargs = mock_formatter.call_args
    assert fmt_call_kwargs.get('aggregation') == agg_dict # Check parsed dict
    mock_save_data.assert_called_once_with(dummy_formatted_data, output_file)


def test_save_dataset_assembly_vectors_missing_info(runner: CliRunner, input_features_dict_npz: Path):
    """Test 'vectors' assembly without required --segment-info."""
    args = [ "save", "dataset", str(input_features_dict_npz), "-o", "out.npz", "--assembly-method", "vectors" ]
    result = runner.invoke(cli, args)
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "Option '--segment-info' is required" in result.stderr


def test_save_dataset_assembly_sequences(runner: CliRunner, input_features_dict_npz: Path, mock_save_data, mocker):
    """Test 'save dataset' with assembly_method='sequences'."""
    input_file = input_features_dict_npz
    input_data_dict = dict(np.load(input_file))
    output_file = input_file.parent / "output_sequences.npz" # Output NPZ
    assembly = 'sequences'
    max_len = 40
    pad_val = -1.0
    trunc = 'pre'
    mock_read = mocker.patch("sygnals.cli.save_cmd.read_data", return_value=input_data_dict)
    # Formatter returns 3D array (1, seq_len, n_feat)
    dummy_formatted_data = np.random.rand(1, max_len, 2)
    mock_formatter = mocker.patch("sygnals.cli.save_cmd.format_feature_sequences", return_value=dummy_formatted_data)

    args = [
        "save", "dataset", str(input_file),
        "--output", str(output_file),
        "--assembly-method", assembly,
        "--max-sequence-length", str(max_len),
        "--padding-value", str(pad_val),
        "--truncation-strategy", trunc
    ]
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nStderr:\n{result.stderr}"
    assert result.exception is None
    mock_read.assert_called_once_with(input_file)
    mock_formatter.assert_called_once()
    # Verify formatter arguments
    fmt_call_args, fmt_call_kwargs = mock_formatter.call_args
    assert 'features_dict' in fmt_call_kwargs
    assert fmt_call_kwargs.get('max_sequence_length') == max_len
    assert fmt_call_kwargs.get('padding_value') == pad_val
    assert fmt_call_kwargs.get('truncation_strategy') == trunc
    assert fmt_call_kwargs.get('output_format') == 'padded_array' # Check forced output format
    # Verify save_data call (should save dict for NPZ)
    mock_save_data.assert_called_once()
    save_call_args, _ = mock_save_data.call_args
    assert isinstance(save_call_args[0], dict)
    assert 'sequences' in save_call_args[0]
    assert_equal(save_call_args[0]['sequences'], dummy_formatted_data)
    assert save_call_args[1] == output_file


def test_save_dataset_assembly_image(runner: CliRunner, input_image_npz: Path, mock_save_data, mocker):
    """Test 'save dataset' with assembly_method='image'."""
    input_file = input_image_npz
    input_data_dict = dict(np.load(input_file))
    output_file = input_file.parent / "output_image.npz" # Output NPZ
    assembly = 'image'
    shape_str = "32,40"
    expected_shape = (32, 40)
    resize_order = 3
    mock_read = mocker.patch("sygnals.cli.save_cmd.read_data", return_value=input_data_dict)
    dummy_formatted_data = np.random.rand(*expected_shape)
    mock_formatter = mocker.patch("sygnals.cli.save_cmd.format_features_as_image", return_value=dummy_formatted_data)

    args = [
        "save", "dataset", str(input_file),
        "--output", str(output_file),
        "--assembly-method", assembly,
        "--output-shape", shape_str,
        "--resize-order", str(resize_order),
        "--no-normalize" # Test disabling normalization
    ]
    result = runner.invoke(cli, args)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nStderr:\n{result.stderr}"
    assert result.exception is None
    mock_read.assert_called_once_with(input_file)
    mock_formatter.assert_called_once()
    # Verify formatter arguments
    fmt_call_args, fmt_call_kwargs = mock_formatter.call_args
    assert 'feature_map' in fmt_call_kwargs
    assert_equal(fmt_call_kwargs['feature_map'], input_data_dict['spectrogram']) # Check correct array extracted
    assert fmt_call_kwargs.get('output_shape') == expected_shape
    assert fmt_call_kwargs.get('resize_order') == resize_order
    assert fmt_call_kwargs.get('normalize') is False # Check --no-normalize flag
    # Verify save_data call (should save dict for NPZ)
    mock_save_data.assert_called_once()
    save_call_args, _ = mock_save_data.call_args
    assert isinstance(save_call_args[0], dict)
    assert 'image_features' in save_call_args[0]
    assert_equal(save_call_args[0]['image_features'], dummy_formatted_data)
    assert save_call_args[1] == output_file


def test_save_dataset_assembly_image_bad_input(runner: CliRunner, input_features_dict_npz: Path, mocker):
    """Test 'image' assembly with unsuitable input data (dict of 1D arrays)."""
    input_file = input_features_dict_npz
    input_data_dict = dict(np.load(input_file))
    output_file = input_file.parent / "output_image_error.npz"
    assembly = 'image'
    mock_read = mocker.patch("sygnals.cli.save_cmd.read_data", return_value=input_data_dict)
    mock_formatter = mocker.patch("sygnals.cli.save_cmd.format_features_as_image")
    mock_save = mocker.patch("sygnals.cli.save_cmd.save_data")

    args = [ "save", "dataset", str(input_file), "-o", str(output_file), "--assembly-method", assembly ]
    result = runner.invoke(cli, args)

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "Input data type <class 'dict'> not suitable for 'image' assembly" in result.stderr
    mock_formatter.assert_not_called()
    mock_save.assert_not_called()


def test_save_dataset_invalid_input_file(runner: CliRunner, tmp_path: Path, mocker):
    """Test 'save dataset' with input that cannot be read."""
    mock_read = mocker.patch("sygnals.cli.save_cmd.read_data", side_effect=ValueError("Cannot read this"))
    input_file = tmp_path / "input.xyz" # Use invalid extension
    input_file.touch()
    output_file = tmp_path / "output.npz"
    args = [ "save", "dataset", str(input_file), "--output", str(output_file) ]
    result = runner.invoke(cli, args)

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "Error during dataset assembly/saving: Cannot read this" in result.stderr
