# tests/test_data_handler.py

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import json
from pathlib import Path

# Import functions to test
from sygnals.core.data_handler import (
    read_data, save_data, filter_data, run_sql_query,
    ReadResult, SaveInput, # Import type aliases if needed for clarity
    SUPPORTED_READ_FORMATS, SUPPORTED_WRITE_FORMATS # Import constants
)
# Import audio I/O constants for checking
# FIX: Changed imported names to match those defined in sygnals.core.audio.io
from sygnals.core.audio.io import SUPPORTED_READ_EXTENSIONS as AUDIO_READ_EXTENSIONS
from sygnals.core.audio.io import SUPPORTED_WRITE_EXTENSIONS as AUDIO_WRITE_EXTENSIONS

# --- Test read_data ---

def test_read_csv(tmp_path: Path):
    """Test reading a basic CSV file."""
    csv_file = tmp_path / "test.csv"
    csv_content = "time,value\n0,1.0\n1,2.0\n2,3.0\n"
    csv_file.write_text(csv_content)
    data = read_data(csv_file)
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 3
    assert list(data.columns) == ["time", "value"]
    assert data["value"].iloc[1] == 2.0

def test_read_json_records(tmp_path: Path):
    """Test reading a JSON file (records orientation)."""
    json_file = tmp_path / "test_records.json"
    json_data = [{"time": 0, "value": 1.0}, {"time": 1, "value": 2.0}]
    json_file.write_text(json.dumps(json_data))
    data = read_data(json_file)
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2
    assert list(data.columns) == ["time", "value"]
    assert data["value"].iloc[1] == 2.0

def test_read_json_lines(tmp_path: Path):
    """Test reading a JSON file (lines orientation)."""
    json_file = tmp_path / "test_lines.json"
    # Write line-delimited JSON
    with open(json_file, "w") as f:
        f.write('{"time": 0, "value": 1.0}\n')
        f.write('{"time": 1, "value": 2.0}\n')
    data = read_data(json_file)
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2
    assert list(data.columns) == ["time", "value"]
    assert data["value"].iloc[1] == 2.0

def test_read_npz(tmp_path: Path):
    """Test reading an NPZ file returns a dictionary."""
    npz_file = tmp_path / "test.npz"
    array1 = np.arange(5)
    array2 = np.linspace(0, 1, 5)
    np.savez(npz_file, first=array1, second=array2)

    data = read_data(npz_file)
    assert isinstance(data, dict)
    assert "first" in data
    assert "second" in data
    assert_array_equal(data["first"], array1)
    assert_allclose(data["second"], array2)

def test_read_audio_delegation(tmp_path: Path, mocker):
    """Test that read_data delegates to audio loader for audio formats."""
    wav_file = tmp_path / "dummy.wav"
    dummy_audio = np.array([0.1, 0.2, 0.1], dtype=np.float64)
    dummy_sr = 16000
    target_sr = 8000 # Test resampling request
    # Use soundfile directly to create the dummy file
    import soundfile as sf
    sf.write(wav_file, dummy_audio, dummy_sr)

    # Mock the actual audio loading function from audio.io
    mock_load_audio = mocker.patch("sygnals.core.data_handler.load_audio_file", return_value=(dummy_audio / 2, target_sr)) # Simulate resampling

    # Call read_data with the WAV file path and target SR
    result_data, result_sr = read_data(wav_file, sr=target_sr)

    # Assert that the mock was called correctly with the target SR
    mock_load_audio.assert_called_once_with(wav_file, sr=target_sr)

    # Assert that the returned data matches the mock's return value
    assert_allclose(result_data, dummy_audio / 2)
    assert result_sr == target_sr
    assert isinstance(result_data, np.ndarray)
    assert result_data.dtype == np.float64 # Should return float64

def test_read_unsupported_format(tmp_path: Path):
    """Test reading an unsupported file format."""
    unsupported_file = tmp_path / "test.xyz"
    unsupported_file.touch()
    with pytest.raises(ValueError, match="Unsupported file format: '.xyz'"):
        read_data(unsupported_file)

def test_read_file_not_found():
    """Test reading a non-existent file."""
    with pytest.raises(FileNotFoundError):
        read_data("non_existent_file.csv")

def test_read_directory_error(tmp_path: Path):
    """Test reading a directory path."""
    with pytest.raises(ValueError, match="Input path is not a file"):
        read_data(tmp_path)

# --- Test save_data ---

def test_save_dataframe_csv(tmp_path: Path):
    """Test saving a DataFrame to CSV."""
    df = pd.DataFrame({"colA": [1, 2], "colB": ["x", "y"]})
    out_file = tmp_path / "out.csv"
    save_data(df, out_file)
    assert out_file.exists()
    loaded_df = pd.read_csv(out_file)
    pd.testing.assert_frame_equal(df, loaded_df)

def test_save_dataframe_json(tmp_path: Path):
    """Test saving a DataFrame to JSON."""
    df = pd.DataFrame({"colA": [1, 2], "colB": ["x", "y"]})
    out_file = tmp_path / "out.json"
    save_data(df, out_file)
    assert out_file.exists()
    loaded_df = pd.read_json(out_file, orient='records')
    # JSON float precision might differ slightly, convert types if needed for exact match
    pd.testing.assert_frame_equal(df.astype(str), loaded_df.astype(str))

def test_save_dataframe_npz(tmp_path: Path):
    """Test saving a DataFrame to NPZ."""
    df = pd.DataFrame({"colA": [1.5, 2.5], "colB": [3, 4]})
    out_file = tmp_path / "out.npz"
    save_data(df, out_file)
    assert out_file.exists()
    loaded_data = np.load(out_file)
    assert "colA" in loaded_data
    assert "colB" in loaded_data
    assert_allclose(loaded_data["colA"], df["colA"].values)
    assert_array_equal(loaded_data["colB"], df["colB"].values)
    loaded_data.close()

def test_save_numpy_array_npz(tmp_path: Path):
    """Test saving a single NumPy array to NPZ."""
    arr = np.random.rand(5, 3)
    out_file = tmp_path / "out.npz"
    save_data(arr, out_file)
    assert out_file.exists()
    loaded_data = np.load(out_file)
    assert "data" in loaded_data # Default key
    assert_allclose(loaded_data["data"], arr)
    loaded_data.close()

def test_save_numpy_array_csv(tmp_path: Path):
    """Test saving a single 1D NumPy array to CSV."""
    arr = np.linspace(0, 1, 5)
    out_file = tmp_path / "out.csv"
    save_data(arr, out_file)
    assert out_file.exists()
    loaded_df = pd.read_csv(out_file)
    assert list(loaded_df.columns) == ['value'] # Default column name
    assert_allclose(loaded_df['value'].values, arr)

def test_save_dict_npz(tmp_path: Path):
    """Test saving a dictionary of NumPy arrays to NPZ."""
    data_dict = {"arr1": np.arange(5), "arr2": np.ones((2, 2))}
    out_file = tmp_path / "out.npz"
    save_data(data_dict, out_file)
    assert out_file.exists()
    loaded_data = np.load(out_file)
    assert "arr1" in loaded_data
    assert "arr2" in loaded_data
    assert_array_equal(loaded_data["arr1"], data_dict["arr1"])
    assert_array_equal(loaded_data["arr2"], data_dict["arr2"])
    loaded_data.close()

def test_save_audio_delegation(tmp_path: Path, mocker):
    """Test that save_data delegates to audio saver for audio formats."""
    dummy_audio = np.array([0.1, -0.1, 0.2], dtype=np.float64)
    dummy_sr = 8000
    audio_tuple: SaveInput = (dummy_audio, dummy_sr) # Explicit type hint
    out_file = tmp_path / "out.wav"
    subtype = 'PCM_16'

    # Mock the actual audio saving function from audio.io
    mock_save_audio = mocker.patch("sygnals.core.data_handler.save_audio_file")

    # Call save_data with the audio tuple
    save_data(audio_tuple, out_file, audio_subtype=subtype) # sr implicitly from tuple

    # Assert that the mock was called correctly
    mock_save_audio.assert_called_once_with(dummy_audio, dummy_sr, out_file, subtype=subtype)

def test_save_audio_delegation_with_sr_override(tmp_path: Path, mocker):
    """Test audio delegation when sr is explicitly provided, overriding tuple sr."""
    dummy_audio = np.array([0.1, -0.1, 0.2], dtype=np.float64)
    tuple_sr = 8000
    save_sr = 16000 # Override sr
    audio_tuple: SaveInput = (dummy_audio, tuple_sr)
    out_file = tmp_path / "out.flac"
    subtype = 'FLOAT'

    mock_save_audio = mocker.patch("sygnals.core.data_handler.save_audio_file")
    save_data(audio_tuple, out_file, sr=save_sr, audio_subtype=subtype)
    # Should be called with the overridden sr
    mock_save_audio.assert_called_once_with(dummy_audio, save_sr, out_file, subtype=subtype)


def test_save_unsupported_format(tmp_path: Path):
    """Test saving to an unsupported file format."""
    df = pd.DataFrame({"a": [1]})
    out_file = tmp_path / "out.xyz"
    with pytest.raises(ValueError, match="Unsupported output file format: '.xyz'"):
        save_data(df, out_file)

def test_save_type_format_mismatch(tmp_path: Path):
    """Test saving incompatible data type and format."""
    # Numpy array to JSON
    arr = np.arange(5)
    out_file_json = tmp_path / "out.json"
    with pytest.raises(ValueError, match="Cannot save single NumPy array directly to format '.json'"):
        save_data(arr, out_file_json)

    # Dict of arrays to CSV
    data_dict = {"arr1": np.arange(5)}
    out_file_csv = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="Cannot save dictionary of NumPy arrays to format '.csv'"):
        save_data(data_dict, out_file_csv)

    # Audio tuple to NPZ
    audio_tuple = (np.zeros(10), 8000)
    out_file_npz = tmp_path / "out.npz"
    with pytest.raises(ValueError, match="Cannot save audio data tuple to non-audio format '.npz'"):
        save_data(audio_tuple, out_file_npz)

def test_save_unsupported_data_type(tmp_path: Path):
    """Test saving an unsupported data type."""
    unsupported_data = [1, 2, 3] # Python list
    out_file = tmp_path / "out.csv"
    with pytest.raises(TypeError, match="Unsupported data type for saving: <class 'list'>"):
        save_data(unsupported_data, out_file)


# --- Test filter_data and run_sql_query ---

@pytest.fixture
def sample_filter_df() -> pd.DataFrame:
    """DataFrame for filtering/SQL tests."""
    return pd.DataFrame({
        "time": [0, 1, 2, 3, 4],
        "value": [10, 20, 5, 25, 15],
        "channel": ['A', 'B', 'A', 'B', 'A']
    })

def test_filter_data(sample_filter_df):
    """Test filtering a DataFrame using a query expression."""
    df = sample_filter_df
    result = filter_data(df, "value > 10 and channel == 'B'")
    expected_df = pd.DataFrame({"time": [1, 3], "value": [20, 25], "channel": ['B', 'B']}, index=[1, 3])
    pd.testing.assert_frame_equal(result, expected_df)

def test_filter_data_invalid_expr(sample_filter_df):
    """Test filtering with an invalid expression."""
    df = sample_filter_df
    with pytest.raises(ValueError, match="Data filtering expression failed"):
        filter_data(df, "invalid_column > 10") # Raises underlying pandas/numexpr error

def test_filter_data_wrong_type():
    """Test filtering with non-DataFrame input."""
    with pytest.raises(TypeError):
        filter_data([1, 2, 3], "value > 1") # type: ignore

def test_run_sql_query(sample_filter_df):
    """Test running an SQL query on a DataFrame."""
    df = sample_filter_df
    query = "SELECT time, value FROM df WHERE channel = 'A' ORDER BY value DESC"
    res = run_sql_query(df, query)
    expected_df = pd.DataFrame({"time": [4, 0, 2], "value": [15, 10, 5]})
    # Note: Index might not be preserved by pandasql, compare values
    pd.testing.assert_frame_equal(res, expected_df, check_index=False)

def test_run_sql_query_invalid_query(sample_filter_df):
    """Test running an invalid SQL query."""
    df = sample_filter_df
    query = "SELECT non_existent_col FROM df"
    # pandasql raises PandaSQLException which often wraps sqlite3.OperationalError
    # Match broadly for robustness
    with pytest.raises(RuntimeError, match="SQL query execution failed"):
        run_sql_query(df, query)

def test_run_sql_query_wrong_type():
    """Test SQL query with non-DataFrame input."""
    with pytest.raises(TypeError):
        run_sql_query([1, 2, 3], "SELECT * FROM df") # type: ignore
