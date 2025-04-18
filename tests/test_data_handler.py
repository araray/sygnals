# tests/test_data_handler.py

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import json
from pathlib import Path

# Import functions to test
from sygnals.core.data_handler import read_data, save_data, filter_data, run_sql_query
# Import audio I/O to mock or verify delegation (optional)
# from sygnals.core.audio.io import load_audio_file, save_audio_file

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

def test_read_json(tmp_path: Path):
    """Test reading a JSON file (records orientation)."""
    json_file = tmp_path / "test.json"
    json_data = [{"time": 0, "value": 1.0}, {"time": 1, "value": 2.0}]
    json_file.write_text(json.dumps(json_data))
    data = read_data(json_file)
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2
    assert list(data.columns) == ["time", "value"]
    assert data["value"].iloc[1] == 2.0

def test_read_npz(tmp_path: Path):
    """Test reading an NPZ file."""
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
    # Create a dummy WAV file
    wav_file = tmp_path / "dummy.wav"
    dummy_audio = np.array([0.1, 0.2, 0.1])
    dummy_sr = 16000
    # Use soundfile directly to create the dummy file
    import soundfile as sf
    sf.write(wav_file, dummy_audio, dummy_sr)

    # Mock the actual audio loading function
    mock_load_audio = mocker.patch("sygnals.core.data_handler.load_audio_file", return_value=(dummy_audio, dummy_sr))

    # Call read_data with the WAV file path
    result_data, result_sr = read_data(wav_file, sr=dummy_sr) # Pass sr

    # Assert that the mock was called correctly
    mock_load_audio.assert_called_once_with(wav_file, sr=dummy_sr)

    # Assert that the returned data matches the mock's return value
    assert_array_equal(result_data, dummy_audio)
    assert result_sr == dummy_sr

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
    # JSON float precision might differ slightly, convert types if needed
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
    dummy_audio = np.array([0.1, -0.1, 0.2])
    dummy_sr = 8000
    audio_tuple = (dummy_audio, dummy_sr)
    out_file = tmp_path / "out.wav"
    subtype = 'PCM_16'

    # Mock the actual audio saving function
    mock_save_audio = mocker.patch("sygnals.core.data_handler.save_audio_file")

    # Call save_data with the audio tuple
    save_data(audio_tuple, out_file, audio_subtype=subtype) # sr implicitly from tuple

    # Assert that the mock was called correctly
    mock_save_audio.assert_called_once_with(dummy_audio, dummy_sr, out_file, subtype=subtype)

def test_save_audio_delegation_with_sr_override(tmp_path: Path, mocker):
    """Test audio delegation when sr is explicitly provided."""
    dummy_audio = np.array([0.1, -0.1, 0.2])
    tuple_sr = 8000
    save_sr = 16000 # Override sr
    audio_tuple = (dummy_audio, tuple_sr)
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
    arr = np.arange(5)
    out_file = tmp_path / "out.json" # Cannot save numpy array directly to json via save_data
    with pytest.raises(ValueError, match="Cannot save NumPy array directly to format '.json'"):
        save_data(arr, out_file)

    audio_tuple = (np.zeros(10), 8000)
    out_file_csv = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="Cannot save audio data tuple to non-audio format '.csv'"):
        save_data(audio_tuple, out_file_csv)


# --- Test filter_data and run_sql_query (remain largely the same) ---

def test_filter_data():
    """Test filtering a DataFrame using a query expression."""
    df = pd.DataFrame({"time": [0, 1, 2, 3], "value": [10, 20, 5, 25], "channel": ['A', 'B', 'A', 'B']})
    result = filter_data(df, "value > 10 and channel == 'B'")
    assert len(result) == 2
    assert set(result["value"]) == {20, 25}
    assert all(result["channel"] == 'B')

def test_run_sql_query():
    """Test running an SQL query on a DataFrame."""
    df = pd.DataFrame({"time": [0, 1, 2], "value": [10, 20, 30], "label": ['x', 'y', 'x']})
    query = "SELECT time, value FROM df WHERE label = 'x' ORDER BY time DESC"
    res = run_sql_query(df, query)
    assert len(res) == 2
    assert list(res.columns) == ["time", "value"]
    assert list(res["time"]) == [2, 0] # Check order
    assert list(res["value"]) == [30, 10]
