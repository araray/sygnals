# sygnals/core/data_handler.py

"""
Handles reading and writing data from/to various file formats.

Supports tabular data (CSV, JSON) via Pandas, numerical data (NPZ) via NumPy,
and delegates audio file handling (WAV, FLAC, OGG, MP3) to the
`sygnals.core.audio.io` module. Also provides utilities for filtering
and querying Pandas DataFrames.
"""

import os
import sys
import logging
from pathlib import Path
# Import necessary types
from typing import Union, Any, Dict, Tuple, Optional, Literal

import pandas as pd
import pandasql as ps
import numpy as np
from numpy.typing import NDArray

# Import audio I/O functions from the dedicated module
from .audio.io import load_audio as load_audio_file
from .audio.io import save_audio as save_audio_file
# Import supported audio extensions for consistency
from .audio.io import SUPPORTED_READ_EXTENSIONS as AUDIO_READ_EXTENSIONS
from .audio.io import SUPPORTED_WRITE_EXTENSIONS as AUDIO_WRITE_EXTENSIONS

logger = logging.getLogger(__name__)

# --- Supported Formats ---
TABULAR_READ_FORMATS = {".csv", ".json"}
ARRAY_READ_FORMATS = {".npz"}
SUPPORTED_READ_FORMATS = TABULAR_READ_FORMATS | ARRAY_READ_FORMATS | AUDIO_READ_EXTENSIONS

TABULAR_WRITE_FORMATS = {".csv", ".json"}
ARRAY_WRITE_FORMATS = {".npz"}
SUPPORTED_WRITE_FORMATS = TABULAR_WRITE_FORMATS | ARRAY_WRITE_FORMATS | AUDIO_WRITE_EXTENSIONS


# --- Type Aliases for Clarity ---
# ReadResult defines the possible types returned by read_data
ReadResult = Union[
    pd.DataFrame,                # For CSV, JSON
    Dict[str, NDArray[Any]],     # For NPZ
    Tuple[NDArray[np.float64], int] # For audio files: (data, sr)
]

# SaveInput defines the possible types accepted by save_data
SaveInput = Union[
    pd.DataFrame,
    NDArray[Any],                # Single NumPy array
    Dict[str, NDArray[Any]],     # Dictionary of NumPy arrays (for NPZ)
    Tuple[NDArray[np.float64], int] # Audio tuple: (data, sr)
]


# --- Core I/O Functions ---

def read_data(
    file_path: Union[str, Path],
    sr: Optional[int] = None # Target sample rate specifically for audio loading
) -> ReadResult:
    """
    Loads data from a supported file format (CSV, JSON, NPZ, WAV, FLAC, OGG, MP3).

    - Reads CSV/JSON into Pandas DataFrame.
    - Reads NPZ into a dictionary of NumPy arrays.
    - Reads audio files using `sygnals.core.audio.io.load_audio`, returning (data, sr).
    - Reads from stdin (assumed CSV) if file_path is '-'.

    Args:
        file_path: Path object or string path to the input file, or '-' for stdin.
        sr: Target sampling rate (only used when loading audio files). If None,
            the native sampling rate is used by the underlying audio loader.

    Returns:
        Loaded data in the appropriate format (DataFrame, dict of arrays, or audio tuple).
        See `ReadResult` type alias.

    Raises:
        ValueError: If the file format is not supported or arguments are invalid.
        FileNotFoundError: If the file does not exist (and is not '-').
        Exception: For underlying read errors from pandas, numpy, or audio loader.

    Example:
        >>> # Read CSV
        >>> df = read_data("data.csv")
        >>> # Read NPZ
        >>> arrays_dict = read_data("arrays.npz")
        >>> # Read WAV at native sample rate
        >>> audio_data, sample_rate = read_data("audio.wav")
        >>> # Read WAV and resample to 16kHz
        >>> audio_data_16k, sr_16k = read_data("audio.wav", sr=16000)
    """
    if isinstance(file_path, Path):
        file_path_str = str(file_path.resolve())
    else:
        file_path_str = str(file_path)

    logger.info(f"Reading data from: {file_path_str}")

    if file_path_str == "-":
        logger.info("Reading data from stdin (assuming CSV format).")
        try:
            # Read directly from stdin stream
            return pd.read_csv(sys.stdin)
        except Exception as e:
            logger.error(f"Error reading from stdin: {e}")
            raise ValueError("Failed to read CSV data from stdin.") from e
    else:
        fpath = Path(file_path_str)
        if not fpath.exists():
             raise FileNotFoundError(f"Input file not found: {fpath}")
        if not fpath.is_file():
             raise ValueError(f"Input path is not a file: {fpath}")

        ext = fpath.suffix.lower()
        if ext not in SUPPORTED_READ_FORMATS:
            raise ValueError(f"Unsupported file format: '{ext}'. Supported read formats: {SUPPORTED_READ_FORMATS}")

        try:
            if ext in TABULAR_READ_FORMATS:
                if ext == ".csv":
                    logger.debug(f"Reading CSV file: {fpath}")
                    return pd.read_csv(fpath)
                elif ext == ".json":
                    logger.debug(f"Reading JSON file: {fpath}")
                    try:
                        # Attempt standard JSON read first (often list of records)
                        return pd.read_json(fpath, orient='records')
                    except ValueError:
                        logger.warning(f"Failed to read JSON with orient='records' for {fpath.name}, trying default line-delimited.")
                        # Fallback for line-delimited JSON
                        return pd.read_json(fpath, lines=True)

            elif ext in ARRAY_READ_FORMATS:
                if ext == ".npz":
                    logger.debug(f"Loading NPZ file: {fpath}")
                    # Use try-finally to ensure file handle is closed
                    npz_file = None
                    try:
                        npz_file = np.load(fpath)
                        # Create a standard dictionary from the NpzFile object keys
                        data_dict = {key: npz_file[key] for key in npz_file.files}
                        logger.debug(f"NPZ file loaded with keys: {list(data_dict.keys())}")
                        return data_dict
                    finally:
                        if npz_file is not None:
                            npz_file.close()

            elif ext in AUDIO_READ_EXTENSIONS:
                # Delegate to audio loader in audio.io
                logger.debug(f"Detected audio format '{ext}', delegating to audio loader (target sr={sr}).")
                # Pass the target sampling rate `sr` to the audio loader
                return load_audio_file(fpath, sr=sr) # Returns tuple (data, sr)

        except Exception as e:
            logger.error(f"Error reading file {fpath}: {e}")
            # Re-raise as a more specific error if possible, or generic exception
            raise RuntimeError(f"Failed to read data from {fpath}.") from e

    # This line should ideally not be reached if logic is correct
    raise RuntimeError("Unexpected error occurred in read_data function.")


def save_data(
    data: SaveInput,
    output_path: Union[str, Path],
    sr: Optional[int] = None, # Required when saving audio tuple if sr differs from tuple
    audio_subtype: Optional[str] = 'PCM_16' # Default audio subtype
):
    """
    Saves data to a specified file format (CSV, JSON, NPZ, WAV, FLAC, OGG).

    - Saves Pandas DataFrame to CSV/JSON (or NPZ by converting columns).
    - Saves single NumPy array to NPZ (key 'data') or CSV (single 'value' column).
    - Saves dict of NumPy arrays to NPZ.
    - Saves audio tuple (data: ndarray, sr: int) to WAV/FLAC/OGG using
      `sygnals.core.audio.io.save_audio`.

    Args:
        data: The data to save (see `SaveInput` type alias).
        output_path: Path object or string path for the output file.
                     The extension determines the save format.
        sr: Sampling rate (Hz). Required only when saving audio data passed as a
            tuple, especially if the desired save SR differs from the SR in the tuple.
            If `data` is an audio tuple and `sr` is None, the SR from the tuple is used.
        audio_subtype: Subtype for saving audio files (e.g., 'PCM_16', 'FLOAT').
                 Passed to `sygnals.core.audio.io.save_audio`. (Default: 'PCM_16')

    Raises:
        ValueError: If output format/data type mismatch, or missing required args (like sr for audio tuple).
        TypeError: If the input `data` type is not supported for saving to the specified format.
        Exception: For underlying write errors from pandas, numpy, or audio saver.

    Example:
        >>> df = pd.DataFrame({'val': [1, 2]})
        >>> save_data(df, "output.csv")
        >>> arr_dict = {'x': np.arange(3), 'y': np.ones(3)}
        >>> save_data(arr_dict, "output.npz")
        >>> audio_data, audio_sr = read_data("input.wav")
        >>> save_data((audio_data, audio_sr), "output.flac") # Save audio tuple
        >>> save_data((audio_data, audio_sr), "output_16k.wav", sr=16000) # Save and resample (if audio saver supports it) - NOTE: save_audio doesn't resample
    """
    fpath = Path(output_path).resolve()
    ext = fpath.suffix.lower()
    logger.info(f"Saving data to: {fpath} (format: {ext})")

    if ext not in SUPPORTED_WRITE_FORMATS:
         raise ValueError(f"Unsupported output file format: '{ext}'. Supported write formats: {SUPPORTED_WRITE_FORMATS}")

    try:
        # Ensure output directory exists before writing
        fpath.parent.mkdir(parents=True, exist_ok=True)

        # --- Handle different input data types ---
        if isinstance(data, pd.DataFrame):
            if ext in TABULAR_WRITE_FORMATS:
                if ext == ".csv":
                    data.to_csv(fpath, index=False)
                elif ext == ".json":
                    # Use records orientation for compatibility with read_json
                    data.to_json(fpath, orient="records", indent=2)
            elif ext == ".npz":
                 # Convert DataFrame columns to a dictionary of arrays for NPZ
                 logger.debug("Converting DataFrame to dict of arrays for NPZ saving.")
                 np.savez(fpath, **{col: data[col].values for col in data.columns})
            else:
                 raise ValueError(f"Cannot save DataFrame to format '{ext}'. Supported: {TABULAR_WRITE_FORMATS | {'.npz'}}")

        elif isinstance(data, np.ndarray):
             # Handle saving a single NumPy array
             if ext == ".npz":
                 # Save with a default key 'data'
                 logger.debug("Saving single NumPy array to NPZ with key 'data'.")
                 np.savez(fpath, data=data)
             elif ext == ".csv":
                 # Convert array to DataFrame for CSV saving
                 logger.debug("Converting NumPy array to DataFrame for CSV saving.")
                 if data.ndim == 1:
                     pd.DataFrame(data, columns=['value']).to_csv(fpath, index=False)
                 elif data.ndim == 2:
                     # Save 2D array without header/index by default
                     pd.DataFrame(data).to_csv(fpath, index=False, header=False)
                 else:
                     raise ValueError("Cannot save NumPy array with >2 dimensions as CSV.")
             else:
                 raise ValueError(f"Cannot save single NumPy array directly to format '{ext}'. Use NPZ or CSV.")

        elif isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()):
             # Handle saving a dictionary of NumPy arrays
             if ext == ".npz":
                 logger.debug(f"Saving dictionary of {len(data)} NumPy arrays to NPZ.")
                 np.savez(fpath, **data)
             else:
                 raise ValueError(f"Cannot save dictionary of NumPy arrays to format '{ext}'. Use NPZ.")

        elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray) and isinstance(data[1], int):
             # Assume it's audio data tuple: (ndarray, int_sr)
             if ext in AUDIO_WRITE_EXTENSIONS:
                 logger.debug(f"Detected audio data tuple, delegating to audio saver.")
                 audio_data, audio_sr_from_tuple = data
                 # Use the explicitly provided `sr` if available, otherwise use the one from the tuple
                 save_sr = sr if sr is not None else audio_sr_from_tuple
                 # Note: save_audio_file currently does NOT resample. The `sr` parameter here
                 # primarily dictates the metadata written to the file header.
                 if sr is not None and sr != audio_sr_from_tuple:
                      logger.warning(f"Explicit 'sr' ({sr} Hz) provided for saving differs from 'sr' in audio tuple "
                                     f"({audio_sr_from_tuple} Hz). The saved file header will use {sr} Hz, "
                                     f"but the audio data itself is NOT resampled by save_data/save_audio_file.")

                 # Delegate to the audio saving function
                 save_audio_file(audio_data, save_sr, fpath, subtype=audio_subtype)
             else:
                  raise ValueError(f"Cannot save audio data tuple to non-audio format '{ext}'. Supported audio formats: {AUDIO_WRITE_EXTENSIONS}")

        else:
            # Raise error for unsupported input data types
            raise TypeError(f"Unsupported data type for saving: {type(data)}. See SaveInput type alias.")

        logger.info(f"Data successfully saved to {fpath}")

    except Exception as e:
        logger.error(f"Error saving file {fpath}: {e}")
        # Re-raise exception for upstream handling
        raise RuntimeError(f"Failed to save data to {fpath}.") from e


# --- SQL and Filtering ---

def run_sql_query(data: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Executes an SQL query on a Pandas DataFrame using pandasql.

    Allows SQL-like querying of DataFrame content. The DataFrame is accessible
    as 'df' within the SQL query.

    Args:
        data: The input DataFrame (will be referred to as 'df' in the query).
        query: The SQL query string (e.g., "SELECT colA, SUM(colB) FROM df WHERE colC > 10 GROUP BY colA").

    Returns:
        A DataFrame containing the query result. Returns an empty DataFrame if the
        query result is None or not tabular.

    Raises:
        TypeError: If input `data` is not a Pandas DataFrame.
        Exception: If the SQL query fails to execute via pandasql (e.g., syntax error, database error).
    """
    logger.info(f"Running SQL query on DataFrame: {query}")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' for run_sql_query must be a Pandas DataFrame.")
    try:
        # Environment for pandasql to find the DataFrame
        env = {'df': data}
        result = ps.sqldf(query, env)
        # Ensure the result is a DataFrame, handle cases where sqldf might return other types
        if result is None:
             logger.warning("SQL query returned None. Returning empty DataFrame.")
             return pd.DataFrame()
        if not isinstance(result, pd.DataFrame):
             logger.warning(f"SQL query did not return a DataFrame (returned {type(result)}). Returning empty DataFrame.")
             return pd.DataFrame()
        logger.debug(f"SQL query returned DataFrame with shape {result.shape}")
        return result
    except Exception as e:
        logger.error(f"SQL query failed: {e}\nQuery: {query}")
        # Re-raise the exception (could be sqldf error or underlying sqlite error)
        raise RuntimeError("SQL query execution failed.") from e


def filter_data(data: pd.DataFrame, filter_expr: str) -> pd.DataFrame:
    """
    Filters data using a Pandas query expression.

    Provides a way to select rows based on column values using Pandas' built-in
    query syntax. See `pandas.DataFrame.query` documentation for expression syntax.

    Args:
        data: The input DataFrame.
        filter_expr: The Pandas query string (e.g., 'value > 10 and channel == "A"').

    Returns:
        A DataFrame containing the filtered rows.

    Raises:
        TypeError: If input `data` is not a Pandas DataFrame.
        Exception: If the query expression is invalid for the DataFrame (e.g., syntax error, invalid column name).
    """
    logger.info(f"Filtering DataFrame with expression: {filter_expr}")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' for filter_data must be a Pandas DataFrame.")
    try:
        filtered_df = data.query(filter_expr)
        logger.debug(f"Filtering resulted in DataFrame with shape {filtered_df.shape}")
        return filtered_df
    except Exception as e:
        logger.error(f"Data filtering failed: {e}\nExpression: {filter_expr}")
        # Re-raise the exception (likely an error in the filter_expr)
        raise ValueError("Data filtering expression failed.") from e
