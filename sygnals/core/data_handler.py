# sygnals/core/data_handler.py

"""
Handles reading and writing data from/to various file formats.
Supports tabular data (CSV, JSON) via Pandas and numerical data (NPZ).
Delegates audio file handling to sygnals.core.audio.io.
"""

import os
import sys
import logging
from pathlib import Path
# Import necessary types
from typing import Union, Any, Dict, Tuple, Optional

import pandas as pd
import pandasql as ps
import numpy as np
from numpy.typing import NDArray

# Import audio I/O functions
from .audio.io import load_audio as load_audio_file
from .audio.io import save_audio as save_audio_file

logger = logging.getLogger(__name__)

# Define supported formats explicitly
TABULAR_READ_FORMATS = {".csv", ".json"}
ARRAY_READ_FORMATS = {".npz"}
AUDIO_READ_FORMATS = {".wav", ".flac", ".ogg", ".mp3"} # Add more as supported by librosa/soundfile
SUPPORTED_READ_FORMATS = TABULAR_READ_FORMATS | ARRAY_READ_FORMATS | AUDIO_READ_FORMATS

TABULAR_WRITE_FORMATS = {".csv", ".json"}
ARRAY_WRITE_FORMATS = {".npz"}
AUDIO_WRITE_FORMATS = {".wav", ".flac", ".ogg"} # Add more as supported by soundfile
SUPPORTED_WRITE_FORMATS = TABULAR_WRITE_FORMATS | ARRAY_WRITE_FORMATS | AUDIO_WRITE_FORMATS


# Define return types for clarity
# ReadResult can be:
# - pd.DataFrame (for CSV, JSON)
# - Dict[str, NDArray[Any]] (for NPZ)
# - Tuple[NDArray[np.float64], int] (for audio files: data, sr)
ReadResult = Union[pd.DataFrame, Dict[str, NDArray[Any]], Tuple[NDArray[np.float64], int]]

# SaveInput can be:
# - pd.DataFrame
# - NDArray[Any] (single array, saved to NPZ with key 'data' or converted for CSV)
# - Dict[str, NDArray[Any]] (saved to NPZ)
# - Tuple[NDArray[np.float64], int] (audio data, sr - requires sr arg for save_data)
SaveInput = Union[pd.DataFrame, NDArray[Any], Dict[str, NDArray[Any]], Tuple[NDArray[np.float64], int]]


def read_data(
    file_path: Union[str, Path],
    sr: Optional[int] = None # Target sample rate for audio
) -> ReadResult:
    """
    Loads data from a supported file format.

    - Reads CSV/JSON into Pandas DataFrame.
    - Reads NPZ into a dictionary of NumPy arrays.
    - Reads audio files (WAV, FLAC, OGG, MP3) into a tuple (data: ndarray, sr: int),
      delegating to `sygnals.core.audio.io.load_audio`.
    - Reads from stdin (assumed CSV) if file_path is '-'.

    Args:
        file_path: Path to the input file or '-' for stdin.
        sr: Target sampling rate (only used when loading audio files). If None,
            the native sampling rate is used unless resampling occurs in the
            underlying audio loading function.

    Returns:
        Loaded data in the appropriate format (DataFrame, dict of arrays, or audio tuple).

    Raises:
        ValueError: If the file format is not supported or arguments are invalid.
        FileNotFoundError: If the file does not exist (and is not '-').
        Exception: For underlying read errors from pandas, numpy, or audio loader.
    """
    file_path_str = str(file_path)
    logger.info(f"Reading data from: {file_path_str}")

    if file_path_str == "-":
        logger.info("Reading data from stdin (assuming CSV format).")
        try:
            # Read directly from stdin stream
            return pd.read_csv(sys.stdin)
        except Exception as e:
            logger.error(f"Error reading from stdin: {e}")
            raise
    else:
        fpath = Path(file_path)
        if not fpath.exists():
             raise FileNotFoundError(f"Input file not found: {fpath}")

        ext = fpath.suffix.lower()
        if ext not in SUPPORTED_READ_FORMATS:
            raise ValueError(f"Unsupported file format: '{ext}'. Supported read formats: {SUPPORTED_READ_FORMATS}")

        try:
            if ext in TABULAR_READ_FORMATS:
                if ext == ".csv":
                    return pd.read_csv(fpath)
                elif ext == ".json":
                    try:
                        # Attempt standard JSON read first (often list of records)
                        return pd.read_json(fpath, orient='records')
                    except ValueError:
                        logger.warning("Failed to read JSON with orient='records', trying default line-delimited.")
                        # Fallback for line-delimited JSON
                        return pd.read_json(fpath, lines=True)

            elif ext in ARRAY_READ_FORMATS:
                if ext == ".npz":
                    # np.load returns a lazy NpzFile object, convert to dict for easier handling
                    logger.debug(f"Loading NPZ file: {fpath}")
                    npz_data = np.load(fpath)
                    # Create a standard dictionary from the NpzFile object keys
                    data_dict = {key: npz_data[key] for key in npz_data.files}
                    npz_data.close() # Good practice to close the file handle
                    logger.debug(f"NPZ file loaded with keys: {list(data_dict.keys())}")
                    return data_dict

            elif ext in AUDIO_READ_FORMATS:
                # Delegate to audio loader in audio.io
                logger.debug(f"Detected audio format '{ext}', delegating to audio loader.")
                # Pass the target sampling rate `sr` to the audio loader
                return load_audio_file(fpath, sr=sr)

        except Exception as e:
            logger.error(f"Error reading file {fpath}: {e}")
            raise

    # This line should ideally not be reached if logic is correct
    raise RuntimeError("Unexpected error occurred in read_data function.")


def save_data(
    data: SaveInput,
    output_path: Union[str, Path],
    sr: Optional[int] = None, # Required when saving audio tuple
    audio_subtype: Optional[str] = 'PCM_16' # Default audio subtype
):
    """
    Saves data to a specified file format.

    - Saves Pandas DataFrame to CSV/JSON (or NPZ by converting columns).
    - Saves NumPy array or dict of arrays to NPZ (or single array to CSV by converting).
    - Saves audio tuple (data: ndarray, sr: int) to WAV/FLAC/OGG, delegating to
      `sygnals.core.audio.io.save_audio`.

    Args:
        data: The data to save (DataFrame, ndarray, dict of arrays, or audio tuple).
        output_path: Path to the output file. The extension determines the format.
        sr: Sampling rate (required only when saving audio data passed as a tuple).
            If `data` is an audio tuple, this `sr` overrides the one in the tuple.
        audio_subtype: Subtype for saving audio files (e.g., 'PCM_16', 'FLOAT').
                 Passed to `sygnals.core.audio.io.save_audio`.

    Raises:
        ValueError: If output format/data type mismatch, or missing required args (like sr for audio tuple).
        TypeError: If the input `data` type is not supported for saving.
        Exception: For underlying write errors from pandas, numpy, or audio saver.
    """
    fpath = Path(output_path)
    ext = fpath.suffix.lower()
    logger.info(f"Saving data to: {fpath} (format: {ext})")

    if ext not in SUPPORTED_WRITE_FORMATS:
         raise ValueError(f"Unsupported output file format: '{ext}'. Supported write formats: {SUPPORTED_WRITE_FORMATS}")

    try:
        # Ensure output directory exists before writing
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            if ext in TABULAR_WRITE_FORMATS:
                if ext == ".csv":
                    data.to_csv(fpath, index=False)
                elif ext == ".json":
                    # Use records orientation for compatibility with read_json
                    data.to_json(fpath, orient="records", indent=2)
            elif ext in ARRAY_WRITE_FORMATS and ext == ".npz":
                 # Convert DataFrame columns to a dictionary of arrays for NPZ
                 logger.debug("Converting DataFrame to dict of arrays for NPZ saving.")
                 np.savez(fpath, **{col: data[col].values for col in data.columns})
            else:
                 raise ValueError(f"Cannot save DataFrame to format '{ext}'. Supported: {TABULAR_WRITE_FORMATS | {'.npz'}}")

        elif isinstance(data, np.ndarray):
             # Handle saving a single NumPy array
             if ext in ARRAY_WRITE_FORMATS and ext == ".npz":
                 # Save with a default key 'data'
                 logger.debug("Saving single NumPy array to NPZ with key 'data'.")
                 np.savez(fpath, data=data)
             elif ext in TABULAR_WRITE_FORMATS and ext == ".csv":
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
                 raise ValueError(f"Cannot save NumPy array directly to format '{ext}'. Use NPZ or CSV.")

        elif isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()):
             # Handle saving a dictionary of NumPy arrays
             if ext in ARRAY_WRITE_FORMATS and ext == ".npz":
                 logger.debug("Saving dictionary of NumPy arrays to NPZ.")
                 np.savez(fpath, **data)
             else:
                 raise ValueError(f"Cannot save dictionary of NumPy arrays to format '{ext}'. Use NPZ.")

        elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray) and isinstance(data[1], int):
             # Assume it's audio data tuple: (ndarray, int_sr)
             if ext in AUDIO_WRITE_FORMATS:
                 logger.debug(f"Detected audio data tuple, delegating to audio saver.")
                 audio_data, audio_sr_from_tuple = data
                 # Use the explicitly provided `sr` if available, otherwise use the one from the tuple
                 save_sr = sr if sr is not None else audio_sr_from_tuple
                 if save_sr is None: # Check again after potential override logic
                      raise ValueError("Sampling rate 'sr' must be provided via argument or tuple when saving audio data.")
                 # Delegate to the audio saving function
                 save_audio_file(audio_data, save_sr, fpath, subtype=audio_subtype)
             else:
                  raise ValueError(f"Cannot save audio data tuple to non-audio format '{ext}'. Supported audio formats: {AUDIO_WRITE_FORMATS}")

        else:
            # Raise error for unsupported input data types
            raise TypeError(f"Unsupported data type for saving: {type(data)}. "
                            f"Expected DataFrame, NumPy array, dict of arrays, or audio tuple (ndarray, sr).")

        logger.info(f"Data successfully saved to {fpath}")

    except Exception as e:
        logger.error(f"Error saving file {fpath}: {e}")
        raise


# --- SQL and Filtering (Keep as is for now, may refactor later) ---

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
        query result is not tabular.

    Raises:
        Exception: If the SQL query fails to execute via pandasql.
    """
    logger.info(f"Running SQL query on DataFrame: {query}")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' for run_sql_query must be a Pandas DataFrame.")
    try:
        # Environment for pandasql to find the DataFrame
        env = {'df': data}
        result = ps.sqldf(query, env)
        # Ensure the result is a DataFrame, handle cases where sqldf might return other types
        if not isinstance(result, pd.DataFrame):
             logger.warning(f"SQL query did not return a DataFrame (returned {type(result)}). Returning empty DataFrame.")
             return pd.DataFrame()
        logger.debug(f"SQL query returned DataFrame with shape {result.shape}")
        return result
    except Exception as e:
        logger.error(f"SQL query failed: {e}\nQuery: {query}")
        raise


def filter_data(data: pd.DataFrame, filter_expr: str) -> pd.DataFrame:
    """
    Filters data using a Pandas query expression.

    Provides a way to select rows based on column values using Pandas' built-in
    query syntax.

    Args:
        data: The input DataFrame.
        filter_expr: The Pandas query string (e.g., 'value > 10 and channel == "A"').

    Returns:
        A DataFrame containing the filtered rows.

    Raises:
        Exception: If the query expression is invalid for the DataFrame.
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
        raise
