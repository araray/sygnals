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
from typing import Union, Any, Dict, Tuple

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
ReadResult = Union[pd.DataFrame, Dict[str, NDArray[Any]], Tuple[NDArray[np.float64], int]]
SaveInput = Union[pd.DataFrame, NDArray[Any], Dict[str, NDArray[Any]], Tuple[NDArray[np.float64], int]]


def read_data(
    file_path: Union[str, Path],
    sr: Optional[int] = None # Target sample rate for audio
) -> ReadResult:
    """
    Loads data from a supported file format.

    - Reads CSV/JSON into Pandas DataFrame.
    - Reads NPZ into a dictionary of NumPy arrays.
    - Reads audio files (WAV, FLAC, OGG, MP3) into a tuple (data: ndarray, sr: int).
    - Reads from stdin (assumed CSV) if file_path is '-'.

    Args:
        file_path: Path to the input file or '-' for stdin.
        sr: Target sampling rate (only used when loading audio files).

    Returns:
        Loaded data in the appropriate format (DataFrame, dict, or tuple).

    Raises:
        ValueError: If the file format is not supported or arguments are invalid.
        FileNotFoundError: If the file does not exist (and is not '-').
        Exception: For underlying read errors.
    """
    file_path_str = str(file_path)
    logger.info(f"Reading data from: {file_path_str}")

    if file_path_str == "-":
        logger.info("Reading data from stdin (assuming CSV format).")
        try:
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
            raise ValueError(f"Unsupported file format: '{ext}'. Supported: {SUPPORTED_READ_FORMATS}")

        try:
            if ext in TABULAR_READ_FORMATS:
                if ext == ".csv":
                    return pd.read_csv(fpath)
                elif ext == ".json":
                    try:
                        return pd.read_json(fpath, orient='records')
                    except ValueError:
                        logger.warning("Failed to read JSON with orient='records', trying default.")
                        return pd.read_json(fpath) # Fallback

            elif ext in ARRAY_READ_FORMATS:
                if ext == ".npz":
                    # np.load returns a lazy NpzFile object, convert to dict
                    npz_data = np.load(fpath)
                    # Create a standard dictionary from the NpzFile object
                    data_dict = {key: npz_data[key] for key in npz_data.files}
                    npz_data.close() # Good practice to close the file handle
                    return data_dict

            elif ext in AUDIO_READ_FORMATS:
                # Delegate to audio loader
                logger.debug(f"Detected audio format '{ext}', delegating to audio loader.")
                return load_audio_file(fpath, sr=sr) # Pass sr argument

        except Exception as e:
            logger.error(f"Error reading file {fpath}: {e}")
            raise

    # Should not be reached
    raise RuntimeError("Unexpected error in read_data function.")


def save_data(
    data: SaveInput,
    output_path: Union[str, Path],
    sr: Optional[int] = None, # Required when saving audio tuple
    audio_subtype: Optional[str] = 'PCM_16' # Default audio subtype
):
    """
    Saves data to a specified file format.

    - Saves Pandas DataFrame to CSV/JSON.
    - Saves NumPy array or dict of arrays to NPZ.
    - Saves audio tuple (data: ndarray, sr: int) to WAV/FLAC/OGG.

    Args:
        data: The data to save (DataFrame, ndarray, dict, or audio tuple).
        output_path: Path to the output file.
        sr: Sampling rate (required only when saving audio data tuple).
        audio_subtype: Subtype for saving audio files (e.g., 'PCM_16', 'FLOAT').

    Raises:
        ValueError: If output format/data type mismatch, or missing required args (like sr for audio).
        Exception: For underlying write errors.
    """
    fpath = Path(output_path)
    ext = fpath.suffix.lower()
    logger.info(f"Saving data to: {fpath} (format: {ext})")

    if ext not in SUPPORTED_WRITE_FORMATS:
         raise ValueError(f"Unsupported output file format: '{ext}'. Supported: {SUPPORTED_WRITE_FORMATS}")

    try:
        # Ensure output directory exists
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            if ext in TABULAR_WRITE_FORMATS:
                if ext == ".csv":
                    data.to_csv(fpath, index=False)
                elif ext == ".json":
                    data.to_json(fpath, orient="records", indent=2)
            elif ext in ARRAY_WRITE_FORMATS and ext == ".npz":
                 logger.debug("Converting DataFrame to dict of arrays for NPZ saving.")
                 np.savez(fpath, **{col: data[col].values for col in data.columns})
            else:
                 raise ValueError(f"Cannot save DataFrame to format '{ext}'. Supported: {TABULAR_WRITE_FORMATS | {'.npz'}}")

        elif isinstance(data, np.ndarray):
             if ext in ARRAY_WRITE_FORMATS and ext == ".npz":
                 logger.debug("Saving single NumPy array to NPZ with key 'data'.")
                 np.savez(fpath, data=data)
             elif ext in TABULAR_WRITE_FORMATS and ext == ".csv":
                 logger.debug("Converting NumPy array to DataFrame for CSV saving.")
                 if data.ndim == 1:
                     pd.DataFrame(data, columns=['value']).to_csv(fpath, index=False)
                 elif data.ndim == 2:
                     pd.DataFrame(data).to_csv(fpath, index=False, header=False)
                 else:
                     raise ValueError("Cannot save NumPy array with >2 dimensions as CSV.")
             else:
                 raise ValueError(f"Cannot save NumPy array directly to format '{ext}'. Use NPZ or CSV.")

        elif isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()):
             if ext in ARRAY_WRITE_FORMATS and ext == ".npz":
                 logger.debug("Saving dictionary of NumPy arrays to NPZ.")
                 np.savez(fpath, **data)
             else:
                 raise ValueError(f"Cannot save dictionary of NumPy arrays to format '{ext}'. Use NPZ.")

        elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray) and isinstance(data[1], int):
             # Assume it's audio data (ndarray, sr)
             if ext in AUDIO_WRITE_FORMATS:
                 logger.debug(f"Detected audio data tuple, delegating to audio saver.")
                 audio_data, audio_sr = data
                 # Use the provided sr if available, otherwise use the one from the tuple
                 save_sr = sr if sr is not None else audio_sr
                 if save_sr is None: # Check again after potential override
                      raise ValueError("Sampling rate 'sr' must be provided when saving audio data tuple.")
                 save_audio_file(audio_data, save_sr, fpath, subtype=audio_subtype)
             else:
                  raise ValueError(f"Cannot save audio data tuple to non-audio format '{ext}'. Supported: {AUDIO_WRITE_FORMATS}")

        else:
            raise TypeError(f"Unsupported data type for saving: {type(data)}. "
                            f"Expected DataFrame, NumPy array, dict of arrays, or audio tuple (ndarray, sr).")

    except Exception as e:
        logger.error(f"Error saving file {fpath}: {e}")
        raise


# --- SQL and Filtering (Unchanged) ---

def run_sql_query(data: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Executes an SQL query on a Pandas DataFrame using pandasql.

    Args:
        data: The input DataFrame (will be referred to as 'df' in the query).
        query: The SQL query string.

    Returns:
        A DataFrame containing the query result.

    Raises:
        Exception: If the SQL query fails.
    """
    logger.info(f"Running SQL query: {query}")
    try:
        env = {'df': data}
        result = ps.sqldf(query, env)
        if not isinstance(result, pd.DataFrame):
             logger.warning(f"SQL query did not return a DataFrame (returned {type(result)}). Returning empty DataFrame.")
             return pd.DataFrame()
        return result
    except Exception as e:
        logger.error(f"SQL query failed: {e}\nQuery: {query}")
        raise


def filter_data(data: pd.DataFrame, filter_expr: str) -> pd.DataFrame:
    """
    Filters data using a Pandas query expression.

    Args:
        data: The input DataFrame.
        filter_expr: The Pandas query string (e.g., 'value > 10 and channel == "A"').

    Returns:
        A DataFrame containing the filtered rows.

    Raises:
        Exception: If the query expression is invalid.
    """
    logger.info(f"Filtering data with expression: {filter_expr}")
    try:
        return data.query(filter_expr)
    except Exception as e:
        logger.error(f"Data filtering failed: {e}\nExpression: {filter_expr}")
        raise
