# sygnals/core/data_handler.py

"""
Handles reading and writing data from/to various file formats (CSV, JSON, etc.)
using Pandas DataFrames. Also includes basic data manipulation utilities.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Union, Any, Dict

import pandas as pd
import pandasql as ps
from numpy.typing import NDArray
import numpy as np


logger = logging.getLogger(__name__)

# Supported formats (can be expanded)
SUPPORTED_READ_FORMATS = [".csv", ".json"] # Add .wav, .flac, .ogg handled by audio_handler
SUPPORTED_WRITE_FORMATS = [".csv", ".json", ".npz"] # Add .wav, .flac, .ogg handled by audio_handler


def read_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads data from a supported file format into a Pandas DataFrame.
    Handles CSV and JSON. Reads from stdin if file_path is '-'.

    Args:
        file_path: Path to the input file or '-' for stdin.

    Returns:
        A Pandas DataFrame containing the loaded data.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file does not exist (and is not '-').
        Exception: For other Pandas reading errors.
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
            # Check if it's an audio format handled elsewhere
            if ext not in [".wav", ".flac", ".ogg", ".mp3"]: # Add other audio formats if needed
                raise ValueError(f"Unsupported file format for generic data reading: '{ext}'. "
                                 f"Supported: {SUPPORTED_READ_FORMATS}. "
                                 f"Use audio commands for audio files.")
            else:
                # This function shouldn't handle audio directly, raise appropriate error or let caller handle
                raise ValueError(f"Attempted to read audio file '{ext}' with generic read_data. Use audio loading functions.")


        try:
            if ext == ".csv":
                return pd.read_csv(fpath)
            elif ext == ".json":
                # Try reading as records orientation first, common for list of dicts
                try:
                    return pd.read_json(fpath, orient='records')
                except ValueError:
                    logger.warning("Failed to read JSON with orient='records', trying default orientation.")
                    # Fallback to default read_json behavior
                    return pd.read_json(fpath)
            # Add readers for other formats like npz, parquet later if needed
            # elif ext == ".npz":
            #     data = np.load(fpath)
            #     # Convert npz structure to DataFrame (requires convention)
            #     # Example: assume dict of 1D arrays
            #     return pd.DataFrame({k: v for k, v in data.items() if v.ndim == 1})

        except Exception as e:
            logger.error(f"Error reading file {fpath}: {e}")
            raise
    # Should not be reached if logic is correct
    raise RuntimeError("Unexpected error in read_data function.")


def save_data(data: Union[pd.DataFrame, NDArray[Any], Dict[str, NDArray[Any]]], output_path: Union[str, Path]):
    """
    Saves data (Pandas DataFrame, NumPy array/dict) to a specified file format.
    Handles CSV, JSON, NPZ.

    Args:
        data: The data to save. Can be a DataFrame, a single NumPy array,
              or a dictionary of NumPy arrays (for NPZ).
        output_path: Path to the output file.

    Raises:
        ValueError: If the output format is not supported or data type is incompatible.
        Exception: For Pandas/NumPy writing errors.
    """
    fpath = Path(output_path)
    ext = fpath.suffix.lower()
    logger.info(f"Saving data to: {fpath} (format: {ext})")

    if ext not in SUPPORTED_WRITE_FORMATS:
         # Check if it's an audio format handled elsewhere
        if ext not in [".wav", ".flac", ".ogg"]: # Add other audio formats if needed
            raise ValueError(f"Unsupported output file format: '{ext}'. "
                             f"Supported: {SUPPORTED_WRITE_FORMATS}. "
                             f"Use audio saving functions for audio files.")
        else:
             raise ValueError(f"Attempted to save data as audio file '{ext}' with generic save_data. Use audio saving functions.")


    try:
        # Ensure output directory exists
        fpath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            if ext == ".csv":
                data.to_csv(fpath, index=False)
            elif ext == ".json":
                data.to_json(fpath, orient="records", indent=2)
            elif ext == ".npz":
                 # Convert DataFrame to dict of arrays for saving
                 np.savez(fpath, **{col: data[col].values for col in data.columns})
            else:
                 raise ValueError(f"Cannot save DataFrame to format '{ext}'.") # Should be caught earlier
        elif isinstance(data, np.ndarray):
             if ext == ".npz":
                 # Save single array with a default key 'data'
                 np.savez(fpath, data=data)
             elif ext == ".csv":
                 # Convert 1D/2D array to DataFrame for saving
                 if data.ndim == 1:
                     pd.DataFrame(data, columns=['value']).to_csv(fpath, index=False)
                 elif data.ndim == 2:
                     pd.DataFrame(data).to_csv(fpath, index=False, header=False) # No header for generic array
                 else:
                     raise ValueError("Cannot save NumPy array with >2 dimensions as CSV.")
             elif ext == ".json":
                  raise ValueError("Direct saving of NumPy array to JSON is not supported via save_data. Convert to DataFrame or list first.")
             else:
                 raise ValueError(f"Cannot save NumPy array to format '{ext}'.")
        elif isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()):
             if ext == ".npz":
                 np.savez(fpath, **data)
             else:
                 raise ValueError(f"Cannot save dictionary of NumPy arrays to format '{ext}'. Use NPZ.")
        else:
            raise TypeError(f"Unsupported data type for saving: {type(data)}. "
                            "Expected DataFrame, NumPy array, or dict of NumPy arrays.")

    except Exception as e:
        logger.error(f"Error saving file {fpath}: {e}")
        raise


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
        # pandasql uses a temporary SQLite database in memory
        env = {'df': data}
        result = ps.sqldf(query, env)
        # Ensure result is a DataFrame, even if query returns scalar/empty
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

# --- Normalization (Example - move to ml_utils later?) ---

def normalize_column(data: pd.DataFrame, column_name: str) -> pd.Series:
    """
    Normalizes a specific column in a DataFrame to the range [0, 1].

    Args:
        data: Input DataFrame.
        column_name: Name of the column to normalize.

    Returns:
        A Pandas Series with the normalized data.

    Raises:
        KeyError: If the column_name does not exist.
    """
    if column_name not in data.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")

    logger.debug(f"Normalizing column: {column_name}")
    col = data[column_name]
    min_val = col.min()
    max_val = col.max()
    range_val = max_val - min_val
    if range_val == 0:
        logger.warning(f"Column '{column_name}' has zero range. Returning original data.")
        # Avoid division by zero, return 0.5 or original data? Returning original for now.
        return col
    return (col - min_val) / range_val
