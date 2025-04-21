# sygnals/core/data_handler.py

"""
Handles reading and writing data from/to various file formats.

Supports tabular data (CSV, JSON) via Pandas, numerical data (NPZ) via NumPy,
and delegates audio file handling (WAV, FLAC, OGG, MP3) to the
`sygnals.core.audio.io` module. Can also utilize plugin-provided data handlers.
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

# Import PluginRegistry type hint (avoid circular import by using string)
if sys.version_info >= (3, 8):
    from typing import TYPE_CHECKING
else:
    from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from ..plugins.api import PluginRegistry # Use string import for type checking

logger = logging.getLogger(__name__)

# --- Supported Formats (Core) ---
# These represent formats handled directly by this module or the core audio module.
# Plugin-supported formats are handled dynamically via the registry.
CORE_TABULAR_READ_FORMATS = {".csv", ".json"}
CORE_ARRAY_READ_FORMATS = {".npz"}
CORE_AUDIO_READ_FORMATS = AUDIO_READ_EXTENSIONS
SUPPORTED_READ_FORMATS = CORE_TABULAR_READ_FORMATS | CORE_ARRAY_READ_FORMATS | CORE_AUDIO_READ_FORMATS

CORE_TABULAR_WRITE_FORMATS = {".csv", ".json"}
CORE_ARRAY_WRITE_FORMATS = {".npz"}
CORE_AUDIO_WRITE_FORMATS = AUDIO_WRITE_EXTENSIONS
SUPPORTED_WRITE_FORMATS = CORE_TABULAR_WRITE_FORMATS | CORE_ARRAY_WRITE_FORMATS | CORE_AUDIO_WRITE_FORMATS


# --- Type Aliases for Clarity ---
# ReadResult defines the possible types returned by read_data
ReadResult = Union[
    pd.DataFrame,                # For CSV, JSON, Parquet (via plugin)
    Dict[str, NDArray[Any]],     # For NPZ, HDF5 (via plugin)
    Tuple[NDArray[np.float64], int] # For audio files: (data, sr)
]

# SaveInput defines the possible types accepted by save_data
SaveInput = Union[
    pd.DataFrame,
    NDArray[Any],                # Single NumPy array
    Dict[str, NDArray[Any]],     # Dictionary of NumPy arrays (for NPZ, HDF5)
    Tuple[NDArray[np.float64], int] # Audio tuple: (data, sr)
]


# --- Core I/O Functions ---

def read_data(
    file_path: Union[str, Path],
    sr: Optional[int] = None, # Target sample rate specifically for audio loading
    registry: Optional['PluginRegistry'] = None # Optional plugin registry
) -> ReadResult:
    """
    Loads data from a supported file format (CSV, JSON, NPZ, WAV, FLAC, OGG, MP3,
    or plugin-supported formats like Parquet, HDF5).

    Checks for plugin readers based on file extension before falling back to core handlers.

    Args:
        file_path: Path object or string path to the input file, or '-' for stdin.
        sr: Target sampling rate (only used when loading audio files). If None,
            the native sampling rate is used by the underlying audio loader.
        registry: Optional PluginRegistry instance to check for custom readers.

    Returns:
        Loaded data in the appropriate format (DataFrame, dict of arrays, or audio tuple).
        See `ReadResult` type alias.

    Raises:
        ValueError: If the file format is not supported (by core or plugins) or arguments are invalid.
        FileNotFoundError: If the file does not exist (and is not '-').
        Exception: For underlying read errors from pandas, numpy, audio loader, or plugin reader.
    """
    if isinstance(file_path, Path):
        file_path_str = str(file_path.resolve())
    else:
        file_path_str = str(file_path)

    logger.info(f"Reading data from: {file_path_str}")

    if file_path_str == "-":
        # Handle stdin (assume CSV for now, plugins typically won't handle stdin)
        logger.info("Reading data from stdin (assuming CSV format).")
        try:
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

        # 1. Check for Plugin Reader
        if registry:
            plugin_reader = registry.get_reader(ext)
            if plugin_reader:
                logger.info(f"Using plugin-provided reader for extension '{ext}'.")
                try:
                    # Pass relevant kwargs if needed, currently just path
                    # TODO: Consider how to pass reader-specific args from CLI/config
                    return plugin_reader(path=fpath)
                except Exception as e:
                    logger.error(f"Plugin reader for '{ext}' failed: {e}", exc_info=True)
                    raise RuntimeError(f"Plugin reader failed for {fpath}.") from e

        # 2. Fallback to Core Handlers
        if ext not in SUPPORTED_READ_FORMATS:
            # Check core formats only if no plugin handled it
            supported_list = SUPPORTED_READ_FORMATS
            if registry:
                 supported_list |= set(registry.list_readers()) # Add plugin formats to error message
            raise ValueError(f"Unsupported file format: '{ext}'. Supported formats: {supported_list}")

        try:
            if ext in CORE_TABULAR_READ_FORMATS:
                if ext == ".csv":
                    logger.debug(f"Reading CSV file: {fpath}")
                    return pd.read_csv(fpath)
                elif ext == ".json":
                    logger.debug(f"Reading JSON file: {fpath}")
                    try:
                        return pd.read_json(fpath, orient='records')
                    except ValueError:
                        logger.warning(f"Failed to read JSON with orient='records' for {fpath.name}, trying default line-delimited.")
                        return pd.read_json(fpath, lines=True)

            elif ext in CORE_ARRAY_READ_FORMATS:
                if ext == ".npz":
                    logger.debug(f"Loading NPZ file: {fpath}")
                    npz_file = None
                    try:
                        npz_file = np.load(fpath)
                        data_dict = {key: npz_file[key] for key in npz_file.files}
                        logger.debug(f"NPZ file loaded with keys: {list(data_dict.keys())}")
                        return data_dict
                    finally:
                        if npz_file is not None:
                            npz_file.close()

            elif ext in CORE_AUDIO_READ_FORMATS:
                logger.debug(f"Detected core audio format '{ext}', delegating to audio loader (target sr={sr}).")
                return load_audio_file(fpath, sr=sr) # Returns tuple (data, sr)

        except Exception as e:
            logger.error(f"Core handler failed reading file {fpath}: {e}")
            raise RuntimeError(f"Failed to read data from {fpath}.") from e

    # Should not be reached
    raise RuntimeError("Unexpected error occurred in read_data function.")


def save_data(
    data: SaveInput,
    output_path: Union[str, Path],
    sr: Optional[int] = None, # Required when saving audio tuple if sr differs from tuple
    audio_subtype: Optional[str] = 'PCM_16', # Default audio subtype
    registry: Optional['PluginRegistry'] = None # Optional plugin registry
):
    """
    Saves data to a specified file format (CSV, JSON, NPZ, WAV, FLAC, OGG,
    or plugin-supported formats like Parquet, HDF5).

    Checks for plugin writers based on file extension before falling back to core handlers.

    Args:
        data: The data to save (see `SaveInput` type alias).
        output_path: Path object or string path for the output file.
                     The extension determines the save format.
        sr: Sampling rate (Hz). Relevant only when saving audio data (passed as kwarg to plugin writers).
        audio_subtype: Subtype for saving audio files (e.g., 'PCM_16', 'FLOAT').
                 Passed to core audio saver and as kwarg to plugin writers. (Default: 'PCM_16')
        registry: Optional PluginRegistry instance to check for custom writers.

    Raises:
        ValueError: If output format/data type mismatch, or missing required args.
        TypeError: If the input `data` type is not supported for saving to the specified format.
        Exception: For underlying write errors from pandas, numpy, audio saver, or plugin writer.
    """
    fpath = Path(output_path).resolve()
    ext = fpath.suffix.lower()
    logger.info(f"Saving data to: {fpath} (format: {ext})")

    # Ensure output directory exists before writing
    try:
        fpath.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {fpath.parent}: {e}")
        raise

    # 1. Check for Plugin Writer
    if registry:
        plugin_writer = registry.get_writer(ext)
        if plugin_writer:
            logger.info(f"Using plugin-provided writer for extension '{ext}'.")
            try:
                # Pass sr and subtype as kwargs for potential use by plugin writer
                plugin_writer(data, fpath, sr=sr, audio_subtype=audio_subtype)
                logger.info(f"Data successfully saved by plugin writer to {fpath}")
                return # Exit after successful plugin write
            except Exception as e:
                logger.error(f"Plugin writer for '{ext}' failed: {e}", exc_info=True)
                raise RuntimeError(f"Plugin writer failed for {fpath}.") from e

    # 2. Fallback to Core Handlers
    if ext not in SUPPORTED_WRITE_FORMATS:
        # Check core formats only if no plugin handled it
        supported_list = SUPPORTED_WRITE_FORMATS
        if registry:
            supported_list |= set(registry.list_writers())
        raise ValueError(f"Unsupported output file format: '{ext}'. Supported formats: {supported_list}")

    try:
        # --- Handle different input data types for Core Handlers ---
        if isinstance(data, pd.DataFrame):
            if ext in CORE_TABULAR_WRITE_FORMATS:
                if ext == ".csv":
                    data.to_csv(fpath, index=False)
                elif ext == ".json":
                    data.to_json(fpath, orient="records", indent=2)
            elif ext == ".npz":
                 logger.debug("Converting DataFrame to dict of arrays for NPZ saving.")
                 np.savez(fpath, **{col: data[col].values for col in data.columns})
            else:
                 raise ValueError(f"Cannot save DataFrame to core format '{ext}'. Supported: {CORE_TABULAR_WRITE_FORMATS | {'.npz'}}")

        elif isinstance(data, np.ndarray):
             if ext == ".npz":
                 logger.debug("Saving single NumPy array to NPZ with key 'data'.")
                 np.savez(fpath, data=data)
             elif ext == ".csv":
                 logger.debug("Converting NumPy array to DataFrame for CSV saving.")
                 if data.ndim == 1:
                     pd.DataFrame(data, columns=['value']).to_csv(fpath, index=False)
                 elif data.ndim == 2:
                     pd.DataFrame(data).to_csv(fpath, index=False, header=False)
                 else:
                     raise ValueError("Cannot save NumPy array with >2 dimensions as CSV.")
             else:
                 raise ValueError(f"Cannot save single NumPy array directly to core format '{ext}'. Use NPZ or CSV.")

        elif isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()):
             if ext == ".npz":
                 logger.debug(f"Saving dictionary of {len(data)} NumPy arrays to NPZ.")
                 np.savez(fpath, **data)
             else:
                 raise ValueError(f"Cannot save dictionary of NumPy arrays to core format '{ext}'. Use NPZ.")

        elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray) and isinstance(data[1], int):
             # Assume audio tuple (data, sr_tuple)
             if ext in CORE_AUDIO_WRITE_FORMATS:
                 logger.debug(f"Detected audio data tuple, delegating to core audio saver.")
                 audio_data, audio_sr_from_tuple = data
                 save_sr = sr if sr is not None else audio_sr_from_tuple
                 if sr is not None and sr != audio_sr_from_tuple:
                      logger.warning(f"Explicit 'sr' ({sr} Hz) provided for saving differs from 'sr' in audio tuple "
                                     f"({audio_sr_from_tuple} Hz). The saved file header will use {sr} Hz, "
                                     f"but the audio data itself is NOT resampled by save_data/save_audio_file.")
                 save_audio_file(audio_data, save_sr, fpath, subtype=audio_subtype)
             else:
                  raise ValueError(f"Cannot save audio data tuple to non-audio core format '{ext}'. Supported audio formats: {CORE_AUDIO_WRITE_FORMATS}")

        else:
            raise TypeError(f"Unsupported data type for core saving handlers: {type(data)}. See SaveInput type alias.")

        logger.info(f"Data successfully saved by core handler to {fpath}")

    except (ValueError, TypeError) as e:
        logger.error(f"Error saving file {fpath}: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error saving file {fpath}: {e}")
        raise


# --- SQL and Filtering (Unchanged) ---

def run_sql_query(data: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Executes an SQL query on a Pandas DataFrame using pandasql.

    Args:
        data: The input DataFrame (will be referred to as 'df' in the query).
        query: The SQL query string (e.g., "SELECT colA, SUM(colB) FROM df WHERE colC > 10 GROUP BY colA").

    Returns:
        A DataFrame containing the query result. Returns an empty DataFrame if the
        query result is None or not tabular.

    Raises:
        TypeError: If input `data` is not a Pandas DataFrame.
        Exception: If the SQL query fails to execute via pandasql.
    """
    logger.info(f"Running SQL query on DataFrame: {query}")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' for run_sql_query must be a Pandas DataFrame.")
    try:
        env = {'df': data}
        result = ps.sqldf(query, env)
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
        raise RuntimeError("SQL query execution failed.") from e


def filter_data(data: pd.DataFrame, filter_expr: str) -> pd.DataFrame:
    """
    Filters data using a Pandas query expression.

    Args:
        data: The input DataFrame.
        filter_expr: The Pandas query string (e.g., 'value > 10 and channel == "A"').

    Returns:
        A DataFrame containing the filtered rows.

    Raises:
        TypeError: If input `data` is not a Pandas DataFrame.
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
        raise ValueError("Data filtering expression failed.") from e
