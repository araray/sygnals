# sygnals_parquet/parquet_io.py

"""
Core Parquet reading and writing functions for the sygnals-parquet plugin.
Uses pandas with pyarrow engine by default.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Union # Import necessary types

import pandas as pd
# Import pyarrow to check for it, although pandas handles the direct call
try:
    import pyarrow
    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False

# Import types from data_handler for type hints if needed
from sygnals.core.data_handler import ReadResult, SaveInput

logger = logging.getLogger(__name__)

def read_parquet(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """
    Reads data from a Parquet file into a Pandas DataFrame.

    Args:
        path: Path to the input Parquet file.
        **kwargs: Additional keyword arguments passed directly to pandas.read_parquet().
                  Common arguments include 'columns', 'filters'.

    Returns:
        A Pandas DataFrame containing the data from the Parquet file.

    Raises:
        ImportError: If pyarrow (or configured engine) is not installed.
        FileNotFoundError: If the path does not exist.
        Exception: For errors during pandas/pyarrow reading.
    """
    if not _PYARROW_AVAILABLE:
        # Check can be done here or rely on pandas to raise ImportError
        raise ImportError("The 'pyarrow' library is required to read Parquet files. Please install it.")

    fpath = Path(path)
    if not fpath.is_file():
        raise FileNotFoundError(f"Parquet file not found: {fpath}")

    logger.debug(f"Reading Parquet file: {fpath} with kwargs: {kwargs}")
    try:
        # Use pandas.read_parquet, which uses pyarrow by default if installed
        df = pd.read_parquet(fpath, engine='pyarrow', **kwargs)
        logger.info(f"Successfully read Parquet file {fpath.name}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading Parquet file {fpath}: {e}")
        raise RuntimeError(f"Failed to read Parquet file {fpath}.") from e

def save_parquet(data: SaveInput, path: Union[str, Path], **kwargs: Any):
    """
    Saves data (expected to be a Pandas DataFrame) to a Parquet file.

    Args:
        data: The data to save. Must be a Pandas DataFrame for Parquet format.
        path: Path where the output Parquet file will be saved.
        **kwargs: Additional keyword arguments passed directly to DataFrame.to_parquet().
                  Common arguments include 'engine', 'compression', 'index'.

    Raises:
        TypeError: If the input data is not a Pandas DataFrame.
        ImportError: If pyarrow (or configured engine) is not installed.
        Exception: For errors during pandas/pyarrow writing.
    """
    if not _PYARROW_AVAILABLE:
        raise ImportError("The 'pyarrow' library is required to write Parquet files. Please install it.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Data type {type(data)} not supported for Parquet saving. Expected Pandas DataFrame.")

    fpath = Path(path)
    # Ensure output directory exists (should be handled by data_handler.save_data, but good practice)
    fpath.parent.mkdir(parents=True, exist_ok=True)

    # Default to not saving the DataFrame index unless specified in kwargs
    save_index = kwargs.pop('index', False)

    logger.debug(f"Saving DataFrame (shape {data.shape}) to Parquet file: {fpath} with index={save_index}, kwargs: {kwargs}")
    try:
        # Use DataFrame.to_parquet
        data.to_parquet(fpath, engine='pyarrow', index=save_index, **kwargs)
        logger.info(f"Successfully saved DataFrame to Parquet file: {fpath.name}")
    except Exception as e:
        logger.error(f"Error saving Parquet file {fpath}: {e}")
        raise RuntimeError(f"Failed to save Parquet file {fpath}.") from e
