# sygnals_hdf5/hdf5_io.py

"""
Core HDF5 reading and writing functions for the sygnals-hdf5 plugin.
Uses h5py for basic dataset I/O.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Union # Import necessary types

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Import types from data_handler for type hints if needed
from sygnals.core.data_handler import ReadResult, SaveInput

logger = logging.getLogger(__name__)

def read_hdf5(path: Union[str, Path], **kwargs: Any) -> Dict[str, NDArray[Any]]:
    """
    Reads top-level datasets from an HDF5 file into a dictionary.

    NOTE: This basic implementation only reads datasets directly under the root group.
          Nested groups and attributes are currently ignored.

    Args:
        path: Path to the input HDF5 file (.h5 or .hdf5).
        **kwargs: Additional keyword arguments (currently unused by this basic reader).

    Returns:
        A dictionary where keys are dataset names and values are NumPy arrays.

    Raises:
        ImportError: If h5py is not installed.
        FileNotFoundError: If the path does not exist.
        Exception: For errors during h5py reading.
    """
    # Check for h5py dependency
    try:
        import h5py
    except ImportError:
        raise ImportError("The 'h5py' library is required to read HDF5 files. Please install it.")

    fpath = Path(path)
    if not fpath.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {fpath}")

    logger.debug(f"Reading HDF5 file: {fpath} (reading top-level datasets)")
    data_dict: Dict[str, NDArray[Any]] = {}
    try:
        with h5py.File(fpath, 'r') as hf:
            for key in hf.keys():
                item = hf[key]
                if isinstance(item, h5py.Dataset):
                    # Read dataset into memory as NumPy array
                    data_dict[key] = item[()] # [()] reads the entire dataset
                    logger.debug(f"Read dataset '{key}' with shape {data_dict[key].shape} and dtype {data_dict[key].dtype}")
                else:
                    logger.warning(f"Skipping item '{key}' in HDF5 file (not a dataset at root level). Type: {type(item)}")
        logger.info(f"Successfully read {len(data_dict)} datasets from HDF5 file: {fpath.name}")
        return data_dict
    except Exception as e:
        logger.error(f"Error reading HDF5 file {fpath}: {e}")
        raise RuntimeError(f"Failed to read HDF5 file {fpath}.") from e

def save_hdf5(data: SaveInput, path: Union[str, Path], **kwargs: Any):
    """
    Saves data (dictionary of NumPy arrays or Pandas DataFrame) to an HDF5 file.

    - If `data` is a dictionary, keys become dataset names, values (NumPy arrays) are saved.
    - If `data` is a DataFrame, it's converted to a dictionary of columns (as NumPy arrays) first.
    - Existing files are overwritten.

    Args:
        data: The data to save. Expected to be a dictionary where values are NumPy arrays,
              or a Pandas DataFrame.
        path: Path where the output HDF5 file (.h5 or .hdf5) will be saved.
        **kwargs: Additional keyword arguments passed to h5py.File.create_dataset().
                  Common arguments include 'compression' (e.g., "gzip"), 'chunks'.

    Raises:
        TypeError: If the input data is not a supported type (dict of ndarray, DataFrame).
        ImportError: If h5py is not installed.
        Exception: For errors during h5py writing.
    """
    # Check for h5py dependency
    try:
        import h5py
    except ImportError:
        raise ImportError("The 'h5py' library is required to write HDF5 files. Please install it.")

    fpath = Path(path)
    # Ensure output directory exists
    fpath.parent.mkdir(parents=True, exist_ok=True)

    data_to_write: Dict[str, NDArray[Any]]

    # Convert DataFrame to dict if necessary
    if isinstance(data, pd.DataFrame):
        logger.debug(f"Converting DataFrame (shape {data.shape}) to dict of arrays for HDF5 saving.")
        data_to_write = {col: data[col].values for col in data.columns}
    elif isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()):
        data_to_write = data # Already a dictionary of arrays
    else:
        raise TypeError(f"Data type {type(data)} not supported for HDF5 saving. Expected dict of NumPy arrays or Pandas DataFrame.")

    logger.debug(f"Saving {len(data_to_write)} arrays to HDF5 file: {fpath} with kwargs: {kwargs}")
    try:
        with h5py.File(fpath, 'w') as hf: # 'w' mode overwrites if file exists
            for key, array_data in data_to_write.items():
                # Create dataset for each key-value pair
                hf.create_dataset(key, data=array_data, **kwargs)
        logger.info(f"Successfully saved data to HDF5 file: {fpath.name}")
    except Exception as e:
        logger.error(f"Error saving HDF5 file {fpath}: {e}")
        raise RuntimeError(f"Failed to save HDF5 file {fpath}.") from e
