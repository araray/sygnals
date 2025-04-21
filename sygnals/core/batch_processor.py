# sygnals/core/batch_processor.py

"""
Provides functionality for processing batches of files.
Applies specified transformations to data read from files in an input directory
and saves the results to an output directory.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path # Use pathlib for path operations
# Import necessary types
from typing import Tuple, Optional, Union, Literal, Dict, Any

# Import necessary functions from other core modules
from sygnals.core.data_handler import read_data, save_data, ReadResult, SUPPORTED_READ_FORMATS
from sygnals.core.dsp import compute_fft
# Correct the import name for the wavelet transform function
from sygnals.core.transforms import discrete_wavelet_transform

logger = logging.getLogger(__name__)

def process_batch(input_dir: str, output_dir: str, transform: str):
    """
    Process multiple files in a directory and apply the given transform.

    Reads supported files (CSV, NPZ, WAV, FLAC, OGG, MP3), extracts relevant
    data (e.g., 'value' from CSV, 'data' from NPZ, audio signal), applies
    'fft' or 'wavelet' transform, and saves the result appropriately
    (CSV for FFT, NPZ for Wavelet).

    Args:
        input_dir: Path to the directory containing input files.
        output_dir: Path to the directory where processed files will be saved.
        transform: The name of the transform to apply ('fft' or 'wavelet').

    Raises:
        FileNotFoundError: If the input directory does not exist.
        ValueError: If an invalid transform name is provided.
        # Individual file errors are logged and skipped.
    """
    input_path = Path(input_dir)
    output_path_dir = Path(output_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if transform not in ["fft", "wavelet"]:
        # Raise error early if transform is invalid for the whole batch
        raise ValueError(f"Invalid transform specified: '{transform}'. Supported: 'fft', 'wavelet'.")

    # Ensure output directory exists
    output_path_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting batch processing from '{input_dir}' to '{output_dir}' with transform '{transform}'")

    processed_count = 0
    skipped_count = 0

    for file_path in input_path.iterdir():
        if not file_path.is_file():
            logger.debug(f"Skipping non-file item: {file_path.name}")
            continue

        # Check if the file extension is supported for reading
        if file_path.suffix.lower() not in SUPPORTED_READ_FORMATS:
            logger.warning(f"Skipping file {file_path.name}: Unsupported read format '{file_path.suffix}'.")
            skipped_count += 1
            continue

        # Determine output path and format based on transform
        output_stem = f"{file_path.stem}_processed"
        if transform == "fft":
            output_file_path = output_path_dir / f"{output_stem}.csv"
        elif transform == "wavelet":
            output_file_path = output_path_dir / f"{output_stem}.npz"
        else:
            # This case should not be reached due to check above, but handle defensively
            logger.error(f"Internal error: Unexpected transform '{transform}'. Skipping file {file_path.name}.")
            skipped_count += 1
            continue

        try:
            logger.debug(f"Processing file: {file_path.name}")
            # Read data - read_data handles different formats
            # Pass sr=None to load audio at native rate initially
            data_result: ReadResult = read_data(file_path, sr=None)

            signal_data: Optional[NDArray[np.float64]] = None
            fs: float = 1.0 # Default sample rate if not available

            # --- Extract signal data based on type ---
            if isinstance(data_result, pd.DataFrame):
                if "value" in data_result.columns:
                    # Ensure 'value' column is numeric and convert to float64 numpy array
                    signal_data = pd.to_numeric(data_result["value"], errors='coerce').dropna().values.astype(np.float64)
                    # Try to infer sample rate if 'time' column exists
                    if 'time' in data_result.columns and len(data_result['time']) > 1:
                        time_diff = np.diff(data_result['time'])
                        if np.allclose(time_diff, time_diff[0]): # Check if time steps are constant
                             fs = 1.0 / time_diff[0]
                             logger.debug(f"Inferred sample rate {fs:.2f} Hz from 'time' column in {file_path.name}")
                else:
                    logger.warning(f"Skipping CSV {file_path.name}: No 'value' column found.")
                    skipped_count += 1
                    continue

            elif isinstance(data_result, tuple) and len(data_result) == 2: # Audio data (signal, sr)
                signal_data, fs_read = data_result
                fs = float(fs_read) # Use sample rate from audio file
                if signal_data.ndim > 1: # Handle multi-channel audio - use mean
                    logger.warning(f"Audio file {file_path.name} is multi-channel ({signal_data.shape}). Converting to mono by averaging.")
                    signal_data = np.mean(signal_data, axis=0).astype(np.float64) # Average channels
                # Ensure float64
                signal_data = signal_data.astype(np.float64)

            elif isinstance(data_result, dict): # NPZ data
                # Assume the primary data is under the key 'data'
                if 'data' in data_result and isinstance(data_result['data'], np.ndarray):
                    signal_data = data_result['data']
                    if signal_data.ndim > 1: # Flatten multi-dimensional arrays if necessary
                        logger.warning(f"NPZ data key 'data' in {file_path.name} is multi-dimensional {signal_data.shape}. Flattening.")
                        signal_data = signal_data.flatten().astype(np.float64)
                    else:
                         signal_data = signal_data.astype(np.float64)
                    # Check for sample rate if saved in NPZ
                    if 'fs' in data_result and isinstance(data_result['fs'], (int, float)):
                         fs = float(data_result['fs'])
                         logger.debug(f"Using sample rate {fs:.2f} Hz from NPZ file {file_path.name}")
                else:
                    logger.warning(f"Skipping NPZ {file_path.name}: Could not find a NumPy array under the key 'data'.")
                    skipped_count += 1
                    continue
            else:
                 # This case might occur if read_data returns an unexpected type
                 logger.warning(f"Skipping file {file_path.name}: Unsupported data structure returned by read_data: {type(data_result)}.")
                 skipped_count += 1
                 continue

            # Check if signal data is empty after extraction/conversion
            if signal_data is None or signal_data.size == 0:
                logger.warning(f"Skipping file {file_path.name}: No valid signal data extracted.")
                skipped_count += 1
                continue
            # --- End data extraction ---

            logger.debug(f"Extracted signal data (shape: {signal_data.shape}, fs: {fs:.2f} Hz) for {file_path.name}. Applying transform '{transform}'...")

            # --- Apply the specified transform ---
            if transform == "fft":
                freqs, spectrum = compute_fft(signal_data, fs=fs)
                # Save result as DataFrame (CSV)
                result_to_save: Any = pd.DataFrame({
                    "Frequency (Hz)": freqs,
                    "Magnitude": np.abs(spectrum),
                    "Phase": np.angle(spectrum)
                })
                logger.debug(f"FFT computed for {file_path.name}. Result shape: {result_to_save.shape}")

            elif transform == "wavelet":
                coeffs = discrete_wavelet_transform(signal_data) # Uses default wavelet/level
                # Save result as dictionary of arrays (NPZ)
                result_to_save = {f"Level_{i}": c for i, c in enumerate(coeffs)}
                logger.debug(f"Wavelet transform computed for {file_path.name}. Number of coefficient arrays: {len(coeffs)}")

            else:
                 # Should not be reached, but handle defensively
                 logger.error(f"Internal Error: Transform '{transform}' not handled in saving block for {file_path.name}.")
                 skipped_count += 1
                 continue

            # --- Save the result ---
            logger.debug(f"Attempting to save processed data for {file_path.name} to {output_file_path}")
            save_data(result_to_save, output_file_path) # save_data handles type checking
            logger.info(f"Successfully processed and saved: {file_path.name} -> {output_file_path.name}")
            processed_count += 1

        except FileNotFoundError:
             # This might happen if the input file disappears during processing
             logger.error(f"Input file not found during batch processing: {file_path.name}. Skipping.")
             skipped_count += 1
        except ValueError as e:
             # Catch specific errors during processing (e.g., invalid data format)
             logger.error(f"Value error processing file {file_path.name}: {e}. Skipping.")
             skipped_count += 1
        except Exception as e:
            # Catch any other unexpected errors during read, process, or save
            logger.error(f"Unexpected error processing file {file_path.name}: {e}", exc_info=True)
            skipped_count += 1

    logger.info(f"Batch processing finished. Processed: {processed_count}, Skipped: {skipped_count}")
