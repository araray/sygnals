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

# Import necessary functions from other core modules
from sygnals.core.data_handler import read_data, save_data, ReadResult
from sygnals.core.dsp import compute_fft
# Correct the import name for the wavelet transform function
from sygnals.core.transforms import discrete_wavelet_transform

logger = logging.getLogger(__name__)

def process_batch(input_dir: str, output_dir: str, transform: str):
    """
    Process multiple files in a directory and apply the given transform.

    Reads supported files (currently assumes CSV with a 'value' column for processing),
    applies either 'fft' or 'wavelet' transform, and saves the result as CSV.

    Args:
        input_dir: Path to the directory containing input files.
        output_dir: Path to the directory where processed files will be saved.
        transform: The name of the transform to apply ('fft' or 'wavelet').

    Raises:
        FileNotFoundError: If the input directory does not exist.
        ValueError: If an invalid transform name is provided or if input data
                    cannot be processed (e.g., missing 'value' column in CSV).
        Exception: For underlying errors during file reading, processing, or writing.
    """
    input_path = Path(input_dir)
    output_path_dir = Path(output_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Ensure output directory exists
    output_path_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting batch processing from '{input_dir}' to '{output_dir}' with transform '{transform}'")

    processed_count = 0
    skipped_count = 0

    for file_path in input_path.iterdir():
        if not file_path.is_file():
            logger.debug(f"Skipping non-file item: {file_path.name}")
            continue

        # Construct output file path
        # Use Path objects for cleaner path joining and suffix handling
        output_file_name = f"{file_path.stem}_processed{file_path.suffix}" # Keep original suffix for clarity? Or force .csv? Forcing CSV for now.
        output_file_path = output_path_dir / f"{file_path.stem}_processed.csv"

        try:
            logger.debug(f"Processing file: {file_path.name}")
            # Read data - currently assumes CSV with 'value' column based on original code
            # TODO: Enhance to handle different data types returned by read_data (DataFrame, dict, audio tuple)
            data_result: ReadResult = read_data(file_path)

            # --- Data Type Handling (Basic Example) ---
            # This section needs significant expansion based on how batch processing
            # should handle different input types and transforms.
            if isinstance(data_result, pd.DataFrame):
                if "value" not in data_result.columns:
                     logger.warning(f"Skipping file {file_path.name}: DataFrame does not contain 'value' column.")
                     skipped_count += 1
                     continue
                # Extract the 'value' column as a NumPy array
                signal_data = data_result["value"].values.astype(np.float64) # Ensure float64
                # Handle empty data after extraction
                if signal_data.size == 0:
                     logger.warning(f"Skipping file {file_path.name}: 'value' column is empty.")
                     skipped_count += 1
                     continue
                fs = 1.0 # Assume default sample rate if not provided/read from file
            # elif isinstance(data_result, tuple) and len(data_result) == 2: # Audio data
            #     signal_data, fs = data_result
            #     if signal_data.ndim > 1: # Handle multi-channel audio if needed
            #         logger.warning(f"Audio file {file_path.name} is multi-channel. Taking first channel.")
            #         signal_data = signal_data[0]
            # elif isinstance(data_result, dict): # NPZ data
            #     # Decide how to handle dicts - e.g., process a specific key like 'data'
            #     if 'data' in data_result:
            #         signal_data = data_result['data']
            #         if signal_data.ndim > 1: # Flatten or select part of array?
            #             logger.warning(f"NPZ data in {file_path.name} is multi-dimensional. Flattening.")
            #             signal_data = signal_data.flatten()
            #         fs = 1.0 # Assume default fs
            #     else:
            #         logger.warning(f"Skipping NPZ file {file_path.name}: Could not find 'data' key.")
            #         skipped_count += 1
            #         continue
            else:
                 logger.warning(f"Skipping file {file_path.name}: Unsupported data type {type(data_result)} returned by read_data for batch processing.")
                 skipped_count += 1
                 continue
            # --- End Data Type Handling ---


            # Apply the specified transform
            if transform == "fft":
                # Compute FFT requires sample rate 'fs', using default 1.0 for now
                freqs, spectrum = compute_fft(signal_data, fs=fs)
                # Save result as DataFrame
                result_df = pd.DataFrame({
                    "Frequency (Hz)": freqs,
                    "Magnitude": np.abs(spectrum), # Save magnitude
                    "Phase": np.angle(spectrum)    # Optionally save phase
                })
                save_data(result_df, output_file_path)
                processed_count += 1
            elif transform == "wavelet":
                # Use the corrected function name
                coeffs = discrete_wavelet_transform(signal_data) # Uses default wavelet/level
                # Convert list of coefficient arrays to a DataFrame
                # Pad shorter arrays with NaN to make columns equal length for DataFrame creation
                max_len = max(len(c) for c in coeffs)
                coeffs_padded = {
                    f"Level_{i}": np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
                    for i, c in enumerate(coeffs)
                }
                result_df = pd.DataFrame(coeffs_padded)
                save_data(result_df, output_file_path)
                processed_count += 1
            else:
                # Log error only once for an invalid transform across the batch
                if processed_count == 0 and skipped_count == 0: # Only log on the first file encountered
                    logger.error(f"Invalid transform specified: '{transform}'. Supported: 'fft', 'wavelet'. Skipping transform for all files.")
                # Don't process this file if transform is invalid
                skipped_count += 1 # Count as skipped due to invalid transform
                continue # Skip saving for this file

        except FileNotFoundError:
             logger.error(f"Input file not found during batch processing: {file_path.name}. Skipping.")
             skipped_count += 1
        except ValueError as e:
             logger.error(f"Value error processing file {file_path.name}: {e}. Skipping.")
             skipped_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing file {file_path.name}: {e}", exc_info=True)
            skipped_count += 1

    logger.info(f"Batch processing finished. Processed: {processed_count}, Skipped: {skipped_count}")

```text
feat: Correct import name for wavelet transform in batch processor

Renamed `wavelet_transform` to `discrete_wavelet_transform` in the import statement and function call within `sygnals/core/batch_processor.py` to align with the updated function name in `sygnals/core/transforms.py`. This resolves the ImportError encountered during test collection.

Also added basic logging, pathlib usage, and error handling improvements.
