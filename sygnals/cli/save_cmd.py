# sygnals/cli/save_cmd.py

"""
CLI command for saving processed data, including assembling datasets.
"""

import logging
import click
import numpy as np
import pandas as pd
from pathlib import Path
import json # For parsing dict options
from typing import Optional, List, Tuple, Any, Dict, Union
import warnings # Import warnings

# Import core components
from sygnals.core.data_handler import read_data, save_data, ReadResult, NDArray
# Import formatters
from sygnals.core.ml_utils.formatters import (
    format_feature_vectors_per_segment,
    format_feature_sequences,
    format_features_as_image
)
from sygnals.config.models import SygnalsConfig # For accessing config if needed

logger = logging.getLogger(__name__)

# --- Helper to parse shape string ---
def _parse_shape(ctx, param, value: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parses a 'height,width' string into a tuple of integers."""
    if value is None:
        return None
    try:
        parts = [int(p.strip()) for p in value.split(',')]
        if len(parts) != 2 or parts[0] <= 0 or parts[1] <= 0:
            raise click.BadParameter("must be two positive integers separated by comma (e.g., '128,64').")
        return (parts[0], parts[1])
    except ValueError:
        raise click.BadParameter("must be two positive integers separated by comma (e.g., '128,64').")

# --- Helper to parse aggregation JSON ---
def _parse_aggregation(ctx, param, value: Optional[str]) -> Union[str, Dict[str, str]]:
    """Parses aggregation option, accepting a string method or a JSON dict string."""
    if value is None or not value.strip().startswith('{'):
        return value or 'mean' # Return string if not JSON dict, default to 'mean'
    try:
        agg_dict = json.loads(value)
        if not isinstance(agg_dict, dict):
            raise click.BadParameter("must be a valid JSON dictionary string (e.g., '{\"feat1\": \"mean\", \"feat2\": \"std\"}') or a single string method.")
        # Optional: Validate keys/values further if needed
        return agg_dict
    except json.JSONDecodeError:
        raise click.BadParameter("must be a valid JSON dictionary string or a single string method.")


# --- Main Save Command Group ---
@click.group("save")
@click.pass_context
def save_cmd(ctx):
    """Save processed data or assemble datasets."""
    pass

# --- Save Dataset Subcommand ---
@save_cmd.command("dataset")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option("-o", "--output", type=click.Path(resolve_path=True), required=True,
              help="Output file path for the dataset (e.g., dataset.npz, features.csv).")
@click.option("--format", "output_format_override", type=str, default=None,
              help="Explicit output format (e.g., 'npz', 'csv'). Overrides extension.")
@click.option("--assembly-method", type=click.Choice(['none', 'vectors', 'sequences', 'image'], case_sensitive=False),
              default='none', show_default=True,
              help="Method to assemble features into dataset structure.")
# Options for 'vectors' assembly
@click.option("--segment-info", type=click.Path(exists=True, dir_okay=False, resolve_path=True), default=None,
              help="Path to segment info file (CSV with 'start_frame', 'end_frame'). Required for 'vectors' assembly.")
@click.option("--aggregation", type=str, default='mean', callback=_parse_aggregation,
              help="Aggregation method(s) for 'vectors' assembly. Either a single method ('mean', 'std', etc.) "
                   "or a JSON string mapping feature names to methods (e.g., '{\"mfcc_0\":\"mean\",\"zcr\":\"max\"}').")
# Options for 'sequences' assembly
@click.option("--max-sequence-length", type=int, default=None,
              help="Maximum length for 'sequences' assembly. Truncates or pads sequences.")
@click.option("--padding-value", type=float, default=0.0, show_default=True,
              help="Value used for padding in 'sequences' assembly.")
@click.option("--truncation-strategy", type=click.Choice(['pre', 'post'], case_sensitive=False), default='post', show_default=True,
              help="Truncation strategy ('pre' or 'post') for 'sequences' assembly.")
# Options for 'image' assembly
@click.option("--output-shape", type=str, default=None, callback=_parse_shape,
              help="Target output shape 'height,width' (e.g., '128,64') for 'image' assembly.")
@click.option("--resize-order", type=click.IntRange(0, 5), default=1, show_default=True,
              help="Interpolation order (0-5) for resizing in 'image' assembly.")
@click.option("--no-normalize", is_flag=True, default=False,
              help="Disable normalization to [0, 1] for 'image' assembly.")
@click.pass_context
def save_dataset(
    ctx,
    input_file: str,
    output: str,
    output_format_override: Optional[str],
    assembly_method: str,
    segment_info: Optional[str],
    aggregation: Union[str, Dict[str, str]],
    max_sequence_length: Optional[int],
    padding_value: float,
    truncation_strategy: str,
    output_shape: Optional[Tuple[int, int]],
    resize_order: int,
    no_normalize: bool
):
    """
    Save features or assemble them into a dataset structure.

    Reads features from INPUT_FILE and saves them to OUTPUT.
    The --assembly-method option determines how features are structured:
    - none: Saves the input data directly.
    - vectors: Aggregates frame-level features into one vector per segment. Requires --segment-info.
    - sequences: Stacks features into sequences, with optional padding/truncation.
    - image: Formats features (e.g., spectrogram) into an image-like array, with optional resize/normalization.
    """
    input_path = Path(input_file)
    output_path = Path(output)
    logger.info(f"Running save dataset on: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Assembly method: {assembly_method}")

    # Determine final output path based on override
    if output_format_override:
        output_path = output_path.with_suffix(f".{output_format_override.lower()}")
        logger.info(f"Output format overridden to: '{output_format_override}'. Final path: {output_path}")

    try:
        # 1. Read Input Data (typically features - CSV, NPZ)
        data_result: ReadResult = read_data(input_path)
        data_to_save: Any = None # Initialize variable to hold final data for saving

        # --- Apply Assembly Logic ---
        if assembly_method == 'none':
            data_to_save = data_result
            logger.info("Assembly method 'none': Saving input data directly.")

        elif assembly_method == 'vectors':
            logger.info(f"Applying 'vectors' assembly with aggregation: {aggregation}")
            if segment_info is None:
                raise click.UsageError("Option '--segment-info' is required for assembly method 'vectors'.")
            if not isinstance(data_result, (dict, pd.DataFrame)):
                 raise click.UsageError(f"Input data type {type(data_result)} not suitable for 'vectors' assembly. Expected dict of 1D arrays (from NPZ) or DataFrame.")

            # Convert DataFrame to dict of 1D arrays if necessary
            features_dict: Dict[str, NDArray[np.float64]]
            if isinstance(data_result, pd.DataFrame):
                 # Assume all numeric columns (excluding potential 'time' index/column) are features
                 if 'time' in data_result.columns:
                     features_dict = {col: data_result[col].values.astype(np.float64) for col in data_result.columns if col != 'time' and pd.api.types.is_numeric_dtype(data_result[col])}
                 elif isinstance(data_result.index, (pd.TimedeltaIndex, pd.DatetimeIndex, pd.PeriodIndex)):
                      # Handle time index if 'time' column is not present
                      features_dict = {col: data_result[col].values.astype(np.float64) for col in data_result.columns if pd.api.types.is_numeric_dtype(data_result[col])}
                 else:
                      # Assume no time column/index, use all numeric columns
                      features_dict = {col: data_result[col].values.astype(np.float64) for col in data_result.columns if pd.api.types.is_numeric_dtype(data_result[col])}

                 if not features_dict:
                      raise ValueError("No suitable numeric feature columns found in input DataFrame.")
            elif isinstance(data_result, dict):
                 # Assume dict contains 1D feature arrays, filter out non-1D arrays
                 features_dict = {k: v.astype(np.float64) for k, v in data_result.items() if isinstance(v, np.ndarray) and v.ndim == 1}
                 if not features_dict:
                      raise ValueError("No suitable 1D feature arrays found in input dictionary (NPZ).")
            else: # Should be caught by initial check, but defensive
                 raise TypeError("Unexpected input data type for 'vectors' assembly.")


            # Read segment info (assume CSV with 'start_frame', 'end_frame')
            try:
                # FIX: Explicitly use pandas.read_csv
                segment_df = pd.read_csv(segment_info)
                if not {'start_frame', 'end_frame'}.issubset(segment_df.columns):
                    raise ValueError("Segment info file must contain 'start_frame' and 'end_frame' columns.")
                segment_indices = list(zip(segment_df['start_frame'].astype(int), segment_df['end_frame'].astype(int)))
                # Optional: Read labels if available
                segment_labels = segment_df['label'].tolist() if 'label' in segment_df.columns else None
            except Exception as e:
                raise click.UsageError(f"Error reading segment info file '{segment_info}': {e}")

            # Call formatter
            data_to_save = format_feature_vectors_per_segment(
                features_dict=features_dict,
                segment_indices=segment_indices,
                aggregation=aggregation,
                output_format='dataframe' if output_path.suffix.lower() in ['.csv', '.json'] else 'numpy', # Choose output based on save format
                segment_labels=segment_labels
            )

        elif assembly_method == 'sequences':
            logger.info(f"Applying 'sequences' assembly: max_len={max_sequence_length}, padding={padding_value}, trunc={truncation_strategy}")
            if not isinstance(data_result, (dict, pd.DataFrame)):
                 raise click.UsageError(f"Input data type {type(data_result)} not suitable for 'sequences' assembly. Expected dict of 1D arrays (from NPZ) or DataFrame.")

            # Convert DataFrame/Dict to dict of 1D arrays
            features_dict_seq: Dict[str, NDArray[np.float64]]
            if isinstance(data_result, pd.DataFrame):
                 if 'time' in data_result.columns:
                     features_dict_seq = {col: data_result[col].values.astype(np.float64) for col in data_result.columns if col != 'time' and pd.api.types.is_numeric_dtype(data_result[col])}
                 elif isinstance(data_result.index, (pd.TimedeltaIndex, pd.DatetimeIndex, pd.PeriodIndex)):
                      features_dict_seq = {col: data_result[col].values.astype(np.float64) for col in data_result.columns if pd.api.types.is_numeric_dtype(data_result[col])}
                 else:
                      features_dict_seq = {col: data_result[col].values.astype(np.float64) for col in data_result.columns if pd.api.types.is_numeric_dtype(data_result[col])}
                 if not features_dict_seq:
                      raise ValueError("No suitable numeric feature columns found in input DataFrame.")
            elif isinstance(data_result, dict):
                 features_dict_seq = {k: v.astype(np.float64) for k, v in data_result.items() if isinstance(v, np.ndarray) and v.ndim == 1}
                 if not features_dict_seq:
                      raise ValueError("No suitable 1D feature arrays found in input dictionary (NPZ).")
            else:
                 raise TypeError("Unexpected input data type for 'sequences' assembly.")

            # Call formatter
            # Force output to be suitable for saving (e.g., padded array for NPZ)
            formatter_output = format_feature_sequences(
                features_dict=features_dict_seq,
                max_sequence_length=max_sequence_length,
                padding_value=padding_value,
                truncation_strategy=truncation_strategy, # type: ignore
                output_format='padded_array' # Force padded array for saving
            )
            # formatter_output is shape (1, seq_len, n_features)
            # Save as dict for NPZ, or potentially reshape/convert for CSV/JSON?
            if output_path.suffix.lower() != '.npz':
                 logger.warning(f"Saving sequence data (shape {formatter_output.shape}) to non-NPZ format '{output_path.suffix}'. Result might be unexpected. Saving first sequence as 2D array.")
                 # Save the first (only) sequence as a 2D array if not NPZ
                 if formatter_output.shape[0] == 1:
                      data_to_save = formatter_output[0] # Save as (seq_len, n_features)
                 else: # Should not happen with current formatter output
                      data_to_save = formatter_output
            else:
                 # Save the 3D array directly, maybe with a key like 'sequences'
                 data_to_save = {'sequences': formatter_output}


        elif assembly_method == 'image':
            logger.info(f"Applying 'image' assembly: shape={output_shape}, order={resize_order}, normalize={not no_normalize}")
            feature_map: Optional[NDArray[np.float64]] = None
            if isinstance(data_result, np.ndarray) and data_result.ndim == 2:
                feature_map = data_result.astype(np.float64)
            elif isinstance(data_result, pd.DataFrame):
                # Assume DataFrame contains the 2D feature map (e.g., spectrogram)
                # Try converting directly, assuming all columns are numeric parts of the map
                try:
                    # Select only numeric columns for the map
                    numeric_df = data_result.select_dtypes(include=np.number)
                    if numeric_df.empty:
                        raise ValueError("DataFrame contains no numeric columns to form a feature map.")
                    feature_map = numeric_df.values.astype(np.float64)
                    if feature_map.ndim != 2 or feature_map.size == 0:
                         raise ValueError("DataFrame does not represent a valid 2D feature map.")
                except Exception as e:
                    raise click.UsageError(f"Could not convert input DataFrame to 2D feature map for 'image' assembly: {e}")
            elif isinstance(data_result, dict):
                # Try to find a suitable 2D array in the dictionary (e.g., 'spectrogram', 'data')
                potential_keys = ['spectrogram', 'data', 'image'] + list(data_result.keys())
                for key in potential_keys:
                    value = data_result.get(key)
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        feature_map = value.astype(np.float64)
                        logger.debug(f"Using array under key '{key}' from input dictionary for 'image' assembly.")
                        break
                if feature_map is None:
                    raise click.UsageError("Could not find a suitable 2D NumPy array in input dictionary for 'image' assembly.")
            else:
                raise click.UsageError(f"Input data type {type(data_result)} not suitable for 'image' assembly. Expected 2D NumPy array (e.g., spectrogram from NPZ or CSV).")

            # Call formatter
            data_to_save = format_features_as_image(
                feature_map=feature_map,
                output_shape=output_shape,
                resize_order=resize_order,
                normalize=not no_normalize
            )
            # If saving to NPZ, wrap in a dict
            if output_path.suffix.lower() == '.npz':
                 data_to_save = {'image_features': data_to_save}

        # --- Save Formatted Data ---
        if data_to_save is None:
             # This should only happen if assembly method logic fails unexpectedly
             raise click.ClickException("Internal error: Formatted data is None after assembly.")

        save_data(data_to_save, output_path) # save_data handles different types

        click.echo(f"Successfully saved dataset from '{input_path.name}' to '{output_path.name}' (assembly: {assembly_method}).")

    except FileNotFoundError:
        # Should be caught by initial check, but handle defensively
        raise click.UsageError(f"Input file not found: {input_path}")
    except (ValueError, TypeError, click.UsageError) as e:
        # Catch specific errors from formatters or data handling
        logger.error(f"Error during dataset assembly/saving: {e}", exc_info=True) # Log traceback for these errors
        raise click.UsageError(f"Error during dataset assembly/saving: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during dataset saving: {e}", exc_info=True)
        raise click.Abort(f"An unexpected error occurred: {e}") # Use Abort for unexpected errors
