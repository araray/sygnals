# sygnals/cli/features_cmd.py

"""
CLI commands related to feature extraction and transformation.
"""

import logging
import click
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict

# Import core components
from sygnals.core.data_handler import read_data, save_data, ReadResult
from sygnals.core.features.manager import extract_features # For 'extract' subcommand later
from sygnals.core.ml_utils.scaling import apply_scaling # For 'transform scale'
from sygnals.config.models import SygnalsConfig # For accessing config if needed

logger = logging.getLogger(__name__)

# --- Main Features Command Group ---
@click.group("features")
@click.pass_context
def features_cmd(ctx):
    """Extract, transform, and manage signal features."""
    pass

# --- Extract Subcommand (Placeholder/Future) ---
@features_cmd.command("extract")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option("-o", "--output", type=click.Path(resolve_path=True), required=True,
              help="Output file path for extracted features (e.g., features.csv, features.npz).")
@click.option("-f", "--feature", "features", multiple=True, required=True,
              help="Feature(s) to extract (e.g., 'rms_energy', 'mfcc'). Use 'all' for all known features. Can be repeated.")
@click.option("--frame-length", type=int, default=2048, show_default=True, help="Analysis frame length (samples).")
@click.option("--hop-length", type=int, default=512, show_default=True, help="Hop length between frames (samples).")
# Add more options for feature params, output format etc. later
@click.pass_context
def features_extract(
    ctx,
    input_file: str,
    output: str,
    features: Tuple[str], # Click collects multiple options into a tuple
    frame_length: int,
    hop_length: int
):
    """Extract features from an audio signal."""
    input_path = Path(input_file)
    output_path = Path(output)
    feature_list = list(features) # Convert tuple to list
    if len(feature_list) == 1 and feature_list[0].lower() == 'all':
         feature_list = ['all'] # Keep 'all' as a single item list

    logger.info(f"Running feature extraction on: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Features: {feature_list}")
    logger.info(f"Params: frame={frame_length}, hop={hop_length}")

    try:
        # 1. Read Input Audio
        data_result: ReadResult = read_data(input_path)
        if not isinstance(data_result, tuple) or len(data_result) != 2:
             raise click.UsageError(f"Input file '{input_path.name}' is not recognized as audio.")
        signal_data, sr = data_result
        if signal_data.ndim != 1:
             logger.warning("Input audio is multi-channel. Converting to mono by averaging for feature extraction.")
             signal_data = np.mean(signal_data, axis=0)

        # Determine output format based on extension
        output_ext = output_path.suffix.lower()
        if output_ext == '.csv':
             output_format = 'dataframe'
        elif output_ext == '.npz':
             output_format = 'dict_of_arrays'
        else:
             # Default to dataframe and let save_data handle format conversion if possible
             logger.warning(f"Output format '{output_ext}' not explicitly CSV or NPZ. Defaulting to DataFrame format internally.")
             output_format = 'dataframe'


        # 2. Extract Features using the manager
        # TODO: Add way to pass feature_params from CLI
        extracted_data = extract_features(
            y=signal_data,
            sr=sr,
            features=feature_list,
            frame_length=frame_length,
            hop_length=hop_length,
            output_format=output_format # Request format needed for saving
        )

        if isinstance(extracted_data, pd.DataFrame) and extracted_data.empty:
             logger.warning("Feature extraction resulted in empty DataFrame.")
             click.echo("Warning: No features extracted or signal too short.")
             return
        elif isinstance(extracted_data, dict) and not any(k != 'time' for k in extracted_data):
             logger.warning("Feature extraction resulted in empty dictionary (only time).")
             click.echo("Warning: No features extracted or signal too short.")
             return

        # 3. Save Features
        save_data(extracted_data, output_path) # Pass DataFrame or Dict directly

        click.echo(f"Successfully extracted features from '{input_path.name}' and saved to '{output_path.name}'.")

    except FileNotFoundError:
        raise click.UsageError(f"Input file not found: {input_path}")
    except ValueError as e:
        raise click.UsageError(f"Error during feature extraction: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during feature extraction: {e}", exc_info=True)
        raise click.Abort(f"An unexpected error occurred: {e}")


# --- Transform Subcommand Group ---
@features_cmd.group("transform")
@click.pass_context
def features_transform(ctx):
    """Transform extracted features (e.g., scaling)."""
    pass


# --- Transform Scale Subcommand ---
@features_transform.command("scale")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option("-o", "--output", type=click.Path(resolve_path=True), required=True,
              help="Output file path for scaled features (e.g., features_scaled.csv).")
@click.option("--scaler", type=click.Choice(['standard', 'minmax', 'robust'], case_sensitive=False),
              default='standard', show_default=True, help="Type of scaler to apply.")
# Add options for scaler parameters later if needed (e.g., --with-mean False)
@click.pass_context
def transform_scale(ctx, input_file: str, output: str, scaler: str):
    """Apply scaling (normalization) to feature data."""
    input_path = Path(input_file)
    output_path = Path(output)
    logger.info(f"Running feature scaling on: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Scaler type: {scaler}")

    try:
        # 1. Read Input Features (CSV or NPZ)
        data_result: ReadResult = read_data(input_path)

        feature_matrix: Optional[NDArray[np.float64]] = None
        original_columns: Optional[List[str]] = None
        time_index: Optional[pd.Index] = None # Store time index if reading DataFrame

        if isinstance(data_result, pd.DataFrame):
             # Assume columns are features, potentially excluding a 'time' column
             if 'time' in data_result.columns:
                 feature_df = data_result.drop(columns=['time'])
                 time_index = data_result.index # Preserve original index if it's time-based
             elif isinstance(data_result.index, pd.TimedeltaIndex) or isinstance(data_result.index, pd.DatetimeIndex):
                  feature_df = data_result
                  time_index = data_result.index
             else:
                  feature_df = data_result
             # Convert to NumPy array, ensuring numeric types
             try:
                 feature_matrix = feature_df.select_dtypes(include=np.number).values.astype(np.float64)
                 original_columns = feature_df.select_dtypes(include=np.number).columns.tolist()
             except Exception as e:
                 raise ValueError(f"Could not convert DataFrame columns to numeric features: {e}")

        elif isinstance(data_result, dict):
             # Assume NPZ: find the main data array (e.g., 'features', 'data')
             # This logic might need refinement based on how features are saved to NPZ
             potential_keys = ['features', 'data'] + list(data_result.keys())
             data_key = None
             for key in potential_keys:
                 if isinstance(data_result.get(key), np.ndarray) and data_result[key].ndim >= 1:
                     data_key = key
                     break
             if data_key:
                 feature_matrix = data_result[data_key].astype(np.float64)
                 original_columns = [f"feature_{i}" for i in range(feature_matrix.shape[1])] if feature_matrix.ndim == 2 else ['feature_0']
                 logger.info(f"Using array under key '{data_key}' from NPZ file for scaling.")
                 # Handle potential 1D array from NPZ
                 if feature_matrix.ndim == 1:
                      feature_matrix = feature_matrix.reshape(-1, 1)
             else:
                 raise ValueError("Could not find a suitable NumPy array in the input NPZ file.")
        else:
            raise click.UsageError(f"Input file '{input_path.name}' format not suitable for feature scaling (expected CSV or NPZ with features).")

        if feature_matrix is None or feature_matrix.size == 0:
             raise ValueError("No valid numeric feature data found in the input file.")
        if feature_matrix.ndim != 2:
             raise ValueError(f"Feature matrix must be 2D (samples/frames x features), got shape {feature_matrix.shape}")

        # 2. Apply Scaling
        # TODO: Add way to pass scaler_params from CLI
        scaled_features, fitted_scaler = apply_scaling(
            features=feature_matrix,
            scaler_type=scaler # type: ignore
        )
        # TODO: Optionally save the fitted scaler instance?

        # 3. Save Scaled Features
        output_ext = output_path.suffix.lower()
        if output_ext == '.csv':
            # Reconstruct DataFrame with original columns and index if possible
            scaled_df = pd.DataFrame(scaled_features, columns=original_columns, index=time_index)
            # If time was dropped, add it back if index was time-based
            if time_index is not None and 'time' not in scaled_df.columns:
                 # This assumes index is time, might need adjustment
                 scaled_df.index.name = 'time'
                 scaled_df = scaled_df.reset_index()
            save_data(scaled_df, output_path)
        elif output_ext == '.npz':
             # Save as NPZ, potentially preserving other arrays from input dict
             if isinstance(data_result, dict):
                 output_dict = data_result.copy() # Start with original dict
                 output_dict[data_key] = scaled_features # Overwrite with scaled data
                 save_data(output_dict, output_path)
             else: # Original was DataFrame, save only scaled features
                 save_data({'scaled_features': scaled_features}, output_path)
        else:
             # Attempt saving as CSV by default if format unknown
             logger.warning(f"Unsupported output format '{output_ext}' for scaled features. Saving as CSV.")
             scaled_df = pd.DataFrame(scaled_features, columns=original_columns, index=time_index)
             if time_index is not None and 'time' not in scaled_df.columns:
                 scaled_df.index.name = 'time'
                 scaled_df = scaled_df.reset_index()
             output_path_csv = output_path.with_suffix('.csv')
             save_data(scaled_df, output_path_csv)
             logger.warning(f"Scaled features saved to {output_path_csv} instead.")


        click.echo(f"Successfully scaled features from '{input_path.name}' using '{scaler}' scaler, saved to '{output_path.name}'.")

    except FileNotFoundError:
        raise click.UsageError(f"Input file not found: {input_path}")
    except ValueError as e:
        raise click.UsageError(f"Error during feature scaling: {e}")
    except ImportError as e:
         # Catch missing scikit-learn
         raise click.UsageError(f"Missing dependency for scaling: {e}. Try `pip install scikit-learn`.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during feature scaling: {e}", exc_info=True)
        raise click.Abort(f"An unexpected error occurred: {e}")
