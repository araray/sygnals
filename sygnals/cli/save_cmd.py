# sygnals/cli/save_cmd.py

"""
CLI command for saving processed data, including assembling datasets.
"""

import logging
import click
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict

# Import core components
from sygnals.core.data_handler import read_data, save_data, ReadResult
# Import formatters when implemented
# from sygnals.core.ml_utils.formatters import (
#     format_feature_vectors_per_segment,
#     format_feature_sequences,
#     format_features_as_image
# )
from sygnals.config.models import SygnalsConfig # For accessing config if needed

logger = logging.getLogger(__name__)

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
              help="Method to assemble features into dataset structure ('none' just saves input).")
# Add options for specific assembly methods later (e.g., --aggregation, --max-length)
@click.pass_context
def save_dataset(
    ctx,
    input_file: str,
    output: str,
    output_format_override: Optional[str],
    assembly_method: str
):
    """
    Save features or assemble them into a dataset structure.

    Currently, only 'none' assembly method is implemented (saves input as is).
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
        # 1. Read Input Data (assumed to be features - CSV, NPZ)
        data_result: ReadResult = read_data(input_path)

        # --- Placeholder for Assembly Logic ---
        if assembly_method == 'none':
            # Just save the input data as is
            data_to_save = data_result
            logger.info("Assembly method 'none': Saving input data directly.")
        elif assembly_method == 'vectors':
            logger.warning("Assembly method 'vectors' is not yet implemented. Saving input data directly.")
            # TODO: Implement call to format_feature_vectors_per_segment
            # Requires segment info and aggregation parameters (passed via CLI options)
            data_to_save = data_result # Placeholder
        elif assembly_method == 'sequences':
            logger.warning("Assembly method 'sequences' is not yet implemented. Saving input data directly.")
            # TODO: Implement call to format_feature_sequences
            # Requires parameters like max_length (passed via CLI options)
            data_to_save = data_result # Placeholder
        elif assembly_method == 'image':
            logger.warning("Assembly method 'image' is not yet implemented. Saving input data directly.")
            # TODO: Implement call to format_features_as_image
            # Requires parameters like output_shape (passed via CLI options)
            # Ensure input data is suitable (e.g., spectrogram)
            if isinstance(data_result, (dict, tuple)): # Cannot format dict/tuple as image easily
                 logger.error("Cannot use 'image' assembly method on non-DataFrame/array input without further processing.")
                 data_to_save = data_result # Save original
            else: # Assume DataFrame or Array is spectrogram-like
                 data_to_save = data_result # Placeholder
        else:
             # Should not happen due to click.Choice
             raise click.UsageError(f"Internal error: Unknown assembly method '{assembly_method}'")

        # 3. Save Formatted Data
        # Note: save_data needs to handle the potential formats returned by formatters
        save_data(data_to_save, output_path)

        click.echo(f"Successfully saved dataset from '{input_path.name}' to '{output_path.name}' (assembly: {assembly_method}).")

    except FileNotFoundError:
        raise click.UsageError(f"Input file not found: {input_path}")
    except ValueError as e:
        raise click.UsageError(f"Error during dataset saving: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during dataset saving: {e}", exc_info=True)
        raise click.Abort(f"An unexpected error occurred: {e}")
