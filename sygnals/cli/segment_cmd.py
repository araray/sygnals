# sygnals/cli/segment_cmd.py

"""
CLI command for segmenting signals.
"""

import logging
import click
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Any

# Import core components
from sygnals.core.data_handler import read_data, save_data, ReadResult
from sygnals.core.segmentation import segment_fixed_length # Add other methods later
from sygnals.config.models import SygnalsConfig # For accessing config if needed

logger = logging.getLogger(__name__)

# --- Helper to save segments ---
def _save_segments(
    segments: List[NDArray[np.float64]],
    sr: int,
    output_path: Path,
    base_filename: str
):
    """Saves a list of signal segments to individual files."""
    output_path.mkdir(parents=True, exist_ok=True)
    num_digits = len(str(len(segments))) # For zero-padding filenames
    saved_count = 0
    for i, seg_data in enumerate(segments):
        # Construct filename (e.g., base_segment_001.wav)
        segment_filename = f"{base_filename}_segment_{i+1:0{num_digits}d}.wav"
        segment_output_path = output_path / segment_filename
        try:
            # Save as WAV file for now
            save_data((seg_data, sr), segment_output_path, audio_subtype='PCM_16')
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save segment {i+1} to {segment_output_path}: {e}")
    logger.info(f"Saved {saved_count} segments to directory: {output_path}")


# --- Main Segment Command Group ---
@click.group("segment")
@click.pass_context
def segment_cmd(ctx):
    """Segment signals using various methods."""
    # Access config if needed: ctx.obj['config']
    pass


# --- Fixed-Length Segmentation Subcommand ---
@segment_cmd.command("fixed-length")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option("-o", "--output", type=click.Path(resolve_path=True), required=True,
              help="Output directory to save segmented files.")
@click.option("--length", type=float, required=True, help="Segment length in seconds.")
@click.option("--overlap", type=float, default=0.0, show_default=True,
              help="Overlap ratio between segments (0.0 to < 1.0).")
@click.option("--pad/--no-pad", is_flag=True, default=True, show_default=True,
              help="Pad the last segment if shorter than 'length'.")
@click.option("--min-length", type=float, default=None,
              help="Minimum segment length in seconds to keep (useful with --pad).")
@click.pass_context
def segment_fixed_length_cmd(
    ctx,
    input_file: str,
    output: str,
    length: float,
    overlap: float,
    pad: bool,
    min_length: Optional[float]
):
    """Segment signal into fixed-length chunks."""
    input_path = Path(input_file)
    output_path = Path(output)
    logger.info(f"Running fixed-length segmentation on: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Params: length={length}s, overlap={overlap}, pad={pad}, min_length={min_length}s")

    try:
        # 1. Read Input Data (only audio makes sense for segmentation)
        data_result: ReadResult = read_data(input_path)
        if not isinstance(data_result, tuple) or len(data_result) != 2:
             raise click.UsageError(f"Input file '{input_path.name}' is not recognized as audio. Segmentation requires audio input.")
        signal_data, sr = data_result
        if signal_data.ndim != 1:
             logger.warning("Input audio is multi-channel. Converting to mono by averaging for segmentation.")
             signal_data = np.mean(signal_data, axis=0)

        # 2. Perform Segmentation
        segments: List[NDArray[np.float64]] = segment_fixed_length(
            y=signal_data,
            sr=sr,
            segment_length_sec=length,
            overlap_ratio=overlap,
            pad=pad,
            min_segment_length_sec=min_length
        )

        if not segments:
             logger.warning("Segmentation resulted in zero segments.")
             click.echo("Warning: No segments generated.")
             return # Exit cleanly

        # 3. Save Segments
        base_filename = input_path.stem # Use input filename stem for output segments
        _save_segments(segments, sr, output_path, base_filename)

        click.echo(f"Successfully segmented '{input_path.name}' into {len(segments)} segments in '{output_path}'.")

    except FileNotFoundError:
        raise click.UsageError(f"Input file not found: {input_path}")
    except ValueError as e:
        # Catch specific value errors from core functions
        raise click.UsageError(f"Error during segmentation: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during segmentation: {e}", exc_info=True)
        # Use click.Abort for unexpected errors
        raise click.Abort(f"An unexpected error occurred: {e}")


# --- Add other segmentation subcommands (silence, event) later ---
# @segment_cmd.command("by-silence")
# ...
#
# @segment_cmd.command("by-event")
# ...
