# sygnals/cli/augment_cmd.py

"""
CLI command for applying data augmentation techniques to signals.
"""

import logging
import click
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Any

# Import core components
from sygnals.core.data_handler import read_data, save_data, ReadResult
from sygnals.core.augment import add_noise, pitch_shift, time_stretch
from sygnals.config.models import SygnalsConfig # For accessing config if needed

logger = logging.getLogger(__name__)

# --- Main Augment Command Group ---
@click.group("augment")
@click.pass_context
def augment_cmd(ctx):
    """Apply data augmentation techniques to audio signals."""
    # Access config if needed: ctx.obj['config']
    pass

# --- Common Options ---
input_option = click.argument("input_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
output_option = click.option("-o", "--output", type=click.Path(resolve_path=True), required=True,
                             help="Output file path for the augmented audio.")

# --- Add Noise Subcommand ---
@augment_cmd.command("add-noise")
@input_option
@output_option
@click.option("--snr", type=float, required=True, help="Target Signal-to-Noise Ratio (SNR) in dB.")
@click.option("--noise-type", type=click.Choice(['gaussian', 'white', 'pink', 'brown'], case_sensitive=False),
              default='gaussian', show_default=True, help="Type of noise to add.")
@click.option("--seed", type=int, default=None, help="Random seed for noise generation.")
@click.pass_context
def augment_add_noise(ctx, input_file: str, output: str, snr: float, noise_type: str, seed: Optional[int]):
    """Add noise to the audio signal."""
    input_path = Path(input_file)
    output_path = Path(output)
    logger.info(f"Running 'add-noise' augmentation on: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Params: snr_db={snr}, noise_type='{noise_type}', seed={seed}")

    try:
        # 1. Read Input Audio
        data_result: ReadResult = read_data(input_path)
        if not isinstance(data_result, tuple) or len(data_result) != 2:
             raise click.UsageError(f"Input file '{input_path.name}' is not recognized as audio.")
        signal_data, sr = data_result
        if signal_data.ndim != 1:
             logger.warning("Input audio is multi-channel. Converting to mono by averaging before adding noise.")
             signal_data = np.mean(signal_data, axis=0)

        # 2. Apply Augmentation
        augmented_signal = add_noise(
            y=signal_data,
            snr_db=snr,
            noise_type=noise_type, # type: ignore # click.Choice ensures valid type
            seed=seed
        )

        # 3. Save Augmented Audio
        # Use default subtype or allow override via config/option later
        save_data((augmented_signal, sr), output_path)

        click.echo(f"Successfully applied '{noise_type}' noise (SNR={snr} dB) to '{input_path.name}', saved to '{output_path.name}'.")

    except FileNotFoundError:
        raise click.UsageError(f"Input file not found: {input_path}")
    except ValueError as e:
        raise click.UsageError(f"Error during noise addition: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during add-noise augmentation: {e}", exc_info=True)
        raise click.Abort(f"An unexpected error occurred: {e}")


# --- Pitch Shift Subcommand ---
@augment_cmd.command("pitch-shift")
@input_option
@output_option
@click.option("--steps", type=float, required=True, help="Number of semitones to shift (positive or negative).")
@click.option("--bins-per-octave", type=int, default=12, show_default=True, help="Number of bins per octave.")
@click.pass_context
def augment_pitch_shift(ctx, input_file: str, output: str, steps: float, bins_per_octave: int):
    """Shift the pitch of the audio signal."""
    input_path = Path(input_file)
    output_path = Path(output)
    logger.info(f"Running 'pitch-shift' augmentation on: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Params: n_steps={steps}, bins_per_octave={bins_per_octave}")

    try:
        # 1. Read Input Audio
        data_result: ReadResult = read_data(input_path)
        if not isinstance(data_result, tuple) or len(data_result) != 2:
             raise click.UsageError(f"Input file '{input_path.name}' is not recognized as audio.")
        signal_data, sr = data_result
        # Pitch shift works on mono or multi-channel handled by librosa

        # 2. Apply Augmentation
        augmented_signal = pitch_shift(
            y=signal_data,
            sr=sr,
            n_steps=steps,
            bins_per_octave=bins_per_octave
        )

        # 3. Save Augmented Audio
        save_data((augmented_signal, sr), output_path)

        click.echo(f"Successfully applied pitch shift (steps={steps}) to '{input_path.name}', saved to '{output_path.name}'.")

    except FileNotFoundError:
        raise click.UsageError(f"Input file not found: {input_path}")
    except ValueError as e:
        raise click.UsageError(f"Error during pitch shifting: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during pitch-shift augmentation: {e}", exc_info=True)
        raise click.Abort(f"An unexpected error occurred: {e}")


# --- Time Stretch Subcommand ---
@augment_cmd.command("time-stretch")
@input_option
@output_option
@click.option("--rate", type=float, required=True, help="Factor to stretch time (>1 speeds up, <1 slows down).")
@click.pass_context
def augment_time_stretch(ctx, input_file: str, output: str, rate: float):
    """Stretch the time duration of the audio signal without changing pitch."""
    input_path = Path(input_file)
    output_path = Path(output)
    logger.info(f"Running 'time-stretch' augmentation on: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Params: rate={rate}")

    if rate <= 0:
         raise click.UsageError("Stretch rate must be positive.")

    try:
        # 1. Read Input Audio
        data_result: ReadResult = read_data(input_path)
        if not isinstance(data_result, tuple) or len(data_result) != 2:
             raise click.UsageError(f"Input file '{input_path.name}' is not recognized as audio.")
        signal_data, sr = data_result
        # Time stretch works on mono or multi-channel handled by librosa

        # 2. Apply Augmentation
        augmented_signal = time_stretch(
            y=signal_data,
            rate=rate
        )

        # 3. Save Augmented Audio
        save_data((augmented_signal, sr), output_path)

        click.echo(f"Successfully applied time stretch (rate={rate}) to '{input_path.name}', saved to '{output_path.name}'.")

    except FileNotFoundError:
        raise click.UsageError(f"Input file not found: {input_path}")
    except ValueError as e:
        raise click.UsageError(f"Error during time stretching: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during time-stretch augmentation: {e}", exc_info=True)
        raise click.Abort(f"An unexpected error occurred: {e}")
