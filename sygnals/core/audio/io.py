# sygnals/core/audio/io.py

"""
Handles loading and saving of audio files using libraries like librosa and soundfile.
"""

import logging
from pathlib import Path
# Import necessary types
from typing import Tuple, Optional

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Get available formats from soundfile
# Store them in a way that's easy to check against extensions
_available_read_formats = sf.available_formats()
_available_write_formats = sf.available_formats() # Write formats might have subtype restrictions

# Create sets of supported extensions (lowercase, including dot)
SUPPORTED_READ_EXTENSIONS = {f".{fmt.lower()}" for fmt in _available_read_formats}
# Add common formats librosa might handle via other backends (like audioread for mp3)
SUPPORTED_READ_EXTENSIONS.add(".mp3")

SUPPORTED_WRITE_EXTENSIONS = {f".{fmt.lower()}" for fmt in _available_write_formats}
# Filter write formats based on common usage or known soundfile support
# (e.g., MP3 writing often requires external libraries/is not standard in sf)
SUPPORTED_WRITE_EXTENSIONS -= {".mp3"} # Example: Explicitly remove mp3 if not reliably supported

logger.debug(f"Supported audio read extensions: {SUPPORTED_READ_EXTENSIONS}")
logger.debug(f"Supported audio write extensions: {SUPPORTED_WRITE_EXTENSIONS}")


def load_audio(
    file_path: Path,
    sr: Optional[int] = None,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None
) -> Tuple[NDArray[np.float64], int]:
    """
    Loads an audio file using librosa.

    Librosa uses soundfile and potentially other backends (like audioread)
    to load various audio formats.

    Args:
        file_path: Path object for the audio file.
        sr: Target sampling rate. If None, uses the native sampling rate.
            Resampling is performed if the native rate differs from the target.
        mono: If True, convert signal to mono by averaging channels.
        offset: Start reading after this time (in seconds) from the beginning
                of the file.
        duration: Only load up to this much audio (in seconds). Reads from the
                  `offset` if specified.

    Returns:
        A tuple containing:
        - data (NDArray[np.float64]): The audio time series as a NumPy array.
                                      Shape is (n_samples,) if mono=True,
                                      or (n_channels, n_samples) if mono=False.
                                      Values are typically normalized to [-1.0, 1.0].
        - sample_rate (int): The sampling rate of the loaded audio time series
                             (will match `sr` if resampling occurred).

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For librosa/soundfile/audioread loading errors (e.g., unsupported
                   format, corrupted file).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio input file not found: {file_path}")
    if not file_path.is_file():
         raise ValueError(f"Input path is not a file: {file_path}")

    logger.info(f"Loading audio from: {file_path} (sr={sr}, mono={mono}, offset={offset}, duration={duration})")
    try:
        # Librosa handles various formats using soundfile and potentially audioread
        # res_type specifies the resampling quality if sr is different from native
        data, sample_rate = librosa.load(
            file_path,
            sr=sr,
            mono=mono,
            offset=offset,
            duration=duration,
            res_type='kaiser_best' # High-quality resampling method
        )
        # Ensure data is float64 as expected by downstream functions
        # Librosa usually returns float32, so conversion is often needed
        if data.dtype != np.float64:
            data = data.astype(np.float64)

        logger.debug(f"Audio loaded successfully. Shape: {data.shape}, SR: {sample_rate}")
        return data, sample_rate
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        # Re-raise the exception to be handled upstream
        raise


def save_audio(
    data: NDArray[np.float64],
    sr: int,
    output_path: Path,
    subtype: Optional[str] = 'PCM_16' # Common default subtype for WAV
):
    """
    Saves audio data to a file using soundfile.

    Determines format from the output file extension. Data should be float64.
    For PCM subtypes, data should ideally be within [-1.0, 1.0] to avoid clipping.

    Args:
        data: Audio time series data (float64 NumPy array).
              Shape (n_samples,) for mono or (n_channels, n_samples) for multi-channel.
              Soundfile expects (n_samples, n_channels) for multi-channel, so transpose if needed.
        sr: Sampling rate (integer, Hz).
        output_path: Path object where to save the audio file.
        subtype: Soundfile subtype string (e.g., 'PCM_16', 'FLOAT', 'VORBIS').
                 Determines the bit depth and encoding. See `soundfile.available_subtypes()`.
                 If None, soundfile chooses a default based on the format.

    Raises:
        ValueError: If the output format (from extension) or subtype is invalid/unsupported,
                    or if data shape is incorrect for multi-channel.
        Exception: For soundfile writing errors (e.g., permission denied).
    """
    logger.info(f"Saving audio to: {output_path} (sr={sr}, subtype={subtype})")

    # Validate output path extension
    ext = output_path.suffix.lower()
    if ext not in SUPPORTED_WRITE_EXTENSIONS:
         raise ValueError(f"Unsupported audio output extension: '{ext}'. "
                          f"Supported extensions: {SUPPORTED_WRITE_EXTENSIONS}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data: soundfile expects shape (n_samples, n_channels)
    if data.ndim == 1:
        # Mono: shape is already correct implicitly (n_samples,)
        pass
    elif data.ndim == 2:
        # Multi-channel: librosa often returns (n_channels, n_samples)
        # Transpose if necessary
        if data.shape[0] < data.shape[1]: # Assume channels dimension is smaller
            logger.debug(f"Transposing multi-channel data from {data.shape} to {(data.shape[1], data.shape[0])} for soundfile.")
            data = data.T
        # Now shape should be (n_samples, n_channels)
    else:
        raise ValueError(f"Input data must be 1D (mono) or 2D (multi-channel), got shape {data.shape}")

    # Check data range for PCM formats and warn/clip if necessary
    if subtype and 'PCM' in subtype:
        max_abs_val = np.max(np.abs(data))
        if max_abs_val > 1.0:
            logger.warning(f"Audio data exceeds range [-1, 1] (max abs: {max_abs_val:.4f}) "
                           f"for PCM subtype '{subtype}'. Clipping data to prevent wrap-around.")
            data = np.clip(data, -1.0, 1.0)

    # Extract format string expected by soundfile (e.g., 'WAV', 'FLAC')
    file_format = ext[1:].upper()

    try:
        # Write the file using soundfile
        sf.write(output_path, data, sr, subtype=subtype, format=file_format)
        logger.info(f"Audio successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving audio file {output_path}: {e}")
        # Re-raise the exception
        raise
