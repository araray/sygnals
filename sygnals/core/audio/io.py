# sygnals/core/audio/io.py

"""
Handles loading and saving of audio files using libraries like librosa and soundfile.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

SUPPORTED_READ_FORMATS = sf.available_formats()
SUPPORTED_WRITE_FORMATS = sf.available_formats() # Check subtypes later if needed

def load_audio(
    file_path: Path,
    sr: Optional[int] = None,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None
) -> Tuple[NDArray[np.float64], int]:
    """
    Loads an audio file using librosa.

    Args:
        file_path: Path to the audio file.
        sr: Target sampling rate. If None, uses the native sampling rate.
        mono: Convert signal to mono.
        offset: Start reading after this time (in seconds).
        duration: Only load up to this much audio (in seconds).

    Returns:
        A tuple containing:
        - data (NDArray[np.float64]): The audio time series.
        - sample_rate (int): The sampling rate of the audio time series.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For librosa loading errors.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio input file not found: {file_path}")

    logger.info(f"Loading audio from: {file_path} (sr={sr}, mono={mono}, offset={offset}, duration={duration})")
    try:
        # Librosa handles various formats using soundfile and potentially audioread
        data, sample_rate = librosa.load(
            file_path,
            sr=sr,
            mono=mono,
            offset=offset,
            duration=duration,
            res_type='kaiser_best' # Default librosa resample quality
        )
        # Ensure data is float64 as expected by downstream functions
        return data.astype(np.float64, copy=False), sample_rate
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise

def save_audio(
    data: NDArray[np.float64],
    sr: int,
    output_path: Path,
    subtype: Optional[str] = 'PCM_16' # Common default subtype
):
    """
    Saves audio data to a file using soundfile.

    Determines format from the output file extension.

    Args:
        data: Audio time series data (float64). Should be in range [-1.0, 1.0] for PCM formats.
        sr: Sampling rate.
        output_path: Path to save the audio file.
        subtype: Soundfile subtype string (e.g., 'PCM_16', 'FLOAT', 'VORBIS').
                 If None, soundfile chooses a default based on format.

    Raises:
        ValueError: If the output format or subtype is invalid.
        Exception: For soundfile writing errors.
    """
    logger.info(f"Saving audio to: {output_path} (sr={sr}, subtype={subtype})")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract format from extension
    file_format = output_path.suffix[1:].upper() # e.g., 'WAV', 'FLAC'

    if file_format not in SUPPORTED_WRITE_FORMATS:
         raise ValueError(f"Unsupported audio output format: '{file_format}'. "
                          f"Supported formats: {list(SUPPORTED_WRITE_FORMATS.keys())}")

    # Normalize data for PCM formats if necessary (simple clipping for now)
    if subtype and 'PCM' in subtype:
        if np.max(np.abs(data)) > 1.0:
            logger.warning(f"Audio data exceeds range [-1, 1] for PCM subtype '{subtype}'. Clipping data.")
            data = np.clip(data, -1.0, 1.0)

    try:
        sf.write(output_path, data, sr, subtype=subtype, format=file_format)
        logger.info(f"Audio successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving audio file {output_path}: {e}")
        raise
