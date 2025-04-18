# sygnals/core/audio/features.py

"""
Functions for extracting audio-specific features.
Leverages libraries like librosa.
"""

import logging
import librosa
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# --- Basic Features ---

def zero_crossing_rate(
    y: NDArray[np.float64],
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True
) -> NDArray[np.float64]:
    """
    Computes the zero-crossing rate (ZCR).

    Args:
        y: Audio time series (float64).
        frame_length: Length of the frame over which to compute ZCR.
        hop_length: Number of samples to advance between frames.
        center: If True, pad `y` to center frames. Affects the time alignment.

    Returns:
        Zero crossing rate for each frame (1D array, float64).
    """
    logger.debug(f"Calculating Zero Crossing Rate: frame={frame_length}, hop={hop_length}")
    try:
        zcr = librosa.feature.zero_crossing_rate(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length,
            center=center
        )
        # librosa returns shape (1, N), flatten to 1D
        return zcr[0].astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error calculating Zero Crossing Rate: {e}")
        raise

def rms_energy(
    y: NDArray[np.float64],
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True
) -> NDArray[np.float64]:
    """
    Computes the Root Mean Square (RMS) energy for each frame.

    Args:
        y: Audio time series (float64).
        frame_length: Length of the frame over which to compute RMS.
        hop_length: Number of samples to advance between frames.
        center: If True, pad `y` to center frames.

    Returns:
        RMS energy for each frame (1D array, float64).
    """
    logger.debug(f"Calculating RMS Energy: frame={frame_length}, hop={hop_length}")
    try:
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length,
            center=center
        )
        # librosa returns shape (1, N), flatten to 1D
        return rms[0].astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error calculating RMS Energy: {e}")
        raise

# --- Pitch Features ---

def fundamental_frequency(
    y: NDArray[np.float64],
    sr: int,
    fmin: float = librosa.note_to_hz('C2'),
    fmax: float = librosa.note_to_hz('C7'),
    method: str = 'pyin' # 'pyin' or 'yin'
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimates the fundamental frequency (pitch) using pYIN or YIN algorithm.

    Args:
        y: Audio time series (float64).
        sr: Sampling rate.
        fmin: Minimum frequency estimate (Hz).
        fmax: Maximum frequency estimate (Hz).
        method: Pitch estimation algorithm ('pyin' or 'yin').

    Returns:
        A tuple containing:
        - times (NDArray[np.float64]): Time points for each estimate.
        - f0 (NDArray[np.float64]): Fundamental frequency estimates (NaN for unvoiced).
        - voiced_flag (NDArray[np.float64]): Boolean flag indicating if frame is voiced.
        - voiced_probs (NDArray[np.float64]): Voicing probability estimates.
    """
    logger.debug(f"Estimating Fundamental Frequency (Pitch) using {method}: fmin={fmin}, fmax={fmax}")
    try:
        if method == 'pyin':
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y=y,
                fmin=fmin,
                fmax=fmax,
                sr=sr
            )
        elif method == 'yin':
             f0 = librosa.yin(
                 y=y,
                 fmin=fmin,
                 fmax=fmax,
                 sr=sr
             )
             # YIN doesn't directly provide voiced_flag/probs, can estimate based on thresholding
             # For simplicity, return dummy arrays for now, or implement thresholding
             voiced_flag = np.isfinite(f0) # Simple estimate: voiced if f0 is not NaN
             voiced_probs = voiced_flag.astype(float) # Simple estimate
        else:
             raise ValueError(f"Unsupported pitch estimation method: {method}")

        # Calculate corresponding time points based on default hop length used by pyin/yin
        # (Typically frame_length=2048, hop_length=512 for pyin)
        # This might need adjustment if librosa changes defaults
        hop_length = 512 # Assume default hop_length
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

        # Ensure float64 output
        return (
            times.astype(np.float64, copy=False),
            f0.astype(np.float64, copy=False),
            voiced_flag.astype(np.float64, copy=False), # Store bools as float 0/1
            voiced_probs.astype(np.float64, copy=False)
        )
    except Exception as e:
        logger.error(f"Error estimating fundamental frequency: {e}")
        raise

# --- Other Audio Metrics (from old audio_handler) ---

def get_basic_audio_metrics(y: NDArray[np.float64], sr: int) -> Dict[str, Any]:
    """
    Calculates basic summary metrics for the entire audio signal.

    Args:
        y: Audio time series (float64).
        sr: Sampling rate.

    Returns:
        Dictionary containing 'duration_seconds', 'rms_global', 'peak_amplitude'.
    """
    logger.debug("Calculating basic global audio metrics.")
    try:
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        # Calculate global RMS (note: different from frame-based RMS energy)
        rms_global = np.sqrt(np.mean(y**2))
        peak_amplitude = np.max(np.abs(y))

        return {
            "duration_seconds": float(duration_seconds),
            "rms_global": float(rms_global),
            "peak_amplitude": float(peak_amplitude),
        }
    except Exception as e:
        logger.error(f"Error calculating basic audio metrics: {e}")
        raise

# --- Add more features as needed: HNR, Jitter, Shimmer, Onset Detection ---
