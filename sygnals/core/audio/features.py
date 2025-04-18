# sygnals/core/audio/features.py

"""
Functions for extracting audio-specific features.
Leverages libraries like librosa.
"""

import logging
import librosa
import numpy as np
from numpy.typing import NDArray
# Import necessary types
from typing import Optional, Dict, Any, Tuple

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

    The rate at which the signal changes sign. It's often used as a measure
    of the noisiness of a signal or the presence of high-frequency content.

    Args:
        y: Audio time series (float64).
        frame_length: Length of the frame over which to compute ZCR.
        hop_length: Number of samples to advance between frames.
        center: If True, pad `y` to center frames. Affects the time alignment.

    Returns:
        Zero crossing rate for each frame (1D array, float64). Values are typically
        between 0 and 1, representing the proportion of zero crossings within the frame.
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

    RMS energy is a measure of the signal's amplitude or loudness over short windows.

    Args:
        y: Audio time series (float64).
        frame_length: Length of the frame over which to compute RMS.
        hop_length: Number of samples to advance between frames.
        center: If True, pad `y` to center frames.

    Returns:
        RMS energy for each frame (1D array, float64). Values are non-negative.
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
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimates the fundamental frequency (pitch) using pYIN or YIN algorithm.

    Pitch is the perceived fundamental frequency of a sound.

    Args:
        y: Audio time series (float64).
        sr: Sampling rate.
        fmin: Minimum frequency estimate (Hz). Defaults to C2 (~65 Hz).
        fmax: Maximum frequency estimate (Hz). Defaults to C7 (~2093 Hz).
        method: Pitch estimation algorithm ('pyin' or 'yin'). 'pyin' also provides
                voicing probability.

    Returns:
        A tuple containing:
        - times (NDArray[np.float64]): Time points for each estimate (center of frame).
        - f0 (NDArray[np.float64]): Fundamental frequency estimates in Hz. NaN for unvoiced frames.
        - voiced_flag (NDArray[np.float64]): Boolean flag indicating if frame is voiced (1.0 if voiced, 0.0 otherwise).
        - voiced_probs (NDArray[np.float64]): Voicing probability estimates (only for 'pyin', otherwise same as voiced_flag).
    """
    logger.debug(f"Estimating Fundamental Frequency (Pitch) using {method}: fmin={fmin}, fmax={fmax}")
    try:
        if method == 'pyin':
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y=y,
                fmin=fmin,
                fmax=fmax,
                sr=sr
                # Default frame/hop lengths are used by pyin internally
            )
        elif method == 'yin':
             f0 = librosa.yin(
                 y=y,
                 fmin=fmin,
                 fmax=fmax,
                 sr=sr
                 # Default frame/hop lengths are used by yin internally
             )
             # YIN doesn't directly provide voiced_flag/probs. Estimate based on NaN.
             voiced_flag = np.isfinite(f0) # Simple estimate: voiced if f0 is not NaN
             voiced_probs = voiced_flag.astype(float) # Simple probability estimate
        else:
             raise ValueError(f"Unsupported pitch estimation method: {method}. Choose 'pyin' or 'yin'.")

        # Calculate corresponding time points based on default hop length used by pyin/yin
        # (Typically frame_length=2048, hop_length=512 for pyin)
        # This might need adjustment if librosa changes defaults or if frame/hop are passed explicitly
        hop_length = 512 # Assume default librosa hop_length for pitch estimation
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

        # Ensure float64 output and handle potential NaNs in f0
        f0_out = np.nan_to_num(f0, nan=0.0).astype(np.float64, copy=False) # Replace NaN with 0 for consistency
        voiced_flag_out = voiced_flag.astype(np.float64, copy=False) # Store bools as float 0/1
        voiced_probs_out = voiced_probs.astype(np.float64, copy=False)
        times_out = times.astype(np.float64, copy=False)

        return (times_out, f0_out, voiced_flag_out, voiced_probs_out)
    except Exception as e:
        logger.error(f"Error estimating fundamental frequency: {e}")
        raise

# --- Voice Quality Features (Placeholders) ---

def harmonic_to_noise_ratio(
    y: NDArray[np.float64],
    sr: int,
    frame_length: int = 2048, # Parameters might be needed for actual implementation
    hop_length: int = 512,
    **kwargs: Any
) -> NDArray[np.float64]:
    """
    Estimates the Harmonic-to-Noise Ratio (HNR) [Placeholder].

    HNR measures the ratio of harmonic energy (related to periodic, pitched sound)
    to noise energy (related to aperiodic or random components) in the signal,
    often used as an indicator of voice quality (e.g., hoarseness).

    NOTE: This is a placeholder implementation and returns an array of NaNs.
          A real implementation would likely involve autocorrelation or cepstral analysis.

    Args:
        y: Audio time series (float64).
        sr: Sampling rate.
        frame_length: Analysis frame length (samples).
        hop_length: Hop length between frames (samples).
        **kwargs: Additional parameters for future implementation.

    Returns:
        An array of NaNs with the expected length based on framing.
    """
    logger.warning("harmonic_to_noise_ratio feature is a placeholder and returns NaN.")
    # Calculate expected number of frames to return NaN array of correct size
    num_frames = len(librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)[0])
    return np.full(num_frames, np.nan, dtype=np.float64)

def jitter(
    y: NDArray[np.float64],
    sr: int,
    f0: Optional[NDArray[np.float64]] = None, # Requires fundamental frequency estimates
    method: str = 'local', # Common methods: 'local', 'rap', 'ppq5'
    **kwargs: Any
) -> NDArray[np.float64]:
    """
    Estimates the frequency jitter (periodicity variations) [Placeholder].

    Jitter refers to the short-term variations in the fundamental frequency (F0)
    of voiced speech. It's often used as a measure of voice instability or roughness.
    Calculation typically requires accurate F0 estimates.

    NOTE: This is a placeholder implementation and returns an array of NaNs.
          A real implementation requires F0 tracking and period difference calculations.

    Args:
        y: Audio time series (float64).
        sr: Sampling rate.
        f0: Optional pre-computed fundamental frequency contour (Hz). If None,
            it would need to be calculated internally (not done in placeholder).
        method: Jitter calculation method (e.g., 'local', 'rap').
        **kwargs: Additional parameters for future implementation.

    Returns:
        An array of NaNs with the expected length based on F0 contour or framing.
    """
    logger.warning("jitter feature is a placeholder and returns NaN.")
    # Determine output length based on f0 if provided, otherwise based on standard framing
    if f0 is not None:
        num_frames = len(f0)
    else:
        # Estimate based on default framing if f0 is not available
        frame_length = kwargs.get('frame_length', 2048)
        hop_length = kwargs.get('hop_length', 512)
        num_frames = len(librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)[0])
    return np.full(num_frames, np.nan, dtype=np.float64)

def shimmer(
    y: NDArray[np.float64],
    sr: int,
    frame_length: int = 2048, # Parameters might be needed for actual implementation
    hop_length: int = 512,
    method: str = 'local_db', # Common methods: 'local', 'local_db', 'apq11'
    **kwargs: Any
) -> NDArray[np.float64]:
    """
    Estimates the amplitude shimmer (amplitude variations) [Placeholder].

    Shimmer refers to the short-term variations in the amplitude of voiced speech
    peaks. It's another measure often associated with voice quality and perceived roughness.

    NOTE: This is a placeholder implementation and returns an array of NaNs.
          A real implementation requires peak amplitude tracking within voiced segments.

    Args:
        y: Audio time series (float64).
        sr: Sampling rate.
        frame_length: Analysis frame length (samples).
        hop_length: Hop length between frames (samples).
        method: Shimmer calculation method (e.g., 'local', 'local_db').
        **kwargs: Additional parameters for future implementation.

    Returns:
        An array of NaNs with the expected length based on framing.
    """
    logger.warning("shimmer feature is a placeholder and returns NaN.")
    # Calculate expected number of frames to return NaN array of correct size
    num_frames = len(librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)[0])
    return np.full(num_frames, np.nan, dtype=np.float64)


# --- Other Audio Metrics (from old audio_handler) ---

def get_basic_audio_metrics(y: NDArray[np.float64], sr: int) -> Dict[str, Any]:
    """
    Calculates basic summary metrics for the entire audio signal.

    Provides high-level information about the audio duration and overall amplitude levels.

    Args:
        y: Audio time series (float64).
        sr: Sampling rate.

    Returns:
        Dictionary containing:
        - 'duration_seconds': Total duration of the audio in seconds.
        - 'rms_global': Root Mean Square value calculated over the entire signal.
        - 'peak_amplitude': Maximum absolute amplitude value in the signal.
    """
    logger.debug("Calculating basic global audio metrics.")
    try:
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        # Calculate global RMS (note: different from frame-based RMS energy)
        rms_global = np.sqrt(np.mean(y**2)) if y.size > 0 else 0.0
        peak_amplitude = np.max(np.abs(y)) if y.size > 0 else 0.0

        return {
            "duration_seconds": float(duration_seconds),
            "rms_global": float(rms_global),
            "peak_amplitude": float(peak_amplitude),
        }
    except Exception as e:
        logger.error(f"Error calculating basic audio metrics: {e}")
        raise

# --- Add more features as needed: Onset Detection ---
# Example: Onset Detection (using librosa)
def detect_onsets(
    y: NDArray[np.float64],
    sr: int,
    **kwargs: Any # Pass additional args to librosa.onset.onset_detect
) -> NDArray[np.int64]:
    """
    Detects note onset events (start times) in an audio signal.

    Args:
        y: Audio time series (float64).
        sr: Sampling rate.
        **kwargs: Additional keyword arguments passed to `librosa.onset.onset_detect`,
                  such as `hop_length`, `backtrack`, `units`.

    Returns:
        Array containing the frame indices of detected onsets. Use `librosa.frames_to_time`
        to convert these to seconds if needed.
    """
    logger.debug(f"Detecting onsets with parameters: {kwargs}")
    try:
        # Default units='frames'
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, **kwargs)
        return onset_frames.astype(np.int64, copy=False) # Return frame indices as integers
    except Exception as e:
        logger.error(f"Error detecting onsets: {e}")
        raise
