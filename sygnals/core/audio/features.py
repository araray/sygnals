# sygnals/core/audio/features.py

"""
Functions for extracting audio-specific features.

Leverages libraries like librosa for common audio features like ZCR, RMS, pitch,
and onset detection. Includes approximate implementations for more complex
voice quality features (HNR, Jitter, Shimmer).
"""

import logging
import librosa
import numpy as np
from numpy.typing import NDArray
# Import necessary types
from typing import Optional, Dict, Any, Tuple, Literal, Union
import warnings # Import warnings for placeholder features

logger = logging.getLogger(__name__)

# Define a small epsilon for safe division and log calculations
_EPSILON = np.finfo(np.float64).eps

# --- Basic Frame-Based Features ---

def zero_crossing_rate(
    y: NDArray[np.float64],
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    **kwargs: Any # Allow additional args for librosa compatibility if needed
) -> NDArray[np.float64]:
    """
    Computes the zero-crossing rate (ZCR) for each frame using librosa.

    The ZCR is the rate at which the signal changes sign within each frame.
    It's often used as a simple measure of the noisiness of a signal or the
    presence of high-frequency content (characteristic of fricatives in speech).

    Args:
        y: Audio time series (1D float64).
        frame_length: Length of the frame over which to compute ZCR (samples).
        hop_length: Number of samples to advance between frames.
        center: If True, pad `y` to center frames. Affects the time alignment
                and number of frames. (Default: True)
        **kwargs: Additional keyword arguments passed to `librosa.feature.zero_crossing_rate`.

    Returns:
        Zero crossing rate for each frame (1D array, float64). Values are typically
        between 0 and 1, representing the proportion of zero crossings within the frame.

    Raises:
        ValueError: If input `y` is not 1D.
        Exception: For errors during librosa calculation.
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    logger.debug(f"Calculating Zero Crossing Rate: frame={frame_length}, hop={hop_length}, center={center}")
    try:
        zcr = librosa.feature.zero_crossing_rate(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length,
            center=center,
            **kwargs
        )
        # librosa returns shape (1, N), flatten to 1D
        return zcr[0].astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error calculating Zero Crossing Rate: {e}")
        raise

def rms_energy(
    y: Optional[NDArray[np.float64]] = None,
    *, # Force keyword arguments after y
    S: Optional[NDArray[np.float64]] = None, # Can compute from signal y or spectrogram S
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: str = 'constant',
    **kwargs: Any # Allow additional args for librosa compatibility
) -> NDArray[np.float64]:
    """
    Computes the Root Mean Square (RMS) energy for each frame using librosa.

    RMS energy is a measure of the signal's amplitude or loudness over short windows.
    It can be computed either from the time-domain signal `y` or from a
    pre-computed magnitude spectrogram `S`.

    Args:
        y: Audio time series (1D float64). Required if `S` is not provided.
        S: Magnitude spectrogram (non-negative values). If provided, `y` is ignored.
           Shape: (n_freq_bins, n_frames).
        frame_length: Length of the frame over which to compute RMS (samples).
                      Used if calculating from `y`. (Default: 2048)
        hop_length: Number of samples to advance between frames. Used if calculating
                    from `y`. (Default: 512)
        center: If True and calculating from `y`, pad `y` to center frames. (Default: True)
        pad_mode: Padding mode if `center=True` and calculating from `y`. (Default: 'constant')
        **kwargs: Additional keyword arguments passed to `librosa.feature.rms`.

    Returns:
        RMS energy for each frame (1D array, float64). Values are non-negative.

    Raises:
        ValueError: If input `y` is not 1D when used, or if neither `y` nor `S` is provided.
        Exception: For errors during librosa calculation.
    """
    if S is None and y is None:
        raise ValueError("Either audio time series 'y' or magnitude spectrogram 'S' must be provided.")
    if y is not None and y.ndim != 1:
        raise ValueError("Input audio data 'y' must be a 1D array.")
    if S is not None and S.ndim != 2:
        raise ValueError("Input spectrogram 'S' must be a 2D array.")

    logger.debug(f"Calculating RMS Energy: frame={frame_length}, hop={hop_length}, center={center}")
    try:
        rms = librosa.feature.rms(
            y=y,
            S=S,
            frame_length=frame_length,
            hop_length=hop_length,
            center=center,
            pad_mode=pad_mode,
            **kwargs
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
    fmin: Optional[float] = None, # Use librosa default if None
    fmax: Optional[float] = None, # Use librosa default if None
    method: Literal['pyin', 'yin'] = 'pyin',
    hop_length: Optional[int] = None, # Allow overriding default hop
    **kwargs: Any # Allow additional args for librosa pitch functions
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimates the fundamental frequency (pitch) using pYIN or YIN algorithm via librosa.

    Pitch is the perceived fundamental frequency of a sound, crucial for speech and music analysis.

    Args:
        y: Audio time series (1D float64).
        sr: Sampling rate (Hz).
        fmin: Minimum frequency estimate (Hz). Defaults to librosa's internal default (~30-60 Hz).
        fmax: Maximum frequency estimate (Hz). Defaults to librosa's internal default (~Nyquist/4).
        method: Pitch estimation algorithm ('pyin' or 'yin'). 'pyin' also provides
                voicing probability. (Default: 'pyin')
        hop_length: Hop length for analysis frames. If None, uses librosa's internal default
                    (typically 512 for pyin/yin).
        **kwargs: Additional keyword arguments passed to `librosa.pyin` or `librosa.yin`.

    Returns:
        A tuple containing:
        - times (NDArray[np.float64]): Time points for each estimate (center of frame, seconds).
        - f0 (NDArray[np.float64]): Fundamental frequency estimates in Hz. NaN for unvoiced frames.
        - voiced_flag (NDArray[np.float64]): Boolean flag indicating if frame is voiced (1.0 if voiced, 0.0 otherwise).
        - voiced_probs (NDArray[np.float64]): Voicing probability estimates (from 'pyin', otherwise same as voiced_flag).

    Raises:
        ValueError: If input `y` is not 1D or method is invalid.
        Exception: For errors during librosa calculation.
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    logger.debug(f"Estimating Fundamental Frequency (Pitch) using {method}: fmin={fmin}, fmax={fmax}, hop={hop_length}")

    # Set defaults for fmin/fmax if None (librosa handles this internally, but good practice)
    fmin = fmin if fmin is not None else librosa.note_to_hz('C2')
    fmax = fmax if fmax is not None else librosa.note_to_hz('C7')

    # Set hop_length for time calculation if not provided
    hop_length_calc = hop_length if hop_length is not None else 512 # Default used by librosa pitch funcs

    try:
        if method == 'pyin':
            # Fill NaN with a placeholder (e.g., 0) temporarily if needed, but keep original NaNs
            f0_raw, voiced_flag_raw, voiced_probs_raw = librosa.pyin(
                y=y,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                hop_length=hop_length, # Pass hop_length if specified
                fill_na=np.nan, # Explicitly request NaN for unvoiced
                **kwargs
            )
        elif method == 'yin':
             f0_raw = librosa.yin(
                 y=y,
                 fmin=fmin,
                 fmax=fmax,
                 sr=sr,
                 hop_length=hop_length, # Pass hop_length if specified
                 **kwargs
             )
             # YIN doesn't directly provide voiced_flag/probs. Estimate based on NaN.
             voiced_flag_raw = np.isfinite(f0_raw) # Voiced if f0 is not NaN
             voiced_probs_raw = voiced_flag_raw.astype(float) # Simple probability estimate
        else:
             raise ValueError(f"Unsupported pitch estimation method: {method}. Choose 'pyin' or 'yin'.")

        # Calculate corresponding time points based on hop length used
        times = librosa.times_like(f0_raw, sr=sr, hop_length=hop_length_calc)

        # Ensure float64 output. Keep NaNs in f0 for unvoiced frames.
        f0_out = f0_raw.astype(np.float64, copy=False)
        voiced_flag_out = voiced_flag_raw.astype(np.float64, copy=False) # Store bools as float 0/1
        voiced_probs_out = voiced_probs_raw.astype(np.float64, copy=False)
        times_out = times.astype(np.float64, copy=False)

        return (times_out, f0_out, voiced_flag_out, voiced_probs_out)
    except Exception as e:
        logger.error(f"Error estimating fundamental frequency: {e}")
        raise

# --- Voice Quality Features (Approximate Implementations) ---

def harmonic_to_noise_ratio(
    y: NDArray[np.float64],
    sr: int, # sr is needed for context, but not directly used here
    frame_length: int = 2048, # Frame length for HPSS STFT
    hop_length: Optional[int] = None, # Hop length for HPSS STFT
    harmonic_margin: Union[float, Tuple[float, ...]] = 1.0,
    percussive_margin: Union[float, Tuple[float, ...]] = 1.0,
    **kwargs: Any # Allow additional args for HPSS if needed
) -> NDArray[np.float64]:
    """
    Estimates the Harmonic-to-Noise Ratio (HNR) using HPSS [Approximation].

    Calculates the ratio of energy in the harmonic component to the energy in the
    percussive component for each frame, obtained via librosa.effects.hpss.
    This serves as a rough approximation of HNR. Higher values suggest more
    harmonic content relative to noise/percussive content.

    NOTE: This is NOT a standard HNR calculation (which often uses autocorrelation
          or cepstral methods). It approximates noise with the percussive component.
          Results may differ significantly from Praat or other standard HNR tools.

    Args:
        y: Audio time series (1D float64).
        sr: Sampling rate (Hz). Unused in this implementation but kept for signature consistency.
        frame_length: Frame length (samples) used for the internal STFT in HPSS. (Default: 2048)
        hop_length: Hop length (samples) used for the internal STFT in HPSS. Defaults to frame_length // 4.
        harmonic_margin: Margin parameter for HPSS separation (controls harmonic smoothness).
        percussive_margin: Margin parameter for HPSS separation (controls percussive sparsity).
        **kwargs: Additional keyword arguments passed to `librosa.effects.hpss`.

    Returns:
        Approximated HNR in dB for each frame (1D array, float64). Returns NaN if
        percussive energy is zero for a frame. Length matches the number of frames
        produced by the internal STFT of HPSS.
    """
    warnings.warn("harmonic_to_noise_ratio feature is an approximation based on HPSS energy ratio.", UserWarning, stacklevel=2)
    logger.warning("harmonic_to_noise_ratio feature is an approximation based on HPSS energy ratio.")
    if y.ndim != 1:
        raise ValueError("Input audio 'y' must be 1D for HPSS-based HNR.")

    hop_length_calc = hop_length if hop_length is not None else frame_length // 4

    try:
        # 1. Perform HPSS
        # HPSS uses STFT internally, parameters affect separation
        y_harmonic, y_percussive = librosa.effects.hpss(
            y,
            kernel_size=(31, 31), # Default kernel sizes
            margin=(harmonic_margin, percussive_margin),
            # Pass STFT parameters if needed, though hpss uses its own defaults if not specified
            # n_fft=frame_length, # HPSS uses its own internal STFT logic
            # hop_length=hop_length_calc,
            **kwargs
        )

        # 2. Calculate RMS energy per frame for harmonic and percussive components
        # Use the same framing parameters as the feature manager likely uses elsewhere
        rms_harmonic = rms_energy(y=y_harmonic, frame_length=frame_length, hop_length=hop_length_calc, center=True)
        rms_percussive = rms_energy(y=y_percussive, frame_length=frame_length, hop_length=hop_length_calc, center=True)

        # Ensure lengths match (should match if using same framing)
        num_frames = min(len(rms_harmonic), len(rms_percussive))
        rms_harmonic = rms_harmonic[:num_frames]
        rms_percussive = rms_percussive[:num_frames]

        # 3. Calculate HNR approximation in dB
        # HNR = 10 * log10 (Energy_harmonic / Energy_percussive)
        # Energy is proportional to RMS^2
        power_harmonic = rms_harmonic**2
        power_percussive = rms_percussive**2

        # Avoid division by zero and log(0)
        # If percussive power is near zero, HNR is very high (set to large number or NaN?)
        # If harmonic power is near zero but percussive is not, HNR is very low.
        hnr_db = np.full(num_frames, np.nan, dtype=np.float64) # Initialize with NaN
        valid_indices = (power_percussive > _EPSILON) & (power_harmonic > _EPSILON)
        hnr_db[valid_indices] = 10 * np.log10(power_harmonic[valid_indices] / power_percussive[valid_indices])

        # Handle cases where only one component has energy
        only_harmonic = (power_harmonic > _EPSILON) & (power_percussive <= _EPSILON)
        hnr_db[only_harmonic] = 80.0 # Assign a large dB value for very high HNR
        only_percussive = (power_harmonic <= _EPSILON) & (power_percussive > _EPSILON)
        hnr_db[only_percussive] = -80.0 # Assign a very low dB value

        logger.debug(f"Calculated approximate HNR for {num_frames} frames.")
        return hnr_db

    except Exception as e:
        logger.error(f"Error calculating approximate HNR using HPSS: {e}")
        # Return NaN array of expected length if possible
        num_frames = 1 + len(y) // hop_length_calc if hop_length_calc > 0 else 0
        return np.full(num_frames, np.nan, dtype=np.float64)


def jitter(
    y: NDArray[np.float64],
    sr: int,
    f0: Optional[NDArray[np.float64]] = None, # Requires fundamental frequency estimates
    voiced_flag: Optional[NDArray[np.float64]] = None, # Requires voicing information
    method: Literal['local_abs'] = 'local_abs', # Only local absolute jitter implemented
    f0_min: float = 75.0, # Min F0 for period calculation
    f0_max: float = 600.0, # Max F0 for period calculation
    hop_length: Optional[int] = None # Hop length used for F0 estimation
    # **kwargs: Any # Reserved for future methods/params
) -> NDArray[np.float64]:
    """
    Estimates the frequency jitter (local, absolute) [Approximation].

    Calculates the mean absolute difference between periods of consecutive voiced frames.
    Requires F0 estimates and voicing information, typically obtained from `fundamental_frequency`.

    NOTE: This is a simplified Jitter calculation based on frame-level F0.
          Standard Jitter measures (like Jita, RAP, PPQ) often require more precise
          period marking within the time-domain signal and are usually calculated
          using specialized tools (e.g., Praat). Results may differ significantly.

    Args:
        y: Audio time series (1D float64). Used only to determine output length if f0 is None.
        sr: Sampling rate (Hz).
        f0: Fundamental frequency contour in Hz (1D array). NaN indicates unvoiced frames.
        voiced_flag: Voicing flag contour (1D array, 1.0=voiced, 0.0=unvoiced).
                     Must correspond frame-by-frame with f0.
        method: Jitter calculation method. Currently only 'local_abs' (mean absolute
                difference between consecutive periods in seconds) is implemented.
        f0_min: Minimum F0 value (Hz) to consider for period calculation. F0 values
                below this (even if voiced) are ignored. (Default: 75.0)
        f0_max: Maximum F0 value (Hz) to consider for period calculation. F0 values
                above this (even if voiced) are ignored. (Default: 600.0)
        hop_length: Hop length (samples) used for F0 analysis. Needed only to determine
                    output array size if f0 is not provided.

    Returns:
        Jitter value (local, absolute difference in seconds) for each frame (1D array, float64).
        Returns NaN for unvoiced frames or frames where jitter cannot be calculated
        (e.g., start of a voiced segment, F0 out of range).
    """
    warnings.warn("Jitter feature is an approximation based on frame-level F0 period differences.", UserWarning, stacklevel=2)
    logger.warning("Jitter feature is an approximation based on frame-level F0 period differences.")
    if method != 'local_abs':
        raise NotImplementedError(f"Jitter method '{method}' not implemented. Only 'local_abs' is available.")

    if f0 is None or voiced_flag is None:
        logger.warning("F0 or voiced_flag not provided. Calculating F0 internally using pyin.")
        # Calculate F0 internally if not provided
        hop_length_calc = hop_length if hop_length is not None else 512
        try:
            _, f0, voiced_flag, _ = fundamental_frequency(y, sr, fmin=f0_min, fmax=f0_max, method='pyin', hop_length=hop_length_calc)
        except Exception as e:
            logger.error(f"Internal F0 calculation failed for Jitter: {e}")
            num_frames = 1 + len(y) // hop_length_calc if hop_length_calc > 0 else 0
            return np.full(num_frames, np.nan, dtype=np.float64)

    if len(f0) != len(voiced_flag):
        raise ValueError("Length of f0 and voiced_flag must match.")

    num_frames = len(f0)
    jitter_values = np.full(num_frames, np.nan, dtype=np.float64)

    # Calculate periods (T0 = 1/F0) for valid, voiced frames within range
    periods = np.full(num_frames, np.nan, dtype=np.float64)
    valid_f0_mask = (voiced_flag > 0.5) & np.isfinite(f0) & (f0 >= f0_min) & (f0 <= f0_max)
    periods[valid_f0_mask] = 1.0 / f0[valid_f0_mask]

    # Calculate absolute difference between consecutive valid periods
    period_diffs = np.abs(np.diff(periods)) # Will contain NaN where periods[i] or periods[i+1] is NaN

    # Jitter for frame i is based on diff between period i and period i-1
    # So, jitter_values[i] corresponds to period_diffs[i-1]
    # Assign NaN to the first frame and any frame where the diff is NaN
    jitter_values[1:] = period_diffs

    # Ensure NaN remains where diff calculation was invalid
    # FIX: Apply boolean mask only to the slice where period_diffs applies
    # Create mask for the diffs array
    invalid_diff_mask = ~np.isfinite(period_diffs)
    # Apply this mask to the corresponding part of jitter_values (from index 1 onwards)
    jitter_values[1:][invalid_diff_mask] = np.nan
    # --- End Fix ---

    # Explicitly set NaN for the first frame and unvoiced frames
    jitter_values[0] = np.nan
    jitter_values[voiced_flag <= 0.5] = np.nan # Also NaN for originally unvoiced

    logger.debug(f"Calculated approximate local absolute jitter for {num_frames} frames.")
    return jitter_values


def shimmer(
    y: NDArray[np.float64],
    sr: int, # sr needed for context if calculating RMS internally
    voiced_flag: Optional[NDArray[np.float64]] = None, # Requires voicing information
    method: Literal['local_rms_rel'] = 'local_rms_rel', # Only local relative RMS shimmer implemented
    frame_length: int = 2048, # Frame length for RMS calculation
    hop_length: Optional[int] = None, # Hop length for RMS calculation
    center: bool = True, # Centering for RMS calculation
    # **kwargs: Any # Reserved for future methods/params
) -> NDArray[np.float64]:
    """
    Estimates the amplitude shimmer (local, relative RMS difference) [Approximation].

    Calculates the mean absolute relative difference between the RMS energy of
    consecutive voiced frames. Requires voicing information.

    NOTE: This is a simplified Shimmer calculation based on frame-level RMS energy.
          Standard Shimmer measures (like ShdB, APQ) often require more precise
          peak amplitude tracking within fundamental periods and are usually calculated
          using specialized tools (e.g., Praat). Results may differ significantly.

    Args:
        y: Audio time series (1D float64).
        sr: Sampling rate (Hz). Used if calculating RMS internally.
        voiced_flag: Voicing flag contour (1D array, 1.0=voiced, 0.0=unvoiced).
                     If None, it will be calculated internally using pyin F0.
        method: Shimmer calculation method. Currently only 'local_rms_rel' (mean absolute
                relative difference between consecutive voiced frame RMS values) is implemented.
        frame_length: Frame length (samples) for RMS calculation. (Default: 2048)
        hop_length: Hop length (samples) for RMS calculation. Defaults to frame_length // 4.
        center: Whether RMS analysis uses centered frames. (Default: True)

    Returns:
        Shimmer value (local, relative RMS difference) for each frame (1D array, float64).
        Returns NaN for unvoiced frames or frames where shimmer cannot be calculated
        (e.g., start of a voiced segment).
    """
    warnings.warn("Shimmer feature is an approximation based on frame-level relative RMS differences.", UserWarning, stacklevel=2)
    logger.warning("Shimmer feature is an approximation based on frame-level relative RMS differences.")
    if y.ndim != 1:
        raise ValueError("Input audio 'y' must be 1D for shimmer calculation.")
    if method != 'local_rms_rel':
        raise NotImplementedError(f"Shimmer method '{method}' not implemented. Only 'local_rms_rel' is available.")

    hop_length_calc = hop_length if hop_length is not None else frame_length // 4
    if hop_length_calc <= 0:
         raise ValueError("Hop length must be positive.")

    # Calculate RMS energy
    try:
        rms_frames = rms_energy(y=y, frame_length=frame_length, hop_length=hop_length_calc, center=center)
    except Exception as e:
        logger.error(f"Internal RMS calculation failed for Shimmer: {e}")
        num_frames_est = 1 + len(y) // hop_length_calc if hop_length_calc > 0 else 0
        return np.full(num_frames_est, np.nan, dtype=np.float64)

    # Get voicing information if not provided
    if voiced_flag is None:
        logger.warning("voiced_flag not provided for Shimmer. Calculating F0 internally using pyin.")
        try:
            # Use F0 defaults matching Jitter if needed
            f0_min_sh = 75.0
            f0_max_sh = 600.0
            _, _, voiced_flag, _ = fundamental_frequency(y, sr, fmin=f0_min_sh, fmax=f0_max_sh, method='pyin', hop_length=hop_length_calc)
        except Exception as e:
            logger.error(f"Internal F0/voicing calculation failed for Shimmer: {e}")
            return np.full(len(rms_frames), np.nan, dtype=np.float64)

    # Ensure RMS and voicing align (take min length)
    num_frames = min(len(rms_frames), len(voiced_flag))
    if num_frames == 0: return np.array([], dtype=np.float64)

    rms_frames = rms_frames[:num_frames]
    voiced_flag = voiced_flag[:num_frames]
    shimmer_values = np.full(num_frames, np.nan, dtype=np.float64)

    # Calculate relative difference between RMS of consecutive *voiced* frames
    for i in range(1, num_frames):
        # Check if both current and previous frames are voiced
        if voiced_flag[i] > 0.5 and voiced_flag[i-1] > 0.5:
            # Calculate relative difference: 2 * |rms[i] - rms[i-1]| / (rms[i] + rms[i-1])
            rms_i = rms_frames[i]
            rms_prev = rms_frames[i-1]
            sum_rms = rms_i + rms_prev
            if sum_rms > _EPSILON: # Avoid division by zero
                shimmer_values[i] = 2.0 * np.abs(rms_i - rms_prev) / sum_rms
            else:
                 # If sum is zero, difference is also zero (or should be)
                 shimmer_values[i] = 0.0

    logger.debug(f"Calculated approximate local relative RMS shimmer for {num_frames} frames.")
    return shimmer_values


# --- Other Global Audio Metrics ---

def get_basic_audio_metrics(y: NDArray[np.float64], sr: int) -> Dict[str, float]:
    """
    Calculates basic summary metrics for the entire audio signal.

    Provides high-level information about the audio duration and overall amplitude levels.

    Args:
        y: Audio time series (1D or multi-channel float64). If multi-channel,
           metrics (RMS, peak) are calculated after converting to mono by averaging.
        sr: Sampling rate (Hz).

    Returns:
        Dictionary containing:
        - 'duration_seconds': Total duration of the audio in seconds (float).
        - 'rms_global': Root Mean Square value calculated over the entire signal (float).
        - 'peak_amplitude': Maximum absolute amplitude value in the signal (float).

    Raises:
        Exception: For errors during librosa calculation.
    """
    logger.debug("Calculating basic global audio metrics.")
    try:
        # Handle multi-channel by converting to mono first for RMS/peak calculation
        if y.ndim > 1:
            logger.warning("Input signal is multi-channel. Converting to mono for global RMS/peak calculation.")
            # Ensure averaging happens along the correct axis (assuming channels are dim 0)
            y_mono = np.mean(y, axis=0) if y.shape[0] < y.shape[1] else np.mean(y, axis=1)
            y_mono = y_mono.astype(np.float64)
        else:
            y_mono = y

        duration_seconds = librosa.get_duration(y=y, sr=sr) # Use original y for duration
        # Calculate global RMS on mono signal
        rms_global = np.sqrt(np.mean(y_mono**2)) if y_mono.size > 0 else 0.0
        peak_amplitude = np.max(np.abs(y_mono)) if y_mono.size > 0 else 0.0

        return {
            "duration_seconds": float(duration_seconds),
            "rms_global": float(rms_global),
            "peak_amplitude": float(peak_amplitude),
        }
    except Exception as e:
        logger.error(f"Error calculating basic audio metrics: {e}")
        raise

# --- Onset Detection ---

def detect_onsets(
    y: Optional[NDArray[np.float64]] = None,
    *, # Force keyword arguments after y
    sr: Optional[int] = None, # Required if y is provided
    onset_envelope: Optional[NDArray[np.float64]] = None, # Can use pre-computed envelope
    hop_length: int = 512,
    units: Literal['frames', 'samples', 'time'] = 'frames',
    **kwargs: Any # Pass additional args to librosa.onset.onset_detect
) -> NDArray[Any]:
    """
    Detects note onset events (start times) in an audio signal using librosa.

    Onsets mark the beginning of transient events in the audio, often corresponding
    to note starts or percussive hits.

    Can operate directly on the time series `y` or a pre-computed onset strength envelope.

    Args:
        y: Audio time series (1D float64). Required if `onset_envelope` is not provided.
        sr: Sampling rate (Hz). Required if `y` is provided.
        onset_envelope: Pre-computed onset strength envelope. If provided, `y` and `sr`
                        are ignored for envelope calculation, but `sr` and `hop_length`
                        are still needed if `units` is 'time' or 'samples'.
        hop_length: Hop length used to compute the onset envelope (samples). (Default: 512)
                    Crucial for correct time/sample conversion if `units` != 'frames'.
        units: Units for the returned onset locations ('frames', 'samples', 'time'). (Default: 'frames')
        **kwargs: Additional keyword arguments passed to `librosa.onset.onset_detect`,
                  such as `backtrack`, `energy`, `peak_pick parameters`.

    Returns:
        Array containing the detected onset locations in the specified units.
        Dtype is int64 for 'frames'/'samples', float64 for 'time'.

    Raises:
        ValueError: If required inputs (`y`/`sr` or `onset_envelope`) are missing,
                    or if `units` requires `sr`/`hop_length` which are not available.
        Exception: For errors during librosa calculation.
    """
    if y is None and onset_envelope is None:
        raise ValueError("Either audio time series 'y' or 'onset_envelope' must be provided.")
    if y is not None and sr is None:
        raise ValueError("Sampling rate 'sr' must be provided when using time series 'y'.")
    if units in ['samples', 'time'] and sr is None:
         # We need sr to convert frames to time/samples even if using pre-computed envelope
         raise ValueError(f"Sampling rate 'sr' is required when units='{units}'.")

    logger.debug(f"Detecting onsets: units={units}, hop_length={hop_length}, kwargs={kwargs}")
    try:
        # librosa.onset.onset_detect handles using y or onset_envelope
        onsets = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            onset_envelope=onset_envelope,
            hop_length=hop_length,
            units=units,
            **kwargs
        )
        # Ensure correct dtype based on units
        if units in ['frames', 'samples']:
            return onsets.astype(np.int64, copy=False)
        else: # units == 'time'
            return onsets.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error detecting onsets: {e}")
        raise
