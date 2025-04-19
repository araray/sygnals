# sygnals/core/audio/features.py

"""
Functions for extracting audio-specific features.

Leverages libraries like librosa for common audio features like ZCR, RMS, pitch,
and onset detection. Includes placeholders for more complex voice quality features.
"""

import logging
import librosa
import numpy as np
from numpy.typing import NDArray
# Import necessary types
from typing import Optional, Dict, Any, Tuple, Literal

logger = logging.getLogger(__name__)

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

    Example:
        >>> sr = 22050
        >>> y_noise = np.random.randn(sr)
        >>> y_sine = np.sin(2 * np.pi * 440.0 * np.arange(sr)/sr)
        >>> zcr_noise = zero_crossing_rate(y_noise)
        >>> zcr_sine = zero_crossing_rate(y_sine)
        >>> print(f"Mean ZCR (Noise): {np.mean(zcr_noise):.3f}")
        >>> print(f"Mean ZCR (Sine): {np.mean(zcr_sine):.3f}")
        Mean ZCR (Noise): 0.XXX # Expected higher value
        Mean ZCR (Sine): 0.XXX # Expected lower value (approx 2*440/sr)
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

    Example:
        >>> sr = 22050
        >>> y = np.sin(2 * np.pi * 440.0 * np.arange(sr)/sr) * 0.5 # Amplitude 0.5
        >>> rms = rms_energy(y=y)
        >>> # Expected RMS for sine A*sin(wt) is A/sqrt(2)
        >>> print(f"Mean RMS: {np.mean(rms):.3f} (Expected approx: {0.5/np.sqrt(2):.3f})")
        Mean RMS: 0.XXX (Expected approx: 0.354)
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
        - f0 (NDArray[np.float64]): Fundamental frequency estimates in Hz. 0.0 for unvoiced frames.
        - voiced_flag (NDArray[np.float64]): Boolean flag indicating if frame is voiced (1.0 if voiced, 0.0 otherwise).
        - voiced_probs (NDArray[np.float64]): Voicing probability estimates (from 'pyin', otherwise same as voiced_flag).

    Raises:
        ValueError: If input `y` is not 1D or method is invalid.
        Exception: For errors during librosa calculation.

    Example:
        >>> sr = 22050
        >>> y = librosa.tone(frequency=261.63, sr=sr, duration=1) # Middle C
        >>> times, f0, vf, vp = fundamental_frequency(y, sr=sr, method='pyin')
        >>> voiced_f0 = f0[vf > 0.5] # Get f0 for voiced frames
        >>> print(f"Mean F0 (voiced): {np.mean(voiced_f0):.2f} Hz")
        Mean F0 (voiced): 261.XX Hz
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
            f0_raw, voiced_flag_raw, voiced_probs_raw = librosa.pyin(
                y=y,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                hop_length=hop_length, # Pass hop_length if specified
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
             voiced_flag_raw = np.isfinite(f0_raw) # Simple estimate: voiced if f0 is not NaN
             voiced_probs_raw = voiced_flag_raw.astype(float) # Simple probability estimate
        else:
             raise ValueError(f"Unsupported pitch estimation method: {method}. Choose 'pyin' or 'yin'.")

        # Calculate corresponding time points based on hop length used
        times = librosa.times_like(f0_raw, sr=sr, hop_length=hop_length_calc)

        # Ensure float64 output and handle potential NaNs in f0
        # Replace NaN with 0.0 for unvoiced frames for consistency
        f0_out = np.nan_to_num(f0_raw, nan=0.0).astype(np.float64, copy=False)
        voiced_flag_out = voiced_flag_raw.astype(np.float64, copy=False) # Store bools as float 0/1
        voiced_probs_out = voiced_probs_raw.astype(np.float64, copy=False)
        times_out = times.astype(np.float64, copy=False)

        return (times_out, f0_out, voiced_flag_out, voiced_probs_out)
    except Exception as e:
        logger.error(f"Error estimating fundamental frequency: {e}")
        raise

# --- Voice Quality Features (Placeholders) ---

def harmonic_to_noise_ratio(
    y: NDArray[np.float64],
    sr: int, # sr is needed for context, but not directly used in placeholder calculation
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True, # Assume centering consistent with other features
    **kwargs: Any
) -> NDArray[np.float64]:
    """
    Estimates the Harmonic-to-Noise Ratio (HNR) [Placeholder].

    HNR measures the ratio of harmonic energy (related to periodic, pitched sound)
    to noise energy (related to aperiodic or random components) in the signal,
    often used as an indicator of voice quality (e.g., hoarseness, breathiness).
    Higher HNR generally indicates a clearer, more harmonic voice.

    NOTE: This is a placeholder implementation and returns an array of NaNs.
          A real implementation would likely involve autocorrelation, cepstral analysis,
          or specialized algorithms (e.g., from Praat).

    Args:
        y: Audio time series (1D float64).
        sr: Sampling rate (unused in placeholder calculation but kept for signature consistency).
        frame_length: Analysis frame length (samples).
        hop_length: Hop length between frames (samples).
        center: Whether the analysis is centered (affects frame count calculation).
        **kwargs: Additional parameters for future implementation.

    Returns:
        An array of NaNs with the expected length based on framing parameters.
        Shape: (num_frames,). Dtype: float64.
    """
    logger.warning("harmonic_to_noise_ratio feature is a placeholder and returns NaN.")
    # Calculate expected number of frames consistent with librosa's centered framing
    if center:
        # Use librosa's frame count calculation for centered frames
        num_frames = 1 + len(y) // hop_length
    else:
        # Frame count calculation for non-centered analysis
        if len(y) >= frame_length:
            num_frames = 1 + (len(y) - frame_length) // hop_length
        else:
            num_frames = 0 # No full frames possible

    logger.debug(f"Placeholder HNR: Calculated num_frames = {num_frames} based on framing parameters.")
    return np.full(num_frames, np.nan, dtype=np.float64)

def jitter(
    y: NDArray[np.float64],
    sr: int, # sr is needed for context, but not directly used in placeholder calculation
    f0: Optional[NDArray[np.float64]] = None, # Requires fundamental frequency estimates
    method: str = 'local', # Common methods: 'local', 'rap', 'ppq5'
    frame_length: int = 2048, # Added frame/hop/center for consistency
    hop_length: int = 512,
    center: bool = True,
    **kwargs: Any
) -> NDArray[np.float64]:
    """
    Estimates the frequency jitter (periodicity variations) [Placeholder].

    Jitter refers to the short-term variations in the fundamental frequency (F0)
    period length of voiced speech. It's often used as a measure of voice instability
    or roughness. Calculation typically requires accurate F0 estimates and period marking.

    NOTE: This is a placeholder implementation and returns an array of NaNs.
          A real implementation requires F0 tracking and period difference calculations,
          often implemented in specialized libraries (e.g., Praat via Parselmouth).

    Args:
        y: Audio time series (1D float64).
        sr: Sampling rate (unused in placeholder calculation).
        f0: Optional pre-computed fundamental frequency contour (Hz). If None,
            it would need to be calculated internally (not done in placeholder).
        method: Jitter calculation method (e.g., 'local', 'rap', 'ppq5'). Placeholder ignores this.
        frame_length: Analysis frame length (samples). Used to determine output length if f0 is None.
        hop_length: Hop length between frames (samples). Used to determine output length if f0 is None.
        center: Whether the analysis is centered. Used to determine output length if f0 is None.
        **kwargs: Additional parameters for future implementation.

    Returns:
        An array of NaNs with the expected length based on F0 contour or framing.
        Shape: (num_frames,). Dtype: float64.
    """
    logger.warning("jitter feature is a placeholder and returns NaN.")
    # Determine output length based on f0 if provided, otherwise based on standard framing
    if f0 is not None:
        num_frames = len(f0)
    else:
        # Calculate expected number of frames consistent with librosa's centered framing
        if center:
            num_frames = 1 + len(y) // hop_length
        else:
            if len(y) >= frame_length:
                num_frames = 1 + (len(y) - frame_length) // hop_length
            else:
                num_frames = 0
    logger.debug(f"Placeholder Jitter: Calculated num_frames = {num_frames}.")
    return np.full(num_frames, np.nan, dtype=np.float64)

def shimmer(
    y: NDArray[np.float64],
    sr: int, # sr is needed for context, but not directly used in placeholder calculation
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True, # Assume centering consistent with other features
    method: str = 'local_db', # Common methods: 'local', 'local_db', 'apq11'
    **kwargs: Any
) -> NDArray[np.float64]:
    """
    Estimates the amplitude shimmer (amplitude variations) [Placeholder].

    Shimmer refers to the short-term variations in the peak amplitude of consecutive
    fundamental frequency periods in voiced speech. It's another measure often
    associated with voice quality and perceived roughness or breathiness.

    NOTE: This is a placeholder implementation and returns an array of NaNs.
          A real implementation requires peak amplitude tracking within voiced segments,
          often implemented in specialized libraries (e.g., Praat via Parselmouth).

    Args:
        y: Audio time series (1D float64).
        sr: Sampling rate (unused in placeholder calculation).
        frame_length: Analysis frame length (samples).
        hop_length: Hop length between frames (samples).
        center: Whether the analysis is centered.
        method: Shimmer calculation method (e.g., 'local', 'local_db', 'apq11'). Placeholder ignores this.
        **kwargs: Additional parameters for future implementation.

    Returns:
        An array of NaNs with the expected length based on framing parameters.
        Shape: (num_frames,). Dtype: float64.
    """
    logger.warning("shimmer feature is a placeholder and returns NaN.")
    # Calculate expected number of frames consistent with librosa's centered framing
    if center:
        num_frames = 1 + len(y) // hop_length
    else:
        if len(y) >= frame_length:
            num_frames = 1 + (len(y) - frame_length) // hop_length
        else:
            num_frames = 0
    logger.debug(f"Placeholder Shimmer: Calculated num_frames = {num_frames}.")
    return np.full(num_frames, np.nan, dtype=np.float64)


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
            y_mono = np.mean(y, axis=0) # Average across channels (assuming channels are first dim if shape[0] < shape[1])
            # Librosa's get_duration works correctly on multi-channel input based on samples dim
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

    Example:
        >>> sr = 22050
        >>> clicks = librosa.clicks(times=[0.5, 1.0, 1.5], sr=sr, length=sr*2)
        >>> onset_times = detect_onsets(y=clicks, sr=sr, units='time')
        >>> print(np.round(onset_times, 2))
        [0.5  1.  1.5]
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
