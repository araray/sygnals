# sygnals/core/features/manager.py

"""
Manages the extraction of various features from signal data.

Handles framing, calculation of intermediate representations like spectrograms,
calling appropriate feature functions, and aggregating results into specified formats.
"""

import logging
import numpy as np
import pandas as pd
import librosa # Needed for framing, time/frequency utils, and some features
from numpy.typing import NDArray
# Import necessary types
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Set, Literal

# Import feature dictionaries and specific functions needed for dispatch
from .time_domain import TIME_DOMAIN_FEATURES
from .frequency_domain import FREQUENCY_DOMAIN_FEATURES, spectral_contrast
from .cepstral import CEPSTRAL_FEATURES, mfcc
# Import audio features that are often calculated per frame or need special handling
from ..audio.features import (
    zero_crossing_rate,
    rms_energy,
    harmonic_to_noise_ratio, # Placeholder
    jitter,                  # Placeholder
    shimmer                  # Placeholder
)

logger = logging.getLogger(__name__)

# --- Feature Definitions and Grouping ---

# Combine all available frame-based features (time-domain + specific audio)
# These operate on individual frames or use librosa's frame-based calculation
_FRAME_BASED_FEATURES: Dict[str, Callable] = {
    **TIME_DOMAIN_FEATURES,
    "zero_crossing_rate": zero_crossing_rate,
    "rms_energy": rms_energy,
    "hnr": harmonic_to_noise_ratio, # Placeholder
    "jitter": jitter,               # Placeholder
    "shimmer": shimmer,             # Placeholder
}

# Features requiring the magnitude spectrum of each frame
_SPECTRUM_BASED_FEATURES: Dict[str, Callable] = {
    **FREQUENCY_DOMAIN_FEATURES
    # Excludes spectral_contrast which needs the full spectrogram
}

# Features requiring the full spectrogram (magnitude or power)
_SPECTROGRAM_BASED_FEATURES: Dict[str, Callable] = {
    "spectral_contrast": spectral_contrast,
}

# Features requiring the log-power Mel spectrogram
_MELSPEC_BASED_FEATURES: Dict[str, Callable] = {
    "mfcc": mfcc,
}

# Combine all known feature names for validation
_ALL_KNOWN_FEATURES: Set[str] = (
    set(_FRAME_BASED_FEATURES.keys()) |
    set(_SPECTRUM_BASED_FEATURES.keys()) |
    set(_SPECTROGRAM_BASED_FEATURES.keys()) |
    set(_MELSPEC_BASED_FEATURES.keys())
)

# --- Custom Exception ---
class FeatureExtractionError(Exception):
    """Custom exception for errors during feature extraction."""
    pass

# --- Main Extraction Function ---

def extract_features(
    y: NDArray[np.float64],
    sr: int,
    features: List[str],
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: str = "hann",
    feature_params: Optional[Dict[str, Dict[str, Any]]] = None,
    output_format: Literal['dataframe', 'dict_of_arrays'] = 'dataframe'
) -> Union[pd.DataFrame, Dict[str, NDArray[Any]]]:
    """
    Extracts specified features from an audio signal.

    Handles framing, calculation of intermediate representations (STFT, Mel
    spectrogram), calling appropriate feature functions, and formatting the output.
    Intermediate calculations are cached to avoid redundant work.

    Args:
        y: Input audio time series (1D NumPy array, float64).
        sr: Sampling rate (Hz).
        features: List of feature names to extract (e.g., ['rms_energy', 'spectral_centroid', 'mfcc']).
                  Use 'all' to attempt extraction of all known standard features.
                  See internal dictionaries (_FRAME_BASED_FEATURES, etc.) for available names.
        frame_length: Analysis frame length in samples (default: 2048). Also used as n_fft
                      for spectral features unless overridden in feature_params.
        hop_length: Hop length between frames in samples (default: 512).
        center: Whether to pad the signal so frames are centered (default: True).
                Affects frame count and time alignment.
        window: Window function to apply for FFT-based features (default: "hann").
        feature_params: Dictionary containing parameters specific to certain features.
                        Keys are feature names, values are dictionaries of parameters.
                        Example: {'mfcc': {'n_mfcc': 20}, 'spectral_rolloff': {'roll_percent': 0.9}}
        output_format: Format for the returned features ('dataframe' or 'dict_of_arrays').
                       'dataframe': Returns a Pandas DataFrame indexed by time (seconds).
                       'dict_of_arrays': Returns a dictionary where keys are feature names
                                         (including 'time') and values are NumPy arrays.

    Returns:
        A Pandas DataFrame or a dictionary of NumPy arrays containing the extracted features
        and corresponding frame times. Returns empty if no features are extracted or
        the signal is too short.

    Raises:
        ValueError: If an unknown feature is requested, required parameters are missing,
                    or parameters are invalid.
        FeatureExtractionError: If a non-recoverable error occurs during extraction.
    """
    logger.info(f"Starting feature extraction for features: {features}")
    logger.debug(f"Parameters: sr={sr}, frame_length(n_fft)={frame_length}, hop={hop_length}, center={center}, window={window}")
    if feature_params:
        logger.debug(f"Feature-specific parameters: {feature_params}")

    feature_params = feature_params or {}
    results: Dict[str, NDArray[Any]] = {} # Store final feature arrays here
    processed_feature_names: Set[str] = set() # Track successfully processed base feature names

    # --- Handle 'all' features request ---
    if features == ['all']:
        features = sorted(list(_ALL_KNOWN_FEATURES))
        logger.info(f"Extracting all available features: {features}")

    # --- Input Validation ---
    unknown_features = [f for f in features if f not in _ALL_KNOWN_FEATURES]
    if unknown_features:
        raise ValueError(f"Unknown feature(s) requested: {unknown_features}. Available: {sorted(list(_ALL_KNOWN_FEATURES))}")
    if y.ndim != 1:
        raise ValueError("Input audio signal 'y' must be a 1D array.")

    # --- Determine Number of Frames and Times ---
    # Calculate expected number of frames based on librosa's centered framing logic
    if center:
        # Librosa's frame count for centered frames: 1 + N // hop_length
        num_frames = 1 + len(y) // hop_length
    else:
        # Frame count for non-centered analysis
        if len(y) >= frame_length:
            num_frames = 1 + (len(y) - frame_length) // hop_length
        else:
            num_frames = 0 # No full frames possible

    if num_frames <= 0:
        logger.warning("Signal is too short for the given frame/hop length and centering setting. No frames generated.")
        return pd.DataFrame() if output_format == 'dataframe' else {'time': np.array([], dtype=np.float64)}

    logger.debug(f"Expecting {num_frames} frames based on signal length and parameters.")

    # Calculate frame times (center times of frames if center=True)
    frame_indices = np.arange(num_frames)
    # Use n_fft=frame_length for time calculation consistency with STFT
    frame_times = librosa.frames_to_time(frame_indices, sr=sr, hop_length=hop_length, n_fft=frame_length if center else None)
    results['time'] = frame_times.astype(np.float64) # Store times early

    # --- Caching for Intermediate Representations ---
    # These will be computed on demand if needed by any requested feature
    _S_mag: Optional[NDArray[np.float64]] = None
    _S_mel_log: Optional[NDArray[np.float64]] = None
    _fft_freqs: Optional[NDArray[np.float64]] = None

    def get_fft_magnitude_and_freqs() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculates and caches STFT magnitude and frequencies."""
        nonlocal _S_mag, _fft_freqs
        if _S_mag is None or _fft_freqs is None:
            logger.debug("Calculating STFT magnitude and FFT frequencies...")
            try:
                # Use frame_length as n_fft by default
                stft_result = librosa.stft(
                    y=y, n_fft=frame_length, hop_length=hop_length,
                    win_length=frame_length, window=window, center=center
                )
                # Check if STFT frame count matches expected num_frames, adjust if necessary
                stft_num_frames = stft_result.shape[1]
                if stft_num_frames != num_frames:
                    logger.warning(f"STFT frames ({stft_num_frames}) mismatch calculated frame times ({num_frames}). "
                                   f"Adjusting frame count and times to match STFT output.")
                    nonlocal frame_times # Allow modification of outer scope variable
                    frame_times = librosa.times_like(stft_result, sr=sr, hop_length=hop_length, n_fft=frame_length)
                    results['time'] = frame_times.astype(np.float64) # Update times in results dict
                    # Update num_frames globally within this function if needed elsewhere?
                    # For now, subsequent loops will use stft_num_frames implicitly via S_mag shape

                _S_mag = np.abs(stft_result).astype(np.float64)
                _fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length).astype(np.float64)
                logger.debug(f"STFT calculated. Shape: {_S_mag.shape}")
            except Exception as e:
                raise FeatureExtractionError(f"Error calculating STFT: {e}")
        return _S_mag, _fft_freqs

    def get_log_mel_spectrogram() -> NDArray[np.float64]:
        """Calculates and caches the log-power Mel spectrogram."""
        nonlocal _S_mel_log
        if _S_mel_log is None:
            logger.debug("Calculating log-power Mel spectrogram...")
            try:
                S_mag, _ = get_fft_magnitude_and_freqs() # Ensure STFT is computed first
                # Get MFCC specific params from feature_params or use librosa defaults
                mfcc_p = feature_params.get('mfcc', {})
                n_mels = mfcc_p.get('n_mels', 128)
                fmin = mfcc_p.get('fmin', 0.0)
                fmax = mfcc_p.get('fmax', sr / 2.0)
                power = mfcc_p.get('power', 2.0) # Power for melspectrogram

                S_mel = librosa.feature.melspectrogram(
                    S=S_mag**power, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
                    n_fft=frame_length # Pass n_fft for consistency
                )
                _S_mel_log = librosa.power_to_db(S_mel, ref=np.max).astype(np.float64)
                logger.debug(f"Log-Mel spectrogram calculated. Shape: {_S_mel_log.shape}")
            except Exception as e:
                raise FeatureExtractionError(f"Error calculating Mel spectrogram: {e}")
        return _S_mel_log

    # --- Extract Features ---
    for feature_name in features:
        if feature_name in processed_feature_names:
            continue # Skip if already processed (e.g., via 'all' expansion)

        logger.debug(f"Attempting to extract feature: {feature_name}")
        params = feature_params.get(feature_name, {})
        current_num_frames = len(results['time']) # Use current frame count based on time array

        try:
            feature_result_list: List[Tuple[str, NDArray[Any]]] = [] # Store tuples of (name, array)

            # --- Frame-Based Features ---
            if feature_name in _FRAME_BASED_FEATURES:
                func = _FRAME_BASED_FEATURES[feature_name]
                # Features calculated by librosa over the whole signal (ZCR, RMS) or placeholders
                if feature_name in ["zero_crossing_rate", "rms_energy", "hnr", "jitter", "shimmer"]:
                     func_params = {
                         'frame_length': frame_length,
                         'hop_length': hop_length,
                         'center': center,
                         **params # Add feature-specific params
                     }
                     # Pass relevant context (y, sr) if needed
                     if feature_name in ["hnr", "jitter", "shimmer"]:
                         func_params['sr'] = sr
                         feature_array = func(y=y, **func_params)
                     elif feature_name in ["zero_crossing_rate", "rms_energy"]:
                         feature_array = func(y=y, **func_params)
                     else: # Should not happen
                          logger.error(f"Internal logic error: Unhandled feature '{feature_name}' in frame-based block.")
                          continue
                     feature_result_list.append((feature_name, feature_array))

                else: # Custom time-domain features applied frame-by-frame
                    logger.debug(f"Applying time-domain feature '{feature_name}' frame-by-frame.")
                    # Generate frames using librosa.util.frame
                    # Note: This framing might differ slightly from librosa's internal framing for ZCR/RMS
                    if center:
                        # Pad signal symmetrically for centered frames
                        pad_width = frame_length // 2
                        y_padded = np.pad(y, pad_width, mode='constant') # Default pad is 0
                        y_framed = librosa.util.frame(y_padded, frame_length=frame_length, hop_length=hop_length)
                    else:
                        y_framed = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

                    # Ensure frame count matches expected num_frames based on time array
                    if y_framed.shape[1] != current_num_frames:
                         logger.warning(f"Manual frame count ({y_framed.shape[1]}) for '{feature_name}' mismatch "
                                        f"expected ({current_num_frames}). Adjusting frames.")
                         # Adjust if necessary (e.g., trim extra frames)
                         if y_framed.shape[1] > current_num_frames:
                             y_framed = y_framed[:, :current_num_frames]
                         # Padding is complex if too few frames, might indicate upstream issue

                    frame_results = [func(y_framed[:, i], **params) for i in range(y_framed.shape[1])]
                    feature_array = np.array(frame_results, dtype=np.float64)
                    feature_result_list.append((feature_name, feature_array))

            # --- Spectrum-Based Features ---
            elif feature_name in _SPECTRUM_BASED_FEATURES:
                S_mag, fft_freqs = get_fft_magnitude_and_freqs()
                current_num_frames = S_mag.shape[1] # Update frame count based on actual STFT
                func = _SPECTRUM_BASED_FEATURES[feature_name]
                # Apply frame-by-frame (column-by-column of S_mag)
                frame_results = [func(S_mag[:, i], fft_freqs, **params) for i in range(current_num_frames)]
                feature_array = np.array(frame_results, dtype=np.float64)
                feature_result_list.append((feature_name, feature_array))

            # --- Spectrogram-Based Features ---
            elif feature_name in _SPECTROGRAM_BASED_FEATURES:
                S_mag, fft_freqs = get_fft_magnitude_and_freqs()
                current_num_frames = S_mag.shape[1] # Update frame count
                func = _SPECTROGRAM_BASED_FEATURES[feature_name]
                # Pass the full spectrogram S
                feature_output = func(S=S_mag, sr=sr, freqs=fft_freqs, **params)
                # Handle multi-band output (like spectral_contrast)
                if feature_name == "spectral_contrast":
                    n_bands_contrast = feature_output.shape[0] - 1
                    for i in range(n_bands_contrast):
                        band_name = f"contrast_band_{i}"
                        feature_result_list.append((band_name, feature_output[i, :current_num_frames]))
                    delta_name = "contrast_delta"
                    feature_result_list.append((delta_name, feature_output[n_bands_contrast, :current_num_frames]))
                else: # Assume single output array otherwise
                    feature_result_list.append((feature_name, feature_output))

            # --- Mel Spectrogram-Based Features ---
            elif feature_name in _MELSPEC_BASED_FEATURES:
                S_mel_log = get_log_mel_spectrogram()
                current_num_frames = S_mel_log.shape[1] # Update frame count
                func = _MELSPEC_BASED_FEATURES[feature_name]
                # Get specific params, merge with general framing params if needed by func
                mfcc_params = feature_params.get('mfcc', {})
                mfcc_params.setdefault('n_fft', frame_length) # Ensure defaults are available if needed by librosa
                mfcc_params.setdefault('hop_length', hop_length)
                # Call the function (e.g., mfcc)
                feature_output = func(S=S_mel_log, sr=sr, **mfcc_params) # Pass sr just in case
                # Handle multi-coefficient output (like MFCC)
                if feature_name == "mfcc":
                    n_coeffs = feature_output.shape[0]
                    for i in range(n_coeffs):
                        coeff_name = f"mfcc_{i}"
                        feature_result_list.append((coeff_name, feature_output[i, :current_num_frames]))
                else: # Assume single output array otherwise
                     feature_result_list.append((feature_name, feature_output))

            # --- Store results and mark processed ---
            for name, array in feature_result_list:
                 # Ensure array is float64
                 if not np.issubdtype(array.dtype, np.floating):
                      array = array.astype(np.float64)
                 # Pad or truncate array if its length doesn't match current_num_frames
                 if len(array) != current_num_frames:
                      logger.warning(f"Feature '{name}' array length ({len(array)}) mismatch expected frames "
                                     f"({current_num_frames}). Adjusting length (padding with NaN or truncating).")
                      if len(array) > current_num_frames:
                          array = array[:current_num_frames]
                      else: # len(array) < current_num_frames
                          padded_array = np.full(current_num_frames, np.nan, dtype=np.float64)
                          padded_array[:len(array)] = array
                          array = padded_array
                 results[name] = array
                 processed_feature_names.add(name) # Add derived names like mfcc_0

            processed_feature_names.add(feature_name) # Mark the original request name as processed
            logger.debug(f"Successfully processed feature: {feature_name}")

        except Exception as e:
            logger.error(f"Error extracting feature '{feature_name}': {e}", exc_info=False) # Set exc_info=False for cleaner logs unless debugging
            # Optionally raise, or just log and continue to next feature
            # raise FeatureExtractionError(f"Failed to extract feature '{feature_name}': {e}") from e

    # --- Final Output Formatting ---
    final_num_frames = len(results['time'])
    if final_num_frames == 0: # Re-check in case STFT adjustment failed badly
         logger.warning("No valid frames found after processing. Returning empty result.")
         return pd.DataFrame() if output_format == 'dataframe' else {'time': np.array([], dtype=np.float64)}

    # Filter results dictionary to include only successfully processed features + time
    # And ensure all arrays have the final correct length
    final_results = {'time': results['time']}
    for name, arr in results.items():
        if name == 'time': continue # Already added
        if name in processed_feature_names:
            if isinstance(arr, np.ndarray) and arr.ndim == 1:
                if len(arr) == final_num_frames:
                    final_results[name] = arr
                else:
                    # This shouldn't happen if padding/truncating above worked, but double-check
                    logger.error(f"Internal Error: Final length mismatch for feature '{name}' "
                                 f"({len(arr)} vs {final_num_frames}). Skipping feature in output.")
            else:
                 logger.warning(f"Skipping feature '{name}' in final output due to unexpected type/shape: {type(arr)}/{getattr(arr, 'shape', 'N/A')}")

    if len(final_results) <= 1: # Only 'time' key left
        logger.warning("No features were successfully extracted or passed final checks.")
        return pd.DataFrame() if output_format == 'dataframe' else {'time': final_results.get('time', np.array([], dtype=np.float64))}

    # --- Create DataFrame or return Dictionary ---
    if output_format == 'dataframe':
        try:
            # Use time array for index
            time_index = pd.to_timedelta(final_results.pop('time'), unit='s')
            df = pd.DataFrame(final_results, index=time_index)
            df.index.name = 'time'
            logger.info(f"Feature extraction complete. DataFrame shape: {df.shape}")
            return df
        except Exception as e:
             raise FeatureExtractionError(f"Error creating output DataFrame: {e}")
    elif output_format == 'dict_of_arrays':
        logger.info(f"Feature extraction complete. Returning dictionary of {len(final_results)} arrays.")
        return final_results
    else:
        # This case should not be reached due to Literal type hint, but handle defensively
        raise ValueError(f"Unsupported output format: {output_format}. Choose 'dataframe' or 'dict_of_arrays'.")
