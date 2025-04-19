# sygnals/core/features/manager.py

"""
Manages the extraction of various features from signal data.
Handles framing, calculation of intermediate representations like spectrograms,
calling appropriate feature functions, and aggregating results.
"""

import logging
import numpy as np
import pandas as pd
import librosa # Needed for framing, time/frequency utils, and some features
from numpy.typing import NDArray
# Import necessary types
from typing import List, Dict, Any, Optional, Tuple, Union

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


# Import DSP functions needed for spectral features (if not using librosa's versions)
# from ..dsp import compute_fft # Example if using custom FFT

logger = logging.getLogger(__name__)

# Combine available features for easier lookup and validation
# Start with time-domain features
_AVAILABLE_FEATURES_DICT = {**TIME_DOMAIN_FEATURES}
# Add frame-based audio features
_AVAILABLE_FEATURES_DICT["zero_crossing_rate"] = zero_crossing_rate
_AVAILABLE_FEATURES_DICT["rms_energy"] = rms_energy
# Add placeholder audio features (treated as frame-based for now)
_AVAILABLE_FEATURES_DICT["hnr"] = harmonic_to_noise_ratio
_AVAILABLE_FEATURES_DICT["jitter"] = jitter
_AVAILABLE_FEATURES_DICT["shimmer"] = shimmer
# DO NOT add FREQUENCY_DOMAIN_FEATURES here - they need special handling below

# List features requiring special handling (e.g., operate on full spectrogram or signal)
_SPECIAL_HANDLING_FEATURES = {"mfcc", "spectral_contrast"}
# Add frequency domain features here if they need specific handling different from time-domain
# For now, they are handled in their own elif block below.


class FeatureExtractionError(Exception):
    """Custom exception for errors during feature extraction."""
    pass


def extract_features(
    y: NDArray[np.float64],
    sr: int,
    features: List[str],
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: str = "hann",
    feature_params: Optional[Dict[str, Dict[str, Any]]] = None,
    output_format: str = 'dataframe' # 'dataframe' or 'dict_of_arrays'
) -> Union[pd.DataFrame, Dict[str, NDArray[Any]]]:
    """
    Extracts specified features from an audio signal.

    Handles framing, FFT/Mel spectrogram calculation (if needed), calling appropriate
    feature functions (frame-based or spectrogram-based), and formatting the output.

    Args:
        y: Input audio time series (1D NumPy array, float64).
        sr: Sampling rate (Hz).
        features: List of feature names to extract (e.g., ['rms_energy', 'spectral_centroid', 'mfcc']).
                  Use 'all' to attempt extraction of all known standard features.
        frame_length: Analysis frame length in samples (default: 2048). Used for STFT n_fft
                      and potentially by time-domain features if they require explicit frames.
        hop_length: Hop length between frames in samples (default: 512).
        center: Whether to pad the signal so frames are centered (default: True). This is
                primarily handled by librosa functions like stft, rms, zcr.
        window: Window function to apply for FFT-based features (default: "hann").
        feature_params: Dictionary containing parameters specific to certain features.
                        Keys are feature names, values are dictionaries of parameters.
                        Example: {'mfcc': {'n_mfcc': 20}, 'spectral_rolloff': {'roll_percent': 0.9}}
        output_format: Format for the returned features ('dataframe' or 'dict_of_arrays').
                       'dataframe': Returns a Pandas DataFrame indexed by time.
                       'dict_of_arrays': Returns a dictionary where keys are feature names
                                         (including 'time') and values are NumPy arrays.

    Returns:
        A Pandas DataFrame or a dictionary of NumPy arrays containing the extracted features.
        DataFrame index or 'time' key in dictionary corresponds to frame times (in seconds).

    Raises:
        ValueError: If an unknown feature is requested or parameters are invalid.
        FeatureExtractionError: If an error occurs during any stage of extraction.
    """
    logger.info(f"Starting feature extraction for features: {features}")
    logger.debug(f"Parameters: sr={sr}, frame_length(n_fft)={frame_length}, hop={hop_length}, center={center}, window={window}")
    if feature_params:
        logger.debug(f"Feature-specific parameters: {feature_params}")

    feature_params = feature_params or {}
    results: Dict[str, NDArray[Any]] = {} # Store results here, key=feature_name, value=array
    processed_features = set() # Track which features (including derived ones like mfcc_0) are done

    # --- Handle 'all' features request ---
    if features == ['all']:
        # Combine standard features and special handling ones
        # Include FREQUENCY_DOMAIN_FEATURES keys here for 'all'
        features = list(_AVAILABLE_FEATURES_DICT.keys()) + list(FREQUENCY_DOMAIN_FEATURES.keys()) + list(_SPECIAL_HANDLING_FEATURES)
        # Remove duplicates if any feature exists in multiple places
        features = sorted(list(set(features)))
        logger.info(f"Extracting all available features: {features}")

    # --- Input Validation ---
    # Update known features to include frequency domain ones
    all_known_features = set(_AVAILABLE_FEATURES_DICT.keys()) | set(FREQUENCY_DOMAIN_FEATURES.keys()) | _SPECIAL_HANDLING_FEATURES
    unknown_features = [f for f in features if f not in all_known_features]
    if unknown_features:
        raise ValueError(f"Unknown feature(s) requested: {unknown_features}. Available: {sorted(list(all_known_features))}")

    # --- Determine Number of Frames and Times ---
    # Calculate expected number of frames based on librosa's centered framing logic
    # This count should align with the output of librosa features like stft, rms, zcr when center=True
    if center:
        num_frames = 1 + int(np.floor(len(y) / hop_length))
    else:
        # Calculate frames for non-centered analysis
        # This might need adjustment based on how non-centered features are implemented
        if len(y) >= frame_length:
            num_frames = 1 + int(np.floor((len(y) - frame_length) / hop_length))
        else:
            num_frames = 0 # No full frames possible if signal < frame_length and not centered

    if num_frames == 0:
        logger.warning("Signal is too short for the given frame/hop length and centering setting, no frames generated.")
        return pd.DataFrame() if output_format == 'dataframe' else {'time': np.array([])}

    logger.debug(f"Expecting {num_frames} frames based on signal length and parameters.")

    # Calculate frame times (center times of frames if center=True)
    frame_indices = np.arange(num_frames)
    # Use n_fft=frame_length for time calculation when center=True for consistency with STFT
    frame_times = librosa.frames_to_time(frame_indices, sr=sr, hop_length=hop_length, n_fft=frame_length if center else None)
    results['time'] = frame_times # Store times early

    # --- Pre-calculate Spectrograms if needed ---
    # These variables will store the computed spectrograms
    S_mag: Optional[NDArray[np.float64]] = None
    S_mel_log: Optional[NDArray[np.float64]] = None
    fft_freqs: Optional[NDArray[np.float64]] = None

    # Determine if any requested feature requires FFT or Mel spectrogram
    needs_fft_spectrum = any(f in FREQUENCY_DOMAIN_FEATURES for f in features)
    needs_fft_spectrogram = "spectral_contrast" in features
    needs_mel = "mfcc" in features

    if needs_fft_spectrum or needs_fft_spectrogram or needs_mel:
        logger.debug("Calculating STFT for spectral/cepstral features...")
        try:
            # Calculate STFT using librosa - handles windowing, centering, and short signals
            stft_result = librosa.stft(
                y, # Use original signal `y`
                n_fft=frame_length,
                hop_length=hop_length,
                win_length=frame_length, # Use frame_length for win_length unless specified otherwise
                window=window,
                center=center,
            )
            # Verify STFT frame count matches expected num_frames
            stft_num_frames = stft_result.shape[1]
            if stft_num_frames != num_frames:
                 logger.warning(f"STFT frames ({stft_num_frames}) mismatch calculated frame times ({num_frames}). "
                                f"This might happen with very short signals or specific edge padding. "
                                f"Using STFT frame count ({stft_num_frames}) for subsequent features.")
                 # Adjust num_frames and frame_times based on the actual STFT output
                 num_frames = stft_num_frames
                 frame_indices = np.arange(num_frames)
                 frame_times = librosa.frames_to_time(frame_indices, sr=sr, hop_length=hop_length, n_fft=frame_length if center else None)
                 results['time'] = frame_times # Update times in results

            S_mag = np.abs(stft_result) # Magnitude spectrogram (n_freq_bins, n_frames)
            fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length) # Frequencies for STFT bins
            logger.debug(f"STFT calculated. Shape: {S_mag.shape}")

            if needs_mel:
                 logger.debug("Calculating Mel spectrogram for MFCC...")
                 # Get MFCC specific params from feature_params or use librosa defaults
                 mfcc_p = feature_params.get('mfcc', {})
                 n_mels = mfcc_p.get('n_mels', 128)
                 fmin = mfcc_p.get('fmin', 0.0)
                 fmax = mfcc_p.get('fmax', sr / 2.0)
                 power = mfcc_p.get('power', 2.0) # Power for melspectrogram (usually 2.0)

                 # Calculate Mel spectrogram from power spectrogram (S_mag**power)
                 S_mel = librosa.feature.melspectrogram(
                     S=S_mag**power,
                     sr=sr,
                     n_mels=n_mels,
                     fmin=fmin,
                     fmax=fmax,
                     # Other params like n_fft, hop_length are inferred from S
                 )
                 # Convert to log-power Mel spectrogram (dB scale) - common input for MFCC
                 S_mel_log = librosa.power_to_db(S_mel, ref=np.max)
                 logger.debug(f"Log-Mel spectrogram calculated. Shape: {S_mel_log.shape}")

        except Exception as e:
            raise FeatureExtractionError(f"Error calculating STFT/Mel spectrogram: {e}")

    # --- Extract Features Iteratively ---
    for feature_name in features:
        if feature_name in processed_features:
            continue

        logger.debug(f"Extracting feature: {feature_name}")
        params = feature_params.get(feature_name, {})
        feature_result: Optional[NDArray[Any]] = None # Initialize result for this feature

        try:
            # --- Handle frame-based features (Time Domain, ZCR, RMS, Placeholders) ---
            if feature_name in _AVAILABLE_FEATURES_DICT:
                func = _AVAILABLE_FEATURES_DICT[feature_name]
                # Features calculated by librosa over the whole signal (ZCR, RMS, Placeholders)
                if feature_name in ["zero_crossing_rate", "rms_energy", "hnr", "jitter", "shimmer"]:
                     func_params = {
                         'frame_length': frame_length,
                         'hop_length': hop_length,
                         'center': center,
                         **params # Add feature-specific params
                     }
                     # Pass relevant context (y, sr) if needed by the function signature
                     # Check function signature or requirements (placeholders need y, sr)
                     if feature_name in ["hnr", "jitter", "shimmer"]:
                         func_params['sr'] = sr # Pass sr to placeholders
                         feature_result = func(y=y, **func_params)
                     elif feature_name in ["zero_crossing_rate", "rms_energy"]:
                         # These librosa features only need y and framing params
                         feature_result = func(y=y, **func_params)
                     else:
                          # Should not happen based on current list, but handle defensively
                          logger.warning(f"Unhandled feature '{feature_name}' in librosa feature block.")
                          continue

                else:
                    # Apply other time-domain features frame-by-frame
                    # These functions expect a single frame as input
                    # We need to generate the frames explicitly here if needed
                    logger.debug(f"Applying time-domain feature '{feature_name}' frame-by-frame.")
                    try:
                         # Generate frames using librosa.util.frame, handling potential short signals
                         # Use the original signal `y` and let librosa handle padding if center=True
                         # Note: This framing is specific to these custom time-domain funcs
                         if center:
                              y_framed = librosa.util.frame(np.pad(y, frame_length // 2, mode='constant'), frame_length=frame_length, hop_length=hop_length)
                         else:
                              y_framed = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

                         # Ensure frame count matches expected num_frames
                         if y_framed.shape[1] != num_frames:
                              logger.warning(f"Manual frame count ({y_framed.shape[1]}) mismatch for '{feature_name}' vs expected ({num_frames}). Using {num_frames} frames.")
                              # Adjust if necessary, although this indicates potential inconsistency
                              if y_framed.shape[1] > num_frames: y_framed = y_framed[:, :num_frames]
                              # Padding might be complex here, maybe skip feature if mismatch is large?

                         feature_result_list = [func(y_framed[:, i], **params) for i in range(y_framed.shape[1])]
                         # Pad result if manual framing produced fewer frames than expected
                         if len(feature_result_list) < num_frames:
                              pad_width = num_frames - len(feature_result_list)
                              # Pad with NaN or last value? Use NaN for safety.
                              feature_result_list.extend([np.nan] * pad_width)
                         feature_result = np.array(feature_result_list)

                    except Exception as frame_err:
                         # Catch framing errors specifically for time-domain features if signal is too short
                         logger.error(f"Error framing signal for time-domain feature '{feature_name}': {frame_err}. Skipping feature.")
                         continue # Skip this feature

                # Store and mark as processed if result is valid
                if feature_result is not None:
                     results[feature_name] = feature_result
                     processed_features.add(feature_name)

            # --- Handle frequency-domain features (operating on single frame spectrum) ---
            elif feature_name in FREQUENCY_DOMAIN_FEATURES:
                if S_mag is None or fft_freqs is None:
                    raise FeatureExtractionError(f"FFT magnitude spectrum 'S_mag' not available for feature '{feature_name}'")
                func = FREQUENCY_DOMAIN_FEATURES[feature_name]
                # Apply frame-by-frame (column-by-column of S_mag)
                feature_result = np.array([func(S_mag[:, i], fft_freqs, **params) for i in range(num_frames)])
                results[feature_name] = feature_result
                processed_features.add(feature_name)

            # --- Handle spectral_contrast (operates on full spectrogram) ---
            elif feature_name == "spectral_contrast":
                 if S_mag is None or fft_freqs is None:
                     raise FeatureExtractionError(f"FFT magnitude spectrum 'S_mag' or 'fft_freqs' not available for feature '{feature_name}'")
                 feature_result_bands = spectral_contrast(S=S_mag, sr=sr, freqs=fft_freqs, **params)
                 n_bands_contrast = feature_result_bands.shape[0] - 1
                 for i in range(n_bands_contrast):
                     band_name = f"contrast_band_{i}"
                     results[band_name] = feature_result_bands[i, :num_frames] # Ensure correct length
                     processed_features.add(band_name)
                 delta_name = "contrast_delta"
                 results[delta_name] = feature_result_bands[n_bands_contrast, :num_frames] # Ensure correct length
                 processed_features.add(delta_name)
                 processed_features.add(feature_name) # Mark original request as processed

            # --- Handle MFCC (operates on log-Mel spectrogram or signal) ---
            elif feature_name == "mfcc":
                 mfcc_params = feature_params.get('mfcc', {})
                 mfcc_params.setdefault('n_fft', frame_length)
                 mfcc_params.setdefault('hop_length', hop_length)
                 mfcc_params.setdefault('center', center)
                 mfcc_params.setdefault('window', window)
                 # Prefer passing pre-calculated S_mel_log
                 feature_result_coeffs = mfcc(y=y if S_mel_log is None else None,
                                              sr=sr,
                                              S=S_mel_log,
                                              **mfcc_params)
                 n_mfcc_coeffs = feature_result_coeffs.shape[0]
                 for i in range(n_mfcc_coeffs):
                     mfcc_name = f"mfcc_{i}"
                     results[mfcc_name] = feature_result_coeffs[i, :num_frames] # Ensure correct length
                     processed_features.add(mfcc_name)
                 processed_features.add(feature_name) # Mark original request as processed

            # --- Log success for the original feature name ---
            logger.debug(f"Successfully processed feature: {feature_name}")

        except Exception as e:
            logger.error(f"Error extracting feature '{feature_name}': {e}", exc_info=True)
            raise FeatureExtractionError(f"Failed to extract feature '{feature_name}': {e}")

    # --- Format Output ---
    if not results or len(results) == 1 and 'time' in results: # Check if only 'time' is present
        logger.warning("No features were successfully extracted.")
        return pd.DataFrame() if output_format == 'dataframe' else {'time': results.get('time', np.array([]))}

    # Ensure all result arrays have the correct length (num_frames) before creating output
    final_results = {}
    time_array = results.pop('time') # Remove time array temporarily

    for name, arr in results.items():
        if not isinstance(arr, np.ndarray):
             logger.warning(f"Result for '{name}' is not a NumPy array ({type(arr)}). Skipping.")
             continue
        if arr.ndim != 1:
             logger.warning(f"Result for '{name}' is not 1D (shape {arr.shape}). Skipping.")
             continue

        if len(arr) == num_frames:
            final_results[name] = arr
        elif len(arr) > num_frames:
             logger.warning(f"Feature '{name}' array length ({len(arr)}) > num_frames ({num_frames}). Truncating.")
             final_results[name] = arr[:num_frames]
        elif len(arr) < num_frames:
             logger.warning(f"Feature '{name}' array length ({len(arr)}) < num_frames ({num_frames}). Padding with NaN.")
             padded_arr = np.full(num_frames, np.nan)
             padded_arr[:len(arr)] = arr
             final_results[name] = padded_arr
        # else: # Should be covered by len(arr) == num_frames
        #     logger.error(f"Feature '{name}' has unexpected length {len(arr)} vs {num_frames}. Skipping.")

    # Add time array back for dict output
    if output_format == 'dict_of_arrays':
        final_results['time'] = time_array
        logger.info(f"Feature extraction complete. Returning dictionary of {len(final_results)} arrays.")
        return final_results

    # Create DataFrame for dataframe output
    elif output_format == 'dataframe':
        if not final_results: # Check again after potential skips
             logger.warning("No valid features remained after length checks.")
             return pd.DataFrame()
        try:
            df = pd.DataFrame(final_results)
            df.index = pd.to_timedelta(time_array, unit='s')
            df.index.name = 'time'
            logger.info(f"Feature extraction complete. DataFrame shape: {df.shape}")
            return df
        except Exception as e:
             raise FeatureExtractionError(f"Error creating output DataFrame: {e}")
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Choose 'dataframe' or 'dict_of_arrays'.")
