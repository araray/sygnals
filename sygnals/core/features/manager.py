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
# Add frequency-domain features (that operate per-frame on spectrum)
_AVAILABLE_FEATURES_DICT.update(FREQUENCY_DOMAIN_FEATURES)
# Add placeholder audio features
_AVAILABLE_FEATURES_DICT["hnr"] = harmonic_to_noise_ratio
_AVAILABLE_FEATURES_DICT["jitter"] = jitter
_AVAILABLE_FEATURES_DICT["shimmer"] = shimmer
# Add cepstral features (MFCC handled specially below)
# _AVAILABLE_FEATURES_DICT.update(CEPSTRAL_FEATURES) # MFCC needs special handling

# List features requiring special handling (e.g., operate on full spectrogram or signal)
_SPECIAL_HANDLING_FEATURES = {"mfcc", "spectral_contrast"}
# Add placeholders here if they need signal-level calculation later, but for now treat as frame-based
# _PLACEHOLDER_FEATURES = {"hnr", "jitter", "shimmer"}


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
        frame_length: Analysis frame length in samples (default: 2048).
        hop_length: Hop length between frames in samples (default: 512).
        center: Whether to pad the signal so frames are centered (default: True).
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
    logger.debug(f"Parameters: sr={sr}, frame={frame_length}, hop={hop_length}, center={center}, window={window}")
    if feature_params:
        logger.debug(f"Feature-specific parameters: {feature_params}")

    feature_params = feature_params or {}
    results: Dict[str, NDArray[Any]] = {} # Store results here, key=feature_name, value=array
    processed_features = set() # Track which features (including derived ones like mfcc_0) are done

    # --- Handle 'all' features request ---
    if features == ['all']:
        # Combine standard features and special handling ones
        features = list(_AVAILABLE_FEATURES_DICT.keys()) + list(_SPECIAL_HANDLING_FEATURES)
        # Remove duplicates if any feature exists in both
        features = sorted(list(set(features)))
        logger.info(f"Extracting all available features: {features}")

    # --- Input Validation ---
    all_known_features = set(_AVAILABLE_FEATURES_DICT.keys()) | _SPECIAL_HANDLING_FEATURES
    unknown_features = [f for f in features if f not in all_known_features]
    if unknown_features:
        raise ValueError(f"Unknown feature(s) requested: {unknown_features}. Available: {sorted(list(all_known_features))}")

    # --- Framing ---
    # Use librosa.util.frame for consistency with feature functions that might use it
    try:
        # Note: librosa.util.frame doesn't handle centering padding itself.
        # Centering is handled by librosa.stft and feature functions like rms/zcr.
        # We frame the original signal here mainly for time-domain features that need raw frames.
        if center:
            # Pad signal for centering before framing if needed by time-domain funcs
            pad_width = frame_length // 2
            y_padded = np.pad(y, pad_width, mode='constant') # Pad with zeros for framing
            frames = librosa.util.frame(y_padded, frame_length=frame_length, hop_length=hop_length)
        else:
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

        # frames shape is (frame_length, num_frames)
        num_frames = frames.shape[1]
        if num_frames == 0:
            logger.warning("Signal is too short for the given frame/hop length, no frames generated.")
            return pd.DataFrame() if output_format == 'dataframe' else {'time': np.array([])}
        logger.debug(f"Signal framed into {num_frames} frames.")
    except Exception as e:
        raise FeatureExtractionError(f"Error during signal framing: {e}")

    # Calculate frame times (center times of frames)
    # Use n_fft=frame_length as reference for time calculation if centering
    frame_times = librosa.times_like(frames[0], sr=sr, hop_length=hop_length, n_fft=frame_length if center else None)
    # Ensure frame_times has the correct length matching num_frames
    if len(frame_times) != num_frames:
         logger.warning(f"Mismatch between number of frames ({num_frames}) and calculated frame times ({len(frame_times)}). Adjusting times array.")
         # This can happen with certain edge padding cases, adjust times array length
         frame_times = frame_times[:num_frames] # Simple truncation, might need refinement

    # --- Pre-calculate Spectrograms if needed ---
    S_mag: Optional[NDArray[np.float64]] = None
    S_mel_log: Optional[NDArray[np.float64]] = None
    fft_freqs: Optional[NDArray[np.float64]] = None
    # mel_freqs: Optional[NDArray[np.float64]] = None # Not strictly needed by MFCC func

    # Determine if any requested feature requires FFT or Mel spectrogram
    needs_fft_spectrum = any(f in FREQUENCY_DOMAIN_FEATURES for f in features)
    needs_fft_spectrogram = "spectral_contrast" in features
    needs_mel = "mfcc" in features

    if needs_fft_spectrum or needs_fft_spectrogram or needs_mel:
        logger.debug("Calculating STFT for spectral/cepstral features...")
        try:
            # Calculate STFT using librosa - handles windowing and centering
            stft_result = librosa.stft(
                y, # Use original signal `y` for librosa's STFT which handles padding
                n_fft=frame_length, # Use frame_length as n_fft by default
                hop_length=hop_length,
                win_length=frame_length, # Ensure window matches frame if possible
                window=window,
                center=center,
            )
            # Ensure STFT result has expected number of frames
            if stft_result.shape[1] != num_frames:
                 logger.warning(f"STFT frames ({stft_result.shape[1]}) mismatch framing ({num_frames}). Using STFT frame count.")
                 num_frames = stft_result.shape[1] # Adjust frame count based on STFT result
                 frame_times = frame_times[:num_frames] # Adjust times accordingly

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
                     # n_fft, hop_length, win_length, window, center are inferred from S
                     n_mels=n_mels,
                     fmin=fmin,
                     fmax=fmax,
                 )
                 # Convert to log-power Mel spectrogram (dB scale) - common input for MFCC
                 S_mel_log = librosa.power_to_db(S_mel, ref=np.max)
                 # mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax) # For context if needed
                 logger.debug(f"Log-Mel spectrogram calculated. Shape: {S_mel_log.shape}")

        except Exception as e:
            raise FeatureExtractionError(f"Error calculating STFT/Mel spectrogram: {e}")

    # --- Extract Features Iteratively ---
    for feature_name in features:
        if feature_name in processed_features:
            continue

        logger.debug(f"Extracting feature: {feature_name}")
        params = feature_params.get(feature_name, {})

        try:
            # --- Handle frame-based features (Time, ZCR, RMS, Placeholders) ---
            # Check if feature is in the main dictionary (includes time-domain, basic audio, placeholders)
            if feature_name in _AVAILABLE_FEATURES_DICT:
                func = _AVAILABLE_FEATURES_DICT[feature_name]
                # Features calculated by librosa over the whole signal (ZCR, RMS, Placeholders)
                # These functions handle their own framing based on provided args
                if feature_name in ["zero_crossing_rate", "rms_energy", "hnr", "jitter", "shimmer"]:
                     # Pass necessary framing parameters if the function needs them
                     func_params = {
                         'frame_length': frame_length,
                         'hop_length': hop_length,
                         'center': center,
                         **params # Add feature-specific params
                     }
                     # Jitter might need f0, but placeholder doesn't use it yet
                     # if feature_name == 'jitter' and 'f0' in results: # Example dependency
                     #     func_params['f0'] = results['f0']

                     feature_result = func(y=y, sr=sr, **func_params)
                     # Ensure result length matches num_frames derived from STFT/framing
                     if len(feature_result) != num_frames:
                          logger.warning(f"Length mismatch for {feature_name} ({len(feature_result)} vs {num_frames}). Adjusting.")
                          # Simple truncation/padding - might need refinement
                          if len(feature_result) > num_frames:
                              feature_result = feature_result[:num_frames]
                          else:
                              # Pad with the last value or NaN? Use NaN for placeholders.
                              pad_val = np.nan if feature_name in ["hnr", "jitter", "shimmer"] else feature_result[-1] if len(feature_result)>0 else 0
                              feature_result = np.pad(feature_result, (0, num_frames - len(feature_result)), mode='constant', constant_values=pad_val)
                     results[feature_name] = feature_result
                else:
                    # Apply other time-domain features frame-by-frame using our pre-calculated frames
                    feature_result = np.array([func(frames[:, i], **params) for i in range(num_frames)])
                    results[feature_name] = feature_result
                processed_features.add(feature_name)

            # --- Handle frequency-domain features (operating on single frame spectrum) ---
            elif feature_name in FREQUENCY_DOMAIN_FEATURES:
                if S_mag is None or fft_freqs is None:
                    # This should not happen if logic above is correct, but check defensively
                    raise FeatureExtractionError(f"FFT magnitude spectrum 'S_mag' not available for feature '{feature_name}'")
                func = FREQUENCY_DOMAIN_FEATURES[feature_name]
                # Apply frame-by-frame (column-by-column of S_mag)
                feature_result = np.array([func(S_mag[:, i], fft_freqs, **params) for i in range(num_frames)])
                results[feature_name] = feature_result
                processed_features.add(feature_name)

            # --- Handle spectral_contrast (operates on full spectrogram) ---
            elif feature_name == "spectral_contrast":
                 if S_mag is None:
                     raise FeatureExtractionError(f"FFT magnitude spectrum 'S_mag' not available for feature '{feature_name}'")
                 # Needs full magnitude spectrogram S, sr, and freqs
                 feature_result = spectral_contrast(S=S_mag, sr=sr, freqs=fft_freqs, **params)
                 # Result shape (n_bands+1, time), store each band as separate feature
                 n_bands_contrast = feature_result.shape[0] - 1
                 for i in range(n_bands_contrast):
                     band_name = f"contrast_band_{i}"
                     # Ensure length matches num_frames after potential STFT adjustment
                     results[band_name] = feature_result[i, :num_frames]
                     processed_features.add(band_name)
                 delta_name = "contrast_delta"
                 results[delta_name] = feature_result[n_bands_contrast, :num_frames] # Overall contrast
                 processed_features.add(delta_name)
                 processed_features.add(feature_name) # Mark original request as processed

            # --- Handle MFCC (operates on log-Mel spectrogram or signal) ---
            elif feature_name == "mfcc":
                 # MFCC function itself needs either y or S (log-Mel spec)
                 mfcc_params = feature_params.get('mfcc', {})
                 # Ensure necessary STFT params used for Mel spec are passed if recalculating inside mfcc
                 mfcc_params.setdefault('n_fft', frame_length)
                 mfcc_params.setdefault('hop_length', hop_length)
                 mfcc_params.setdefault('center', center)
                 mfcc_params.setdefault('window', window)
                 # Call MFCC function - prefer passing pre-calculated S_mel_log
                 feature_result = mfcc(y=y if S_mel_log is None else None, # Pass y only if S not available
                                       sr=sr,
                                       S=S_mel_log, # Pass pre-calculated log-Mel spectrogram
                                       **mfcc_params)
                 # Result shape (n_mfcc, time), store each coefficient as separate feature
                 n_mfcc_coeffs = feature_result.shape[0]
                 for i in range(n_mfcc_coeffs):
                     mfcc_name = f"mfcc_{i}"
                     # Ensure length matches num_frames after potential STFT adjustment
                     results[mfcc_name] = feature_result[i, :num_frames]
                     processed_features.add(mfcc_name)
                 processed_features.add(feature_name) # Mark original request as processed

            # --- Log success for the original feature name ---
            logger.debug(f"Successfully processed feature: {feature_name}")

        except Exception as e:
            logger.error(f"Error extracting feature '{feature_name}': {e}", exc_info=True)
            # Option: Continue to next feature or re-raise? Re-raise for now.
            raise FeatureExtractionError(f"Failed to extract feature '{feature_name}': {e}")

    # --- Format Output ---
    if not results:
        logger.warning("No features were successfully extracted.")
        return pd.DataFrame() if output_format == 'dataframe' else {'time': np.array([])}

    # Ensure all result arrays have the correct length (num_frames) before creating output
    final_results = {}
    for name, arr in results.items():
        if arr.ndim == 1 and len(arr) == num_frames:
            final_results[name] = arr
        elif arr.ndim == 1 and len(arr) > num_frames:
             logger.warning(f"Feature '{name}' array length ({len(arr)}) > num_frames ({num_frames}). Truncating.")
             final_results[name] = arr[:num_frames]
        elif arr.ndim == 1 and len(arr) < num_frames:
             logger.warning(f"Feature '{name}' array length ({len(arr)}) < num_frames ({num_frames}). Padding with NaN.")
             padded_arr = np.full(num_frames, np.nan) # Use NaN for padding
             padded_arr[:len(arr)] = arr
             final_results[name] = padded_arr
        else:
             # This case might occur if a feature function returns unexpected shape
             logger.error(f"Feature '{name}' has unexpected shape {arr.shape} or length != {num_frames}. Skipping.")

    # Add time array for dict output
    if output_format == 'dict_of_arrays':
        final_results['time'] = frame_times
        logger.info(f"Feature extraction complete. Returning dictionary of {len(final_results)} arrays.")
        return final_results

    # Create DataFrame for dataframe output
    elif output_format == 'dataframe':
        try:
            # Create DataFrame from the verified/adjusted results
            df = pd.DataFrame(final_results)
            # Set index to time
            df.index = pd.to_timedelta(frame_times, unit='s')
            df.index.name = 'time'
            logger.info(f"Feature extraction complete. DataFrame shape: {df.shape}")
            return df
        except Exception as e:
             # Catch errors during DataFrame creation (e.g., length mismatches if checks failed)
             raise FeatureExtractionError(f"Error creating output DataFrame: {e}")
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Choose 'dataframe' or 'dict_of_arrays'.")
