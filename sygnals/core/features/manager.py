# sygnals/core/features/manager.py

"""
Manages the extraction of various features from signal data.
Handles framing, calling appropriate feature functions, and aggregating results.
"""

import logging
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Dict, Any, Optional, Tuple

# Import feature dictionaries and functions
from .time_domain import TIME_DOMAIN_FEATURES
from .frequency_domain import FREQUENCY_DOMAIN_FEATURES, spectral_contrast
from .cepstral import CEPSTRAL_FEATURES, mfcc
from ..audio.features import zero_crossing_rate, rms_energy # Import audio features directly if needed

# Import DSP functions needed for spectral features
from ..dsp import compute_fft

logger = logging.getLogger(__name__)

# Combine available features (excluding those needing special handling like MFCC/Contrast initially)
# Manager will decide how to call these based on requirements (frame vs spectrogram)
AVAILABLE_FEATURES = {**TIME_DOMAIN_FEATURES}
# Add audio features that operate per-frame
AVAILABLE_FEATURES["zero_crossing_rate"] = zero_crossing_rate
AVAILABLE_FEATURES["rms_energy"] = rms_energy


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
) -> Union[pd.DataFrame, Dict[str, NDArray[np.float64]]]:
    """
    Extracts specified features from an audio signal.

    Handles framing, FFT calculation (if needed), calling feature functions,
    and formatting the output.

    Args:
        y: Input audio time series (float64).
        sr: Sampling rate.
        features: List of feature names to extract (e.g., ['rms_energy', 'spectral_centroid', 'mfcc']).
        frame_length: Analysis frame length in samples.
        hop_length: Hop length between frames in samples.
        center: Whether to pad the signal so frames are centered.
        window: Window function to apply for FFT-based features.
        feature_params: Dictionary containing parameters specific to certain features.
                        Example: {'mfcc': {'n_mfcc': 20}, 'spectral_rolloff': {'roll_percent': 0.9}}
        output_format: Format for the returned features ('dataframe' or 'dict_of_arrays').

    Returns:
        A Pandas DataFrame or a dictionary of NumPy arrays containing the extracted features.
        DataFrame index/keys correspond to frame times. Dictionary keys are feature names.

    Raises:
        ValueError: If an unknown feature is requested.
        FeatureExtractionError: If an error occurs during extraction.
    """
    logger.info(f"Starting feature extraction for features: {features}")
    logger.debug(f"Parameters: sr={sr}, frame={frame_length}, hop={hop_length}, center={center}, window={window}")
    if feature_params:
        logger.debug(f"Feature-specific parameters: {feature_params}")

    feature_params = feature_params or {}
    results: Dict[str, NDArray[np.float64]] = {}
    processed_features = set()

    # --- Input Validation ---
    unknown_features = [f for f in features if f not in AVAILABLE_FEATURES and f not in CEPSTRAL_FEATURES and f != "spectral_contrast"]
    if unknown_features:
        available = list(AVAILABLE_FEATURES.keys()) + list(CEPSTRAL_FEATURES.keys()) + ["spectral_contrast"]
        raise ValueError(f"Unknown feature(s) requested: {unknown_features}. Available: {available}")

    # --- Framing ---
    # Use librosa.util.frame for consistency with feature functions
    try:
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        # frames shape is (frame_length, num_frames)
        num_frames = frames.shape[1]
        logger.debug(f"Signal framed into {num_frames} frames.")
    except Exception as e:
        raise FeatureExtractionError(f"Error during signal framing: {e}")

    # Calculate frame times (useful for DataFrame index)
    frame_times = librosa.times_like(frames[0], sr=sr, hop_length=hop_length, n_fft=frame_length) # Use frame_length for n_fft reference if centered

    # --- Pre-calculate Spectrograms if needed ---
    S_mag: Optional[NDArray[np.float64]] = None
    S_mel_log: Optional[NDArray[np.float64]] = None
    fft_freqs: Optional[NDArray[np.float64]] = None
    mel_freqs: Optional[NDArray[np.float64]] = None # Not directly used by MFCC but good context

    needs_fft = any(f in FREQUENCY_DOMAIN_FEATURES for f in features) or "spectral_contrast" in features
    needs_mel = "mfcc" in features

    if needs_fft or needs_mel:
        logger.debug("Calculating STFT for spectral/cepstral features...")
        try:
            # Calculate STFT using librosa for consistency
            stft_result = librosa.stft(
                y,
                n_fft=frame_length, # Use frame_length as n_fft by default
                hop_length=hop_length,
                win_length=frame_length, # Ensure window matches frame
                window=window,
                center=center,
            )
            S_mag = np.abs(stft_result) # Magnitude spectrogram
            fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
            logger.debug(f"STFT calculated. Shape: {S_mag.shape}")

            if needs_mel:
                 logger.debug("Calculating Mel spectrogram for MFCC...")
                 # Get MFCC specific params
                 mfcc_p = feature_params.get('mfcc', {})
                 n_mels = mfcc_p.get('n_mels', 128) # Default librosa n_mels
                 fmin = mfcc_p.get('fmin', 0)
                 fmax = mfcc_p.get('fmax', sr / 2.0)
                 power = mfcc_p.get('power', 2.0) # Power for melspectrogram

                 S_mel = librosa.feature.melspectrogram(
                     S=S_mag**power, # Pass power spectrogram
                     sr=sr,
                     n_fft=frame_length, # Must match STFT n_fft
                     hop_length=hop_length, # Must match STFT hop_length
                     n_mels=n_mels,
                     fmin=fmin,
                     fmax=fmax,
                     # win_length, window, center are implicitly handled by S
                 )
                 S_mel_log = librosa.power_to_db(S_mel, ref=np.max)
                 mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
                 logger.debug(f"Log-Mel spectrogram calculated. Shape: {S_mel_log.shape}")

        except Exception as e:
            raise FeatureExtractionError(f"Error calculating STFT/Mel spectrogram: {e}")

    # --- Extract Features ---
    for feature_name in features:
        if feature_name in processed_features:
            continue

        logger.debug(f"Extracting feature: {feature_name}")
        params = feature_params.get(feature_name, {})

        try:
            if feature_name in AVAILABLE_FEATURES:
                func = AVAILABLE_FEATURES[feature_name]
                # Check if function needs frame/hop lengths (like ZCR, RMS)
                if feature_name in ["zero_crossing_rate", "rms_energy"]:
                     # These are calculated by librosa over the whole signal
                     feature_result = func(y=y, frame_length=frame_length, hop_length=hop_length, center=center, **params)
                     results[feature_name] = feature_result[:num_frames] # Ensure length matches frames
                else:
                    # Apply frame-by-frame for other time-domain features
                    feature_result = np.array([func(frames[:, i], **params) for i in range(num_frames)])
                    results[feature_name] = feature_result

            elif feature_name in FREQUENCY_DOMAIN_FEATURES:
                if S_mag is None or fft_freqs is None:
                    raise FeatureExtractionError(f"FFT magnitude spectrum 'S_mag' not available for feature '{feature_name}'")
                func = FREQUENCY_DOMAIN_FEATURES[feature_name]
                # Apply frame-by-frame (column-by-column of S_mag)
                feature_result = np.array([func(S_mag[:, i], fft_freqs, **params) for i in range(num_frames)])
                results[feature_name] = feature_result

            elif feature_name == "spectral_contrast":
                 if S_mag is None:
                     raise FeatureExtractionError(f"FFT magnitude spectrum 'S_mag' not available for feature '{feature_name}'")
                 # Needs full spectrogram S, sr, and freqs
                 feature_result = spectral_contrast(S=S_mag, sr=sr, freqs=fft_freqs, **params)
                 # Result shape (n_bands+1, time), store each band as separate feature
                 n_bands_contrast = feature_result.shape[0] - 1
                 for i in range(n_bands_contrast):
                     results[f"contrast_band_{i}"] = feature_result[i, :num_frames]
                 results["contrast_delta"] = feature_result[n_bands_contrast, :num_frames] # Overall contrast

            elif feature_name == "mfcc":
                 if S_mel_log is None and y is None: # Need y if S_mel_log wasn't precalculated
                     raise FeatureExtractionError(f"Neither Log-Mel spectrogram 'S_mel_log' nor signal 'y' available for feature '{feature_name}'")
                 # Pass relevant params from feature_params['mfcc']
                 mfcc_params = feature_params.get('mfcc', {})
                 # Ensure n_fft, hop_length etc. used here match spectrogram calculation if S is used
                 mfcc_params.setdefault('n_fft', frame_length)
                 mfcc_params.setdefault('hop_length', hop_length)
                 mfcc_params.setdefault('center', center)
                 # Call MFCC function
                 feature_result = mfcc(y=y, sr=sr, S=S_mel_log, **mfcc_params)
                 # Result shape (n_mfcc, time), store each coefficient as separate feature
                 n_mfcc_coeffs = feature_result.shape[0]
                 for i in range(n_mfcc_coeffs):
                     results[f"mfcc_{i}"] = feature_result[i, :num_frames]

            processed_features.add(feature_name)
            # Add derived features (contrast bands, mfcc coeffs) to processed set
            if feature_name == "spectral_contrast":
                 processed_features.update([f"contrast_band_{i}" for i in range(n_bands_contrast)] + ["contrast_delta"])
            if feature_name == "mfcc":
                 processed_features.update([f"mfcc_{i}" for i in range(n_mfcc_coeffs)])


        except Exception as e:
            logger.error(f"Error extracting feature '{feature_name}': {e}", exc_info=True)
            # Optionally continue to next feature or re-raise
            raise FeatureExtractionError(f"Failed to extract feature '{feature_name}': {e}")

    # --- Format Output ---
    if not results:
        logger.warning("No features were successfully extracted.")
        return pd.DataFrame() if output_format == 'dataframe' else {}

    # Ensure all result arrays have the correct length (num_frames)
    final_results = {}
    for name, arr in results.items():
        if arr.ndim == 1 and len(arr) == num_frames:
            final_results[name] = arr
        elif arr.ndim == 1 and len(arr) > num_frames:
             logger.warning(f"Feature '{name}' array length ({len(arr)}) > num_frames ({num_frames}). Truncating.")
             final_results[name] = arr[:num_frames]
        elif arr.ndim == 1 and len(arr) < num_frames:
             logger.warning(f"Feature '{name}' array length ({len(arr)}) < num_frames ({num_frames}). Padding with NaN.")
             padded_arr = np.full(num_frames, np.nan)
             padded_arr[:len(arr)] = arr
             final_results[name] = padded_arr
        else:
             logger.error(f"Feature '{name}' has unexpected shape {arr.shape} or length. Skipping.")


    if output_format == 'dataframe':
        try:
            df = pd.DataFrame(final_results)
            df.index = pd.to_timedelta(frame_times, unit='s')
            df.index.name = 'time'
            logger.info(f"Feature extraction complete. DataFrame shape: {df.shape}")
            return df
        except Exception as e:
             raise FeatureExtractionError(f"Error creating output DataFrame: {e}")
    elif output_format == 'dict_of_arrays':
        logger.info(f"Feature extraction complete. Returning dictionary of arrays.")
        # Add frame_times to the dictionary
        final_results['time'] = frame_times
        return final_results
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
