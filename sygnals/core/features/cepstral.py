# sygnals/core/features/cepstral.py

"""
Functions for extracting cepstral features, primarily Mel-Frequency Cepstral Coefficients (MFCCs).
Cepstral analysis is often used in speech and audio processing as the cepstrum can
separate the source (e.g., vocal cords) from the filter (e.g., vocal tract).
"""

import logging
import librosa
import numpy as np
from numpy.typing import NDArray
# Import necessary types
from typing import Optional, Dict, Any # Added Any and Dict for CEPSTRAL_FEATURES

logger = logging.getLogger(__name__)

def mfcc(
    y: Optional[NDArray[np.float64]] = None,
    sr: int = 22050,
    S: Optional[NDArray[np.float64]] = None, # Can compute from signal y or spectrogram S
    n_mfcc: int = 13,
    dct_type: int = 2,
    norm: str = 'ortho',
    lifter: int = 0,
    **kwargs: Any # Pass additional args to librosa.feature.mfcc (n_fft, hop_length, etc.)
) -> NDArray[np.float64]:
    """
    Computes Mel-Frequency Cepstral Coefficients (MFCCs) using librosa.

    MFCCs are widely used features in automatic speech recognition (ASR) and
    music information retrieval (MIR). They represent the short-term power
    spectrum of a sound on a nonlinear mel scale of frequency, followed by
    a discrete cosine transform (DCT) to decorrelate the coefficients.

    Can compute directly from audio time series `y` or from a pre-computed
    log-power Mel spectrogram `S`.

    Args:
        y: Audio time series (float64). Required if `S` is not provided.
        sr: Sampling rate. Required if `y` is provided or if needed for kwargs
            like `hop_length` when `S` is provided but `hop_length` isn't.
        S: Pre-computed log-power Mel spectrogram (e.g., from librosa.power_to_db(librosa.feature.melspectrogram)).
           If provided, `y` is ignored for MFCC calculation itself, but `sr` might
           still be needed if `hop_length` or other time-dependent parameters
           are passed via `kwargs` and not already accounted for in `S`.
        n_mfcc: Number of MFCCs to return (typically 13 to 40). The 0th coefficient
                often represents overall energy and is sometimes discarded.
        dct_type: Discrete Cosine Transform type (1, 2, or 3). Type 2 is standard.
        norm: DCT normalization type ('ortho' or None). 'ortho' ensures Parseval's theorem.
        lifter: Cepstral liftering coefficient. If > 0, applies sinusoidal liftering
                which can de-emphasize higher-order MFCCs. 0 means no liftering.
        **kwargs: Additional keyword arguments passed to `librosa.feature.mfcc`,
                  which in turn passes them to `librosa.feature.melspectrogram` if `S`
                  is not provided. Common args include:
                  - `n_fft`: FFT window size (default: 2048).
                  - `hop_length`: Hop length for STFT (default: 512).
                  - `win_length`: Window length (default: n_fft).
                  - `window`: Window function (default: 'hann').
                  - `n_mels`: Number of Mel bands (default: 128).
                  - `fmin`: Minimum frequency for Mel bands (default: 0).
                  - `fmax`: Maximum frequency for Mel bands (default: sr/2.0).

    Returns:
        MFCC sequence (shape: (n_mfcc, time_frames)). Each column corresponds to an MFCC
        vector for a single time frame.
    """
    logger.debug(f"Calculating MFCCs: n_mfcc={n_mfcc}, dct_type={dct_type}, norm={norm}, lifter={lifter}, kwargs={kwargs}")

    if S is None and y is None:
        raise ValueError("Either audio time series 'y' or Mel spectrogram 'S' must be provided.")
    if S is not None and y is not None:
        logger.warning("Both 'y' and 'S' provided for MFCC calculation. Using pre-computed 'S'. Ensure 'S' is a log-power Mel spectrogram.")
    if S is None and y is not None and sr is None:
         raise ValueError("Sampling rate 'sr' must be provided when calculating MFCCs from time series 'y'.")

    try:
        # librosa.feature.mfcc handles the logic of using y or S
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            S=S, # Pass the log-power mel spectrogram if available
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            norm=norm,
            lifter=lifter,
            **kwargs
        )
        # Ensure output type
        return mfccs.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error calculating MFCCs: {e}")
        raise

# Dictionary mapping feature names to functions for the manager
# Note: MFCC requires specific setup (y or S, sr, kwargs), which needs
# to be handled correctly by the calling code (e.g., Feature Manager).
CEPSTRAL_FEATURES: Dict[str, Any] = {
    "mfcc": mfcc,
    # Add other cepstral features like Linear Prediction Coefficients (LPC) if needed
    # "lpc": lpc_function, # Example
}
