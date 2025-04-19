# sygnals/core/features/cepstral.py

"""
Functions for extracting cepstral features, primarily Mel-Frequency Cepstral Coefficients (MFCCs).

Cepstral analysis is often used in speech and audio processing as the cepstrum can
help separate the source (e.g., vocal cords) from the filter (e.g., vocal tract),
making features like MFCCs effective for tasks like speech recognition and speaker identification.
"""

import logging
import librosa
import numpy as np
from numpy.typing import NDArray
# Import necessary types
from typing import Optional, Dict, Any, Union # Added Union

logger = logging.getLogger(__name__)

def mfcc(
    y: Optional[NDArray[np.float64]] = None,
    sr: Optional[int] = None, # Made sr optional, required only if y is provided
    S: Optional[NDArray[np.float64]] = None, # Can compute from signal y or spectrogram S
    n_mfcc: int = 13,
    dct_type: int = 2,
    norm: Optional[Literal['ortho']] = 'ortho', # Librosa uses Optional[Literal['ortho']]
    lifter: float = 0.0, # Librosa uses float for lifter
    **kwargs: Any # Pass additional args to librosa.feature.mfcc (n_fft, hop_length, etc.)
) -> NDArray[np.float64]:
    """
    Computes Mel-Frequency Cepstral Coefficients (MFCCs) using librosa.

    MFCCs represent the short-term power spectrum of a sound on a non-linear
    Mel scale of frequency, followed by a Discrete Cosine Transform (DCT) to
    decorrelate the coefficients. They are widely used in audio analysis.

    This function can compute MFCCs either directly from an audio time series `y`
    or from a pre-computed log-power Mel spectrogram `S`.

    Args:
        y: Audio time series (1D float64). Required if `S` is not provided.
        sr: Sampling rate (Hz). Required if `y` is provided.
        S: Pre-computed log-power Mel spectrogram (e.g., from
           `librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, **kwargs))`).
           If provided, `y` and `sr` are ignored for the Mel spectrogram calculation
           step itself, but `sr` might still be needed if time-dependent parameters
           like `hop_length` are passed via `kwargs` without being implicitly
           derivable from `S`. Shape: (n_mels, n_frames).
        n_mfcc: Number of MFCCs to return (typically 13 to 40). The 0th coefficient
                (MFCC 0) often represents overall log-energy and is sometimes
                excluded in certain applications. (Default: 13)
        dct_type: Discrete Cosine Transform type (1, 2, or 3). Type 2 is standard. (Default: 2)
        norm: DCT normalization type ('ortho' or None). 'ortho' ensures Parseval's
              theorem holds (preserves energy). (Default: 'ortho')
        lifter: Cepstral liftering coefficient. If > 0, applies sinusoidal liftering
                which can de-emphasize higher-order MFCCs, potentially improving
                robustness to noise. `lifter = 22` is a common value for speech.
                0 means no liftering. (Default: 0.0)
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
                  - `power`: Exponent for the magnitude melspectrogram (default: 2.0).

    Returns:
        MFCC sequence (shape: (n_mfcc, n_frames), dtype: float64). Each column
        corresponds to an MFCC vector for a single time frame.

    Raises:
        ValueError: If neither `y` nor `S` is provided, or if `sr` is missing when `y` is provided.
        Exception: For errors during librosa calculation.

    Example:
        >>> sr = 22050
        >>> y = librosa.chirp(fmin=100, fmax=5000, sr=sr, duration=2)
        >>> # Compute MFCCs directly from audio
        >>> mfccs_from_y = mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512)
        >>> print(mfccs_from_y.shape)
        (20, 87) # Example shape
        >>> # Compute MFCCs from pre-calculated log-Mel spectrogram
        >>> S_mel_log = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512))
        >>> mfccs_from_S = mfcc(S=S_mel_log, sr=sr, n_mfcc=20) # sr might still be needed by librosa internally
        >>> np.allclose(mfccs_from_y, mfccs_from_S)
        True
    """
    logger.debug(f"Calculating MFCCs: n_mfcc={n_mfcc}, dct_type={dct_type}, norm={norm}, lifter={lifter}, kwargs={kwargs}")

    if S is None and y is None:
        raise ValueError("Either audio time series 'y' or Mel spectrogram 'S' must be provided.")
    if S is None and sr is None:
         raise ValueError("Sampling rate 'sr' must be provided when calculating MFCCs from time series 'y'.")
    if S is not None and y is not None:
        logger.warning("Both 'y' and 'S' provided for MFCC calculation. Using pre-computed 'S'. "
                       "Ensure 'S' is a log-power Mel spectrogram for correct results.")

    try:
        # librosa.feature.mfcc handles the logic of using y or S
        # Pass sr regardless, as librosa might use it internally even if S is provided
        # (e.g., for default fmax calculation if not specified in kwargs)
        mfccs_result = librosa.feature.mfcc(
            y=y,
            sr=sr,
            S=S, # Pass the log-power mel spectrogram if available
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            norm=norm,
            lifter=float(lifter), # Ensure lifter is float
            **kwargs
        )
        # Ensure output type is float64
        return mfccs_result.astype(np.float64, copy=False)
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
