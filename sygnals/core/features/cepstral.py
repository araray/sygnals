# sygnals/core/features/cepstral.py

"""
Functions for extracting cepstral features, primarily MFCCs.
"""

import logging
import librosa
import numpy as np
from numpy.typing import NDArray
from typing import Optional

logger = logging.getLogger(__name__)

def mfcc(
    y: Optional[NDArray[np.float64]] = None,
    sr: int = 22050,
    S: Optional[NDArray[np.float64]] = None, # Can compute from signal y or spectrogram S
    n_mfcc: int = 13,
    dct_type: int = 2,
    norm: str = 'ortho',
    lifter: int = 0,
    **kwargs # Pass additional args to librosa.feature.mfcc (n_fft, hop_length, etc.)
) -> NDArray[np.float64]:
    """
    Computes Mel-Frequency Cepstral Coefficients (MFCCs) using librosa.

    Can compute directly from audio time series `y` or from a pre-computed
    Mel spectrogram `S`.

    Args:
        y: Audio time series (float64). Required if `S` is not provided.
        sr: Sampling rate. Required if `y` is provided.
        S: Pre-computed log-power Mel spectrogram (e.g., from librosa.feature.melspectrogram).
           If provided, `y` and `sr` are ignored for MFCC calculation itself,
           but might be needed for kwargs like hop_length if not specified.
        n_mfcc: Number of MFCCs to return.
        dct_type: Discrete Cosine Transform type (1, 2, or 3). Default: 2.
        norm: DCT normalization type ('ortho' or None). Default: 'ortho'.
        lifter: Cepstral liftering coefficient. 0 for no liftering. Default: 0.
        **kwargs: Additional keyword arguments passed to `librosa.feature.mfcc`,
                  such as `n_fft`, `hop_length`, `win_length`, `window`,
                  `center`, `pad_mode`, `power`, `n_mels`, `fmin`, `fmax`.

    Returns:
        MFCC sequence (shape: (n_mfcc, time)).
    """
    logger.debug(f"Calculating MFCCs: n_mfcc={n_mfcc}, dct_type={dct_type}, norm={norm}, lifter={lifter}, kwargs={kwargs}")

    if S is None and y is None:
        raise ValueError("Either audio time series 'y' or Mel spectrogram 'S' must be provided.")
    if S is not None and y is not None:
        logger.warning("Both 'y' and 'S' provided for MFCC calculation. Using pre-computed 'S'.")

    try:
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            S=S,
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            norm=norm,
            lifter=lifter,
            **kwargs
        )
        return mfccs.astype(np.float64, copy=False)
    except Exception as e:
        logger.error(f"Error calculating MFCCs: {e}")
        raise

# Dictionary mapping feature names to functions
# Note: MFCC requires specific setup (y or S, sr, kwargs), handled in manager
CEPSTRAL_FEATURES = {
    "mfcc": mfcc,
    # Add other cepstral features like LPC if needed
}
