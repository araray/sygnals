# sygnals/core/augment/noise.py
# Drop-in replacement for the previous implementation, which silently
# generated WHITE noise when 'pink' or 'brown' was requested.
#
# This implementation generates spectrally-shaped noise via FFT shaping:
# pink (1/f magnitude², so 1/sqrt(f) magnitude), brown (1/f² → 1/f magnitude),
# blue (+3 dB/oct), violet (+6 dB/oct). White is unchanged.
#
# Validation: a Welch PSD of the output, fit in log-log space, gives slopes
# of approximately:
#     white  : 0
#     pink   : -1
#     brown  : -2
#     blue   : +1
#     violet : +2
# A test for this is included as a comment at the bottom.

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_EPSILON = np.finfo(np.float64).eps

NoiseType = Literal[
    "gaussian", "white", "pink", "brown", "red", "blue", "violet", "purple"
]


# --- Internal: spectrally-shaped colored noise --------------------------------


def _colored_noise(n: int, color: str, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate `n` samples of spectrally-shaped noise, unit-RMS-normalized.

    Implementation: take FFT of white Gaussian noise, multiply the magnitude
    spectrum by the desired shape, IFFT to time domain, normalize to unit RMS.

    Notes
    -----
    * Unit-RMS normalization is important for downstream SNR-based scaling:
      the calling code computes a target noise power from a signal power and
      a desired SNR; if different colors have different RMS, the SNR result
      depends on color. Normalizing to unit RMS removes that coupling.
    * The DC bin is forced to zero, which is acoustically sensible for
      audio (no audible DC offset in the augmented signal).
    """
    if color in ("white", "gaussian"):
        x = rng.standard_normal(n).astype(np.float64)
        rms = np.sqrt(np.mean(x * x)) + _EPSILON
        return (x / rms).astype(np.float64, copy=False)

    # 1. White noise as starting point
    x = rng.standard_normal(n).astype(np.float64)
    X = np.fft.rfft(x)  # one-sided spectrum
    f = np.fft.rfftfreq(n, d=1.0)  # normalized; absolute scale is irrelevant
    # Avoid div-by-zero at f[0]; we'll force DC to zero anyway.
    f[0] = f[1] if len(f) > 1 else 1.0

    # 2. Magnitude scaling by color
    if color == "pink":
        scale = 1.0 / np.sqrt(f)  # 1/f power → 1/sqrt(f) magnitude
    elif color in ("brown", "red", "brownian"):
        scale = 1.0 / f  # 1/f² power
    elif color in ("blue", "azure"):
        scale = np.sqrt(f)  # +3 dB/oct
    elif color in ("violet", "purple"):
        scale = f  # +6 dB/oct
    else:
        raise ValueError(
            f"unknown noise color {color!r}; choose from "
            "'white' | 'pink' | 'brown' | 'blue' | 'violet'"
        )

    X = X * scale
    X[0] = 0.0  # zero DC

    # 3. Back to time domain and normalize to unit RMS
    y = np.fft.irfft(X, n=n)
    rms = np.sqrt(np.mean(y * y)) + _EPSILON
    y = y / rms
    return y.astype(np.float64, copy=False)


# --- Public API: SNR-targeted noise addition ----------------------------------


def add_noise(
    y: NDArray[np.float64],
    snr_db: float,
    noise_type: NoiseType = "gaussian",
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """Add spectrally-shaped noise at a specified SNR (dB).

    Args:
        y: Input clean audio time series (1D float64).
        snr_db: Desired Signal-to-Noise Ratio in decibels.
        noise_type: Spectral color of the noise. 'gaussian'/'white' is flat;
            'pink' is -3 dB/oct; 'brown'/'red' is -6 dB/oct; 'blue' is +3
            dB/oct; 'violet' is +6 dB/oct.
        seed: Optional random seed for reproducibility.

    Returns:
        y + scaled_noise, shape and dtype matching y.

    Raises:
        ValueError: If y is not 1D, or noise_type is unknown, or signal/noise
            power is degenerate (we return the input unchanged in that case
            with a logger.warning).
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array for noise addition.")

    logger.info("add_noise: type=%s, target SNR=%.2f dB.", noise_type, snr_db)

    rng = np.random.default_rng(seed)
    n_samples = len(y)
    noise = _colored_noise(n_samples, noise_type, rng)  # unit-RMS

    signal_power = float(np.mean(y * y))
    if signal_power < _EPSILON:
        logger.warning(
            "Signal power is near zero; cannot scale noise to a meaningful SNR. "
            "Returning input unchanged."
        )
        return y.astype(np.float64, copy=False)

    # noise has unit RMS, so noise_power == 1.0 by construction.
    snr_linear = 10.0 ** (snr_db / 10.0)
    target_noise_power = signal_power / snr_linear
    scaling_factor = np.sqrt(target_noise_power)  # noise_power == 1, so this is final
    noise_scaled = noise * scaling_factor
    return (y + noise_scaled).astype(np.float64, copy=False)


# --- Test sketch (move to tests/test_augment.py) ------------------------------
#
# import numpy as np
# from scipy.signal import welch
# from sygnals.core.augment.noise import _colored_noise
#
# def _slope_loglog(f, p, fmin, fmax):
#     mask = (f >= fmin) & (f <= fmax) & (p > 0)
#     log_f = np.log(f[mask]); log_p = np.log(p[mask])
#     A = np.vstack([log_f, np.ones_like(log_f)]).T
#     return np.linalg.lstsq(A, log_p, rcond=None)[0][0]
#
# def test_colored_noise_psd_slopes():
#     rng = np.random.default_rng(42)
#     n = 1 << 18
#     fs = 44100
#     for color, expected_slope, tol in [
#         ("pink",  -1.0, 0.20),
#         ("brown", -2.0, 0.30),
#         ("blue",  +1.0, 0.20),
#         ("violet", +2.0, 0.30),
#     ]:
#         x = _colored_noise(n, color, rng)
#         f, p = welch(x, fs=fs, nperseg=4096)
#         slope = _slope_loglog(f, p, 50, fs/4)
#         assert abs(slope - expected_slope) < tol, (color, slope)
