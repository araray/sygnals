# sygnals/core/features/peak_interp.py
"""
Parabolic and Jacobsen peak interpolation for sub-bin frequency estimation.

The naive `argmax` over an FFT magnitude spectrum has bin-resolution
quantization error of up to fs/(2N). Parabolic interpolation reduces that
error to O((fs/(2N))^3) for a parabola fit of three samples around the
peak; Jacobsen's complex-domain estimator [2] is even better for sinusoids
in noise.

References
----------
[1] J. O. Smith III, "Spectral Audio Signal Processing", CCRMA Stanford.
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral.html
[2] E. Jacobsen and P. Kootsookos, "Fast, Accurate Frequency Estimators",
    IEEE Signal Process. Mag., 24(3):123-125, May 2007.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def parabolic_peak_bin(mag: NDArray[np.float64]) -> Tuple[float, float]:
    """Parabolic interpolation around the largest bin.

    Args:
        mag: 1D magnitude (or log-magnitude — the latter often gives a
            slightly better fit because a sinusoidal peak in dB is closer
            to a parabola).

    Returns:
        (bin_offset, peak_mag) where bin_offset is in (-0.5, 0.5).

    The interpolated peak frequency in Hz is `(k + bin_offset) * df`
    where `k = argmax(mag)` and `df = fs / N_fft`.
    """
    if mag.ndim != 1:
        raise ValueError("mag must be 1D")
    k = int(np.argmax(mag))
    if k == 0 or k == mag.size - 1:
        return 0.0, float(mag[k])

    a, b, c = mag[k - 1], mag[k], mag[k + 1]
    denom = a - 2.0 * b + c
    if abs(denom) < np.finfo(np.float64).eps:
        return 0.0, float(b)

    offset = 0.5 * (a - c) / denom
    # Interpolated peak magnitude (parabola apex)
    peak = b - 0.25 * (a - c) * offset
    return float(offset), float(peak)


def jacobsen_peak_bin(spec: NDArray[np.complex128]) -> float:
    """Jacobsen's complex-spectrum frequency estimator.

    Operates on the complex DFT (not the magnitude), so it benefits from
    phase information. Better than parabolic for sinusoids buried in
    moderate noise.

    Args:
        spec: 1D complex DFT (one-sided or two-sided is fine).

    Returns:
        bin_offset in (-0.5, 0.5).
    """
    mag = np.abs(spec)
    k = int(np.argmax(mag))
    if k == 0 or k == spec.size - 1:
        return 0.0
    X_m = spec[k - 1]
    X_0 = spec[k]
    X_p = spec[k + 1]
    delta = -np.real((X_p - X_m) / (2.0 * X_0 - X_m - X_p))
    # Clamp to (-0.5, 0.5) to guard against pathological cases
    if not np.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -0.5, 0.5))


def dominant_frequency_interp(
    mag: NDArray[np.float64],
    fs: float,
    *,
    method: str = "parabolic",
) -> float:
    """High-level wrapper: interpolated dominant-frequency in Hz.

    Args:
        mag: 1D magnitude spectrum (length N_fft//2 + 1 for r2c FFT).
        fs: Sample rate.
        method: "parabolic" or "argmax".

    Returns:
        Dominant frequency in Hz.
    """
    if mag.size < 2:
        return 0.0
    n_fft = (mag.size - 1) * 2
    df = fs / n_fft
    k = int(np.argmax(mag))
    if method == "argmax":
        return float(k * df)
    if method == "parabolic":
        offset, _ = parabolic_peak_bin(mag)
        return float((k + offset) * df)
    raise ValueError(f"unknown method {method!r}")
