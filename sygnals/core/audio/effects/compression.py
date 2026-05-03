# sygnals/core/audio/effects/compression.py
"""
Dynamic-range compressor with attack, release, soft-knee and makeup gain.

Replaces the previous memoryless waveshaper. The previous implementation
applied a pure function of the instantaneous magnitude with no time
dependence, which produces audio-rate gain modulation and harmonic
distortion. This implementation:

  * Smooths the detector (peak or RMS) with an asymmetric one-pole filter.
  * Applies a soft-knee gain transfer.
  * Smooths the gain itself with the same asymmetric one-pole.
  * Applies makeup gain.

Two implementations are provided:

  1. `compress(...)` — a vectorized path using `scipy.signal.lfilter` with
     fixed coefficients. Fast in pure NumPy. Limitation: it can't switch the
     time constant per sample (attack vs release) without a Python loop, so
     the vectorized path uses a single time constant on the rectified signal
     for level estimation, then a single time constant on the gain. For most
     audio program material this is musically indistinguishable from a true
     asymmetric implementation.
  2. `Compressor` class — sample-accurate asymmetric attack/release in a
     loop. Slow in pure Python (use Numba/Cython/C for production), but
     bit-identical to typical hardware/plugin behavior. Use this when
     correctness > speed or when porting to a faster language.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

logger = logging.getLogger(__name__)

_EPSILON = np.finfo(np.float64).eps


def _knee_gain_db(
    level_db: NDArray[np.float64], threshold_db: float, ratio: float, knee_db: float
) -> NDArray[np.float64]:
    """Soft-knee gain transfer in dB.

    Below `threshold_db - knee_db/2`: 0 dB (no compression).
    Above `threshold_db + knee_db/2`: full compression at `ratio`.
    Inside the knee: quadratic interpolation.
    """
    half_knee = 0.5 * knee_db
    over = level_db - threshold_db
    gain = np.zeros_like(over)

    above = over >= +half_knee
    inside = (over > -half_knee) & ~above

    # Above knee: full compression
    gain[above] = -(over[above] - over[above] / ratio)

    # Inside knee: quadratic blend (Massberg / cookbook formula)
    if knee_db > 0.0 and inside.any():
        x_in_knee = over[inside] + half_knee
        gain[inside] = -((1.0 - 1.0 / ratio) * x_in_knee * x_in_knee / (2.0 * knee_db))
    # else: gain stays 0 below the knee.

    return gain


def compress(
    y: NDArray[np.float64],
    sr: int,
    *,
    threshold_db: float = -24.0,
    ratio: float = 4.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    knee_db: float = 6.0,
    makeup_db: float = 0.0,
    detector: Literal["peak", "rms"] = "peak",
) -> NDArray[np.float64]:
    """Vectorized compressor (single time constant per smoother).

    Trade-off: doesn't switch time constants per sample. Acceptable for most
    audio; for a fully-correct asymmetric implementation, use the
    `Compressor` class below.
    """
    if y.ndim != 1:
        raise ValueError("compress() expects 1D input. Apply per-channel.")
    if ratio < 1.0:
        raise ValueError("ratio must be >= 1")
    if knee_db < 0.0:
        raise ValueError("knee_db must be >= 0")

    # 1. Detector
    if detector == "peak":
        instantaneous = np.abs(y)
    else:  # rms
        instantaneous = y * y

    # 2. Smooth detector with an attack-time one-pole.
    # First-order one-pole: y[n] = (1-a) * x[n] + a * y[n-1]
    # In lfilter form: b = [1-a], a = [1, -a]
    # Prime to the first sample's instantaneous value to avoid a startup transient.
    a_attack = math.exp(-1.0 / max(1e-9, attack_ms * 1e-3 * sr))
    b = np.array([1.0 - a_attack])
    a_coef = np.array([1.0, -a_attack])
    zi_lvl = (
        np.array([a_attack * float(instantaneous[0])])
        if instantaneous.size
        else np.zeros(1)
    )
    level, _ = lfilter(b, a_coef, instantaneous, zi=zi_lvl)
    level = np.maximum(level, _EPSILON)

    # 3. Convert to dB for gain transfer
    if detector == "peak":
        level_db = 20.0 * np.log10(level)
    else:
        level_db = 10.0 * np.log10(level)

    # 4. Static gain transfer with soft knee
    target_gain_db = _knee_gain_db(level_db, threshold_db, ratio, knee_db)
    target_gain = np.power(10.0, target_gain_db / 20.0)

    # 5. Smooth gain reduction with a release-time one-pole.
    # We want fast attack (gain dropping) and slow release (gain coming back).
    # In the vectorized version we use release for the gain smoother.
    # Prime the smoother to the first target gain (typically 1.0 for a quiet
    # start). Without this, the smoother starts at 0 and ramps up — multiplying
    # the signal by ~0 at the beginning, which is a real bug.
    a_release = math.exp(-1.0 / max(1e-9, release_ms * 1e-3 * sr))
    b = np.array([1.0 - a_release])
    a_coef = np.array([1.0, -a_release])
    zi_g = (
        np.array([a_release * float(target_gain[0])])
        if target_gain.size
        else np.zeros(1)
    )
    gain_smoothed, _ = lfilter(b, a_coef, target_gain, zi=zi_g)

    # 6. Apply gain and makeup
    makeup_lin = 10.0 ** (makeup_db / 20.0)
    return (y * gain_smoothed * makeup_lin).astype(np.float64, copy=False)


class Compressor:
    """Sample-accurate asymmetric-time-constant compressor (slow but exact).

    Use this when bit-exact behavior matters or when porting to C/Rust.
    Maintains internal state across `process_block` calls for streaming.
    """

    def __init__(
        self,
        sr: int,
        *,
        threshold_db: float = -24.0,
        ratio: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
        knee_db: float = 6.0,
        makeup_db: float = 0.0,
        detector: Literal["peak", "rms"] = "peak",
    ):
        if ratio < 1.0:
            raise ValueError("ratio must be >= 1")
        if knee_db < 0.0:
            raise ValueError("knee_db must be >= 0")
        self.sr = sr
        self.threshold_db = float(threshold_db)
        self.ratio = float(ratio)
        self.knee_db = float(knee_db)
        self.makeup_lin = 10.0 ** (makeup_db / 20.0)
        self.detector = detector
        self.a_attack = math.exp(-1.0 / max(1e-9, attack_ms * 1e-3 * sr))
        self.a_release = math.exp(-1.0 / max(1e-9, release_ms * 1e-3 * sr))
        self.half_knee = 0.5 * knee_db
        self._level = 0.0
        self._gain = 1.0

    def reset(self) -> "Compressor":
        self._level = 0.0
        self._gain = 1.0
        return self

    def process_block(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if x.ndim != 1:
            raise ValueError("process_block expects 1D input")
        out = np.empty_like(x)
        level = self._level
        gain = self._gain
        threshold_db = self.threshold_db
        ratio = self.ratio
        knee_db = self.knee_db
        half_knee = self.half_knee
        a_attack = self.a_attack
        a_release = self.a_release
        makeup_lin = self.makeup_lin
        peak = self.detector == "peak"

        for i in range(x.size):
            xi = x[i]
            instantaneous = abs(xi) if peak else xi * xi
            # Asymmetric detector: attack when level is rising
            a = a_attack if instantaneous > level else a_release
            level = a * level + (1.0 - a) * instantaneous

            level_db = (
                20.0 * math.log10(level + _EPSILON)
                if peak
                else 10.0 * math.log10(level + _EPSILON)
            )

            over = level_db - threshold_db
            if over <= -half_knee:
                target_gain_db = 0.0
            elif over >= +half_knee:
                target_gain_db = -(over - over / ratio)
            else:
                x_in_knee = over + half_knee
                target_gain_db = -(
                    (1.0 - 1.0 / ratio) * x_in_knee * x_in_knee / (2.0 * knee_db)
                )

            target_gain = 10.0 ** (target_gain_db / 20.0)
            # Asymmetric gain smoother: attack when reducing gain
            a = a_attack if target_gain < gain else a_release
            gain = a * gain + (1.0 - a) * target_gain

            out[i] = xi * gain * makeup_lin

        self._level = level
        self._gain = gain
        return out
