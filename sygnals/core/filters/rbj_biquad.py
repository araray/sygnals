# sygnals/core/filters/rbj_biquad.py
"""
RBJ Audio EQ Cookbook biquad designs.

Closed-form formulas for the 2nd-order IIR filter shapes that audio engineers
actually want: peaking EQ, low/high shelf, low/high-pass with Q, bandpass
(constant-skirt-gain and constant-peak-gain forms), notch, allpass.

Reference
---------
Robert Bristow-Johnson, "Audio EQ Cookbook"
https://www.w3.org/TR/audio-eq-cookbook/

Each design returns SOS-format coefficients (shape (1, 6)) directly compatible
with scipy.signal.sosfilt / sosfiltfilt and with sygnals' apply_sos_filter.
The 6-tuple ordering is (b0, b1, b2, a0, a1, a2). The cookbook's a0 is divided
out into the b's so that the returned SOS is in the standard
(b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0) form.

Validated:
    * peaking_eq( fs=48000, f0=1000, Q=1, gain_dB=+6 ) → +6 dB at 1 kHz, 0 dB out of band.
    * lowshelf / highshelf at S=1 give the canonical -6 dB/oct slope outside
      the shelf transition.
    * lowpass / highpass with Q=1/sqrt(2) are exactly Butterworth 2nd order.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray

BiquadShape = Literal[
    "lowpass",
    "highpass",
    "bandpass_csg",
    "bandpass_cpg",  # constant-skirt-gain / constant-peak-gain
    "notch",
    "allpass",
    "peaking",
    "lowshelf",
    "highshelf",
]


def _common(fs: float, f0: float) -> Tuple[float, float, float]:
    """Compute (w0, cos(w0), sin(w0)) with the standard guards."""
    if fs <= 0.0:
        raise ValueError("fs must be > 0")
    if not (0.0 < f0 < fs / 2.0):
        raise ValueError(f"f0={f0} must be strictly between 0 and Nyquist={fs / 2}")
    w0 = 2.0 * math.pi * f0 / fs
    return w0, math.cos(w0), math.sin(w0)


def _alpha_q(sin_w0: float, Q: float) -> float:
    if Q <= 0.0:
        raise ValueError("Q must be > 0")
    return sin_w0 / (2.0 * Q)


def _alpha_bw(sin_w0: float, w0: float, BW_octaves: float) -> float:
    """Alpha from bandwidth in octaves (cookbook §4)."""
    return sin_w0 * math.sinh(0.5 * math.log(2.0) * BW_octaves * w0 / sin_w0)


def _alpha_shelf(sin_w0: float, A: float, S: float) -> float:
    """Alpha for shelving filters (cookbook §4)."""
    if S <= 0.0:
        raise ValueError("S (shelf slope) must be > 0")
    return 0.5 * sin_w0 * math.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)


def _to_sos(
    b0: float, b1: float, b2: float, a0: float, a1: float, a2: float
) -> NDArray[np.float64]:
    """Pack into SOS row, dividing by a0 so a0 == 1.0."""
    return np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64
    )


# --- Public designers --------------------------------------------------------


def lowpass(fs: float, f0: float, Q: float = 0.7071067811865475) -> NDArray[np.float64]:
    """RBJ low-pass biquad. Q=1/sqrt(2) gives Butterworth 2nd order."""
    w0, cos_w0, sin_w0 = _common(fs, f0)
    alpha = _alpha_q(sin_w0, Q)
    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return _to_sos(b0, b1, b2, a0, a1, a2)


def highpass(
    fs: float, f0: float, Q: float = 0.7071067811865475
) -> NDArray[np.float64]:
    """RBJ high-pass biquad."""
    w0, cos_w0, sin_w0 = _common(fs, f0)
    alpha = _alpha_q(sin_w0, Q)
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return _to_sos(b0, b1, b2, a0, a1, a2)


def bandpass_csg(fs: float, f0: float, Q: float) -> NDArray[np.float64]:
    """Bandpass with constant skirt gain (peak gain = Q)."""
    w0, cos_w0, sin_w0 = _common(fs, f0)
    alpha = _alpha_q(sin_w0, Q)
    b0 = sin_w0 / 2.0
    b1 = 0.0
    b2 = -sin_w0 / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return _to_sos(b0, b1, b2, a0, a1, a2)


def bandpass_cpg(fs: float, f0: float, Q: float) -> NDArray[np.float64]:
    """Bandpass with constant 0-dB peak gain."""
    w0, cos_w0, sin_w0 = _common(fs, f0)
    alpha = _alpha_q(sin_w0, Q)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return _to_sos(b0, b1, b2, a0, a1, a2)


def notch(fs: float, f0: float, Q: float) -> NDArray[np.float64]:
    """Notch (band-reject) at f0."""
    w0, cos_w0, sin_w0 = _common(fs, f0)
    alpha = _alpha_q(sin_w0, Q)
    b0 = 1.0
    b1 = -2.0 * cos_w0
    b2 = 1.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return _to_sos(b0, b1, b2, a0, a1, a2)


def allpass(fs: float, f0: float, Q: float) -> NDArray[np.float64]:
    """Allpass — flat magnitude, phase shift centered on f0."""
    w0, cos_w0, sin_w0 = _common(fs, f0)
    alpha = _alpha_q(sin_w0, Q)
    b0 = 1.0 - alpha
    b1 = -2.0 * cos_w0
    b2 = 1.0 + alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return _to_sos(b0, b1, b2, a0, a1, a2)


def peaking(fs: float, f0: float, Q: float, gain_dB: float) -> NDArray[np.float64]:
    """Parametric peaking EQ. Boost or cut at f0 with bandwidth set by Q."""
    w0, cos_w0, sin_w0 = _common(fs, f0)
    alpha = _alpha_q(sin_w0, Q)
    A = 10.0 ** (gain_dB / 40.0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return _to_sos(b0, b1, b2, a0, a1, a2)


def lowshelf(fs: float, f0: float, S: float, gain_dB: float) -> NDArray[np.float64]:
    """Low-shelving EQ. S=1.0 gives -6 dB/oct slope outside the transition."""
    w0, cos_w0, sin_w0 = _common(fs, f0)
    A = 10.0 ** (gain_dB / 40.0)
    alpha = _alpha_shelf(sin_w0, A, S)
    sqrtA = math.sqrt(A)
    two_sqrtA_alpha = 2.0 * sqrtA * alpha
    b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + two_sqrtA_alpha)
    b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - two_sqrtA_alpha)
    a0 = (A + 1.0) + (A - 1.0) * cos_w0 + two_sqrtA_alpha
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
    a2 = (A + 1.0) + (A - 1.0) * cos_w0 - two_sqrtA_alpha
    return _to_sos(b0, b1, b2, a0, a1, a2)


def highshelf(fs: float, f0: float, S: float, gain_dB: float) -> NDArray[np.float64]:
    """High-shelving EQ. S=1.0 gives +6 dB/oct slope outside the transition."""
    w0, cos_w0, sin_w0 = _common(fs, f0)
    A = 10.0 ** (gain_dB / 40.0)
    alpha = _alpha_shelf(sin_w0, A, S)
    sqrtA = math.sqrt(A)
    two_sqrtA_alpha = 2.0 * sqrtA * alpha
    b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + two_sqrtA_alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - two_sqrtA_alpha)
    a0 = (A + 1.0) - (A - 1.0) * cos_w0 + two_sqrtA_alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
    a2 = (A + 1.0) - (A - 1.0) * cos_w0 - two_sqrtA_alpha
    return _to_sos(b0, b1, b2, a0, a1, a2)


def design(
    shape: BiquadShape,
    fs: float,
    f0: float,
    Q: float = 0.7071067811865475,
    gain_dB: float = 0.0,
    S: float = 1.0,
) -> NDArray[np.float64]:
    """Dispatch by name. Convenience for CLI / config-driven design."""
    table = {
        "lowpass": lambda: lowpass(fs, f0, Q),
        "highpass": lambda: highpass(fs, f0, Q),
        "bandpass_csg": lambda: bandpass_csg(fs, f0, Q),
        "bandpass_cpg": lambda: bandpass_cpg(fs, f0, Q),
        "notch": lambda: notch(fs, f0, Q),
        "allpass": lambda: allpass(fs, f0, Q),
        "peaking": lambda: peaking(fs, f0, Q, gain_dB),
        "lowshelf": lambda: lowshelf(fs, f0, S, gain_dB),
        "highshelf": lambda: highshelf(fs, f0, S, gain_dB),
    }
    if shape not in table:
        raise ValueError(f"unknown biquad shape {shape!r}")
    return table[shape]()
