# sygnals/core/audio/effects/reverb.py
"""
Schroeder synthetic reverberator.

Replaces the previous implementation, which generated a reverb impulse
response as exp(-t/tau) * white_noise. That has none of the structure of an
actual room IR (no direct sound, no early reflections, no diffusion, no
frequency-dependent decay), and it requires a long FFT-based convolution
pass over the whole signal.

Algorithm
---------
Schroeder 1962 [1]: 4 parallel feedback comb filters in parallel, summed
into a series of 3 nested allpass filters. The combs build the late
reverb's exponential decay; the allpasses break up the periodicity and
add diffusion.

Implementation note: each comb is a recursive filter
    y[n] = x[n] + g * y[n-D]
which can be written as `lfilter(b=[1], a=[1, 0, ..., 0, -g], x)` with
the -g coefficient at index D. SciPy `lfilter` runs this in C, which is
~100× faster than a Python `for` loop.

The allpass filter
    y[n] = -g * x[n] + x[n-D] + g * y[n-D]
is `lfilter(b=[-g, 0, ..., 0, 1], a=[1, 0, ..., 0, g], x)` with both
coefficients at index D.

References
----------
[1] M. R. Schroeder, "Natural Sounding Artificial Reverberation",
    J. Audio Eng. Soc., 10(3):219-223, 1962.
[2] J. A. Moorer, "About This Reverberation Business",
    Computer Music Journal, 3(2):13-28, 1979. Section on Schroeder reverb
    constants.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

logger = logging.getLogger(__name__)

# Moorer's tuning of Schroeder's reverb. Mutually-prime delays in milliseconds
# minimize repetitive comb-spectrum artifacts.
_COMB_DELAYS_MS = (29.7, 37.1, 41.1, 43.7)
_ALLPASS_DELAYS_MS = (5.0, 1.7, 0.7)
_ALLPASS_FEEDBACK = 0.5


def _comb(
    x: NDArray[np.float64], delay_samples: int, gain: float
) -> NDArray[np.float64]:
    """Feedback comb filter via lfilter (C-speed)."""
    if delay_samples < 1:
        delay_samples = 1
    a = np.zeros(delay_samples + 1, dtype=np.float64)
    a[0] = 1.0
    a[delay_samples] = -gain
    return lfilter([1.0], a, x)


def _allpass(
    x: NDArray[np.float64], delay_samples: int, gain: float
) -> NDArray[np.float64]:
    """Schroeder allpass via lfilter (C-speed)."""
    if delay_samples < 1:
        delay_samples = 1
    b = np.zeros(delay_samples + 1, dtype=np.float64)
    a = np.zeros(delay_samples + 1, dtype=np.float64)
    b[0] = -gain
    b[delay_samples] = 1.0
    a[0] = 1.0
    a[delay_samples] = gain
    return lfilter(b, a, x)


def schroeder_reverb(
    y: NDArray[np.float64],
    sr: int,
    *,
    rt60: float = 1.0,
    wet: float = 0.3,
    dry: float = 0.7,
) -> NDArray[np.float64]:
    """Schroeder synthetic reverb.

    Args:
        y: 1D input signal.
        sr: Sample rate (Hz).
        rt60: Reverberation time in seconds (energy decay by 60 dB).
        wet: Gain for the reverberated signal (default 0.3).
        dry: Gain for the unprocessed signal (default 0.7).

    Returns:
        Reverberated signal of the same length as `y`.

    Notes
    -----
    * For accurate RT60, the comb gains are derived from the standard
      relation `g = 10^(-3 * D_sec / RT60)`, which gives an exponential
      decay with -60 dB at `RT60` seconds.
    * The output is **not** lengthened to include the full reverb tail.
      For that, zero-pad the input by ~rt60 seconds before calling.
    """
    if y.ndim != 1:
        raise ValueError("schroeder_reverb expects 1D input. Apply per-channel.")
    if rt60 <= 0:
        raise ValueError("rt60 must be > 0")

    # Parallel combs
    accum = np.zeros_like(y)
    for d_ms in _COMB_DELAYS_MS:
        d_sec = d_ms * 1e-3
        d_samples = max(1, int(round(d_sec * sr)))
        gain = 10.0 ** (-3.0 * d_sec / rt60)
        accum = accum + _comb(y, d_samples, gain)
    accum = accum / len(_COMB_DELAYS_MS)

    # Series allpasses
    z = accum
    for d_ms in _ALLPASS_DELAYS_MS:
        d_samples = max(1, int(round(d_ms * 1e-3 * sr)))
        z = _allpass(z, d_samples, _ALLPASS_FEEDBACK)

    return (dry * y + wet * z).astype(np.float64, copy=False)


def reverb_impulse_response(
    sr: int,
    *,
    rt60: float = 1.0,
    duration: float = 2.0,
) -> NDArray[np.float64]:
    """Return the Schroeder reverb impulse response for inspection."""
    n = max(1, int(round(duration * sr)))
    impulse = np.zeros(n, dtype=np.float64)
    impulse[0] = 1.0
    return schroeder_reverb(impulse, sr, rt60=rt60, wet=1.0, dry=0.0)
