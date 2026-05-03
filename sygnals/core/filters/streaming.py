# sygnals/core/filters/streaming.py
"""
Streaming filter classes that preserve state across `process_block` calls.

This is the foundation for any real-time / live / long-file / pipeline use of
sygnals filters. The current sygnals 1.0 filter API is offline-only
(`apply_sos_filter` calls `sosfiltfilt`, which is zero-phase but cannot
process incremental blocks). These classes complement that API for the
streaming case.

Design
------
* Each instance owns a `(n_sections, 2)` state array (`_zi`) for SciPy's
  Direct-Form II Transposed biquad cascade.
* `process_block(x)` runs `scipy.signal.sosfilt` with explicit `zi`,
  updates `zi`, and returns the filtered output. No allocations beyond
  the output buffer.
* `reset()` clears state to zero.
* `prime(level)` initializes state to the steady-state response for a DC
  input of `level`. This is what `scipy.signal.filtfilt` uses internally
  to suppress edge transients; useful for the first block of long-file
  pipelines.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import sosfilt, sosfilt_zi


class SOSFilter:
    """Cascaded biquad filter with persistent state.

    Example
    -------
    >>> from sygnals.core.filters.rbj_biquad import lowpass
    >>> sos = lowpass(fs=48000, f0=1000)
    >>> filt = SOSFilter(sos)
    >>> y_part1 = filt.process_block(x_block_1)
    >>> y_part2 = filt.process_block(x_block_2)   # picks up where y_part1 ended
    >>> filt.reset()                              # back to a fresh filter
    """

    __slots__ = ("_sos", "_zi", "_initialized", "_zi_template")

    def __init__(self, sos: NDArray[np.float64]):
        sos = np.asarray(sos, dtype=np.float64)
        if sos.ndim != 2 or sos.shape[1] != 6:
            raise ValueError(f"sos must have shape (n_sections, 6); got {sos.shape}")
        self._sos = sos
        self._zi_template = sosfilt_zi(self._sos)  # (n_sections, 2), DC-prime template
        self._zi: NDArray[np.float64] = np.zeros_like(self._zi_template)
        self._initialized = False

    @property
    def sos(self) -> NDArray[np.float64]:
        return self._sos

    @property
    def n_sections(self) -> int:
        return self._sos.shape[0]

    @property
    def state(self) -> NDArray[np.float64]:
        """Read-only view of internal state (for diagnostics / checkpointing)."""
        return self._zi.view()

    def reset(self) -> "SOSFilter":
        """Clear filter state to zero."""
        self._zi[:] = 0.0
        self._initialized = False
        return self

    def prime(self, level: float = 0.0) -> "SOSFilter":
        """Pre-load filter state to the DC steady-state for `level`.

        This is useful for the first block of a long-file process: it
        suppresses the startup transient that you'd otherwise get with
        zero-state initialization, much like `scipy.signal.filtfilt` does
        internally.
        """
        self._zi = self._zi_template * level
        self._initialized = True
        return self

    def process_block(
        self, x: NDArray[np.float64], out: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Filter a block of samples; update state.

        Args:
            x: 1D input block (float64).
            out: Optional preallocated output buffer of the same shape.

        Returns:
            Filtered output block.
        """
        if x.ndim != 1:
            raise ValueError("process_block requires a 1D input")
        if x.size == 0:
            if out is None:
                return np.empty(0, dtype=np.float64)
            return out
        if not self._initialized:
            # Default to DC-prime with the first sample's value to match the
            # convention used by sosfiltfilt (avoids large transients on
            # signals that don't start at zero).
            self.prime(float(x[0]))
        y, self._zi = sosfilt(self._sos, x.astype(np.float64, copy=False), zi=self._zi)
        if out is None:
            return y.astype(np.float64, copy=False)
        out[:] = y
        return out

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.process_block(x)


class FIRFilter:
    """Streaming FIR filter using overlap-save convolution.

    Holds the last `len(taps) - 1` input samples as state so that
    successive blocks produce the same output as if the entire signal
    had been convolved at once (modulo numerical tolerance).
    """

    __slots__ = ("_taps", "_history", "_initialized")

    def __init__(self, taps: NDArray[np.float64]):
        taps = np.asarray(taps, dtype=np.float64)
        if taps.ndim != 1:
            raise ValueError("taps must be 1D")
        if taps.size < 1:
            raise ValueError("taps must have at least 1 sample")
        self._taps = taps
        self._history: NDArray[np.float64] = np.zeros(taps.size - 1, dtype=np.float64)
        self._initialized = False

    @property
    def order(self) -> int:
        return self._taps.size - 1

    def reset(self) -> "FIRFilter":
        self._history[:] = 0.0
        self._initialized = False
        return self

    def process_block(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if x.ndim != 1:
            raise ValueError("process_block requires a 1D input")
        n = self._taps.size
        if x.size == 0:
            return np.empty(0, dtype=np.float64)
        # Build extended buffer = history + x, convolve with 'valid' to drop
        # the warmup, leaving exactly len(x) output samples.
        extended = np.concatenate([self._history, x])
        y = np.convolve(extended, self._taps, mode="valid")
        # Save the last n-1 samples of x for next call's history.
        if x.size >= n - 1:
            self._history[:] = x[-(n - 1) :]
        else:
            # x is shorter than the filter order; shift history left.
            shift = x.size
            self._history[:-shift] = self._history[shift:]
            self._history[-shift:] = x
        self._initialized = True
        return y.astype(np.float64, copy=False)

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.process_block(x)
