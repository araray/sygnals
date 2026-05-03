# tests/test_sygnals_fixes.py
"""
Test suite for the sygnals 1.0.x correctness/security fixes.

Covers:
  * custom_exec safe replacement (security)
  * colored noise (correctness)
  * RBJ biquad designs (correctness)
  * streaming SOS / FIR filters (correctness, state preservation)
  * compressor (correctness, time response)
  * Schroeder reverb (correctness)
  * chorus O(N) replacement (correctness, performance)
  * parabolic peak interpolation (correctness, sub-bin accuracy)

Run with:
    pytest -v tests/test_sygnals_fixes.py

Hypothesis property tests are gated behind a flag so the suite is fast
by default. To run them:
    SYGNALS_FAST_TESTS=0 pytest -v tests/test_sygnals_fixes.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter, sosfilt, sosfreqz, welch

# ---------------------------------------------------------------------------
# Module loading helpers — works whether the modules are installed or just
# sitting in the same directory as this test file.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    candidates = [
        _HERE / filename,
        _HERE.parent / filename,
        _HERE.parent / "output" / filename,
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location(name, p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError(f"could not locate {filename} in any of {candidates}")


custom_exec_safe = _load("custom_exec_safe", "custom_exec_safe.py")
noise_fixed = _load("noise_fixed", "noise_fixed.py")
rbj = _load("rbj_biquad", "rbj_biquad.py")
streaming = _load("streaming_filters", "streaming_filters.py")
compressor_fixed = _load("compressor_fixed", "compressor_fixed.py")
reverb = _load("reverb_schroeder", "reverb_schroeder.py")
chorus_fixed = _load("chorus_fixed", "chorus_fixed.py")
peak_interp = _load("peak_interp", "peak_interp.py")


# ---------------------------------------------------------------------------
# 1. custom_exec safe replacement
# ---------------------------------------------------------------------------


class TestCustomExecSafe:
    """Verify the ast-based evaluator allows legitimate use and blocks attacks."""

    @pytest.mark.parametrize(
        "expr, vars_, expected",
        [
            ("2 * x + y", {"x": 5, "y": 3}, 13),
            ("(a + b) * c", {"a": 1, "b": 2, "c": 3}, 9),
            ("x ** 2", {"x": 4}, 16),
            ("x // 2", {"x": 7}, 3),
            ("x % 3", {"x": 10}, 1),
        ],
    )
    def test_arithmetic(self, expr, vars_, expected):
        assert custom_exec_safe.evaluate_expression(expr, vars_) == expected

    @pytest.mark.parametrize(
        "expr, expected",
        [
            ("sin(0.5 * pi)", 1.0),
            ("cos(0)", 1.0),
            ("sqrt(16)", 4.0),
            ("exp(1)", math.e),
            ("log(e)", 1.0),
        ],
    )
    def test_whitelisted_functions(self, expr, expected):
        result = custom_exec_safe.evaluate_expression(expr, {})
        assert math.isclose(float(result), expected, rel_tol=1e-9, abs_tol=1e-9)

    def test_conditional_expression(self):
        assert custom_exec_safe.evaluate_expression("1 if x > 0 else -1", {"x": 5}) == 1
        assert (
            custom_exec_safe.evaluate_expression("1 if x > 0 else -1", {"x": -5}) == -1
        )

    def test_logical_short_circuit(self):
        evaluate = custom_exec_safe.evaluate_expression
        assert evaluate("a and b", {"a": True, "b": False}) is False
        assert evaluate("a or b", {"a": False, "b": True}) is True
        # Short-circuit: division by zero in unreached branch must not raise
        assert evaluate("x > 0 and 1.0 / x", {"x": 0}) is False

    def test_subscripting(self):
        evaluate = custom_exec_safe.evaluate_expression
        assert evaluate("a[0]", {"a": [10, 20, 30]}) == 10
        assert evaluate("a[1:3]", {"a": [10, 20, 30, 40]}) == [20, 30]

    def test_mean_over_array(self):
        result = custom_exec_safe.evaluate_expression(
            "mean(arr)", {"arr": np.array([1.0, 2.0, 3.0, 4.0])}
        )
        assert math.isclose(float(result), 2.5)

    # --- Attack vectors ---

    def test_blocks_subclass_traversal(self):
        with pytest.raises(custom_exec_safe.UnsafeExpressionError):
            custom_exec_safe.evaluate_expression(
                "[c for c in ().__class__.__base__.__subclasses__() "
                "if c.__name__ == 'BuiltinImporter']",
                {},
            )

    def test_blocks_dunder_attribute(self):
        with pytest.raises(custom_exec_safe.UnsafeExpressionError):
            custom_exec_safe.evaluate_expression("x.__class__", {"x": 1})

    def test_blocks_attribute_access_entirely(self):
        with pytest.raises(custom_exec_safe.UnsafeExpressionError):
            custom_exec_safe.evaluate_expression("foo.bar", {"foo": object()})

    def test_blocks_indirect_call(self):
        with pytest.raises(custom_exec_safe.UnsafeExpressionError):
            custom_exec_safe.evaluate_expression("sin.__call__(0)", {})

    def test_blocks_underscore_names(self):
        with pytest.raises(custom_exec_safe.UnsafeExpressionError):
            custom_exec_safe.evaluate_expression("_secret + 1", {"_secret": 42})

    def test_blocks_lambda(self):
        with pytest.raises(SyntaxError):
            # `f = lambda x: x` is a statement, not an expression — ast.parse(mode='eval') rejects it
            custom_exec_safe.evaluate_expression("f = lambda x: x", {})

    def test_blocks_lambda_expression(self):
        # `lambda x: x` IS a valid expression but the ast.Lambda node is not in our whitelist
        with pytest.raises(custom_exec_safe.UnsafeExpressionError):
            custom_exec_safe.evaluate_expression("(lambda x: x)(1)", {})

    def test_blocks_comprehension(self):
        with pytest.raises(custom_exec_safe.UnsafeExpressionError):
            custom_exec_safe.evaluate_expression("[i for i in range(10)]", {})

    def test_blocks_walrus(self):
        with pytest.raises(custom_exec_safe.UnsafeExpressionError):
            custom_exec_safe.evaluate_expression("(x := 5)", {})

    def test_blocks_undefined_name(self):
        with pytest.raises(NameError):
            custom_exec_safe.evaluate_expression("foo + 1", {})


# ---------------------------------------------------------------------------
# 2. Colored noise
# ---------------------------------------------------------------------------


class TestColoredNoise:
    """Verify spectral slope and unit-RMS normalization."""

    @pytest.mark.parametrize(
        "color, expected_slope, tol",
        [
            ("white", 0.0, 0.15),
            ("pink", -1.0, 0.20),
            ("brown", -2.0, 0.30),
            ("blue", +1.0, 0.20),
            ("violet", +2.0, 0.30),
        ],
    )
    def test_psd_slope(self, color, expected_slope, tol):
        rng = np.random.default_rng(42)
        n, fs = 1 << 17, 44100
        x = noise_fixed._colored_noise(n, color, rng)
        f, p = welch(x, fs=fs, nperseg=4096)
        # Fit log P vs log f over the 50 Hz - fs/4 band
        mask = (f >= 50) & (f <= fs / 4) & (p > 0)
        slope = np.polyfit(np.log(f[mask]), np.log(p[mask]), 1)[0]
        assert abs(slope - expected_slope) < tol, (
            f"{color}: slope={slope:.3f}, expected {expected_slope}±{tol}"
        )

    @pytest.mark.parametrize("color", ["white", "pink", "brown", "blue", "violet"])
    def test_unit_rms(self, color):
        rng = np.random.default_rng(0)
        x = noise_fixed._colored_noise(1 << 16, color, rng)
        rms = float(np.sqrt(np.mean(x * x)))
        assert math.isclose(rms, 1.0, abs_tol=1e-10), f"{color}: rms={rms}"

    def test_zero_dc(self):
        rng = np.random.default_rng(0)
        for color in ("pink", "brown", "blue", "violet"):
            x = noise_fixed._colored_noise(1 << 14, color, rng)
            mean = float(np.mean(x))
            assert abs(mean) < 1e-10, f"{color}: mean={mean}"

    def test_unknown_color_raises(self):
        with pytest.raises(ValueError):
            noise_fixed._colored_noise(1024, "octarine", np.random.default_rng(0))

    def test_add_noise_snr(self):
        rng = np.random.default_rng(0)
        sr, n = 16000, 16000
        # Pure tone: very predictable signal power
        t = np.arange(n) / sr
        sig = np.sin(2 * np.pi * 1000 * t)
        sig_power = np.mean(sig * sig)
        for snr_db in (0, 10, 20, 30):
            y = noise_fixed.add_noise(sig, snr_db, "white", seed=1)
            noise = y - sig
            noise_power = np.mean(noise * noise)
            achieved_snr_db = 10 * np.log10(sig_power / noise_power)
            assert abs(achieved_snr_db - snr_db) < 0.5, (
                f"target {snr_db} dB, got {achieved_snr_db:.2f} dB"
            )


# ---------------------------------------------------------------------------
# 3. RBJ Audio EQ Cookbook biquads
# ---------------------------------------------------------------------------


class TestRBJBiquads:
    """Verify the canonical RBJ designs match expected frequency response."""

    @staticmethod
    def _response_db(sos, fs, freq):
        w, h = sosfreqz(sos, worN=4096, fs=fs)
        idx = int(np.argmin(np.abs(w - freq)))
        return 20.0 * np.log10(np.abs(h[idx]) + 1e-30)

    def test_lowpass_butterworth_cutoff(self):
        """LP at f0=1 kHz with Q=1/sqrt(2) is Butterworth → -3 dB at cutoff."""
        fs, f0 = 48000, 1000.0
        sos = rbj.lowpass(fs, f0)
        assert -3.5 < self._response_db(sos, fs, f0) < -2.5

    def test_lowpass_passband_flat(self):
        fs, f0 = 48000, 1000.0
        sos = rbj.lowpass(fs, f0)
        assert -1.0 < self._response_db(sos, fs, f0 / 4) < 0.5

    def test_lowpass_stopband_attenuated(self):
        fs, f0 = 48000, 1000.0
        sos = rbj.lowpass(fs, f0)
        assert self._response_db(sos, fs, f0 * 4) < -20.0

    def test_highpass_cutoff(self):
        fs, f0 = 48000, 1000.0
        sos = rbj.highpass(fs, f0)
        assert -3.5 < self._response_db(sos, fs, f0) < -2.5

    def test_peaking_eq_boost_at_center(self):
        fs, f0, Q, gain = 48000, 1000.0, 1.0, 6.0
        sos = rbj.peaking(fs, f0, Q, gain)
        center_db = self._response_db(sos, fs, f0)
        assert abs(center_db - gain) < 0.1

    def test_peaking_eq_unity_far_from_center(self):
        fs, f0, Q, gain = 48000, 1000.0, 1.0, 6.0
        sos = rbj.peaking(fs, f0, Q, gain)
        # Far from center, should be ~0 dB (unity)
        assert abs(self._response_db(sos, fs, 50.0)) < 0.5
        assert abs(self._response_db(sos, fs, 20000.0)) < 0.5

    def test_peaking_cut_at_center(self):
        fs, f0, Q, gain = 48000, 2000.0, 1.5, -9.0
        sos = rbj.peaking(fs, f0, Q, gain)
        assert abs(self._response_db(sos, fs, f0) - gain) < 0.1

    def test_notch_attenuates_at_center(self):
        fs, f0, Q = 48000, 1000.0, 10.0
        sos = rbj.notch(fs, f0, Q)
        # The cookbook notch has a true zero at f0; we must evaluate exactly
        # at f0, not on a grid that may skip the null.
        w, h = sosfreqz(sos, worN=[2 * np.pi * f0 / fs])
        depth_db = 20.0 * np.log10(np.abs(h[0]) + 1e-30)
        assert depth_db < -100.0, f"notch depth at f0 = {depth_db:.2f} dB"

    def test_lowshelf_dc_gain(self):
        fs, f0, gain = 48000, 200.0, +6.0
        sos = rbj.lowshelf(fs, f0, S=1.0, gain_dB=gain)
        # Way below the shelf, gain should approach +6 dB
        assert abs(self._response_db(sos, fs, 20.0) - gain) < 0.5

    def test_lowshelf_above_shelf_unity(self):
        fs, f0, gain = 48000, 200.0, +6.0
        sos = rbj.lowshelf(fs, f0, S=1.0, gain_dB=gain)
        assert abs(self._response_db(sos, fs, 8000.0)) < 0.5

    def test_highshelf_above_shelf_gain(self):
        fs, f0, gain = 48000, 5000.0, +6.0
        sos = rbj.highshelf(fs, f0, S=1.0, gain_dB=gain)
        # Way above the shelf, gain should approach +6 dB
        assert abs(self._response_db(sos, fs, 20000.0) - gain) < 0.5

    def test_allpass_unity_magnitude(self):
        fs, f0, Q = 48000, 1000.0, 1.0
        sos = rbj.allpass(fs, f0, Q)
        for f in (100, 500, 1000, 2000, 10000):
            assert abs(self._response_db(sos, fs, f)) < 0.01

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            rbj.lowpass(48000, 24000)  # f0 == Nyquist
        with pytest.raises(ValueError):
            rbj.lowpass(48000, 0)
        with pytest.raises(ValueError):
            rbj.peaking(48000, 1000, Q=0, gain_dB=6)
        with pytest.raises(ValueError):
            rbj.design("nonsense", 48000, 1000)


# ---------------------------------------------------------------------------
# 4. Streaming filters
# ---------------------------------------------------------------------------


class TestStreamingFilters:
    """Block-by-block output must match whole-signal output."""

    def test_sosfilter_block_equivalence(self):
        fs, f0 = 48000, 1000.0
        sos = rbj.lowpass(fs, f0)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(48000)

        # Whole-signal reference (matching the streaming filter's prime() behavior)
        from scipy.signal import sosfilt_zi

        zi = sosfilt_zi(sos) * x[0]
        y_ref, _ = sosfilt(sos, x, zi=zi)

        # Streaming
        filt = streaming.SOSFilter(sos)
        block_size = 1024
        blocks = []
        for i in range(0, len(x), block_size):
            blocks.append(filt.process_block(x[i : i + block_size]))
        y_streamed = np.concatenate(blocks)

        np.testing.assert_allclose(y_streamed, y_ref, atol=1e-10, rtol=1e-10)

    def test_sosfilter_reset(self):
        sos = rbj.lowpass(48000, 1000)
        filt = streaming.SOSFilter(sos)
        x = np.ones(1024)
        y1 = filt.process_block(x).copy()
        filt.reset()
        y2 = filt.process_block(x)
        np.testing.assert_allclose(y1, y2)

    def test_firfilter_block_equivalence(self):
        rng = np.random.default_rng(0)
        taps = rng.standard_normal(64)
        x = rng.standard_normal(8192)
        y_ref = np.convolve(x, taps, mode="full")[: len(x)]

        filt = streaming.FIRFilter(taps)
        block_size = 256
        blocks = []
        for i in range(0, len(x), block_size):
            blocks.append(filt.process_block(x[i : i + block_size]))
        y_streamed = np.concatenate(blocks)
        np.testing.assert_allclose(y_streamed, y_ref, atol=1e-12, rtol=1e-12)

    def test_invalid_sos_shape_raises(self):
        with pytest.raises(ValueError):
            streaming.SOSFilter(np.zeros((3, 5)))
        with pytest.raises(ValueError):
            streaming.SOSFilter(np.zeros(6))


# ---------------------------------------------------------------------------
# 5. Compressor
# ---------------------------------------------------------------------------


class TestCompressor:
    """Verify gain reduction, time response, and RMS detector behavior."""

    def test_no_compression_below_threshold(self):
        sr = 48000
        x = 0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)  # ~-20 dBFS peak
        y = compressor_fixed.compress(
            x,
            sr,
            threshold_db=-6.0,
            ratio=4.0,
            attack_ms=1.0,
            release_ms=10.0,
            knee_db=0.0,
        )
        # Below threshold → output should equal input
        np.testing.assert_allclose(y, x, atol=0.005)

    def test_compression_above_threshold(self):
        sr = 48000
        x = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)  # 0 dBFS peak
        y = compressor_fixed.compress(
            x,
            sr,
            threshold_db=-12.0,
            ratio=4.0,
            attack_ms=1.0,
            release_ms=20.0,
            knee_db=0.0,
        )
        # The gain smoother takes a few release-time-constants to reach
        # steady state. Look at the second half of the signal.
        peak_in_late = np.max(np.abs(x[sr // 2 :]))
        peak_out_late = np.max(np.abs(y[sr // 2 :]))
        assert peak_out_late < peak_in_late * 0.6, (
            f"steady-state peak: in={peak_in_late:.3f}, out={peak_out_late:.3f}"
        )

    def test_makeup_gain(self):
        sr = 48000
        x = 0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        y = compressor_fixed.compress(
            x,
            sr,
            threshold_db=-30.0,
            ratio=4.0,
            attack_ms=0.1,
            release_ms=1.0,
            knee_db=0.0,
            makeup_db=6.0,
        )
        # Quasi-steady state: makeup gain ~doubles the signal
        # (after compression we have less than 2x — so test is loose)
        assert np.max(np.abs(y[-1000:])) > np.max(np.abs(x))

    def test_class_streaming_equivalence(self):
        sr = 48000
        rng = np.random.default_rng(0)
        x = 0.5 * rng.standard_normal(sr)
        cmp1 = compressor_fixed.Compressor(
            sr, threshold_db=-12, ratio=4, attack_ms=5, release_ms=50
        )
        y_whole = cmp1.process_block(x)

        cmp2 = compressor_fixed.Compressor(
            sr, threshold_db=-12, ratio=4, attack_ms=5, release_ms=50
        )
        block_size = 1024
        blocks = []
        for i in range(0, len(x), block_size):
            blocks.append(cmp2.process_block(x[i : i + block_size]))
        y_streamed = np.concatenate(blocks)
        np.testing.assert_allclose(y_streamed, y_whole, atol=1e-12, rtol=1e-12)

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            compressor_fixed.compress(np.zeros(100), 48000, ratio=0.5)


# ---------------------------------------------------------------------------
# 6. Schroeder reverb
# ---------------------------------------------------------------------------


class TestReverb:
    def test_impulse_response_decays(self):
        sr = 48000
        ir = reverb.reverb_impulse_response(sr, rt60=0.5, duration=1.0)
        # First-100ms RMS should be much larger than last-100ms RMS
        early = ir[: int(0.1 * sr)]
        late = ir[-int(0.1 * sr) :]
        early_rms = np.sqrt(np.mean(early * early))
        late_rms = np.sqrt(np.mean(late * late))
        assert early_rms > 4.0 * late_rms

    def test_dry_only_passthrough(self):
        sr = 48000
        x = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        y = reverb.schroeder_reverb(x, sr, rt60=0.5, wet=0.0, dry=1.0)
        np.testing.assert_allclose(y, x, atol=1e-12)

    def test_wet_only_no_direct(self):
        sr = 48000
        x = np.zeros(sr)
        x[100] = 1.0
        y = reverb.schroeder_reverb(x, sr, rt60=0.5, wet=1.0, dry=0.0)
        # There must be reverb energy
        assert np.max(np.abs(y)) > 0.0
        # Direct sample (sample 100) should not pass through unchanged: combs
        # delay it. (Resonant combs CAN produce peaks > 1.0 elsewhere; that's
        # not a bug, just feedback resonance.)
        assert abs(y[100]) < 0.5

    def test_invalid_rt60(self):
        with pytest.raises(ValueError):
            reverb.schroeder_reverb(np.zeros(100), 48000, rt60=0.0)


# ---------------------------------------------------------------------------
# 7. Chorus (correctness + performance vs. a reference O(N) implementation)
# ---------------------------------------------------------------------------


class TestChorus:
    def test_no_voices_no_change_dry_only(self):
        sr = 48000
        rng = np.random.default_rng(0)
        x = 0.3 * rng.standard_normal(sr)
        y = chorus_fixed.apply_chorus(
            x, sr, voices=2, depth_ms=2.0, delay_ms=20.0, wet=0.0, dry=1.0
        )
        np.testing.assert_allclose(y, x, atol=1e-12)

    def test_output_length(self):
        sr = 48000
        x = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        y = chorus_fixed.apply_chorus(x, sr)
        assert y.shape == x.shape

    def test_completes_in_reasonable_time(self):
        """The catastrophic-perf-bug fix: 1 second of audio should process in < 5s."""
        sr = 44100
        x = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        t0 = time.time()
        chorus_fixed.apply_chorus(x, sr, voices=3)
        elapsed = time.time() - t0
        # Pure-Python loop, O(N · V): a few hundred ms expected. Allow 5s headroom.
        assert elapsed < 5.0, f"chorus took {elapsed:.2f}s for 1s of audio"

    def test_invalid_feedback(self):
        with pytest.raises(ValueError):
            chorus_fixed.apply_chorus(np.zeros(100), 48000, feedback=1.5)


# ---------------------------------------------------------------------------
# 8. Parabolic peak interpolation
# ---------------------------------------------------------------------------


class TestPeakInterp:
    @pytest.mark.parametrize("offset", [0.0, 0.1, 0.25, 0.4, -0.3])
    def test_parabolic_subbin_accuracy(self, offset):
        """Synthesize a magnitude spectrum that is a parabola peaked at offset."""
        N = 32
        k = N // 2
        bins = np.arange(N)
        # Parabolic shape centered at k+offset, peak 1.0
        mag = np.maximum(0.0, 1.0 - 0.05 * (bins - (k + offset)) ** 2)
        result_offset, result_peak = peak_interp.parabolic_peak_bin(mag)
        # Interpolation accuracy: a clean parabola should recover the offset exactly
        assert abs(result_offset - offset) < 0.02, (
            f"offset error: {result_offset - offset}"
        )

    def test_dominant_frequency_subbin(self):
        """Sinusoid at a non-bin-center frequency: parabolic better than argmax."""
        fs, n = 48000, 4096
        # Test multiple frequencies including one near a bin edge (worst case
        # for argmax, best case for parabolic) and one near a bin center
        # (best case for argmax, less marginal benefit from parabolic).
        bin_width = fs / n  # ≈ 11.72 Hz
        for f_true in (1234.567, 1000.0 + 0.5 * bin_width, 2500.0 + 0.1 * bin_width):
            t = np.arange(n) / fs
            x = np.sin(2 * np.pi * f_true * t)
            from scipy.signal import get_window

            win = get_window("hann", n, fftbins=True)
            spec = np.fft.rfft(x * win)
            mag = np.abs(spec)
            f_argmax = peak_interp.dominant_frequency_interp(mag, fs, method="argmax")
            f_parab = peak_interp.dominant_frequency_interp(mag, fs, method="parabolic")
            err_argmax = abs(f_argmax - f_true)
            err_parab = abs(f_parab - f_true)
            # argmax has at most half-bin error
            assert err_argmax <= bin_width / 2 + 0.01
            # parabolic on Hann window: typical error ~5% of a bin
            # (per J. O. Smith's analysis of bias)
            assert err_parab < bin_width / 10, (
                f"f={f_true}: parabolic err={err_parab:.3f} Hz, bin={bin_width:.2f} Hz"
            )
            # parabolic should be at least as good as argmax
            assert err_parab <= err_argmax + 1e-9


# ---------------------------------------------------------------------------
# 9. Hypothesis property tests (gated)
# ---------------------------------------------------------------------------

if not int(os.environ.get("SYGNALS_FAST_TESTS", "1")):
    try:
        from hypothesis import given, settings
        from hypothesis import strategies as st
        from hypothesis.extra.numpy import arrays

        class TestProperties:
            @given(
                arrays(
                    np.float64,
                    st.integers(64, 4096),
                    elements=st.floats(
                        -1.0, 1.0, allow_nan=False, allow_infinity=False
                    ),
                )
            )
            @settings(max_examples=50, deadline=None)
            def test_lowpass_streaming_matches_offline(self, x):
                sos = rbj.lowpass(48000, 5000)
                from scipy.signal import sosfilt_zi

                zi = sosfilt_zi(sos) * x[0]
                y_ref, _ = sosfilt(sos, x, zi=zi)
                filt = streaming.SOSFilter(sos)
                # Run in random-sized blocks
                blocks = []
                i = 0
                rng = np.random.default_rng()
                while i < len(x):
                    bs = int(rng.integers(1, max(2, len(x) // 4 + 1)))
                    blocks.append(filt.process_block(x[i : i + bs]))
                    i += bs
                y_streamed = np.concatenate(blocks)
                np.testing.assert_allclose(y_streamed, y_ref, atol=1e-9)

            @given(st.text(min_size=1, max_size=50))
            @settings(max_examples=200, deadline=None)
            def test_evaluator_never_returns_module(self, expr):
                """No matter what the user types, we should NEVER return a module/class."""
                try:
                    result = custom_exec_safe.evaluate_expression(expr, {})
                except Exception:
                    return  # any exception is acceptable
                # If it succeeded, the result must not be a module or class
                import types

                assert not isinstance(result, types.ModuleType)
                assert not isinstance(result, type)

    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
