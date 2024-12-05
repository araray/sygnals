import unittest
import numpy as np
from sygnals.core.dsp import (
    compute_fft, compute_ifft,
    low_pass_filter, high_pass_filter,
    band_pass_filter, apply_window
)

class TestDSP(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 1000  # Sampling rate in Hz
        self.signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, self.sample_rate))  # 10 Hz sine wave

    def test_fft(self):
        freqs, spectrum = compute_fft(self.signal, self.sample_rate)
        self.assertEqual(len(freqs), len(self.signal))
        self.assertTrue(np.isclose(max(spectrum), self.sample_rate / 2))  # Peak in magnitude

    def test_ifft(self):
        _, spectrum = compute_fft(self.signal, self.sample_rate)
        reconstructed_signal = compute_ifft(spectrum)
        self.assertTrue(np.allclose(self.signal, reconstructed_signal, atol=1e-6))

    def test_low_pass_filter(self):
        filtered_signal = low_pass_filter(self.signal, cutoff=15, fs=self.sample_rate)
        self.assertEqual(len(filtered_signal), len(self.signal))

    def test_high_pass_filter(self):
        filtered_signal = high_pass_filter(self.signal, cutoff=5, fs=self.sample_rate)
        self.assertEqual(len(filtered_signal), len(self.signal))

    def test_band_pass_filter(self):
        filtered_signal = band_pass_filter(self.signal, low_cutoff=5, high_cutoff=15, fs=self.sample_rate)
        self.assertEqual(len(filtered_signal), len(self.signal))

    def test_apply_window(self):
        windowed_signal = apply_window(self.signal, window_type="hamming")
        self.assertEqual(len(windowed_signal), len(self.signal))
