import pytest
import numpy as np
from sygnals.utils.visualizations import plot_spectrogram, plot_fft, plot_waveform

def test_plot_spectrogram(tmp_path):
    sr = 1000
    t = np.arange(0,1,1/sr)
    x = np.sin(2*np.pi*50*t)
    out_file = tmp_path / "spectrogram.png"
    plot_spectrogram(x, sr, str(out_file))
    assert out_file.exists()

def test_plot_fft(tmp_path):
    sr = 1000
    t = np.arange(0,1,1/sr)
    x = np.sin(2*np.pi*100*t)
    out_file = tmp_path / "fft.png"
    plot_fft(x, sr, str(out_file))
    assert out_file.exists()

def test_plot_waveform(tmp_path):
    sr = 1000
    x = np.random.randn(sr)
    out_file = tmp_path / "waveform.png"
    plot_waveform(x, sr, str(out_file))
    assert out_file.exists()
