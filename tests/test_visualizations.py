# tests/test_visualizations.py

import pytest
import numpy as np
from pathlib import Path # Use pathlib

# Import the updated and new visualization functions
from sygnals.utils.visualizations import (
    plot_spectrogram,
    plot_fft_magnitude, # Renamed from plot_fft
    plot_waveform,
    plot_fft_phase,     # New function
    plot_scalogram      # New function
)

# --- Test Data Generation ---

@pytest.fixture
def sine_signal():
    """Generate a simple sine wave."""
    sr = 1000.0
    duration = 1.0
    freq = 100.0
    t = np.arange(0, duration, 1/sr)
    x = 0.8 * np.sin(2 * np.pi * freq * t).astype(np.float64)
    return x, sr

@pytest.fixture
def noise_signal():
    """Generate random noise."""
    sr = 1000.0
    duration = 1.0
    x = (np.random.rand(int(sr * duration)) * 2 - 1).astype(np.float64) # Noise in [-1, 1]
    return x, sr

# --- Test Functions ---

def test_plot_spectrogram(tmp_path: Path, sine_signal):
    """Test generating a spectrogram plot."""
    x, sr = sine_signal
    out_file = tmp_path / "spectrogram.png"
    plot_spectrogram(x, sr, str(out_file))
    assert out_file.exists()
    assert out_file.stat().st_size > 100 # Check if file is not empty

def test_plot_fft_magnitude(tmp_path: Path, sine_signal):
    """Test generating an FFT magnitude plot (renamed test)."""
    x, sr = sine_signal
    out_file = tmp_path / "fft_magnitude.png"
    # Call the renamed function
    plot_fft_magnitude(x, sr, str(out_file))
    assert out_file.exists()
    assert out_file.stat().st_size > 100

def test_plot_fft_phase(tmp_path: Path, sine_signal):
    """Test generating an FFT phase plot."""
    x, sr = sine_signal
    out_file = tmp_path / "fft_phase.png"
    # Test both wrapped and unwrapped phase
    plot_fft_phase(x, sr, str(out_file), unwrap=False, title="FFT Phase (Wrapped)")
    assert out_file.exists()
    assert out_file.stat().st_size > 100

    out_file_unwrapped = tmp_path / "fft_phase_unwrapped.png"
    plot_fft_phase(x, sr, str(out_file_unwrapped), unwrap=True, title="FFT Phase (Unwrapped)")
    assert out_file_unwrapped.exists()
    assert out_file_unwrapped.stat().st_size > 100

def test_plot_waveform(tmp_path: Path, noise_signal):
    """Test generating a waveform plot."""
    x, sr = noise_signal
    out_file = tmp_path / "waveform.png"
    plot_waveform(x, sr, str(out_file))
    assert out_file.exists()
    assert out_file.stat().st_size > 100

def test_plot_waveform_max_samples(tmp_path: Path, noise_signal):
    """Test waveform plot with max_samples limit."""
    x, sr = noise_signal
    max_samples = 100
    out_file = tmp_path / "waveform_limited.png"
    plot_waveform(x, sr, str(out_file), max_samples=max_samples)
    assert out_file.exists()
    assert out_file.stat().st_size > 100
    # Verification could involve loading the image and checking dimensions if needed,
    # but file existence and non-zero size is a basic check.

def test_plot_scalogram(tmp_path: Path, sine_signal):
    """Test generating a wavelet scalogram plot."""
    x, sr = sine_signal
    out_file = tmp_path / "scalogram.png"
    # Use default scales and wavelet ('morl')
    plot_scalogram(x, str(out_file), sr=sr)
    assert out_file.exists()
    assert out_file.stat().st_size > 100

def test_plot_scalogram_custom_scales(tmp_path: Path, sine_signal):
    """Test scalogram with custom scales."""
    x, sr = sine_signal
    out_file = tmp_path / "scalogram_custom.png"
    scales = np.geomspace(1, 128, num=50) # Example custom scales
    plot_scalogram(x, str(out_file), scales=scales, wavelet='cmor1.5-1.0', sr=sr)
    assert out_file.exists()
    assert out_file.stat().st_size > 100

# Add tests for edge cases like empty data if desired
def test_plot_empty_data(tmp_path: Path):
    """Test plotting functions with empty data."""
    x = np.array([], dtype=np.float64)
    sr = 1000.0
    out_file_base = tmp_path / "empty_plot"

    # Expect functions to handle empty data gracefully (e.g., log warning, not raise error)
    plot_spectrogram(x, sr, str(out_file_base / "spec.png"))
    plot_fft_magnitude(x, sr, str(out_file_base / "fft_mag.png"))
    plot_fft_phase(x, sr, str(out_file_base / "fft_phase.png"))
    plot_waveform(x, sr, str(out_file_base / "wave.png"))
    plot_scalogram(x, str(out_file_base / "scal.png"), sr=sr)

    # Check that no plot files were actually created (or are empty)
    assert not (out_file_base / "spec.png").exists() or (out_file_base / "spec.png").stat().st_size == 0
    # ... add similar checks for other plot types if strict behavior is needed
