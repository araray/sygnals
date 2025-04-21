# tests/test_visualizations.py

import pytest
import numpy as np
from pathlib import Path # Use pathlib
import matplotlib.pyplot as plt # Import plt to check if figures are closed

# Import the updated and new visualization functions
from sygnals.utils.visualizations import (
    plot_spectrogram,
    plot_fft_magnitude,
    plot_waveform,
    plot_fft_phase,
    plot_scalogram
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
def chirp_signal():
    """Generate a chirp signal for scalogram testing."""
    sr = 1000.0
    duration = 1.0
    import librosa # Use librosa directly for chirp generation
    x = librosa.chirp(fmin=50, fmax=250, sr=sr, duration=duration).astype(np.float64)
    return x, sr

@pytest.fixture
def noise_signal():
    """Generate random noise."""
    sr = 1000.0
    duration = 1.0
    rng = np.random.default_rng(42)
    x = (rng.random(int(sr * duration)) * 2 - 1).astype(np.float64) # Noise in [-1, 1]
    return x, sr

# --- Helper to check if file exists and is not empty ---
def check_plot_saved(filepath: Path):
    """Checks if a plot file exists and has size > 100 bytes."""
    assert filepath.exists(), f"Plot file not found: {filepath}"
    assert filepath.stat().st_size > 100, f"Plot file is too small (likely empty): {filepath}"
    # Check if all matplotlib figures are closed
    assert plt.get_fignums() == [], "Plot figure was not closed properly."


# --- Test Functions ---

def test_plot_spectrogram(tmp_path: Path, sine_signal):
    """Test generating a spectrogram plot with default and custom options."""
    x, sr = sine_signal
    out_file_default = tmp_path / "spectrogram_default.png"
    out_file_custom = tmp_path / "spectrogram_custom.png"

    # Test default
    plot_spectrogram(x, sr, str(out_file_default))
    check_plot_saved(out_file_default)

    # Test custom options
    plot_spectrogram(x, sr, str(out_file_custom),
                     f_min=50, f_max=150, db_scale=False, cmap='magma',
                     nperseg=512, noverlap=256, title="Custom Spectrogram")
    check_plot_saved(out_file_custom)

def test_plot_fft_magnitude(tmp_path: Path, sine_signal):
    """Test generating an FFT magnitude plot."""
    x, sr = sine_signal
    out_file = tmp_path / "fft_magnitude.png"
    plot_fft_magnitude(x, sr, str(out_file), window='blackman', n=2048) # Test window and n
    check_plot_saved(out_file)

def test_plot_fft_phase(tmp_path: Path, sine_signal):
    """Test generating an FFT phase plot (wrapped and unwrapped)."""
    x, sr = sine_signal
    out_file_wrap = tmp_path / "fft_phase_wrapped.png"
    out_file_unwrap = tmp_path / "fft_phase_unwrapped.png"

    # Test wrapped phase
    plot_fft_phase(x, sr, str(out_file_wrap), unwrap=False, title="FFT Phase (Wrapped)")
    check_plot_saved(out_file_wrap)

    # Test unwrapped phase
    plot_fft_phase(x, sr, str(out_file_unwrap), unwrap=True, title="FFT Phase (Unwrapped)")
    check_plot_saved(out_file_unwrap)

def test_plot_waveform(tmp_path: Path, noise_signal):
    """Test generating a waveform plot."""
    x, sr = noise_signal
    out_file = tmp_path / "waveform.png"
    plot_waveform(x, sr, str(out_file))
    check_plot_saved(out_file)

def test_plot_waveform_max_samples(tmp_path: Path, noise_signal):
    """Test waveform plot with max_samples limit."""
    x, sr = noise_signal
    max_samples = 100
    out_file = tmp_path / "waveform_limited.png"
    plot_waveform(x, sr, str(out_file), max_samples=max_samples)
    check_plot_saved(out_file)

def test_plot_scalogram(tmp_path: Path, chirp_signal):
    """Test generating a wavelet scalogram plot with frequency axis."""
    x, sr = chirp_signal # Chirp is good for scalograms
    out_file = tmp_path / "scalogram_freq.png"
    # Use default scales and wavelet ('morl'), provide sr for frequency axis
    # FIX: Removed pytest.warns(None) context manager
    plot_scalogram(x, str(out_file), sr=sr, wavelet='cmor1.5-1.0') # Use complex morlet
    check_plot_saved(out_file)


def test_plot_scalogram_custom_scales_no_sr(tmp_path: Path, chirp_signal):
    """Test scalogram with custom scales and no sr (scale axis)."""
    x, sr = chirp_signal
    out_file = tmp_path / "scalogram_scale_axis.png"
    scales = np.geomspace(1, 128, num=50) # Example custom scales
    # FIX: Removed pytest.warns(None) context manager
    plot_scalogram(x, str(out_file), scales=scales, wavelet='gaus1', sr=None) # No sr
    check_plot_saved(out_file)


# Test edge cases like empty data
def test_plot_empty_data(tmp_path: Path):
    """Test plotting functions with empty data."""
    x = np.array([], dtype=np.float64)
    sr = 1000.0
    out_file_base = tmp_path / "empty_plot"

    # Expect functions to handle empty data gracefully (log warning, not raise error, no file)
    plot_spectrogram(x, sr, str(out_file_base / "spec.png"))
    plot_fft_magnitude(x, sr, str(out_file_base / "fft_mag.png"))
    plot_fft_phase(x, sr, str(out_file_base / "fft_phase.png"))
    plot_waveform(x, sr, str(out_file_base / "wave.png"))
    plot_scalogram(x, str(out_file_base / "scal.png"), sr=sr)

    # Check that no plot files were actually created
    assert not (out_file_base / "spec.png").exists()
    assert not (out_file_base / "fft_mag.png").exists()
    assert not (out_file_base / "fft_phase.png").exists()
    assert not (out_file_base / "wave.png").exists()
    assert not (out_file_base / "scal.png").exists()
    # Check that no figures are left open
    assert plt.get_fignums() == [], "Plot figures were left open after empty data tests."
