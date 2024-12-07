import pytest
import numpy as np
from sygnals.core.transforms import wavelet_transform, wavelet_reconstruction

def test_wavelet_transform():
    data = np.sin(np.linspace(0,2*np.pi,1024))
    coeffs = wavelet_transform(data, wavelet='db4', level=3)
    assert len(coeffs) == 4  # level=3 should produce 4 sets of coefficients

def test_wavelet_reconstruction():
    data = np.random.randn(1024)
    coeffs = wavelet_transform(data, wavelet='db4', level=3)
    rec = wavelet_reconstruction(coeffs, wavelet='db4')
    assert np.allclose(data, rec, atol=1e-7)
