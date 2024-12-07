import numpy as np
import pywt
from scipy.fft import fft, ifft


# Fast Fourier Transform
def fft(data):
    """Compute the FFT of a signal."""
    return np.fft.fft(data)


def ifft(spectrum):
    """Compute the inverse FFT."""
    return np.fft.ifft(spectrum).real


# Wavelet Transform
def wavelet_transform(data, wavelet="db4", level=3):
    """Perform Wavelet Transform."""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs


def wavelet_reconstruction(coeffs, wavelet="db4"):
    """Reconstruct signal from Wavelet coefficients."""
    return pywt.waverec(coeffs, wavelet)


# Laplace Transform
def laplace_transform(data, s_values):
    """Compute Laplace Transform numerically."""
    return np.array(
        [np.sum(data * np.exp(-s * np.arange(len(data)))) for s in s_values]
    )


def inverse_laplace_transform(transform, t_values):
    """Compute inverse Laplace Transform numerically (simplified example)."""
    return np.array([np.sum(transform * np.exp(s * t)) for t in t_values])
