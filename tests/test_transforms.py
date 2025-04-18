# tests/test_transforms.py

import pytest
import numpy as np
from numpy.testing import assert_allclose # Use assert_allclose for float comparisons

# Import the renamed functions
from sygnals.core.transforms import discrete_wavelet_transform, inverse_discrete_wavelet_transform
# Import hilbert_transform and laplace_transform_numerical if tests are added later
# from sygnals.core.transforms import hilbert_transform, laplace_transform_numerical

def test_discrete_wavelet_transform():
    """Test the Discrete Wavelet Transform function."""
    # Use a simple signal where levels are easy to track
    data = np.sin(np.linspace(0, 4 * np.pi, 1024)).astype(np.float64) # Two cycles
    wavelet = 'db4'
    level = 3
    # Call the renamed function
    coeffs = discrete_wavelet_transform(data, wavelet=wavelet, level=level)

    # Check the number of coefficient arrays returned
    # wavedec returns N+1 arrays: [cAn, cDn, cDn-1, ..., cD1]
    assert isinstance(coeffs, list)
    assert len(coeffs) == level + 1

    # Check dtypes
    for c in coeffs:
        assert c.dtype == np.float64

    # Check coefficient lengths (approximate, depends on wavelet and mode)
    # Lengths roughly halve at each level
    expected_len_cA = len(data) // (2**level)
    assert len(coeffs[0]) > expected_len_cA - 10 # Allow some variation due to filter length
    assert len(coeffs[0]) < expected_len_cA + 10
    for i in range(1, level + 1):
        expected_len_cD = len(data) // (2**(level - i + 1))
        assert len(coeffs[i]) > expected_len_cD - 10
        assert len(coeffs[i]) < expected_len_cD + 10


def test_inverse_discrete_wavelet_transform():
    """Test the Inverse Discrete Wavelet Transform (reconstruction)."""
    # Use a random signal for better reconstruction test
    original_data = np.random.randn(1024).astype(np.float64)
    wavelet = 'sym5' # Use a different wavelet
    level = 4

    # Decompose
    coeffs = discrete_wavelet_transform(original_data, wavelet=wavelet, level=level)

    # Reconstruct using the renamed function
    reconstructed_data = inverse_discrete_wavelet_transform(coeffs, wavelet=wavelet)

    # Check dtype
    assert reconstructed_data.dtype == np.float64

    # Check if reconstruction is close to the original
    # Allow for small numerical errors. Length might differ slightly depending on wavelet/mode.
    # Compare the overlapping part if lengths differ.
    min_len = min(len(original_data), len(reconstructed_data))
    assert_allclose(original_data[:min_len], reconstructed_data[:min_len], atol=1e-9) # Use assert_allclose

# --- Tests for other transforms (add later if needed) ---

# def test_hilbert_transform():
#     # Test Hilbert transform properties (e.g., envelope extraction)
#     pass

# def test_laplace_transform_numerical():
#     # Test numerical Laplace on a known function (e.g., exp(-at))
#     pass
