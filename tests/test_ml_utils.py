# tests/test_ml_utils.py

"""
Tests for ML utility functions in sygnals.core.ml_utils (scaling, formatters).
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal

# Import functions/classes to test
from sygnals.core.ml_utils.scaling import apply_scaling, _SKLEARN_AVAILABLE
from sygnals.core.ml_utils.formatters import ( # Import placeholders
    format_feature_vectors_per_segment,
    format_feature_sequences,
    format_features_as_image
)

# Conditionally import sklearn for checking fitted scalers
if _SKLEARN_AVAILABLE:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
else:
    # Define dummy types if sklearn not present, tests will be skipped
    StandardScaler = type("StandardScaler", (), {})
    MinMaxScaler = type("MinMaxScaler", (), {})
    RobustScaler = type("RobustScaler", (), {})


# --- Test Fixtures ---

@pytest.fixture
def sample_features_array() -> np.ndarray:
    """Generate a sample 2D feature array."""
    # Use seed for reproducibility
    rng = np.random.default_rng(123)
    # Create features with different scales
    feat1 = rng.normal(loc=10, scale=2, size=(50, 1))
    feat2 = rng.normal(loc=0, scale=0.1, size=(50, 1))
    feat3 = rng.uniform(low=-5, high=5, size=(50, 1))
    features = np.hstack([feat1, feat2, feat3]).astype(np.float64)
    return features

@pytest.fixture
def sample_features_1d() -> np.ndarray:
    """Generate a sample 1D feature array."""
    rng = np.random.default_rng(456)
    return rng.normal(loc=5, scale=3, size=100).astype(np.float64)


# --- Scaling Tests ---

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_apply_scaling_standard(sample_features_array):
    """Test StandardScaler via apply_scaling."""
    features = sample_features_array
    scaled_features, scaler = apply_scaling(features, scaler_type='standard')

    assert isinstance(scaler, StandardScaler)
    assert scaled_features.shape == features.shape
    assert scaled_features.dtype == np.float64
    # Check if mean is close to 0 and std dev is close to 1 for scaled features
    assert_allclose(np.mean(scaled_features, axis=0), 0.0, atol=1e-7)
    assert_allclose(np.std(scaled_features, axis=0), 1.0, atol=1e-7)

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_apply_scaling_minmax(sample_features_array):
    """Test MinMaxScaler via apply_scaling."""
    features = sample_features_array
    scaled_features, scaler = apply_scaling(features, scaler_type='minmax')

    assert isinstance(scaler, MinMaxScaler)
    assert scaled_features.shape == features.shape
    assert scaled_features.dtype == np.float64
    # Check if features are scaled to default range [0, 1]
    assert_allclose(np.min(scaled_features, axis=0), 0.0, atol=1e-7)
    assert_allclose(np.max(scaled_features, axis=0), 1.0, atol=1e-7)

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_apply_scaling_robust(sample_features_array):
    """Test RobustScaler via apply_scaling."""
    features = sample_features_array
    scaled_features, scaler = apply_scaling(features, scaler_type='robust')

    assert isinstance(scaler, RobustScaler)
    assert scaled_features.shape == features.shape
    assert scaled_features.dtype == np.float64
    # RobustScaler centers based on median and scales based on IQR.
    # Check if median is close to 0 (if centering is enabled by default)
    assert_allclose(np.median(scaled_features, axis=0), 0.0, atol=1e-7)

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_apply_scaling_1d_input(sample_features_1d):
    """Test apply_scaling with 1D input."""
    features = sample_features_1d
    scaled_features, scaler = apply_scaling(features, scaler_type='standard')

    assert isinstance(scaler, StandardScaler)
    # Output should still be 2D (n_samples, 1) after internal reshape
    assert scaled_features.shape == (len(features), 1)
    assert scaled_features.dtype == np.float64
    assert_allclose(np.mean(scaled_features), 0.0, atol=1e-7)
    assert_allclose(np.std(scaled_features), 1.0, atol=1e-7)

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_apply_scaling_pre_fitted(sample_features_array):
    """Test applying a pre-fitted scaler."""
    features = sample_features_array
    # Fit a scaler first
    scaler_to_fit = StandardScaler()
    scaler_to_fit.fit(features[:25]) # Fit on first half

    # Apply the fitted scaler to the second half
    scaled_features, used_scaler = apply_scaling(
        features[25:],
        fit=False, # Important: do not refit
        scaler_instance=scaler_to_fit
    )

    assert used_scaler is scaler_to_fit # Should return the same instance
    assert scaled_features.shape == (25, features.shape[1])
    # Mean/std of the transformed second half won't necessarily be 0/1
    # because it was scaled using parameters from the first half.
    # We can check if inverse transform works approximately.
    original_second_half = used_scaler.inverse_transform(scaled_features)
    assert_allclose(original_second_half, features[25:], atol=1e-7)

def test_apply_scaling_no_sklearn():
    """Test that scaling raises ImportError if sklearn is unavailable."""
    if not _SKLEARN_AVAILABLE:
        features = np.random.rand(10, 2)
        with pytest.raises(ImportError, match="scikit-learn package is required"):
            apply_scaling(features, scaler_type='standard')
    else:
        pytest.skip("scikit-learn is installed, skipping this test.")

def test_apply_scaling_invalid_type(sample_features_array):
    """Test apply_scaling with invalid scaler type."""
    features = sample_features_array
    with pytest.raises(ValueError, match="Unsupported scaler_type"):
        apply_scaling(features, scaler_type='invalid_scaler') # type: ignore

def test_apply_scaling_no_fit_no_instance(sample_features_array):
    """Test apply_scaling with fit=False but no scaler_instance."""
    features = sample_features_array
    with pytest.raises(ValueError, match="`scaler_instance` must be provided when `fit=False`"):
        apply_scaling(features, fit=False, scaler_instance=None)

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_apply_scaling_unfitted_instance(sample_features_array):
    """Test apply_scaling with fit=False and an unfitted scaler instance."""
    features = sample_features_array
    unfitted_scaler = StandardScaler() # Not fitted
    with pytest.raises(ValueError, match="Provided `scaler_instance` does not appear to be fitted"):
        apply_scaling(features, fit=False, scaler_instance=unfitted_scaler)


# --- Formatter Tests (Placeholders) ---

def test_format_feature_vectors_placeholder():
    """Test the placeholder for format_feature_vectors_per_segment."""
    features_dict = {'feat1': np.random.rand(100), 'feat2': np.random.rand(100)}
    # Test DataFrame output
    result_df = format_feature_vectors_per_segment(features_dict, output_format='dataframe')
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty
    # Test NumPy output
    result_np = format_feature_vectors_per_segment(features_dict, output_format='numpy')
    assert isinstance(result_np, np.ndarray)
    assert result_np.shape == (1, 0) # Expect shape (1, 0) for empty array

def test_format_feature_sequences_placeholder():
    """Test the placeholder for format_feature_sequences."""
    features_dict = {'feat1': np.random.rand(50), 'feat2': np.random.rand(50)}
    # Test list output
    result_list = format_feature_sequences(features_dict, output_format='list_of_arrays')
    assert isinstance(result_list, list)
    assert len(result_list) == 0
    # Test padded array output
    max_len = 60
    result_padded = format_feature_sequences(features_dict, max_sequence_length=max_len, output_format='padded_array')
    assert isinstance(result_padded, np.ndarray)
    assert result_padded.shape == (1, max_len, 0) # Expect shape (1, max_len, 0 features)

def test_format_features_as_image_placeholder():
    """Test the placeholder for format_features_as_image."""
    spectrogram = np.random.rand(64, 100) # Example spectrogram shape
    result = format_features_as_image(spectrogram)
    assert isinstance(result, np.ndarray)
    # Placeholder returns the input directly
    assert_equal(result, spectrogram)
    # Test invalid input shape
    with pytest.raises(ValueError, match="Input spectrogram_data must be a 2D array"):
        format_features_as_image(np.random.rand(10)) # Pass 1D array
