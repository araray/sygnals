# tests/test_ml_utils.py

"""
Tests for ML utility functions in sygnals.core.ml_utils (scaling, formatters).
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Any # Added List, Tuple, Any

# Import functions/classes to test
from sygnals.core.ml_utils.scaling import apply_scaling, _SKLEARN_AVAILABLE
from sygnals.core.ml_utils.formatters import ( # Import implemented formatters
    format_feature_vectors_per_segment,
    format_feature_sequences,
    format_features_as_image,
    AGGREGATION_FUNCS # Import aggregation functions for testing
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
    rng = np.random.default_rng(123)
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

@pytest.fixture
def sample_features_dict() -> Dict[str, NDArray[np.float64]]:
    """Generate a sample dictionary of 1D feature arrays (like from feature extraction)."""
    n_frames = 50
    rng = np.random.default_rng(789)
    return {
        "rms": rng.random(n_frames).astype(np.float64) * 0.5,
        "zcr": rng.random(n_frames).astype(np.float64) * 0.1,
        "centroid": rng.random(n_frames).astype(np.float64) * 1000 + 500,
    }

@pytest.fixture
def sample_segment_indices() -> List[Tuple[int, int]]:
    """Generate sample segment indices (start_frame, end_frame)."""
    # Corresponds to 50 frames total
    return [
        (0, 15),   # Segment 0: frames 0-14
        (15, 30),  # Segment 1: frames 15-29
        (30, 50),  # Segment 2: frames 30-49
    ]

@pytest.fixture
def sample_feature_map() -> NDArray[np.float64]:
    """Generate a sample 2D feature map (like a spectrogram)."""
    rng = np.random.default_rng(101)
    # Shape (n_bins, n_frames)
    return rng.random((64, 80)).astype(np.float64) * 10


# --- Scaling Tests (Keep Existing) ---

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_apply_scaling_standard(sample_features_array):
    features = sample_features_array
    scaled_features, scaler = apply_scaling(features, scaler_type='standard')
    assert isinstance(scaler, StandardScaler)
    assert scaled_features.shape == features.shape and scaled_features.dtype == np.float64
    assert_allclose(np.mean(scaled_features, axis=0), 0.0, atol=1e-7)
    assert_allclose(np.std(scaled_features, axis=0), 1.0, atol=1e-7)

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_apply_scaling_minmax(sample_features_array):
    features = sample_features_array
    scaled_features, scaler = apply_scaling(features, scaler_type='minmax')
    assert isinstance(scaler, MinMaxScaler)
    assert scaled_features.shape == features.shape and scaled_features.dtype == np.float64
    assert_allclose(np.min(scaled_features, axis=0), 0.0, atol=1e-7)
    assert_allclose(np.max(scaled_features, axis=0), 1.0, atol=1e-7)

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_apply_scaling_robust(sample_features_array):
    features = sample_features_array
    scaled_features, scaler = apply_scaling(features, scaler_type='robust')
    assert isinstance(scaler, RobustScaler)
    assert scaled_features.shape == features.shape and scaled_features.dtype == np.float64
    assert_allclose(np.median(scaled_features, axis=0), 0.0, atol=1e-7)

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_apply_scaling_1d_input(sample_features_1d):
    features = sample_features_1d
    scaled_features, scaler = apply_scaling(features, scaler_type='standard')
    assert isinstance(scaler, StandardScaler)
    assert scaled_features.shape == (len(features), 1) and scaled_features.dtype == np.float64
    assert_allclose(np.mean(scaled_features), 0.0, atol=1e-7)
    assert_allclose(np.std(scaled_features), 1.0, atol=1e-7)

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed, skipping scaling tests")
def test_apply_scaling_pre_fitted(sample_features_array):
    features = sample_features_array
    scaler_to_fit = StandardScaler()
    scaler_to_fit.fit(features[:25])
    scaled_features, used_scaler = apply_scaling(features[25:], fit=False, scaler_instance=scaler_to_fit)
    assert used_scaler is scaler_to_fit
    assert scaled_features.shape == (25, features.shape[1])
    original_second_half = used_scaler.inverse_transform(scaled_features)
    assert_allclose(original_second_half, features[25:], atol=1e-7)

def test_apply_scaling_no_sklearn():
    if not _SKLEARN_AVAILABLE:
        features = np.random.rand(10, 2)
        with pytest.raises(ImportError, match="scikit-learn package is required"):
            apply_scaling(features, scaler_type='standard')
    else: pytest.skip("scikit-learn is installed")

def test_apply_scaling_invalid_type(sample_features_array):
    features = sample_features_array
    with pytest.raises(ValueError): apply_scaling(features, scaler_type='invalid_scaler') # type: ignore

def test_apply_scaling_no_fit_no_instance(sample_features_array):
    features = sample_features_array
    with pytest.raises(ValueError): apply_scaling(features, fit=False, scaler_instance=None)

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_apply_scaling_unfitted_instance(sample_features_array):
    features = sample_features_array
    unfitted_scaler = StandardScaler()
    with pytest.raises(ValueError): apply_scaling(features, fit=False, scaler_instance=unfitted_scaler)


# --- Formatter Tests ---

# Tests for format_feature_vectors_per_segment
def test_format_vectors_basic_mean_df(sample_features_dict, sample_segment_indices):
    """Test basic vector formatting with mean aggregation (DataFrame output)."""
    features = sample_features_dict
    segments = sample_segment_indices
    num_segments = len(segments)
    feature_names = list(features.keys())

    result_df = format_feature_vectors_per_segment(features, segments, aggregation='mean', output_format='dataframe')

    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape == (num_segments, len(feature_names))
    assert list(result_df.columns) == feature_names
    assert result_df.index.name == "segment_index"

    # Check aggregation for one segment and one feature
    seg0_rms_mean = np.mean(features['rms'][segments[0][0]:segments[0][1]])
    assert np.isclose(result_df.loc[0, 'rms'], seg0_rms_mean)

def test_format_vectors_mixed_agg_numpy(sample_features_dict, sample_segment_indices):
    """Test vector formatting with mixed aggregation (NumPy output)."""
    features = sample_features_dict
    segments = sample_segment_indices
    num_segments = len(segments)
    feature_names = list(features.keys()) # ['rms', 'zcr', 'centroid']
    aggregation = {'rms': 'max', 'zcr': 'min', 'centroid': 'std'} # Use std for centroid

    result_np = format_feature_vectors_per_segment(features, segments, aggregation=aggregation, output_format='numpy')

    assert isinstance(result_np, np.ndarray)
    assert result_np.shape == (num_segments, len(feature_names))
    assert result_np.dtype == np.float64

    # Check aggregation for segment 1
    seg1_start, seg1_end = segments[1]
    expected_rms_max = np.max(features['rms'][seg1_start:seg1_end])
    expected_zcr_min = np.min(features['zcr'][seg1_start:seg1_end])
    expected_centroid_std = np.std(features['centroid'][seg1_start:seg1_end])

    assert np.isclose(result_np[1, feature_names.index('rms')], expected_rms_max)
    assert np.isclose(result_np[1, feature_names.index('zcr')], expected_zcr_min)
    assert np.isclose(result_np[1, feature_names.index('centroid')], expected_centroid_std)

def test_format_vectors_with_labels(sample_features_dict, sample_segment_indices):
    """Test vector formatting with segment labels."""
    features = sample_features_dict
    segments = sample_segment_indices
    labels = ['speech', 'music', 'noise']

    result_df = format_feature_vectors_per_segment(features, segments, segment_labels=labels, output_format='dataframe')

    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.index) == labels

def test_format_vectors_invalid_input(sample_features_dict, sample_segment_indices):
    """Test vector formatting with invalid inputs."""
    # Mismatched feature lengths
    bad_features = sample_features_dict.copy()
    bad_features['rms'] = bad_features['rms'][:-1] # Shorter rms
    with pytest.raises(ValueError, match="All feature arrays.*must have the same length"):
        format_feature_vectors_per_segment(bad_features, sample_segment_indices)

    # Invalid segment index
    bad_segments = [(0, 10), (10, 60)] # Segment 1 end index > num_frames (50)
    with pytest.warns(UserWarning, match=r"Invalid segment indices \(10, 60\)"): # Check warning for skip
         result = format_feature_vectors_per_segment(sample_features_dict, bad_segments)
         # Expect NaN row for the invalid segment
         assert np.isnan(result.iloc[1]).all()

    # Unknown aggregation
    with pytest.raises(ValueError, match="Unknown global aggregation function"):
        format_feature_vectors_per_segment(sample_features_dict, sample_segment_indices, aggregation='unknown')
    with pytest.raises(ValueError, match="Unknown aggregation function 'bad' for feature 'rms'"):
        format_feature_vectors_per_segment(sample_features_dict, sample_segment_indices, aggregation={'rms': 'bad'})

    # Mismatched labels
    with pytest.raises(ValueError, match="Length of segment_labels must match"):
        format_feature_vectors_per_segment(sample_features_dict, sample_segment_indices, segment_labels=['a', 'b'])


# Tests for format_feature_sequences
def test_format_sequences_basic_list(sample_features_dict):
    """Test basic sequence formatting (list output)."""
    features = sample_features_dict
    num_frames = len(features['rms'])
    num_features = len(features)

    result_list = format_feature_sequences(features, output_format='list_of_arrays')

    assert isinstance(result_list, list)
    assert len(result_list) == 1
    seq_array = result_list[0]
    assert isinstance(seq_array, np.ndarray)
    assert seq_array.shape == (num_frames, num_features)
    assert seq_array.dtype == np.float64
    # Check if columns match original features (order might vary based on dict keys)
    # assert_allclose(seq_array[:, 0], features['rms']) # This depends on key order

def test_format_sequences_padding_trunc(sample_features_dict):
    """Test sequence formatting with padding and truncation."""
    features = sample_features_dict
    num_frames = len(features['rms']) # 50
    num_features = len(features)

    # Padding
    max_len_pad = 60
    result_pad = format_feature_sequences(features, max_sequence_length=max_len_pad, padding_value=-1.0, output_format='padded_array')
    assert isinstance(result_pad, np.ndarray)
    assert result_pad.shape == (1, max_len_pad, num_features)
    assert np.all(result_pad[0, num_frames:, :] == -1.0) # Check padding value
    assert_allclose(result_pad[0, :num_frames, list(features.keys()).index('rms')], features['rms']) # Check original data part

    # Truncation (post)
    max_len_trunc = 40
    result_trunc_post = format_feature_sequences(features, max_sequence_length=max_len_trunc, truncation_strategy='post', output_format='padded_array')
    assert result_trunc_post.shape == (1, max_len_trunc, num_features)
    assert_allclose(result_trunc_post[0, :, list(features.keys()).index('rms')], features['rms'][:max_len_trunc])

    # Truncation (pre)
    result_trunc_pre = format_feature_sequences(features, max_sequence_length=max_len_trunc, truncation_strategy='pre', output_format='padded_array')
    assert result_trunc_pre.shape == (1, max_len_trunc, num_features)
    assert_allclose(result_trunc_pre[0, :, list(features.keys()).index('rms')], features['rms'][num_frames - max_len_trunc:])

def test_format_sequences_invalid(sample_features_dict):
    """Test sequence formatting with invalid inputs."""
    # Mismatched lengths
    bad_features = sample_features_dict.copy()
    bad_features['rms'] = bad_features['rms'][:-1]
    with pytest.raises(ValueError, match="All feature arrays.*must have the same length"):
        format_feature_sequences(bad_features)

    # Invalid truncation strategy
    with pytest.raises(ValueError, match="Unknown truncation_strategy"):
        format_feature_sequences(sample_features_dict, max_sequence_length=40, truncation_strategy='middle') # type: ignore


# Tests for format_features_as_image
def test_format_image_basic(sample_feature_map):
    """Test basic image formatting (no resize/norm)."""
    feature_map = sample_feature_map
    result = format_features_as_image(feature_map, normalize=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == feature_map.shape
    assert result.dtype == np.float64
    assert_equal(result, feature_map) # Should be identical

def test_format_image_resize(sample_feature_map):
    """Test image formatting with resizing."""
    feature_map = sample_feature_map
    target_shape = (32, 40) # Different aspect ratio
    result = format_features_as_image(feature_map, output_shape=target_shape, normalize=False)
    assert result.shape == target_shape
    assert result.dtype == np.float64

def test_format_image_normalize(sample_feature_map):
    """Test image formatting with normalization."""
    feature_map = sample_feature_map
    result = format_features_as_image(feature_map, normalize=True)
    assert result.shape == feature_map.shape
    assert result.dtype == np.float64
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0
    # Check if range is actually close to 0-1 (unless input was constant)
    if np.ptp(feature_map) > 1e-9: # ptp = peak-to-peak (range)
        assert np.isclose(np.min(result), 0.0)
        assert np.isclose(np.max(result), 1.0)

def test_format_image_resize_normalize(sample_feature_map):
    """Test image formatting with both resizing and normalization."""
    feature_map = sample_feature_map
    target_shape = (100, 100)
    result = format_features_as_image(feature_map, output_shape=target_shape, normalize=True)
    assert result.shape == target_shape
    assert result.dtype == np.float64
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0
    if np.ptp(feature_map) > 1e-9:
        assert np.isclose(np.min(result), 0.0)
        assert np.isclose(np.max(result), 1.0)

def test_format_image_invalid(sample_feature_map):
    """Test image formatting with invalid inputs."""
    # Invalid input shape
    with pytest.raises(ValueError, match="Input feature_map must be a 2D array"):
        format_features_as_image(np.random.rand(10)) # 1D input

    # Invalid output_shape
    with pytest.raises(ValueError, match="output_shape must be a tuple of two positive integers"):
        format_features_as_image(sample_feature_map, output_shape=(10,)) # type: ignore
    with pytest.raises(ValueError, match="output_shape must be a tuple of two positive integers"):
        format_features_as_image(sample_feature_map, output_shape=(-10, 10))
