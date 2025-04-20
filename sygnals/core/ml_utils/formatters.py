# sygnals/core/ml_utils/formatters.py

"""
Functions for formatting extracted features into structures suitable for ML models.

Handles tasks like creating feature vectors per segment, sequences of vectors,
or image-like arrays (e.g., from spectrograms).
"""

import logging
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Union, List, Dict, Optional, Any, Literal, Tuple, Callable
# Import zoom for image resizing
from scipy.ndimage import zoom
import warnings # Import the warnings module

logger = logging.getLogger(__name__)

# Define a small epsilon for safe division and log calculations
_EPSILON = np.finfo(np.float64).eps

# --- Aggregation Functions ---
# Define common aggregation functions to handle potential NaNs
# TODO: Consider adding more complex aggregations if needed

def _nanagg(func: Callable) -> Callable:
    """Decorator to handle NaN values gracefully in aggregation functions."""
    def wrapper(a: NDArray[np.float64], *args, **kwargs) -> np.float64:
        if a.size == 0 or np.all(np.isnan(a)):
            return np.nan
        # Filter out NaNs before applying the function
        valid_a = a[~np.isnan(a)]
        if valid_a.size == 0: # Handle case where only NaNs were present
             return np.nan
        return func(valid_a, *args, **kwargs)
    return wrapper

AGGREGATION_FUNCS: Dict[str, Callable] = {
    'mean': _nanagg(np.mean),
    'std': _nanagg(np.std),
    'median': _nanagg(np.median),
    'min': _nanagg(np.min),
    'max': _nanagg(np.max),
    # Add others like 'sum', 'first', 'last' if needed
}

# --- Formatter Functions ---

def format_feature_vectors_per_segment(
    features_dict: Dict[str, NDArray[np.float64]],
    segment_indices: List[Tuple[int, int]], # List of (start_frame, end_frame) tuples (exclusive end)
    aggregation: Union[str, Dict[str, str]] = 'mean',
    output_format: Literal['dataframe', 'numpy'] = 'dataframe',
    segment_labels: Optional[List[Any]] = None # Optional labels for segments
) -> Union[pd.DataFrame, NDArray[np.float64]]:
    """
    Formats features into vectors, one vector per segment, by aggregating frame features.

    Args:
        features_dict: Dictionary where keys are feature names and values are
                       1D NumPy arrays (n_frames,) representing frame-level features.
                       All arrays must have the same length.
        segment_indices: List of tuples defining segment boundaries as
                         (start_frame_index, end_frame_index). Indices are inclusive
                         of start, exclusive of end, relative to the feature arrays.
        aggregation: Method(s) to aggregate frame-level features within each segment.
                     - If string: Aggregation function name ('mean', 'std', 'median',
                       'min', 'max') applied to all features.
                     - If dict: Maps feature names to aggregation function names.
                       Features not in the dict use 'mean'.
                     (Default: 'mean')
        output_format: 'dataframe' returns a Pandas DataFrame indexed by segment number
                       (or labels if provided). 'numpy' returns a 2D NumPy array
                       (n_segments, n_features). (Default: 'dataframe')
        segment_labels: Optional list of labels corresponding to each segment in
                        `segment_indices`. Used for DataFrame index if provided.
                        Must have the same length as `segment_indices`.

    Returns:
        A DataFrame or NumPy array containing the aggregated feature vectors per segment.
        Returns empty DataFrame/array if no segments or features.

    Raises:
        ValueError: If input dictionary is empty, feature lengths mismatch,
                    segment indices are invalid (but handled with warning),
                    aggregation method is unknown,
                    or segment_labels length mismatches segment_indices length.
    """
    if not features_dict:
        logger.warning("Input features_dict is empty. Returning empty result.")
        return pd.DataFrame() if output_format == 'dataframe' else np.empty((0, 0), dtype=np.float64)

    feature_names = list(features_dict.keys())
    # Validate feature array lengths
    frame_counts = [len(arr) for arr in features_dict.values()]
    if not frame_counts: # Should not happen if features_dict is not empty
        return pd.DataFrame() if output_format == 'dataframe' else np.empty((0, 0), dtype=np.float64)

    num_frames = frame_counts[0]
    if not all(count == num_frames for count in frame_counts):
        raise ValueError(f"All feature arrays in features_dict must have the same length. Found lengths: {frame_counts}")

    if not segment_indices:
        logger.warning("No segment indices provided. Returning empty result.")
        return pd.DataFrame() if output_format == 'dataframe' else np.empty((0, len(feature_names)), dtype=np.float64)

    if segment_labels is not None and len(segment_labels) != len(segment_indices):
        raise ValueError("Length of segment_labels must match length of segment_indices.")

    logger.info(f"Formatting {len(feature_names)} features into vectors for {len(segment_indices)} segments using aggregation: {aggregation}")

    num_segments = len(segment_indices)
    num_features = len(feature_names)
    aggregated_vectors = np.full((num_segments, num_features), np.nan, dtype=np.float64)

    # Determine aggregation function for each feature
    agg_funcs: Dict[str, Callable] = {}
    if isinstance(aggregation, str):
        if aggregation not in AGGREGATION_FUNCS:
            raise ValueError(f"Unknown global aggregation function: '{aggregation}'. Available: {list(AGGREGATION_FUNCS.keys())}")
        agg_func = AGGREGATION_FUNCS[aggregation]
        agg_funcs = {name: agg_func for name in feature_names}
    elif isinstance(aggregation, dict):
        default_agg_func = AGGREGATION_FUNCS['mean']
        for name in feature_names:
            method_name = aggregation.get(name, 'mean')
            if method_name not in AGGREGATION_FUNCS:
                raise ValueError(f"Unknown aggregation function '{method_name}' for feature '{name}'. Available: {list(AGGREGATION_FUNCS.keys())}")
            agg_funcs[name] = AGGREGATION_FUNCS[method_name]
    else:
        raise TypeError("aggregation must be a string or a dictionary.")

    # Aggregate features for each segment
    for i, (start_frame, end_frame) in enumerate(segment_indices):
        # Validate segment indices
        # FIX: Changed condition to check start < end
        if not (0 <= start_frame < num_frames and start_frame < end_frame and end_frame <= num_frames):
            warning_msg = (f"Invalid segment indices ({start_frame}, {end_frame}) for num_frames={num_frames}. "
                           f"Skipping segment {i}.")
            logger.warning(warning_msg)
            # --- FIX: Emit UserWarning ---
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
            # --- End Fix ---
            continue # Skip invalid segment, result will have NaN row

        for j, feat_name in enumerate(feature_names):
            segment_data = features_dict[feat_name][start_frame:end_frame]
            agg_func = agg_funcs[feat_name]
            aggregated_vectors[i, j] = agg_func(segment_data)

    # Format output
    if output_format == 'dataframe':
        index = segment_labels if segment_labels is not None else pd.RangeIndex(num_segments, name="segment_index")
        df = pd.DataFrame(aggregated_vectors, columns=feature_names, index=index)
        # Optionally drop rows where all features are NaN (due to skipped segments)
        # df = df.dropna(axis=0, how='all')
        return df
    elif output_format == 'numpy':
        return aggregated_vectors
    else:
        raise ValueError(f"Unknown output_format: '{output_format}'. Choose 'dataframe' or 'numpy'.")


def format_feature_sequences(
    features_dict: Dict[str, NDArray[np.float64]],
    max_sequence_length: Optional[int] = None,
    padding_value: float = 0.0,
    truncation_strategy: Literal['pre', 'post'] = 'post',
    output_format: Literal['list_of_arrays', 'padded_array'] = 'list_of_arrays'
) -> Union[List[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Formats features into sequences (e.g., for RNNs/Transformers).

    Stacks features for each frame into a sequence. Handles padding or truncation
    to ensure uniform sequence length if required. Assumes input dictionary
    represents features for a single sequence.

    Args:
        features_dict: Dictionary where keys are feature names and values are
                       1D NumPy arrays (n_frames,) representing frame-level features.
                       All arrays must have the same length.
        max_sequence_length: If specified (> 0), sequences will be padded or truncated
                             to this length.
        padding_value: Value used for padding shorter sequences. (Default: 0.0)
        truncation_strategy: 'pre' (remove from beginning) or 'post' (remove from end).
                             (Default: 'post')
        output_format: 'list_of_arrays' returns a list containing one NumPy array
                       (sequence_length, n_features). 'padded_array' returns a single
                       3D NumPy array (1, max_sequence_length, n_features). (Default: 'list_of_arrays')

    Returns:
        List containing one sequence array, or a single 3D padded array.
        Returns empty list/array if input is empty.

    Raises:
        ValueError: If input is invalid (empty dict, mismatched lengths, invalid params).
    """
    if not features_dict:
        logger.warning("Input features_dict is empty. Returning empty result.")
        return [] if output_format == 'list_of_arrays' else np.empty((1, max_sequence_length or 0, 0), dtype=np.float64)

    feature_names = list(features_dict.keys())
    # Validate feature array lengths
    frame_counts = [len(arr) for arr in features_dict.values()]
    if not frame_counts:
        return [] if output_format == 'list_of_arrays' else np.empty((1, max_sequence_length or 0, 0), dtype=np.float64)

    num_frames = frame_counts[0]
    if not all(count == num_frames for count in frame_counts):
        raise ValueError(f"All feature arrays in features_dict must have the same length. Found lengths: {frame_counts}")

    if num_frames == 0:
        logger.warning("Input features have zero length (no frames). Returning empty result.")
        return [] if output_format == 'list_of_arrays' else np.empty((1, max_sequence_length or 0, len(feature_names)), dtype=np.float64)

    num_features = len(feature_names)
    logger.info(f"Formatting {num_features} features into sequence (original length: {num_frames} frames).")

    # Stack features into (n_frames, n_features) array
    sequence_array = np.stack([features_dict[name] for name in feature_names], axis=1).astype(np.float64)

    # Apply padding/truncation if max_sequence_length is specified
    final_sequence_length = num_frames
    if max_sequence_length is not None and max_sequence_length > 0:
        final_sequence_length = max_sequence_length
        if num_frames > max_sequence_length:
            # Truncate
            logger.debug(f"Truncating sequence from {num_frames} to {max_sequence_length} frames (strategy: {truncation_strategy}).")
            if truncation_strategy == 'post':
                sequence_array = sequence_array[:max_sequence_length, :]
            elif truncation_strategy == 'pre':
                sequence_array = sequence_array[num_frames - max_sequence_length:, :]
            else:
                raise ValueError(f"Unknown truncation_strategy: '{truncation_strategy}'. Choose 'pre' or 'post'.")
        elif num_frames < max_sequence_length:
            # Pad
            logger.debug(f"Padding sequence from {num_frames} to {max_sequence_length} frames (value: {padding_value}).")
            pad_width = max_sequence_length - num_frames
            # Padding applied after the sequence ((before, after), (before, after))
            padding = ((0, pad_width), (0, 0))
            sequence_array = np.pad(sequence_array, pad_width=padding, mode='constant', constant_values=padding_value)
        # else: num_frames == max_sequence_length, no action needed

    # Format output
    if output_format == 'list_of_arrays':
        return [sequence_array]
    elif output_format == 'padded_array':
        # Reshape to (1, sequence_length, n_features)
        return np.expand_dims(sequence_array, axis=0)
    else:
        raise ValueError(f"Unknown output_format: '{output_format}'. Choose 'list_of_arrays' or 'padded_array'.")


def format_features_as_image(
    feature_map: NDArray[np.float64], # Expects spectrogram-like data
    output_shape: Optional[Tuple[int, int]] = None, # Target (height, width)
    resize_order: int = 1, # Order for interpolation (1=bilinear)
    normalize: bool = True # Whether to normalize values to [0, 1]
) -> NDArray[np.float64]:
    """
    Formats features (typically a 2D map like a spectrogram) into an image-like array.

    Optionally resizes the feature map and normalizes its values to the range [0, 1].

    Args:
        feature_map: 2D NumPy array representing the feature map (e.g., Mel spectrogram).
                     Shape: (n_frequency_bins, n_time_frames).
        output_shape: Optional target shape (height, width) for resizing. If None,
                      no resizing is performed.
        resize_order: The order of the spline interpolation for resizing (0-5).
                      0: Nearest neighbor. 1: Bilinear (default). 3: Bicubic.
        normalize: If True (default), scales the output image values to the range [0, 1].

    Returns:
        A 2D NumPy array representing the formatted image (float64).

    Raises:
        ValueError: If input feature_map is not 2D or resize parameters are invalid.
    """
    if feature_map.ndim != 2:
        raise ValueError("Input feature_map must be a 2D array.")
    if feature_map.size == 0:
        logger.warning("Input feature_map is empty. Returning empty array.")
        return np.empty(output_shape or (0, 0), dtype=np.float64)

    logger.info(f"Formatting feature map (shape {feature_map.shape}) as image.")
    image_out = feature_map.astype(np.float64) # Ensure float64

    # --- Resizing ---
    if output_shape is not None:
        if not isinstance(output_shape, tuple) or len(output_shape) != 2 or \
           not all(isinstance(dim, int) and dim > 0 for dim in output_shape):
            raise ValueError("output_shape must be a tuple of two positive integers (height, width).")

        if image_out.shape == output_shape:
            logger.debug("Output shape matches input shape, no resizing needed.")
        else:
            logger.debug(f"Resizing feature map from {image_out.shape} to {output_shape} (order={resize_order}).")
            zoom_factors = (output_shape[0] / image_out.shape[0], output_shape[1] / image_out.shape[1])
            try:
                image_out = zoom(image_out, zoom_factors, order=resize_order, mode='nearest')
                # Ensure exact output shape after zoom (can sometimes be off by 1 due to rounding)
                if image_out.shape != output_shape:
                     logger.warning(f"Zoom result shape {image_out.shape} differs slightly from target {output_shape}. Clipping/Padding.")
                     # Basic clipping/padding to force shape - more robust resizing might be needed
                     h, w = output_shape
                     img_h, img_w = image_out.shape
                     image_out = image_out[:h, :w] # Clip if larger
                     # Pad if smaller (less likely with zoom > 0)
                     pad_h = h - image_out.shape[0]
                     pad_w = w - image_out.shape[1]
                     if pad_h > 0 or pad_w > 0:
                          image_out = np.pad(image_out, ((0, pad_h), (0, pad_w)), mode='constant')

            except Exception as e:
                 logger.error(f"Error during image resizing: {e}")
                 raise ValueError("Failed to resize feature map.") from e

    # --- Normalization ---
    if normalize:
        logger.debug("Normalizing image values to [0, 1].")
        min_val = np.min(image_out)
        max_val = np.max(image_out)
        range_val = max_val - min_val
        if range_val < _EPSILON:
            # Handle constant image (avoid division by zero)
            # Map to 0 or 0.5 depending on preference
            image_out = np.full_like(image_out, 0.0)
        else:
            image_out = (image_out - min_val) / range_val

    return image_out
