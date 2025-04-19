# tests/test_segmentation.py

"""
Tests for signal segmentation functions in sygnals.core.segmentation.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from typing import Optional, Union, Literal, Dict, Any, Tuple

# Import segmentation functions to test
from sygnals.core.segmentation import (
    segment_fixed_length,
    segment_by_silence, # Placeholder
    segment_by_event   # Placeholder
)

# --- Test Fixtures ---

@pytest.fixture
def sample_signal_long() -> Tuple[np.ndarray, int]:
    """Generate a moderately long signal for segmentation tests."""
    sr = 1000 # Use easy sample rate
    duration = 5.3 # seconds (not an exact multiple of segment lengths)
    signal = np.arange(int(sr * duration)).astype(np.float64) / sr # Simple ramp
    return signal, sr

# --- Test segment_fixed_length ---

def test_segment_fixed_length_no_overlap_no_pad(sample_signal_long):
    """Test fixed-length segmentation without overlap or padding."""
    signal, sr = sample_signal_long
    seg_len_sec = 1.0
    seg_len_samples = int(seg_len_sec * sr) # 1000 samples

    segments = segment_fixed_length(signal, sr, segment_length_sec=seg_len_sec, overlap_ratio=0.0, pad=False)

    assert isinstance(segments, list)
    expected_num_segments = len(signal) // seg_len_samples # Integer division
    assert len(segments) == expected_num_segments # Should discard last partial segment
    for i, seg in enumerate(segments):
        assert isinstance(seg, np.ndarray)
        assert seg.dtype == np.float64
        assert len(seg) == seg_len_samples
        # Check content matches original signal slice
        expected_content = signal[i * seg_len_samples : (i + 1) * seg_len_samples]
        assert_array_equal(seg, expected_content)

def test_segment_fixed_length_with_overlap_no_pad(sample_signal_long):
    """Test fixed-length segmentation with overlap, no padding."""
    signal, sr = sample_signal_long
    seg_len_sec = 1.0
    overlap_ratio = 0.5
    seg_len_samples = int(seg_len_sec * sr) # 1000
    hop_len_samples = int(seg_len_samples * (1.0 - overlap_ratio)) # 500

    segments = segment_fixed_length(signal, sr, segment_length_sec=seg_len_sec, overlap_ratio=overlap_ratio, pad=False)

    assert isinstance(segments, list)
    # Calculate expected number of segments without padding
    expected_num_segments = 0
    start = 0
    while start + seg_len_samples <= len(signal):
        expected_num_segments += 1
        start += hop_len_samples
    assert len(segments) == expected_num_segments

    for i, seg in enumerate(segments):
        assert len(seg) == seg_len_samples
        start_idx = i * hop_len_samples
        expected_content = signal[start_idx : start_idx + seg_len_samples]
        assert_array_equal(seg, expected_content)

def test_segment_fixed_length_with_padding(sample_signal_long):
    """Test fixed-length segmentation with padding enabled."""
    signal, sr = sample_signal_long
    seg_len_sec = 1.0
    overlap_ratio = 0.25
    seg_len_samples = int(seg_len_sec * sr) # 1000
    hop_len_samples = int(seg_len_samples * (1.0 - overlap_ratio)) # 750

    segments = segment_fixed_length(signal, sr, segment_length_sec=seg_len_sec, overlap_ratio=overlap_ratio, pad=True)

    assert isinstance(segments, list)
    # Calculate expected number of segments with padding
    # Last segment starts at last_start such that last_start < len(signal)
    # Total segments = floor((len(signal) - seg_len) / hop) + 1 (approx, depends on exact boundary)
    # Easier: check if last segment start is correct and length is padded
    expected_num_segments = 0
    start = 0
    while start < len(signal):
        expected_num_segments += 1
        start += hop_len_samples
    assert len(segments) == expected_num_segments

    # Check last segment
    last_segment = segments[-1]
    assert len(last_segment) == seg_len_samples # Should be padded to full length
    # Check content of the non-padded part of the last segment
    last_segment_start_idx = (len(segments) - 1) * hop_len_samples
    original_part_len = len(signal) - last_segment_start_idx
    assert original_part_len > 0 # Must have some original part
    assert_array_equal(last_segment[:original_part_len], signal[last_segment_start_idx:])
    # Check padded part is zeros
    assert_array_equal(last_segment[original_part_len:], np.zeros(seg_len_samples - original_part_len))

def test_segment_fixed_length_with_min_length(sample_signal_long):
    """Test fixed-length segmentation with padding and min_length filtering."""
    signal, sr = sample_signal_long
    seg_len_sec = 1.0
    overlap_ratio = 0.0
    min_len_sec = 0.4 # Minimum length to keep (400 samples)
    seg_len_samples = int(seg_len_sec * sr) # 1000
    hop_len_samples = seg_len_samples # No overlap

    segments = segment_fixed_length(
        signal, sr,
        segment_length_sec=seg_len_sec,
        overlap_ratio=overlap_ratio,
        pad=True, # Padding enabled
        min_segment_length_sec=min_len_sec
    )

    # Signal length 5300. Segments start at 0, 1000, 2000, 3000, 4000, 5000.
    # Segment 0-4 are full length (1000 samples).
    # Segment 5 starts at 5000, original length is 300 samples.
    # Since 300 < min_len (400), this last segment should be discarded even with padding.
    expected_num_segments = 5 # Only the full segments
    assert len(segments) == expected_num_segments
    for seg in segments:
        assert len(seg) == seg_len_samples # All kept segments are full length

def test_segment_fixed_length_short_signal(sample_signal_long):
    """Test fixed-length segmentation on a signal shorter than segment length."""
    signal_short, sr = sample_signal_long
    signal_short = signal_short[:500] # 0.5 seconds
    seg_len_sec = 1.0
    seg_len_samples = int(seg_len_sec * sr) # 1000

    # Test without padding
    segments_no_pad = segment_fixed_length(signal_short, sr, seg_len_sec, pad=False)
    assert len(segments_no_pad) == 0

    # Test with padding
    segments_pad = segment_fixed_length(signal_short, sr, seg_len_sec, pad=True)
    assert len(segments_pad) == 1
    assert len(segments_pad[0]) == seg_len_samples
    assert_array_equal(segments_pad[0][:len(signal_short)], signal_short)
    assert_array_equal(segments_pad[0][len(signal_short):], np.zeros(seg_len_samples - len(signal_short)))

    # Test with padding and min_length that discards the segment
    segments_pad_min = segment_fixed_length(signal_short, sr, seg_len_sec, pad=True, min_segment_length_sec=0.6)
    assert len(segments_pad_min) == 0

def test_segment_fixed_length_invalid_params(sample_signal_long):
    """Test fixed-length segmentation with invalid parameters."""
    signal, sr = sample_signal_long
    with pytest.raises(ValueError):
        segment_fixed_length(signal, sr, segment_length_sec=0) # Zero length
    with pytest.raises(ValueError):
        segment_fixed_length(signal, sr, segment_length_sec=1.0, overlap_ratio=1.0) # Invalid overlap
    with pytest.raises(ValueError):
        segment_fixed_length(signal, sr, segment_length_sec=1.0, overlap_ratio=-0.1) # Invalid overlap


# --- Placeholder Function Tests ---

def test_segment_by_silence_placeholder(sample_signal_long):
    """Test the placeholder for segment_by_silence."""
    signal, sr = sample_signal_long
    result = segment_by_silence(signal, sr)
    assert isinstance(result, list)
    assert len(result) == 0

def test_segment_by_event_placeholder(sample_signal_long):
    """Test the placeholder for segment_by_event."""
    signal, sr = sample_signal_long
    event_times = np.array([0.5, 1.5, 2.5])
    result = segment_by_event(signal, sr, event_times_sec=event_times)
    assert isinstance(result, list)
    assert len(result) == 0
