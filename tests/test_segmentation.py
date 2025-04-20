# tests/test_segmentation.py

"""
Tests for signal segmentation functions in sygnals.core.segmentation.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from typing import Optional, Union, Literal, Dict, Any, Tuple, List # Added List import

# Import segmentation functions to test
from sygnals.core.segmentation import (
    segment_fixed_length,
    segment_by_silence, # Implemented
    segment_by_event   # Implemented
)

# --- Test Fixtures ---

@pytest.fixture
def sample_signal_long() -> Tuple[np.ndarray, int]:
    """Generate a moderately long signal for segmentation tests."""
    sr = 1000 # Use easy sample rate
    duration = 5.3 # seconds (not an exact multiple of segment lengths)
    signal = np.arange(int(sr * duration)).astype(np.float64) / sr # Simple ramp
    return signal, sr

@pytest.fixture
def sample_signal_with_silence() -> Tuple[np.ndarray, int, List[Tuple[float, float]]]:
    """Generate a signal with silent gaps."""
    sr = 1000
    segments_info = [
        (0.5, 0.8), # Signal 1 (duration 0.3s) at time 0.5s
        (1.5, 2.2), # Signal 2 (duration 0.7s) at time 1.5s
        (3.0, 3.1)  # Signal 3 (duration 0.1s) at time 3.0s
    ]
    total_duration = 4.0
    signal = np.zeros(int(sr * total_duration), dtype=np.float64)
    # Add some noise in non-silent parts
    rng = np.random.default_rng(55)
    expected_segments_sec = []
    for start_sec, end_sec in segments_info:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        signal[start_sample:end_sample] = rng.normal(0, 0.3, end_sample - start_sample)
        expected_segments_sec.append((start_sec, end_sec))

    return signal, sr, expected_segments_sec


# --- Test segment_fixed_length (Existing Tests - Keep) ---

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
    expected_num_segments = 0
    start = 0
    while start < len(signal):
        expected_num_segments += 1
        start += hop_len_samples
    assert len(segments) == expected_num_segments

    # Check last segment
    last_segment = segments[-1]
    assert len(last_segment) == seg_len_samples # Should be padded to full length
    last_segment_start_idx = (len(segments) - 1) * hop_len_samples
    original_part_len = len(signal) - last_segment_start_idx
    assert original_part_len > 0 # Must have some original part
    assert_array_equal(last_segment[:original_part_len], signal[last_segment_start_idx:])
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
    expected_num_segments = 5 # Last segment (300 samples) should be discarded
    assert len(segments) == expected_num_segments
    for seg in segments:
        assert len(seg) == seg_len_samples

def test_segment_fixed_length_short_signal(sample_signal_long):
    """Test fixed-length segmentation on a signal shorter than segment length."""
    signal_short, sr = sample_signal_long
    signal_short = signal_short[:500] # 0.5 seconds
    seg_len_sec = 1.0
    seg_len_samples = int(seg_len_sec * sr) # 1000

    segments_no_pad = segment_fixed_length(signal_short, sr, seg_len_sec, pad=False)
    assert len(segments_no_pad) == 0
    segments_pad = segment_fixed_length(signal_short, sr, seg_len_sec, pad=True)
    assert len(segments_pad) == 1
    assert len(segments_pad[0]) == seg_len_samples
    assert_array_equal(segments_pad[0][:len(signal_short)], signal_short)
    assert_array_equal(segments_pad[0][len(signal_short):], np.zeros(seg_len_samples - len(signal_short)))
    segments_pad_min = segment_fixed_length(signal_short, sr, seg_len_sec, pad=True, min_segment_length_sec=0.6)
    assert len(segments_pad_min) == 0

def test_segment_fixed_length_invalid_params(sample_signal_long):
    """Test fixed-length segmentation with invalid parameters."""
    signal, sr = sample_signal_long
    with pytest.raises(ValueError): segment_fixed_length(signal, sr, segment_length_sec=0)
    with pytest.raises(ValueError): segment_fixed_length(signal, sr, segment_length_sec=1.0, overlap_ratio=1.0)
    with pytest.raises(ValueError): segment_fixed_length(signal, sr, segment_length_sec=1.0, overlap_ratio=-0.1)


# --- Test segment_by_silence ---

def test_segment_by_silence_basic(sample_signal_with_silence):
    """Test basic segmentation by silence."""
    signal, sr, expected_segments_sec = sample_signal_with_silence
    padding_sec = 0.05
    padding_samples = int(padding_sec * sr)

    segments = segment_by_silence(
        signal, sr,
        threshold_db=-30, # Lower threshold to clearly separate noise from silence
        min_silence_duration_sec=0.2, # Silence between seg 1&2 (0.7s) and 2&3 (0.8s) are long enough
        min_segment_duration_sec=0.05, # Keep all segments
        padding_sec=padding_sec
    )

    assert isinstance(segments, list)
    assert len(segments) == len(expected_segments_sec) # Should find 3 segments

    # Check segment boundaries (approximate due to framing and padding)
    for i, (start_s, end_s) in enumerate(segments):
        expected_start_sec, expected_end_sec = expected_segments_sec[i]
        # Check if segment overlaps significantly with expected time range, considering padding
        assert start_s / sr < expected_start_sec + padding_sec # Start time check (allow for padding)
        assert end_s / sr > expected_end_sec - padding_sec   # End time check (allow for padding)
        # Check segment duration is reasonable
        assert (end_s - start_s) / sr > (expected_end_sec - expected_start_sec) - 0.1 # Allow slight deviation

def test_segment_by_silence_min_duration(sample_signal_with_silence):
    """Test filtering segments by min_segment_duration_sec."""
    signal, sr, _ = sample_signal_with_silence
    # Segment 3 has duration 0.1s

    segments = segment_by_silence(
        signal, sr,
        threshold_db=-30,
        min_silence_duration_sec=0.2,
        min_segment_duration_sec=0.2, # Set min duration > 0.1s
        padding_sec=0.0 # No padding for simplicity
    )

    assert len(segments) == 2 # Segment 3 should be discarded

def test_segment_by_silence_min_silence(sample_signal_with_silence):
    """Test effect of min_silence_duration_sec."""
    signal, sr, _ = sample_signal_with_silence
    # Gap between 1 and 2 is 0.7s, gap between 2 and 3 is 0.8s

    # Require very long silence -> should merge segments
    segments = segment_by_silence(
        signal, sr,
        threshold_db=-30,
        min_silence_duration_sec=1.0, # Longer than any gap
        min_segment_duration_sec=0.05,
        padding_sec=0.0
    )

    assert len(segments) == 1 # All segments should be merged

def test_segment_by_silence_all_silent():
    """Test segmentation on a completely silent signal."""
    sr = 1000
    signal = np.zeros(2 * sr, dtype=np.float64)
    segments = segment_by_silence(signal, sr)
    assert len(segments) == 0

def test_segment_by_silence_no_silence():
    """Test segmentation on a signal with no significant silence."""
    sr = 1000
    signal = np.random.rand(3 * sr).astype(np.float64) - 0.5
    segments = segment_by_silence(signal, sr, min_silence_duration_sec=0.1)
    assert len(segments) == 1 # Should return one segment covering the whole signal
    assert segments[0] == (0, len(signal)) # Check boundaries (allowing for padding effects if any)


# --- Test segment_by_event ---

def test_segment_by_event_fixed_duration(sample_signal_long):
    """Test segmentation by event with fixed duration."""
    signal, sr = sample_signal_long
    event_times = np.array([0.5, 1.8, 3.2, 5.0]) # Events within the 5.3s signal
    seg_dur_sec = 0.4
    seg_dur_samples = int(seg_dur_sec * sr)
    half_dur_samples = seg_dur_samples // 2

    segments = segment_by_event(signal, sr, event_times, segment_duration_sec=seg_dur_sec)

    assert isinstance(segments, list)
    assert len(segments) == len(event_times)

    for i, (start_s, end_s) in enumerate(segments):
        event_sample = int(event_times[i] * sr)
        expected_start = event_sample - half_dur_samples
        expected_end = expected_start + seg_dur_samples
        # Check boundaries after clipping
        assert start_s == max(0, expected_start)
        assert end_s == min(len(signal), expected_end)
        # Check duration (might be shorter at boundaries)
        assert (end_s - start_s) <= seg_dur_samples

def test_segment_by_event_pre_post(sample_signal_long):
    """Test segmentation by event with pre/post padding."""
    signal, sr = sample_signal_long
    event_times = np.array([0.2, 2.5, 5.1]) # Include events near boundaries
    pre_sec = 0.1
    post_sec = 0.3
    pre_samples = int(pre_sec * sr)
    post_samples = int(post_sec * sr)
    expected_duration_samples = pre_samples + post_samples

    segments = segment_by_event(signal, sr, event_times, pre_event_sec=pre_sec, post_event_sec=post_sec)

    assert isinstance(segments, list)
    assert len(segments) == len(event_times)

    for i, (start_s, end_s) in enumerate(segments):
        event_sample = int(event_times[i] * sr)
        expected_start = event_sample - pre_samples
        expected_end = event_sample + post_samples
        # Check boundaries after clipping
        assert start_s == max(0, expected_start)
        assert end_s == min(len(signal), expected_end)
        # Check duration (might be shorter at boundaries)
        assert (end_s - start_s) <= expected_duration_samples

def test_segment_by_event_no_events(sample_signal_long):
    """Test segmentation by event with no events provided."""
    signal, sr = sample_signal_long
    event_times = np.array([])
    segments = segment_by_event(signal, sr, event_times)
    assert len(segments) == 0

def test_segment_by_event_invalid_params(sample_signal_long):
    """Test segmentation by event with invalid parameters."""
    signal, sr = sample_signal_long
    event_times = np.array([1.0])
    with pytest.raises(ValueError):
        segment_by_event(signal, sr, event_times, segment_duration_sec=-0.5)
    with pytest.raises(ValueError):
        segment_by_event(signal, sr, event_times, pre_event_sec=-0.1)
    with pytest.raises(ValueError):
        segment_by_event(signal, sr, event_times, post_event_sec=-0.1)
