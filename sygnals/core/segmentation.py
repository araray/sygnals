# sygnals/core/segmentation.py

"""
Core logic for segmenting signals based on various criteria.

Provides functions to divide a signal into meaningful chunks (segments or frames)
using methods like fixed-length windows, silence detection, or event detection.
"""

import logging
import numpy as np
from numpy.typing import NDArray # Import NDArray
# Import necessary types
from typing import List, Tuple, Optional, Literal, Union, Dict, Any
import librosa # Used for framing and potentially event detection

logger = logging.getLogger(__name__)

# --- Segmentation Methods ---

def segment_fixed_length(
    y: NDArray[np.float64],
    sr: int,
    segment_length_sec: float,
    overlap_ratio: float = 0.0,
    pad: bool = True,
    min_segment_length_sec: Optional[float] = None
) -> List[NDArray[np.float64]]:
    """
    Segments the signal into fixed-length windows with a specified overlap.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate (Hz).
        segment_length_sec: Desired length of each segment in seconds.
        overlap_ratio: Overlap between consecutive segments as a ratio of
                       segment_length_sec (0.0 to < 1.0). Default: 0.0 (no overlap).
        pad: If True (default), pad the end of the signal with zeros to ensure
             the last segment is included, even if shorter than segment_length_sec
             (unless filtered by min_segment_length_sec). If False, discard the
             last partial segment.
        min_segment_length_sec: If set, discard segments shorter than this duration
                                (in seconds), checked *before* padding.

    Returns:
        List of NumPy arrays, where each array is a segment of the signal (float64).

    Raises:
        ValueError: If parameters are invalid (e.g., length <= 0, overlap >= 1).
    """
    if y.ndim != 1:
        raise ValueError("Input signal y must be 1D.")
    if segment_length_sec <= 0:
        raise ValueError("segment_length_sec must be positive.")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be between 0.0 and < 1.0.")

    segment_length_samples = int(segment_length_sec * sr)
    if segment_length_samples == 0:
         logger.warning(f"Segment length in samples is 0 for {segment_length_sec}s and sr={sr}. No segments generated.")
         return []

    hop_length_samples = int(segment_length_samples * (1.0 - overlap_ratio))
    # Ensure hop length is at least 1 sample
    hop_length_samples = max(1, hop_length_samples)

    min_samples = int(min_segment_length_sec * sr) if min_segment_length_sec is not None else 0

    logger.info(f"Segmenting signal (length {len(y)}) into fixed segments: "
                f"len={segment_length_sec}s ({segment_length_samples} samples), "
                f"overlap={overlap_ratio*100:.1f}% ({hop_length_samples} hop samples), pad={pad}, min_len={min_samples} samples")

    segments = []
    start_sample = 0
    total_samples = len(y)

    while start_sample < total_samples:
        end_sample = start_sample + segment_length_samples
        segment = y[start_sample:end_sample]
        original_segment_len = len(segment) # Store original length before potential padding

        # FIX: Check minimum length *before* padding
        if min_samples > 0 and original_segment_len < min_samples:
            logger.debug(f"Discarding segment starting at {start_sample} (original length {original_segment_len} < min {min_samples}).")
            segment = None # Mark for skipping
        elif end_sample > total_samples: # Handle last segment padding/truncation only if not discarded
            if pad:
                # Pad the last segment with zeros to match desired length
                padding_needed = segment_length_samples - original_segment_len
                segment = np.pad(segment, (0, padding_needed), mode='constant')
            else:
                # Discard the last partial segment if padding is disabled
                segment = None # Mark for skipping

        if segment is not None:
            # Ensure segment length matches expected length if padding occurred
            if pad and len(segment) != segment_length_samples:
                 # This might happen if the original segment was already discarded by min_len check
                 # but somehow segment is not None. Log a warning.
                 logger.warning(f"Segment length mismatch after potential padding/filtering. Expected {segment_length_samples}, got {len(segment)}. Skipping.")
            else:
                 segments.append(segment.astype(np.float64, copy=False)) # Ensure float64

        # Move to the next segment start position
        start_sample += hop_length_samples

        # Break if padding is disabled and we've processed the last possible full start
        if not pad and start_sample + segment_length_samples > total_samples:
             break

    logger.debug(f"Generated {len(segments)} fixed-length segments.")
    return segments


def segment_by_silence(
    y: NDArray[np.float64],
    sr: int,
    threshold_db: float = -40.0,
    min_silence_duration_sec: float = 0.1,
    min_segment_duration_sec: float = 0.2,
    padding_sec: float = 0.05
) -> List[Tuple[int, int]]:
    """
    Segments the signal based on periods of silence [Placeholder].

    Identifies non-silent regions separated by silence longer than a threshold.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate (Hz).
        threshold_db: Energy threshold (in dB relative to max) below which a frame
                      is considered silent. (Default: -40.0)
        min_silence_duration_sec: Minimum duration of silence (in seconds) required
                                 to split segments. (Default: 0.1)
        min_segment_duration_sec: Minimum duration of a non-silent segment (in seconds)
                                 to be kept. (Default: 0.2)
        padding_sec: Amount of padding (in seconds) to add to the beginning and end
                     of each detected non-silent segment. (Default: 0.05)

    Returns:
        List of tuples, where each tuple represents a segment as (start_sample, end_sample).
        [Placeholder - returns empty list]
    """
    logger.warning("segment_by_silence is a placeholder and returns empty list.")
    # TODO: Implement silence detection logic (e.g., using RMS energy per frame)
    # TODO: Identify contiguous non-silent regions based on thresholds and durations
    # TODO: Apply padding and minimum segment length filtering
    # TODO: Return list of (start_sample, end_sample) tuples
    return []


def segment_by_event(
    y: NDArray[np.float64],
    sr: int,
    event_times_sec: NDArray[np.float64], # Times of events (e.g., onsets)
    segment_duration_sec: Optional[float] = None, # If set, create fixed-duration segments around events
    pre_event_sec: float = 0.05, # Time before event to include
    post_event_sec: float = 0.2 # Time after event to include (if segment_duration_sec is None)
) -> List[Tuple[int, int]]:
    """
    Segments the signal based on specified event times [Placeholder].

    Creates segments centered around or starting at given event timestamps.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate (Hz).
        event_times_sec: NumPy array of event timestamps in seconds (e.g., from onset detection).
        segment_duration_sec: If provided, creates fixed-duration segments of this length,
                              centered around each event time (adjusting for boundaries).
                              If None, uses `pre_event_sec` and `post_event_sec`.
        pre_event_sec: Duration (in seconds) before each event time to include in the segment
                       (used if `segment_duration_sec` is None). (Default: 0.05)
        post_event_sec: Duration (in seconds) after each event time to include in the segment
                        (used if `segment_duration_sec` is None). (Default: 0.2)

    Returns:
        List of tuples, where each tuple represents a segment as (start_sample, end_sample).
        Segments are clipped to signal boundaries.
        [Placeholder - returns empty list]
    """
    logger.warning("segment_by_event is a placeholder and returns empty list.")
    # TODO: Convert event times to sample indices
    # TODO: Calculate start/end samples for each segment based on chosen method (fixed duration or pre/post)
    # TODO: Clip segment boundaries to signal length (0 to len(y))
    # TODO: Return list of (start_sample, end_sample) tuples
    return []
