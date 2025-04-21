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
import warnings # Import warnings module

logger = logging.getLogger(__name__)

# Define a small epsilon for safe division and log calculations
_EPSILON = np.finfo(np.float64).eps

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
        segment_slice = y[start_sample:end_sample]
        original_segment_len = len(segment_slice) # Store original length before potential padding
        segment_to_add = None # Initialize segment to add

        # Check minimum length *before* padding
        if min_samples > 0 and original_segment_len < min_samples:
            logger.debug(f"Discarding segment starting at {start_sample} (original length {original_segment_len} < min {min_samples}).")
        elif end_sample > total_samples: # Handle last segment padding/truncation only if not discarded
            if pad:
                # Pad the last segment with zeros to match desired length
                padding_needed = segment_length_samples - original_segment_len
                segment_to_add = np.pad(segment_slice, (0, padding_needed), mode='constant')
            else:
                # Discard the last partial segment if padding is disabled
                pass # segment_to_add remains None
        else:
             # Full segment, no padding needed
             segment_to_add = segment_slice

        if segment_to_add is not None:
            # Ensure segment length matches expected length if padding occurred
            if pad and len(segment_to_add) != segment_length_samples:
                 logger.warning(f"Segment length mismatch after potential padding/filtering. Expected {segment_length_samples}, got {len(segment_to_add)}. Skipping.")
            else:
                 segments.append(segment_to_add.astype(np.float64, copy=False)) # Ensure float64

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
    padding_sec: float = 0.05,
    # FIX: Changed default frame_length from 1024 to 512 for potentially better resolution
    frame_length: int = 512, # Frame length for RMS calculation
    hop_length: Optional[int] = None # Hop length for RMS calculation
) -> List[Tuple[int, int]]:
    """
    Segments the signal based on periods of silence.

    Identifies non-silent regions separated by silence longer than a threshold.
    Uses frame-based RMS energy to detect silence.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate (Hz).
        threshold_db: Energy threshold (in dB relative to max RMS energy) below which
                      a frame is considered silent. (Default: -40.0)
        min_silence_duration_sec: Minimum duration of consecutive silence (in seconds)
                                 required to split segments. (Default: 0.1)
        min_segment_duration_sec: Minimum duration of a non-silent segment (in seconds)
                                 to be kept. (Default: 0.2)
        padding_sec: Amount of padding (in seconds) to add to the beginning and end
                     of each detected non-silent segment. (Default: 0.05)
        frame_length: Frame length (samples) for RMS energy calculation. (Default: 512)
        hop_length: Hop length (samples) for RMS energy calculation. Defaults to frame_length // 4.

    Returns:
        List of tuples, where each tuple represents a non-silent segment as (start_sample, end_sample).
        Indices are inclusive of start, exclusive of end.

    Raises:
        ValueError: If input `y` is not 1D or parameters are invalid.
    """
    if y.ndim != 1:
        raise ValueError("Input signal y must be 1D.")
    if threshold_db > 0:
        raise ValueError("threshold_db must be non-positive (e.g., -40.0).")
    if min_silence_duration_sec < 0 or min_segment_duration_sec < 0 or padding_sec < 0:
        raise ValueError("Durations and padding must be non-negative.")

    hop_length_calc = hop_length if hop_length is not None else frame_length // 4
    if hop_length_calc <= 0:
         raise ValueError("Hop length must be positive.")

    logger.info(f"Segmenting by silence: threshold={threshold_db}dB, min_silence={min_silence_duration_sec}s, "
                f"min_segment={min_segment_duration_sec}s, padding={padding_sec}s, frame_len={frame_length}")

    # 1. Calculate RMS energy per frame
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length_calc)[0]
    if rms_frames.size == 0:
        logger.warning("Could not calculate RMS frames (signal likely too short). Returning no segments.")
        return []

    # 2. Determine threshold in linear amplitude
    max_rms = np.max(rms_frames)
    if max_rms < _EPSILON:
        logger.warning("Signal maximum RMS is near zero. Assuming entire signal is silent.")
        return [] # No non-silent segments if max RMS is zero
    threshold_linear = max_rms * (10.0**(threshold_db / 20.0))

    # 3. Identify silent frames
    silent_frames_mask = rms_frames < threshold_linear

    # 4. Find contiguous blocks of silence longer than min_silence_duration
    min_silence_frames = int(np.ceil(min_silence_duration_sec * sr / hop_length_calc))
    if min_silence_frames <= 0: min_silence_frames = 1 # Need at least 1 frame

    silent_blocks = []
    start_idx = -1
    for i, is_silent in enumerate(silent_frames_mask):
        if is_silent and start_idx == -1:
            start_idx = i
        elif not is_silent and start_idx != -1:
            if (i - start_idx) >= min_silence_frames:
                silent_blocks.append((start_idx, i)) # Store frame indices (exclusive end)
            start_idx = -1 # Reset start index
    # Check for trailing silence block
    if start_idx != -1 and (len(silent_frames_mask) - start_idx) >= min_silence_frames:
        silent_blocks.append((start_idx, len(silent_frames_mask)))

    # 5. Determine non-silent segments (regions between long silences)
    non_silent_segments = []
    last_silence_end_frame = 0
    for silence_start, silence_end in silent_blocks:
        segment_start_frame = last_silence_end_frame
        segment_end_frame = silence_start
        if segment_end_frame > segment_start_frame:
            non_silent_segments.append((segment_start_frame, segment_end_frame))
        last_silence_end_frame = silence_end

    # Add the final segment after the last long silence
    if last_silence_end_frame < len(silent_frames_mask):
        non_silent_segments.append((last_silence_end_frame, len(silent_frames_mask)))

    # 6. Convert frame indices to sample indices and apply padding/min duration
    segments_samples = []
    min_segment_samples = int(min_segment_duration_sec * sr)
    padding_samples = int(padding_sec * sr)
    total_samples = len(y)

    for start_frame, end_frame in non_silent_segments:
        # Convert frame indices to sample indices (use frame start)
        start_sample = librosa.frames_to_samples(start_frame, hop_length=hop_length_calc)
        # Convert end frame index to sample index (end of the last frame in the segment)
        # Use end_frame directly as it's exclusive
        end_sample = librosa.frames_to_samples(end_frame, hop_length=hop_length_calc)

        # Apply padding
        start_sample_padded = max(0, start_sample - padding_samples)
        end_sample_padded = min(total_samples, end_sample + padding_samples)

        # Check minimum segment duration *after* padding
        if (end_sample_padded - start_sample_padded) >= min_segment_samples:
            segments_samples.append((start_sample_padded, end_sample_padded))
        else:
            logger.debug(f"Discarding segment ({start_sample_padded}, {end_sample_padded}) due to min_segment_duration.")


    # 7. Merge overlapping segments (can happen due to padding)
    if not segments_samples:
        return []

    merged_segments = []
    # Sort segments just in case they are out of order (shouldn't happen with current logic)
    segments_samples.sort(key=lambda x: x[0])
    current_start, current_end = segments_samples[0]

    for next_start, next_end in segments_samples[1:]:
        if next_start < current_end: # Overlap detected
            current_end = max(current_end, next_end) # Merge by extending the end
        else: # No overlap, start new segment
            merged_segments.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged_segments.append((current_start, current_end)) # Add the last segment

    logger.debug(f"Generated {len(merged_segments)} segments by silence.")
    return merged_segments


def segment_by_event(
    y: NDArray[np.float64],
    sr: int,
    event_times_sec: NDArray[np.float64], # Times of events (e.g., onsets)
    segment_duration_sec: Optional[float] = None, # If set, create fixed-duration segments around events
    pre_event_sec: float = 0.05, # Time before event to include
    post_event_sec: float = 0.2 # Time after event to include (if segment_duration_sec is None)
) -> List[Tuple[int, int]]:
    """
    Segments the signal based on specified event times.

    Creates segments centered around or defined relative to given event timestamps.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate (Hz).
        event_times_sec: NumPy array of event timestamps in seconds (e.g., from onset detection).
        segment_duration_sec: If provided (> 0), creates fixed-duration segments of this length,
                              centered around each event time (adjusting for boundaries).
                              If None or 0, uses `pre_event_sec` and `post_event_sec`.
        pre_event_sec: Duration (in seconds) before each event time to include in the segment
                       (used if `segment_duration_sec` is None/0). (Default: 0.05)
        post_event_sec: Duration (in seconds) after each event time to include in the segment
                        (used if `segment_duration_sec` is None/0). (Default: 0.2)

    Returns:
        List of tuples, where each tuple represents a segment as (start_sample, end_sample).
        Segments are clipped to signal boundaries [0, len(y)). Indices are inclusive of start,
        exclusive of end. Returns empty list if no events provided.

    Raises:
        ValueError: If input `y` is not 1D or parameters are invalid.
    """
    if y.ndim != 1:
        raise ValueError("Input signal y must be 1D.")
    if pre_event_sec < 0 or post_event_sec < 0:
        raise ValueError("pre_event_sec and post_event_sec must be non-negative.")
    if segment_duration_sec is not None and segment_duration_sec <= 0:
        raise ValueError("segment_duration_sec must be positive if specified.")

    if event_times_sec is None or len(event_times_sec) == 0:
        logger.warning("No event times provided for segmentation.")
        return []

    logger.info(f"Segmenting by {len(event_times_sec)} events. Method: {'fixed duration' if segment_duration_sec else 'pre/post padding'}.")

    # Convert event times to sample indices
    event_samples = (event_times_sec * sr).astype(int)
    total_samples = len(y)
    segments_samples = []

    for event_sample in event_samples:
        start_sample: int
        end_sample: int

        if segment_duration_sec is not None and segment_duration_sec > 0:
            # Fixed duration centered around event
            duration_samples = int(segment_duration_sec * sr)
            half_duration = duration_samples // 2
            start_sample = event_sample - half_duration
            end_sample = start_sample + duration_samples # Exclusive end
        else:
            # Use pre/post padding
            pre_samples = int(pre_event_sec * sr)
            post_samples = int(post_event_sec * sr)
            start_sample = event_sample - pre_samples
            end_sample = event_sample + post_samples # Exclusive end

        # Clip segment boundaries to signal limits [0, total_samples)
        start_sample_clipped = max(0, start_sample)
        end_sample_clipped = min(total_samples, end_sample)

        # Ensure start is less than end after clipping (avoid empty segments)
        if start_sample_clipped < end_sample_clipped:
            segments_samples.append((start_sample_clipped, end_sample_clipped))
        else:
            logger.debug(f"Skipping segment for event at sample {event_sample} as clipped boundaries are invalid "
                         f"({start_sample_clipped}, {end_sample_clipped}).")

    # Optional: Merge overlapping segments if desired (not done by default for event-based)
    # ... merge logic similar to segment_by_silence if needed ...

    logger.debug(f"Generated {len(segments_samples)} segments by event.")
    return segments_samples
