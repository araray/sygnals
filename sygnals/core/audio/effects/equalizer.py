# sygnals/core/audio/effects/equalizer.py

"""
Implementation of audio equalizer (EQ) effects using scipy filters.
Includes basic Graphic EQ and Parametric EQ functionality.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from scipy.signal import iirdesign, sosfiltfilt, freqs_zpk # For filter design and analysis
# Import necessary types
from typing import List, Tuple, Dict, Optional, Union, Literal, Any

logger = logging.getLogger(__name__)

# --- Helper Function for Filter Design ---

def _design_eq_filter(
    filter_type: Literal['peak', 'low_shelf', 'high_shelf', 'notch'],
    fs: float,
    freq: float,
    gain_db: float = 0.0,
    q: float = 1.0, # Quality factor, typically sqrt(0.5) for -3dB bandwidth
    order: int = 2 # Order for shelf/notch, peak is usually 2nd order implicitly
) -> Optional[NDArray[np.float64]]:
    """
    Designs a single EQ filter stage (peak, shelf, notch) using iirdesign.

    Args:
        filter_type: Type of filter ('peak', 'low_shelf', 'high_shelf', 'notch').
        fs: Sampling frequency (Hz).
        freq: Center frequency (peak/notch) or cutoff frequency (shelf) in Hz.
        gain_db: Gain in dB for peak/shelf filters. Ignored for notch.
        q: Quality factor (controls bandwidth for peak/notch). Higher Q = narrower band.
        order: Filter order (mainly for shelf/notch).

    Returns:
        Filter coefficients in Second-Order Sections (SOS) format (ndarray),
        or None if design fails or gain is negligible.

    Raises:
        ValueError: If parameters are invalid (e.g., freq >= fs/2).
    """
    nyquist = fs / 2.0
    if not 0 < freq < nyquist:
        raise ValueError(f"EQ frequency ({freq} Hz) must be between 0 and Nyquist ({nyquist} Hz).")
    if filter_type in ['peak', 'notch'] and q <= 0:
        raise ValueError("Q factor must be positive for peak/notch filters.")

    # --- Convert gain_db to linear amplitude gain ---
    # gain_lin = 10.0**(gain_db / 20.0)
    # Note: iirdesign often works with passband/stopband ripple/attenuation in dB.
    # For peaking/shelving EQs, designing directly can be complex.
    # Using approximations or cookbook formulas might be simpler for basic EQs.
    # However, let's try a simplified approach with iirdesign for peaking/shelving.
    # We define passband/stopband edges very close to the center frequency and use gain.

    # --- Filter Design using iirdesign (can be complex for standard EQ types) ---
    # This approach might not perfectly match standard audio EQ designs without careful parameter mapping.
    # Consider using dedicated audio EQ libraries or cookbook formulas for more precise results.

    # --- Simplified Peaking EQ Design (using iirdesign as bandpass/bandstop) ---
    # This is an approximation. Passband gain controls the peak height.
    if filter_type == 'peak':
        if abs(gain_db) < 0.1: return None # No significant gain change

        # Approximate bandwidth based on Q factor
        bw = freq / q
        f_low = freq - bw / 2.0
        f_high = freq + bw / 2.0

        # Ensure frequencies are within valid range
        f_low = max(1e-3 * nyquist, f_low) # Avoid 0 Hz
        f_high = min(0.999 * nyquist, f_high) # Avoid Nyquist

        if f_low >= f_high:
             logger.warning(f"Calculated bandwidth is too narrow or invalid for peak EQ at {freq} Hz, Q={q}. Skipping filter.")
             return None

        # Define passband and stopband edges (very narrow stopband for peak)
        wp = [f_low / nyquist, f_high / nyquist]
        # Stopband edges slightly outside passband
        ws = [(f_low * 0.9) / nyquist, (f_high * 1.1) / nyquist]
        ws[0] = max(1e-6, ws[0]) # Ensure ws[0] > 0
        ws[1] = min(0.999999, ws[1]) # Ensure ws[1] < 1

        # Define gain/attenuation (gpass near 0dB, gstop defines attenuation/boost)
        # This mapping is approximate for peaking EQ via iirdesign
        gpass = 0.1 # Minimal ripple/attenuation in passband (dB)
        gstop = abs(gain_db) # Attenuation/gain in stopband (relative to passband)

        try:
            # Design as bandpass if gain > 0, bandstop if gain < 0 (approximation)
            filter_btype = 'bandpass' if gain_db > 0 else 'bandstop'
            # Order needs to be sufficient, maybe 4 for reasonable peak/dip
            sos = iirdesign(wp, ws, gpass, gstop, analog=False, ftype='butter', output='sos', order=4)
            # Note: Gain scaling might need adjustment based on actual response
            # This is a simplified approach. Cookbooks are better for precise peaking EQ.
            # If gain_db < 0, the filter is designed as bandstop, effectively creating a dip.
            # If gain_db > 0, it's designed as bandpass. We might need to add this filtered
            # signal back to the original scaled by the gain, which is complex.
            # --> Sticking to placeholder/warning for peak for now due to iirdesign complexity for this.
            logger.warning(f"Peaking EQ design using iirdesign is experimental. Consider dedicated EQ implementations.")
            # For now, return None to indicate peak filter not fully implemented this way
            return None # Temporarily disable peak filter via iirdesign

        except Exception as e:
            logger.error(f"Failed to design peaking EQ filter stage at {freq} Hz: {e}")
            return None

    # --- Shelf Filter Design ---
    elif filter_type in ['low_shelf', 'high_shelf']:
        if abs(gain_db) < 0.1: return None # No significant gain change
        # Design parameters for shelf filters
        gpass = 1.0 # Passband ripple (dB) - keep low
        gstop = abs(gain_db) # Stopband attenuation/gain (dB)
        wp = freq / nyquist # Passband edge frequency (normalized)
        # Stopband edge frequency - further away from cutoff
        ws = (freq * 0.5 / nyquist) if filter_type == 'low_shelf' else (freq * 1.5 / nyquist)
        ws = max(1e-6, min(0.999999, ws)) # Clamp ws

        # Determine filter type for iirdesign based on shelf type and gain
        if filter_type == 'low_shelf':
            btype = 'lowpass' if gain_db > 0 else 'highpass' # Boost = lowpass, Cut = highpass
            # Adjust wp/ws if cutting (passband becomes the attenuated part)
            if gain_db < 0: wp, ws = ws, wp; gpass, gstop = gstop, gpass
        else: # high_shelf
            btype = 'highpass' if gain_db > 0 else 'lowpass' # Boost = highpass, Cut = lowpass
            if gain_db < 0: wp, ws = ws, wp; gpass, gstop = gstop, gpass

        try:
            sos = iirdesign(wp, ws, gpass, gstop, analog=False, ftype='butter', output='sos', order=order)
            # Note: Gain might need adjustment. iirdesign aims for attenuation/ripple specs.
            # For a shelf boost, the gain might not reach the target dB accurately without scaling.
            logger.warning("Shelf EQ design using iirdesign provides approximate gain. Consider dedicated EQ implementations.")
            return sos.astype(np.float64, copy=False) # Return designed SOS
        except Exception as e:
            logger.error(f"Failed to design {filter_type} filter stage at {freq} Hz: {e}")
            return None

    # --- Notch Filter Design ---
    elif filter_type == 'notch':
         # Design as a narrow bandstop filter
         bw = freq / q # Bandwidth
         f_low = max(1e-3 * nyquist, freq - bw / 2.0)
         f_high = min(0.999 * nyquist, freq + bw / 2.0)
         if f_low >= f_high:
             logger.warning(f"Calculated bandwidth is too narrow for notch filter at {freq} Hz, Q={q}. Skipping.")
             return None

         wp = [(f_low * 0.9) / nyquist, (f_high * 1.1) / nyquist] # Passband edges
         ws = [f_low / nyquist, f_high / nyquist] # Stopband edges (notch center)
         wp[0] = max(1e-6, wp[0])
         wp[1] = min(0.999999, wp[1])

         gpass = 1.0 # Low passband ripple
         gstop = 30.0 # Significant attenuation in the notch (e.g., 30 dB)

         try:
             sos = iirdesign(wp, ws, gpass, gstop, analog=False, ftype='butter', output='sos', order=order)
             return sos.astype(np.float64, copy=False)
         except Exception as e:
             logger.error(f"Failed to design notch filter stage at {freq} Hz: {e}")
             return None
    else:
        logger.error(f"Unknown filter type requested: {filter_type}")
        return None


# --- Main EQ Functions ---

def apply_graphic_eq(
    y: NDArray[np.float64],
    sr: int,
    band_gains: List[Tuple[float, float]], # List of (center_frequency_hz, gain_db)
    q_factor: float = 1.414 # Default Q for moderate bandwidth (approx sqrt(2))
) -> NDArray[np.float64]:
    """
    Applies a basic graphic equalizer effect using peaking filters.

    NOTE: Peaking filter design via iirdesign is complex and experimental here.
          This function currently logs warnings and may not apply peaking filters correctly.
          Consider using dedicated audio EQ libraries for reliable results.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate of `y`.
        band_gains: A list of tuples, where each tuple represents a frequency band
                    and its desired gain. Each tuple is (center_frequency_hz, gain_db).
                    Example: [(100, -6.0), (1000, 3.0), (5000, -2.0)]
        q_factor: The quality factor (Q) for the peaking filters, controlling bandwidth.
                  Higher Q means narrower bands. (Default: ~1.414).

    Returns:
        The processed audio time series (float64). May be identical to input if
        peaking filter design fails or gains are negligible.

    Raises:
        ValueError: If input `y` is not 1D or parameters are invalid.
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    if not isinstance(band_gains, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in band_gains):
         raise ValueError("band_gains must be a list of (center_frequency_hz, gain_db) tuples.")
    if q_factor <= 0:
        raise ValueError("q_factor must be positive.")

    logger.info(f"Applying Graphic EQ: {len(band_gains)} bands, Q={q_factor:.2f}")
    logger.warning("Graphic EQ (peaking filters) implementation using iirdesign is experimental and may not function correctly.")

    processed_y = y.copy().astype(np.float64) # Start with a copy

    for freq, gain_db in band_gains:
        if abs(gain_db) < 0.1: # Skip negligible gains
            continue
        logger.debug(f"Designing graphic EQ stage: Freq={freq} Hz, Gain={gain_db} dB")
        try:
            # Attempt to design a peaking filter stage
            sos = _design_eq_filter(filter_type='peak', fs=sr, freq=freq, gain_db=gain_db, q=q_factor)

            if sos is not None:
                # Apply the filter stage sequentially
                # WARNING: Applying peaking filters designed this way sequentially might not
                # produce the desired combined frequency response. Additive synthesis or
                # parallel filter banks are often used for graphic EQs.
                # This implementation will likely just apply the last designed filter if any succeed.
                # --> Returning input signal until peak filter design is robust.
                logger.warning(f"Skipping application of experimental peak filter at {freq} Hz.")
                # processed_y = sosfiltfilt(sos, processed_y)
            else:
                 logger.warning(f"Skipping EQ stage for Freq={freq} Hz, Gain={gain_db} dB (design failed or gain negligible).")

        except ValueError as e:
            logger.warning(f"Skipping EQ stage for Freq={freq} Hz due to invalid parameters: {e}")
        except Exception as e:
            logger.error(f"Error processing EQ stage for Freq={freq} Hz: {e}")

    # Return the signal (potentially unmodified if peaking filters were skipped)
    return processed_y # Return the potentially modified signal

def apply_parametric_eq(
    y: NDArray[np.float64],
    sr: int,
    params: List[Dict[str, Any]] # List of filter parameter dictionaries
) -> NDArray[np.float64]:
    """
    Applies a parametric equalizer effect using multiple filter stages.

    Supports 'low_shelf', 'high_shelf', and 'notch' filters designed using
    `scipy.signal.iirdesign`. Peaking filters are currently experimental/skipped.

    Args:
        y: Input audio time series (1D float64).
        sr: Sampling rate of `y`.
        params: A list of dictionaries, where each dictionary defines one filter stage.
                Required keys per dictionary:
                 - 'type': str ('low_shelf', 'high_shelf', 'notch', 'peak' [experimental])
                 - 'freq': float (Center/cutoff frequency in Hz)
                Optional keys:
                 - 'gain_db': float (Gain in dB for shelf/peak, default: 0.0)
                 - 'q': float (Quality factor for notch/peak, default: 1.0)
                 - 'order': int (Filter order for shelf/notch, default: 2)

    Returns:
        The processed audio time series (float64).

    Raises:
        ValueError: If input `y` is not 1D or parameters are invalid/missing.
    """
    if y.ndim != 1:
        raise ValueError("Input audio data must be a 1D array.")
    if not isinstance(params, list) or not all(isinstance(item, dict) for item in params):
         raise ValueError("params must be a list of dictionaries defining filter stages.")

    logger.info(f"Applying Parametric EQ: {len(params)} stages.")

    processed_y = y.copy().astype(np.float64) # Start with a copy

    for i, stage_params in enumerate(params):
        # Validate required parameters for the stage
        if 'type' not in stage_params or 'freq' not in stage_params:
            raise ValueError(f"EQ stage {i} missing required 'type' or 'freq' parameter.")

        filter_type = stage_params['type'].lower()
        freq = stage_params['freq']
        gain_db = stage_params.get('gain_db', 0.0) # Default gain 0 dB
        q = stage_params.get('q', 1.0)             # Default Q 1.0
        order = stage_params.get('order', 2)       # Default order 2

        logger.debug(f"Designing parametric EQ stage {i}: type={filter_type}, freq={freq}, gain={gain_db}, q={q}, order={order}")

        if filter_type == 'peak':
             logger.warning("Peaking filters in parametric EQ are experimental and currently skipped.")
             continue # Skip peak filters for now

        try:
            # Design the filter stage
            sos = _design_eq_filter(filter_type=filter_type, fs=sr, freq=freq, gain_db=gain_db, q=q, order=order)

            if sos is not None:
                # Apply the filter stage sequentially using zero-phase filtering
                processed_y = sosfiltfilt(sos, processed_y)
                logger.debug(f"Applied {filter_type} filter stage {i}.")
            else:
                 logger.debug(f"Skipping parametric EQ stage {i} (design failed or gain negligible).")

        except ValueError as e:
            logger.warning(f"Skipping parametric EQ stage {i} due to invalid parameters: {e}")
        except Exception as e:
            logger.error(f"Error processing parametric EQ stage {i}: {e}")

    return processed_y # Return the modified signal
