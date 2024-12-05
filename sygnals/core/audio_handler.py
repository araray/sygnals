import librosa
import numpy as np
import soundfile as sf

# Load and save audio
def load_audio(file_path, sr=None):
    """Load an audio file using Librosa."""
    data, sample_rate = librosa.load(file_path, sr=sr)
    return data, sample_rate

def save_audio(data, sr, output_path):
    """Save audio data to a WAV file."""
    sf.write(output_path, data, sr)

# Audio metrics
def get_audio_metrics(data, sr):
    """Compute audio metrics such as RMS, peak amplitude, and duration."""
    rms = np.sqrt(np.mean(data**2))
    duration = len(data) / sr
    peak_amplitude = np.max(np.abs(data))
    return {
        "rms": rms,
        "duration (seconds)": duration,
        "peak_amplitude": peak_amplitude
    }

# Audio slicing
def slice_audio(data, sr, start_time, end_time):
    """Extract a portion of audio data between start_time and end_time."""
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    return data[start_sample:end_sample]

# Audio effects
def time_stretch(data, factor):
    """Stretch audio in time (does not affect pitch)."""
    return librosa.effects.time_stretch(data, factor)

def pitch_shift(data, sr, semitones):
    """Shift the pitch of audio."""
    return librosa.effects.pitch_shift(data, sr, n_steps=semitones)

def dynamic_range_compression(data, threshold=0.1):
    """Apply simple dynamic range compression to normalize the audio."""
    max_amplitude = np.max(np.abs(data))
    if max_amplitude > threshold:
        return data / max_amplitude * threshold
    return data
