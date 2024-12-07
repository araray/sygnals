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

def save_audio_as_csv(dataframe, output_path):
    """Save audio data to a CSV file."""
    dataframe.to_csv(output_path, index=False)

def save_audio_as_json(dataframe, output_path):
    """Save audio data to a JSON file."""
    dataframe.to_json(output_path, orient="records", indent=2)

# Audio metrics
def get_audio_metrics(data, sr):
    """Calculate audio metrics."""
    rms = np.sqrt(np.mean(data**2))
    peak_amplitude = np.max(np.abs(data))
    duration = len(data) / sr
    # Convert numpy types to Python types for JSON compatibility
    return {
        "rms": float(rms),
        "peak_amplitude": float(peak_amplitude),
        "duration (seconds)": float(duration)
    }

# Audio slicing
def slice_audio(data, sr, start_time, end_time):
    """Extract a portion of audio data between start_time and end_time."""
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    return data[start_sample:end_sample]

# Audio effects
def time_stretch(data, rate):
    """Stretch audio in time (does not affect pitch)."""
    return librosa.effects.time_stretch(y=data, rate=rate)

def pitch_shift(data, sr, n_steps):
    """Shift the pitch of audio."""
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def dynamic_range_compression(data, threshold=0.1):
    """Apply simple dynamic range compression to normalize the audio."""
    max_amplitude = np.max(np.abs(data))
    if max_amplitude > threshold:
        return data / max_amplitude * threshold
    return data
