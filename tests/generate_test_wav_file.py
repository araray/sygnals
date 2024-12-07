import numpy as np
import soundfile as sf

# Generate a sine wave
duration = 2.0  # seconds
sampling_rate = 44100  # Hz
frequency = 440.0  # Hz (A4)
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

# Save as a WAV file
sf.write("test_audio.wav", sine_wave, sampling_rate)
