import pytest
import numpy as np
import pandas as pd
import soundfile as sf
from sygnals.core.audio_handler import load_audio, save_audio, slice_audio, time_stretch, pitch_shift, dynamic_range_compression, save_audio_as_csv, save_audio_as_json
import librosa

@pytest.mark.parametrize("audio_file", ["tests/data/test_audio_001.wav"])
def test_load_audio(audio_file):
    data, sr = load_audio(audio_file)
    assert isinstance(data, np.ndarray)
    assert sr > 0

def test_save_audio(tmp_path):
    # Generate a simple sine wave
    sr = 22050
    t = np.linspace(0,1,sr)
    data = np.sin(2*np.pi*440*t)
    out_file = tmp_path / "out.wav"
    save_audio(data, sr, str(out_file))
    assert out_file.exists()
    loaded_data, loaded_sr = sf.read(str(out_file))
    assert np.allclose(data, loaded_data, atol=1e-5)
    assert sr == loaded_sr

def test_slice_audio():
    sr = 22050
    data = np.arange(sr*2)  # 2 seconds of data
    sliced = slice_audio(data, sr, start_time=0.5, end_time=1.0)
    assert len(sliced) == sr*(1.0-0.5)  # half a second of samples

def test_time_stretch():
    sr = 22050
    data = np.ones(sr) # 1 second of ones
    stretched = time_stretch(data, rate=2.0)
    # time_stretch should produce ~ 0.5s of data if rate=2.0
    assert 0.4*sr < len(stretched) < 0.6*sr

def test_pitch_shift():
    sr = 22050
    data = np.ones(sr)
    shifted = pitch_shift(data, sr, n_steps=2)
    assert len(shifted) == len(data)  # pitch shift doesn't change length

def test_dynamic_range_compression():
    data = np.array([0.5, 1.0, 0.2])
    compressed = dynamic_range_compression(data, threshold=0.5)
    assert np.max(np.abs(compressed)) <= 0.5

def test_save_audio_as_csv(tmp_path):
    df = pd.DataFrame({"time":[0,0.001,0.002],"amplitude":[0.1,0.2,0.3]})
    out = tmp_path / "audio.csv"
    save_audio_as_csv(df, str(out))
    loaded = pd.read_csv(out)
    assert loaded.equals(df)

def test_save_audio_as_json(tmp_path):
    df = pd.DataFrame({"time":[0,0.001],"amplitude":[0.1,0.2]})
    out = tmp_path / "audio.json"
    save_audio_as_json(df, str(out))
    loaded = pd.read_json(out)
    assert loaded.equals(df)
