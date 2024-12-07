import os
import pandas as pd

from sygnals.core.data_handler import read_data, save_data
from sygnals.core.dsp import compute_fft
from sygnals.core.transforms import wavelet_transform


def process_batch(input_dir, output_dir, transform):
    """Process multiple files in a directory and apply the given transform."""
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"{file_name}_processed.csv")

        data = read_data(input_path).values.flatten()

        if transform == "fft":
            freqs, spectrum = compute_fft(data)
            result = pd.DataFrame({
                "Frequency (Hz)": freqs,
                "Magnitude": spectrum
            })
            save_data(result, output_path)
        elif transform == "wavelet":
            coeffs = wavelet_transform(data)
            # coeffs is a list, convert it appropriately
            result = pd.DataFrame({f"Level {i+1}": c for i, c in enumerate(coeffs)})
            save_data(result, output_path)
