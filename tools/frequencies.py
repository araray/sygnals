import json
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np


def process_wav_json(json_data, output_file, sample_rate=44100):
    """
    Processes WAV-like data in JSON format, performs FFT, and saves a frequency spectrum plot to a file.

    Parameters:
        json_data (str or dict): The JSON data as a string or dictionary.
        output_file (str): The path to save the frequency spectrum plot.
        sample_rate (int): The sample rate of the audio data (default is 44100 Hz).
    """
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    # Extract amplitude values
    amplitudes = [point["amplitude"] for point in data]

    # Perform FFT
    fft_result = np.fft.fft(amplitudes)
    frequencies = np.fft.fftfreq(len(amplitudes), 1 / sample_rate)

    # Get the magnitude of the FFT (absolute value of complex numbers)
    magnitude = np.abs(fft_result)

    # Plot the positive half of the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies[: len(frequencies) // 2], magnitude[: len(magnitude) // 2])
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Save the plot to the specified file
    plt.savefig(output_file)
    plt.close()
    print(f"Frequency spectrum plot saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        output_file = data_file + ".png"
        with open(data_file) as data:
            json_data = json.load(data)
        process_wav_json(json_data, output_file, sample_rate=44100)
