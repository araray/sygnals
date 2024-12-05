# Sygnals

**Sygnals** is a versatile command-line interface (CLI) for signal and audio processing. It allows you to analyze, manipulate, and transform time-series and audio data with a wide range of DSP features.

---

## Features
- Analyze data and audio files.
- Apply DSP operations like FFT, Wavelet, and Laplace transforms.
- Filter signals using low-pass, high-pass, and band-pass filters.
- Manipulate data using SQL-like queries and Pandas transformations.
- Process audio files with effects like time-stretching, pitch-shifting, and dynamic range compression.
- Generate visualizations such as spectrograms, FFT plots, and waveforms.
- Extend functionality with custom plugins.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/sygnals.git
cd sygnals
pip install .
```

---

## Usage

Run the CLI tool:
```bash
sygnals --help
```

### Analyze Data or Audio
```bash
sygnals analyze input.wav --output json
```

### Apply Filters
```bash
sygnals filter input.csv --low-pass 100 --output filtered.csv
```

### Transform Data
```bash
sygnals transform input.csv --fft --output fft_result.csv
```

### Visualize Data
```bash
sygnals visualize input.wav --type spectrogram --output spectrogram.png
```

---

## Custom Plugins
Write custom Python plugins in the `plugins/` directory. Example:

```python
from sygnals.core.plugin_manager import register_plugin

@register_plugin
def my_custom_plugin(data):
    return data * 2
```

List plugins:
```bash
sygnals plugin --list
```

Run a plugin:
```bash
sygnals plugin my_custom_plugin input.csv --output output.csv
```
