
# Sygnals: A CLI Toolkit for Signal and Audio Processing in Data Science Workflows

**Sygnals** is a command-line interface (CLI) tool designed to support **data science and machine learning workflows** by providing robust capabilities for pre-processing, analyzing, transforming, and featurizing time-series and audio data. It emphasizes compatibility with common data science libraries and frameworks (like Pandas and NumPy) and offers an extensible plugin architecture.

This toolkit facilitates the preparation of raw signal and audio data for subsequent analysis, feature engineering, and ingestion into pipelines for **training Artificial Intelligence (AI) or Neural Network (NN) models**.

---

## Real-World Problems

Sygnals can be used to tackle various real-world signal and audio processing problems, particularly those involving data preparation for machine learning:

*   **Environmental Sound Classification:**
    *   Segment long audio recordings into shorter clips (`sygnals segment fixed-length`).
    *   Extract relevant features like MFCCs and Spectral Contrast from each segment (`sygnals features extract`).
    *   Assemble the features into a dataset of feature vectors per segment (`sygnals save dataset --assembly-method vectors`) for training a classifier.
*   **Machinery Health Monitoring:**
    *   Record vibration data from machines over time (outside Sygnals).
    *   Load vibration data (e.g., from CSV).
    *   Compute time-domain features (RMS, Crest Factor) and frequency-domain features (PSD Welch) over time windows (`sygnals dsp psd-welch`, `sygnals features extract`).
    *   Monitor changes in these features over time or use them to train anomaly detection models (`sygnals features transform scale`, `sygnals save dataset`).
*   **Speech Analysis for Voice Pathology Detection:**
    *   Process speech recordings.
    *   Extract pitch (Fundamental Frequency), Jitter, and Shimmer features (`sygnals features extract`).
    *   Analyze or visualize the variation of these features (`sygnals visualize`).
    *   Use these features to train models to identify voice characteristics associated with pathologies (`sygnals save dataset`).
*   **Audio Augmentation for Robust ML Models:**
    *   Take a dataset of clean audio (e.g., speech commands).
    *   Generate augmented versions by adding different types and levels of noise (`sygnals augment add-noise`).
    *   Generate augmented versions by slightly shifting pitch (`sygnals augment pitch-shift`) or stretching time (`sygnals augment time-stretch`).
    *   Use the combined clean and augmented data to train models that are more robust to real-world variations.
*   **Filter Design and Analysis:**
    *   Analyze the frequency content of a signal (`sygnals dsp fft`).
    *   Design and analyze a filter (e.g., bandpass) to isolate a specific frequency range or remove noise (`sygnals filter analyze`).
    *   Apply the designed filter to the signal (`sygnals filter apply`).
*   **Custom Data Format Support:**
    *   If your data is stored in a specific format (e.g., a proprietary binary format, custom HDF5 structure, Parquet), write a plugin (`sygnals plugin scaffold`, `docs/developing_plugins.md`) to add a reader/writer for that format using `register_data_readers`/`register_data_writers`. Install the plugin (`sygnals plugin install`), and Sygnals commands will then be able to load/save your custom files.

---

## Features

Sygnals provides a comprehensive suite of tools, accessible via the command line, tailored for handling and preparing complex signal and audio data:

### Core Capabilities
*   **Robust Data Loading & Saving:** Load and save data in various formats including CSV, JSON, NPZ, WAV, FLAC, and Ogg. Support for additional formats like Parquet and HDF5 is available via plugins (e.g., installable via `pip install sygnals-parquet`).
*   **Layered Configuration:** Powerful configuration management via CLI arguments, environment variables, and TOML config files (`sygnals.toml`).
*   **Detailed Logging:** Configurable logging system with console output (using Rich) and persistent file logging.

### Digital Signal Processing (DSP) & Transforms (`sygnals dsp`)
*   **Time-Frequency Analysis:** Compute Fast Fourier Transform (FFT), Inverse FFT (IFFT), Short-Time Fourier Transform (STFT), and Constant-Q Transform (CQT).
*   **Wavelet Transforms:** Perform Discrete Wavelet Transform (DWT) and Inverse DWT (IDWT).
*   **Other Transforms:** Calculate Hilbert Transform (for analytic signal) and perform basic Numerical Laplace Transform.
*   **Analysis Techniques:** Compute Autocorrelation, Cross-correlation, and Power Spectral Density (using Periodogram and Welch's methods).
*   **Convolution:** Apply 1D Convolution.

### Filtering (`sygnals filter`)
*   **Filter Design & Analysis:** Analyze and plot the frequency response of filters (currently Butterworth).
*   **Filter Application:** Design and apply Butterworth IIR filters (low-pass, high-pass, band-pass, band-stop) using numerically stable Second-Order Sections (SOS) with zero-phase filtering.

### Audio Processing & Analysis (`sygnals audio`)
*   **Audio I/O:** Dedicated handling for common audio file formats (WAV, FLAC, Ogg, MP3 read; WAV, FLAC, Ogg write). Supports resampling during load.
*   **Effects (`sygnals audio effects`):** Apply audio effects and data augmentation techniques: Gain adjustment, basic Spectral Noise Reduction, HPSS-based Transient Shaping, Mid/Side Stereo Widening, Tremolo, Chorus, Flanger, Convolution Reverb (using generated IR), simple Dynamic Range Compression.
*   **Analysis (`sygnals audio analyze`):** Perform audio analysis including basic metrics (duration, RMS, peak) and onset detection.

### Signal Segmentation (`sygnals segment`)
*   **Fixed-Length Segmentation (`sygnals segment fixed-length`):** Divide audio into segments based on fixed length. Underlying utilities for silence and event-based segmentation are also available in the core library for programmatic use.

### Feature Engineering (`sygnals features`)
*   **Feature Extraction (`sygnals features extract`):** Extract a wide range of frame-level features from audio signals including:
    *   **Time Domain:** Mean Amplitude, Standard Deviation, Skewness, Kurtosis, Peak Amplitude, Crest Factor, Signal Entropy.
    *   **Frequency Domain:** Spectral Centroid, Spectral Bandwidth, Spectral Flatness, Spectral Rolloff, Dominant Frequency, Spectral Contrast.
    *   **Cepstral:** Mel-Frequency Cepstral Coefficients (MFCCs).
    *   **Audio Specific:** Zero Crossing Rate, RMS Energy (frame-based), Pitch (Fundamental Frequency).
    *   *Approximate* voice quality features: Harmonic-to-Noise Ratio (HNR), Jitter, and Shimmer. *Note: These are simplified approximations based on frame-level analysis.*
*   **Feature Transformation (`sygnals features transform scale`):** Apply transformations to extracted feature data, such as Standard, MinMax, and Robust Scaling. Requires the `scikit-learn` Python package (`pip install scikit-learn`).
*   **Feature Listing (`sygnals features list`):** List all available features.
*   **Show Feature Parameters (`sygnals features show-params`):** Display parameters for a specific feature.

### Data Augmentation (`sygnals augment`)
*   **Add Noise (`sygnals augment add-noise`):** Add noise to the signal at a specified SNR.
*   **Pitch Shift (`sygnals augment pitch-shift`):** Shift the pitch without changing duration.
*   **Time Stretch (`sygnals augment time-stretch`):** Stretch or compress duration without changing pitch.

### Dataset Assembly (`sygnals save`)
*   **Save Dataset (`sygnals save dataset`):** Format extracted features or processed data into structures suitable for ML models. Supported assembly methods include creating vectors per segment, stacking sequences, and formatting 2D feature maps as image-like arrays.

### Visualization (`sygnals visualize`)
*   **Generate Plots:** Create and save plots for Waveforms, Spectrograms, FFT Magnitude, FFT Phase, and Wavelet Scalograms.

### Plugin System (`sygnals plugin`)
*   **Extensibility:** Easily extend Sygnals by writing custom Python plugins for new filters, transforms, features, effects, visualizations, data handlers (readers/writers), and potentially CLI commands.
*   **Management:** CLI commands to list discovered plugins, enable/disable them, scaffold new plugin projects, and install/uninstall plugins.

### Other Utilities
*   **Configuration Inspection (`sygnals show-config`):** Display the currently loaded configuration.
*   **Safe Expression Evaluation:** Underlying utility (`sygnals.core.custom_exec`) for safely evaluating mathematical expressions within a restricted environment.
*   **Basic Database Storage:** Underlying utilities (`sygnals.core.storage`) for saving/querying data in SQLite databases (for programmatic use).
*   **Logging Utilities (`sygnals.utils.logging_config`):** Configuration of Python standard logging with Rich console handler.


---

## Installation

To get started with Sygnals, clone the repository and install the required dependencies. It is highly recommended to use a Python virtual environment.

```bash
# Ensure you have Git and Python 3.8+ installed

# Clone the Sygnals repository
git clone https://github.com/araray/sygnals.git
cd sygnals

# (Optional but Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate # On Windows use `.venv\Scripts\activate`

# Install Sygnals and its dependencies
# The '.[dev]' part includes dependencies needed for development and testing (like pytest, black, mypy, ruff)
pip install .[dev]
```

After installation, verify the `sygnals` command is available in your environment:

```bash
sygnals --version
```

## Usage

Sygnals is primarily used via the command line, following a `sygnals [GLOBAL OPTIONS] COMMAND [ARGUMENTS]` structure. Use the `--help` option to explore commands and their parameters.

### Global Options

*   `-v`, `--verbose`: Increase verbosity (show `INFO` messages). Can be repeated (`-vv` for `DEBUG`).
*   `-q`, `--quiet`: Suppress most console output (only shows `CRITICAL` errors).

### Top-Level Commands

The implemented top-level command groups are: `segment`, `features`, `augment`, `save`, `plugin`, `filter`, `visualize`, `audio`, `dsp`, `show-config`.

```bash
# Get overall help
sygnals --help

# Get help for a specific command group (e.g., features)
sygnals features --help

# Get help for a specific subcommand (e.g., features transform scale)
sygnals features transform scale --help
```

Here are examples of common workflows:

### `sygnals segment`

Segments signals into smaller pieces.

#### `sygnals segment fixed-length <INPUT_FILE>`

Segments a signal into fixed-length windows with optional overlap and padding. Requires audio input.

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input audio file (e.g., `.wav`, `.flac`).
*   **Options:**
    *   `-o, --output DIRECTORY`: **Required.** Output path. This is a *directory* where individual segment audio files will be saved (e.g., `segment_001.wav`, `segment_002.wav`).
    *   `--length FLOAT`: **Required.** Segment length in seconds. Must be positive.
    *   `--overlap FLOAT`: Overlap ratio between `0.0` (no overlap) and `1.0` (exclusive). Default: `0.0`.
    *   `--pad / --no-pad`: Pad the last segment with zeros if shorter than `--length`. Default: `--pad`.
    *   `--min-length FLOAT`: Minimum segment length in seconds to keep. Segments shorter than this (after potential padding) are discarded. Default: `None`.

*   **Example:** Segment `audio.wav` into 1.5-second segments with 50% overlap, saving them to `./segments/`.

    ```bash
    sygnals segment fixed-length audio.wav \
        --output ./segments/ \
        --length 1.5 \
        --overlap 0.5
    ```

### `sygnals features`

Extracts and transforms signal features.

#### `sygnals features extract <INPUT_FILE>`

Extracts features from a signal or audio file. Requires audio input for most features.

```bash
sygnals features extract <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file (typically audio for feature extraction).
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path (e.g., `features.csv`, `features.npz`). The format is inferred from the extension. Output contains features per frame, plus a 'time' column/array.
    *   `-f, --feature NAME`: **Required (can be repeated).** Name of the feature(s) to extract. Use `all` to extract most known standard features. Supported features are listed in the "Implemented DSP, Audio Processing, and Features" section. Features like `mfcc` or `spectral_contrast` will produce multiple output columns (e.g., `mfcc_0`, `mfcc_1`, ..., `contrast_band_0`, ...).
    *   `--frame-length INTEGER`: Analysis frame length in samples (used for framing time-domain features and as default `n_fft` for spectral features). Default: `2048`.
    *   `--hop-length INTEGER`: Hop length between frames in samples. Default: `512`.
    *   *(Note: Specific parameters for individual features like `n_mfcc` can often be set via a configuration file.)*

*   **Example:** Extract RMS energy, Spectral Centroid, and all default MFCCs (13 coefficients) from `voice.wav`, saving to `voice_features.csv`.

    ```bash
    sygnals features extract voice.wav \
        --output voice_features.csv \
        --feature rms_energy \
        --feature spectral_centroid \
        --feature mfcc \
        --frame-length 1024 \
        --hop-length 256
    ```

#### `sygnals features transform scale <INPUT_FILE>`

Applies feature scaling (normalization) to an existing feature file. Requires the `scikit-learn` Python package (`pip install scikit-learn`). Input must be a tabular format (like CSV or NPZ with a suitable array).

```bash
sygnals features transform scale <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input feature file (e.g., `.csv`, `.npz`).
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path for the scaled features. Format is inferred from the extension.
    *   `--scaler TYPE`: Type of scaler: `standard` (StandardScaler), `minmax` (MinMaxScaler), `robust` (RobustScaler). Default: `standard`.
    *   *(Note: Scaler parameters can often be set via a configuration file.)*

*   **Example:** Apply standard scaling to features in `features.csv`, saving to `features_scaled.csv`.

    ```bash
    sygnals features transform scale features.csv \
        --output features_scaled.csv \
        --scaler standard
    ```

#### `sygnals features list`

Lists all available core features that can be extracted with descriptions and domains.

*   **Example:**

    ```bash
    sygnals features list
    ```

#### `sygnals features show-params <NAME>`

Show available parameters for a specific feature.

*   **Arguments:**
    *   `NAME`: The name of the feature.

*   **Example:**

    ```bash
    sygnals features show-params mfcc
    ```

### `sygnals augment`

Applies data augmentation techniques to audio signals. Requires audio input.

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input audio file.
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path for the augmented audio.

#### `sygnals augment add-noise <INPUT_FILE>`

Adds noise to the signal at a specified Signal-to-Noise Ratio (SNR).

*   **Options:**
    *   `--snr FLOAT`: **Required.** Target Signal-to-Noise Ratio in dB.
    *   `--noise-type TYPE`: Type of noise: `gaussian` (alias `white`), `pink` (placeholder), `brown` (placeholder). Default: `gaussian`. *Note: 'pink' and 'brown' currently add white noise.*
    *   `--seed INTEGER`: Random seed for noise generation reproducibility.

*   **Example:** Add Gaussian noise with 10 dB SNR to `clean.wav`, saving to `noisy.wav`.

    ```bash
    sygnals augment add-noise clean.wav -o noisy.wav --snr 10.0
    ```

#### `sygnals augment pitch-shift <INPUT_FILE>`

Shifts the pitch of the audio signal without changing duration. Requires `librosa`.

*   **Options:**
    *   `--steps FLOAT`: **Required.** Number of semitones to shift (positive for up, negative for down). Can be fractional.
    *   `--bins-per-octave INTEGER`: Number of bins per octave for the pitch shift calculation. Default: `12`.

*   **Example:** Shift `voice.wav` up by 2 semitones, saving to `voice_higher.wav`.

    ```bash
    sygnals augment pitch-shift voice.wav -o voice_higher.wav --steps 2.0
    ```

#### `sygnals augment time-stretch <INPUT_FILE>`

Stretches or compresses the time duration of the audio signal without changing pitch. Requires `librosa`.

*   **Options:**
    *   `--rate FLOAT`: **Required.** Stretch factor (`>1` speeds up, `<1` slows down). Must be positive.

*   **Example:** Slow down `music.wav` to 90% of its original speed, saving to `music_slow.wav`.

    ```bash
    sygnals augment time-stretch music.wav -o music_slow.wav --rate 0.9
    ```

### `sygnals save`

Saves processed data or assembles datasets.

#### `sygnals save dataset <INPUT_FILE>`

Saves processed data, potentially applying formatting for ML model ingestion. The input file typically contains features or processed data (e.g., from `features extract`, `features transform scale`).

```bash
sygnals save dataset <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file.
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path for the dataset. Format is inferred from the extension, but can be overridden by `--format`.
    *   `--format EXT`: Explicitly set output format (e.g., `npz`, `csv`), overriding the output file extension. Supports core formats and plugin-added writers.
    *   `--assembly-method METHOD`: How to structure the output dataset. Default: `none`.
        *   `none`: Saves the input data directly without restructuring. Input can be any format loadable by `data_handler`. Output format is determined by `--output` extension or `--format`.
        *   `vectors`: Aggregates frame-level features (from input file) into one vector per segment. Input file should contain frame-level features (e.g., CSV or NPZ dict of 1D arrays). Requires `--segment-info`. Output is typically CSV (DataFrame) or NPZ (NumPy array).
        *   `sequences`: Stacks frame-level features (from input file) into sequences. Input file should contain frame-level features (e.g., CSV or NPZ dict of 1D arrays) for a single signal/segment. Handles optional padding/truncation. Output is typically NPZ (3D NumPy array).
        *   `image`: Formats 2D feature maps (e.g., spectrograms from input file) into an image-like array. Input file should contain a 2D array (NPZ) or a suitable DataFrame (CSV). Optional resize/normalization. Output is typically NPZ.
    *   `--segment-info FILE`: Path to segment information (e.g., CSV with `start_frame`, `end_frame` columns). Required for `vectors` assembly method.
    *   `--aggregation TEXT`: Aggregation method(s) for `vectors` assembly. Either a single method name (`mean`, `std`, `max`, `min`, `median`) or a JSON string mapping feature names to methods (e.g., `{"mfcc_0": "mean", "zcr": "max"}`). Default: `mean`.
    *   `--max-sequence-length INTEGER`: Maximum length for `sequences` assembly. Truncates or pads sequences to this length.
    *   `--padding-value FLOAT`: Value used for padding shorter sequences in `sequences` assembly. Default: `0.0`.
    *   `--truncation-strategy [pre|post]`: Truncation strategy (`pre` (remove from beginning) or `post` (remove from end)) for `sequences` assembly. Default: `post`.
    *   `--output-shape TEXT`: Target `height,width` for `image` assembly (e.g., `128,64`). Must be two positive integers separated by a comma.
    *   `--resize-order INTEGER`: Interpolation order (0-5) for resizing in `image` assembly. Default: `1`.
    *   `--no-normalize`: Disable normalization to `[0, 1]` for `image` assembly.

*   **Example 1:** Simply save an NPZ file `features.npz` to a CSV file `final_features.csv`.

    ```bash
    sygnals save dataset features.npz \
        --output final_features.csv \
        --format csv \
        --assembly-method none
    ```
*   **Example 2:** Assemble features from `frame_features.npz` into vectors per segment using `mean` aggregation, based on `segments.csv`, saving to `segment_vectors.npz`.

    ```bash
    sygnals save dataset frame_features.npz \
        --output segment_vectors.npz \
        --assembly-method vectors \
        --segment-info segments.csv \
        --aggregation mean \
        --format npz
    ```
*   **Example 3:** Assemble features from `frame_features.npz` into sequences padded/truncated to length 100, saving to `sequences.npz`.

    ```bash
    sygnals save dataset frame_features.npz \
        --output sequences.npz \
        --assembly-method sequences \
        --max-sequence-length 100
    ```
*   **Example 4:** Assemble a spectrogram from `spectrogram.npz` into a 64x64 image, normalizing values, saving to `spectrogram_image.npz`.

    ```bash
    sygnals save dataset spectrogram.npz \
        --output spectrogram_image.npz \
        --assembly-method image \
        --output-shape 64,64
    ```

### `sygnals plugin`

Manage Sygnals plugins.

#### `sygnals plugin list`

Lists all discovered plugins, their version, status (loaded, disabled, incompatible), and source (local or entry point).

*   **Example:**

    ```bash
    sygnals plugin list
    ```

#### `sygnals plugin enable <NAME>`

Enables a specific plugin by its name. The change takes effect on the next Sygnals run.

*   **Arguments:**
    *   `NAME`: The unique name of the plugin to enable.

*   **Example:**

    ```bash
    sygnals plugin enable my-custom-feature-plugin
    ```

#### `sygnals plugin disable <NAME>`

Disables a specific plugin by its name. The change takes effect on the next Sygnals run.

*   **Arguments:**
    *   `NAME`: The unique name of the plugin to disable.

*   **Example:**

    ```bash
    sygnals plugin disable experimental-filter
    ```

#### `sygnals plugin scaffold <NAME>`

Generates a template directory structure for creating a new plugin.

*   **Arguments:**
    *   `NAME`: The desired name for the new plugin.
*   **Options:**
    *   `--dest PATH`: Destination directory for the plugin template. Default: `.`.

*   **Example:**

    ```bash
    sygnals plugin scaffold my-new-plugin --dest ./plugins_dev/
    ```

#### `sygnals plugin install <SOURCE>`

Installs a plugin from a local path or PyPI.

*   **Arguments:**
    *   `SOURCE`: A local directory path containing a plugin (`plugin.toml` must exist) or a package name from PyPI.
*   **Options:**
    *   `--force`: Force reinstall/overwrite if the plugin already exists locally or via pip.

*   **Example 1:** Install a plugin from a local directory `./my-plugin-source/`.

    ```bash
    sygnals plugin install ./my-plugin-source/
    ```

*   **Example 2:** Install a plugin from PyPI (e.g., assuming a package named `sygnals-awesome-plugin`).

    ```bash
    sygnals plugin install sygnals-awesome-plugin
    ```

#### `sygnals plugin uninstall <NAME>`

Uninstalls a plugin by its name. Handles plugins installed locally or via pip (entry points).

*   **Arguments:**
    *   `NAME`: The unique name of the plugin to uninstall.
*   **Options:**
    *   `-y, --yes`: Do not ask for confirmation before uninstalling.

*   **Example:** Uninstall the plugin named `sygnals-awesome-plugin`.

    ```bash
    sygnals plugin uninstall sygnals-awesome-plugin
    ```

### `sygnals filter`

Apply filters or analyze filter characteristics.

#### `sygnals filter analyze`

Analyze and plot the frequency response of a filter.

```bash
sygnals filter analyze [OPTIONS]
```

*   **Options:**
    *   `--type [lowpass|highpass|bandpass|bandstop]`: **Required.** Type of the filter (e.g., 'lowpass').
    *   `--cutoff TEXT`: **Required.** Cutoff frequency (Hz). Single value for low/highpass, comma-separated for bandpass/stop (e.g., '100' or '50,200').
    *   `--fs FLOAT`: **Required.** Sampling frequency of the signal (Hz).
    *   `--order INTEGER`: Order of the Butterworth filter. Default: `5`.
    *   `-o, --output FILE`: **Required.** Output file path for the frequency response plot (e.g., 'filter_response.png').

*   **Example:** Analyze and plot a 6th order Butterworth bandpass filter response from 100 Hz to 500 Hz at 8000 Hz sampling rate, saving the plot to `bandpass_response.png`.

    ```bash
    sygnals filter analyze \
        --type bandpass \
        --cutoff 100,500 \
        --fs 8000 \
        --order 6 \
        --output bandpass_response.png
    ```

#### `sygnals filter apply <INPUT_FILE>`

Apply a Butterworth filter to a signal or audio file. Uses zero-phase filtering.

```bash
sygnals filter apply <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file (audio, CSV with 'value', or NPZ with 'data').
*   **Options:**
    *   `--type [lowpass|highpass|bandpass|bandstop]`: **Required.** Type of the filter to apply.
    *   `--cutoff TEXT`: **Required.** Cutoff frequency (Hz). Single value for low/highpass, comma-separated for bandpass/stop.
    *   `--fs FLOAT`: Sampling frequency (Hz). Required if input file does not contain SR information.
    *   `--order INTEGER`: Order of the Butterworth filter. Default: `5`.
    *   `-o, --output FILE`: **Required.** Output file path for the filtered data.

*   **Example:** Apply a 5th order lowpass filter with a cutoff of 500 Hz to `audio.wav` (assuming SR can be read from the file), saving to `audio_lp.wav`.

    ```bash
    sygnals filter apply audio.wav \
        --type lowpass \
        --cutoff 500 \
        --output audio_lp.wav
    ```

### `sygnals visualize`

Generate various signal and audio visualizations. Requires `matplotlib`.

#### `sygnals visualize waveform <INPUT_FILE>`

Generate a waveform (amplitude vs. time) plot.

```bash
sygnals visualize waveform <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file (audio, CSV with 'value', or NPZ with 'data').
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path for the plot (e.g., '.png', '.svg').
    *   `--sr FLOAT`: Sampling rate (Hz). Required if input file does not contain SR.
    *   `--max-samples INTEGER`: Limit plotting to the first N samples.
    *   `--title TEXT`: Plot title. Default: 'Waveform'.

*   **Example:** Plot the waveform of `signal.csv` (assuming SR=1000), saving to `waveform.png`.

    ```bash
    sygnals visualize waveform signal.csv --sr 1000 -o waveform.png
    ```

#### `sygnals visualize spectrogram <INPUT_FILE>`

Generate a spectrogram (time-frequency representation) plot. Requires `scipy`.

```bash
sygnals visualize spectrogram <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file (audio, CSV with 'value', or NPZ with 'data').
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path for the plot.
    *   `--sr FLOAT`: **Required.** Sampling rate (Hz).
    *   `--f-min FLOAT`: Minimum frequency to display (Hz). Default: `0.0`.
    *   `--f-max FLOAT`: Maximum frequency to display (Hz). Defaults to Nyquist.
    *   `--window TEXT`: Window function for STFT. Default: 'hann'.
    *   `--nperseg INTEGER`: Length of each segment for STFT (samples).
    *   `--noverlap INTEGER`: Overlap between segments (samples). Defaults to nperseg // 2.
    *   `--db-scale / --linear-scale`: Display power in dB or linear scale. Default: `--db-scale`.
    *   `--cmap TEXT`: Matplotlib colormap name. Default: 'viridis'.
    *   `--title TEXT`: Plot title. Default: 'Spectrogram'.

*   **Example:** Plot a spectrogram of `audio.wav` (assuming SR=44100), showing frequencies from 100 Hz to 8000 Hz in dB scale, saving to `spectrogram.png`.

    ```bash
    sygnals visualize spectrogram audio.wav \
        --sr 44100 \
        --f-min 100 --f-max 8000 \
        -o spectrogram.png
    ```

#### `sygnals visualize fft-magnitude <INPUT_FILE>`

Generate a plot of the FFT magnitude spectrum. Requires `scipy`.

```bash
sygnals visualize fft-magnitude <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file (audio, CSV with 'value', or NPZ with 'data').
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path for the plot.
    *   `--sr FLOAT`: **Required.** Sampling rate (Hz).
    *   `--window TEXT`: Window function applied before FFT. Default: 'hann'.
    *   `--n INTEGER`: FFT length (for padding/truncation). Defaults to signal length.
    *   `--title TEXT`: Plot title. Default: 'FFT Magnitude Spectrum'.

*   **Example:** Plot the magnitude spectrum of `signal.wav` (SR=22050), saving to `fft_mag.png`.

    ```bash
    sygnals visualize fft-magnitude signal.wav --sr 22050 -o fft_mag.png
    ```

#### `sygnals visualize fft-phase <INPUT_FILE>`

Generate a plot of the FFT phase spectrum. Requires `scipy`.

```bash
sygnals visualize fft-phase <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file (audio, CSV with 'value', or NPZ with 'data').
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path for the plot.
    *   `--sr FLOAT`: **Required.** Sampling rate (Hz).
    *   `--window TEXT`: Window function applied before FFT. Default: 'hann'.
    *   `--unwrap / --no-unwrap`: Unwrap the phase to make it continuous. Default: `--unwrap`.
    *   `--n INTEGER`: FFT length. Defaults to signal length.
    *   `--title TEXT`: Plot title. Default: 'FFT Phase Spectrum'.

*   **Example:** Plot the unwrapped phase spectrum of `signal.wav` (SR=22050), saving to `fft_phase.png`.

    ```bash
    sygnals visualize fft-phase signal.wav --sr 22050 -o fft_phase.png
    ```

#### `sygnals visualize scalogram <INPUT_FILE>`

Generate a Wavelet Transform Scalogram plot using CWT. Requires `pywt`.

```bash
sygnals visualize scalogram <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file (audio, CSV with 'value', or NPZ with 'data').
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path for the plot.
    *   `--sr FLOAT`: Sampling rate (Hz). If provided, y-axis shows approximate frequencies; otherwise, it shows scales.
    *   `--scales TEXT`: Wavelet scales. Integer count (> 0) or comma-separated list of positive floats (e.g., '64' or '1.0,2.0,4.0'). Default: '64'.
    *   `--wavelet TEXT`: Continuous wavelet name (e.g., 'morl', 'cmorB-C', 'gausN'). Default: 'morl'.
    *   `--cmap TEXT`: Matplotlib colormap name. Default: 'viridis'.
    *   `--title TEXT`: Plot title. Default: 'Wavelet Scalogram'.

*   **Example:** Plot a scalogram of `signal.wav` (SR=22050) using 128 scales, saving to `scalogram.png`.

    ```bash
    sygnals visualize scalogram signal.wav --sr 22050 --scales 128 -o scalogram.png
    ```

### `sygnals audio`

Apply audio-specific processing, effects, or analysis.

#### `sygnals audio effects`

Apply various audio effects.

*   **Subcommands:** `gain`, `reverb`, `delay`, `tremolo`, `chorus`, `flanger`, `compression`, `hpss`, `stereo-widen`. Each requires an `INPUT_FILE` and `-o, --output FILE` path. Options are specific to each effect.

*   **Example: Apply Gain:**

    ```bash
    sygnals audio effects gain audio.wav -o audio_loud.wav --gain-db 6.0
    ```

*   **Example: Apply Reverb:**

    ```bash
    sygnals audio effects reverb audio.wav -o audio_reverb.wav --decay-time 1.0 --wet-level 0.4
    ```

*   **Example: Apply simple Compression:**

    ```bash
    sygnals audio effects compression audio.wav -o audio_comp.wav --threshold 0.7 --ratio 5.0
    ```

#### `sygnals audio analyze`

Perform audio analysis and calculate metrics.

*   **Subcommands:** `metrics`, `onsets`.

*   **Example: Calculate Metrics:**

    ```bash
    sygnals audio analyze metrics audio.wav --sr 44100 # SR can often be read from file
    ```

*   **Example: Detect Onsets:**

    ```bash
    sygnals audio analyze onsets audio.wav --sr 44100 -o onsets.csv --units time
    ```

### `sygnals dsp`

Perform core Digital Signal Processing (DSP) operations.

#### `sygnals dsp fft <INPUT_FILE>`

Compute the Fast Fourier Transform (FFT).

```bash
sygnals dsp fft <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file (audio, CSV with 'value', or NPZ with 'data').
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path (e.g., '.csv', '.npz'). CSV includes Frequency, Magnitude, Phase. NPZ includes frequencies, spectrum (complex), fs.
    *   `--fs FLOAT`: **Required.** Sampling frequency (Hz).
    *   `--window TEXT`: Window function. Default: 'hann'.
    *   `--n INTEGER`: FFT length. Defaults to signal length.

*   **Example:** Compute FFT of `signal.csv` (SR=1000), save to `fft_result.csv`.

    ```bash
    sygnals dsp fft signal.csv --fs 1000 -o fft_result.csv
    ```

#### `sygnals dsp ifft <INPUT_FILE>`

Compute the Inverse Fast Fourier Transform (IFFT). Input is typically complex spectrum data (e.g., from `dsp fft`).

```bash
sygnals dsp ifft <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input spectrum file (e.g., NPZ from `dsp fft` or CSV with Frequency, Magnitude, Phase).
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path (e.g., '.csv', '.npz', audio formats if SR available).
    *   `--n INTEGER`: Length of the output signal. Defaults to spectrum length.

*   **Example:** Compute IFFT from `fft_result.npz`, save reconstructed signal to `reconstructed.wav` (requires SR to be present in NPZ or inferrable).

    ```bash
    sygnals dsp ifft fft_result.npz -o reconstructed.wav
    ```

#### `sygnals dsp convolution <INPUT_FILE_1> <INPUT_FILE_2>`

Apply convolution to a 1D signal using a 1D kernel.

```bash
sygnals dsp convolution <INPUT_FILE_1> <INPUT_FILE_2> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE_1`: Path to the input signal file (audio, CSV with 'value', or NPZ with 'data').
    *   `INPUT_FILE_2`: Path to the input kernel file (CSV with 'value', or NPZ with 'data').
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path (e.g., '.csv', '.npz', audio if in 'same' mode and SR available).
    *   `--mode [full|valid|same]`: Convolution mode. Default: 'same'.

*   **Example:** Convolve `signal.csv` with `kernel.csv` in 'same' mode, save to `convolved.csv`.

    ```bash
    sygnals dsp convolution signal.csv kernel.csv -o convolved.csv --mode same
    ```

#### `sygnals dsp correlation <INPUT_FILE_1> [INPUT_FILE_2]`

Compute cross-correlation or autocorrelation.

```bash
sygnals dsp correlation <INPUT_FILE_1> [INPUT_FILE_2] [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE_1`: Path to the first input signal file.
    *   `INPUT_FILE_2`: Optional path to the second input signal file. If omitted, computes autocorrelation of `INPUT_FILE_1`.
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path (e.g., '.csv', '.npz').
    *   `--mode [full|valid|same]`: Correlation mode. Default: 'full'.
    *   `--method [auto|direct|fft]`: Computation method. Default: 'auto'.

*   **Example 1:** Compute autocorrelation of `signal.csv`.

    ```bash
    sygnals dsp correlation signal.csv -o autocorrelation.csv
    ```

*   **Example 2:** Compute cross-correlation of `signal_a.csv` and `signal_b.csv`.

    ```bash
    sygnals dsp correlation signal_a.csv signal_b.csv -o crosscorrelation.csv
    ```

#### `sygnals dsp psd-periodogram <INPUT_FILE>`

Estimate Power Spectral Density using Periodogram.

```bash
sygnals dsp psd-periodogram <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file (audio, CSV with 'value', or NPZ with 'data').
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path (e.g., '.csv', '.npz').
    *   `--fs FLOAT`: **Required.** Sampling frequency (Hz).
    *   `--window TEXT`: Window function. Default: 'hann'.
    *   `--nfft INTEGER`: FFT length.
    *   `--detrend [none|constant|linear]`: Detrending method. Default: 'constant'.
    *   `--scaling [density|spectrum]`: Scaling method. Default: 'density'.

*   **Example:** Compute PSD of `signal.wav` (SR=44100), save to `psd_periodogram.csv`.

    ```bash
    sygnals dsp psd-periodogram signal.wav --fs 44100 -o psd_periodogram.csv
    ```

#### `sygnals dsp psd-welch <INPUT_FILE>`

Estimate Power Spectral Density using Welch's method.

```bash
sygnals dsp psd-welch <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file.
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path.
    *   `--fs FLOAT`: **Required.** Sampling frequency (Hz).
    *   `--window TEXT`: Window function. Default: 'hann'.
    *   `--nperseg INTEGER`: Length of each segment.
    *   `--noverlap INTEGER`: Number of points to overlap.
    *   `--nfft INTEGER`: FFT length.
    *   `--detrend [none|constant|linear]`: Detrending method. Default: 'constant'.
    *   `--scaling [density|spectrum]`: Scaling method. Default: 'density'.

*   **Example:** Compute PSD of `audio.wav` (SR=44100) using Welch's method, with 1024 samples per segment, save to `psd_welch.npz`.

    ```bash
    sygnals dsp psd-welch audio.wav --fs 44100 --nperseg 1024 -o psd_welch.npz
    ```

#### `sygnals dsp dwt <INPUT_FILE>`

Compute the Discrete Wavelet Transform (DWT).

```bash
sygnals dsp dwt <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file.
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path (supports .npz).
    *   `--wavelet TEXT`: Discrete wavelet name (e.g., 'db4'). Default: 'db4'.
    *   `--level INTEGER`: Decomposition level. Defaults to max possible level.
    *   `--mode TEXT`: Signal extension mode. Default: 'symmetric'.

*   **Example:** Compute DWT of `signal.csv` using 'sym5' wavelet at level 5, save coefficients to `dwt_coeffs.npz`.

    ```bash
    sygnals dsp dwt signal.csv --wavelet sym5 --level 5 -o dwt_coeffs.npz
    ```

#### `sygnals dsp idwt <INPUT_FILE>`

Compute the Inverse Discrete Wavelet Transform (IDWT) from coefficients.

```bash
sygnals dsp idwt <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input coefficient file (e.g., NPZ from `dsp dwt`).
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path (e.g., '.csv', '.npz', audio formats).
    *   `--wavelet TEXT`: **Required.** Discrete wavelet name used for decomposition.
    *   `--mode TEXT`: Signal extension mode used for decomposition. Default: 'symmetric'.

*   **Example:** Reconstruct signal from `dwt_coeffs.npz` using 'sym5' wavelet, save to `reconstructed_signal.wav` (requires SR to be available in NPZ).

    ```bash
    sygnals dsp idwt dwt_coeffs.npz --wavelet sym5 -o reconstructed_signal.wav
    ```

#### `sygnals dsp hilbert <INPUT_FILE>`

Compute the Analytic Signal using the Hilbert Transform.

```bash
sygnals dsp hilbert <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file.
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path (e.g., '.csv', '.npz'). Output is complex.

*   **Example:** Compute Hilbert transform of `signal.csv`, save analytic signal to `analytic_signal.npz`.

    ```bash
    sygnals dsp hilbert signal.csv -o analytic_signal.npz
    ```

#### `sygnals dsp laplace <INPUT_FILE>`

Compute the Numerical Laplace Transform for specified s-values.

```bash
sygnals dsp laplace <INPUT_FILE> [OPTIONS]
```

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file.
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path (e.g., '.csv', '.npz').
    *   `--s-values TEXT`: **Required.** Comma-separated list of complex 's' values (sigma+j*omega), e.g., '0.1+0.5j,1.0'.
    *   `--t-step FLOAT`: Time step between samples (1/fs). Required if FS cannot be determined from input.

*   **Example:** Compute Laplace transform of `signal.csv` (assuming t_step=0.001), for s=1.0 and s=0.5+0.2j, save to `laplace_result.csv`.

    ```bash
    sygnals dsp laplace signal.csv --t-step 0.001 --s-values '1.0,0.5+0.2j' -o laplace_result.csv
    ```

### `sygnals show-config`

Show the currently loaded configuration, optionally filtered by section path.

```bash
sygnals show-config [SECTION_PATH]
```

*   `SECTION_PATH`: Optional path to a configuration section or parameter (e.g., `paths`, `logging.log_level_file`).

*   **Example 1:** Show the full configuration.

    ```bash
    sygnals show-config
    ```

*   **Example 2:** Show only the logging configuration section.

    ```bash
    sygnals show-config logging
    ```

*   **Example 3:** Show the default sample rate parameter.

    ```bash
    sygnals show-config defaults.default_sample_rate
    ```

---

## Implemented DSP, Audio Processing, and Features

Sygnals provides access to a variety of signal processing algorithms and features through its core Python modules. Many of these are utilized by the CLI commands, while others are available for programmatic use. The implementation relies on standard libraries like NumPy, SciPy, Pandas, and Librosa, and PyWavelets for DWT.

*   **Digital Filters (`sygnals.core.filters`):**
    *   Butterworth IIR filters (Low-pass, High-pass, Band-pass, Band-stop) designed using `scipy.signal.butter`.
    *   Applied using zero-phase filtering (`scipy.signal.sosfiltfilt`) for numerical stability and phase preservation.
*   **Frequency Domain Analysis (`sygnals.core.dsp`, `sygnals.core.features.frequency_domain`):**
    *   Fast Fourier Transform (FFT) and Inverse FFT (IFFT) using `scipy.fft`.
    *   Short-Time Fourier Transform (STFT) and Constant-Q Transform (CQT) using `librosa`.
    *   Power Spectral Density (PSD) estimation (Periodogram, Welch's method) using `scipy.signal`.
    *   Frame-based spectral features (`sygnals.core.features.frequency_domain`): Spectral Centroid, Bandwidth (p-norm), Flatness (Wiener entropy ratio), Rolloff (percentile frequency), Dominant Frequency (peak frequency bin). Calculated from magnitude spectra.
    *   Spectral Contrast (`sygnals.core.features.frequency_domain`) calculated from magnitude spectrogram.
*   **Time Domain Analysis (`sygnals.core.dsp`, `sygnals.core.features.time_domain`, `sygnals.core.audio.features`):**
    *   Convolution (`scipy.signal.fftconvolve`) and Auto/Cross-correlation (`scipy.signal.correlate`).
    *   Amplitude Envelope detection (Hilbert transform magnitude via `scipy.signal.hilbert`, or frame-based RMS).
    *   Windowing functions (`scipy.signal.get_window`) applied for spectral analysis.
    *   Frame-based time features (`sygnals.core.features.time_domain`): Mean Absolute Amplitude, Standard Deviation, Skewness (amplitude distribution asymmetry), Kurtosis (amplitude distribution peakedness), Peak Absolute Amplitude, Crest Factor (peak-to-RMS ratio), Signal Entropy (amplitude distribution entropy - binned).
    *   Audio-specific frame features (`sygnals.core.audio.features`): Zero-Crossing Rate (ZCR), Root Mean Square (RMS) Energy.
*   **Audio-Specific Features (`sygnals.core.audio.features`):**
    *   Fundamental Frequency (Pitch) estimation (pYIN, YIN algorithms) using `librosa.pyin` / `librosa.yin`.
    *   Basic Global Audio Metrics (Duration, Global RMS, Peak Amplitude).
    *   **Approximate Voice Quality Features:** Harmonic-to-Noise Ratio (HNR), Jitter, Shimmer. *Note: These implementations are simplified for frame-based analysis and may yield different results than traditional methods or tools like Praat.*
    *   Onset Detection (`librosa.onset.onset_detect`).
*   **Audio Effects (`sygnals.core.audio.effects`):**
    *   `sygnals.core.audio.effects.compression`: Simple Dynamic Range Compression.
    *   `sygnals.core.audio.effects.delay`: Simple Delay with feedback.
    *   `sygnals.core.audio.effects.chorus`: Chorus effect.
    *   `sygnals.core.audio.effects.flanger`: Flanger effect.
    *   `sygnals.core.audio.effects.tremolo`: Tremolo (amplitude modulation).
    *   `sygnals.core.audio.effects.utility`: Gain Adjustment, basic Spectral Subtraction Noise Reduction, Transient Shaping (HPSS-based), Mid/Side Stereo Widening.
    *   `sygnals.core.audio.effects.pitch_shift`: Pitch Shifting (used in `augment`).
    *   `sygnals.core.audio.effects.time_stretch`: Time Stretching (used in `augment`).
    *   `sygnals.core.audio.effects.reverb`: Simple Convolution Reverb (using a generated IR).
    *   `sygnals.core.audio.effects.equalizer`: Experimental Graphic and Parametric EQ (using `scipy.signal` filters - *Note: Peaking EQ design is currently experimental*).
*   **Wavelet Transforms (`sygnals.core.transforms`):**
    *   Discrete Wavelet Transform (DWT) and Inverse DWT using `pywt`.
    *   Hilbert Transform using `scipy.signal.hilbert`.
    *   Numerical Laplace Transform (basic numerical approximation).
*   **ML Utilities (`sygnals.core.ml_utils`):**
    *   `sygnals.core.ml_utils.scaling`: StandardScaler, MinMaxScaler, RobustScaler wrappers using `scikit-learn` (if installed).
    *   `sygnals.core.ml_utils.formatters`: Dataset Formatting functions (`format_feature_vectors_per_segment`, `format_feature_sequences`, `format_features_as_image`).
*   **Data Handling (`sygnals.core.data_handler`):**
    *   Unified interface for reading/writing CSV, JSON, NPZ files using Pandas/NumPy.
    *   Delegation to `sygnals.core.audio.io` for WAV, FLAC, Ogg, MP3 files (using `soundfile` and `librosa`).
    *   Extensible via plugin-registered readers/writers.
    *   Basic SQL querying (`pandasql`) and Pandas filtering on DataFrames.
*   **Custom Execution (`sygnals.core.custom_exec`):** Safe evaluation of mathematical expressions using a restricted environment.
*   **Logging Utilities (`sygnals.utils.logging_config`):** Configuration of Python standard logging with Rich console handler.
*   **Visualization Utilities (`sygnals.utils.visualizations`):** Functions for generating Matplotlib plots (Spectrogram, FFT Magnitude/Phase, Waveform, Scalogram).

This section provides a detailed overview of the algorithms and modules that constitute the Sygnals core functionality as implemented in the provided source code.

---

## Plugin System

Sygnals features a flexible plugin system allowing users and developers to extend its functionality without modifying the core codebase. Plugins can add new filters, transforms, features, effects, visualizations, data handlers (readers/writers), and even custom CLI commands.

### Using Plugins

Plugins installed via pip (declaring a `sygnals.plugins` entry point) or placed in the configured local plugin directory (`~/.config/sygnals/plugins/` by default) are automatically discovered by Sygnals on startup.

You can manage discovered plugins using the `sygnals plugin` command group:

*   `sygnals plugin list`: See all discovered plugins and their current status (loaded, disabled, incompatible).
*   `sygnals plugin enable <name>`: Enable a discovered plugin for the next run.
*   `sygnals plugin disable <name>`: Disable a discovered plugin for the next run.

Once a plugin is loaded, its registered functionalities become available. For instance, a plugin registering a new data reader for the `.mydata` extension will make that format loadable by any command that uses `data_handler.read_data`. Similarly, a plugin registering a feature named `my_plugin_feature` will make it available via `sygnals features extract --feature my_plugin_feature`.

### Developing Plugins

For developers, Sygnals provides a clear API (`sygnals.plugins.api.SygnalsPluginBase`) and registration hooks. Plugins are Python packages/modules that:

1.  Contain a `plugin.toml` manifest file in their root directory with metadata (name, version, Sygnals API compatibility, entry point).
2.  Implement a main class inheriting from `sygnals.plugins.api.SygnalsPluginBase`.
3.  Implement `register_*` methods (e.g., `register_filters`, `register_feature_extractors`, `register_data_readers`, `register_cli_commands`) to register custom callables/classes with the `sygnals.plugins.api.PluginRegistry`.
4.  Optionally implement `setup()` (called during loading with global config) and `teardown()` (called on application exit) lifecycle hooks.

The `sygnals plugin scaffold <name>` command helps generate the basic file structure for a new plugin.

**Key Plugin Extension Points (via `PluginRegistry` methods):**

*   `add_filter`: Register new signal filtering functions.
*   `add_transform`: Register new signal transformation functions.
*   `add_feature`: Register new frame-level or signal-level feature calculation functions.
*   `add_effect`: Register new audio processing effects.
*   `add_augmenter`: Register new data augmentation functions.
*   `add_reader`: **Add support for reading new file formats** by registering functions for specific file extensions (e.g., `.parquet`, `.hdf5`). Function must accept `path` and `**kwargs` and return `ReadResult`.
*   `add_writer`: **Add support for writing to new file formats** by registering functions for specific file extensions. Function must accept `data`, `path`, and `**kwargs` and return `None`.
*   `add_cli_command`: Add custom top-level or grouped CLI commands (requires `click` in the plugin's dependencies).

Plugins are loaded based on their `sygnals_api` requirement matching the core Sygnals version, ensuring compatibility. The loader handles managing the enabled/disabled state persistently.

See the `docs/developing_plugins.md` guide for detailed instructions on creating and registering custom plugin functionality.

---

## Configuration

Sygnals uses a flexible, layered configuration system based on TOML files, environment variables, and command-line arguments. Settings are loaded and applied in order of precedence (highest first):

1.  **Command-Line Arguments:** Flags and options provided directly when running `sygnals` commands (e.g., `--output output.csv`).
2.  **Environment Variables:** Variables prefixed with `SYGNALS_` (e.g., `SYGNALS_DEFAULTS_DEFAULT_SAMPLE_RATE=48000`). Nested configuration sections are represented by underscores (e.g., `SYGNALS_PATHS_OUTPUT_DIR=/tmp/sygnals`).
3.  **User Config File:** `sygnals.toml` located in the user's configuration directory (e.g., `~/.config/sygnals/sygnals.toml` on Linux/macOS). This path is determined by the `appdirs` library or similar platform-specific conventions.
4.  **Project Config File:** `sygnals.toml` located in the current working directory (`./sygnals.toml`).
5.  **Internal Defaults:** Hardcoded default values within the application defined using Pydantic models (`sygnals.config.models`).

You can create `sygnals.toml` files to customize default behaviors, paths, processing parameters (like FFT window size, filter order, or specific feature parameters), and logging settings. An example `sygnals.toml` might look like this:

```toml
# Example sygnals.toml

[defaults]
default_sample_rate = 44100
default_fft_window = "blackman"

[paths]
plugin_dir = "~/.config/sygnals/plugins"
output_dir = "./processed_output"

[parameters.mfcc]
n_mfcc = 20
hop_length = 256

[logging]
log_file_enabled = true
log_level_file = "INFO"
log_directory = "./sygnals_logs" # Log directory can be set here or in [paths]
log_filename_template = "{timestamp:%Y%m%d_%H%M%S}.log"
```

This system allows you to define global settings in your user config, override them for specific projects in a project config, and make temporary adjustments via environment variables or CLI flags. Plugin-specific configuration can be added in a `[plugins.<plugin_name>]` section if the plugin supports it.

See the `docs/configuration_guide.md` guide for a detailed breakdown of all available configuration options.

---

## Logging System

Sygnals includes a configurable logging system to provide feedback during execution and assist with debugging. It uses Python's standard `logging` module, enhanced with Rich for console output and configurable file logging.

Logging behavior is controlled primarily via the `[logging]` section in `sygnals.toml` and command-line verbosity flags (`-v`, `-vv`, `-q`).

*   **Console Output:** Controlled by the `--verbosity` flag (or `-v`, `-vv`, `-q`).
    *   Default (0 verbosity): Shows `WARNING` and `ERROR` messages.
    *   `-v` (1 verbosity): Also shows `INFO` messages.
    *   `-vv` (2 verbosity): Also shows `DEBUG` messages (most detailed).
    *   `-q` (-1 verbosity): Suppresses all console output except critical errors.
    Console output uses Rich for formatting.
*   **File Logging:** Enabled/disabled via the `log_file_enabled` setting in `sygnals.toml`. Log files are written to the directory specified in `[paths].log_directory` using a configurable filename template (`log_filename_template`). The minimum logging level for files (`log_level_file`) can be set independently from the console level, allowing for detailed debug logs to a file while keeping the console clean. The format of log entries is also configurable (`log_format`).

Log messages use hierarchical names (e.g., `sygnals.core.dsp`, `sygnals.cli.features_cmd`), which can be useful for filtering logs if needed.

---

## Code Quality and Testing

Sygnals is developed with a strong emphasis on code quality and reliability. The project incorporates:

*   **Linting and Formatting:** Code style is enforced using tools like `Black` and `ruff` via pre-commit hooks and Continuous Integration (CI) checks.
*   **Type Hinting:** Comprehensive type hints are used throughout the codebase, with static analysis performed by `mypy` in CI.
*   **Error Handling:** Specific and informative exceptions are implemented to provide helpful feedback to users and developers.
*   **Modularity:** The codebase is structured into modular components (e.g., `core/dsp`, `core/audio`, `plugins/`) to enhance maintainability and separation of concerns.
*   **Testing:** A comprehensive testing framework is in place (`pytest`), including:
    *   Unit tests for core algorithms and functions across various modules (`tests/test_*.py`).
    *   Integration tests for common CLI workflows (`tests/test_cli_*.py`).
    *   Tests for the plugin system (`tests/test_plugin_system.py`).
    *   Continuous Integration (CI) pipelines automatically run tests on commits and pull requests to prevent regressions.

This commitment ensures that Sygnals is not only powerful but also stable, maintainable, and provides a solid foundation for future development and contributions.

---

## Project Directory Structure

The Sygnals codebase is organized to provide logical separation of concerns and enhance maintainability. The core structure (simplified) is as follows:

```
sygnals_repo/
├── sygnals/                  # Main Python package source
│   ├── __init__.py           # Package initialization
│   ├── version.py          # Package version (__version__)
│   ├── cli/                  # CLI commands and entry points
│   │   ├── main.py           # Main CLI application entry point (Click)
│   │   ├── base_cmd.py       # Base command setup (config, logging, plugins)
│   │   ├── augment_cmd.py    # Data augmentation command group
│   │   ├── features_cmd.py   # Feature extraction and transform command group
│   │   ├── plugin_cmd.py     # Plugin management command group
│   │   ├── save_cmd.py       # Data saving and dataset assembly command group
│   │   └── segment_cmd.py    # Signal segmentation command group
│   ├── config/               # Configuration loading and models (Pydantic)
│   │   ├── loaders.py
│   │   └── models.py
│   ├── core/                 # Core processing logic modules
│   │   ├── __init__.py
│   │   ├── audio/            # Audio-specific modules (effects, io, features)
│   │   ├── data_handler.py   # General data I/O and handling
│   │   ├── dsp.py              # Core DSP algorithms
│   │   ├── features/         # General feature extraction logic and management
│   │   ├── filters.py        # Digital filter design and application
│   │   ├── ml_utils/         # ML-specific utilities (scaling, formatters)
│   │   ├── plugin_manager.py # Legacy plugin discovery (likely superseded by plugins/loader.py)
│   │   ├── batch_processor.py # Batch processing logic (may be refactored)
│   │   ├── custom_exec.py    # Safe expression evaluation
│   │   └── storage.py        # Database storage utilities (SQLite)
│   ├── plugins/              # Plugin system API, loader, and scaffold logic
│   │   ├── api.py            # Plugin base class and registry
│   │   ├── loader.py         # Plugin discovery and loading
│   │   └── scaffold.py       # Plugin template generation
│   └── utils/                # General utilities
│       ├── logging_config.py # Logging setup
│       └── visualizations.py # Visualization generation functions
├── docs/                     # Documentation source files (Markdown, potentially Sphinx/MkDocs config)
│   ├── index.md              # Overview/Homepage
│   ├── usage.md              # General Usage Guide (planned content - *not provided*)
│   ├── developing_plugins.md # Plugin Development Guide (*provided*)
│   ├── configuration_guide.md# Configuration Guide (*provided*)
│   ├── contributing.md       # Contribution Guide (planned content - *not provided*)
│   └── ...                   # Other guides, API reference (auto-generated)
├── examples/                 # Example usage scripts/notebooks
├── plugins_contrib/          # Example directory for user-contributed/local plugins
│   ├── sygnals-hdf5/         # Example HDF5 plugin (*provided*)
│   └── sygnals-parquet/      # Example Parquet plugin (*provided*)
├── tests/                    # Unit and integration tests
│   ├── test_*.py             # Individual test files (*provided*)
│   └── test_cli_*.py         # Individual CLI test files (*provided*)
├── sygnals.toml.example      # Example configuration file (*not provided*)
├── pyproject.toml            # Project build metadata and dependencies (PEP 518/621) (*provided*)
└── README.md                 # This file (the one you are reading)
```

This structure facilitates development, testing, and understanding the separation between core logic, CLI interface, utilities, and the extensible plugin system.

---

## Contributing

We welcome contributions to Sygnals! If you are interested in contributing, please refer to the `docs/contributing.md` guide (located in the `docs/` directory) for detailed instructions on:

*   Setting up a development environment.
*   Coding standards and style guides.
*   Running tests and ensuring quality checks pass.
*   Submitting pull requests.
*   Code of conduct.

Contributions can include bug fixes, new features, documentation improvements, or examples.

---

## License

Sygnals is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). See the `LICENSE` file in the repository root for the full text (if available).

---

