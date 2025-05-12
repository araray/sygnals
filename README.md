
# Sygnals: A CLI Toolkit for Signal and Audio Processing in Data Science Workflows

**Sygnals** is a command-line interface (CLI) tool designed to support **data science and machine learning workflows** by providing robust capabilities for pre-processing, analyzing, transforming, and featurizing time-series and audio data. It emphasizes compatibility with common data science libraries and frameworks (like Pandas and NumPy) and offers an extensible plugin architecture.

This toolkit facilitates the preparation of raw signal and audio data for subsequent analysis, feature engineering, and ingestion into pipelines for **training Artificial Intelligence (AI) or Neural Network (NN) models**.

---

## Features

Sygnals provides a comprehensive suite of tools, accessible via the command line, tailored for handling and preparing complex signal and audio data:

### Core Capabilities
*   **Robust Data Loading & Saving:** Load and save data in various formats including CSV, JSON, NPZ, WAV, FLAC, and Ogg. Support for additional formats like Parquet and HDF5 is available via plugins (e.g., installable via `pip install sygnals-parquet`).
*   **Layered Configuration:** Powerful configuration management via CLI arguments, environment variables, and TOML config files (`sygnals.toml`).
*   **Detailed Logging:** Configurable logging system with console output (using Rich) and persistent file logging.

### Digital Signal Processing (DSP) & Transforms
*   **Time-Frequency Analysis:** Compute Fast Fourier Transform (FFT), Inverse FFT (IFFT), Short-Time Fourier Transform (STFT), and Constant-Q Transform (CQT).
*   **Wavelet Transforms:** Perform Discrete Wavelet Transform (DWT).
*   **Other Transforms:** Calculate Hilbert Transform (for analytic signal) and perform basic Numerical Laplace Transform.
*   **Analysis Techniques:** Compute Autocorrelation, Cross-correlation, and Power Spectral Density (using Periodogram and Welch's methods).
*   **Filtering:** Design and apply Butterworth IIR filters (low-pass, high-pass, band-pass, band-stop) using numerically stable Second-Order Sections (SOS) with zero-phase filtering.
*   **Windowing:** Apply standard window functions (e.g., Hann, Hamming) to signal frames before spectral analysis.
*   **Amplitude Envelope:** Estimate signal amplitude envelope using Hilbert transform magnitude or frame-based RMS energy.

### Audio Processing & Augmentation
*   **Audio I/O:** Dedicated handling for common audio file formats (WAV, FLAC, Ogg, MP3 read; WAV, FLAC, Ogg write). Supports resampling during load.
*   **Basic Metrics:** Calculate global audio metrics like duration, RMS energy, and peak amplitude.
*   **Segmentation (`sygnals segment`):** Divide audio into segments based on fixed length. Underlying utilities for silence and event-based segmentation are also available in the core library.
*   **Effects & Augmentation (`sygnals augment`):** Apply audio effects and data augmentation techniques: Gain adjustment, basic Spectral Noise Reduction, HPSS-based Transient Shaping, Mid/Side Stereo Widening, Tremolo, Chorus, Flanger, Convolution Reverb (using generated IR), Pitch Shifting, and Time Stretching.

### Feature Engineering (`sygnals features`)
*   **Feature Extraction (`sygnals features extract`):** Extract a wide range of frame-level features from audio signals including:
    *   **Time Domain:** Mean Amplitude, Standard Deviation, Skewness, Kurtosis, Peak Amplitude, Crest Factor, Signal Entropy.
    *   **Frequency Domain:** Spectral Centroid, Spectral Bandwidth, Spectral Flatness, Spectral Rolloff, Dominant Frequency, Spectral Contrast.
    *   **Cepstral:** Mel-Frequency Cepstral Coefficients (MFCCs).
    *   **Audio Specific:** Zero Crossing Rate, RMS Energy (frame-based), Onset Detection, Pitch (Fundamental Frequency).
    *   *Approximate* voice quality features: Harmonic-to-Noise Ratio (HNR), Jitter, and Shimmer. *Note: These are simplified approximations based on frame-level analysis.*
*   **Feature Transformation (`sygnals features transform scale`):** Apply transformations to extracted feature data, such as Standard, MinMax, and Robust Scaling. Requires the `scikit-learn` Python package (`pip install scikit-learn`).
*   **Dataset Assembly (`sygnals save dataset`):** Format extracted features or processed data into structures suitable for ML models. Supported assembly methods include creating vectors per segment, stacking sequences, and formatting 2D feature maps as image-like arrays.

### Visualization
*   **Visualization Utilities:** Underlying plotting utilities exist (`sygnals.utils.visualizations`) to generate plots for analysis (Waveforms, Spectrograms, FFT Magnitude and Phase Spectra, Wavelet Scalograms). Command-line interfaces to generate these plots directly are not yet fully implemented in the core CLI but can be used programmatically or integrated into scripts.

### Plugin System (`sygnals plugin`)
*   **Extensibility:** Easily extend Sygnals by writing custom Python plugins for new filters, transforms, features, effects, visualizations, data handlers (readers/writers), and potentially CLI commands.
*   **Management:** CLI commands to list discovered plugins, enable/disable them, scaffold new plugin projects, and install/uninstall plugins.

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

The implemented top-level command groups are: `segment`, `features`, `augment`, `save`, `plugin`.

```bash
# Get overall help
sygnals --help

# Get help for a specific command group (e.g., features)
sygnals features --help

# Get help for a specific subcommand (e.g., features transform scale)
sygnals features transform scale --help
```

Here are examples of common workflows:

### Signal Segmentation (`sygnals segment`)

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

### Feature Engineering (`sygnals features`)

Extracts and transforms signal features.

#### `sygnals features extract <INPUT_FILE>`

Extracts features from a signal or audio file. Requires audio input for most features.

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file (typically audio for feature extraction).
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path (e.g., `features.csv`, `features.npz`). The format is inferred from the extension. Output contains features per frame, plus a 'time' column/array.
    *   `-f, --feature NAME`: **Required (can be repeated).** Name of the feature(s) to extract. Use `all` to extract most known standard features. Supported features are listed in the "Implemented DSP, Audio Processing, and Features" section. Features like `mfcc` or `spectral_contrast` will produce multiple output columns (e.g., `mfcc_0`, `mfcc_1`, ..., `contrast_band_0`, ...).
    *   `--frame-length INTEGER`: Analysis frame length in samples. Used for framing time-domain features and as default `n_fft` for spectral features. Default: `2048`.
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

### Data Augmentation (`sygnals augment`)

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

#### `sygnals save`

Saves processed data or assembles datasets.

##### `sygnals save dataset <INPUT_FILE>`

Saves processed data, potentially applying formatting for ML model ingestion. The input file typically contains features or processed data (e.g., from `features extract`, `features transform scale`).

*   **Arguments:**
    *   `INPUT_FILE`: Path to the input file.
*   **Options:**
    *   `-o, --output FILE`: **Required.** Output file path for the dataset. Format is inferred from the extension, but can be overridden by `--format`.
    *   `--format EXT`: Explicitly set output format (e.g., `npz`, `csv`), overriding the output file extension. Supports core formats and plugin-added writers.
    *   `--assembly-method METHOD`: Method to assemble features into a dataset structure. Default: `none`.
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

#### `sygnals plugin`

Manage Sygnals plugins.

##### `sygnals plugin list`

Lists all discovered plugins, their version, status (loaded, disabled, incompatible), and source (local or entry point).

*   **Example:**

    ```bash
    sygnals plugin list
    ```

##### `sygnals plugin enable <NAME>`

Enables a specific plugin by its name. The change takes effect on the next Sygnals run.

*   **Arguments:**
    *   `NAME`: The unique name of the plugin to enable.

*   **Example:**

    ```bash
    sygnals plugin enable my-custom-feature-plugin
    ```

##### `sygnals plugin disable <NAME>`

Disables a specific plugin by its name. The change takes effect on the next Sygnals run.

*   **Arguments:**
    *   `NAME`: The unique name of the plugin to disable.

*   **Example:**

    ```bash
    sygnals plugin disable experimental-filter
    ```

##### `sygnals plugin scaffold <NAME>`

Generates a template directory structure for creating a new plugin.

*   **Arguments:**
    *   `NAME`: The desired name for the new plugin.
*   **Options:**
    *   `--dest PATH`: Destination directory for the plugin template. Default: `.`.

*   **Example:**

    ```bash
    sygnals plugin scaffold my-new-plugin --dest ./plugins_dev/
    ```

##### `sygnals plugin install <SOURCE>`

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

##### `sygnals plugin uninstall <NAME>`

Uninstalls a plugin by its name. Handles plugins installed locally or via pip (entry points).

*   **Arguments:**
    *   `NAME`: The unique name of the plugin to uninstall.
*   **Options:**
    *   `-y, --yes`: Do not ask for confirmation before uninstalling.

*   **Example:** Uninstall the plugin named `sygnals-awesome-plugin`.

    ```bash
    sygnals plugin uninstall sygnals-awesome-plugin
    ```

---

## Implemented DSP, Audio Processing, and Features

Sygnals provides access to a variety of signal processing algorithms and features through its core Python modules. Many of these are utilized by the CLI commands, while others are available for programmatic use.

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

Sygnals features a flexible plugin system allowing users and developers to extend its functionality without modifying the core codebase. Plugins can add new filters, transforms, feature extractors, audio effects, data augmenters, data readers/writers, and even custom CLI commands.

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

See the `docs/configuration_guide.md` (or relevant documentation) for a detailed breakdown of all available configuration options.

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
│   │   ├── dsp/              # Core DSP algorithms
│   │   ├── features/         # General feature extraction logic and management
│   │   ├── filters/          # Digital filter design and application
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
│   ├── usage.md              # General Usage Guide (planned content)
│   ├── developing_plugins.md # Plugin Development Guide
│   ├── configuration_guide.md# Configuration Guide (planned content)
│   ├── contributing.md       # Contribution Guide (planned content)
│   └── ...                   # Other guides, API reference (auto-generated)
├── examples/                 # Example usage scripts/notebooks
├── plugins_contrib/          # Example directory for user-contributed/local plugins
│   ├── sygnals-hdf5/         # Example HDF5 plugin
│   └── sygnals-parquet/      # Example Parquet plugin
├── tests/                    # Unit and integration tests
│   ├── test_*.py             # Individual test files
│   └── test_cli_*.py
├── sygnals.toml.example      # Example configuration file
├── pyproject.toml            # Project build metadata and dependencies (PEP 518/621)
└── README.md                 # This file (the one you are reading)
```

This structure facilitates development, testing, and understanding the separation between core logic, CLI interface, utilities, and the extensible plugin system.

---

## Contributing

We welcome contributions to Sygnals! If you are interested in contributing, please refer to the `CONTRIBUTING.md` guide (located in the `docs/` directory) for detailed instructions on:

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
