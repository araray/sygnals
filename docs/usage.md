# Sygnals Usage Guide

**Sygnals** is a versatile command-line interface (CLI) for signal and audio processing, tailored for data science workflows. It allows you to load, analyze, manipulate, transform, and save time-series and audio data. This guide covers common workflows and command usage.

---

## 1. Getting Started

### Installation

If you haven't already, install Sygnals (ensure you have Python >= 3.11):

```bash
# Clone the repository (if you haven't)
# git clone [https://github.com/araray/sygnals.git](https://github.com/araray/sygnals.git)
# cd sygnals

# Install using pip (preferably in a virtual environment)
pip install .
# Or for development:
# pip install -e .[dev]
```

### Basic Help

To see the main help message and available commands:

```bash
sygnals --help
```

To get help for a specific command (e.g., `segment`):

```bash
sygnals segment --help
```

To get help for a subcommand (e.g., `segment fixed-length`):

```bash
sygnals segment fixed-length --help
```

### Verbosity

Control the amount of information printed to the console:

* `-v`: Verbose mode (shows INFO messages).
* `-vv`: Debug mode (shows DEBUG messages, very detailed).
* `-q`: Quiet mode (suppresses INFO and WARNING messages).

Example:

```bash
sygnals -vv features extract ... # Run feature extraction with debug output
```

---

## 2. Core Concepts

* **Pipeline:** Sygnals commands can often be chained together using standard shell pipes (`|`) where logical, allowing you to build processing pipelines (though complex pipelines might be better managed via configuration or scripting).
* **Data Handling:** Sygnals uses `pandas` for tabular data (CSV, JSON) and `numpy` for numerical data (NPZ, audio waveforms). It aims to handle common formats seamlessly.
* **Configuration:** Sygnals uses a layered configuration system (`sygnals.toml`, environment variables, CLI options). See the `configuration_guide.md` for details.
* **Plugins:** Functionality can be extended via plugins. See `developing_plugins.md`.

---

## 3. Common Workflows & Commands

### 3.1. Loading and Saving Data

Sygnals commands typically take an input file path as an argument and require an output path using the `-o` or `--output` option. The file format is usually inferred from the extension (`.csv`, `.json`, `.npz`, `.wav`, `.flac`, etc.).

Supported input formats include CSV, JSON, NPZ, WAV, FLAC, OGG, MP3.
Supported output formats include CSV, JSON, NPZ, WAV, FLAC, OGG.

### 3.2. Segmentation (`sygnals segment`)

Divides a signal into smaller segments based on specified criteria. This is often a prerequisite for feature extraction in machine learning.

**`segment fixed-length`**: Creates segments of a fixed duration with optional overlap.

* **Purpose:** Useful when you need uniformly sized chunks of the signal, common for STFT-based feature extraction or training models that expect fixed input lengths.
* **Arguments:**
    * `INPUT_FILE`: Path to the input audio file (e.g., `input.wav`).
* **Options:**
    * `-o, --output DIRECTORY`: **Required.** Directory where segment files (e.g., `input_segment_001.wav`, `input_segment_002.wav`, ...) will be saved.
    * `--length SECONDS`: **Required.** The duration of each segment in seconds.
    * `--overlap RATIO`: Overlap between consecutive segments (0.0 for no overlap, 0.5 for 50% overlap, etc.). Default: `0.0`.
    * `--pad / --no-pad`: Whether to pad the last segment with zeros if it's shorter than `--length`. Default: `--pad`.
    * `--min-length SECONDS`: If set (and `--pad` is used), segments shorter than this duration (after potential padding) will be discarded.

**Example:** Segment `speech.wav` into 2-second chunks with 25% overlap, saving them to the `segments/` directory. Discard any trailing segment shorter than 0.5 seconds.

```bash
sygnals segment fixed-length speech.wav \
    --output ./segments/ \
    --length 2.0 \
    --overlap 0.25 \
    --pad \
    --min-length 0.5
```

*(Note: `segment by-silence` and `segment by-event` are planned but currently placeholders).*

### 3.3. Feature Extraction (`sygnals features extract`)

Extracts various features from audio signals. Features quantify characteristics of the signal in different domains (time, frequency, cepstral).

* **Purpose:** Generate numerical representations of audio suitable for analysis or machine learning models.
* **Arguments:**
    * `INPUT_FILE`: Path to the input audio file.
* **Options:**
    * `-o, --output FILE`: **Required.** Output file path (e.g., `features.csv`, `features.npz`). Format determined by extension.
    * `-f, --feature NAME`: **Required (can be repeated).** Name of the feature(s) to extract (e.g., `rms_energy`, `spectral_centroid`, `mfcc`, `zcr`). Use `--feature all` to extract all known standard features.
    * `--frame-length SAMPLES`: Length of the analysis window in samples. Also used as `n_fft` for spectral features. Default: `2048`.
    * `--hop-length SAMPLES`: Step size between consecutive analysis windows in samples. Default: `512`.
    * *(Future: Options to pass specific parameters to individual features, e.g., `--param mfcc.n_mfcc=20`)*

**Example 1:** Extract RMS energy and Zero Crossing Rate, saving to CSV.

```bash
sygnals features extract input.wav \
    -o energy_zcr.csv \
    --feature rms_energy \
    --feature zero_crossing_rate \
    --frame-length 1024 \
    --hop-length 512
```

**Example 2:** Extract 13 MFCCs and Spectral Centroid, saving to NPZ.

```bash
sygnals features extract dialogue.wav \
    -o mfcc_centroid.npz \
    --feature mfcc \
    --feature spectral_centroid
    # Uses default frame/hop lengths (2048/512)
    # Uses default n_mfcc=13 for mfcc feature
```

### 3.4. Feature Transformation (`sygnals features transform`)

Applies transformations to existing feature data, such as scaling.

**`transform scale`**: Applies feature scaling (normalization) to standardize the range of features.

* **Purpose:** Scaling is often crucial for machine learning algorithms that are sensitive to feature ranges (e.g., SVMs, KNN, Neural Networks).
* **Arguments:**
    * `INPUT_FILE`: Path to the input feature file (CSV or NPZ containing feature data).
* **Options:**
    * `-o, --output FILE`: **Required.** Output file path for the scaled features.
    * `--scaler TYPE`: Type of scaler: `standard` (zero mean, unit variance), `minmax` (scales to a range, typically [0, 1]), `robust` (uses median and IQR, less sensitive to outliers). Default: `standard`.
    * *(Future: Options for scaler-specific parameters, e.g., `minmax` range)*

**Example:** Apply standard scaling (Z-score normalization) to features stored in `features.csv`.

```bash
sygnals features transform scale features.csv \
    -o features_scaled.csv \
    --scaler standard
```

**Example:** Apply MinMax scaling to scale features in `features.npz` to the range [0, 1].

```bash
sygnals features transform scale features.npz \
    -o features_minmax.npz \
    --scaler minmax
```

### 3.5. Data Augmentation (`sygnals augment`)

Creates modified versions of audio files by applying transformations like adding noise, shifting pitch, or stretching time.

* **Purpose:** Increase the size and diversity of training datasets for machine learning, potentially improving model robustness and generalization.
* **Arguments:**
    * `INPUT_FILE`: Path to the input audio file.
* **Options:**
    * `-o, --output FILE`: **Required.** Output file path for the augmented audio.

**Subcommands:**

* **`add-noise`**: Adds noise to the signal.
    * `--snr DB`: **Required.** Target Signal-to-Noise Ratio in dB (lower means more noise).
    * `--noise-type TYPE`: `gaussian` (white), `pink`, `brown`. Default: `gaussian`. (Pink/Brown are placeholders).
    * `--seed INT`: Optional random seed for reproducibility.
* **`pitch-shift`**: Changes the pitch without changing duration.
    * `--steps STEPS`: **Required.** Number of semitones to shift (positive or negative, can be fractional).
    * `--bins-per-octave INT`: Defaults to 12.
* **`time-stretch`**: Changes duration without changing pitch.
    * `--rate RATE`: **Required.** Stretch factor (>1 speeds up/shortens, <1 slows down/lengthens).

**Example 1:** Create a noisy version of `clean.wav` with 10dB SNR.

```bash
sygnals augment add-noise clean.wav -o noisy_10db.wav --snr 10.0
```

**Example 2:** Create a version of `voice.wav` shifted down by 1.5 semitones.

```bash
sygnals augment pitch-shift voice.wav -o voice_down.wav --steps -1.5
```

**Example 3:** Create a version of `music.wav` slowed down to 90% of original speed.

```bash
sygnals augment time-stretch music.wav -o music_slow.wav --rate 0.9
```

### 3.6. Saving Datasets (`sygnals save dataset`)

Saves processed data, potentially applying formatting for ML model ingestion.

* **Purpose:** Assemble processed features or signals into a final dataset file.
* **Arguments:**
    * `INPUT_FILE`: Path to the input file containing processed data (e.g., features.csv, features.npz).
* **Options:**
    * `-o, --output FILE`: **Required.** Output file path for the dataset.
    * `--format EXT`: Explicitly set output format (e.g., `csv`, `npz`), overriding the extension in `--output`.
    * `--assembly-method METHOD`: How to structure the output dataset. Default: `none`.
        * `none`: Saves the input data directly without restructuring.
        * `vectors` (Placeholder): Aggregate features per segment into vectors.
        * `sequences` (Placeholder): Format features as sequences (e.g., for RNNs).
        * `image` (Placeholder): Format features (e.g., spectrogram) as image-like arrays (e.g., for CNNs).

**Example 1:** Simply save the processed features from `scaled_features.csv` to a new file `final_dataset.csv` (no restructuring).

```bash
sygnals save dataset scaled_features.csv -o final_dataset.csv --assembly-method none
```

**Example 2:** Save features from `features.npz` to a CSV file, overriding the format.

```bash
sygnals save dataset features.npz -o dataset_from_npz.csv --format csv
```

*(Note: Assembly methods other than 'none' are currently placeholders and will just save the input data directly with a warning).*

### 3.7. Plugin Management (`sygnals plugin`)

Manage external plugins that extend Sygnals functionality.

* `plugin list`: Show discovered plugins and their status (loaded, disabled, incompatible).
* `plugin enable NAME`: Enable a discovered plugin for the next run.
* `plugin disable NAME`: Disable a discovered plugin.
* `plugin scaffold NAME`: Generate a template directory structure for creating a new plugin.
* `plugin install SOURCE`: Install a plugin from PyPI or a local directory.
* `plugin uninstall NAME`: Uninstall a plugin.

**Example:** List all available plugins.

```bash
sygnals plugin list
```

**Example:** Disable a plugin named `my-custom-filter`.

```bash
sygnals plugin disable my-custom-filter
```

**Example:** Create a template for a new plugin named `spectral-flux`.

```bash
sygnals plugin scaffold spectral-flux --dest ./my_sygnals_plugins/
```

See `developing_plugins.md` and `adding_new_plugin.md` for more details.

---

## 4. Machine Learning Data Preparation Example

Let's illustrate a common workflow for preparing audio data for a classification task:

**Goal:** Extract MFCC features from 1-second segments of audio files in a directory, scale them, and save as a single dataset file.

**Assumptions:**
* Audio files are in `./audio_files/`.
* We want segments saved temporarily in `./temp_segments/`.
* Features extracted to `./temp_features/`.
* Scaled features saved to `./temp_scaled/`.
* Final dataset saved as `speech_features_dataset.npz`.

**Steps:**

1.  **Segment each audio file:**
    ```bash
    # Create directories first
    mkdir -p ./temp_segments/ ./temp_features/ ./temp_scaled/
    
    # Loop through audio files (example for bash)
    for f in ./audio_files/*.wav; do
      echo "Segmenting $f..."
      # Create subdirectory for each original file's segments
      fname=$(basename "$f" .wav)
      mkdir -p "./temp_segments/$fname"
      sygnals segment fixed-length "$f" \
          -o "./temp_segments/$fname/" \
          --length 1.0 --overlap 0.0 --no-pad # 1-sec segments, no overlap/padding
    done
    ```

2.  **Extract MFCCs from each segment:**
    ```bash
    echo "Extracting features..."
    # Loop through segment directories
    find ./temp_segments/ -name '*.wav' | while read seg_file; do
      # Construct output path for features
      out_name=$(basename "$seg_file" .wav)_mfcc.csv
      out_dir="./temp_features/$(basename "$(dirname "$seg_file")")"
      mkdir -p "$out_dir"
      out_path="$out_dir/$out_name"
    
      sygnals features extract "$seg_file" \
          -o "$out_path" \
          --feature mfcc # Use default n_mfcc=13, frame/hop
    done
    ```
    *(Note: This saves features for each segment separately. A more advanced workflow might combine features during extraction or use batch processing if available).*

3.  **Scale the features (example for one file, loop similarly):**
    ```bash
    echo "Scaling features..."
    # Example for one feature file (loop through all in temp_features)
    sygnals features transform scale ./temp_features/file1/file1_segment_001_mfcc.csv \
        -o ./temp_scaled/file1_segment_001_mfcc_scaled.csv \
        --scaler standard
    # Add loop here for all files in temp_features
    ```

4.  **Assemble the dataset (Placeholder):**
    *This step requires more advanced logic not yet fully implemented in `save dataset`.* You would typically write a script to:
    * Load all scaled feature files (`.csv` or `.npz`).
    * Potentially aggregate features per segment (e.g., take the mean MFCC vector).
    * Combine features from all segments/files into a single array or DataFrame.
    * Add corresponding labels (if available).
    * Save the final combined data.

    **Using `save dataset` (current placeholder functionality):**
    If you had combined scaled features into one file (e.g., `all_scaled_features.csv`), you could just "save" it:
    ```bash
    # Assuming all_scaled_features.csv exists
    # sygnals save dataset all_scaled_features.csv -o speech_features_dataset.npz --format npz
    ```
