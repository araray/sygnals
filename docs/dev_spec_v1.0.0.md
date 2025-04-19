## Project Specification: Sygnals v1.0.0

### 1. Introduction

**Sygnals v1.0.0** is designed as a powerful command-line toolkit for signal and audio processing, specifically tailored to support **data science workflows**. It facilitates the manipulation, transformation, feature extraction, and preparation of signal/audio data for ingestion into other data analysis platforms or for **training Artificial Intelligence (AI) / Neural Network (NN) models**. Building upon the initial `sygnals` capabilities, this version emphasizes robust data handling, a comprehensive suite of feature engineering tools, flexible dataset creation capabilities, interoperability with common data science libraries and frameworks, high-quality code, and **extensive documentation**.

**Goals:**

- Provide a versatile CLI for pre-processing, analyzing, and featurizing signal/audio data for machine learning applications.
- Enable the creation of structured, analysis-ready datasets from raw signal sources.
- Offer a wide array of configurable DSP algorithms, transforms, and feature extraction techniques relevant to data science.
- Support common data formats used in data science and ML pipelines (e.g., CSV, NPZ, Parquet).
- Maintain extensibility through a refined plugin system for custom processing steps.
- Ensure reproducibility through robust configuration management and pipeline definition capabilities.
- **Deliver exceptional developer and user experience through thorough inline code documentation and comprehensive external guides.**

### 2. Core Concepts

- **Data Science Pipeline:** Extends the signal processing pipeline concept to explicitly support stages common in ML data preparation: Load -> Pre-process (Filter, Resample) -> Segment -> Extract Features -> Transform Features (Scale, Normalize) -> Augment -> Format Dataset -> Save.
- **Feature Engineering:** The process of using domain knowledge and signal processing techniques to create features (measurable properties or characteristics) from raw signal data that make machine learning algorithms work better. Sygnals provides tools for both standard and custom feature calculation.
- **Dataset Assembly:** Capabilities to organize extracted features and processed signals into structured formats (e.g., feature vectors per segment, sequences of features) suitable for input to ML models.
- **Interoperability:** Emphasis on using data formats and representations (like NumPy arrays, Pandas DataFrames) easily consumable by libraries like Scikit-learn, TensorFlow, PyTorch, Pandas, etc.
- **Signal/Audio Representation:** Handling data primarily as numerical arrays (NumPy) or structured tables (Pandas DataFrames) for compatibility with the data science ecosystem.
- **DSP Core:** Modules providing implementations of various DSP algorithms (filters, transforms, analysis techniques).
- **Audio Core:** Specific functionalities for audio processing, including effects, metrics, and format handling.
- **Visualization Engine:** Generates various plots and visual representations of signals and their analyses.
- **Plugin Architecture:** Allows users to extend functionality by adding custom filters, transforms, analysis methods, or even entire commands.
- **Configuration Management:** Layered system (defaults, file, environment variables, CLI args) for managing processing parameters, default behaviors, and external service integrations (if any).

### 3. Features

#### 3.1. Code Quality & Maintainability Enhancements

- **Linting & Formatting:** Integrate and enforce code style using tools like `Black` and `Flake8` via pre-commit hooks and CI checks.
- **Type Hinting:** Implement comprehensive type hinting across the codebase and use `mypy` for static analysis.
- **Error Handling:** Implement specific, informative exception classes. Improve error reporting in the CLI, providing context and potential solutions. Refine error handling in file I/O and core processing functions.
- **Modularity Review:** Refactor large modules like `cli.py` and consolidate overlapping functionalities (e.g., Butterworth filters in `dsp.py` and `filters.py`). Introduce clearer separation of concerns between core logic and CLI presentation.
- **Dependency Management:** Use `pyproject.toml` (with Poetry or PDM) for managing dependencies. Pin critical dependencies or use version ranges cautiously. Regularly review and remove unused dependencies.

#### 3.2. Comprehensive Testing Framework

- **Increased Coverage:** Significantly expand unit test coverage for all core modules (`dsp`, `filters`, `transforms`, `audio_handler`, `data_handler`, `visualizations`, `plugin_manager`). Add tests for edge cases, parameter variations, and different data types/formats.
- **Integration Tests:** Develop integration tests for common CLI workflows (`analyze`, `filter`, `transform`, `visualize`, `audio effect`, `features extract`, `plugin` commands) ensuring commands work together correctly.
- **Test Fixtures:** Utilize `pytest` fixtures more effectively for setup/teardown (e.g., generating test signals/audio files, creating temporary directories, loading dummy plugins).
- **CI Integration:** Implement Continuous Integration (GitHub Actions, GitLab CI) to automatically run tests, linters, and type checkers on commits and pull requests.
- **Refactor `run_tests.sh`:** Replace the shell script with direct `pytest` execution within the CI pipeline for better control, reporting, and maintainability. Add more specific output validation beyond file existence.

#### 3.3. Enhanced Documentation (Inline and External)

- Inline Documentation:

    - **Docstrings (NumPy/Google Style):** Mandate comprehensive docstrings for *all* public modules, classes, functions, and methods. Docstrings must detail purpose, parameters (including types), return values (including types), raised exceptions, usage examples (`Examples` section), and relevant notes (`Notes` section, e.g., on algorithm complexity or assumptions). Target modules like `dsp.py`, `filters.py`, `transforms.py`, and all new core modules.
    - **Code Comments:** Use clear, concise inline comments to explain complex logic, non-obvious sections, workarounds, or important algorithms within function/method bodies. Focus on the *why* not just the *what*.

- External Documentation (Generated via Sphinx/MkDocs from Docstrings & Markdown Files):

    - **`README.md`:** A high-level overview of the project: what it is, key features, quick start installation, basic usage example, links to full documentation, contribution overview, license. Keep it concise and engaging.

    - `usage.md` (or equivalent section):

         Gorgeous, in-depth, explanatory guide covering common workflows with practical examples. Should cover:

        - Basic file loading, analysis, and visualization.

        - Applying various filters and transforms.

        - Audio processing (effects, metrics).

        - Detailed ML Data Preparation Examples:

             Step-by-step guides for tasks like:

            - Extracting features (MFCCs, spectral features) from audio segments.
            - Creating feature vectors for classification/regression.
            - Generating spectrogram images for CNN input.
            - Applying data augmentation techniques.

        - Using the plugin system (finding, installing, using plugins).

        - Piping commands together.

    - **`configuration_guide.md` (or equivalent section):** Detailed explanation of the layered configuration system (CLI > Env > User Config > Project Config > Defaults). Provide an exhaustive reference for *every* section and parameter in `sygnals.toml`, explaining its purpose, possible values, default, rationale for default, impact, and corresponding environment variable/CLI flag (modeled after the Semantiscan example).

    - Plugin Documentation:

        - **`adding_new_plugin.md` (or equivalent section):** User-focused guide on how to find, install (`sygnals plugin install`), enable (`sygnals plugin enable`), disable, list, and uninstall plugins using the CLI commands. Explain the concepts of entry-point vs. local plugins.
        - **`developing_plugin.md` (or equivalent section):** Developer-focused guide detailing the plugin architecture, the `SygnalsPluginBase` API, registration hooks (`register_*` methods), the plugin manifest (`plugin.toml`) structure and fields, versioning/compatibility (`sygnals_api`), how to use the scaffold tool (`sygnals plugin scaffold`), best practices, and testing strategies for plugins. Include a worked example of creating a simple plugin (e.g., a custom filter or feature).

    - **API Reference:** Automatically generated reference documentation from the inline docstrings for all public modules, classes, and functions.

    - **Contribution Guide:** Instructions for setting up a development environment, running tests, coding standards, submitting pull requests, and the code of conduct.

#### 3.4. Flexible Configuration System (Modeled after Semantiscan examples)

- Layered Configuration:

     Settings are loaded and applied in the following order of precedence (highest precedence first):

    1. **Command-Line Arguments:** Flags like `--output-format`, `--filter-order`, `--log-level-file`.
    2. **Environment Variables:** Variables like `SYGNALS_DEFAULT_SAMPLE_RATE`, `SYGNALS_LOG_DIRECTORY`.
    3. **User Config File:** `sygnals.toml` in `~/.config/sygnals/`.
    4. **Project Config File:** `sygnals.toml` in the current working directory.
    5. **Internal Defaults:** Hardcoded defaults within the application.

- Configuration File (`sygnals.toml`):

     Uses TOML format. Key sections detailed below:

    - `[defaults]`:

        - `default_sample_rate`: Default sample rate assumed for data files if not specified (e.g., `44100`).
        - `default_input_format`: Default format assumed for input files (e.g., `"wav"`, `"csv"`).
        - `default_output_format`: Default format for saving outputs (e.g., `"wav"`, `"csv"`, `"npz"`).
        - `default_fft_window`: Default window function for FFT/STFT (e.g., `"hann"`).
        - `default_filter_order`: Default order for filters like Butterworth (e.g., `5`).
        - *(Add other relevant processing defaults, e.g., default effect parameters)*

    - `[paths]`:

        - `plugin_dir`: Path to local user plugins (e.g., `"~/.config/sygnals/plugins"`).
        - `cache_dir`: Path for temporary or cached files (e.g., `"./.sygnals_cache"`).
        - `output_dir`: Default directory for saving results if `--output` is not a full path (e.g., `"./sygnals_output"`).

    - `[parameters]`:

         Centralized parameters for reusable components:

        - `[parameters.stft]`: `window_size = 1024`, `overlap = 512`.
        - `[parameters.mfcc]`: `n_mfcc = 13`, `n_fft = 2048`, `hop_length = 512`.
        - `[parameters.segmentation.fixed_length]`: `length = 1.0`, `overlap = 0.5`.
        - `[parameters.scaling.standard_scaler]`: `with_mean = true`, `with_std = true`.
        - *(Add other parameter groups for filters, effects, specific features, etc.)*

    - **`[logging]`:** (See Section 3.11 below).

    - `[discovery]`:

         (Optional - primarily for commands scanning directories)

        - `excluded_dirs = ["__pycache__", ".git", "venv", ".venv"]`
        - `excluded_files = [".DS_Store", "*.tmp", "*.swp"]`
        - `use_gitignore = true`

#### 3.5. Expanded DSP Algorithms & Transforms

- Core Transforms:
    - **FFT/IFFT:** Enhanced with windowing (`scipy.signal.get_window`), padding, explicit FFT length control, access to phase information.
    - **STFT (Short-Time Fourier Transform):** Core implementation with configurable window type, length, and overlap. Outputs suitable for feature matrices.
    - **Wavelet Transforms:** Support Continuous Wavelet Transform (CWT), Discrete Wavelet Transform (DWT) (`pywt.wavedec`), explicit wavelet family selection.
    - **CQT (Constant-Q Transform):** For frequency analysis with logarithmic spacing (musical audio).
    - **MFCC (Mel-Frequency Cepstral Coefficients):** Core feature extraction for audio.
    - **Hilbert Transform:** For analytic signal computation (envelope/phase).
    - **(Advanced) Z-Transform:** Numerical Z-transform analysis capabilities (possibly via plugins).
- Filtering:
    - **Filter Types:** Butterworth, Chebyshev (I/II), Elliptic, Bessel, FIR (`scipy.signal.firwin`).
    - **Zero-Phase Filtering:** Option using `scipy.signal.filtfilt`.
    - **Filter Order Control:** Explicit parameter for filter order.
    - **Convolutional Filters:** Apply custom 1D kernels via `scipy.signal.convolve` or `fftconvolve`.
- Analysis Techniques:
    - **Cepstral Analysis:** Real and complex cepstrum.
    - **Correlation:** Autocorrelation, Cross-correlation.
    - **Power Spectral Density (PSD):** Welch's method, Periodogram.
    - **Fundamental Frequency (Pitch) Estimation:** YIN, Cepstrum method, etc.
    - **Envelope Detection:** Amplitude envelope via Hilbert or RMS.
    - **Zero-Crossing Rate.**

#### 3.6. Advanced Audio Effects & Augmentation

- **Standard Effects:** Reverb, Delay, Chorus, Flanger, Phaser, Parametric/Graphic EQ, Tremolo, Vibrato, Distortion/Overdrive.
- **Utility Effects:** Noise Reduction (Spectral Subtraction), Transient Shaping, Stereo Widening, Faders.
- Data Augmentation Effects:
    - Pitch Shifting (configurable steps/range).
    - Time Stretching (configurable rate/range).
    - Adding Noise (Gaussian, Pink, Brown, etc., with SNR control).
    - Clipping (configurable threshold).
    - Gain Adjustment (configurable dB range).
    - (Plugin) Room Impulse Response (RIR) Convolution.
- **Parameterization:** Detailed control via CLI options and configuration.

#### 3.7. Advanced Data Wrangling & Handling (Enhanced for Data Science)

- **Multi-Channel/Variate Handling:** Explicit support in core functions. Channel selection/mixing/splitting.
- **Resampling Methods:** Configurable methods (e.g., `scipy`, `librosa`, `resampy`).
- **Time-Series Alignment:** DTW implementation.
- Signal Segmentation:
    - **Methods:** Threshold crossing, event detection (peaks, onsets via `librosa`), silence detection, fixed/variable-length windows (with overlap).
    - **Output:** Segments as lists of arrays or fed directly into feature extraction.
- Feature Extraction Framework:
    - **Command:** `sygnals features extract ...`
    - Built-in Features:
        - Time Domain: RMS, ZCR, Mean, Std Dev, Skewness, Kurtosis, Peak, Crest Factor, Entropy.
        - Frequency Domain: Spectral Centroid, Bandwidth, Contrast, Flatness, Rolloff, Dominant Frequency.
        - Cepstral: MFCCs.
        - Audio Specific: Pitch, HNR, Jitter/Shimmer.
    - **Configuration:** Specify features, parameters (FFT size, n_mfcc), framing (window, hop).
    - **Output:** Structured formats (CSV, JSON, NPZ) with labels (feature name, time index/segment ID).
- Feature Transformation & Selection:
    - **Scaling/Normalization:** Implement `StandardScaler`, `MinMaxScaler`, `RobustScaler`. Command: `sygnals features transform scale ...`
    - **Selection:** Allow selecting features by name (`--include-features`, `--exclude-features`).
- Data Format Expansion:
    - **Input:** CSV, JSON, WAV, FLAC, Ogg, NPZ, (Plugin) Parquet, HDF5.
    - **Output:** CSV, JSON, NPZ, WAV, FLAC, Ogg, (Plugin) Parquet, HDF5.
- Dataset Assembly:
    - Options via `sygnals save dataset ...` command.
    - Format as feature vectors per segment, sequences of vectors, or image-like arrays (spectrograms). Output compatible with NumPy/Pandas.

#### 3.8. Expanded Visualization Options

- **Core Plots:** Enhanced Waveform (multi-channel), Spectrogram (Mel, Log), FFT/Phase plots.
- **New Plot Types:** Phase Spectrum, Pole-Zero Plot, Wavelet Scalogram, Lissajous Figures, Constellation Diagram, Heatmaps.
- **Feature Visualization:** Plots for feature values over time, histograms, scatter plots.
- **Customization:** Control titles, labels, legends, colors, styles, figure size, resolution via CLI/config. Overlay capability.

#### 3.9. Plugin System (Modeled after Semantiscan Plugin Spec)

1. Overview

    The Plugin System enables third-party and community contributions to extend core sygnals features—such as filters, transforms, feature extractors, visualization types, audio effects, augmentation techniques, and CLI commands—without altering the core codebase.

2. **Objectives**

    - **Extensibility:** Allow drop-in plugins to register new behaviors through well-defined hooks.
    - **Compatibility:** Use semantic versioning to enforce core-API compatibility.
    - **Discoverability:** Provide CLI commands (`sygnals plugin list/install/enable/disable/scaffold`) to manage plugins.
    - **Safety:** Fail fast on incompatible plugins; isolate plugin errors from core.

3. **Architecture**

    Code snippet

    ```
    flowchart LR
      A[sygnals CLI] -->|load| B[PluginLoader]
      B --> C{Entry-Points}
      B --> D{Local Plugins}
      C --> E[PluginRegistry]
      D --> E
      E --> F[Filters]
      E --> G[Transforms]
      E --> H[Feature Extractors]
      E --> I[Visualizations]
      E --> K[Audio Effects]
      E --> L[Augmenters]
      E --> M[CLI Commands]
      F & G & H & I & K & L & M --> J[Core Workflows]
    ```

    - **Entry-Points:** Discovered via `setuptools` (`sygnals.plugins`).
    - **Local Plugins:** Path defined in `[paths].plugin_dir` (e.g., `~/.config/sygnals/plugins/<name>/plugin.toml`).
    - **PluginRegistry:** Central registry for all extension points.
    - **Core Workflows:** CLI execution, Data Loading, Preprocessing, Segmentation, Feature Extraction, Augmentation, Visualization, Saving.

4. Plugin Manifest (plugin.toml)

    Each plugin root must include plugin.toml:

    Ini, TOML

    ```
    # Unique plugin identifier (e.g., "sygnals-kalman-filter")
    name            = "<unique-id>"
    # Plugin version (SemVer, e.g., "0.1.0")
    version         = "<semver>"
    # Core Sygnals API compatibility range (PEP440 specifier, e.g., ">=1.0.0,<2.0.0")
    sygnals_api     = ">=<core-min>,<core-max>"
    # Human-readable summary
    description     = "Short summary of the plugin"
    # <module>:<class> implementing SygnalsPluginBase
    entry_point     = "module.path:ClassName"
    # Optional extra pip installable dependencies required by the plugin
    dependencies    = ["numpy>=1.20.0", "some-other-lib"] # optional
    ```

5. Plugin Interface (sygnals.plugins.api.SygnalsPluginBase)

    All plugins must subclass SygnalsPluginBase and implement registration hooks:

    Python

    ```
    from sygnals.plugins.api import SygnalsPluginBase, PluginRegistry
    
    class MyCustomPlugin(SygnalsPluginBase):
        @property
        def name(self) -> str:
            """Return the unique name specified in plugin.toml."""
            return "<unique-id>"
    
        @property
        def version(self) -> str:
            """Return the version specified in plugin.toml."""
            return "<semver>"
    
        def register_filters(self, registry: PluginRegistry):
            """Register custom filter functions or classes.
    
            Args:
                registry: The central plugin registry instance. Use methods like
                          registry.add_filter(name: str, filter_callable: Callable).
            """
            # Example: registry.add_filter("my_filter_name", my_filter_function_or_class)
            pass
    
        def register_transforms(self, registry: PluginRegistry):
            """Register custom transform functions or classes."""
            # Example: registry.add_transform("my_transform_name", MyTransformProcessor)
            pass
    
        def register_feature_extractors(self, registry: PluginRegistry):
            """Register custom feature extraction functions."""
            # Example: registry.add_feature("my_feature_name", my_feature_extractor_func)
            pass
    
        def register_visualizations(self, registry: PluginRegistry):
            """Register custom visualization functions."""
            # Example: registry.add_visualization("my_plot_name", my_plotting_function)
            pass
    
        def register_audio_effects(self, registry: PluginRegistry):
            """Register custom audio effect processors."""
            # Example: registry.add_effect("my_effect_name", my_audio_effect_processor)
            pass
    
        def register_augmenters(self, registry: PluginRegistry):
            """Register custom data augmentation functions."""
            # Example: registry.add_augmenter("my_augmenter_name", my_data_augment_func)
            pass
    
        # Optional hooks
        def register_cli_commands(self, registry: PluginRegistry):
            """Register custom Click command groups or commands."""
            # Example: registry.add_cli_command(my_custom_click_command_group)
            pass
    
        def setup(self, config: dict):
            """Optional setup hook called once during plugin loading.
    
            Useful for initializing resources based on the global Sygnals configuration.
    
            Args:
                config: The resolved Sygnals configuration dictionary.
            """
            pass
    
        def teardown(self):
            """Optional teardown hook called during application shutdown."""
            # Cleanup resources (e.g., close files, release locks).
            pass
    ```

6. **Discovery & Loading**

    - **Entry-Points:** Scan `pkg_resources` (or `importlib.metadata`) for `sygnals.plugins` entry points.
    - **Local Directory:** Scan configured `[paths].plugin_dir`.
    - **Loading:** Parse `plugin.toml`, check `sygnals_api` compatibility against core version using `packaging.specifiers`. Import `entry_point` module/class. Instantiate plugin. Call `setup`. Call relevant `register_*` methods on the central `PluginRegistry`. Wrap loading in `try/except` to isolate faulty plugins.

7. **Versioning & Compatibility**

    - Core API Version: Defined in `sygnals/version.py`.
    - Manifest Specifier (`sygnals_api`): Evaluated using `packaging.specifiers.SpecifierSet`.
    - On Mismatch: Plugin is skipped with a warning (e.g., `Plugin X vY incompatible with sygnals vZ`).

8. Plugin Management CLI (sygnals plugin ...)

    Integrate plugin management commands:

    - `sygnals plugin list`: Show installed entry-point and local plugins, versions, API compatibility, and enabled status.
    - `sygnals plugin install <source>`: Install a plugin (e.g., from PyPI via `pip`, or copy a local directory). Handles adding entry point if needed or copying to local dir.
    - `sygnals plugin uninstall <name>`: Uninstall/remove a plugin (via `pip` or deleting local dir).
    - `sygnals plugin enable <name>`: Mark plugin as enabled in a state file (e.g., `~/.config/sygnals/plugins.yaml`).
    - `sygnals plugin disable <name>`: Mark plugin as disabled in the state file.
    - `sygnals plugin scaffold [dest]`: Create a starter plugin template directory.

9. Scaffold Tool (sygnals plugin scaffold)

    Generates a template directory structure:

    ```
    <dest>/
    ├── plugin.toml          # Pre-filled manifest template
    ├── pyproject.toml       # Or setup.py, with entry-point config
    └── <plugin_name_pkg>/
        ├── __init__.py      # Imports plugin class
        └── plugin.py        # Stub subclass of SygnalsPluginBase with documentation
    ```

10. **Security & Isolation**

    - **Error Handling:** Use `try/except` during loading and execution to prevent plugin errors from crashing `sygnals`. Log errors clearly.
    - **Dependency Conflicts:** Plugins declare dependencies in `plugin.toml` (informational) and `pyproject.toml`/`setup.py` (installation). User/pip manages conflict resolution.

11. **Testing Strategy**

    - **Unit Tests:** Test manifest parsing, registry operations, loader logic, compatibility checks.
    - **Integration Tests:** Test plugin CLI commands (`list`, `enable`, etc.). Test core `sygnals` commands using functionality provided by dummy test plugins (e.g., a custom filter, a custom feature).
    - **CI Pipeline:** Run tests, linting, and type checking.

#### 3.10. Command-Line Interface (CLI) Refinements

- **Structure:** Organize commands logically (e.g., `load`, `preprocess filter/resample`, `segment`, `features extract/transform`, `augment`, `visualize`, `save dataset`, `plugin`). Use `click` groups for organization.
- **Configuration Overrides:** Implement clear CLI flags for overriding key config settings (e.g., `--filter-order 8`, `--log-level-file DEBUG`, `-o output.csv`, `--format npz`).
- **Verbosity Control:** Implement `--verbosity LEVEL` (`debug`, `verbose`, `normal`, `silent`). Add shortcuts: `-v` (verbose), `-vv` (debug), `-q` (silent). Control console log level accordingly.
- **Logging Flags:** Implement `--log-dir`, `--no-log-file`, `--log-level-file`, `--log-filename` flags.
- **Plugin Subcommand:** Implement `sygnals plugin` group as specified in Section 3.9.
- **Progress Indicators:** Use `rich.progress` for long operations (batch processing, feature extraction on large files).
- **Piping:** Support piping data between `sygnals` commands where logical (e.g., `sygnals load file.wav | sygnals preprocess filter ... | sygnals save dataset --format npz -`). Requires careful handling of data streams (e.g., using temporary files or in-memory objects).
- **Interactive Mode (Optional):** Consider a REPL mode (`sygnals interactive`) for easier sequential command execution.

#### 3.11. Logging System (Modeled after Semantiscan examples)

- Configuration:

     Controlled via the 

    ```
    [logging]
    ```

     section in 

    ```
    sygnals.toml
    ```

     (and overrides):

    - `log_file_enabled` (`true`/`false`): Enable/disable persistent file logging. Default: `true`. CLI: `--no-log-file`. Env: `SYGNALS_LOG_FILE_ENABLED`.
    - `log_directory` (Path): Directory for log files. Default: `"./sygnals_logs"`. CLI: `--log-dir`. Env: `SYGNALS_LOG_DIRECTORY`.
    - `log_filename_template` (String): Naming pattern for log files, uses `{timestamp:strftime_format}`. Default: `"sygnals_run_{timestamp:%Y%m%d_%H%M%S}.log"`. CLI: `--log-filename`. Env: `SYGNALS_LOG_FILENAME_TEMPLATE`.
    - `log_level_file` (String): Minimum level for file logs (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Default: `"DEBUG"`. CLI: `--log-level-file`. Env: `SYGNALS_LOG_LEVEL_FILE`.
    - `log_format` (String): Format string for file log entries using Python `logging` codes. Default: `"%(asctime)s [%(levelname)-8s] %(name)-30s - %(message)s (%(filename)s:%(lineno)d)"`.
    - `log_level_console` (String): *Default* minimum level for console output, primarily controlled by `--verbosity`. Default: `"INFO"`.

- **Runtime Control:** Console verbosity controlled by `--verbosity` flag mapping to `logging` levels. File logging level set independently via `log_level_file`.

- **Implementation:** Use Python's standard `logging` module. Configure handlers (Console using `rich.logging.RichHandler`, File) and formatters based on the final resolved configuration. Use logger names based on module hierarchy (e.g., `sygnals.core.dsp`, `sygnals.cli.features_cmd`).

### 4. Project Directory Structure (Illustrative - Updated)

```
sygnals_vnext/
├── sygnals/                  # Main package source
│   ├── __init__.py
│   ├── version.py          # Contains __version__ for API compatibility
│   ├── cli/                  # CLI commands and entry points
│   │   ├── __init__.py
│   │   ├── main.py           # Main CLI app (Click)
│   │   ├── base_cmd.py       # Base command setup (config/logging loading)
│   │   ├── load_cmd.py
│   │   ├── preprocess_cmd.py # Group for filter, resample etc.
│   │   ├── segment_cmd.py
│   │   ├── features_cmd.py   # Group for extract, transform
│   │   ├── augment_cmd.py
│   │   ├── visualize_cmd.py
│   │   ├── save_cmd.py
│   │   └── plugin_cmd.py     # Plugin management commands
│   ├── config/               # Configuration loading
│   │   ├── __init__.py
│   │   ├── loaders.py        # Loads from file, env, defaults
│   │   └── models.py         # Pydantic models for config structure
│   ├── core/                 # Core processing logic
│   │   ├── __init__.py
│   │   ├── common/           # Shared data structures (SignalData obj?), constants
│   │   ├── data_handler.py   # Enhanced data I/O, format handling
│   │   ├── dsp/              # DSP algorithms (FFT, STFT, CQT, Corr, PSD...)
│   │   ├── filters/          # Filter implementations (Butter, Cheby, FIR...)
│   │   ├── transforms/       # Wavelet, MFCC, Hilbert...
│   │   ├── audio/            # Audio-specific logic
│   │   │   ├── __init__.py
│   │   │   ├── effects/      # Reverb, Delay, EQ, Noise Reduction...
│   │   │   ├── io.py         # Audio file I/O (WAV, FLAC, OGG)
│   │   │   └── features.py   # Pitch, Onset, ZCR...
│   │   ├── features/         # Dedicated module for feature extraction logic
│   │   │   ├── __init__.py
│   │   │   ├── time_domain.py
│   │   │   ├── frequency_domain.py
│   │   │   ├── cepstral.py
│   │   │   └── manager.py    # Dispatches feature calculation
│   │   ├── augment/          # Data augmentation logic
│   │   │   ├── __init__.py
│   │   │   ├── noise.py
│   │   │   └── effects_based.py
│   │   └── ml_utils/         # Scalers, dataset formatters
│   │       ├── __init__.py
│   │       ├── scaling.py
│   │       └── formatters.py
│   ├── plugins/              # Plugin management and API
│   │   ├── __init__.py
│   │   ├── api.py            # Base classes (SygnalsPluginBase), registry (PluginRegistry)
│   │   ├── loader.py         # Discovers and loads plugins
│   │   └── scaffold.py       # Logic for `plugin scaffold`
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── logging_config.py # Sets up logging based on config
│   │   └── plot_utils.py     # Base plotting helpers
│   └── visualizations/       # Visualization generation logic
│       ├── __init__.py
│       ├── manager.py        # Dispatches plotting based on type
│       ├── plot_waveform.py
│       ├── plot_spectrum.py
│       ├── plot_spectrogram.py
│       └── ...               # Other plot types
├── docs/                     # Documentation source (Sphinx/MkDocs)
│   ├── index.md
│   ├── installation.md
│   ├── usage.md
│   ├── configuration_guide.md
│   ├── plugins/
│   │   ├── adding_new_plugin.md
│   │   └── developing_plugin.md
│   ├── tutorials/            # Specific workflow examples
│   │   └── ml_data_prep.md
│   ├── api/                  # Auto-generated API reference
│   └── contributing.md
├── examples/                 # Example usage scripts/notebooks (incl. ML prep)
├── plugins_contrib/          # Directory for user-provided plugins (example)
├── tests/                    # Unit and integration tests (incl. plugin tests)
├── sygnals.toml.example      # Example configuration file matching spec
├── pyproject.toml            # Project metadata and dependencies (Poetry/PDM)
├── README.md                 # Concise overview, links to docs
└── .gitignore
```

### 5. Core Workflows (Updated Examples)

#### 5.1. ML Dataset Preparation Workflow (showing potential plugin use)

1. **Goal:** Create a dataset of scaled spectral features from audio segments for environmental sound classification, using a custom peak detection plugin.

2. **Assumed Plugin:** `sygnals-peak-detector` installed and enabled, providing `segment --method peak_events` and `features extract --features peak_rate`.

3. Commands:

    Bash

    ```
    sygnals load input.wav \
    | sygnals preprocess filter --type bandpass --low-cutoff 300 --high-cutoff 8000 \
    | sygnals segment --method peak_events --threshold 0.5 --min-distance 0.1 \
    | sygnals features extract --features spectral_centroid,spectral_bandwidth,peak_rate --frame-len 0.04 --frame-hop 0.02 \
    | sygnals features transform scale --scaler standard \
    | sygnals save dataset --format npz --output sound_features_scaled.npz --verbosity verbose
    ```

4. Process:

    - Load config (finds `sygnals-peak-detector` plugin).
    - Plugin `setup` is called. Plugin registers `peak_events` segmentation method and `peak_rate` feature extractor.
    - Load `input.wav`.
    - Apply bandpass filter.
    - Segment the filtered audio using the plugin's `peak_events` method.
    - For each segment, extract Spectral Centroid, Bandwidth (built-in), and `peak_rate` (plugin feature).
    - Apply StandardScaler to the combined feature matrix.
    - Save the scaled features (and potentially segment metadata) to `sound_features_scaled.npz`.
    - Plugin `teardown` might be called on exit.

5. Logging & Output:

    - **Console (`verbose`):** Will show INFO and DEBUG level messages, including details about configuration loaded, files processed, filter applied, number of segments found, features extracted per segment, scaling parameters, and final save location. Progress bars likely shown for segmentation and feature extraction.
    - **Log File (`DEBUG` level):** Contains highly detailed trace information, including function calls, intermediate values, exact parameters used, plugin loading steps, timing information for different stages, and any low-level warnings or errors.

### 6. Development Plan (Phased Approach - Updated)

- **Phase 1: Foundation & Refactoring:** Implement enhanced config system (`config` module), set up `pyproject.toml`, integrate linters/formatters/type checking, refactor core modules (`cli`, `dsp`/`filters`), establish basic CI pipeline, implement logging system (`utils.logging_config`). **Ensure excellent inline documentation from the start.**
- **Phase 2: Core DSP, Audio, Features:** Implement core DSP/Audio features (Filters, STFT, CQT, Effects), build `features` module with core extractors, enhance data I/O (`data_handler`), add core visualizations. Expand unit tests. **Write comprehensive docstrings and usage examples.**
- **Phase 3: Plugin System:** Implement plugin API (`plugins.api`), loader (`plugins.loader`), registry, version compatibility checks, scaffold tool (`plugins.scaffold`), and CLI commands (`cli.plugin_cmd`). Test plugin loading and basic registration. **Develop initial `developing_plugins.md`.**
- **Phase 4: Data Wrangling & ML Utilities:** Implement advanced segmentation, multi-channel handling, data augmentation (`augment` module), scaling (`ml_utils`), enhance dataset assembly (`save dataset` command). Write integration tests using dummy plugins. **Document these features in `usage.md`.**
- **Phase 5: Documentation & CLI Polish:** Create comprehensive user documentation website structure (`docs/`). Draft initial versions of `README.md`, `installation.md`, `usage.md`, `configuration_guide.md`, `adding_new_plugin.md`, `developing_plugin.md`, `contributing.md`. Refine overall CLI structure, implement piping, progress indicators, finalize configuration options, ensure seamless integration of all components.
- **Phase 6: Finalization & Release:** Complete and polish all documentation (inline and external), ensure high test coverage, perform thorough testing, finalize `README.md`, package for distribution, prepare for release.

------