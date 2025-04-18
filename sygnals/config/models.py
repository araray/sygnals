# sygnals/config/models.py

"""
Pydantic models for defining the structure and validation of the sygnals configuration (sygnals.toml).
Uses Pydantic V2 syntax.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Import V2 components
from pydantic import BaseModel, Field, field_validator, DirectoryPath, FilePath, ConfigDict

# --- Helper Functions ---

def _resolve_path(path: Union[str, Path]) -> Path:
    """Resolves and expands user paths."""
    # Ensure input is converted to Path before expansion/resolution
    return Path(path).expanduser().resolve()

# --- Model Definitions ---

class DefaultsConfig(BaseModel):
    """Default processing parameters."""
    default_sample_rate: int = Field(44100, description="Default sample rate assumed for data if not specified.")
    default_input_format: str = Field("wav", description="Default format assumed for input files (e.g., 'wav', 'csv').")
    default_output_format: str = Field("wav", description="Default format for saving outputs (e.g., 'wav', 'csv', 'npz').")
    default_fft_window: str = Field("hann", description="Default window function for FFT/STFT.")
    default_filter_order: int = Field(5, description="Default order for IIR filters like Butterworth.")
    # Add other relevant defaults as needed

class PathsConfig(BaseModel):
    """Configuration for file paths used by Sygnals."""
    # Use Field validation for types where appropriate, although Pydantic handles Path conversion
    plugin_dir: Path = Field(default=Path("~/.config/sygnals/plugins"), description="Path to local user plugins.")
    cache_dir: Path = Field(default=Path("./.sygnals_cache"), description="Path for temporary or cached files.")
    output_dir: Path = Field(default=Path("./sygnals_output"), description="Default directory for saving results.")
    log_directory: Path = Field(default=Path("./sygnals_logs"), description="Directory for log files.") # Moved from LoggingConfig for consistency

    # Use field_validator with mode='before' to modify input before standard validation
    @field_validator('plugin_dir', 'cache_dir', 'output_dir', 'log_directory', mode='before')
    @classmethod
    def resolve_paths_before_validation(cls, value: Any) -> Path:
        """Resolves paths before Pydantic validates them."""
        if isinstance(value, (str, Path)):
            return _resolve_path(value)
        # Let Pydantic handle other types or raise validation error
        return value

class STFTParams(BaseModel):
    """Parameters for Short-Time Fourier Transform."""
    window_size: int = 1024
    overlap: int = 512
    window: str = "hann" # Can override default_fft_window

class MFCCParams(BaseModel):
    """Parameters for Mel-Frequency Cepstral Coefficients."""
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512

class FixedLengthSegmentationParams(BaseModel):
    """Parameters for fixed-length segmentation."""
    length: float = Field(1.0, gt=0, description="Segment length in seconds, must be positive.")
    overlap: float = Field(0.5, ge=0, lt=1.0, description="Overlap ratio (0.0 to < 1.0).")

class SegmentationParams(BaseModel):
    """Container for different segmentation method parameters."""
    fixed_length: FixedLengthSegmentationParams = Field(default_factory=FixedLengthSegmentationParams)
    # Add other methods like 'silence_detection', 'event_based' later

class StandardScalerParams(BaseModel):
    """Parameters for standard scaling."""
    with_mean: bool = True
    with_std: bool = True

class ScalingParams(BaseModel):
    """Container for different scaling method parameters."""
    standard_scaler: StandardScalerParams = Field(default_factory=StandardScalerParams)
    # Add 'min_max_scaler', 'robust_scaler' later

class ParametersConfig(BaseModel):
    """Centralized parameters for reusable components."""
    stft: STFTParams = Field(default_factory=STFTParams)
    mfcc: MFCCParams = Field(default_factory=MFCCParams)
    segmentation: SegmentationParams = Field(default_factory=SegmentationParams)
    scaling: ScalingParams = Field(default_factory=ScalingParams)
    # Add parameters for filters, effects, etc. as needed

class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    log_file_enabled: bool = Field(True, description="Enable/disable persistent file logging.")
    # log_directory moved to PathsConfig
    log_filename_template: str = Field("sygnals_run_{timestamp:%Y%m%d_%H%M%S}.log", description="Naming pattern for log files.")
    log_level_file: str = Field("DEBUG", description="Minimum level for file logs (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    log_format: str = Field("%(asctime)s [%(levelname)-8s] %(name)-30s - %(message)s (%(filename)s:%(lineno)d)", description="Format string for file log entries.")
    log_level_console: str = Field("INFO", description="Default minimum level for console output (overridden by verbosity flags).")

    # Use field_validator for multiple fields
    @field_validator('log_level_file', 'log_level_console')
    @classmethod
    def check_log_level(cls, value: str) -> str:
        """Validate log level strings."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_value = value.upper()
        if upper_value not in allowed_levels:
            raise ValueError(f"Invalid log level '{value}'. Must be one of {allowed_levels}")
        return upper_value # Return the validated uppercase value

class DiscoveryConfig(BaseModel):
    """Configuration for file/directory discovery processes."""
    excluded_dirs: List[str] = Field(default_factory=lambda: ["__pycache__", ".git", "venv", ".venv", ".pytest_cache"])
    excluded_files: List[str] = Field(default_factory=lambda: [".DS_Store", "*.tmp", "*.swp"])
    use_gitignore: bool = Field(True, description="Whether to respect .gitignore files during discovery.")

class SygnalsConfig(BaseModel):
    """Root configuration model for Sygnals."""
    # Define model config using ConfigDict
    model_config = ConfigDict(
        extra='allow', # Allow extra fields (e.g., for plugin configs)
        validate_assignment=True # Re-validate fields on assignment
    )

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    parameters: ParametersConfig = Field(default_factory=ParametersConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)

    # Placeholder for future plugin-specific configurations
    plugins: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Configurations specific to installed plugins.")
