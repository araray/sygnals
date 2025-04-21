# sygnals/config/loaders.py

"""
Functions for loading and merging Sygnals configuration from various sources.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import toml
from pydantic import ValidationError

from .models import SygnalsConfig

logger = logging.getLogger(__name__) # Use standard logging

# --- Constants ---
ENV_PREFIX = "SYGNALS_"
USER_CONFIG_DIR = Path("~/.config/sygnals").expanduser()
USER_CONFIG_FILE = USER_CONFIG_DIR / "sygnals.toml"
PROJECT_CONFIG_FILE = Path("./sygnals.toml").resolve()

# --- Helper Functions ---

def _load_toml_file(filepath: Path) -> Dict[str, Any]:
    """Loads a TOML file if it exists, returns empty dict otherwise."""
    if filepath.is_file():
        try:
            with open(filepath, 'r') as f:
                return toml.load(f)
        except toml.TomlDecodeError as e:
            logger.warning(f"Error decoding TOML file '{filepath}': {e}. Skipping.")
        except Exception as e:
            logger.warning(f"Could not read config file '{filepath}': {e}. Skipping.")
    return {}

def _deep_merge_dicts(base: Dict, update: Dict) -> Dict:
    """Recursively merges 'update' dict into 'base' dict."""
    merged = base.copy()
    for key, value in update.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            # Update takes precedence
            merged[key] = value
    return merged

def _get_config_from_env() -> Dict[str, Any]:
    """Reads configuration settings from environment variables."""
    env_config = {}
    for env_var, value in os.environ.items():
        if env_var.startswith(ENV_PREFIX):
            # Convert SYGNALS_SECTION_SUBSECTION_KEY to ['section', 'subsection', 'key']
            keys = env_var[len(ENV_PREFIX):].lower().split('_')
            # Try to parse value (simple types for now)
            parsed_value: Any = value
            try:
                # Handle potential boolean values
                if value.lower() in ['true', 'false']:
                    parsed_value = value.lower() == 'true'
                else:
                    # Try int, then float
                    try:
                        parsed_value = int(value)
                    except ValueError:
                        try:
                            parsed_value = float(value)
                        except ValueError:
                            pass # Keep as string
            except Exception:
                 pass # Keep as string if parsing fails

            # Build nested dictionary structure
            d = env_config
            for i, key in enumerate(keys):
                if i == len(keys) - 1:
                    d[key] = parsed_value
                else:
                    d = d.setdefault(key, {})
    return env_config

# --- Main Loading Function ---

def load_configuration(
    config_files: Optional[List[Path]] = None,
    disable_project_config: bool = False,
    disable_user_config: bool = False,
) -> SygnalsConfig:
    """
    Loads Sygnals configuration from defaults, files, and environment variables.

    Precedence (highest first):
    1. Environment Variables (SYGNALS_*)
    2. User Config File (~/.config/sygnals/sygnals.toml)
    3. Project Config File (./sygnals.toml)
    4. Explicitly passed config files (if any)
    5. Internal Defaults (from Pydantic models)

    Args:
        config_files: List of additional config file paths to load.
        disable_project_config: If True, ignores ./sygnals.toml.
        disable_user_config: If True, ignores ~/.config/sygnals/sygnals.toml.

    Returns:
        A validated SygnalsConfig object.
    """
    # 1. Start with internal defaults (implicitly handled by Pydantic model)
    # We create an empty dict and let Pydantic fill defaults later.
    merged_config_dict: Dict[str, Any] = {}

    # 2. Load explicitly passed config files (lowest file precedence)
    if config_files:
        for file_path in reversed(config_files): # Load in reverse for correct precedence
             merged_config_dict = _deep_merge_dicts(merged_config_dict, _load_toml_file(file_path))

    # 3. Load project config file
    if not disable_project_config:
        logger.debug(f"Attempting to load project config: {PROJECT_CONFIG_FILE}")
        project_cfg = _load_toml_file(PROJECT_CONFIG_FILE)
        if project_cfg:
             logger.info(f"Loaded project configuration from {PROJECT_CONFIG_FILE}")
             merged_config_dict = _deep_merge_dicts(merged_config_dict, project_cfg)

    # 4. Load user config file
    if not disable_user_config:
        # Ensure user config directory exists
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Attempting to load user config: {USER_CONFIG_FILE}")
        user_cfg = _load_toml_file(USER_CONFIG_FILE)
        if user_cfg:
            logger.info(f"Loaded user configuration from {USER_CONFIG_FILE}")
            merged_config_dict = _deep_merge_dicts(merged_config_dict, user_cfg)

    # 5. Load environment variables (highest precedence)
    env_cfg = _get_config_from_env()
    if env_cfg:
        logger.debug(f"Applying environment variable configuration: {env_cfg}")
        merged_config_dict = _deep_merge_dicts(merged_config_dict, env_cfg)

    # 6. Validate and instantiate the Pydantic model
    try:
        # Pydantic automatically uses default values for missing fields
        final_config = SygnalsConfig(**merged_config_dict)
        logger.debug("Configuration loaded and validated successfully.")
        # Ensure essential directories exist after loading config
        final_config.paths.log_directory.mkdir(parents=True, exist_ok=True)
        final_config.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        final_config.paths.output_dir.mkdir(parents=True, exist_ok=True)
        # plugin_dir is handled separately by plugin loader
        return final_config
    except ValidationError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        # Provide a default config on validation error to prevent crashing
        # Log the error clearly so the user knows something is wrong
        logger.warning("Falling back to default configuration due to validation errors.")
        return SygnalsConfig() # Return default config
    except Exception as e:
        logger.error(f"An unexpected error occurred during configuration loading: {e}")
        logger.warning("Falling back to default configuration due to an unexpected error.")
        return SygnalsConfig() # Return default config

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # Configure basic logging for testing this module directly
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example: Set an environment variable
    os.environ[f"{ENV_PREFIX}DEFAULTS_DEFAULT_SAMPLE_RATE"] = "48000"
    os.environ[f"{ENV_PREFIX}PATHS_OUTPUT_DIR"] = "/tmp/sygnals_test_output"
    os.environ[f"{ENV_PREFIX}LOGGING_LOG_LEVEL_FILE"] = "INFO"

    # Create dummy config files
    dummy_proj_file = Path("./sygnals.toml")
    dummy_user_dir = Path("~/.config/sygnals").expanduser()
    dummy_user_dir.mkdir(parents=True, exist_ok=True)
    dummy_user_file = dummy_user_dir / "sygnals.toml"

    with open(dummy_proj_file, "w") as f:
        f.write("[defaults]\ndefault_fft_window = 'blackman'\n")
        f.write("[paths]\ncache_dir = './.my_cache'\n")

    with open(dummy_user_file, "w") as f:
        f.write("[logging]\nlog_file_enabled = false\n") # Overrides default
        f.write("[defaults]\ndefault_filter_order = 7\n") # Overrides default

    print("--- Loading Configuration ---")
    config = load_configuration()
    print("\n--- Final Configuration ---")
    print(config.model_dump_json(indent=2))

    # Clean up dummy files and env vars
    dummy_proj_file.unlink()
    dummy_user_file.unlink()
    try:
        dummy_user_dir.rmdir() # Only if empty
    except OSError:
        pass
    del os.environ[f"{ENV_PREFIX}DEFAULTS_DEFAULT_SAMPLE_RATE"]
    del os.environ[f"{ENV_PREFIX}PATHS_OUTPUT_DIR"]
    del os.environ[f"{ENV_PREFIX}LOGGING_LOG_LEVEL_FILE"]
