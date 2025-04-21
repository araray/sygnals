# sygnals/utils/logging_config.py

"""
Configures the logging system for the Sygnals application based on loaded settings.
Uses Rich for enhanced console logging.
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler

from sygnals.config import SygnalsConfig
from sygnals.version import __version__

# --- Constants ---
# Map verbosity levels (from CLI flags) to logging levels
VERBOSITY_MAP = {
    0: logging.WARNING,  # Default (normal)
    1: logging.INFO,     # -v (verbose)
    2: logging.DEBUG,    # -vv (debug)
    -1: logging.CRITICAL + 10 # -q (quiet/silent) - use a level higher than critical
}

# --- Setup Function ---

def setup_logging(config: SygnalsConfig, verbosity: int = 0):
    """
    Configures the root logger based on the provided configuration and verbosity level.

    Args:
        config: The loaded SygnalsConfig object.
        verbosity: An integer representing the desired console verbosity level
                   (e.g., 0 for normal, 1 for verbose, 2 for debug, -1 for quiet).
    """
    log_cfg = config.logging
    paths_cfg = config.paths

    # Determine console log level based on verbosity flag
    console_level = VERBOSITY_MAP.get(verbosity, logging.INFO) # Default to INFO if verbosity is unexpected

    # --- Root Logger Configuration ---
    root_logger = logging.getLogger("sygnals") # Get logger for the main package
    root_logger.setLevel(logging.DEBUG)  # Set root logger to lowest level (DEBUG) to capture everything
                                         # Handlers will filter based on their specific levels.
    root_logger.handlers.clear() # Remove any existing handlers (important for re-configuration)

    # --- Console Handler (Rich) ---
    if console_level <= logging.CRITICAL: # Only add console handler if not quiet
        console_formatter = logging.Formatter("%(message)s") # Simple format for Rich
        console_handler = RichHandler(
            level=console_level,
            show_time=False, # Rich handles time formatting
            show_level=True,
            show_path=False, # Keep console output cleaner
            rich_tracebacks=True,
            markup=True, # Enable Rich markup in log messages
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        logging.getLogger().addHandler(console_handler) # Also add to default root logger if needed elsewhere

    # --- File Handler ---
    if log_cfg.log_file_enabled:
        try:
            log_dir = paths_cfg.log_directory
            log_dir.mkdir(parents=True, exist_ok=True) # Ensure log directory exists

            # Format filename template with timestamp
            timestamp_str = datetime.now().strftime(log_cfg.log_filename_template)
            # Simple approach: replace {timestamp:format} - more robust parsing could be added
            log_filename = log_cfg.log_filename_template.format(timestamp=datetime.now())
            log_filepath = log_dir / log_filename

            file_formatter = logging.Formatter(log_cfg.log_format)
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setLevel(log_cfg.log_level_file) # Use level from config
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

            # Log initial information to the file
            file_logger = logging.getLogger("sygnals.init") # Specific logger for init messages
            file_logger.info(f"--- Sygnals v{__version__} Log Start ---")
            file_logger.info(f"File logging level set to: {log_cfg.log_level_file}")
            file_logger.info(f"Console logging level set to: {logging.getLevelName(console_level)}")
            file_logger.debug(f"Full configuration loaded: {config.model_dump()}") # Log full config at debug level

        except Exception as e:
            # Log error to console if file logging setup fails
            console_logger = logging.getLogger("sygnals.error")
            console_logger.error(f"Failed to configure file logging: {e}", exc_info=True)
            log_cfg.log_file_enabled = False # Disable file logging if setup failed

    # --- Initial Log Messages ---
    init_logger = logging.getLogger("sygnals.init")
    init_logger.info(f"Sygnals v{__version__} initialized.")
    if log_cfg.log_file_enabled:
        init_logger.info(f"Logging to file: {log_filepath}")
    else:
        init_logger.info("File logging is disabled.")

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # Load default config for testing
    test_config = SygnalsConfig()

    print("--- Testing Logging Setup (Verbosity 0 - Normal) ---")
    setup_logging(test_config, verbosity=0)
    logging.getLogger("sygnals.test").debug("This is a DEBUG message (should not appear on console).")
    logging.getLogger("sygnals.test").info("This is an INFO message.")
    logging.getLogger("sygnals.test").warning("This is a WARNING message.")
    logging.getLogger("sygnals.test").error("This is an ERROR message.")

    print("\n--- Testing Logging Setup (Verbosity 1 - Verbose) ---")
    setup_logging(test_config, verbosity=1)
    logging.getLogger("sygnals.test").debug("This is a DEBUG message (should not appear on console).")
    logging.getLogger("sygnals.test").info("This is an INFO message.")

    print("\n--- Testing Logging Setup (Verbosity 2 - Debug) ---")
    setup_logging(test_config, verbosity=2)
    logging.getLogger("sygnals.test").debug("This is a DEBUG message (should appear).")
    logging.getLogger("sygnals.test").info("This is an INFO message.")

    print("\n--- Testing Logging Setup (Verbosity -1 - Quiet) ---")
    setup_logging(test_config, verbosity=-1)
    logging.getLogger("sygnals.test").info("This is an INFO message (should not appear).")
    logging.getLogger("sygnals.test").warning("This is a WARNING message (should not appear).")
    logging.getLogger("sygnals.test").critical("This is a CRITICAL message (should not appear).")

    print("\nCheck log files in:", test_config.paths.log_directory)
