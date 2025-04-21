# sygnals_parquet/plugin.py

"""
Main implementation file for the 'sygnals-parquet' Sygnals plugin.
Registers Parquet file reading and writing capabilities.
"""

import logging
from typing import Dict, Any # Import necessary types

# Import the base class and registry from Sygnals core API
from sygnals.plugins.api import SygnalsPluginBase, PluginRegistry

# Import the actual I/O functions from this plugin package
from .parquet_io import read_parquet, save_parquet

logger = logging.getLogger(__name__)

# --- Plugin Implementation ---

class SygnalsParquetPlugin(SygnalsPluginBase):
    """
    Plugin to add Parquet file support to Sygnals.
    """

    # --- Required Properties ---

    @property
    def name(self) -> str:
        """Return the unique plugin name (must match plugin.toml)."""
        return "sygnals-parquet"

    @property
    def version(self) -> str:
        """Return the plugin version (must match plugin.toml)."""
        # Ideally, read from manifest or package metadata, hardcoded for simplicity here
        return "0.1.0"

    # --- Optional Lifecycle Hooks ---

    def setup(self, config: Dict[str, Any]):
        """Initialize plugin resources (if any)."""
        logger.info(f"Initializing plugin '{self.name}' v{self.version}")
        # Check if pyarrow is available during setup (optional)
        try:
            import pyarrow
            logger.debug("PyArrow library found for Parquet support.")
        except ImportError:
            logger.warning("PyArrow library not found. Parquet functionality might fail if pandas doesn't find it.")
        pass

    def teardown(self):
        """Clean up resources (if any)."""
        logger.info(f"Tearing down plugin '{self.name}'")
        pass

    # --- Registration Hooks ---

    def register_data_readers(self, registry: PluginRegistry):
        """Register the Parquet file reader."""
        try:
            # Check dependency again before registering (optional but safer)
            import pyarrow
            registry.add_reader(".parquet", read_parquet)
            logger.debug(f"Plugin '{self.name}' registered reader for '.parquet'.")
        except ImportError:
            logger.error(f"Plugin '{self.name}' could not register Parquet reader: 'pyarrow' not found.")
        except Exception as e:
             logger.error(f"Plugin '{self.name}' failed during reader registration: {e}", exc_info=True)


    def register_data_writers(self, registry: PluginRegistry):
        """Register the Parquet file writer."""
        try:
            import pyarrow
            registry.add_writer(".parquet", save_parquet)
            logger.debug(f"Plugin '{self.name}' registered writer for '.parquet'.")
        except ImportError:
            logger.error(f"Plugin '{self.name}' could not register Parquet writer: 'pyarrow' not found.")
        except Exception as e:
             logger.error(f"Plugin '{self.name}' failed during writer registration: {e}", exc_info=True)
