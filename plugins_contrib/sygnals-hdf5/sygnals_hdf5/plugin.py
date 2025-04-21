# sygnals_hdf5/plugin.py

"""
Main implementation file for the 'sygnals-hdf5' Sygnals plugin.
Registers HDF5 file reading and writing capabilities.
"""

import logging
from typing import Dict, Any # Import necessary types

# Import the base class and registry from Sygnals core API
from sygnals.plugins.api import SygnalsPluginBase, PluginRegistry

# Import the actual I/O functions from this plugin package
from .hdf5_io import read_hdf5, save_hdf5

logger = logging.getLogger(__name__)

# --- Plugin Implementation ---

class SygnalsHdf5Plugin(SygnalsPluginBase):
    """
    Plugin to add HDF5 file support (basic dataset I/O) to Sygnals.
    """

    # --- Required Properties ---

    @property
    def name(self) -> str:
        """Return the unique plugin name (must match plugin.toml)."""
        return "sygnals-hdf5"

    @property
    def version(self) -> str:
        """Return the plugin version (must match plugin.toml)."""
        # Ideally, read from manifest or package metadata, hardcoded for simplicity here
        return "0.1.0"

    # --- Optional Lifecycle Hooks ---

    def setup(self, config: Dict[str, Any]):
        """Initialize plugin resources (if any)."""
        logger.info(f"Initializing plugin '{self.name}' v{self.version}")
        # Check if h5py is available during setup (optional)
        try:
            import h5py
            logger.debug("h5py library found for HDF5 support.")
        except ImportError:
            logger.warning("h5py library not found. HDF5 functionality will fail.")
        pass

    def teardown(self):
        """Clean up resources (if any)."""
        logger.info(f"Tearing down plugin '{self.name}'")
        pass

    # --- Registration Hooks ---

    def register_data_readers(self, registry: PluginRegistry):
        """Register the HDF5 file reader."""
        try:
            # Check dependency again before registering (optional but safer)
            import h5py
            # Register for both common extensions
            registry.add_reader(".hdf5", read_hdf5)
            registry.add_reader(".h5", read_hdf5)
            logger.debug(f"Plugin '{self.name}' registered reader for '.hdf5' and '.h5'.")
        except ImportError:
            logger.error(f"Plugin '{self.name}' could not register HDF5 reader: 'h5py' not found.")
        except Exception as e:
             logger.error(f"Plugin '{self.name}' failed during reader registration: {e}", exc_info=True)


    def register_data_writers(self, registry: PluginRegistry):
        """Register the HDF5 file writer."""
        try:
            import h5py
            # Register for both common extensions
            registry.add_writer(".hdf5", save_hdf5)
            registry.add_writer(".h5", save_hdf5)
            logger.debug(f"Plugin '{self.name}' registered writer for '.hdf5' and '.h5'.")
        except ImportError:
            logger.error(f"Plugin '{self.name}' could not register HDF5 writer: 'h5py' not found.")
        except Exception as e:
             logger.error(f"Plugin '{self.name}' failed during writer registration: {e}", exc_info=True)
