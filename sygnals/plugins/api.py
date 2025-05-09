# sygnals/plugins/api.py

"""
Defines the core API for Sygnals plugins, including the base plugin class
and the registry for extensions.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Type, Optional, List, Set, Union, Tuple
from pathlib import Path

# Import necessary types for data handlers
import pandas as pd
import numpy as np
from numpy.typing import NDArray
# Define types for reader/writer callables for clarity
# Reader should return ReadResult type from data_handler
ReadResult = Union[pd.DataFrame, Dict[str, NDArray[Any]], Tuple[NDArray[np.float64], int]]
ReaderCallable = Callable[..., ReadResult] # path, **kwargs -> ReadResult
# Writer takes data (SaveInput type from data_handler) and path, **kwargs
SaveInput = Union[pd.DataFrame, NDArray[Any], Dict[str, NDArray[Any]], Tuple[NDArray[np.float64], int]]
WriterCallable = Callable[[SaveInput, Union[str, Path], Any], None] # data, path, **kwargs -> None

# Import Click types for CLI command registration if needed later
try:
    import click
    ClickCommandType = Union[click.Command, click.Group, Callable[..., Any]]
except ImportError:
    click = None # type: ignore
    ClickCommandType = Callable[..., Any] # Fallback type

logger = logging.getLogger(__name__)

# --- Base Plugin Class ---

class SygnalsPluginBase(ABC):
    """
    Abstract base class for all Sygnals plugins.

    Plugins must inherit from this class and implement the abstract properties.
    They can optionally implement `register_*` methods to add functionality
    to Sygnals.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the plugin (from plugin.toml)."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version of the plugin (from plugin.toml)."""
        pass

    # --- Registration Hooks (Optional Implementation by Subclasses) ---

    def register_filters(self, registry: 'PluginRegistry'):
        """
        Register custom filter functions or classes.

        Args:
            registry: The central plugin registry instance. Use methods like
                      registry.add_filter(name: str, filter_callable: Callable).
        """
        pass

    def register_transforms(self, registry: 'PluginRegistry'):
        """Register custom transform functions or classes."""
        pass

    def register_feature_extractors(self, registry: 'PluginRegistry'):
        """Register custom feature extraction functions."""
        pass

    def register_visualizations(self, registry: 'PluginRegistry'):
        """Register custom visualization functions."""
        pass

    def register_audio_effects(self, registry: 'PluginRegistry'):
        """Register custom audio effect processors."""
        pass

    def register_augmenters(self, registry: 'PluginRegistry'):
        """Register custom data augmentation functions."""
        pass

    def register_data_readers(self, registry: 'PluginRegistry'):
        """
        Register custom data reading functions for specific file extensions.

        Args:
            registry: The central plugin registry instance. Use methods like
                      registry.add_reader(extension: str, reader_callable: ReaderCallable).
                      Extension should include the dot (e.g., '.parquet').
        """
        pass # New hook for data readers

    def register_data_writers(self, registry: 'PluginRegistry'):
        """
        Register custom data writing functions for specific file extensions.

        Args:
            registry: The central plugin registry instance. Use methods like
                      registry.add_writer(extension: str, writer_callable: WriterCallable).
                      Extension should include the dot (e.g., '.parquet').
        """
        pass # New hook for data writers

    def register_cli_commands(self, registry: 'PluginRegistry'):
        """
        Register custom Click command groups or commands.

        Requires 'click' to be installed.
        """
        pass

    # --- Lifecycle Hooks (Optional Implementation by Subclasses) ---

    def setup(self, config: Dict[str, Any]):
        """
        Optional setup hook called once during plugin loading.

        Useful for initializing resources based on the global Sygnals configuration.
        The plugin should store any necessary state internally.

        Args:
            config: The resolved Sygnals configuration dictionary.
        """
        logger.debug(f"Plugin '{self.name}' setup hook called.")
        pass

    def teardown(self):
        """Optional teardown hook called during application shutdown."""
        logger.debug(f"Plugin '{self.name}' teardown hook called.")
        pass


# --- Plugin Registry ---

class PluginRegistry:
    """
    Central registry for extensions provided by plugins.

    Holds dictionaries mapping registered names or extensions to the corresponding
    callable or class provided by a plugin.
    """
    def __init__(self):
        self._filters: Dict[str, Callable] = {}
        self._transforms: Dict[str, Callable] = {}
        self._features: Dict[str, Callable] = {}
        self._visualizations: Dict[str, Callable] = {}
        self._effects: Dict[str, Callable] = {}
        self._augmenters: Dict[str, Callable] = {}
        self._readers: Dict[str, ReaderCallable] = {} # New: extension -> reader func
        self._writers: Dict[str, WriterCallable] = {} # New: extension -> writer func
        self._cli_commands: List[ClickCommandType] = []
        self._registered_plugin_names: Set[str] = set() # Track names of loaded plugins

    def add_plugin_name(self, name: str):
        """Record the name of a successfully loaded plugin."""
        self._registered_plugin_names.add(name)

    @property
    def loaded_plugin_names(self) -> List[str]:
        """Return a sorted list of names of successfully loaded plugins."""
        return sorted(list(self._registered_plugin_names))

    # --- Methods to Add Extensions ---

    def add_filter(self, name: str, filter_callable: Callable):
        """Register a custom filter function."""
        if name in self._filters:
            logger.warning(f"Filter '{name}' is already registered. Overwriting.")
        logger.debug(f"Registering filter: '{name}'")
        self._filters[name] = filter_callable

    def add_transform(self, name: str, transform_callable: Callable):
        """Register a custom transform function or class."""
        if name in self._transforms:
            logger.warning(f"Transform '{name}' is already registered. Overwriting.")
        logger.debug(f"Registering transform: '{name}'")
        self._transforms[name] = transform_callable

    def add_feature(self, name: str, feature_callable: Callable):
        """Register a custom feature extraction function."""
        if name in self._features:
            logger.warning(f"Feature '{name}' is already registered. Overwriting.")
        logger.debug(f"Registering feature: '{name}'")
        self._features[name] = feature_callable

    def add_visualization(self, name: str, vis_callable: Callable):
        """Register a custom visualization function."""
        if name in self._visualizations:
            logger.warning(f"Visualization '{name}' is already registered. Overwriting.")
        logger.debug(f"Registering visualization: '{name}'")
        self._visualizations[name] = vis_callable

    def add_effect(self, name: str, effect_callable: Callable):
        """Register a custom audio effect function or processor."""
        if name in self._effects:
            logger.warning(f"Audio effect '{name}' is already registered. Overwriting.")
        logger.debug(f"Registering audio effect: '{name}'")
        self._effects[name] = effect_callable

    def add_augmenter(self, name: str, augmenter_callable: Callable):
        """Register a custom data augmentation function."""
        if name in self._augmenters:
            logger.warning(f"Augmenter '{name}' is already registered. Overwriting.")
        logger.debug(f"Registering augmenter: '{name}'")
        self._augmenters[name] = augmenter_callable

    def add_reader(self, extension: str, reader_callable: ReaderCallable):
        """Register a custom data reader for a file extension."""
        ext_lower = extension.lower()
        if not ext_lower.startswith('.'):
             logger.warning(f"Reader extension '{extension}' should start with a dot (e.g., '.parquet'). Adding dot.")
             ext_lower = '.' + ext_lower
        if ext_lower in self._readers:
            logger.warning(f"Reader for extension '{ext_lower}' is already registered. Overwriting.")
        logger.debug(f"Registering data reader for extension: '{ext_lower}'")
        self._readers[ext_lower] = reader_callable

    def add_writer(self, extension: str, writer_callable: WriterCallable):
        """Register a custom data writer for a file extension."""
        ext_lower = extension.lower()
        if not ext_lower.startswith('.'):
             logger.warning(f"Writer extension '{extension}' should start with a dot (e.g., '.parquet'). Adding dot.")
             ext_lower = '.' + ext_lower
        if ext_lower in self._writers:
            logger.warning(f"Writer for extension '{ext_lower}' is already registered. Overwriting.")
        logger.debug(f"Registering data writer for extension: '{ext_lower}'")
        self._writers[ext_lower] = writer_callable

    def add_cli_command(self, command: ClickCommandType):
        """Register a custom Click command or group."""
        if click is None:
            logger.error("Cannot register CLI command: 'click' library is not installed.")
            return
        if not isinstance(command, (click.Command, click.Group)) and not callable(command):
             logger.error(f"Cannot register CLI command: Expected a Click Command/Group or callable, got {type(command)}.")
             return
        logger.debug(f"Registering CLI command: {getattr(command, 'name', '<callable>')}")
        self._cli_commands.append(command)

    # --- Methods to Get Extensions ---

    def get_filter(self, name: str) -> Optional[Callable]:
        """Get a registered filter by name."""
        return self._filters.get(name)

    def get_transform(self, name: str) -> Optional[Callable]:
        """Get a registered transform by name."""
        return self._transforms.get(name)

    def get_feature(self, name: str) -> Optional[Callable]:
        """Get a registered feature by name."""
        return self._features.get(name)

    def get_visualization(self, name: str) -> Optional[Callable]:
        """Get a registered visualization by name."""
        return self._visualizations.get(name)

    def get_effect(self, name: str) -> Optional[Callable]:
        """Get a registered audio effect by name."""
        return self._effects.get(name)

    def get_augmenter(self, name: str) -> Optional[Callable]:
        """Get a registered augmenter by name."""
        return self._augmenters.get(name)

    def get_reader(self, extension: str) -> Optional[ReaderCallable]:
        """Get a registered data reader by file extension (lowercase, including dot)."""
        return self._readers.get(extension.lower())

    def get_writer(self, extension: str) -> Optional[WriterCallable]:
        """Get a registered data writer by file extension (lowercase, including dot)."""
        return self._writers.get(extension.lower())

    # --- Methods to List Extensions ---

    def list_filters(self) -> List[str]:
        """Return a list of registered filter names."""
        return sorted(list(self._filters.keys()))

    def list_transforms(self) -> List[str]:
        """Return a list of registered transform names."""
        return sorted(list(self._transforms.keys()))

    def list_features(self) -> List[str]:
        """Return a list of registered feature names."""
        return sorted(list(self._features.keys()))

    def list_visualizations(self) -> List[str]:
        """Return a list of registered visualization names."""
        return sorted(list(self._visualizations.keys()))

    def list_effects(self) -> List[str]:
        """Return a list of registered audio effect names."""
        return sorted(list(self._effects.keys()))

    def list_augmenters(self) -> List[str]:
        """Return a list of registered augmenter names."""
        return sorted(list(self._augmenters.keys()))

    def list_readers(self) -> List[str]:
        """Return a list of registered reader extensions."""
        return sorted(list(self._readers.keys()))

    def list_writers(self) -> List[str]:
        """Return a list of registered writer extensions."""
        return sorted(list(self._writers.keys()))

    def get_cli_commands(self) -> List[ClickCommandType]:
        """Return the list of registered CLI commands/groups."""
        return self._cli_commands

    # --- Get All Extensions (Used by Feature Manager etc.) ---
    def get_all_extensions(self) -> Dict[str, Dict[str, Callable]]:
        """Return all registered extensions grouped by type."""
        # Note: Reader/Writer callables might have different signatures, so handle carefully if using this dict directly.
        return {
            "filters": self._filters.copy(),
            "transforms": self._transforms.copy(),
            "features": self._features.copy(),
            "visualizations": self._visualizations.copy(),
            "effects": self._effects.copy(),
            "augmenters": self._augmenters.copy(),
            "readers": self._readers.copy(), # Add readers
            "writers": self._writers.copy(), # Add writers
            # CLI commands are handled separately
        }
