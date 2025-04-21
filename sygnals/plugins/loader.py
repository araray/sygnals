# sygnals/plugins/loader.py

"""
Handles the discovery, loading, and registration of Sygnals plugins.
Includes discovery via entry points and local directories.
"""

import importlib
import importlib.metadata
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Type

import toml
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from packaging.version import Version, InvalidVersion

# Import necessary types and classes from within sygnals
from sygnals.config.models import SygnalsConfig # Use the correct config model import
from sygnals.version import __version__ as core_version

# Import from the same package
from .api import SygnalsPluginBase, PluginRegistry

logger = logging.getLogger(__name__)

# --- Constants ---
PLUGIN_MANIFEST_FILENAME = "plugin.toml"
PLUGIN_STATE_FILENAME = "plugins.yaml" # Simple state file (could use JSON/TOML)
ENTRY_POINT_GROUP = "sygnals.plugins" # Entry point group name

# --- Helper Functions ---

def _find_manifest_for_module(module) -> Optional[Path]:
    """
    Tries to find the plugin.toml manifest file relative to a loaded module.
    Assumes standard packaging structure (e.g., manifest in parent dir of package).
    """
    if not hasattr(module, '__file__') or module.__file__ is None:
        logger.warning(f"Cannot determine file path for module {module.__name__} to find manifest.")
        return None

    module_path = Path(module.__file__).parent
    # Search upwards from the module's directory for plugin.toml
    current_dir = module_path
    # Limit search depth to avoid searching the entire filesystem
    for _ in range(5): # Search up to 5 levels up
        manifest_path = current_dir / PLUGIN_MANIFEST_FILENAME
        if manifest_path.is_file():
            logger.debug(f"Found manifest for module {module.__name__} at {manifest_path}")
            return manifest_path
        if current_dir.parent == current_dir: # Reached root
            break
        current_dir = current_dir.parent

    logger.warning(f"Could not find {PLUGIN_MANIFEST_FILENAME} near module {module.__name__} path {module_path}.")
    return None


def _parse_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    """Parses a plugin.toml manifest file."""
    if not manifest_path.is_file():
        logger.warning(f"Plugin manifest not found: {manifest_path}")
        return None
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = toml.load(f)
        # Basic validation of required fields
        required = ['name', 'version', 'sygnals_api', 'entry_point']
        if not all(key in manifest_data for key in required):
            missing = [key for key in required if key not in manifest_data]
            logger.error(f"Manifest '{manifest_path}' missing required fields: {missing}")
            return None
        return manifest_data
    except toml.TomlDecodeError as e:
        logger.error(f"Error decoding manifest file '{manifest_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading manifest file '{manifest_path}': {e}")
        return None

def _check_compatibility(plugin_name: str, plugin_version: str, api_specifier: str, core_v: str) -> bool:
    """Checks if a plugin's API requirement is compatible with the core version."""
    try:
        spec = SpecifierSet(api_specifier)
        core = Version(core_v)
        if core in spec:
            logger.debug(f"Plugin '{plugin_name}' v{plugin_version} (requires API '{api_specifier}') is compatible with core v{core_v}.")
            return True
        else:
            logger.warning(f"Skipping plugin '{plugin_name}' v{plugin_version}: Incompatible API requirement. "
                           f"Requires '{api_specifier}', core is v{core_v}.")
            return False
    except (InvalidSpecifier, InvalidVersion) as e:
        logger.error(f"Skipping plugin '{plugin_name}' v{plugin_version}: Invalid version specifier "
                     f"in manifest ('{api_specifier}') or core version ('{core_v}'): {e}")
        return False
    except Exception as e:
         logger.error(f"Error checking compatibility for plugin '{plugin_name}': {e}")
         return False

def _import_plugin_entry_point(entry_point_str: str, manifest_path: Optional[Path] = None) -> Optional[Type[SygnalsPluginBase]]:
    """
    Imports the plugin's entry point class.
    Handles temporary sys.path modification for local plugins if manifest_path is provided.
    """
    module_str, class_name = entry_point_str.split(':')
    plugin_root_dir = manifest_path.parent if manifest_path else None
    needs_path_update = False
    original_sys_path = list(sys.path) # Store original path

    try:
        # If it's a local plugin, temporarily add its root to sys.path
        # to handle potential relative imports within the plugin code.
        if plugin_root_dir and str(plugin_root_dir) not in sys.path:
            # Also add the directory *containing* the plugin root,
            # in case the entry point is like 'plugin_pkg.submodule:Class'
            # and the plugin root itself is not directly importable.
            container_dir = plugin_root_dir.parent
            if str(container_dir) not in sys.path:
                 sys.path.insert(0, str(container_dir))
                 needs_path_update = True
                 logger.debug(f"Temporarily added {container_dir} to sys.path for plugin import.")
            # No longer adding plugin_root_dir directly, rely on container dir

        module = importlib.import_module(module_str)
        plugin_class = getattr(module, class_name, None)

        if plugin_class is None:
            logger.error(f"Entry point class '{class_name}' not found in module '{module_str}' "
                         f"(manifest: {manifest_path}).")
            return None
        # Check inheritance using issubclass
        if not issubclass(plugin_class, SygnalsPluginBase):
            logger.error(f"Entry point class '{entry_point_str}' does not inherit from SygnalsPluginBase "
                         f"(manifest: {manifest_path}).")
            return None
        return plugin_class
    except ImportError as e:
        logger.error(f"Failed to import plugin module '{module_str}': {e} (manifest: {manifest_path}). "
                     "Check PYTHONPATH and plugin structure.")
        # Print traceback for import errors to aid debugging
        traceback.print_exc()
        return None
    except Exception as e:
        logger.error(f"Error importing plugin entry point '{entry_point_str}': {e}", exc_info=True)
        return None
    finally:
        # Restore original sys.path if it was modified
        if needs_path_update:
             # Check if the path is still there before removing
             if container_dir and str(container_dir) in sys.path and sys.path[0] == str(container_dir):
                  sys.path.pop(0)
                  logger.debug(f"Removed {container_dir} from sys.path.")
             else:
                  # Restore from original if something went wrong
                  logger.warning(f"sys.path was modified unexpectedly during import of {module_str}. Restoring original path.")
                  sys.path = original_sys_path


# --- Plugin State Management (Basic) ---

def _load_plugin_state(config_dir: Path) -> Dict[str, bool]:
    """Loads enabled/disabled state from a simple YAML/JSON/TOML file."""
    state_file = config_dir / PLUGIN_STATE_FILENAME
    if not state_file.exists():
        return {} # Default to all enabled if state file doesn't exist

    try:
        # Using TOML for simplicity, could use YAML or JSON
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = toml.load(f)
        # Expecting format like: {'plugin_name': {'enabled': True/False}, ...}
        enabled_state = {name: data.get('enabled', True) for name, data in state_data.items() if isinstance(data, dict)}
        return enabled_state
    except Exception as e:
        logger.warning(f"Could not load plugin state file '{state_file}': {e}. Assuming all plugins enabled.")
        return {}

def _save_plugin_state(config_dir: Path, state: Dict[str, bool]):
    """Saves enabled/disabled state."""
    state_file = config_dir / PLUGIN_STATE_FILENAME
    try:
        # Format for saving: {'plugin_name': {'enabled': True/False}, ...}
        save_data = {name: {'enabled': enabled} for name, enabled in state.items()}
        with open(state_file, 'w', encoding='utf-8') as f:
            toml.dump(save_data, f)
    except Exception as e:
        logger.error(f"Could not save plugin state file '{state_file}': {e}")


# --- Plugin Loader Class ---

class PluginLoader:
    """Discovers, loads, and registers Sygnals plugins."""

    def __init__(self, config: SygnalsConfig, registry: PluginRegistry):
        self.config = config
        self.registry = registry
        self.loaded_plugins: Dict[str, SygnalsPluginBase] = {}
        # Store manifest data and path together
        self.plugin_manifests: Dict[str, Tuple[Dict[str, Any], Path]] = {} # name -> (manifest_data, manifest_path)
        self.plugin_sources: Dict[str, str] = {} # Store source ('entry_point' or 'local')
        self.plugin_enabled_state: Dict[str, bool] = {} # Store enabled/disabled status

    def _load_and_register(self, plugin_name: str):
        """
        Loads, validates, and registers a single plugin *after* its manifest
        has been parsed and stored. Uses data stored in self.plugin_manifests.
        """
        # Manifest data and path should already be in self.plugin_manifests
        manifest_tuple = self.plugin_manifests.get(plugin_name)
        if not manifest_tuple:
             logger.error(f"Internal error: Manifest data/path not found for '{plugin_name}' during load attempt.")
             return # Should not happen if called correctly

        manifest_data, manifest_path = manifest_tuple
        plugin_version = manifest_data['version']
        api_spec = manifest_data['sygnals_api']
        entry_point_str = manifest_data['entry_point']

        # 1. Check compatibility
        if not _check_compatibility(plugin_name, plugin_version, api_spec, core_version):
            return # Skip incompatible plugin

        # 2. Check enabled state
        if not self.plugin_enabled_state.get(plugin_name, True): # Default to enabled
             logger.info(f"Skipping disabled plugin '{plugin_name}' v{plugin_version}.")
             return # Skip disabled plugin

        # 3. Import entry point
        # Pass manifest_path to helper for local plugins sys.path handling
        plugin_class = _import_plugin_entry_point(entry_point_str, manifest_path)
        if plugin_class is None:
            return # Skip if import failed

        # 4. Instantiate plugin
        try:
            plugin_instance = plugin_class()
            # Verify instance properties match manifest (optional sanity check)
            if plugin_instance.name != plugin_name or plugin_instance.version != plugin_version:
                 logger.warning(f"Manifest/Instance mismatch for plugin '{plugin_name}'. "
                                f"Manifest: ({plugin_name}, {plugin_version}), "
                                f"Instance: ({plugin_instance.name}, {plugin_instance.version}). Using instance values.")
                 # Update internal name based on instance if needed? For now, just warn.
                 # plugin_name = plugin_instance.name # Be careful with this
        except Exception as e:
            logger.error(f"Failed to instantiate plugin '{plugin_name}' from {entry_point_str}: {e}", exc_info=True)
            return

        # 5. Call setup hook
        try:
            plugin_instance.setup(self.config.model_dump()) # Pass full config dict
        except Exception as e:
            logger.error(f"Error during setup() for plugin '{plugin_name}': {e}", exc_info=True)
            return # Skip registration if setup fails

        # 6. Call registration hooks
        try:
            logger.debug(f"Registering extensions for plugin '{plugin_name}'...")
            # Call all registration hooks, including the new ones
            plugin_instance.register_filters(self.registry)
            plugin_instance.register_transforms(self.registry)
            plugin_instance.register_feature_extractors(self.registry)
            plugin_instance.register_visualizations(self.registry)
            plugin_instance.register_audio_effects(self.registry)
            plugin_instance.register_augmenters(self.registry)
            plugin_instance.register_data_readers(self.registry) # Call new reader hook
            plugin_instance.register_data_writers(self.registry) # Call new writer hook
            plugin_instance.register_cli_commands(self.registry)
        except Exception as e:
            logger.error(f"Error during registration hooks for plugin '{plugin_name}': {e}", exc_info=True)
            # Decide if teardown should be called if registration fails partially
            try:
                plugin_instance.teardown()
            except Exception as te:
                 logger.error(f"Error during teardown after failed registration for plugin '{plugin_name}': {te}")
            return # Skip if registration fails

        # 7. Store successful load
        self.loaded_plugins[plugin_name] = plugin_instance
        # Manifest and source already stored during discovery
        self.registry.add_plugin_name(plugin_name) # Add name to registry's list
        logger.info(f"Successfully loaded and registered plugin: '{plugin_name}' v{plugin_version} ({self.plugin_sources.get(plugin_name, '?')})")


    def discover_and_load(self):
        """Discover plugins from entry points and local directory, then load and register them."""
        logger.info("Starting plugin discovery...")
        # Clear previous discovery results before starting new discovery
        self.plugin_manifests.clear()
        self.plugin_sources.clear()
        # Keep loaded_plugins until teardown, but registry should be fresh if loader is re-run

        # --- Load Enabled State ---
        # State file location - use config directory parent as a convention
        state_file_dir = self.config.paths.plugin_dir.parent
        self.plugin_enabled_state = _load_plugin_state(state_file_dir)
        logger.debug(f"Loaded plugin enabled/disabled state from {state_file_dir}: {self.plugin_enabled_state}")

        # --- Discover Entry Points ---
        logger.debug(f"Scanning for entry points in group '{ENTRY_POINT_GROUP}'...")
        try:
            # Use importlib.metadata to find entry points
            entry_points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
            for ep in entry_points:
                logger.debug(f"Found entry point: name='{ep.name}', value='{ep.value}'")
                try:
                    # Load the module associated with the entry point to find its manifest
                    plugin_module = ep.load() # This actually loads the entry point function/class
                    # We need the *module* containing the entry point
                    # This is tricky, ep.load() gives the object itself.
                    # We might need to parse ep.value to get the module path.
                    if ':' in ep.value:
                        module_path_str = ep.value.split(':', 1)[0]
                        try:
                            containing_module = importlib.import_module(module_path_str)
                            manifest_path = _find_manifest_for_module(containing_module)
                            if manifest_path:
                                manifest_data = _parse_manifest(manifest_path)
                                if manifest_data:
                                    plugin_name = manifest_data['name']
                                    # Check if name matches entry point name (convention)
                                    if plugin_name != ep.name:
                                        logger.warning(f"Entry point name '{ep.name}' differs from manifest name '{plugin_name}' in {manifest_path}. Using manifest name.")

                                    if plugin_name in self.plugin_manifests:
                                        logger.warning(f"Duplicate plugin name '{plugin_name}' found (entry point vs local/other entry point). Prioritizing first discovery.")
                                    else:
                                        self.plugin_manifests[plugin_name] = (manifest_data, manifest_path)
                                        self.plugin_sources[plugin_name] = 'entry_point'
                                else:
                                     logger.warning(f"Could not parse manifest found at {manifest_path} for entry point '{ep.name}'.")
                            else:
                                 logger.warning(f"Could not find manifest file near module for entry point '{ep.name}'. Skipping.")
                        except ImportError:
                             logger.error(f"Could not import module '{module_path_str}' for entry point '{ep.name}'. Skipping.")
                        except Exception as e:
                             logger.error(f"Error processing entry point '{ep.name}': {e}", exc_info=True)

                    else:
                        logger.warning(f"Entry point value '{ep.value}' for '{ep.name}' is not in expected 'module:object' format. Cannot find manifest.")

                except Exception as e:
                    logger.error(f"Failed to load or process entry point '{ep.name}': {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error occurred during entry point discovery: {e}", exc_info=True)

        # --- Discover Local Plugins ---
        local_plugin_dir = self.config.paths.plugin_dir
        if local_plugin_dir.exists() and local_plugin_dir.is_dir():
            logger.debug(f"Scanning local plugin directory: {local_plugin_dir}")
            for item in local_plugin_dir.iterdir():
                if item.is_dir(): # Each plugin in its own subdirectory
                    manifest_path = item / PLUGIN_MANIFEST_FILENAME
                    if manifest_path.exists():
                        logger.debug(f"Found potential local plugin manifest: {manifest_path}")
                        manifest_data = _parse_manifest(manifest_path)
                        if manifest_data:
                            plugin_name = manifest_data['name']
                            if plugin_name in self.plugin_manifests: # Check if already discovered (e.g., via entry point)
                                logger.warning(f"Duplicate plugin name '{plugin_name}' found (local vs entry point/other local). Prioritizing first discovery.")
                            else:
                                self.plugin_manifests[plugin_name] = (manifest_data, manifest_path)
                                self.plugin_sources[plugin_name] = 'local'
        else:
            logger.debug(f"Local plugin directory not found or not a directory: {local_plugin_dir}")


        # --- Load and Register Discovered Plugins ---
        logger.info(f"Found {len(self.plugin_manifests)} potential plugin manifests. Attempting to load...")
        # Iterate through the names found during discovery
        # Use list() to create a copy, allowing modification during iteration if needed (though not done here)
        for name in list(self.plugin_manifests.keys()):
             # Attempt to load and register this specific plugin using stored manifest info
             self._load_and_register(name)


        logger.info(f"Plugin loading complete. {len(self.loaded_plugins)} plugins loaded successfully.")

    def get_plugin_info(self) -> List[Dict[str, Any]]:
        """Return metadata about all discovered plugins (loaded or disabled/failed)."""
        info_list = []
        all_discovered_names = set(self.plugin_manifests.keys())

        for name in sorted(list(all_discovered_names)):
            manifest_tuple = self.plugin_manifests.get(name)
            manifest = manifest_tuple[0] if manifest_tuple else {}
            source = self.plugin_sources.get(name, 'unknown')
            is_loaded = name in self.loaded_plugins
            is_enabled = self.plugin_enabled_state.get(name, True) # Default enabled

            # Determine status more accurately
            status = "unknown"
            if not is_enabled:
                 status = "disabled"
            elif is_loaded:
                 status = "loaded"
            elif manifest: # Manifest exists but not loaded/disabled
                 # Check compatibility again to determine reason
                 if _check_compatibility(name, manifest.get('version','?'), manifest.get('sygnals_api','?'), core_version):
                     # Compatible but failed somewhere after compatibility check
                     status = "error/load_failed"
                 else:
                     status = "error/incompatible"
            else:
                 status = "error/no_manifest" # Should not happen if name is from plugin_manifests keys

            info_list.append({
                "name": name,
                "version": manifest.get("version", "N/A"),
                "api_required": manifest.get("sygnals_api", "N/A"),
                "status": status,
                "source": source,
                "description": manifest.get("description", ""),
                "entry_point": manifest.get("entry_point", "N/A"),
            })
        return info_list

    def enable_plugin(self, name: str) -> bool:
        """Mark a plugin as enabled."""
        # Check manifests first, as plugin might be discovered but not loaded
        if name not in self.plugin_manifests:
             logger.error(f"Cannot enable plugin '{name}': Not found (no manifest discovered).")
             return False
        logger.info(f"Enabling plugin '{name}'.")
        self.plugin_enabled_state[name] = True
        # Use the same logic for state file location as in load
        state_file_dir = self.config.paths.plugin_dir.parent
        _save_plugin_state(state_file_dir, self.plugin_enabled_state)
        return True

    def disable_plugin(self, name: str) -> bool:
        """Mark a plugin as disabled."""
        # Check manifests first
        if name not in self.plugin_manifests:
             logger.error(f"Cannot disable plugin '{name}': Not found (no manifest discovered).")
             return False
        logger.info(f"Disabling plugin '{name}'.")
        self.plugin_enabled_state[name] = False
        # Use the same logic for state file location as in load
        state_file_dir = self.config.paths.plugin_dir.parent
        _save_plugin_state(state_file_dir, self.plugin_enabled_state)
        return True

    def call_teardown_hooks(self):
        """Call the teardown method for all loaded plugins."""
        logger.debug("Calling teardown hooks for loaded plugins...")
        # Iterate safely over a copy of the keys in case teardown modifies the dict
        loaded_names = list(self.loaded_plugins.keys())
        for name in loaded_names:
            instance = self.loaded_plugins.get(name)
            if instance:
                try:
                    instance.teardown()
                except Exception as e:
                    logger.error(f"Error during teardown for plugin '{name}': {e}", exc_info=True)
