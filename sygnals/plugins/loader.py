# sygnals/plugins/loader.py

"""
Handles the discovery, loading, and registration of Sygnals plugins.
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

from sygnals.config import SygnalsConfig
from sygnals.version import __version__ as core_version

# Import from the same package
from .api import SygnalsPluginBase, PluginRegistry

logger = logging.getLogger(__name__)

# --- Constants ---
PLUGIN_MANIFEST_FILENAME = "plugin.toml"
PLUGIN_STATE_FILENAME = "plugins.yaml" # Simple state file (could use JSON/TOML)

# --- Helper Functions ---

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

def _import_plugin_entry_point(entry_point_str: str, manifest_path: Path) -> Optional[Type[SygnalsPluginBase]]:
    """Imports the plugin's entry point class."""
    module_str, class_name = entry_point_str.split(':')
    try:
        # If the plugin is local, we might need to add its directory to sys.path temporarily
        # to allow relative imports within the plugin itself.
        plugin_root_dir = manifest_path.parent
        needs_path_update = str(plugin_root_dir) not in sys.path
        if needs_path_update:
            sys.path.insert(0, str(plugin_root_dir))
            logger.debug(f"Temporarily added {plugin_root_dir} to sys.path for plugin import.")

        module = importlib.import_module(module_str)
        plugin_class = getattr(module, class_name, None)

        if needs_path_update:
            sys.path.pop(0) # Remove path after import
            logger.debug(f"Removed {plugin_root_dir} from sys.path.")

        if plugin_class is None:
            logger.error(f"Entry point class '{class_name}' not found in module '{module_str}' "
                         f"(manifest: {manifest_path}).")
            return None
        if not issubclass(plugin_class, SygnalsPluginBase):
            logger.error(f"Entry point class '{entry_point_str}' does not inherit from SygnalsPluginBase "
                         f"(manifest: {manifest_path}).")
            return None
        return plugin_class
    except ImportError as e:
        logger.error(f"Failed to import plugin module '{module_str}': {e} (manifest: {manifest_path}). "
                     "Check PYTHONPATH and plugin structure.")
        return None
    except Exception as e:
        logger.error(f"Error importing plugin entry point '{entry_point_str}': {e}", exc_info=True)
        return None

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
        self.plugin_manifests: Dict[str, Dict[str, Any]] = {} # Store manifest data by plugin name
        self.plugin_sources: Dict[str, str] = {} # Store source ('entry_point' or 'local')
        self.plugin_enabled_state: Dict[str, bool] = {} # Store enabled/disabled status

    def _load_and_register(self, plugin_name: str, manifest_path: Path):
        """
        Loads, validates, and registers a single plugin *after* its manifest
        has been parsed and stored.
        """
        # Manifest data should already be in self.plugin_manifests
        manifest_data = self.plugin_manifests.get(plugin_name)
        if not manifest_data:
             logger.error(f"Internal error: Manifest data not found for '{plugin_name}' during load attempt.")
             return # Should not happen if called correctly

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
            plugin_instance.register_filters(self.registry)
            plugin_instance.register_transforms(self.registry)
            plugin_instance.register_feature_extractors(self.registry)
            plugin_instance.register_visualizations(self.registry)
            plugin_instance.register_audio_effects(self.registry)
            plugin_instance.register_augmenters(self.registry)
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
        discovered_manifest_paths: Dict[str, Tuple[Path, str]] = {} # name -> (path, source)

        # --- Load Enabled State ---
        # State file should be relative to config dir, not necessarily plugin dir parent
        # Using config.paths.plugin_dir.parent assumes state file is alongside plugin dir folder
        # A better location might be ~/.config/sygnals/ or similar. Using plugin_dir.parent for now.
        state_file_dir = self.config.paths.plugin_dir.parent
        self.plugin_enabled_state = _load_plugin_state(state_file_dir)
        logger.debug(f"Loaded plugin enabled/disabled state from {state_file_dir}: {self.plugin_enabled_state}")

        # --- Discover Entry Points ---
        # (Skipping entry point discovery for now as it's complex to map back to manifests reliably)
        logger.debug("Entry point discovery currently skipped.")

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
                            if plugin_name in self.plugin_manifests: # Check if already discovered
                                logger.warning(f"Duplicate plugin name '{plugin_name}' found. Prioritizing first discovery.")
                            else:
                                # FIX: Store manifest and source immediately after parsing
                                self.plugin_manifests[plugin_name] = manifest_data
                                self.plugin_sources[plugin_name] = 'local'
                                discovered_manifest_paths[plugin_name] = (manifest_path, 'local') # Keep track for loading loop
        else:
            logger.debug(f"Local plugin directory not found or not a directory: {local_plugin_dir}")


        # --- Load and Register Discovered Plugins ---
        logger.info(f"Found {len(self.plugin_manifests)} potential plugin manifests. Attempting to load...")
        # Iterate through the names found during discovery
        for name in list(self.plugin_manifests.keys()): # Use list to avoid dict size change issues
             if name in discovered_manifest_paths: # Check if it was found in this run's discovery
                 path, source = discovered_manifest_paths[name]
                 # Attempt to load and register this specific plugin
                 self._load_and_register(name, path)
             else:
                 # This case might occur if manifest was stored from a previous run but not found now
                 logger.debug(f"Plugin '{name}' manifest exists but was not discovered in this run. Skipping load attempt.")


        logger.info(f"Plugin loading complete. {len(self.loaded_plugins)} plugins loaded successfully.")

    def get_plugin_info(self) -> List[Dict[str, Any]]:
        """Return metadata about all discovered plugins (loaded or disabled)."""
        info_list = []
        all_discovered_names = set(self.plugin_manifests.keys())

        for name in sorted(list(all_discovered_names)):
            manifest = self.plugin_manifests.get(name, {})
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
                     status = "error/load_failed" # Compatible but failed load/import/setup
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

    def enable_plugin(self, name: str):
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

    def disable_plugin(self, name: str):
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
        for name, instance in self.loaded_plugins.items():
            try:
                instance.teardown()
            except Exception as e:
                logger.error(f"Error during teardown for plugin '{name}': {e}", exc_info=True)

# --- Global instance (optional, depends on application structure) ---
# global_plugin_registry = PluginRegistry()
