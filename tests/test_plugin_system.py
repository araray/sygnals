# tests/test_plugin_system.py

"""
Tests for the Sygnals plugin system (loading, registration, management).

Ensures plugins are discovered, loaded correctly based on compatibility and state,
registration hooks function as expected, and CLI commands for plugin management work.
Verifies error handling and state management for plugins.
"""

import pytest
import sys
import traceback # Added for detailed exception printing in tests
from pathlib import Path
import toml # For creating dummy manifests
import logging # Import logging for caplog test
from typing import Optional, Dict, Any, List, Tuple, Callable # Import necessary types
import numpy as np # <-- **Import numpy for use in test assertions**
import pandas as pd # <-- Added import for dummy handlers

# Import components to test
from sygnals.config.models import SygnalsConfig, PathsConfig
from sygnals.plugins.api import PluginRegistry, SygnalsPluginBase
from sygnals.plugins.loader import PluginLoader, _load_plugin_state, _save_plugin_state, PLUGIN_STATE_FILENAME
from sygnals.plugins.scaffold import create_plugin_scaffold
from sygnals.version import __version__ as core_version

from click.testing import CliRunner
from sygnals.cli.main import cli # Import main CLI entry point
# Import base_cmd to potentially patch the loader there
from sygnals.cli import base_cmd as sygnals_base_cmd

# --- Test Fixtures ---

@pytest.fixture
def plugin_registry() -> PluginRegistry:
    """Provides a fresh PluginRegistry instance for each test."""
    # Reset the global registry instance potentially used by base_cmd
    # This ensures test isolation.
    sygnals_base_cmd.plugin_registry = PluginRegistry()
    return sygnals_base_cmd.plugin_registry

@pytest.fixture
def base_config(tmp_path: Path) -> SygnalsConfig:
    """Provides a base SygnalsConfig pointing plugin dir to a temp path."""
    config = SygnalsConfig()
    # Override paths to use temp directory for test isolation
    plugin_dir_path = tmp_path / "sygnals_test_plugins"
    # Use the *parent* of the plugin dir for the state file, as per loader logic
    state_dir_path = plugin_dir_path.parent
    plugin_dir_path.mkdir(parents=True, exist_ok=True)
    state_dir_path.mkdir(parents=True, exist_ok=True) # Ensure state dir exists

    config.paths = PathsConfig(
        plugin_dir=plugin_dir_path,
        cache_dir=tmp_path / ".sygnals_cache",
        output_dir=tmp_path / "sygnals_output",
        log_directory=tmp_path / "sygnals_logs"
    )
    # Store state dir path separately for easy access in tests
    # This is a helper attribute for the test setup itself.
    setattr(config, '_test_state_dir', state_dir_path)
    return config

@pytest.fixture
def plugin_loader(base_config: SygnalsConfig, plugin_registry: PluginRegistry) -> PluginLoader:
    """
    Provides a PluginLoader instance initialized with temp config and fresh registry.

    NOTE: This fixture does NOT call discover_and_load automatically.
          Tests requiring discovery should call it explicitly after setting up plugins.
    """
    loader = PluginLoader(base_config, plugin_registry)
    # Also update the global instance that might be used by CLI commands if not patched
    sygnals_base_cmd.plugin_loader = loader
    return loader

@pytest.fixture
def teardown_sys_path():
    """Fixture to clean up sys.path modifications made during tests."""
    original_sys_path = list(sys.path)
    yield
    # Restore sys.path, be careful if other tests modify it concurrently
    current_sys_path = list(sys.path)
    new_sys_path = [p for p in current_sys_path if p in original_sys_path]
    # Add back original paths that might have been removed, preserving order roughly
    for p in original_sys_path:
        if p not in new_sys_path:
            # Find original position if possible (approximate)
            try:
                 original_index = original_sys_path.index(p)
                 # Avoid inserting duplicates if path somehow got added back
                 if p not in new_sys_path:
                     new_sys_path.insert(min(original_index, len(new_sys_path)), p)
            except ValueError:
                 if p not in new_sys_path:
                     new_sys_path.append(p) # Append if original index not found
    sys.path = new_sys_path


# --- Helper Function to Create Dummy Plugin (Enhanced) ---
def create_dummy_plugin(
    plugin_dir: Path,
    state_dir: Path, # Directory to store enabled/disabled state
    name: str,
    version: str = "0.1.0",
    api_req: str = f">={core_version},<2.0.0", # Compatible API req by default
    entry_point_content: str = "", # Extra Python code for plugin.py
    dependencies: list = [], # Informational dependencies for plugin.toml
    register_hooks: Optional[Dict[str, str]] = None, # Code snippets for registration hooks
    is_enabled: bool = True, # Whether the plugin should be marked as enabled
    create_pyproject: bool = False # Whether to create a dummy pyproject.toml
):
    """
    Creates a dummy plugin directory structure and files for testing purposes.

    Generates plugin.toml, the package directory, __init__.py, plugin.py (with
    optional hook implementations), and optionally pyproject.toml. Also updates
    the plugin enabled/disabled state file.

    Args:
        plugin_dir: The base directory where the plugin's root folder will be created.
        state_dir: The directory where the plugin state file (plugins.yaml/toml) resides.
        name: The name of the dummy plugin (e.g., "my-test-plugin").
        version: The plugin version string.
        api_req: The sygnals_api requirement string for plugin.toml.
        entry_point_content: Additional Python code to append to the generated plugin.py.
        dependencies: List of informational dependencies for plugin.toml.
        register_hooks: A dictionary mapping registration hook names (e.g., "register_filters")
                        to Python code snippets to be inserted into the hook implementation.
        is_enabled: If True, marks the plugin as enabled in the state file; otherwise, disabled.
        create_pyproject: If True, generates a minimal pyproject.toml for testing entry point discovery.

    Returns:
        Tuple[Path, str]: The root path of the created plugin directory and the entry point string.
    """
    plugin_root = plugin_dir / name
    # Ensure clean state if directory exists from previous failed run
    if plugin_root.exists():
        import shutil
        shutil.rmtree(plugin_root)
    plugin_root.mkdir()

    # Generate package and class names from plugin name
    package_name = name.replace("-", "_")
    class_name = "".join(part.capitalize() for part in name.split('-')) + "Plugin"
    entry_module = package_name
    entry_point_path = f"{entry_module}.plugin:{class_name}"

    # Create plugin.toml
    manifest_content = f"""
name = "{name}"
version = "{version}"
sygnals_api = "{api_req}"
description = "A dummy test plugin: {name}"
entry_point = "{entry_point_path}"
dependencies = {dependencies}
"""
    (plugin_root / "plugin.toml").write_text(manifest_content, encoding='utf-8')

    # Create Python package structure
    package_dir = plugin_root / entry_module
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(f"# Init for {package_name}", encoding='utf-8')

    # Generate plugin.py content
    hook_implementations = ""
    if register_hooks:
        for hook_name, hook_code in register_hooks.items():
            # Indent the provided hook code correctly within the method body
            indented_code = "\n".join("        " + line for line in hook_code.strip().split("\n"))
            hook_implementations += f"""
    def {hook_name}(self, registry):
        # Log entry into the hook for debugging test runs
        logger.info(f"Plugin '{name}' executing hook '{hook_name}'")
{indented_code}
"""
    # Ensure 'import numpy as np' is present in the template
    plugin_py_content = f"""
# Dummy plugin file for testing: {name}
import logging
import numpy as np # <-- Ensure this import is present
from pathlib import Path
import pandas as pd
from sygnals.plugins.api import SygnalsPluginBase, PluginRegistry
# Add other common imports needed by hook code snippets if necessary

# Simple callables/handlers for registration testing
def dummy_callable(*args, **kwargs):
    # Basic callable placeholder
    pass

def dummy_reader(path, **kwargs):
    # Simple reader returning a dict with numpy array
    # Requires numpy to be imported above.
    return {{'data': np.array([1])}}

def dummy_writer(data, path, **kwargs):
    # Simple writer placeholder
    pass

logger = logging.getLogger(__name__)

class {class_name}(SygnalsPluginBase):
    # Plugin class implementation

    @property
    def name(self) -> str:
        # Returns the plugin name
        return "{name}"
    @property
    def version(self) -> str:
        # Returns the plugin version
        return "{version}"

    def setup(self, config):
        # Setup hook implementation
        logger.info(f"Plugin '{{self.name}}' setup called.") # Escaped braces
    def teardown(self):
        # Teardown hook implementation
        logger.info(f"Plugin '{{self.name}}' teardown called.") # Escaped braces

{hook_implementations} # Insert generated hook implementations here

{entry_point_content} # Add extra content if needed
"""
    (package_dir / "plugin.py").write_text(plugin_py_content, encoding='utf-8')

    # Create pyproject.toml (optional)
    if create_pyproject:
         pyproject_content = f"""
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "{version}"

# Entry point definition for discovery
[project.entry-points."sygnals.plugins"]
{name} = "{entry_point_path}"
"""
         (plugin_root / "pyproject.toml").write_text(pyproject_content, encoding='utf-8')

    # Update plugin enabled/disabled state file
    # Use the provided state_dir
    current_state = _load_plugin_state(state_dir)
    current_state[name] = is_enabled
    _save_plugin_state(state_dir, current_state)

    return plugin_root, entry_point_path


# --- Test Cases ---

def test_load_valid_local_plugin_registration(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading a valid local plugin and verify its registration hook is called successfully."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir') # Get state dir from config fixture
    plugin_name = "test-register-filter"
    # Define the registration code snippet for the filter hook
    hook_code = 'registry.add_filter("my_test_filter", dummy_callable)'
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, register_hooks={"register_filters": hook_code})

    # Explicitly call discover_and_load *after* creating the plugin
    plugin_loader.discover_and_load()

    # Assertions
    assert plugin_name in plugin_loader.loaded_plugins, "Plugin should be loaded."
    assert plugin_loader.registry.get_filter("my_test_filter") is not None, "Filter should be registered."
    assert plugin_loader.registry.list_filters() == ["my_test_filter"], "Registry should list the registered filter."
    assert plugin_loader.registry.loaded_plugin_names == [plugin_name], "Registry should record the loaded plugin name."


def test_load_plugin_registering_multiple_types(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading a plugin that registers multiple types of extensions (feature and transform)."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-multi-register"
    # Define hooks for registering a feature and a transform
    hooks = {
        "register_feature_extractors": 'registry.add_feature("multi_feature", lambda x: x+1)',
        "register_transforms": 'registry.add_transform("multi_transform", lambda x: x*2)'
    }
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, register_hooks=hooks)

    # Explicitly call discover_and_load
    plugin_loader.discover_and_load()

    # Assertions
    assert plugin_name in plugin_loader.loaded_plugins, "Plugin should be loaded."
    assert plugin_loader.registry.get_feature("multi_feature") is not None, "Feature should be registered."
    assert plugin_loader.registry.get_transform("multi_transform") is not None, "Transform should be registered."
    assert plugin_loader.registry.list_features() == ["multi_feature"], "Registry should list the feature."
    assert plugin_loader.registry.list_transforms() == ["multi_transform"], "Registry should list the transform."
    assert plugin_loader.registry.list_filters() == [], "No filters should be registered by this plugin."


def test_load_multiple_plugins_registration(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading multiple plugins and check combined registry content from both."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin1_name = "plugin-one"
    plugin2_name = "plugin-two"

    # Create two plugins, each registering a different type of extension
    create_dummy_plugin(plugin_dir, state_dir, plugin1_name, register_hooks={"register_filters": 'registry.add_filter("filter_one", dummy_callable)'})
    create_dummy_plugin(plugin_dir, state_dir, plugin2_name, register_hooks={"register_feature_extractors": 'registry.add_feature("feature_two", dummy_callable)'})

    # Explicitly call discover_and_load
    plugin_loader.discover_and_load()

    # Assertions
    assert plugin1_name in plugin_loader.loaded_plugins, "Plugin 1 should be loaded."
    assert plugin2_name in plugin_loader.loaded_plugins, "Plugin 2 should be loaded."
    assert plugin_loader.registry.get_filter("filter_one") is not None, "Filter from plugin 1 should be registered."
    assert plugin_loader.registry.get_feature("feature_two") is not None, "Feature from plugin 2 should be registered."
    assert sorted(plugin_loader.registry.loaded_plugin_names) == sorted([plugin1_name, plugin2_name]), "Registry should record both loaded plugins."


def test_registration_overwriting(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path, caplog):
    """Test that registering the same extension name logs a warning and overwrites the previous registration."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin1_name = "plugin-overwriter-1"
    plugin2_name = "plugin-overwriter-2"
    feature_name = "shared_feature" # The name to be overwritten

    # Plugin 1 registers the feature
    create_dummy_plugin(plugin_dir, state_dir, plugin1_name, register_hooks={"register_feature_extractors": f'registry.add_feature("{feature_name}", lambda: 1)'})
    # Plugin 2 registers the same feature name
    create_dummy_plugin(plugin_dir, state_dir, plugin2_name, register_hooks={"register_feature_extractors": f'registry.add_feature("{feature_name}", lambda: 2)'})

    # Capture log messages at WARNING level or higher
    with caplog.at_level(logging.WARNING):
        # Explicitly call discover_and_load
        plugin_loader.discover_and_load()

    # Assertions
    assert plugin1_name in plugin_loader.loaded_plugins, "Plugin 1 should be loaded."
    assert plugin2_name in plugin_loader.loaded_plugins, "Plugin 2 should be loaded."

    # Check that the warning was logged
    assert f"Feature '{feature_name}' is already registered. Overwriting." in caplog.text, "Warning message for overwriting not found in logs."

    # Check that the feature callable corresponds to the *last* loaded plugin
    registered_feature = plugin_loader.registry.get_feature(feature_name)
    assert registered_feature is not None, "Shared feature should still be registered."
    assert registered_feature() == 2, "Registered feature should be the one from the last loaded plugin."


# --- Tests for Data Handlers ---

def test_load_plugin_registering_reader_writer(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading a plugin that registers data readers and writers."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-io-plugin"
    # Define hooks for reader and writer registration
    hooks = {
        "register_data_readers": 'registry.add_reader(".custom", dummy_reader)',
        "register_data_writers": 'registry.add_writer(".custom", dummy_writer)',
    }
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, register_hooks=hooks)

    # Explicitly call discover_and_load
    plugin_loader.discover_and_load()

    # Assertions
    assert plugin_name in plugin_loader.loaded_plugins, "IO plugin should be loaded."
    # Check registry for the reader/writer
    reader = plugin_loader.registry.get_reader(".custom")
    writer = plugin_loader.registry.get_writer(".custom")
    assert reader is not None and callable(reader), "Reader should be registered and callable."
    assert writer is not None and callable(writer), "Writer should be registered and callable."
    assert plugin_loader.registry.list_readers() == [".custom"], "Registry should list the reader extension."
    assert plugin_loader.registry.list_writers() == [".custom"], "Registry should list the writer extension."

    # Verify the dummy reader works via registry
    # The dummy_reader is defined within the create_dummy_plugin helper
    # Ensure np is imported in the template string
    read_result = reader(path="dummy")
    assert isinstance(read_result, dict) and 'data' in read_result
    # *** FIX: Check type using np imported at the top of this test file ***
    assert isinstance(read_result['data'], np.ndarray)
    assert read_result['data'][0] == 1, "Dummy reader did not return expected value."


def test_data_handler_registration_overwriting(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path, caplog):
    """Test overwriting data handlers for the same extension and check warnings."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin1_name = "plugin-io-1"
    plugin2_name = "plugin-io-2"
    extension = ".mydata"

    # Plugin 1 registers reader/writer
    hooks1 = {
        "register_data_readers": f'registry.add_reader("{extension}", lambda path: 1)',
        "register_data_writers": f'registry.add_writer("{extension}", lambda data, path: None)',
    }
    create_dummy_plugin(plugin_dir, state_dir, plugin1_name, register_hooks=hooks1)
    # Plugin 2 registers reader/writer for the same extension
    hooks2 = {
        "register_data_readers": f'registry.add_reader("{extension}", lambda path: 2)',
        "register_data_writers": f'registry.add_writer("{extension}", lambda data, path: None)',
    }
    create_dummy_plugin(plugin_dir, state_dir, plugin2_name, register_hooks=hooks2)

    # Capture warning logs
    with caplog.at_level(logging.WARNING):
        plugin_loader.discover_and_load()

    # Check warnings
    assert f"Reader for extension '{extension}' is already registered. Overwriting." in caplog.text, "Reader overwrite warning missing."
    assert f"Writer for extension '{extension}' is already registered. Overwriting." in caplog.text, "Writer overwrite warning missing."

    # Check that the handler from the second plugin is active
    reader = plugin_loader.registry.get_reader(extension)
    assert reader is not None, "Reader should still be registered."
    assert reader(path="dummy") == 2, "Active reader should be from the second plugin."


# --- Existing Tests (Keep and Ensure They Still Pass) ---

def test_skip_incompatible_plugin(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test that an incompatible plugin (based on sygnals_api) is skipped."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-incompatible"
    # Require a future API version that doesn't exist
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, api_req=">=99.0.0")

    # Explicitly call discover_and_load
    plugin_loader.discover_and_load()

    # Assertions
    assert plugin_name not in plugin_loader.loaded_plugins, "Incompatible plugin should not be loaded."
    assert plugin_name in plugin_loader.plugin_manifests, "Manifest should be stored even if incompatible."
    assert plugin_loader.plugin_sources[plugin_name] == 'local', "Source should be recorded."
    assert not plugin_loader.registry.loaded_plugin_names, "Registry should not list the incompatible plugin as loaded."


def test_skip_disabled_plugin(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test that a disabled plugin is skipped during loading."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-disabled"
    # Create the plugin but mark it as disabled in the state file
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, is_enabled=False)

    # Explicitly call discover_and_load *after* creating the plugin and setting state
    plugin_loader.discover_and_load()

    # Assertions
    assert plugin_name not in plugin_loader.loaded_plugins, "Disabled plugin should not be loaded."
    assert plugin_name in plugin_loader.plugin_manifests, "Manifest should be found."
    assert plugin_loader.plugin_sources[plugin_name] == 'local', "Source should be recorded."
    assert not plugin_loader.registry.loaded_plugin_names, "Registry should not list the disabled plugin as loaded."


def test_handle_missing_manifest(plugin_loader: PluginLoader, base_config: SygnalsConfig):
    """Test discovery when a plugin directory lacks a manifest file."""
    plugin_dir = base_config.paths.plugin_dir
    # Create a directory that looks like a plugin but has no plugin.toml
    (plugin_dir / "no-manifest-plugin").mkdir()
    (plugin_dir / "no-manifest-plugin" / "plugin.py").touch()

    plugin_loader.discover_and_load()
    # Assertions
    assert not plugin_loader.loaded_plugins, "No plugins should be loaded."
    assert "no-manifest-plugin" not in plugin_loader.plugin_manifests, "Plugin without manifest should not be recorded in manifests."


def test_handle_invalid_manifest(plugin_loader: PluginLoader, base_config: SygnalsConfig):
    """Test loading when a manifest file is invalid (e.g., missing required fields)."""
    plugin_dir = base_config.paths.plugin_dir
    plugin_name = "test-invalid-manifest"
    plugin_root = plugin_dir / plugin_name
    plugin_root.mkdir()
    # Create manifest missing the required 'version' field
    invalid_manifest = f"""
name = "{plugin_name}"
sygnals_api = ">=1.0.0"
entry_point = "invalid.plugin:InvalidPlugin"
"""
    (plugin_root / "plugin.toml").write_text(invalid_manifest, encoding='utf-8')

    plugin_loader.discover_and_load()
    # Assertions
    assert plugin_name not in plugin_loader.loaded_plugins, "Plugin with invalid manifest should not load."
    assert plugin_name not in plugin_loader.plugin_manifests, "Invalid manifest should not be stored."


def test_handle_entry_point_import_error(plugin_loader: PluginLoader, base_config: SygnalsConfig):
    """Test loading when the plugin's entry_point module/class cannot be imported."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-import-error"
    plugin_root = plugin_dir / plugin_name
    plugin_root.mkdir()
    # Correct manifest, but target .py file won't exist or module is wrong
    manifest_content = f"""
name = "{plugin_name}"
version = "0.1.0"
sygnals_api = ">=1.0.0"
entry_point = "non_existent_module.plugin:NonExistentPlugin"
"""
    (plugin_root / "plugin.toml").write_text(manifest_content, encoding='utf-8')
    # Update state file (needed by create_dummy_plugin logic, do manually here)
    current_state = _load_plugin_state(state_dir)
    current_state[plugin_name] = True
    _save_plugin_state(state_dir, current_state)

    plugin_loader.discover_and_load()
    # Assertions
    assert plugin_name not in plugin_loader.loaded_plugins, "Plugin with import error should not load."
    assert plugin_name in plugin_loader.plugin_manifests, "Manifest should be stored even if import fails later."
    assert plugin_loader.plugin_sources[plugin_name] == 'local', "Source should be recorded."


def test_handle_entry_point_not_subclass(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading when entry_point class doesn't inherit from SygnalsPluginBase."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-not-subclass"
    # Create the dummy plugin structure first
    _, entry_point_path_correct = create_dummy_plugin(plugin_dir, state_dir, plugin_name)
    # Modify the generated plugin.py to define a class that doesn't inherit
    package_name = plugin_name.replace("-", "_")
    class_name_wrong = "NotAPlugin" # The class we actually want to test
    plugin_py_path = plugin_dir / plugin_name / package_name / "plugin.py"
    # Define a class that does NOT inherit from SygnalsPluginBase
    plugin_py_content_wrong_class = f"""
import logging
import numpy as np # Added import
from pathlib import Path # Added import
import pandas as pd # Added import
# NOTE: Intentionally NOT importing SygnalsPluginBase
logger = logging.getLogger(__name__)

# Dummy handlers needed by helper
def dummy_callable(*args, **kwargs): pass
def dummy_reader(path, **kwargs): return {{'data': np.array([1])}}
def dummy_writer(data, path, **kwargs): pass

class {class_name_wrong}: # Class doesn't inherit!
    # Need name/version properties for the loader's check, even if not inheriting base
    @property
    def name(self): return "{plugin_name}"
    @property
    def version(self): return "0.1.0"
    pass
"""
    plugin_py_path.write_text(plugin_py_content_wrong_class, encoding='utf-8')

    # Adjust the manifest entry point to point to the wrong class
    manifest_path = plugin_dir / plugin_name / "plugin.toml"
    manifest_data = toml.load(manifest_path)
    manifest_data['entry_point'] = f"{package_name}.plugin:{class_name_wrong}" # Point to wrong class
    manifest_path.write_text(toml.dumps(manifest_data), encoding='utf-8')

    plugin_loader.discover_and_load()
    # Assertions
    assert plugin_name not in plugin_loader.loaded_plugins, "Plugin not inheriting base class should not load."
    assert plugin_name in plugin_loader.plugin_manifests, "Manifest should be stored."
    assert plugin_loader.plugin_sources[plugin_name] == 'local', "Source should be recorded."


def test_plugin_state_save_load(base_config: SygnalsConfig):
    """Test saving and loading plugin enabled/disabled state via helper functions."""
    state_dir = getattr(base_config, '_test_state_dir') # Use dedicated state dir
    state_file = state_dir / PLUGIN_STATE_FILENAME

    # Initial state: file doesn't exist, load should return empty dict
    if state_file.exists(): state_file.unlink() # Ensure clean start
    assert not state_file.exists(), "State file should not exist initially."
    loaded_state1 = _load_plugin_state(state_dir)
    assert loaded_state1 == {}, "Loading non-existent state file should return empty dict."

    # Save some state
    state_to_save = {"plugin-a": True, "plugin-b": False, "plugin-c": True}
    _save_plugin_state(state_dir, state_to_save)
    assert state_file.exists(), "State file should be created after saving."

    # Load the saved state
    loaded_state2 = _load_plugin_state(state_dir)
    assert loaded_state2 == state_to_save, "Loaded state does not match saved state."


def test_plugin_list_cli(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path, mocker):
    """Test the 'sygnals plugin list' command output."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    # Create one loaded, one disabled, one incompatible plugin
    create_dummy_plugin(plugin_dir, state_dir, "plugin-loaded", version="1.0", register_hooks={"register_filters": 'registry.add_filter("f1", dummy_callable)'})
    create_dummy_plugin(plugin_dir, state_dir, "plugin-disabled", version="1.1", is_enabled=False)
    create_dummy_plugin(plugin_dir, state_dir, "plugin-incomp", version="1.2", api_req=">=99.0") # Incompatible API

    # Explicitly call discover_and_load *after* creating plugins
    plugin_loader.discover_and_load()

    # Patch the loader instance and config used within the CLI context
    mocker.patch('sygnals.cli.base_cmd.plugin_loader', plugin_loader)
    mocker.patch('sygnals.cli.base_cmd.load_configuration', return_value=base_config)

    runner = CliRunner()
    result = runner.invoke(cli, ["plugin", "list"])

    # Print output for debugging if needed
    # print("CLI Output:\n", result.output)
    # if result.exception: print("Exception:\n", result.exception)

    # Assertions
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}.\nStderr:\n{result.stderr}"
    assert result.exception is None, "CLI command should not raise an exception."

    # Check for presence and status in the output (allow for formatting variations)
    assert "plugin-loaded" in result.output and "1.0" in result.output and "loaded" in result.output
    assert "plugin-disabled" in result.output and "1.1" in result.output and "disabled" in result.output
    assert "plugin-incomp" in result.output and "1.2" in result.output and "incomp" in result.output # Check for "incompatible" status abbreviation
    assert "local" in result.output # Check source type


def test_plugin_enable_disable_cli(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path, mocker):
    """Test the 'sygnals plugin enable' and 'disable' commands."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "plugin-toggle"
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, is_enabled=True) # Start enabled

    # Patch the loader instance and config used within the CLI context *before* running commands
    mocker.patch('sygnals.cli.base_cmd.plugin_loader', plugin_loader)
    mocker.patch('sygnals.cli.base_cmd.load_configuration', return_value=base_config)

    # We still need the loader to know about the plugin manifest *before* enable/disable is called
    # discover_and_load reads the state file, so call it first.
    plugin_loader.discover_and_load()

    runner = CliRunner(mix_stderr=False) # Keep stderr separate

    # --- Disable the plugin ---
    result_disable = runner.invoke(cli, ["plugin", "disable", plugin_name])
    assert result_disable.exit_code == 0, f"Disable failed: {result_disable.stderr}"
    assert result_disable.exception is None
    assert "disabled" in result_disable.output
    # Verify state file was updated
    state1 = _load_plugin_state(state_dir)
    assert state1.get(plugin_name) is False, "Plugin state should be disabled after disable command."

    # --- Enable the plugin ---
    result_enable = runner.invoke(cli, ["plugin", "enable", plugin_name])
    assert result_enable.exit_code == 0, f"Enable failed: {result_enable.stderr}"
    assert result_enable.exception is None
    assert "enabled" in result_enable.output
    # Verify state file was updated
    state2 = _load_plugin_state(state_dir)
    assert state2.get(plugin_name) is True, "Plugin state should be enabled after enable command."

    # --- Try disabling non-existent plugin ---
    result_disable_bad = runner.invoke(cli, ["plugin", "disable", "non-existent"])
    assert result_disable_bad.exit_code != 0, "Disabling non-existent plugin should fail."
    assert result_disable_bad.exception is not None, "Expected SystemExit for failed disable."
    assert isinstance(result_disable_bad.exception, SystemExit), "Exception should be SystemExit."
    # Check result.output (stdout) instead of result.stderr
    assert "Error" in result_disable_bad.output, "Error message not found in stdout"
    assert "Not found" in result_disable_bad.output, "Plugin not found message missing from stdout"


def test_plugin_scaffold_cli(tmp_path: Path, mocker):
    """Test the 'sygnals plugin scaffold' command."""
    # Mock config loading and plugin discovery to isolate the scaffold command
    mocker.patch('sygnals.cli.base_cmd.load_configuration', return_value=SygnalsConfig())
    mocker.patch('sygnals.plugins.loader.PluginLoader.discover_and_load', return_value=None)

    runner = CliRunner()
    plugin_name = "my-scaffolded-plugin"
    dest_dir = tmp_path / "scaffold_dest"
    # Scaffold command should create the destination directory if needed

    result = runner.invoke(cli, ["plugin", "scaffold", plugin_name, "--dest", str(dest_dir)])

    # Print output for debugging if needed
    # print("Scaffold CLI Output:\n", result.output)
    # if result.exception: print("Exception:\n", result.exception)

    # Assertions
    assert result.exit_code == 0, f"Scaffold command failed: {result.stderr}"
    assert result.exception is None
    assert "Plugin template" in result.output and "created successfully" in result.output

    # Check if expected files/dirs were created
    plugin_root = dest_dir / plugin_name
    package_name = "my_scaffolded_plugin"
    package_dir = plugin_root / package_name
    assert plugin_root.is_dir(), "Plugin root directory not created."
    assert package_dir.is_dir(), "Plugin package directory not created."
    assert (plugin_root / "plugin.toml").is_file(), "plugin.toml not created."
    assert (plugin_root / "pyproject.toml").is_file(), "pyproject.toml not created."
    assert (package_dir / "__init__.py").is_file(), "__init__.py not created."
    assert (package_dir / "plugin.py").is_file(), "plugin.py not created."

    # Check content of one file (e.g., plugin.toml for correct name)
    manifest_content = (plugin_root / "plugin.toml").read_text(encoding='utf-8')
    assert f'name = "{plugin_name}"' in manifest_content, "Plugin name not found in generated plugin.toml."
