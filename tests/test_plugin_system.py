# tests/test_plugin_system.py

"""
Tests for the Sygnals plugin system (loading, registration, management).
"""

import pytest
import sys
import traceback # Added for detailed exception printing in tests
from pathlib import Path
import toml # For creating dummy manifests
import logging # Import logging for caplog test
from typing import Optional, Dict, Any, List, Tuple, Callable # Import necessary types

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
    sygnals_base_cmd.plugin_registry = PluginRegistry()
    return sygnals_base_cmd.plugin_registry

@pytest.fixture
def base_config(tmp_path: Path) -> SygnalsConfig:
    """Provides a base SygnalsConfig pointing plugin dir to a temp path."""
    config = SygnalsConfig()
    # Override paths to use temp directory
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
    setattr(config, '_test_state_dir', state_dir_path) # Use setattr to add temporary attribute
    return config

@pytest.fixture
def plugin_loader(base_config: SygnalsConfig, plugin_registry: PluginRegistry) -> PluginLoader:
    """
    Provides a PluginLoader instance initialized with temp config and fresh registry.
    NOTE: This fixture does NOT call discover_and_load automatically. Tests should call it.
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
    state_dir: Path, # Added state_dir argument
    name: str,
    version: str = "0.1.0",
    api_req: str = f">={core_version},<2.0.0",
    entry_point_content: str = "",
    dependencies: list = [],
    register_hooks: Optional[Dict[str, str]] = None, # e.g., {"register_filters": "registry.add_filter(...)"}
    is_enabled: bool = True,
    create_pyproject: bool = False
):
    """Creates a dummy plugin directory structure and files."""
    plugin_root = plugin_dir / name
    # Ensure clean state if directory exists from previous failed run
    if plugin_root.exists():
        import shutil
        shutil.rmtree(plugin_root)
    plugin_root.mkdir()

    package_name = name.replace("-", "_")
    class_name = "".join(part.capitalize() for part in name.split('-')) + "Plugin"
    entry_module = package_name
    entry_point_path = f"{entry_module}.plugin:{class_name}"

    # plugin.toml
    manifest_content = f"""
name = "{name}"
version = "{version}"
sygnals_api = "{api_req}"
description = "A dummy test plugin: {name}"
entry_point = "{entry_point_path}"
dependencies = {dependencies}
"""
    (plugin_root / "plugin.toml").write_text(manifest_content, encoding='utf-8')

    # Python package structure
    package_dir = plugin_root / entry_module
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(f"# Init for {package_name}", encoding='utf-8')

    # plugin.py (entry point)
    hook_implementations = ""
    if register_hooks:
        for hook_name, hook_code in register_hooks.items():
            # Indent the provided hook code correctly
            indented_code = "\n".join("        " + line for line in hook_code.strip().split("\n"))
            hook_implementations += f"""
    def {hook_name}(self, registry):
        logger.info(f"Plugin {name} executing {hook_name}")
{indented_code}
"""

    plugin_py_content = f"""
# Dummy plugin file for testing: {name}
import logging
from sygnals.plugins.api import SygnalsPluginBase, PluginRegistry
# Add common imports needed by hook code snippets
import numpy as np
from pathlib import Path # Add Path import
import pandas as pd # Add pandas import
def dummy_callable(*args, **kwargs): pass # Simple callable for registration
def dummy_reader(path, **kwargs): return {{'data': np.array([1])}} # Simple reader
def dummy_writer(data, path, **kwargs): pass # Simple writer

logger = logging.getLogger(__name__)

class {class_name}(SygnalsPluginBase):
    @property
    def name(self) -> str: return "{name}"
    @property
    def version(self) -> str: return "{version}"

    def setup(self, config): logger.info(f"Plugin {{self.name}} setup called.") # Escaped braces
    def teardown(self): logger.info(f"Plugin {{self.name}} teardown called.") # Escaped braces

{hook_implementations}

{entry_point_content} # Add extra content if needed
"""
    (package_dir / "plugin.py").write_text(plugin_py_content, encoding='utf-8')

    # pyproject.toml (optional, for entry point testing if needed)
    if create_pyproject:
         pyproject_content = f"""
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "{version}"

[project.entry-points."sygnals.plugins"]
{name} = "{entry_point_path}"
"""
         (plugin_root / "pyproject.toml").write_text(pyproject_content, encoding='utf-8')

    # Update state file if needed (emulate enable/disable)
    # Use the provided state_dir
    current_state = _load_plugin_state(state_dir)
    current_state[name] = is_enabled
    _save_plugin_state(state_dir, current_state)

    return plugin_root, entry_point_path


# --- Test Cases ---

def test_load_valid_local_plugin_registration(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading a plugin and verify its registration hook is called."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir') # Get state dir from config fixture
    plugin_name = "test-register-filter"
    # Define the registration code snippet
    hook_code = 'registry.add_filter("my_test_filter", dummy_callable)'
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, register_hooks={"register_filters": hook_code})

    # Explicitly call discover_and_load *after* creating the plugin
    plugin_loader.discover_and_load()

    assert plugin_name in plugin_loader.loaded_plugins
    assert plugin_loader.registry.get_filter("my_test_filter") is not None
    assert plugin_loader.registry.list_filters() == ["my_test_filter"]
    assert plugin_loader.registry.loaded_plugin_names == [plugin_name]


def test_load_plugin_registering_multiple_types(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading a plugin that registers multiple extension types."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-multi-register"
    hooks = {
        "register_feature_extractors": 'registry.add_feature("multi_feature", lambda x: x+1)',
        "register_transforms": 'registry.add_transform("multi_transform", lambda x: x*2)'
    }
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, register_hooks=hooks)

    # Explicitly call discover_and_load
    plugin_loader.discover_and_load()

    assert plugin_name in plugin_loader.loaded_plugins
    # Assert based on what was *actually* registered by the hooks
    assert plugin_loader.registry.get_feature("multi_feature") is not None
    assert plugin_loader.registry.get_transform("multi_transform") is not None
    assert plugin_loader.registry.list_features() == ["multi_feature"]
    assert plugin_loader.registry.list_transforms() == ["multi_transform"]
    assert plugin_loader.registry.list_filters() == [] # No filters registered


def test_load_multiple_plugins_registration(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading multiple plugins and check combined registry content."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin1_name = "plugin-one"
    plugin2_name = "plugin-two"

    create_dummy_plugin(plugin_dir, state_dir, plugin1_name, register_hooks={"register_filters": 'registry.add_filter("filter_one", dummy_callable)'})
    create_dummy_plugin(plugin_dir, state_dir, plugin2_name, register_hooks={"register_feature_extractors": 'registry.add_feature("feature_two", dummy_callable)'})

    # Explicitly call discover_and_load
    plugin_loader.discover_and_load()

    assert plugin1_name in plugin_loader.loaded_plugins
    assert plugin2_name in plugin_loader.loaded_plugins
    assert plugin_loader.registry.get_filter("filter_one") is not None
    assert plugin_loader.registry.get_feature("feature_two") is not None
    assert sorted(plugin_loader.registry.loaded_plugin_names) == sorted([plugin1_name, plugin2_name])


def test_registration_overwriting(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path, caplog):
    """Test that registering the same name logs a warning and overwrites."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin1_name = "plugin-overwriter-1"
    plugin2_name = "plugin-overwriter-2"
    feature_name = "shared_feature"

    # Plugin 1 registers the feature
    create_dummy_plugin(plugin_dir, state_dir, plugin1_name, register_hooks={"register_feature_extractors": f'registry.add_feature("{feature_name}", lambda: 1)'})
    # Plugin 2 registers the same feature name
    create_dummy_plugin(plugin_dir, state_dir, plugin2_name, register_hooks={"register_feature_extractors": f'registry.add_feature("{feature_name}", lambda: 2)'})

    # Capture log messages
    with caplog.at_level(logging.WARNING): # Import logging at top
        # Explicitly call discover_and_load
        plugin_loader.discover_and_load()

    assert plugin1_name in plugin_loader.loaded_plugins
    assert plugin2_name in plugin_loader.loaded_plugins

    # Check that the warning was logged
    assert f"Feature '{feature_name}' is already registered. Overwriting." in caplog.text

    # Check that the feature callable corresponds to the *last* loaded plugin
    registered_feature = plugin_loader.registry.get_feature(feature_name)
    assert registered_feature is not None
    assert registered_feature() == 2 # Should be the function from plugin 2


# --- Tests for Data Handlers ---

def test_load_plugin_registering_reader_writer(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading a plugin that registers data readers and writers."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-io-plugin"
    hooks = {
        "register_data_readers": 'registry.add_reader(".custom", dummy_reader)',
        "register_data_writers": 'registry.add_writer(".custom", dummy_writer)',
    }
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, register_hooks=hooks)

    # Explicitly call discover_and_load
    plugin_loader.discover_and_load()

    assert plugin_name in plugin_loader.loaded_plugins
    # Check registry for the reader/writer
    reader = plugin_loader.registry.get_reader(".custom")
    writer = plugin_loader.registry.get_writer(".custom")
    assert reader is not None and callable(reader)
    assert writer is not None and callable(writer)
    assert plugin_loader.registry.list_readers() == [".custom"]
    assert plugin_loader.registry.list_writers() == [".custom"]

    # Verify the dummy reader works via registry
    assert reader(path="dummy") == {'data': np.array([1])}


def test_data_handler_registration_overwriting(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path, caplog):
    """Test overwriting data handlers and logging warnings."""
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

    with caplog.at_level(logging.WARNING):
        plugin_loader.discover_and_load()

    # Check warnings
    assert f"Reader for extension '{extension}' is already registered. Overwriting." in caplog.text
    assert f"Writer for extension '{extension}' is already registered. Overwriting." in caplog.text

    # Check that the handler from the second plugin is active
    reader = plugin_loader.registry.get_reader(extension)
    assert reader is not None
    assert reader(path="dummy") == 2 # Should be the function from plugin 2


# --- Existing Tests (Keep and Ensure They Still Pass) ---

def test_skip_incompatible_plugin(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test that an incompatible plugin (based on sygnals_api) is skipped."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-incompatible"
    # Require a future API version
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, api_req=">=99.0.0")

    # Explicitly call discover_and_load
    plugin_loader.discover_and_load()

    assert plugin_name not in plugin_loader.loaded_plugins
    # Manifest should be stored even if incompatible
    assert plugin_name in plugin_loader.plugin_manifests
    assert plugin_loader.plugin_sources[plugin_name] == 'local'
    assert not plugin_loader.registry.loaded_plugin_names


def test_skip_disabled_plugin(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test that a disabled plugin is skipped during loading."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-disabled"
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, is_enabled=False)

    # Explicitly call discover_and_load *after* creating the plugin and setting state
    plugin_loader.discover_and_load()

    # Now the assertion should pass because the loader reads the state file
    # created by create_dummy_plugin before trying to load
    assert plugin_name not in plugin_loader.loaded_plugins
    assert plugin_name in plugin_loader.plugin_manifests # Found but not loaded
    assert plugin_loader.plugin_sources[plugin_name] == 'local'
    assert not plugin_loader.registry.loaded_plugin_names


def test_handle_missing_manifest(plugin_loader: PluginLoader, base_config: SygnalsConfig):
    """Test discovery when a plugin directory lacks a manifest."""
    plugin_dir = base_config.paths.plugin_dir
    (plugin_dir / "no-manifest-plugin").mkdir()
    (plugin_dir / "no-manifest-plugin" / "plugin.py").touch()

    plugin_loader.discover_and_load()
    assert not plugin_loader.loaded_plugins # No plugins should be loaded


def test_handle_invalid_manifest(plugin_loader: PluginLoader, base_config: SygnalsConfig):
    """Test loading when a manifest file is invalid (e.g., missing fields)."""
    plugin_dir = base_config.paths.plugin_dir
    plugin_name = "test-invalid-manifest"
    plugin_root = plugin_dir / plugin_name
    plugin_root.mkdir()
    # Create manifest missing 'version'
    invalid_manifest = f"""
name = "{plugin_name}"
sygnals_api = ">=1.0.0"
entry_point = "invalid.plugin:InvalidPlugin"
"""
    (plugin_root / "plugin.toml").write_text(invalid_manifest, encoding='utf-8')

    plugin_loader.discover_and_load()
    assert plugin_name not in plugin_loader.loaded_plugins
    # Invalid manifest means it shouldn't even be stored
    assert plugin_name not in plugin_loader.plugin_manifests


def test_handle_entry_point_import_error(plugin_loader: PluginLoader, base_config: SygnalsConfig):
    """Test loading when the entry_point module/class cannot be imported."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-import-error"
    plugin_root = plugin_dir / plugin_name
    plugin_root.mkdir()
    # Correct manifest, but target .py file won't exist
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
    assert plugin_name not in plugin_loader.loaded_plugins
    # Manifest should be stored even if import fails later
    assert plugin_name in plugin_loader.plugin_manifests
    assert plugin_loader.plugin_sources[plugin_name] == 'local'


def test_handle_entry_point_not_subclass(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading when entry_point class doesn't inherit from SygnalsPluginBase."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "test-not-subclass"
    # Create the dummy plugin structure first
    _, entry_point_path = create_dummy_plugin(plugin_dir, state_dir, plugin_name)
    # Modify the generated plugin.py
    package_name = plugin_name.replace("-", "_")
    class_name_correct = "".join(part.capitalize() for part in plugin_name.split('-')) + "Plugin"
    class_name_wrong = "NotAPlugin" # The class we actually want to test
    plugin_py_path = plugin_dir / plugin_name / package_name / "plugin.py"
    plugin_py_content_wrong_class = f"""
import logging
# NOTE: Intentionally NOT importing SygnalsPluginBase
logger = logging.getLogger(__name__)

class {class_name_wrong}: # Class doesn't inherit!
    pass

# Keep the original class name definition too, just in case import logic needs it initially
# but the manifest points to the wrong one.
class {class_name_correct}:
    @property
    def name(self): return "{plugin_name}" # Need properties for check later
    @property
    def version(self): return "0.1.0"

"""
    plugin_py_path.write_text(plugin_py_content_wrong_class, encoding='utf-8')

    # Adjust the manifest entry point to point to the wrong class
    manifest_path = plugin_dir / plugin_name / "plugin.toml"
    manifest_data = toml.load(manifest_path)
    manifest_data['entry_point'] = f"{package_name}.plugin:{class_name_wrong}" # Point to wrong class
    manifest_path.write_text(toml.dumps(manifest_data), encoding='utf-8')

    plugin_loader.discover_and_load()
    assert plugin_name not in plugin_loader.loaded_plugins
    # Manifest should be stored even if class validation fails
    assert plugin_name in plugin_loader.plugin_manifests
    assert plugin_loader.plugin_sources[plugin_name] == 'local'


def test_plugin_state_save_load(base_config: SygnalsConfig):
    """Test saving and loading plugin enabled/disabled state."""
    state_dir = getattr(base_config, '_test_state_dir') # Use dedicated state dir
    state_file = state_dir / PLUGIN_STATE_FILENAME

    # Initial state: file doesn't exist, load should return empty dict
    assert not state_file.exists()
    loaded_state1 = _load_plugin_state(state_dir)
    assert loaded_state1 == {}

    # Save some state
    state_to_save = {"plugin-a": True, "plugin-b": False, "plugin-c": True}
    _save_plugin_state(state_dir, state_to_save)
    assert state_file.exists()

    # Load the saved state
    loaded_state2 = _load_plugin_state(state_dir)
    assert loaded_state2 == state_to_save


def test_plugin_list_cli(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path, mocker):
    """Test the 'sygnals plugin list' command."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    # Create one loaded, one disabled, one incompatible
    create_dummy_plugin(plugin_dir, state_dir, "plugin-loaded", version="1.0", register_hooks={"register_filters": 'registry.add_filter("f1", dummy_callable)'})
    create_dummy_plugin(plugin_dir, state_dir, "plugin-disabled", version="1.1", is_enabled=False)
    create_dummy_plugin(plugin_dir, state_dir, "plugin-incomp", version="1.2", api_req=">=99.0")

    # Explicitly call discover_and_load *after* creating plugins
    # This ensures the loader sees the correct state for 'plugin-disabled'
    plugin_loader.discover_and_load()

    # Patch the loader instance used within the CLI context
    mocker.patch('sygnals.cli.base_cmd.plugin_loader', plugin_loader)
    mocker.patch('sygnals.cli.base_cmd.load_configuration', return_value=base_config)

    runner = CliRunner()
    result = runner.invoke(cli, ["plugin", "list"])

    print("CLI Output:\n", result.output) # Print output for debugging
    if result.exception:
        print("Exception:\n", result.exception)
        traceback.print_tb(result.exc_info[2])

    assert result.exit_code == 0
    # Check for presence and status (allow for Rich formatting variations)
    assert "plugin-loaded" in result.output and "1.0" in result.output and "loaded" in result.output
    # Now 'plugin-disabled' should correctly show as disabled
    assert "plugin-disabled" in result.output and "1.1" in result.output and "disabled" in result.output
    # Check for incompatible status - check for substring "incomp" to handle truncation
    assert "plugin-incomp" in result.output and "1.2" in result.output and "incomp" in result.output
    assert "local" in result.output # Check source


def test_plugin_enable_disable_cli(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path, mocker):
    """Test the 'sygnals plugin enable/disable' commands."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = getattr(base_config, '_test_state_dir')
    plugin_name = "plugin-toggle"
    create_dummy_plugin(plugin_dir, state_dir, plugin_name, is_enabled=True) # Start enabled

    # Patch the loader instance used within the CLI context *before* running commands
    # The commands themselves will use this patched loader to modify state
    mocker.patch('sygnals.cli.base_cmd.plugin_loader', plugin_loader)
    mocker.patch('sygnals.cli.base_cmd.load_configuration', return_value=base_config)

    # We still need the loader to know about the plugin manifest *before* enable/disable is called
    plugin_loader.discover_and_load() # Load manifests initially

    runner = CliRunner()

    # Disable the plugin
    result_disable = runner.invoke(cli, ["plugin", "disable", plugin_name])
    print("Disable CLI Output:\n", result_disable.output)
    if result_disable.exception: print("Exception:\n", result_disable.exception)
    assert result_disable.exit_code == 0
    assert "disabled" in result_disable.output
    # Check state file was updated
    state1 = _load_plugin_state(state_dir)
    assert state1.get(plugin_name) is False

    # Enable the plugin
    result_enable = runner.invoke(cli, ["plugin", "enable", plugin_name])
    print("Enable CLI Output:\n", result_enable.output)
    if result_enable.exception: print("Exception:\n", result_enable.exception)
    assert result_enable.exit_code == 0
    assert "enabled" in result_enable.output
    # Check state file was updated
    state2 = _load_plugin_state(state_dir)
    assert state2.get(plugin_name) is True

    # Try disabling non-existent plugin
    result_disable_bad = runner.invoke(cli, ["plugin", "disable", "non-existent"])
    print("Disable Bad CLI Output:\n", result_disable_bad.output)
    if result_disable_bad.exception: print("Exception:\n", result_disable_bad.exception)
    assert result_disable_bad.exit_code == 1 # Should exit with error
    assert "Error" in result_disable_bad.output
    assert "Not found" in result_disable_bad.output # Check specific message


def test_plugin_scaffold_cli(tmp_path: Path, mocker):
    """Test the 'sygnals plugin scaffold' command."""
     # Mock load_configuration to prevent issues finding default config files
    mocker.patch('sygnals.cli.base_cmd.load_configuration', return_value=SygnalsConfig())
    # Mock discover_and_load to prevent scanning during scaffold command
    mocker.patch('sygnals.plugins.loader.PluginLoader.discover_and_load', return_value=None)

    runner = CliRunner()
    plugin_name = "my-scaffolded-plugin"
    dest_dir = tmp_path / "scaffold_dest"
    # No need to mkdir, scaffold command should handle it if needed by create_plugin_scaffold

    result = runner.invoke(cli, ["plugin", "scaffold", plugin_name, "--dest", str(dest_dir)])

    print("Scaffold CLI Output:\n", result.output) # Print output for debugging
    if result.exception:
        print("Exception:\n", result.exception)
        import traceback
        traceback.print_tb(result.exc_info[2])

    assert result.exit_code == 0
    assert "Plugin template" in result.output
    assert "created successfully" in result.output

    # Check if expected files/dirs were created
    plugin_root = dest_dir / plugin_name
    package_name = "my_scaffolded_plugin"
    package_dir = plugin_root / package_name
    assert plugin_root.is_dir()
    assert package_dir.is_dir()
    assert (plugin_root / "plugin.toml").is_file()
    assert (plugin_root / "pyproject.toml").is_file()
    assert (package_dir / "__init__.py").is_file()
    assert (package_dir / "plugin.py").is_file()

    # Check content of one file (e.g., plugin.toml)
    manifest_content = (plugin_root / "plugin.toml").read_text(encoding='utf-8')
    assert f'name = "{plugin_name}"' in manifest_content
