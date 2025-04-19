# tests/test_plugin_system.py

"""
Tests for the Sygnals plugin system (loading, registration, management).
"""

import pytest
import sys
import traceback # Added for detailed exception printing in tests
from pathlib import Path
import toml # For creating dummy manifests
from typing import Optional # Import Optional for type hinting

# Import components to test
from sygnals.config.models import SygnalsConfig, PathsConfig
from sygnals.plugins.api import PluginRegistry, SygnalsPluginBase
from sygnals.plugins.loader import PluginLoader, _load_plugin_state, _save_plugin_state
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
    config.paths = PathsConfig(
        plugin_dir=tmp_path / "sygnals_test_plugins",
        cache_dir=tmp_path / ".sygnals_cache",
        output_dir=tmp_path / "sygnals_output",
        log_directory=tmp_path / "sygnals_logs"
    )
    # Ensure plugin directory exists for the tests
    config.paths.plugin_dir.mkdir(parents=True, exist_ok=True)
    return config

@pytest.fixture
def plugin_loader(base_config: SygnalsConfig, plugin_registry: PluginRegistry) -> PluginLoader:
    """Provides a PluginLoader instance initialized with temp config and fresh registry."""
    loader = PluginLoader(base_config, plugin_registry)
    # Also update the global instance that might be used by CLI commands if not patched
    sygnals_base_cmd.plugin_loader = loader
    return loader

@pytest.fixture
def teardown_sys_path():
    """Fixture to clean up sys.path modifications made during tests."""
    original_sys_path = list(sys.path)
    yield
    sys.path = original_sys_path


# --- Helper Function to Create Dummy Plugin ---
def create_dummy_plugin(
    plugin_dir: Path,
    name: str,
    version: str = "0.1.0",
    api_req: str = f">={core_version},<2.0.0",
    entry_point_content: str = "",
    dependencies: list = [],
    register_hook: Optional[str] = None, # e.g., "register_filters"
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
    # Use the provided 'name' argument for generating package/class names
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
    (plugin_root / "plugin.toml").write_text(manifest_content)

    # Python package structure
    package_dir = plugin_root / entry_module
    package_dir.mkdir()
    (package_dir / "__init__.py").touch()

    # plugin.py (entry point)
    hook_impl = ""
    if register_hook:
        hook_impl = f"""
    def {register_hook}(self, registry):
        # Simple registration for testing
        if "{register_hook}" == "register_filters":
             registry.add_filter("{name}_filter", lambda x: x)
        elif "{register_hook}" == "register_features":
             registry.add_feature("{name}_feature", lambda x: x)
        logger.info(f"Plugin {name} executing {register_hook}")
"""

    plugin_py_content = f"""
import logging
from sygnals.plugins.api import SygnalsPluginBase, PluginRegistry
logger = logging.getLogger(__name__)

class {class_name}(SygnalsPluginBase):
    @property
    def name(self) -> str: return "{name}"
    @property
    def version(self) -> str: return "{version}"

    def setup(self, config): logger.info(f"Plugin {name} setup called.")
    def teardown(self): logger.info(f"Plugin {name} teardown called.")

    {hook_impl}

{entry_point_content} # Add extra content if needed
"""
    (package_dir / "plugin.py").write_text(plugin_py_content)

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
         (plugin_root / "pyproject.toml").write_text(pyproject_content)

    # Update state file if needed (emulate enable/disable)
    # State file is in parent of plugin_dir
    state_dir = plugin_dir.parent
    current_state = _load_plugin_state(state_dir)
    current_state[name] = is_enabled
    _save_plugin_state(state_dir, current_state)

    return plugin_root, entry_point_path


# --- Test Cases ---

def test_load_valid_local_plugin(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading a correctly structured and compatible local plugin."""
    plugin_dir = base_config.paths.plugin_dir
    plugin_name = "test-valid"
    create_dummy_plugin(plugin_dir, plugin_name, register_hook="register_filters")

    plugin_loader.discover_and_load()

    assert plugin_name in plugin_loader.loaded_plugins
    assert plugin_name in plugin_loader.plugin_manifests
    assert plugin_loader.plugin_sources[plugin_name] == 'local'
    assert plugin_loader.registry.get_filter(f"{plugin_name}_filter") is not None
    assert plugin_loader.registry.loaded_plugin_names == [plugin_name]


def test_skip_incompatible_plugin(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test that an incompatible plugin (based on sygnals_api) is skipped."""
    plugin_dir = base_config.paths.plugin_dir
    plugin_name = "test-incompatible"
    # Require a future API version
    create_dummy_plugin(plugin_dir, plugin_name, api_req=">=99.0.0")

    plugin_loader.discover_and_load()

    assert plugin_name not in plugin_loader.loaded_plugins
    # Manifest should be stored even if incompatible
    assert plugin_name in plugin_loader.plugin_manifests
    assert plugin_loader.plugin_sources[plugin_name] == 'local'
    assert not plugin_loader.registry.loaded_plugin_names


def test_skip_disabled_plugin(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test that a disabled plugin is skipped during loading."""
    plugin_dir = base_config.paths.plugin_dir
    plugin_name = "test-disabled"
    create_dummy_plugin(plugin_dir, plugin_name, is_enabled=False)

    plugin_loader.discover_and_load()

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
    (plugin_root / "plugin.toml").write_text(invalid_manifest)

    plugin_loader.discover_and_load()
    assert plugin_name not in plugin_loader.loaded_plugins
    # Invalid manifest means it shouldn't even be stored
    assert plugin_name not in plugin_loader.plugin_manifests


def test_handle_entry_point_import_error(plugin_loader: PluginLoader, base_config: SygnalsConfig):
    """Test loading when the entry_point module/class cannot be imported."""
    plugin_dir = base_config.paths.plugin_dir
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
    (plugin_root / "plugin.toml").write_text(manifest_content)

    plugin_loader.discover_and_load()
    assert plugin_name not in plugin_loader.loaded_plugins
    # Manifest should be stored even if import fails later
    assert plugin_name in plugin_loader.plugin_manifests
    assert plugin_loader.plugin_sources[plugin_name] == 'local'


def test_handle_entry_point_not_subclass(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path):
    """Test loading when entry_point class doesn't inherit from SygnalsPluginBase."""
    plugin_dir = base_config.paths.plugin_dir
    plugin_name = "test-not-subclass"
    # Create a valid class, but doesn't inherit
    plugin_content = """
class NotAPlugin:
    pass
"""
    # Create the dummy plugin structure first
    _, entry_point_path = create_dummy_plugin(plugin_dir, plugin_name)
    # Modify the generated plugin.py
    # FIX: Use plugin_name here instead of undefined 'name'
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
    pass

"""
    plugin_py_path.write_text(plugin_py_content_wrong_class)


    # Adjust the manifest entry point to point to the wrong class
    manifest_path = plugin_dir / plugin_name / "plugin.toml"
    manifest_data = toml.load(manifest_path)
    manifest_data['entry_point'] = f"{package_name}.plugin:{class_name_wrong}" # Point to wrong class
    manifest_path.write_text(toml.dumps(manifest_data))


    plugin_loader.discover_and_load()
    assert plugin_name not in plugin_loader.loaded_plugins
    # Manifest should be stored even if class validation fails
    assert plugin_name in plugin_loader.plugin_manifests
    assert plugin_loader.plugin_sources[plugin_name] == 'local'


def test_plugin_state_save_load(base_config: SygnalsConfig):
    """Test saving and loading plugin enabled/disabled state."""
    state_dir = base_config.paths.plugin_dir.parent
    # Use the constant defined in loader.py for consistency
    state_file = state_dir / PluginLoader.PLUGIN_STATE_FILENAME

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
    # Create one loaded, one disabled, one incompatible using the fixture's loader
    create_dummy_plugin(plugin_dir, "plugin-loaded", version="1.0", register_hook="register_filters")
    create_dummy_plugin(plugin_dir, "plugin-disabled", version="1.1", is_enabled=False)
    create_dummy_plugin(plugin_dir, "plugin-incomp", version="1.2", api_req=">=99.0")

    # Ensure the fixture's loader has processed these plugins
    plugin_loader.discover_and_load()

    # Patch the loader instance used within the CLI context
    # Target the global variable in base_cmd that ConfigGroup uses
    mocker.patch('sygnals.cli.base_cmd.plugin_loader', plugin_loader)
    # Also patch load_configuration to return the test config
    mocker.patch('sygnals.cli.base_cmd.load_configuration', return_value=base_config)


    runner = CliRunner()
    result = runner.invoke(cli, ["plugin", "list"])

    print("CLI Output:\n", result.output) # Print output for debugging
    if result.exception:
        print("Exception:\n", result.exception)
        traceback.print_tb(result.exc_info[2])


    assert result.exit_code == 0
    assert "plugin-loaded" in result.output
    assert "1.0" in result.output
    # FIX: Check for plain text status instead of Rich formatting
    assert "loaded" in result.output
    assert "plugin-disabled" in result.output
    assert "1.1" in result.output
    assert "disabled" in result.output
    assert "plugin-incomp" in result.output
    assert "1.2" in result.output
    assert "error/incompatible" in result.output
    assert "local" in result.output # Check source


def test_plugin_enable_disable_cli(plugin_loader: PluginLoader, base_config: SygnalsConfig, teardown_sys_path, mocker):
    """Test the 'sygnals plugin enable/disable' commands."""
    plugin_dir = base_config.paths.plugin_dir
    state_dir = plugin_dir.parent
    plugin_name = "plugin-toggle"
    create_dummy_plugin(plugin_dir, plugin_name, is_enabled=True) # Start enabled

    # Ensure the fixture's loader has processed this plugin's manifest
    plugin_loader.discover_and_load()

    # Patch the loader instance used within the CLI context
    mocker.patch('sygnals.cli.base_cmd.plugin_loader', plugin_loader)
    mocker.patch('sygnals.cli.base_cmd.load_configuration', return_value=base_config)


    runner = CliRunner()

    # Disable the plugin
    result_disable = runner.invoke(cli, ["plugin", "disable", plugin_name])
    print("Disable CLI Output:\n", result_disable.output)
    if result_disable.exception: print("Exception:\n", result_disable.exception)
    assert result_disable.exit_code == 0
    assert "disabled" in result_disable.output
    # Check state file
    state1 = _load_plugin_state(state_dir)
    assert state1.get(plugin_name) is False

    # Enable the plugin
    result_enable = runner.invoke(cli, ["plugin", "enable", plugin_name])
    print("Enable CLI Output:\n", result_enable.output)
    if result_enable.exception: print("Exception:\n", result_enable.exception)
    assert result_enable.exit_code == 0
    assert "enabled" in result_enable.output
    # Check state file
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
    manifest_content = (plugin_root / "plugin.toml").read_text()
    assert f'name = "{plugin_name}"' in manifest_content
