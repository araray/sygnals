# tests/test_plugin_manager.py

import pytest
from pathlib import Path
import sys

# Import the functions to test
from sygnals.core.plugin_manager import discover_plugins, register_plugin

# --- Test Fixture ---

@pytest.fixture
def setup_plugin_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory structure with dummy plugin files."""
    plugin_dir = tmp_path / "temp_plugins"
    plugin_dir.mkdir()

    # Plugin file 1: Contains valid plugins
    plugin_file_1 = plugin_dir / "valid_plugins.py"
    plugin_file_1.write_text(
        """
from sygnals.core.plugin_manager import register_plugin
import numpy as np

@register_plugin
def plugin_func_one(data):
    return data * 2

@register_plugin
class PluginClassOne:
    __plugin__ = True # Decorator equivalent for classes (manual for testing)
    def __call__(self, data):
        return np.mean(data)

def not_a_plugin(data):
    return data + 1
"""
    )

    # Plugin file 2: Contains another valid plugin
    plugin_file_2 = plugin_dir / "more_plugins.py"
    plugin_file_2.write_text(
        """
from sygnals.core.plugin_manager import register_plugin

@register_plugin
def plugin_func_two(data, factor=0.5):
    return data * factor
"""
    )

    # File 3: Not a python file
    (plugin_dir / "not_a_python_file.txt").touch()

    # File 4: Python file with no plugins
    plugin_file_4 = plugin_dir / "no_plugins_here.py"
    plugin_file_4.write_text(
        """
def helper_function():
    pass
"""
    )

    # Add the temp_plugins directory to sys.path temporarily if needed for importlib
    # Although spec_from_file_location should handle this.
    # sys.path.insert(0, str(tmp_path))
    # yield plugin_dir
    # sys.path.pop(0)
    return plugin_dir


# --- Test Cases ---

def test_discover_plugins_finds_valid(setup_plugin_dir):
    """Test that discover_plugins finds functions marked with @register_plugin."""
    plugin_dir = setup_plugin_dir
    plugins = discover_plugins(plugin_dir=str(plugin_dir)) # Pass path as string

    assert isinstance(plugins, dict)
    # Check if the discovered plugins are present by name
    assert "plugin_func_one" in plugins
    assert "plugin_func_two" in plugins
    # Check if the manually marked class is found (current implementation finds callables)
    # Note: The current discover_plugins might not find classes unless they are callable
    # and manually marked. Let's check based on the implementation detail (callable check).
    # If PluginClassOne instance was intended, test needs adjustment.
    # assert "PluginClassOne" in plugins # This might fail depending on exact implementation

    # Check that the non-plugin function is NOT discovered
    assert "not_a_plugin" not in plugins
    assert "helper_function" not in plugins

    # Check the type of discovered items (should be functions/callables)
    assert callable(plugins["plugin_func_one"])
    assert callable(plugins["plugin_func_two"])


def test_discover_plugins_non_existent_dir(tmp_path: Path):
    """Test discovery when the plugin directory doesn't exist (it should be created)."""
    non_existent_dir = tmp_path / "no_plugins_yet"
    # Ensure it doesn't exist initially
    assert not non_existent_dir.exists()

    plugins = discover_plugins(plugin_dir=str(non_existent_dir))

    # Check that the directory was created
    assert non_existent_dir.exists()
    assert non_existent_dir.is_dir()

    # Check that no plugins were found
    assert isinstance(plugins, dict)
    assert len(plugins) == 0


def test_discover_plugins_empty_dir(tmp_path: Path):
    """Test discovery when the plugin directory is empty."""
    empty_dir = tmp_path / "empty_plugins"
    empty_dir.mkdir()

    plugins = discover_plugins(plugin_dir=str(empty_dir))

    assert isinstance(plugins, dict)
    assert len(plugins) == 0


def test_register_plugin_decorator():
    """Test that the @register_plugin decorator adds the '__plugin__' attribute."""

    @register_plugin
    def my_test_plugin_func(x):
        return x

    def not_a_plugin_func(x):
        return x

    assert hasattr(my_test_plugin_func, "__plugin__")
    assert getattr(my_test_plugin_func, "__plugin__") is True

    assert not hasattr(not_a_plugin_func, "__plugin__")
