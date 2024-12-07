import pytest
from sygnals.core.plugin_manager import discover_plugins

def test_discover_plugins():
    plugins = discover_plugins(plugin_dir='sygnals/plugins')
    # Expect at least the example plugins defined
    assert "custom_filter" in plugins or "square_signal" in plugins
