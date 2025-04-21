# sygnals/plugins/scaffold.py

"""
Provides functionality to generate a basic plugin template structure.
"""

import logging
import re
from pathlib import Path
from packaging.version import Version # Added import

from sygnals.version import __version__ as core_version

logger = logging.getLogger(__name__)

# --- Template Content ---

PLUGIN_TOML_TEMPLATE = """\
# Plugin Manifest for '{plugin_name}'
# See Sygnals documentation for details on manifest fields.

# Unique plugin identifier (lowercase, hyphens allowed)
name = "{plugin_name}"

# Plugin version (Semantic Versioning - [https://semver.org/](https://semver.org/))
version = "0.1.0"

# Sygnals core API compatibility range (PEP 440 specifier)
# Adjust based on the Sygnals versions your plugin supports.
# FIX: Use the pre-calculated next major version key
sygnals_api = ">={core_version_major_minor}.0,<{core_version_next_major}.0.0" # E.g., ">=1.0.0,<2.0.0"

# Human-readable summary of the plugin's purpose
description = "A brief description of what {plugin_name} does."

# Path to the main plugin class implementing SygnalsPluginBase
# Format: <python_module_path>:<ClassName>
entry_point = "{package_name}.plugin:{class_name}"

# Optional: List of additional Python package dependencies required by this plugin
# These are informational for the user; actual installation dependencies
# should be managed in pyproject.toml or setup.py.
# dependencies = [
#     "numpy>=1.20.0",
#     "some-other-lib==1.2.3",
# ]
"""

PYPROJECT_TOML_TEMPLATE = """\
# pyproject.toml for the '{plugin_name}' Sygnals plugin

[build-system]
requires = ["setuptools>=61.0"] # Minimum setuptools version
build-backend = "setuptools.build_meta"

[project]
name = "{plugin_name}"
version = "0.1.0" # Should match plugin.toml
description = "A brief description of what {plugin_name} does." # Should match plugin.toml
readme = "README.md" # Optional: Add a README for your plugin
authors = [
  # {{ name="Your Name", email="your.email@example.com" }}, # Add your details
]
license = {{ text="Apache-2.0" }} # Or choose another license (e.g., MIT)
requires-python = ">=3.8" # Minimum Python version for Sygnals core
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License", # Match license choice
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
]
keywords = ["sygnals", "plugin", "signal processing", "audio"] # Add relevant keywords

# --- Plugin Dependencies ---
dependencies = [
    # Add any libraries your plugin needs, e.g.:
    # "numpy>=1.20",
    # "scipy",
    # Ensure Sygnals itself is NOT listed here, as it's the host application.
    # Sygnals core API compatibility is checked via 'sygnals_api' in plugin.toml.
]

[project.urls]
# Homepage = "[https://github.com/your_username/](https://github.com/your_username/){plugin_name}" # Optional: Link to repo

# --- Entry Point for Sygnals Plugin Discovery ---
# This tells Sygnals how to find your plugin if installed via pip.
[project.entry-points."sygnals.plugins"]
{plugin_name} = "{package_name}.plugin:{class_name}"

# --- Optional Tool Configurations ---
# Add configurations for tools like black, mypy, pytest if desired.
# [tool.black]
# line-length = 88
"""

# FIX: Corrected package name reference in __init__ template
PLUGIN_INIT_TEMPLATE = """\
# {package_name}/__init__.py

# Import the main plugin class to make it accessible
from .{package_name}.plugin import {class_name}

# Optionally define __all__ if needed
__all__ = ["{class_name}"]
"""


PLUGIN_PY_TEMPLATE = """\
# {package_name}/plugin.py

\"\"\"
Main implementation file for the '{plugin_name}' Sygnals plugin.
\"\"\"

import logging
from typing import Dict, Any, Callable # Import necessary types

# Import the base class and registry from Sygnals core API
from sygnals.plugins.api import SygnalsPluginBase, PluginRegistry
# Import other Sygnals components or external libraries as needed
# e.g., import numpy as np
# from sygnals.core.dsp import compute_fft

logger = logging.getLogger(__name__)

# --- Plugin Implementation ---

class {class_name}(SygnalsPluginBase):
    \"\"\"
    Main plugin class for '{plugin_name}'.

    Inherits from SygnalsPluginBase and implements the required properties
    and desired registration hooks.
    \"\"\"

    # --- Required Properties ---

    @property
    def name(self) -> str:
        \"\"\"Return the unique plugin name (must match plugin.toml).\"\"\"
        return "{plugin_name}"

    @property
    def version(self) -> str:
        \"\"\"Return the plugin version (must match plugin.toml).\"\"\"
        return "{plugin_version}" # Read from manifest ideally, hardcoded for template

    # --- Optional Lifecycle Hooks ---

    def setup(self, config: Dict[str, Any]):
        \"\"\"
        Initialize plugin resources here. Called once during loading.
        'config' is the resolved Sygnals configuration dictionary.
        \"\"\"
        logger.info(f"Initializing plugin '{{self.name}}' v{{self.version}}")
        # Example: Store config value or initialize a resource
        # self.my_setting = config.get('plugins', {{}}).get(self.name, {{}}).get('my_setting', 'default')
        pass

    def teardown(self):
        \"\"\"
        Clean up resources here. Called during application shutdown.
        \"\"\"
        logger.info(f"Tearing down plugin '{{self.name}}'")
        pass

    # --- Optional Registration Hooks ---
    # Uncomment and implement the hooks for the functionalities your plugin provides.

    # def register_filters(self, registry: PluginRegistry):
    #     \"\"\"Register custom filter functions.\"\"\"
    #     from . import filters # Example: Assuming filters are in filters.py
    #     registry.add_filter("my_custom_filter", filters.my_filter_func)
    #     logger.debug(f"Plugin '{{self.name}}' registered filter 'my_custom_filter'.")

    # def register_transforms(self, registry: PluginRegistry):
    #     \"\"\"Register custom transforms.\"\"\"
    #     pass # Add transform registration here

    # def register_feature_extractors(self, registry: PluginRegistry):
    #     \"\"\"Register custom feature extractors.\"\"\"
    #     pass # Add feature registration here

    # def register_visualizations(self, registry: PluginRegistry):
    #     \"\"\"Register custom visualizations.\"\"\"
    #     pass # Add visualization registration here

    # def register_audio_effects(self, registry: PluginRegistry):
    #     \"\"\"Register custom audio effects.\"\"\"
    #     pass # Add effect registration here

    # def register_augmenters(self, registry: PluginRegistry):
    #     \"\"\"Register custom data augmenters.\"\"\"
    #     pass # Add augmenter registration here

    # def register_cli_commands(self, registry: PluginRegistry):
    #     \"\"\"Register custom CLI commands/groups.\"\"\"
    #     import click
    #     from . import cli # Example: Assuming commands are in cli.py
    #     registry.add_cli_command(cli.my_command_group)
    #     logger.debug(f"Plugin '{{self.name}}' registered CLI commands.")

"""

# --- Helper Functions ---

def _sanitize_to_package_name(plugin_name: str) -> str:
    """Converts a plugin name (e.g., 'my-cool-plugin') to a valid Python package name ('my_cool_plugin')."""
    # Replace hyphens with underscores
    name = plugin_name.replace('-', '_')
    # Remove any characters not suitable for package names (keep letters, numbers, underscore)
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    # Ensure it starts with a letter or underscore
    if not re.match(r'^[a-zA-Z_]', name):
        name = '_' + name
    return name.lower()

def _sanitize_to_class_name(plugin_name: str) -> str:
    """Converts a plugin name to a CamelCase class name (e.g., 'my-cool-plugin' -> 'MyCoolPlugin')."""
    # Split by hyphen or underscore, capitalize each part, join
    parts = re.split(r'[-_]', plugin_name)
    class_name = "".join(part.capitalize() for part in parts if part)
    # Ensure it starts with a letter
    if not re.match(r'^[a-zA-Z]', class_name):
        # Find first letter and capitalize it, prepend rest
        match = re.search(r'[a-zA-Z]', class_name)
        if match:
            idx = match.start()
            class_name = class_name[idx:].capitalize() + class_name[:idx]
        else:
            class_name = "Plugin" # Fallback if no letters
    return class_name


# --- Scaffold Creation Function ---

def create_plugin_scaffold(plugin_name: str, destination_dir: Path):
    """
    Generates the directory structure and template files for a new plugin.

    Args:
        plugin_name: The desired name for the plugin (e.g., "my-filter").
        destination_dir: The parent directory where the plugin folder will be created.

    Raises:
        FileExistsError: If the target plugin directory already exists.
        OSError: If there are issues creating directories or files.
        ValueError: If the plugin name is invalid.
    """
    if not plugin_name or not re.match(r'^[a-zA-Z0-9_-]+$', plugin_name):
        raise ValueError(f"Invalid plugin name: '{plugin_name}'. Use letters, numbers, hyphens, underscores.")

    plugin_root = destination_dir / plugin_name
    package_name = _sanitize_to_package_name(plugin_name)
    class_name = _sanitize_to_class_name(plugin_name)
    package_dir = plugin_root / package_name

    logger.info(f"Creating plugin scaffold for '{plugin_name}' in '{destination_dir}'")
    logger.debug(f"Package name: {package_name}, Class name: {class_name}")

    # Check if directory already exists
    if plugin_root.exists():
        raise FileExistsError(f"Target directory already exists: {plugin_root}")

    try:
        # Create directories
        plugin_root.mkdir(parents=True)
        package_dir.mkdir()

        # --- Get core version info for templates ---
        core_v = Version(core_version) # Use imported Version
        core_version_major_minor = f"{core_v.major}.{core_v.minor}"
        # FIX: Calculate next major version correctly
        core_version_next_major = core_v.major + 1

        # --- Create template files ---
        template_context = {
            "plugin_name": plugin_name,
            "package_name": package_name,
            "class_name": class_name,
            "plugin_version": "0.1.0", # Default initial version
            "core_version_major_minor": core_version_major_minor,
            # FIX: Add the calculated next major version to the context
            "core_version_next_major": core_version_next_major,
        }

        # plugin.toml
        (plugin_root / "plugin.toml").write_text(
            PLUGIN_TOML_TEMPLATE.format(**template_context), encoding='utf-8'
        )

        # pyproject.toml
        (plugin_root / "pyproject.toml").write_text(
            PYPROJECT_TOML_TEMPLATE.format(**template_context), encoding='utf-8'
        )

        # <package_name>/__init__.py
        (package_dir / "__init__.py").write_text(
            PLUGIN_INIT_TEMPLATE.format(**template_context), encoding='utf-8'
        )

        # <package_name>/plugin.py
        # FIX: Escape curly braces intended for logging f-strings within the template
        (package_dir / "plugin.py").write_text(
            PLUGIN_PY_TEMPLATE.format(**template_context).replace("{{", "{").replace("}}", "}"),
            encoding='utf-8'
        )


        # Optional: Create empty README.md
        (plugin_root / "README.md").write_text(f"# Sygnals Plugin: {plugin_name}\n\nTODO: Add description.", encoding='utf-8')

        logger.info(f"Plugin scaffold created successfully at: {plugin_root}")

    except OSError as e:
        logger.error(f"OS error during scaffold creation: {e}")
        # Consider cleaning up partially created directories/files here if needed
        raise
    except Exception as e:
        logger.error(f"Unexpected error during scaffold creation: {e}")
        raise
