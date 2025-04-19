# Developing Sygnals Plugins

This guide details how to create custom plugins to extend the functionality of the Sygnals CLI toolkit. Plugins allow you to add new filters, transforms, feature extractors, audio effects, visualizations, data augmenters, and even custom CLI commands.

## Plugin Architecture Overview

Sygnals uses a plugin system based on discovery and registration:

1.  **Discovery:** Sygnals looks for plugins in two main locations:
    * **Entry Points:** Python packages installed in the environment that register a `sygnals.plugins` entry point (defined in their `pyproject.toml` or `setup.py`). This is the standard way to distribute plugins via PyPI.
    * **Local Directory:** A specific directory configured in `sygnals.toml` (`paths.plugin_dir`, typically `~/.config/sygnals/plugins/` or a project-local path). Each plugin resides in its own subdirectory within this location.
2.  **Manifest (`plugin.toml`):** Each plugin must have a `plugin.toml` file in its root directory. This file contains essential metadata like the plugin's unique name, version, description, entry point class, and Sygnals core API compatibility requirements.
3.  **Loading & Validation:** The `PluginLoader` parses the manifest, checks if the plugin's required `sygnals_api` version range is compatible with the currently running Sygnals core version, and verifies if the plugin is marked as enabled.
4.  **Instantiation:** If compatible and enabled, the loader imports the specified `entry_point` (a Python class) and instantiates it.
5.  **Registration:** The loader calls various `register_*` methods (e.g., `register_filters`, `register_features`) on the plugin instance. The plugin instance uses the provided `PluginRegistry` object to register its specific extensions (like custom filter functions or feature extractors) under unique names.
6.  **Execution:** When Sygnals performs tasks (like filtering data or extracting features), it consults the `PluginRegistry` to see if any registered plugin provides the requested functionality.

```mermaid
flowchart LR
    A[sygnals CLI] -->|load config| B(Config)
    B --> C[PluginLoader]
    C -->|scan| D{Entry Points}
    C -->|scan| E(Local Plugin Dir)
    D --> F(Manifest Check)
    E --> F
    F -->|compatible & enabled| G[Import & Instantiate Plugin]
    G -->|call setup()| H(Plugin Instance)
    H -->|call register_*()| I[PluginRegistry]
    I --> J{Filters}
    I --> K{Transforms}
    I --> L{Features}
    I --> M{... other extensions}
    A -->|uses| J & K & L & M
```

## The Plugin Interface (`SygnalsPluginBase`)

All plugins **must** inherit from the `sygnals.plugins.api.SygnalsPluginBase` abstract base class.

```python
# sygnals/plugins/api.py (Excerpt)
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Type, Optional, List, Set

# ... other imports ...

class SygnalsPluginBase(ABC):
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

    # --- Registration Hooks (Optional) ---
    def register_filters(self, registry: 'PluginRegistry'): pass
    def register_transforms(self, registry: 'PluginRegistry'): pass
    def register_feature_extractors(self, registry: 'PluginRegistry'): pass
    def register_visualizations(self, registry: 'PluginRegistry'): pass
    def register_audio_effects(self, registry: 'PluginRegistry'): pass
    def register_augmenters(self, registry: 'PluginRegistry'): pass
    def register_cli_commands(self, registry: 'PluginRegistry'): pass

    # --- Lifecycle Hooks (Optional) ---
    def setup(self, config: Dict[str, Any]): pass
    def teardown(self): pass

# ... PluginRegistry definition ...
```

**Key Requirements:**

- Implement the abstract `name` and `version` properties. These **must** return the exact values specified in your `plugin.toml`.

- Optionally, implement any relevant 

    ```
    register_*
    ```

     methods. Inside these methods, use the provided 

    ```
    registry
    ```

     object (an instance of 

    ```
    PluginRegistry
    ```

    ) to add your plugin's functionality. For example, to register a custom filter function named 

    ```
    my_awesome_filter
    ```

    :

    ```python
    def register_filters(self, registry: PluginRegistry):
        from .my_filter_module import my_filter_function # Import your function
        registry.add_filter("my_awesome_filter", my_filter_function)
    ```

- Optionally, implement `setup(config)` to initialize resources based on the global Sygnals configuration when the plugin loads.

- Optionally, implement `teardown()` to clean up resources when Sygnals exits.

## The Plugin Manifest (`plugin.toml`)

This file resides in the root directory of your plugin and provides crucial metadata.

```toml
# plugin.toml

# Unique plugin identifier (lowercase, hyphens allowed, e.g., "my-custom-filter")
name = "<unique-id>"

# Plugin version (Semantic Versioning, e.g., "0.1.0")
version = "<semver>"

# Sygnals core API compatibility range (PEP 440 specifier)
# Defines which versions of Sygnals core this plugin works with.
# Example: ">=1.0.0,<2.0.0" (Works with Sygnals 1.0.0 up to, but not including, 2.0.0)
# Example: "~=1.1" (Works with Sygnals >=1.1.0, <1.2.0)
sygnals_api = ">=<core-min>,<core-max>"

# Human-readable summary of the plugin's purpose
description = "Short summary of the plugin"

# Path to the main plugin class implementing SygnalsPluginBase
# Format: <python_module_path.submodule>:<ClassName>
# Example: "my_custom_filter.plugin:MyCustomFilterPlugin"
entry_point = "module.path:ClassName"

# Optional: List of additional Python package dependencies required by this plugin
# This is primarily for informational purposes. Actual dependencies for installation
# must be declared in your plugin's pyproject.toml or setup.py.
# dependencies = ["numpy>=1.20.0", "some-other-lib"]
```

**Fields:**

- `name`: A unique identifier for your plugin. Used internally and in CLI commands. Keep it concise and descriptive (e.g., `kalman-filter`, `voice-activity-detector`).
- `version`: Your plugin's version, following [Semantic Versioning](https://semver.org/).
- `sygnals_api`: **Crucial.** Specifies the range of Sygnals core versions your plugin is compatible with, using [PEP 440 specifiers](https://www.google.com/search?q=https://peps.python.org/pep-0440/%23version-specifiers). Sygnals will skip loading plugins where the current core version falls outside this range.
- `description`: A brief summary shown in `sygnals plugin list`.
- `entry_point`: The Python import path to your main plugin class (which inherits from `SygnalsPluginBase`).
- `dependencies` (Optional): A list of external Python libraries your plugin needs.

## Versioning and Compatibility

- **Core API Version:** Sygnals core maintains its own version (e.g., `1.0.0`) defined in `sygnals/version.py`.

- Plugin Compatibility (`sygnals_api`):

     Your plugin declares the compatible core API range in 

    ```
    plugin.toml
    ```

    . Use specifiers carefully:

    - `>=1.0.0,<2.0.0`: Compatible with all 1.x versions starting from 1.0.0. Good practice if you rely on features introduced in 1.0 and don't expect breaking changes until 2.0.
    - `~=1.1`: Compatible with >=1.1.0 and &lt;1.2.0. Use if you depend on features specific to 1.1.x.

- **Loading Check:** Before loading, Sygnals compares its core version against your plugin's `sygnals_api` specifier using the `packaging` library. Incompatible plugins are skipped with a warning.

## Getting Started: `sygnals plugin scaffold`

The easiest way to start a new plugin is using the built-in scaffold tool:

```bash
sygnals plugin scaffold <your-plugin-name> [--dest <output-directory>]
```

- Replace `<your-plugin-name>` with your desired plugin name (e.g., `awesome-feature`).
- Optionally use `--dest` to specify where the plugin directory should be created (defaults to the current directory).

This command generates a directory structure like this:

```
<your-plugin-name>/
├── plugin.toml          # Pre-filled manifest template
├── pyproject.toml       # Basic pyproject.toml with entry point config
├── README.md            # Placeholder README
└── <package_name>/      # Python package for your code
    ├── __init__.py      # Imports your main plugin class
    └── plugin.py        # Your main plugin class, inheriting SygnalsPluginBase
```

**Next Steps After Scaffolding:**

1. **Review `plugin.toml`:** Adjust the `description` and confirm the `sygnals_api` range.
2. Review `pyproject.toml`:
    - Add your author details.
    - Choose a license.
    - Add any external Python `dependencies` your plugin requires.
3. **Implement Logic:** Write your core plugin functionality (e.g., filter functions, feature extractors) potentially in new modules within the `<package_name>` directory.
4. Implement `plugin.py`:
    - Fill in the `setup`, `teardown`, and relevant `register_*` methods in your main plugin class.
    - Import and register your implemented functionalities using the `registry` object.
5. **Test:** Write tests for your plugin's logic.
6. Install/Deploy:
    - For local development, you can install your plugin in editable mode: `pip install -e .` from within the `<your-plugin-name>` directory. This makes it discoverable via entry points.
    - Alternatively, copy the entire `<your-plugin-name>` directory into your Sygnals local plugin directory (check `sygnals config get paths.plugin_dir` or `~/.config/sygnals/plugins/`).

## Testing Plugins

- **Unit Tests:** Test your core logic functions (filters, feature extractors) independently, just like any Python code. Use `pytest` and fixtures.
- Integration Tests:
    - Create test signals.
    - Instantiate your plugin class directly in your test code.
    - Create a mock `PluginRegistry`.
    - Call your plugin's `register_*` methods with the mock registry.
    - Verify that the correct functions were registered.
    - Call the registered functions with test data and assert the expected output.
    - (Advanced) You could potentially run `sygnals` CLI commands using `CliRunner` with your test plugin installed in the environment or placed in a temporary local plugin directory configured for the test run.

## Best Practices

- Keep plugins focused on a specific task or set of related functionalities.
- Use clear and unique names for your plugin and its registered extensions.
- Manage dependencies properly in `pyproject.toml`. Avoid depending on Sygnals itself here.
- Define a clear and stable API compatibility range (`sygnals_api`).
- Add logging within your plugin using `logging.getLogger(__name__)`.
- Include documentation (docstrings, README) for your plugin.