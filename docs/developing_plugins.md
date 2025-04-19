# Developing Sygnals Plugins

This guide provides comprehensive instructions for creating custom plugins to extend the functionality of the Sygnals CLI toolkit. Plugins allow you to add new filters, transforms, feature extractors, audio effects, visualizations, data augmenters, and even custom CLI commands, integrating seamlessly with the core Sygnals workflows.

## 1. Plugin Architecture Overview

Sygnals employs a flexible plugin system designed for extensibility, compatibility, and discoverability. The core components interact as follows:

1.  **Discovery:** When Sygnals starts, the `PluginLoader` searches for potential plugins in two primary locations:
    * **Entry Points:** Python packages installed in the environment (e.g., via `pip`) that declare an entry point under the `sygnals.plugins` group in their `pyproject.toml` or `setup.py`. This is the standard method for distributing shareable plugins.
    * **Local Directory:** A specific directory defined in the Sygnals configuration (`sygnals.toml` under `[paths].plugin_dir`, typically `~/.config/sygnals/plugins/` or a project-specific path). Each plugin must reside in its own subdirectory within this location.
2.  **Manifest (`plugin.toml`):** Every plugin (whether local or installed via entry point) **must** contain a `plugin.toml` file in its root directory. This file acts as the plugin's descriptor, containing essential metadata: its unique name, version, target Sygnals API compatibility range, description, and the Python entry point class.
3.  **Loading & Validation:** For each discovered plugin manifest:
    * The `PluginLoader` parses `plugin.toml`.
    * It checks the plugin's `sygnals_api` version requirement against the current Sygnals core version using semantic versioning rules. Incompatible plugins are skipped with a warning.
    * It checks the plugin's enabled/disabled status (managed via the `sygnals plugin enable/disable` commands and stored in `plugins.yaml`). Disabled plugins are skipped.
4.  **Instantiation:** If a plugin is compatible and enabled, the `PluginLoader` imports the Python class specified in the manifest's `entry_point` field (e.g., `my_plugin.plugin:MyPluginClass`). It then creates an instance of this class. The entry point class **must** inherit from `sygnals.plugins.api.SygnalsPluginBase`.
5.  **Setup:** The `setup(config)` method of the plugin instance is called, passing the resolved Sygnals configuration dictionary. This allows the plugin to perform any necessary initialization based on global settings.
6.  **Registration:** The `PluginLoader` calls various `register_*` methods on the plugin instance (e.g., `register_filters`, `register_feature_extractors`, `register_cli_commands`). Inside these methods, the plugin uses the provided `PluginRegistry` object to register its specific extensions (like custom filter functions, feature extractors, or Click commands) under unique names.
7.  **Execution:** During normal operation, when Sygnals needs to perform an action involving an extensible component (e.g., applying a filter named "my_custom_filter"), it consults the `PluginRegistry` to find the corresponding function or class registered by a plugin.
8.  **Teardown:** When Sygnals exits gracefully, the `teardown()` method of each successfully loaded plugin instance is called, allowing plugins to clean up any resources they might have acquired.

```mermaid
flowchart LR
    A[sygnals CLI Start] --> B(Load Config);
    B --> C[PluginLoader];
    C -->|scan| D{Entry Points};
    C -->|scan| E(Local Plugin Dir);
    D --> F(Find Manifests);
    E --> F;
    F --> G{Parse & Validate Manifest};
    G -- compatible & enabled --> H[Import & Instantiate Plugin];
    G -- incompatible / disabled --> I[Skip Plugin];
    H --> J(Call plugin.setup());
    J --> K(Call plugin.register_*());
    K --> L[PluginRegistry];
    A -->|uses extensions| L;
    L --> M{Filters};
    L --> N{Features};
    L --> O{Transforms};
    L --> P{...};
    A --> Q[Sygnals Exit];
    Q --> R(Call plugin.teardown());

    subgraph Plugin Loading
        direction LR
        C; D; E; F; G; H; I; J; K; L; R;
    end

    subgraph Core Execution
        direction LR
        A; B; M; N; O; P; Q;
    end
```

## 2\. The Plugin Interface (`SygnalsPluginBase`)

The foundation of any Sygnals plugin is a class that inherits from the abstract base class `sygnals.plugins.api.SygnalsPluginBase`.

```python
# sygnals/plugins/api.py (Key Parts)
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Type, Optional, List, Set, Union

# Potentially import Click types if needed
try:
    import click
    ClickCommandType = Union[click.Command, click.Group, Callable[..., Any]]
except ImportError:
    click = None
    ClickCommandType = Callable[..., Any]

logger = logging.getLogger(__name__)

class PluginRegistry:
    # ... (Registry methods as defined in api.py) ...
    def add_filter(self, name: str, filter_callable: Callable): ...
    def add_transform(self, name: str, transform_callable: Callable): ...
    def add_feature(self, name: str, feature_callable: Callable): ...
    def add_visualization(self, name: str, vis_callable: Callable): ...
    def add_effect(self, name: str, effect_callable: Callable): ...
    def add_augmenter(self, name: str, augmenter_callable: Callable): ...
    def add_cli_command(self, command: ClickCommandType): ...
    # ... (Getter and Lister methods) ...

class SygnalsPluginBase(ABC):
    """Abstract base class for all Sygnals plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the plugin (must match plugin.toml)."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version of the plugin (must match plugin.toml)."""
        pass

    # --- Registration Hooks (Optional Implementation by Subclasses) ---

    def register_filters(self, registry: PluginRegistry): pass
    def register_transforms(self, registry: PluginRegistry): pass
    # Note: Correct hook name is register_feature_extractors
    def register_feature_extractors(self, registry: PluginRegistry): pass
    def register_visualizations(self, registry: PluginRegistry): pass
    # Note: Correct hook name is register_audio_effects
    def register_audio_effects(self, registry: PluginRegistry): pass
    def register_augmenters(self, registry: PluginRegistry): pass
    def register_cli_commands(self, registry: PluginRegistry): pass

    # --- Lifecycle Hooks (Optional Implementation by Subclasses) ---

    def setup(self, config: Dict[str, Any]): pass
    def teardown(self): pass

```

**Implementation Requirements:**

1.  **Inheritance:** Your main plugin class **must** inherit from `SygnalsPluginBase`.
2.  **Abstract Properties:** You **must** implement the `name` and `version` properties. These should return the exact strings defined in your plugin's `plugin.toml` file.
3.  **Registration Hooks:** Implement the `register_*` methods corresponding to the types of extensions your plugin provides (e.g., `register_filters`, `register_feature_extractors`, `register_audio_effects`). Inside each implemented hook, use the methods of the passed `registry` object (an instance of `PluginRegistry`) to register your functionality.
      * Example: `registry.add_filter("my_custom_filter", my_filter_function)`
      * The `name` you provide during registration is how users will refer to your extension in Sygnals commands or configurations. Choose clear, unique names.
4.  **Lifecycle Hooks (Optional):**
      * `setup(config)`: Implement this method if your plugin needs initialization logic that depends on the global Sygnals configuration (e.g., reading API keys, setting up connections, pre-loading models). Store any necessary state as instance variables within your plugin class.
      * `teardown()`: Implement this method to release any resources (files, network connections, etc.) acquired during `setup` or runtime. This is called when Sygnals exits.

## 3\. The Plugin Manifest (`plugin.toml`)

Every plugin requires a `plugin.toml` file in its root directory (the directory containing the plugin's main package and potentially `pyproject.toml`). This TOML file provides essential metadata for discovery, validation, and loading.

```toml
# plugin.toml Example

# REQUIRED FIELDS

# Unique plugin identifier (lowercase, hyphens allowed, e.g., "my-custom-filter")
# This is used internally and in CLI commands (e.g., 'sygnals plugin disable my-custom-filter').
name = "my-awesome-feature"

# Plugin version (Semantic Versioning - [https://semver.org/](https://semver.org/), e.g., "0.1.0", "1.2.3-alpha.1")
# Used for display and potentially dependency resolution in the future.
version = "0.1.0"

# Sygnals core API compatibility range (PEP 440 Version Specifiers - [https://peps.python.org/pep-0440/#version-specifiers](https://peps.python.org/pep-0440/#version-specifiers))
# CRUCIAL: Defines which versions of Sygnals core this plugin is compatible with.
# The loader will skip plugins where the current core version doesn't match this specifier.
# Examples:
#   ">=1.0.0,<2.0.0"  # Works with Sygnals 1.x starting from 1.0.0
#   "~=1.1"           # Works with Sygnals >=1.1.0, <1.2.0
#   "==1.0.5"         # Works only with Sygnals 1.0.5
sygnals_api = ">=1.0.0,<2.0.0" # Adjust based on the Sygnals version you target

# Human-readable summary of the plugin's purpose. Shown in 'sygnals plugin list'.
description = "Extracts the awesome feature from audio signals."

# Path to the main plugin class implementing SygnalsPluginBase.
# Format: <python_module_path.submodule>:<ClassName>
# Example: "my_awesome_feature.plugin:MyAwesomeFeaturePlugin"
entry_point = "my_awesome_feature.plugin:MyAwesomeFeaturePlugin"


# OPTIONAL FIELDS

# List of additional Python package dependencies required by this plugin.
# This is primarily informational for the user viewing the manifest.
# Actual installation dependencies MUST be declared in the plugin's
# pyproject.toml (recommended) or setup.py for 'pip install' to work correctly.
# Sygnals itself does not automatically install these dependencies.
# dependencies = [
#     "numpy>=1.20.0",
#     "scikit-learn==1.0.2",
# ]

# Author information (optional, for display)
# author = "Your Name"
# author_email = "your.email@example.com"

# Plugin homepage URL (optional, for display)
# url = "[https://github.com/your_username/my-awesome-feature](https://github.com/your_username/my-awesome-feature)"
```

**Key Fields Explained:**

  * `name`: The unique identifier for your plugin. Use lowercase letters, numbers, and hyphens.
  * `version`: Your plugin's version number, following [Semantic Versioning](https://semver.org/).
  * `sygnals_api`: **Critically important.** Defines the compatible range of the Sygnals *core* application your plugin works with. Use [PEP 440 specifiers](https://www.google.com/url?sa=E&source=gmail&q=https://peps.python.org/pep-0440/#version-specifiers). If the running Sygnals version is outside this range, your plugin will not be loaded.
  * `description`: A short summary displayed to users.
  * `entry_point`: The Python import path to your main plugin class (the one inheriting `SygnalsPluginBase`).
  * `dependencies` (Optional): Lists runtime dependencies for informational purposes. *Actual installation dependencies must be handled by your plugin's packaging (`pyproject.toml`)*.

## 4\. Versioning and Compatibility

Robust versioning ensures plugins work correctly with the host application.

  * **Sygnals Core Version:** The main Sygnals application has its own version (e.g., `1.0.0`), defined in `sygnals/version.py`.
  * **Plugin Version:** Your plugin has its own independent version defined in `plugin.toml` (and `pyproject.toml`).
  * **API Compatibility (`sygnals_api`):** This field in your `plugin.toml` links the two. It declares the range of Sygnals *core* versions your plugin is designed to work with.
      * **Best Practice:** Use a range like `>=MAJOR.MINOR.0,<NEXT_MAJOR.0.0`. For example, if developing against Sygnals `1.2.x`, use `sygnals_api = ">=1.2.0,<2.0.0"`. This assumes Sygnals follows semantic versioning, where minor releases (`1.3.0`) are backward-compatible, but major releases (`2.0.0`) may introduce breaking changes.
      * **Tighter Constraints:** If you rely on a feature introduced in a specific patch release (e.g., `1.2.5`), you might use `sygnals_api = ">=1.2.5,<2.0.0"`. If you depend on behavior specific to a minor version range, use `~=1.2` (equivalent to `>=1.2.0,<1.3.0`).
  * **Loading Check:** Before attempting to load a plugin, the `PluginLoader` compares the current `sygnals.__version__` against the plugin's `sygnals_api` specifier using the `packaging` library. If the core version does not fall within the specified range, the plugin is skipped, and a warning is logged.

## 5\. Getting Started: `sygnals plugin scaffold`

The quickest way to create the basic structure for a new plugin is the built-in scaffold command:

```bash
sygnals plugin scaffold <your-plugin-name> [--dest <output-directory>]
```

  * Replace `<your-plugin-name>` with the desired unique name for your plugin (e.g., `spectral-flux-feature`).
  * Use `--dest` (optional) to specify where the plugin's root directory should be created (defaults to the current working directory).

This command generates the following structure:

```
<output-directory>/
└── <your-plugin-name>/       # Plugin Root Directory
    ├── plugin.toml          # Pre-filled manifest template
    ├── pyproject.toml       # Basic pyproject.toml with entry point config
    ├── README.md            # Placeholder README
    └── <package_name>/      # Python package for your plugin code
        ├── __init__.py      # Imports your main plugin class
        └── plugin.py        # Your main plugin class (inheriting SygnalsPluginBase)
```

  * `<package_name>` is automatically generated from `<your-plugin-name>` by replacing hyphens with underscores (e.g., `spectral_flux_feature`).

**Next Steps After Scaffolding:**

1.  **Edit `plugin.toml`:**
      * Update the `description`.
      * Verify the `sygnals_api` range matches the Sygnals version you are targeting.
      * Add optional fields like `author` or `dependencies` if desired.
2.  **Edit `pyproject.toml`:**
      * Add your `authors` details.
      * Choose a `license`.
      * **Crucially, add any external Python libraries your plugin requires to the `[project].dependencies` list.**
      * Update optional `[project.urls]`.
3.  **Implement Logic:** Write the core functionality of your plugin (e.g., the filter algorithm, the feature calculation) within the `<package_name>` directory. You can create additional `.py` files within this package as needed (e.g., `filters.py`, `utils.py`).
4.  **Implement `plugin.py`:**
      * In the generated class (e.g., `SpectralFluxFeaturePlugin`), implement the necessary `register_*` methods.
      * Import your core logic functions/classes from other modules within your package and register them using the `registry` object passed to the hooks.
      * Implement `setup` and `teardown` if needed.
5.  **Write Tests:** Create tests for your plugin's logic (see Testing section below).
6.  **Install/Use:**
      * **Development:** Navigate into the `<your-plugin-name>` directory and run `pip install -e .`. This installs your plugin in "editable" mode, making it discoverable via the `sygnals.plugins` entry point by Sygnals. Changes you make to the code will be reflected immediately without reinstalling.
      * **Local Deployment:** Alternatively, copy the entire `<your-plugin-name>` directory into the Sygnals local plugin directory (find the path using `sygnals config get paths.plugin_dir` or check `~/.config/sygnals/plugins/`). Sygnals will discover it from there on the next run.

## 6\. Plugin Management (CLI)

Sygnals provides CLI commands to manage discovered plugins:

  * `sygnals plugin list`: Shows all discovered plugins (from both entry points and the local directory), their version, source, compatibility status (loaded, disabled, incompatible, error), and description.
  * `sygnals plugin enable <plugin-name>`: Marks a *discovered* plugin as enabled for future runs. Updates the state file (`plugins.yaml`).
  * `sygnals plugin disable <plugin-name>`: Marks a *discovered* plugin as disabled. Updates the state file.
  * `sygnals plugin scaffold ...`: (As described above) Generates a new plugin template.

**Note:** `sygnals plugin install <source>` and `sygnals plugin uninstall <name>` commands are planned but not yet implemented in the core as of spec v1.0.0. Installation currently relies on standard Python packaging tools (`pip`) for entry-point plugins or manual copying for local plugins.

## 7\. Best Practices

  * **Focus:** Keep plugins focused on a specific piece of functionality (e.g., one type of filter, a related set of features).
  * **Naming:** Use clear, descriptive, and unique names for your plugin (`name` in `plugin.toml`) and its registered extensions (the names passed to `registry.add_*` methods). Avoid overly generic names.
  * **Dependencies:** Clearly declare all external Python dependencies in your plugin's `pyproject.toml`. Avoid depending on Sygnals itself in `pyproject.toml`; API compatibility is handled by `sygnals_api` in `plugin.toml`.
  * **Error Handling:** Implement robust error handling within your plugin's logic. Catch expected exceptions and log informative messages. Unhandled exceptions in registration hooks or lifecycle methods may prevent the plugin from loading fully.
  * **Logging:** Use Python's standard `logging` module within your plugin. Get a logger specific to your plugin using `logging.getLogger(__name__)`. Sygnals configures the root logger, so your plugin's logs will be handled according to the user's settings.
  * **Documentation:** Include docstrings for your plugin class, registration methods, and core logic functions. Consider adding a `README.md` to your plugin's root directory explaining its purpose, usage, and any configuration options.
  * **Configuration:** If your plugin needs configuration, consider accessing plugin-specific sections within the main Sygnals configuration dictionary passed to the `setup(config)` method. Users could potentially add a `[plugins.<your-plugin-name>]` section to their `sygnals.toml`.

## 8\. Testing Strategies

Thorough testing is crucial for plugin reliability.

  * **Unit Tests:**
      * Test your core logic functions (filters, feature calculations, etc.) in isolation, just like any standard Python code. Use `pytest` and mock any external dependencies or Sygnals components if necessary.
      * Focus on testing different inputs, edge cases, and expected outputs of your algorithms.
  * **Integration Tests (within your plugin's test suite):**
      * Instantiate your main plugin class directly in your test code.
      * Create a mock `PluginRegistry` instance (or potentially use the real one if dependencies are simple).
      * Call your plugin's `register_*` methods with the registry.
      * Assert that the correct items were registered in the registry (e.g., check `registry.list_features()`).
      * Retrieve the registered callable from the registry and call it with test data, asserting the expected outcome.
      * Test the `setup` and `teardown` methods if they contain significant logic, potentially mocking the `config` dictionary.
  * **Testing with Sygnals Core (More Advanced):**
      * For full end-to-end testing, you might need to install your plugin (e.g., using `pip install -e .` in a test environment) and then use `click.testing.CliRunner` to invoke Sygnals CLI commands that utilize your plugin's functionality. This is more complex to set up but provides the highest confidence.

## 9\. Worked Example: Simple Gain Filter Plugin

Let's create a basic plugin named `simple-gain` that registers a filter to multiply the signal by a constant factor.

**1. Scaffold the Plugin:**

```bash
sygnals plugin scaffold simple-gain --dest ./my_plugins
cd ./my_plugins/simple-gain
```

**2. Edit `plugin.toml`:**

```toml
# ./my_plugins/simple-gain/plugin.toml
name = "simple-gain"
version = "0.1.0"
sygnals_api = ">=1.0.0,<2.0.0" # Assuming compatibility with Sygnals 1.x
description = "Applies a simple gain (multiplication) factor to the signal."
entry_point = "simple_gain.plugin:SimpleGainPlugin"
```

**3. Edit `pyproject.toml`:**

```toml
# ./my_plugins/simple-gain/pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "simple-gain"
version = "0.1.0"
description = "Applies a simple gain (multiplication) factor to the signal."
authors = [ { name="Your Name", email="your.email@example.com" } ] # Add your info
license = { text="MIT" } # Choose a license
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
]
keywords = ["sygnals", "plugin", "audio", "filter", "gain"]

dependencies = [
    "numpy" # Our filter uses numpy
]

[project.entry-points."sygnals.plugins"]
simple-gain = "simple_gain.plugin:SimpleGainPlugin"
```

**4. Implement the Filter Logic:**

Create a new file `simple_gain/filters.py`:

```python
# ./my_plugins/simple-gain/simple_gain/filters.py
import numpy as np
from numpy.typing import NDArray
import logging

logger = logging.getLogger(__name__)

def apply_gain_filter(data: NDArray[np.float64], gain_factor: float = 2.0) -> NDArray[np.float64]:
    """Multiplies the input signal by a gain factor."""
    logger.debug(f"Applying gain factor: {gain_factor}")
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    return (data * gain_factor).astype(np.float64)
```

**5. Implement the Plugin Class (`simple_gain/plugin.py`):**

Modify the scaffolded `plugin.py`:

```python
# ./my_plugins/simple-gain/simple_gain/plugin.py
import logging
from typing import Dict, Any
from sygnals.plugins.api import SygnalsPluginBase, PluginRegistry

# Import our filter function
from .filters import apply_gain_filter

logger = logging.getLogger(__name__)

class SimpleGainPlugin(SygnalsPluginBase):
    @property
    def name(self) -> str: return "simple-gain"
    @property
    def version(self) -> str: return "0.1.0" # Read from manifest ideally

    def setup(self, config: Dict[str, Any]):
        logger.info(f"Initializing plugin '{self.name}' v{self.version}")

    def teardown(self):
        logger.info(f"Tearing down plugin '{self.name}'")

    def register_filters(self, registry: PluginRegistry):
        """Register the custom gain filter."""
        # Register the filter function under the name 'apply_gain'
        # Users would call it via e.g., 'sygnals filter --type apply_gain ...'
        registry.add_filter("apply_gain", apply_gain_filter)
        logger.debug(f"Plugin '{self.name}' registered filter 'apply_gain'.")

```

**6. Install and Use:**

  * Navigate to `./my_plugins/simple-gain` in your terminal.
  * Run `pip install -e .`
  * Run `sygnals plugin list` - you should see `simple-gain` listed as loaded.
  * (Assuming a `sygnals filter` command exists that uses registered filters) You could potentially use it like: `sygnals filter input.wav --type apply_gain --params '{"gain_factor": 1.5}' --output output.wav`

