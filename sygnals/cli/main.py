# sygnals/cli/main.py

"""
Main entry point for the Sygnals CLI application.
Uses Click for command-line interface handling.
Integrates plugin loading.
"""

import click
import logging
import sys
import atexit # For calling teardown hooks
from typing import Dict, Any, List, Optional, Tuple, Set, Type

from sygnals.version import __version__
from .base_cmd import ConfigGroup, verbose_option, quiet_option # ConfigGroup handles config/logging
from sygnals.config import SygnalsConfig
from sygnals.plugins.api import PluginRegistry # Import registry
from sygnals.plugins.loader import PluginLoader # Import loader

# Import command groups/functions
from .plugin_cmd import plugin_cmd
from .segment_cmd import segment_cmd # Import the new segment command group
# from .augment_cmd import augment_cmd # Will be added later
# from .save_cmd import save_cmd # Will be added later
# from .features_cmd import features_cmd # Will be added later

# from .analyze_cmd import analyze
# from .transform_cmd import transform
# from .filter_cmd import filter_cmd
# from .audio_cmd import audio
# from .visualize_cmd import visualize
# ... other commands

logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# --- Global Plugin Instances ---
# Instantiate registry and loader once
# These will be populated during CLI invocation by ConfigGroup
plugin_registry = PluginRegistry()
plugin_loader: Optional[PluginLoader] = None # Initialize later

# --- Teardown Hook Registration ---
def _call_plugin_teardown():
    """Calls teardown hooks for all loaded plugins."""
    global plugin_loader
    if plugin_loader:
        logger.debug("Running plugin teardown hooks...")
        plugin_loader.call_teardown_hooks()
    else:
        logger.debug("No plugin loader instance found, skipping teardown.")

atexit.register(_call_plugin_teardown) # Register teardown to run on exit


# --- Main CLI Group ---
@click.group(context_settings=CONTEXT_SETTINGS, cls=ConfigGroup) # Use ConfigGroup
@click.version_option(__version__, '-V', '--version', package_name='sygnals', prog_name='sygnals')
@verbose_option
@quiet_option
@click.pass_context # Pass context to access config, loader, registry from ctx.obj
def main_cli(ctx, verbose: int, quiet: bool):
    """
    Sygnals v1.0.0: A versatile CLI for signal and audio processing,
    tailored for data science workflows. Includes plugin support.

    Configuration is loaded from:
    Defaults -> ./sygnals.toml -> ~/.config/sygnals/sygnals.toml -> Env Vars

    Use -v for verbose output, -vv for debug output, -q for quiet mode.
    """
    # Config loading, logging setup, and plugin loading happen in ConfigGroup.invoke
    # We can access the context object here if needed
    if isinstance(ctx.obj, dict) and 'config' in ctx.obj and 'plugin_loader' in ctx.obj:
        config: SygnalsConfig = ctx.obj['config']
        loader: PluginLoader = ctx.obj['plugin_loader']
        registry: PluginRegistry = ctx.obj['plugin_registry']
        logger.debug(f"Sygnals CLI group invoked. Config loaded. {len(loader.loaded_plugins)} plugins loaded.")
        # Example: Accessing loaded plugin names from registry
        # logger.debug(f"Loaded plugin names: {registry.loaded_plugin_names}")
    else:
        # This might happen if ConfigGroup failed or context wasn't set up correctly
        logger.error("Context object (config/plugins) not found in main_cli. Setup might have failed.")


# --- Register Commands ---
@main_cli.command()
@click.pass_context
def hello(ctx):
    """A simple example command."""
    # Access context object populated by ConfigGroup
    if isinstance(ctx.obj, dict) and 'config' in ctx.obj:
        config: SygnalsConfig = ctx.obj['config']
        registry: PluginRegistry = ctx.obj['plugin_registry']
        logger.info("Hello command executing.")
        logger.debug(f"Default sample rate from config: {config.defaults.default_sample_rate}")
        logger.debug(f"Registered plugin filters: {registry.list_filters()}")
        click.echo("Hello from Sygnals!")
    else:
         logger.error("Configuration or plugin context not available in 'hello' command.")
         click.echo("Error: Context not loaded correctly.", err=True)
         ctx.exit(1)

# Add the new plugin command group
main_cli.add_command(plugin_cmd)
# Add the new segment command group
main_cli.add_command(segment_cmd)

# Add other commands/groups as they are refactored or created
# main_cli.add_command(features_cmd)
# main_cli.add_command(augment_cmd)
# main_cli.add_command(save_cmd)
# main_cli.add_command(analyze)
# main_cli.add_command(transform)
# main_cli.add_command(filter_cmd, name="filter")
# main_cli.add_command(audio)
# main_cli.add_command(visualize)
# ...

cli = main_cli

if __name__ == "__main__":
    cli()
