# sygnals/cli/base_cmd.py

"""
Base setup for CLI commands, including configuration loading, logging initialization,
and plugin system initialization.
"""

import logging
import click
import sys
from typing import Optional # Import Optional

from sygnals.config import load_configuration, SygnalsConfig
from sygnals.utils.logging_config import setup_logging
# Import plugin components
from sygnals.plugins.api import PluginRegistry
from sygnals.plugins.loader import PluginLoader

logger = logging.getLogger(__name__)

# --- Global Plugin Instances ---
# Instantiate registry here, loader will be created in invoke
# These need to be accessible by the main_cli module for teardown registration
plugin_registry = PluginRegistry()
plugin_loader: Optional[PluginLoader] = None # Initialize later


class ConfigGroup(click.Group):
    """
    A custom Click Group that loads configuration, sets up logging,
    and initializes the plugin system before invoking the group or its subcommands.
    Passes config, loader, and registry via the context object (ctx.obj).
    """
    def invoke(self, ctx: click.Context):
        """
        Overrides the default invoke method to set up config, logging, and plugins.
        """
        # Ensure ctx.obj is initialized as a dictionary
        if ctx.obj is None:
             ctx.obj = {}

        config_loaded_here = False
        plugins_loaded_here = False

        try:
            # 1. Load Configuration
            # Check if config is already loaded (e.g., if nested groups use this)
            if 'config' not in ctx.obj:
                config = load_configuration()
                ctx.obj['config'] = config # Store config in context dict
                config_loaded_here = True
            else:
                config: SygnalsConfig = ctx.obj['config'] # Config already loaded
                logger.debug("Configuration already loaded in context.")

            # 2. Setup Logging based on config and verbosity options
            # Only set up logging if this group instance loaded the config
            if config_loaded_here:
                verbosity = 0
                # Check params on the current context first
                if ctx.params.get('verbose', 0) > 0:
                    verbosity = ctx.params['verbose']
                if ctx.params.get('quiet', False):
                    verbosity = -1
                # If not found, check parent context
                elif ctx.parent and ctx.parent.params.get('verbose', 0) > 0:
                     verbosity = ctx.parent.params['verbose']
                elif ctx.parent and ctx.parent.params.get('quiet', False):
                     verbosity = -1

                setup_logging(config, verbosity)
                logger.debug("Logging setup complete in ConfigGroup.")
            else:
                 logger.debug("Skipping logging setup; already done by parent group.")

            # 3. Initialize and Run Plugin Loader
            # Check if plugins are already loaded
            global plugin_loader, plugin_registry # Use global instances
            if 'plugin_loader' not in ctx.obj:
                logger.debug("Initializing Plugin System...")
                # Ensure registry passed to loader is the global one used elsewhere
                plugin_loader = PluginLoader(config, plugin_registry)
                plugin_loader.discover_and_load() # Discover and load plugins
                ctx.obj['plugin_loader'] = plugin_loader
                ctx.obj['plugin_registry'] = plugin_registry # Add registry too
                plugins_loaded_here = True
                logger.debug("Plugin system initialized and loaded.")
            else:
                # Retrieve existing loader/registry if nested call
                plugin_loader = ctx.obj['plugin_loader']
                plugin_registry = ctx.obj['plugin_registry']
                logger.debug("Plugin system already initialized in context.")

            # 4. Proceed with the actual group/command invocation
            return super().invoke(ctx)

        except click.exceptions.Exit as e:
            # Re-raise Exit exceptions to let Click handle them (e.g., for --help)
            # This prevents logging Exit(0) as a critical error.
            raise e
        except Exception as e:
            # Log other critical errors during setup
            setup_logger = logging.getLogger("sygnals.setup.error")
            # Use repr(e) to potentially get more info than just str(e)
            setup_logger.critical(f"Critical error during CLI setup (config/log/plugin): {repr(e)}", exc_info=True)
            # Use basic print as fallback if logging failed
            print(f"CRITICAL SETUP ERROR: {repr(e)}", file=sys.stderr)
            # Exit cleanly on setup error
            ctx.exit(1)


# --- Common CLI Options ---
# (Keep verbose_option and quiet_option as defined before)
verbose_option = click.option(
    '-v', '--verbose',
    count=True,
    help="Increase verbosity level (-v for INFO, -vv for DEBUG)."
)
quiet_option = click.option(
    '-q', '--quiet',
    is_flag=True,
    default=False,
    help="Suppress all console output except critical errors."
)
