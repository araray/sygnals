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
    Passes config, loader, registry via the context object (ctx.obj).
    Handles exceptions during setup gracefully.
    """
    def invoke(self, ctx: click.Context):
        """
        Overrides the default invoke method to set up config, logging, plugins,
        and handle exceptions during setup. Exceptions during command execution
        are allowed to propagate.
        """
        # Ensure ctx.obj is initialized as a dictionary
        if ctx.obj is None:
             ctx.obj = {}

        config_loaded_here = False
        plugins_loaded_here = False
        setup_success = False

        try:
            # --- Setup Phase ---
            # 1. Load Configuration
            if 'config' not in ctx.obj:
                config = load_configuration()
                ctx.obj['config'] = config
                config_loaded_here = True
            else:
                config: SygnalsConfig = ctx.obj['config']
                logger.debug("Configuration already loaded in context.")

            # 2. Setup Logging based on config and verbosity options
            if config_loaded_here:
                # Determine verbosity from current or parent context params
                verbosity = 0
                verbose_flag = ctx.params.get('verbose', 0)
                quiet_flag = ctx.params.get('quiet', False)
                if quiet_flag:
                    verbosity = -1
                elif verbose_flag > 0:
                    verbosity = verbose_flag
                elif ctx.parent: # Check parent context if flags not on current command
                     parent_verbose = ctx.parent.params.get('verbose', 0)
                     parent_quiet = ctx.parent.params.get('quiet', False)
                     if parent_quiet: verbosity = -1
                     elif parent_verbose > 0: verbosity = parent_verbose

                setup_logging(config, verbosity)
                logger.debug("Logging setup complete in ConfigGroup.")
            else:
                 logger.debug("Skipping logging setup; already done by parent group.")

            # 3. Initialize and Run Plugin Loader
            global plugin_loader, plugin_registry # Use global instances
            if 'plugin_loader' not in ctx.obj:
                logger.debug("Initializing Plugin System...")
                plugin_loader = PluginLoader(config, plugin_registry)
                plugin_loader.discover_and_load()
                ctx.obj['plugin_loader'] = plugin_loader
                ctx.obj['plugin_registry'] = plugin_registry
                plugins_loaded_here = True
                logger.debug("Plugin system initialized and loaded.")
            else:
                # Ensure global vars are updated if context already has them
                # (might happen in nested groups, though less likely with current structure)
                if 'plugin_loader' in ctx.obj: plugin_loader = ctx.obj['plugin_loader']
                if 'plugin_registry' in ctx.obj: plugin_registry = ctx.obj['plugin_registry']
                logger.debug("Plugin system already initialized in context.")

            setup_success = True # Mark setup as successful before invoking command

            # --- Command Execution Phase ---
            # Let exceptions from super().invoke() propagate up to Click/CliRunner
            return super().invoke(ctx)

        except click.exceptions.Exit as e:
            # Let Click handle --help, ctx.exit(), etc. cleanly
            raise e
        except Exception as e:
            # Handle ONLY unexpected errors during the SETUP phase
            if not setup_success:
                error_logger = logging.getLogger("sygnals.error")
                error_logger.critical(f"Critical error during CLI setup: {repr(e)}", exc_info=True)
                # Print basic error to stderr as logging might not be fully set up
                print(f"CRITICAL SETUP ERROR: {repr(e)}", file=sys.stderr)
                # Exit with non-zero code for setup errors
                ctx.exit(1)
            else:
                # If setup was successful, errors during command execution
                # should have been allowed to propagate. This block might
                # not be reached unless super().invoke() itself raises
                # an unexpected non-Click exception.
                error_logger = logging.getLogger("sygnals.error")
                error_logger.critical(f"Unexpected error during command invocation (after setup): {repr(e)}", exc_info=True)
                # Re-raise the original exception to let Click/CliRunner handle it
                raise


# --- Common CLI Options ---
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
