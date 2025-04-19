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
    Handles exceptions gracefully.
    """
    def invoke(self, ctx: click.Context):
        """
        Overrides the default invoke method to set up config, logging, plugins,
        and handle exceptions during command execution.
        """
        # Ensure ctx.obj is initialized as a dictionary
        if ctx.obj is None:
             ctx.obj = {}

        config_loaded_here = False
        plugins_loaded_here = False
        setup_success = False

        try:
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
                verbosity = 0
                if ctx.params.get('verbose', 0) > 0: verbosity = ctx.params['verbose']
                if ctx.params.get('quiet', False): verbosity = -1
                elif ctx.parent and ctx.parent.params.get('verbose', 0) > 0: verbosity = ctx.parent.params['verbose']
                elif ctx.parent and ctx.parent.params.get('quiet', False): verbosity = -1
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
                plugin_loader = ctx.obj['plugin_loader']
                plugin_registry = ctx.obj['plugin_registry']
                logger.debug("Plugin system already initialized in context.")

            setup_success = True # Mark setup as successful before invoking command

            # 4. Proceed with the actual group/command invocation
            return super().invoke(ctx)

        except click.exceptions.Exit as e:
            # Let Click handle --help, ctx.exit(), etc. cleanly
            raise e
        except click.ClickException as e:
            # Handle known Click errors originating from commands (UsageError, MissingParameter, Abort etc.)
            # Log the error, show the user-friendly message via Click, and re-raise.
            # The test runner with catch_exceptions=True should capture the original exception.
            logger.error(f"Command error: {e.__class__.__name__} - {e}", exc_info=True) # Log with traceback
            e.show() # Print the error message to stderr as Click would
            # Re-raise the original Click exception.
            # Click's default handling or the test runner will manage the exit code.
            raise e
        except Exception as e:
            # Handle truly unexpected errors (likely bugs in setup or command)
            error_logger = logging.getLogger("sygnals.error") # Use a general error logger
            if setup_success:
                # Error occurred *during* command execution
                error_logger.critical(f"Unexpected error during command execution: {repr(e)}", exc_info=True)
                print(f"CRITICAL UNEXPECTED ERROR: {repr(e)}", file=sys.stderr)
            else:
                # Error occurred during setup (config/log/plugin)
                error_logger.critical(f"Critical error during CLI setup: {repr(e)}", exc_info=True)
                print(f"CRITICAL SETUP ERROR: {repr(e)}", file=sys.stderr)

            # Exit with non-zero code for unexpected errors
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
