# sygnals/cli/base_cmd.py

"""
Base setup for CLI commands, including configuration loading and logging initialization.
"""

import logging
import click
import sys

from sygnals.config import load_configuration, SygnalsConfig
from sygnals.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Rename to ConfigGroup and inherit from click.Group
class ConfigGroup(click.Group):
    """
    A custom Click Group that loads configuration and sets up logging
    before invoking the group or its subcommands.
    Passes the config object via the context.
    """
    def invoke(self, ctx: click.Context):
        """
        Overrides the default invoke method to set up config and logging first.
        """
        # Ensure ctx.obj is initialized if not already done (e.g., by parent group)
        if ctx.obj is None:
             ctx.obj = {} # Initialize as dict, can store config here

        # 1. Load Configuration
        # Check if config is already loaded (e.g., if nested groups use this)
        if not isinstance(ctx.obj, SygnalsConfig):
            config = load_configuration()
            ctx.obj = config # Store config in context object
            config_loaded_here = True
        else:
            config = ctx.obj # Config already loaded by a parent group
            config_loaded_here = False
            logger.debug("Configuration already loaded in context.")


        # 2. Setup Logging based on verbosity options from context
        # Only set up logging if this group instance loaded the config
        # to avoid redundant setup in nested groups.
        if config_loaded_here:
            verbosity = 0
            # Check params on the current context first
            if ctx.params.get('verbose', 0) > 0:
                verbosity = ctx.params['verbose']
            if ctx.params.get('quiet', False):
                verbosity = -1
            # If not found, check parent context (might be needed if options are on top-level group)
            elif ctx.parent and ctx.parent.params.get('verbose', 0) > 0:
                 verbosity = ctx.parent.params['verbose']
            elif ctx.parent and ctx.parent.params.get('quiet', False):
                 verbosity = -1


            try:
                setup_logging(config, verbosity)
                logger.debug("Config and logging setup complete in ConfigGroup.")
            except Exception as e:
                # Use basic print for critical errors during setup
                print(f"CRITICAL ERROR during logging setup: {e}", file=sys.stderr)
                # Optionally re-raise or exit
                # raise # Or sys.exit(1)
        else:
             logger.debug("Skipping logging setup; already done by parent group.")

        # 3. Proceed with the actual group/command invocation
        try:
            # Call the superclass's invoke method to handle command dispatching
            return super().invoke(ctx)
        except Exception as e:
            # Log unhandled exceptions before exiting
            # Use the logger potentially configured above
            exc_logger = logging.getLogger("sygnals.cli.error")
            exc_logger.critical(f"Unhandled exception during command execution: {e}", exc_info=True)
            # Optionally re-raise or exit with error code
            sys.exit(1) # Exit with non-zero code on error


# --- Common CLI Options ---

# Verbosity options (applied to the main group)
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

# Configuration override options (can be added later)
# config_file_option = click.option(
#     '-c', '--config-file',
#     type=click.Path(exists=True, dir_okay=False, resolve_path=True),
#     multiple=True, # Allow multiple config files
#     help="Path to a Sygnals TOML configuration file (overrides defaults)."
# )
