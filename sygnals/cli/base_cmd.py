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

class ConfigCommand(click.Command):
    """
    A custom Click command that loads configuration and sets up logging.
    Passes the config object via the context.
    """
    def invoke(self, ctx: click.Context):
        # 1. Load Configuration
        # Allow overriding config file via CLI option in the future if needed
        config = load_configuration()
        ctx.obj = config # Store config in context object

        # 2. Setup Logging based on verbosity options from context
        verbosity = 0
        if ctx.params.get('verbose', 0) > 0:
            verbosity = ctx.params['verbose']
        if ctx.params.get('quiet', False):
            verbosity = -1

        try:
            setup_logging(config, verbosity)
            logger.debug("Config and logging setup complete in ConfigCommand.")
        except Exception as e:
            # Use basic print for critical errors during setup
            print(f"CRITICAL ERROR during logging setup: {e}", file=sys.stderr)
            # Optionally re-raise or exit
            # raise # Or sys.exit(1)

        # 3. Proceed with the actual command invocation
        try:
            return super().invoke(ctx)
        except Exception as e:
            # Log unhandled exceptions before exiting
            logger.critical(f"Unhandled exception in command execution: {e}", exc_info=True)
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
