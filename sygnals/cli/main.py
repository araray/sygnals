# sygnals/cli/main.py

"""
Main entry point for the Sygnals CLI application.
Uses Click for command-line interface handling.
"""

import click
import logging

from sygnals.version import __version__
from .base_cmd import ConfigCommand, verbose_option, quiet_option
# Import command groups/functions from other files as they are created/refactored
# from .analyze_cmd import analyze
# from .transform_cmd import transform
# from .filter_cmd import filter_cmd # Renamed to avoid conflict
# from .audio_cmd import audio
# from .visualize_cmd import visualize
# from .plugin_cmd import plugin
# ... other commands

logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

# --- Define the main group object first ---
# Instantiate the group, assigning the custom command class
cli = click.Group(
    name='sygnals-cli', # Explicitly name the group object if needed, separate from function name
    context_settings=CONTEXT_SETTINGS,
    cls=ConfigCommand, # Use the custom class for config/logging setup
    help="""
    Sygnals v1.0.0: A versatile CLI for signal and audio processing,
    tailored for data science workflows.

    Configuration is loaded from:
    Defaults -> ./sygnals.toml -> ~/.config/sygnals/sygnals.toml -> Env Vars

    Use -v for verbose output, -vv for debug output, -q for quiet mode.
    """
)

# --- Define the function that runs when the group is invoked ---
# Decorate this function with the group object itself (@cli)
# Add options like version, verbosity to this function
@cli # Use the group object as the decorator
@click.version_option(__version__, '-V', '--version', package_name='sygnals', prog_name='sygnals') # Add prog_name
@verbose_option
@quiet_option
@click.pass_context # Pass context to access config object (ctx.obj)
def cli_entry_point(ctx, verbose: int, quiet: bool):
    """
    This function is executed when the main 'sygnals' command is run.
    The actual logic for the group (like config loading) happens in ConfigCommand.
    """
    # The ConfigCommand already loaded the config into ctx.obj
    # and set up logging based on verbose/quiet flags.
    config = ctx.obj
    logger.debug(f"Sygnals CLI group invoked with verbosity={verbose}, quiet={quiet}")
    logger.debug(f"Loaded config ID: {id(config)}") # Verify config object is passed


# --- Register Commands ---
# Now add commands using the group object's command decorator
@cli.command() # Use @cli.command() here
@click.pass_context
def hello(ctx):
    """A simple example command."""
    config = ctx.obj # Access config from context
    logger.info("Hello command executing.")
    logger.debug(f"Default sample rate from config: {config.defaults.default_sample_rate}")
    click.echo("Hello from Sygnals!")

# Add other commands/groups as they are refactored or created
# cli.add_command(analyze)
# cli.add_command(transform)
# cli.add_command(filter_cmd, name="filter") # Use name='filter'
# cli.add_command(audio)
# cli.add_command(visualize)
# cli.add_command(plugin)
# ... add other commands (manipulate, batch, math, etc.)


if __name__ == "__main__":
    # This allows running the CLI directly via `python -m sygnals.cli.main`
    # The entry point script defined in pyproject.toml handles `sygnals` command
    cli()
