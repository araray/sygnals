# sygnals/cli/main.py

"""
Main entry point for the Sygnals CLI application.
Uses Click for command-line interface handling.
"""

import click
import logging

from sygnals.version import __version__
# Import the renamed ConfigGroup class
from .base_cmd import ConfigGroup, verbose_option, quiet_option
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

# --- Define the main group function using the decorator ---
# Use the decorator to associate the custom ConfigGroup class
@click.group(context_settings=CONTEXT_SETTINGS, cls=ConfigGroup) # Use ConfigGroup here
@click.version_option(__version__, '-V', '--version', package_name='sygnals', prog_name='sygnals')
@verbose_option
@quiet_option
@click.pass_context # Pass context to access config object (ctx.obj)
def main_cli(ctx, verbose: int, quiet: bool):
    """
    Sygnals v1.0.0: A versatile CLI for signal and audio processing,
    tailored for data science workflows.

    Configuration is loaded from:
    Defaults -> ./sygnals.toml -> ~/.config/sygnals/sygnals.toml -> Env Vars

    Use -v for verbose output, -vv for debug output, -q for quiet mode.
    """
    # This function is executed when the main 'sygnals' command is run.
    # The actual logic for the group (like config loading) happens in ConfigGroup's invoke method.
    config = ctx.obj # Config should be loaded into ctx.obj by ConfigGroup.invoke
    if isinstance(config, SygnalsConfig): # Check if config loading was successful
        logger.debug(f"Sygnals CLI group invoked with verbosity={verbose}, quiet={quiet}")
        logger.debug(f"Loaded config ID: {id(config)}") # Verify config object is passed
    else:
        # Log basic message if config object isn't the expected type (might indicate setup error)
        logger.error("SygnalsConfig object not found in context. Setup might have failed.")


# --- Register Commands ---
# Now add commands using the main group function's name
@main_cli.command() # Use @main_cli.command() here
@click.pass_context
def hello(ctx):
    """A simple example command."""
    config = ctx.obj # Access config from context
    if isinstance(config, SygnalsConfig):
        logger.info("Hello command executing.")
        logger.debug(f"Default sample rate from config: {config.defaults.default_sample_rate}")
        click.echo("Hello from Sygnals!")
    else:
         click.echo("Error: Configuration not loaded correctly.", err=True)


# Add other commands/groups as they are refactored or created
# main_cli.add_command(analyze)
# main_cli.add_command(transform)
# main_cli.add_command(filter_cmd, name="filter") # Use name='filter'
# main_cli.add_command(audio)
# main_cli.add_command(visualize)
# main_cli.add_command(plugin)
# ... add other commands (manipulate, batch, math, etc.)


# Define the entry point for the script execution (e.g., for pyproject.toml)
# This should refer to the group object created by the decorator.
cli = main_cli


if __name__ == "__main__":
    # This allows running the CLI directly via `python -m sygnals.cli.main`
    cli()
