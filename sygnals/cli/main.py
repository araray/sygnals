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

@click.group(context_settings=CONTEXT_SETTINGS, cls=ConfigCommand)
@click.version_option(__version__, '-V', '--version', package_name='sygnals')
@verbose_option
@quiet_option
@click.pass_context # Pass context to access config object (ctx.obj)
def cli(ctx, verbose: int, quiet: bool):
    """
    Sygnals v1.0.0: A versatile CLI for signal and audio processing,
    tailored for data science workflows.

    Configuration is loaded from:
    Defaults -> ./sygnals.toml -> ~/.config/sygnals/sygnals.toml -> Env Vars

    Use -v for verbose output, -vv for debug output, -q for quiet mode.
    """
    # The ConfigCommand already loaded the config into ctx.obj
    # and set up logging based on verbose/quiet flags.
    # We can access the config here if needed for the top-level group,
    # but usually it's more relevant within subcommands.
    config = ctx.obj
    logger.debug(f"Sygnals CLI started with verbosity={verbose}, quiet={quiet}")
    logger.debug(f"Loaded config ID: {id(config)}") # Verify config object is passed

# --- Register Commands ---
# Add commands/groups as they are refactored or created
# cli.add_command(analyze)
# cli.add_command(transform)
# cli.add_command(filter_cmd, name="filter") # Use name='filter'
# cli.add_command(audio)
# cli.add_command(visualize)
# cli.add_command(plugin)
# ... add other commands (manipulate, batch, math, etc.)

# Example placeholder command
@cli.command()
@click.pass_context
def hello(ctx):
    """A simple example command."""
    config = ctx.obj # Access config from context
    logger.info("Hello command executing.")
    logger.debug(f"Default sample rate from config: {config.defaults.default_sample_rate}")
    click.echo("Hello from Sygnals!")

if __name__ == "__main__":
    cli()
