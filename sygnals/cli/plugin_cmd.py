# sygnals/cli/plugin_cmd.py

"""
CLI commands for managing Sygnals plugins.
"""

import logging
import click
from pathlib import Path
from typing import List, Dict, Any

from rich.table import Table
from rich.console import Console

# Import PluginLoader and PluginRegistry types for context hints
# These will be passed via the Click context object (ctx.obj)
from ..plugins.loader import PluginLoader
from ..plugins.api import PluginRegistry

logger = logging.getLogger(__name__)
console = Console()

# --- Plugin Command Group ---

@click.group("plugin")
@click.pass_context
def plugin_cmd(ctx):
    """Manage Sygnals plugins."""
    # Ensure plugin loader/registry are available in context
    if not isinstance(ctx.obj, dict) or 'plugin_loader' not in ctx.obj:
        logger.error("PluginLoader not found in context. Plugin commands require main app setup.")
        # Fallback or raise error? For now, log and potentially fail in subcommands.
        pass

# --- Subcommands ---

@plugin_cmd.command("list")
@click.pass_context
def list_plugins(ctx):
    """List all discovered plugins and their status."""
    if not isinstance(ctx.obj, dict) or 'plugin_loader' not in ctx.obj:
        console.print("[bold red]Error:[/bold red] Plugin system not initialized correctly.")
        ctx.exit(1)

    plugin_loader: PluginLoader = ctx.obj['plugin_loader']
    plugin_info: List[Dict[str, Any]] = plugin_loader.get_plugin_info()

    if not plugin_info:
        console.print("No plugins discovered.")
        return

    table = Table(title="Discovered Sygnals Plugins", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="dim cyan", width=25)
    table.add_column("Version", width=12)
    table.add_column("Status", width=15)
    table.add_column("Source", width=12)
    table.add_column("API Required", width=18)
    # table.add_column("Description") # Optional: Add description

    status_colors = {
        "loaded": "green",
        "disabled": "yellow",
        "error/incompatible": "red",
    }

    for info in plugin_info:
        status = info.get('status', 'unknown')
        color = status_colors.get(status, "white")
        table.add_row(
            info.get('name', 'N/A'),
            info.get('version', 'N/A'),
            f"[{color}]{status}[/{color}]",
            info.get('source', 'N/A'),
            info.get('api_required', 'N/A'),
            # info.get('description', '')
        )

    console.print(table)

@plugin_cmd.command("enable")
@click.argument("name", type=str)
@click.pass_context
def enable_plugin(ctx, name: str):
    """Enable a specific plugin for the next run."""
    if not isinstance(ctx.obj, dict) or 'plugin_loader' not in ctx.obj:
        console.print("[bold red]Error:[/bold red] Plugin system not initialized correctly.")
        ctx.exit(1)

    plugin_loader: PluginLoader = ctx.obj['plugin_loader']

    if plugin_loader.enable_plugin(name):
        console.print(f"Plugin '{name}' marked as [bold green]enabled[/bold green]. Changes will apply on the next run.")
    else:
        # Error message already logged by enable_plugin
        console.print(f"[bold red]Error:[/bold red] Could not enable plugin '{name}'. Check logs for details.")
        ctx.exit(1)


@plugin_cmd.command("disable")
@click.argument("name", type=str)
@click.pass_context
def disable_plugin(ctx, name: str):
    """Disable a specific plugin for the next run."""
    if not isinstance(ctx.obj, dict) or 'plugin_loader' not in ctx.obj:
        console.print("[bold red]Error:[/bold red] Plugin system not initialized correctly.")
        ctx.exit(1)

    plugin_loader: PluginLoader = ctx.obj['plugin_loader']

    if plugin_loader.disable_plugin(name):
        console.print(f"Plugin '{name}' marked as [bold yellow]disabled[/bold yellow]. Changes will apply on the next run.")
    else:
        # Error message already logged by disable_plugin
        console.print(f"[bold red]Error:[/bold red] Could not disable plugin '{name}'. Check logs for details.")
        ctx.exit(1)


@plugin_cmd.command("scaffold")
@click.argument("name", type=str)
@click.option('--dest', default='.', help='Destination directory for the plugin template.', type=click.Path(file_okay=False, writable=True, resolve_path=True))
@click.pass_context
def scaffold_plugin(ctx, name: str, dest: str):
    """Generate a basic plugin template directory."""
    # Logic will be implemented in sygnals.plugins.scaffold
    from ..plugins.scaffold import create_plugin_scaffold # Import here to avoid circular dependency issues

    destination_path = Path(dest)
    try:
        create_plugin_scaffold(name, destination_path)
        console.print(f"Plugin template for '{name}' created successfully in '{destination_path / name}'.")
        console.print("Remember to:")
        console.print(f"  1. Implement your plugin logic in '{destination_path / name / name.replace('-', '_') / 'plugin.py'}'.")
        console.print(f"  2. Update '{destination_path / name / 'plugin.toml'}'.")
        console.print(f"  3. Update '{destination_path / name / 'pyproject.toml'}' (especially dependencies).")
        console.print(f"  4. Install the plugin (e.g., using 'pip install -e {destination_path / name}') or copy it to the local plugin directory.")
    except Exception as e:
        logger.error(f"Failed to create plugin scaffold: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Could not create plugin scaffold for '{name}': {e}")
        ctx.exit(1)

# --- TODO: Add install/uninstall commands (Phase 3+) ---
# These would require interacting with pip or copying files, adding complexity.
# @plugin_cmd.command("install")
# @click.argument("source", type=str)
# def install_plugin(source: str): ...

# @plugin_cmd.command("uninstall")
# @click.argument("name", type=str)
# def uninstall_plugin(name: str): ...
