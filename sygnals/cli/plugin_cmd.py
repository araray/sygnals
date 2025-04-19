# sygnals/cli/plugin_cmd.py

"""
CLI commands for managing Sygnals plugins.
"""

import logging
import click
import subprocess # For running pip commands
import sys # For accessing python executable
import shutil # For copying/removing local plugins
from pathlib import Path
from typing import List, Dict, Any

from rich.table import Table
from rich.console import Console

# Import PluginLoader and PluginRegistry types for context hints
# These will be passed via the Click context object (ctx.obj)
from ..plugins.loader import PluginLoader, PLUGIN_MANIFEST_FILENAME
from ..plugins.api import PluginRegistry
# Import config model to access plugin directory path
from ..config.models import SygnalsConfig

logger = logging.getLogger(__name__)
console = Console()

# --- Helper Functions ---

def _run_pip_command(args: List[str]) -> bool:
    """Runs a pip command using the current Python interpreter."""
    # Construct the command using sys.executable to ensure correct pip version
    command = [sys.executable, "-m", "pip"] + args
    logger.info(f"Running command: {' '.join(command)}")
    try:
        # Use check=True to raise CalledProcessError on failure
        # Capture output to prevent it from cluttering Sygnals output unless debugging
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.debug(f"Pip command stdout:\n{result.stdout}")
        if result.stderr: # Log stderr even on success, as pip might output warnings
             logger.warning(f"Pip command stderr:\n{result.stderr}")
        console.print(f"[green]Pip command successful: {' '.join(args)}[/green]")
        return True
    except FileNotFoundError:
        logger.error(f"Error running pip: '{sys.executable} -m pip' command not found. Is pip installed in the environment?")
        console.print(f"[bold red]Error:[/bold red] 'pip' command not found for the current Python interpreter ({sys.executable}).")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Pip command failed: {' '.join(args)}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Stdout:\n{e.stdout}")
        logger.error(f"Stderr:\n{e.stderr}")
        console.print(f"[bold red]Error:[/bold red] Pip command failed: {' '.join(args)}")
        console.print(f"[red]{e.stderr}[/red]") # Show pip's error output
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred running pip command {' '.join(args)}: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] An unexpected error occurred running pip: {e}")
        return False

def _is_valid_plugin_dir(path: Path) -> bool:
    """Checks if a directory looks like a valid plugin (contains plugin.toml)."""
    return path.is_dir() and (path / PLUGIN_MANIFEST_FILENAME).is_file()

# --- Plugin Command Group ---

@click.group("plugin")
@click.pass_context
def plugin_cmd(ctx):
    """Manage Sygnals plugins."""
    # Ensure plugin loader/registry are available in context
    if not isinstance(ctx.obj, dict) or 'plugin_loader' not in ctx.obj or 'config' not in ctx.obj:
        logger.error("PluginLoader or Config not found in context. Plugin commands require main app setup.")
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
    # Refresh discovery before listing? Optional, might slow down list command.
    # plugin_loader.discover_and_load() # Uncomment if fresh discovery is desired

    plugin_info: List[Dict[str, Any]] = plugin_loader.get_plugin_info()

    if not plugin_info:
        console.print("No plugins discovered.")
        return

    table = Table(title="Discovered Sygnals Plugins", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="dim cyan", width=25)
    table.add_column("Version", width=12)
    table.add_column("Status", width=18) # Wider status for more detail
    table.add_column("Source", width=12)
    table.add_column("API Required", width=18)
    # table.add_column("Description") # Optional: Add description

    status_colors = {
        "loaded": "green",
        "disabled": "yellow",
        "error/incompatible": "red",
        "error/load_failed": "red",
        "error/no_manifest": "red", # Should not happen if listed
        "unknown": "white",
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

# --- Install/Uninstall Commands ---

@plugin_cmd.command("install")
@click.argument("source", type=str)
@click.option('--force', is_flag=True, help='Force reinstall/overwrite if plugin already exists.')
@click.pass_context
def install_plugin(ctx, source: str, force: bool):
    """
    Install a plugin from a local path or PyPI.

    SOURCE can be a local directory path containing a plugin or a package name from PyPI.
    """
    if not isinstance(ctx.obj, dict) or 'config' not in ctx.obj:
        console.print("[bold red]Error:[/bold red] Configuration not loaded correctly.")
        ctx.exit(1)

    config: SygnalsConfig = ctx.obj['config']
    local_plugin_dir = config.paths.plugin_dir

    source_path = Path(source)

    # --- Case 1: Install from Local Directory ---
    if source_path.exists():
        if not source_path.is_dir():
            console.print(f"[bold red]Error:[/bold red] Local source '{source}' exists but is not a directory.")
            ctx.exit(1)
        if not _is_valid_plugin_dir(source_path):
            console.print(f"[bold red]Error:[/bold red] Local source directory '{source}' does not appear to be a valid plugin (missing {PLUGIN_MANIFEST_FILENAME}).")
            ctx.exit(1)

        # Parse manifest to get the plugin name
        manifest_path = source_path / PLUGIN_MANIFEST_FILENAME
        try:
            with open(manifest_path, 'r') as f:
                manifest = toml.load(f)
            plugin_name = manifest.get('name')
            if not plugin_name:
                console.print(f"[bold red]Error:[/bold red] Could not read plugin name from manifest '{manifest_path}'.")
                ctx.exit(1)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] Failed to read plugin manifest '{manifest_path}': {e}")
            ctx.exit(1)

        # Determine destination path in the local plugin directory
        destination_path = local_plugin_dir / plugin_name

        if destination_path.exists():
            if force:
                console.print(f"Plugin '{plugin_name}' already exists locally. Overwriting (--force specified).")
                try:
                    shutil.rmtree(destination_path)
                except OSError as e:
                    console.print(f"[bold red]Error:[/bold red] Failed to remove existing plugin directory '{destination_path}': {e}")
                    ctx.exit(1)
            else:
                console.print(f"[bold red]Error:[/bold red] Plugin '{plugin_name}' already exists in '{local_plugin_dir}'. Use --force to overwrite.")
                ctx.exit(1)

        # Copy the plugin directory
        try:
            shutil.copytree(source_path, destination_path)
            console.print(f"[green]Plugin '{plugin_name}' installed locally from '{source}' to '{destination_path}'.[/green]")
            console.print("Run 'sygnals plugin list' to verify.")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] Failed to copy plugin from '{source}' to '{destination_path}': {e}")
            ctx.exit(1)

    # --- Case 2: Install from PyPI (or potentially other pip sources) ---
    else:
        # Assume 'source' is a package name or other pip-installable target
        console.print(f"Attempting to install plugin '{source}' using pip...")
        pip_args = ["install"]
        if force:
            pip_args.append("--force-reinstall")
        pip_args.append(source)

        if not _run_pip_command(pip_args):
            console.print(f"[bold red]Error:[/bold red] Failed to install plugin '{source}' using pip. See logs for details.")
            ctx.exit(1)
        else:
             console.print(f"[green]Plugin '{source}' installed via pip.[/green]")
             console.print("Run 'sygnals plugin list' to verify (might require restart if entry points changed).")

@plugin_cmd.command("uninstall")
@click.argument("name", type=str)
@click.option('-y', '--yes', is_flag=True, help='Do not ask for confirmation.')
@click.pass_context
def uninstall_plugin(ctx, name: str, yes: bool):
    """
    Uninstall a plugin by its name.

    Handles plugins installed locally or via pip (entry points).
    """
    if not isinstance(ctx.obj, dict) or 'plugin_loader' not in ctx.obj or 'config' not in ctx.obj:
        console.print("[bold red]Error:[/bold red] Plugin system or configuration not initialized correctly.")
        ctx.exit(1)

    plugin_loader: PluginLoader = ctx.obj['plugin_loader']
    config: SygnalsConfig = ctx.obj['config']
    local_plugin_dir = config.paths.plugin_dir

    # Refresh discovery to get latest info before uninstalling
    # This ensures plugin_sources and plugin_manifests are populated
    # Note: This might re-register things, but is needed to find the plugin source.
    # A better approach might be to store discovery info persistently.
    plugin_loader.discover_and_load()

    # Find the plugin's source and manifest
    source_type = plugin_loader.plugin_sources.get(name)
    manifest_tuple = plugin_loader.plugin_manifests.get(name)

    if not source_type or not manifest_tuple:
        console.print(f"[bold red]Error:[/bold red] Plugin '{name}' not found or manifest missing.")
        ctx.exit(1)

    manifest_data, manifest_path = manifest_tuple
    version = manifest_data.get('version', 'N/A')

    # --- Confirm Uninstall ---
    if not yes:
        click.confirm(f"Uninstall plugin '{name}' v{version} (source: {source_type})?", abort=True)

    # --- Perform Uninstall based on source ---
    success = False
    if source_type == 'local':
        plugin_path = local_plugin_dir / name # Assuming dir name matches plugin name
        if plugin_path.exists() and plugin_path.is_dir():
            console.print(f"Removing local plugin directory: {plugin_path}")
            try:
                shutil.rmtree(plugin_path)
                console.print(f"[green]Successfully uninstalled local plugin '{name}'.[/green]")
                success = True
            except OSError as e:
                console.print(f"[bold red]Error:[/bold red] Failed to remove directory '{plugin_path}': {e}")
        else:
            console.print(f"[bold red]Error:[/bold red] Local plugin directory not found for '{name}' at '{plugin_path}'.")

    elif source_type == 'entry_point':
        # Assume package name matches plugin name (this might be fragile)
        package_name = name
        console.print(f"Attempting to uninstall package '{package_name}' using pip...")
        # Add '-y' to pip uninstall to avoid its confirmation prompt
        if _run_pip_command(["uninstall", "-y", package_name]):
            console.print(f"[green]Successfully uninstalled plugin '{name}' (package: {package_name}) via pip.[/green]")
            success = True
        else:
            console.print(f"[bold red]Error:[/bold red] Failed to uninstall plugin '{name}' (package: {package_name}) using pip.")

    else:
        console.print(f"[bold red]Error:[/bold red] Unknown source type '{source_type}' for plugin '{name}'. Cannot uninstall.")

    # --- Clean up State File ---
    if success:
        if name in plugin_loader.plugin_enabled_state:
            del plugin_loader.plugin_enabled_state[name]
            state_file_dir = local_plugin_dir.parent
            _save_plugin_state(state_file_dir, plugin_loader.plugin_enabled_state)
            logger.info(f"Removed state for uninstalled plugin '{name}'.")

    if not success:
        ctx.exit(1)
