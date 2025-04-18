# tests/test_cli.py

import pytest
from click.testing import CliRunner
from sygnals.cli.main import cli # Updated import path

# Minimal tests for now, expand later

def test_cli_help():
    """Test the main help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    # Check for parts of the description or usage
    assert "Sygnals v1.0.0" in result.output
    assert "Usage: cli [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "Commands:" in result.output

def test_cli_version():
    """Test the version option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    # Should output something like: cli, version 1.0.0
    assert "cli, version 1.0.0" in result.output.lower() # Check lower case for robustness

def test_cli_no_command():
    """Test invoking the CLI without any command."""
    runner = CliRunner()
    result = runner.invoke(cli, [])
    # Invoking without a command should show the help message
    assert result.exit_code == 0
    assert "Usage: cli [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "Commands:" in result.output

# Add more tests for specific commands as they are refactored/implemented
# Example:
# def test_cli_hello_command():
#     """Test the placeholder hello command."""
#     runner = CliRunner()
#     result = runner.invoke(cli, ["hello"])
#     assert result.exit_code == 0
#     assert "Hello from Sygnals!" in result.output
