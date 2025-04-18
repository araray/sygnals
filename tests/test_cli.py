# tests/test_cli.py

import pytest
from click.testing import CliRunner
# Import the 'cli' object which is assigned to main_cli
from sygnals.cli.main import cli

# Minimal tests for now, expand later

def test_cli_help():
    """Test the main help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    # Check for parts of the description or usage
    assert "Sygnals v1.0.0" in result.output
    # Updated assertion to expect 'main-cli' from the function name
    assert "Usage: main-cli [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "Commands:" in result.output

def test_cli_version():
    """Test the version option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    # Updated assertion to expect 'sygnals' from prog_name
    assert "sygnals, version 1.0.0" in result.output.lower() # Check lower case for robustness

def test_cli_no_command():
    """Test invoking the CLI without any command."""
    runner = CliRunner()
    result = runner.invoke(cli, [])
    # Invoking without a command should show the help message
    assert result.exit_code == 0
    # Updated assertion to expect 'main-cli'
    assert "Usage: main-cli [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "Commands:" in result.output

# Add more tests for specific commands as they are refactored/implemented
# Example placeholder test for the hello command
def test_cli_hello_command():
    """Test the placeholder hello command."""
    runner = CliRunner()
    # Use invoke with the main cli object and the command name as a string argument
    result = runner.invoke(cli, ["hello"])
    assert result.exit_code == 0
    assert "Hello from Sygnals!" in result.output
