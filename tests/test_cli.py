import pytest
from click.testing import CliRunner
from sygnals.cli import cli

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Sygnals: A versatile CLI for signal and audio processing." in result.output

def test_cli_no_command():
    runner = CliRunner()
    result = runner.invoke(cli, [])
    # Should display help
    assert result.exit_code == 0
    assert "Commands:" in result.output
