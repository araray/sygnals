# tests/test_math_.py

import pytest
from click.testing import CliRunner
from sygnals.cli.main import cli # Updated import path
import pandas as pd
import numpy as np
import os # Import os for path manipulation

# Test the 'math' command (assuming it will be added back later)
@pytest.mark.skip(reason="Math command not yet re-implemented in refactored CLI")
def test_math_command(tmp_path):
    """Test the math command for evaluating expressions."""
    runner = CliRunner()
    out_file = tmp_path / "math_output.csv"
    expression = "np.sin(x * np.pi)" # Use numpy functions available in scope
    x_range = "0,2,0.1" # Range from 0 to 2 with step 0.1

    # Construct the command arguments
    args = [
        "math",
        expression,
        "--x-range", x_range,
        "--output", str(out_file)
    ]

    result = runner.invoke(cli, args)

    # Print output for debugging if the test fails
    if result.exit_code != 0:
        print("CLI Output:\n", result.output)
        print("Exception:\n", result.exception)

    assert result.exit_code == 0, f"CLI command failed with exit code {result.exit_code}"
    assert out_file.exists(), f"Output file {out_file} was not created."

    # Verify the contents of the output file
    try:
        df = pd.read_csv(out_file)
        assert "x" in df.columns
        assert "result" in df.columns
        # Check if the number of rows matches the expected range
        expected_rows = len(np.arange(0, 2, 0.1))
        assert len(df) == expected_rows, f"Expected {expected_rows} rows, found {len(df)}"

        # Optional: Check a specific value
        # Find the row where x is approximately 0.5
        # sin(0.5 * pi) should be 1.0
        val_at_half_pi = df[np.isclose(df['x'], 0.5)]['result'].iloc[0]
        assert np.isclose(val_at_half_pi, 1.0), f"Expected sin(0.5*pi) to be approx 1.0, got {val_at_half_pi}"

    except Exception as e:
        pytest.fail(f"Error reading or verifying the output file {out_file}: {e}")

# Add a test case for potential errors if needed
@pytest.mark.skip(reason="Math command not yet re-implemented in refactored CLI")
def test_math_command_invalid_expr(tmp_path):
     """Test the math command with an invalid expression."""
     runner = CliRunner()
     out_file = tmp_path / "math_error.csv"
     expression = "invalid function(x)"
     x_range = "0,1,0.1"

     args = [
         "math",
         expression,
         "--x-range", x_range,
         "--output", str(out_file)
     ]

     result = runner.invoke(cli, args)
     # Expecting an error exit code
     assert result.exit_code != 0, "Command should fail with an invalid expression."
     # Check if error message is logged or printed (depending on implementation)
     # This might require inspecting stderr or log files in a more complex setup
     assert "error" in result.output.lower() or result.exception is not None
     assert not out_file.exists(), "Output file should not be created on error."
