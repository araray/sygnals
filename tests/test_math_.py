import pytest
from click.testing import CliRunner
from sygnals.cli import cli
import pandas as pd
import numpy as np

def test_math_command(tmp_path):
    runner = CliRunner()
    out_file = tmp_path / "math_output.csv"
    result = runner.invoke(cli, [
        "math",
        "np.sin(x)",
        "--x-range","0,6.283185,0.1",
        "--output", str(out_file)
    ])
    assert result.exit_code == 0
    df = pd.read_csv(out_file)
    # Check we have values of sin from 0 to ~2π
    assert len(df) > 50
    # Optional check: sin(0)=0
    zero_val = df[df['x']==0]['result'].iloc[0]
    assert abs(zero_val) < 1e-7
