import pytest
import pandas as pd
import numpy as np
from sygnals.core.batch_processor import process_batch
from sygnals.core.data_handler import read_data

def test_process_batch(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    # Create test data files
    df = pd.DataFrame({"value": np.sin(np.linspace(0,2*np.pi,1000))})
    file_1 = input_dir / "data1.csv"
    file_2 = input_dir / "data2.csv"
    df.to_csv(file_1, index=False)
    df.to_csv(file_2, index=False)

    process_batch(str(input_dir), str(output_dir), "fft")
    # Check output files
    out_files = list(output_dir.glob("*_processed.csv"))
    assert len(out_files) == 2
