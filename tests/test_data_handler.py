import pytest
import pandas as pd
from sygnals.core.data_handler import read_data, save_data, filter_data, run_sql_query
import json

def test_read_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("time,value\n0,1.0\n1,2.0\n2,3.0\n")
    df = read_data(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df["value"].iloc[0] == 1.0

def test_read_json(tmp_path):
    json_file = tmp_path / "test.json"
    data = [{"time":0,"value":1.0},{"time":1,"value":2.0}]
    json_file.write_text(json.dumps(data))
    df = read_data(str(json_file))
    assert len(df) == 2
    assert df["value"].iloc[1] == 2.0

def test_save_data_csv(tmp_path):
    df = pd.DataFrame({"time":[0,1,2],"value":[1.0,2.0,3.0]})
    out_file = tmp_path / "out.csv"
    save_data(df, str(out_file))
    assert out_file.exists()
    loaded = pd.read_csv(out_file)
    assert loaded.equals(df)

def test_filter_data():
    df = pd.DataFrame({"time": [0,1,2,3], "value": [10,20,5,25]})
    result = filter_data(df, "value > 10")
    assert len(result) == 2
    assert set(result["value"]) == {20,25}

def test_run_sql_query():
    df = pd.DataFrame({"time":[0,1,2],"value":[10,20,30]})
    query = "SELECT * FROM df WHERE value > 15"
    res = run_sql_query(df, query)
    assert len(res) == 2
    assert set(res["value"]) == {20,30}
