import pytest
import pandas as pd
from sygnals.core.storage import save_to_database, query_database

def test_database_storage(tmp_path):
    db = tmp_path / "test.db"
    df = pd.DataFrame({"time":[0,1],"value":[10,20]})
    save_to_database(df, str(db), "mytable")
    res = query_database(str(db), "SELECT * FROM mytable")
    assert res.equals(df)
