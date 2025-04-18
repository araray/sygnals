# tests/test_storage.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path # Use pathlib
import sqlite3 # Import sqlite3 to check for errors

# Import the functions to test
from sygnals.core.storage import save_to_database, query_database

# --- Test Fixture ---

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Creates a sample Pandas DataFrame for testing."""
    return pd.DataFrame({
        "time": np.linspace(0, 1, 5),
        "value": np.random.rand(5) * 10,
        "label": ['A', 'B', 'A', 'B', 'A']
    })

# --- Test Cases ---

def test_save_and_query_database(tmp_path: Path, sample_dataframe):
    """Test saving a DataFrame to a database and querying it back."""
    db_path = tmp_path / "test_storage.db"
    table_name = "signal_data"
    df_original = sample_dataframe

    # Save the DataFrame to the database
    save_to_database(df_original, str(db_path), table_name)

    # Check if the database file was created
    assert db_path.exists()
    assert db_path.stat().st_size > 0

    # Query all data back from the table
    df_queried = query_database(str(db_path), f"SELECT * FROM {table_name}")

    # Verify the queried data matches the original DataFrame
    # Use pandas testing function for robust comparison
    pd.testing.assert_frame_equal(df_original, df_queried, check_dtype=False) # Allow dtype differences (e.g., int vs float)

def test_save_database_replace(tmp_path: Path, sample_dataframe):
    """Test the if_exists='replace' behavior of save_to_database."""
    db_path = tmp_path / "test_replace.db"
    table_name = "replace_test"
    df1 = sample_dataframe
    # Create a second DataFrame with different data but same columns
    df2 = pd.DataFrame({
        "time": [10.0, 11.0],
        "value": [100.0, 200.0],
        "label": ['X', 'Y']
    })

    # Save the first DataFrame
    save_to_database(df1, str(db_path), table_name)
    queried1 = query_database(str(db_path), f"SELECT * FROM {table_name}")
    assert len(queried1) == len(df1)

    # Save the second DataFrame to the same table (should replace)
    save_to_database(df2, str(db_path), table_name)
    queried2 = query_database(str(db_path), f"SELECT * FROM {table_name}")

    # Check that the table now contains data from the second DataFrame
    assert len(queried2) == len(df2)
    pd.testing.assert_frame_equal(df2, queried2, check_dtype=False)

def test_query_database_specific_cols(tmp_path: Path, sample_dataframe):
    """Test querying specific columns."""
    db_path = tmp_path / "test_query_cols.db"
    table_name = "specific_cols"
    df_original = sample_dataframe
    save_to_database(df_original, str(db_path), table_name)

    # Query only 'time' and 'label' columns
    query = f"SELECT time, label FROM {table_name} WHERE label = 'A'"
    df_queried = query_database(str(db_path), query)

    assert list(df_queried.columns) == ["time", "label"]
    assert len(df_queried) == df_original[df_original['label'] == 'A'].shape[0]
    assert all(df_queried['label'] == 'A')

def test_query_database_with_ordering(tmp_path: Path, sample_dataframe):
    """Test querying with an ORDER BY clause."""
    db_path = tmp_path / "test_query_order.db"
    table_name = "ordered_data"
    df_original = sample_dataframe
    save_to_database(df_original, str(db_path), table_name)

    # Query ordered by value descending
    query = f"SELECT value FROM {table_name} ORDER BY value DESC"
    df_queried = query_database(str(db_path), query)

    # Verify the order
    expected_order = df_original['value'].sort_values(ascending=False).tolist()
    actual_order = df_queried['value'].tolist()
    assert actual_order == expected_order

def test_query_database_invalid_query(tmp_path: Path, sample_dataframe):
    """Test executing an invalid SQL query."""
    db_path = tmp_path / "test_invalid_query.db"
    table_name = "test_table"
    save_to_database(sample_dataframe, str(db_path), table_name)

    invalid_query = "SELECT non_existent_column FROM non_existent_table"
    # Expect pandasql/sqlite3 to raise an operational error
    with pytest.raises(sqlite3.OperationalError): # Or potentially pd.io.sql.DatabaseError
        query_database(str(db_path), invalid_query)

def test_query_database_empty_table(tmp_path: Path):
    """Test querying an empty table."""
    db_path = tmp_path / "test_empty.db"
    table_name = "empty_table"
    # Create an empty DataFrame with columns
    df_empty = pd.DataFrame({'colA': pd.Series(dtype='int'), 'colB': pd.Series(dtype='str')})
    save_to_database(df_empty, str(db_path), table_name)

    # Query the empty table
    df_queried = query_database(str(db_path), f"SELECT * FROM {table_name}")

    assert df_queried.empty
    # Check columns are preserved even if empty
    pd.testing.assert_index_equal(df_empty.columns, df_queried.columns)
