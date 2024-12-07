import sqlite3

import pandas as pd


def save_to_database(data, db_path, table_name):
    """Save a Pandas DataFrame to an SQLite database."""
    conn = sqlite3.connect(db_path)
    data.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


def query_database(db_path, query):
    """Execute an SQL query on the SQLite database."""
    conn = sqlite3.connect(db_path)
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result
