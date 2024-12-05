import pandas as pd
import pandasql as ps
import os
import json

# Supported formats
SUPPORTED_FORMATS = ['csv', 'json']

def read_data(file_path):
    """Load data from a file (CSV or JSON) into a Pandas DataFrame."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def save_data(data, output_path):
    """Save a Pandas DataFrame to a CSV or JSON file."""
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()

    if ext == '.csv':
        data.to_csv(output_path, index=False)
    elif ext == '.json':
        data.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported output file format: {ext}")

def run_sql_query(data, query):
    """Execute an SQL query on a Pandas DataFrame."""
    return ps.sqldf(query, locals())

def filter_data(data, filter_expr):
    """Filter data using a Pandas-style filter expression."""
    return data.query(filter_expr)

def normalize(data):
    """Normalize the data in a DataFrame."""
    return (data - data.min()) / (data.max() - data.min())
