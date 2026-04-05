import pandas as pd

def analyze_dataset(df):

    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().sum()
    }

    return info