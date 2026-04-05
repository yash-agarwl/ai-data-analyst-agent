import pandas as pd

def load_data(file):
    df = pd.read_csv(file)
    return df

def clean_data(df):
    df = df.dropna()
    return df