import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def clean_data(df):
    df = df.dropna()
    return df

def feature_engineering(df):
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['congestion_index'] = df['vehicles'] / df['road_capacity']
    return df
