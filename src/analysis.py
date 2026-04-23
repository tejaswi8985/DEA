def peak_analysis(df):
    return df.groupby('hour')['vehicles'].mean()

def zone_analysis(df):
    return df.groupby('zone')['congestion_index'].mean()
