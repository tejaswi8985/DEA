import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb


# ===============================
# 1. LINEAR REGRESSION
# ===============================
def linear_model(df):
    X = df[['hour', 'weekday']]
    y = df['congestion_index']

    model = LinearRegression()
    model.fit(X, y)

    return model


# ===============================
# 2. RANDOM FOREST
# ===============================
def random_forest_model(df):
    X = df[['hour', 'weekday']]
    y = df['congestion_index']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    return model


# ===============================
# 3. XGBOOST
# ===============================
def xgboost_model(df):
    X = df[['hour', 'weekday']]
    y = df['congestion_index']

    model = xgb.XGBRegressor()
    model.fit(X, y)

    return model


# ===============================
# 4. K-MEANS (CLUSTERING)
# ===============================
def kmeans_model(df):
    X = df[['vehicles', 'congestion_index']]

    model = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = model.fit_predict(X)

    return df, model


# ===============================
# 5. ARIMA (TIME SERIES)
# ===============================
def arima_model(df):
    ts = df.set_index('timestamp')['congestion_index']

    model = ARIMA(ts, order=(1,1,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=5)

    return forecast


# ===============================
# 6. MODEL COMPARISON
# ===============================
def compare_models(df):
    X = df[['hour', 'weekday']]
    y = df['congestion_index']

    results = {}

    # Linear
    lr = LinearRegression().fit(X, y)
    results['Linear'] = mean_squared_error(y, lr.predict(X))

    # Random Forest
    rf = RandomForestRegressor().fit(X, y)
    results['RandomForest'] = mean_squared_error(y, rf.predict(X))

    # XGBoost
    xg_model = xgb.XGBRegressor().fit(X, y)
    results['XGBoost'] = mean_squared_error(y, xg_model.predict(X))

    return results