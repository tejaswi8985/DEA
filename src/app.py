import streamlit as st
import plotly.express as px

from preprocessing import *
from analysis import *
from model import *

st.set_page_config(page_title="Traffic Dashboard", layout="wide")

st.title("🚦 Smart Traffic Congestion Dashboard")

# Load Data
df = load_data("data/traffic_data.csv")
df = clean_data(df)
df = feature_engineering(df)

# ===========================
# 🔍 FILTERS
# ===========================
st.sidebar.header("🔍 Filters")

zone_filter = st.sidebar.multiselect(
    "Select Zone",
    options=df['zone'].unique(),
    default=df['zone'].unique()
)

df = df[df['zone'].isin(zone_filter)]

# ===========================
# 📊 METRICS
# ===========================
col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Avg Traffic", int(df['vehicles'].mean()))
col3.metric("Avg Congestion", round(df['congestion_index'].mean(), 2))

# ===========================
# 📈 CHARTS
# ===========================
st.subheader("⏰ Traffic by Hour")

hourly = df.groupby('hour')['vehicles'].mean().reset_index()
fig1 = px.line(hourly, x='hour', y='vehicles', markers=True)
st.plotly_chart(fig1, use_container_width=True)

# ===========================
st.subheader("🌍 Zone Congestion")

zone = df.groupby('zone')['congestion_index'].mean().reset_index()
fig2 = px.bar(zone, x='zone', y='congestion_index', color='zone')
st.plotly_chart(fig2, use_container_width=True)

# ===========================
st.subheader("🔥 Congestion Heatmap")

pivot = df.pivot_table(
    values='congestion_index',
    index='hour',
    columns='weekday'
)

fig3 = px.imshow(pivot, aspect="auto", color_continuous_scale="reds")
st.plotly_chart(fig3, use_container_width=True)

# ===========================
# 🤖 MODEL SECTION
# ===========================
st.subheader("🤖 ML Prediction")

model_choice = st.selectbox(
    "Choose Model",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

hour = st.slider("Hour", 0, 23)
weekday = st.slider("Weekday", 0, 6)

if model_choice == "Linear Regression":
    model = linear_model(df)
elif model_choice == "Random Forest":
    model = random_forest_model(df)
else:
    model = xgboost_model(df)

prediction = model.predict([[hour, weekday]])
st.success(f"🚗 Predicted Congestion Index: {prediction[0]:.2f}")

# ===========================
# 📊 MODEL COMPARISON
# ===========================
st.subheader("📊 Model Performance")

results = compare_models(df)

fig4 = px.bar(
    x=list(results.keys()),
    y=list(results.values()),
    labels={'x': 'Model', 'y': 'MSE'},
    color=list(results.keys())
)

st.plotly_chart(fig4, use_container_width=True)

# ===========================
# 📍 CLUSTERING
# ===========================
st.subheader("📍 Traffic Clusters")

clustered_df, _ = kmeans_model(df)

fig5 = px.scatter(
    clustered_df,
    x='vehicles',
    y='congestion_index',
    color='cluster'
)

st.plotly_chart(fig5, use_container_width=True)

# ===========================
# 📈 ARIMA
# ===========================
st.subheader("📈 Forecast (ARIMA)")

forecast = arima_model(df)
st.write(forecast)

# ===========================
st.markdown("---")
st.caption("🚀 Built by Tejaswi | Smart Traffic Analytics Project")