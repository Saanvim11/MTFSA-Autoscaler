# app.py — MTFSA LIVE DASHBOARD
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from pathlib import Path
import time
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
BASE_DIR = Path(r"C:\Users\SAANVI\Downloads\cloud_autoscaler_project")
MODEL_PATH = BASE_DIR / "models" / "hybrid_autoscaler.keras"
DATA_FILE = BASE_DIR / "data" / "processed" / "r1_d0_to_d5_hourly.parquet"

st.set_page_config(page_title="MTFSA Live", layout="wide")
st.title("MTFSA: Hybrid Few-Shot Autoscaler (LIVE)")
st.markdown("**TL + FSL + Tinch of ML** | RMSE = 0.013 | R² = 0.999")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load data
@st.cache_data
def load_data():
    return pd.read_parquet(DATA_FILE)

hourly = load_data()

# ---------- SIDEBAR ----------
st.sidebar.header("Controls")
selected_func = st.sidebar.selectbox("Select Function", hourly['funcName'].unique())
lookback = st.sidebar.slider("Lookback Hours", 1, 6, 3)
refresh = st.sidebar.button("Refresh Prediction")

# ---------- LIVE PREDICTION ----------
def predict_cold_starts(func_name, lookback=3):
    data = hourly[hourly['funcName'] == func_name].sort_values('hour')
    if len(data) < lookback + 1:
        return None, None, None
    
    recent = data.tail(lookback + 1)
    series = recent['cold_count'].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    
    X = scaled[:-1].reshape(1, lookback, 1)
    y_actual = scaled[-1]
    
    pred_scaled = model.predict(X, verbose=0)[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]
    actual = scaler.inverse_transform([[y_actual]])[0][0]
    mae = abs(pred - actual)
    
    return pred, actual, mae

# ---------- MAIN ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Function: `{selected_func}`")
    func_data = hourly[hourly['funcName'] == selected_func].tail(20)
    
    if len(func_data) >= lookback + 1:
        pred, actual, mae = predict_cold_starts(selected_func, lookback)
        
        st.metric("Predicted Cold Starts", f"{pred:.3f}" if pred else "—")
        st.metric("Actual Last Hour", f"{actual:.1f}" if actual else "—")
        st.metric("MAE", f"{mae:.3f}" if mae else "—")
        
        if pred and pred > 1.5:
            st.error(f"SCALE UP! Pre-warm {int(pred)} pods NOW")
        else:
            st.success("No scaling needed")
    else:
        st.warning("Not enough data")

with col2:
    st.subheader("Live History")
    fig, ax = plt.subplots()
    plot_data = hourly[hourly['funcName'] == selected_func].tail(10)
    ax.plot(plot_data['hour'], plot_data['cold_count'], 'o-', label='Actual', color='blue')
    
    if len(plot_data) >= lookback + 1:
        pred, _, _ = predict_cold_starts(selected_func, lookback)
        if pred:
            ax.plot(plot_data['hour'].iloc[-1], pred, 'go', markersize=12, label=f"Pred: {pred:.2f}")
    
    ax.set_title("Cold Start History")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Cold Starts")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ---------- AUTO REFRESH ----------
if refresh:
    st.success("Prediction refreshed!")
    time.sleep(1)
    st.rerun()

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("**MTFSA v4** — Transfer + Few-Shot + 1-Step Meta-Init")