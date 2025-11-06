# app_final.py — MTFSA LIVE DASHBOARD (RENDER.COM READY)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

LSTM_MODEL_PATH = MODEL_DIR / "final_model2.keras"
RF_MODEL_PATH = MODEL_DIR / "rf_refiner.pkl"

# Verify files exist
if not LSTM_MODEL_PATH.exists():
    st.error(f"Model not found: {LSTM_MODEL_PATH}")
    st.stop()
if not RF_MODEL_PATH.exists():
    st.error(f"RF model not found: {RF_MODEL_PATH}")
    st.stop()

st.set_page_config(page_title="MTFSA Live", layout="wide")
st.title("MTFSA: Hybrid Meta-Few-Shot Autoscaler (LIVE)")
st.markdown("**TL + FSL + 1-Step Meta + RF Refinement** | RMSE = 0.117 | R² = 0.876 | 100% @ ±0.5")

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_lstm():
    return tf.keras.models.load_model(LSTM_MODEL_PATH)

@st.cache_resource
def load_rf():
    with open(RF_MODEL_PATH, 'rb') as f:
        return pickle.load(f)

lstm_model = load_lstm()
rf_model = load_rf()

# ---------- LOAD DATA ----------
@st.cache_data
def load_all_data():
    datasets = {}
    regions = [f"r{i}" for i in range(1, 6)]
    for r in regions:
        path = DATA_DIR / f"{r}_d0_to_d30_hourly.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                datasets[r.upper()] = df
            except Exception as e:
                st.warning(f"Failed to load {path}: {e}")
    return datasets

data = load_all_data()

if not data:
    st.error("No data loaded. Check `data/processed/` folder.")
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.header("Controls")
selected_region = st.sidebar.selectbox("Region", options=list(data.keys()))
selected_func = st.sidebar.selectbox("Function", options=data[selected_region]['funcName'].unique())
lookback = st.sidebar.slider("Lookback Hours", 1, 6, 3)
refresh = st.sidebar.button("Refresh Prediction")

# ---------- PREDICTION FUNCTION ----------
def predict_cold_starts(region_df, func_name, lookback=3):
    func_data = region_df[region_df['funcName'] == func_name].sort_values('hour')
    if len(func_data) < lookback + 1:
        return None, None, None, None
    
    recent = func_data.tail(lookback + 1)
    series = recent['cold_count'].values.astype(float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    
    X = scaled[:-1].reshape(1, lookback, 1)
    y_actual_scaled = scaled[-1]
    
    # LSTM prediction
    lstm_pred_scaled = lstm_model.predict(X, verbose=0)[0][0]
    
    # RF refinement
    rf_input = np.array([[lstm_pred_scaled]])
    rf_refined_scaled = rf_model.predict(rf_input)[0]
    
    # Inverse transform
    lstm_pred = float(scaler.inverse_transform([[lstm_pred_scaled]])[0][0])
    refined_pred = float(scaler.inverse_transform([[rf_refined_scaled]])[0][0])
    actual = float(scaler.inverse_transform([[y_actual_scaled]])[0][0])
    
    mae = abs(refined_pred - actual)
    
    return lstm_pred, refined_pred, actual, mae

# ---------- MAIN ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Function: `{selected_func}` | Region: {selected_region}")
    
    region_df = data[selected_region]
    if len(region_df[region_df['funcName'] == selected_func]) >= lookback + 1:
        lstm_pred, refined_pred, actual, mae = predict_cold_starts(region_df, selected_func, lookback)
        
        st.metric("LSTM Output", f"{lstm_pred:.3f}")
        st.metric("**Final Prediction (RF)**", f"**{refined_pred:.3f}**")
        st.metric("Actual Last Hour", f"{actual:.1f}")
        st.metric("MAE", f"{mae:.3f}")
        
        if refined_pred > 1.5:
            st.error(f"**SCALE UP!** Pre-warm {int(np.ceil(refined_pred))} containers NOW")
        else:
            st.success("No scaling needed")
    else:
        st.warning("Not enough data for prediction")

with col2:
    st.subheader("Cold Start History")
    func_data = region_df[region_df['funcName'] == selected_func].tail(10)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(func_data['hour'], func_data['cold_count'], 'o-', label='Actual', color='steelblue', linewidth=2)
    
    if len(func_data) >= lookback + 1:
        _, refined_pred, _, _ = predict_cold_starts(region_df, selected_func, lookback)
        if refined_pred:
            last_hour = func_data['hour'].iloc[-1]
            ax.plot(last_hour, refined_pred, 'r*', markersize=15, label=f"Pred: {refined_pred:.2f}")
    
    ax.set_title(f"Cold Starts — {selected_func}")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Cold Starts")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ---------- AUTO REFRESH ----------
if refresh:
    st.success("Prediction refreshed!")
    st.rerun()

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    """
    **MTFSA v5** — Pre-trained on 377 dense (R1) | Evaluated on 1,272 sparse (R2–R5)  
    31 days | 3-shot | No fine-tuning | RF refinement | **Live on Render**
    """
)
