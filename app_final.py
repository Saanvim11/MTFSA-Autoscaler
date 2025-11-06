# app_final.py — MTFSA LIVE DASHBOARD (RENDER.COM READY)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

LSTM_MODEL_PATH = MODEL_DIR / "final_model2.keras"
RF_MODEL_PATH = MODEL_DIR / "rf_refiner.pkl"

# ---------- VALIDATE FILES ----------
if not LSTM_MODEL_PATH.exists():
    st.error(f"Missing: {LSTM_MODEL_PATH}")
    st.stop()
if not RF_MODEL_PATH.exists():
    st.error(f"Missing: {RF_MODEL_PATH}")
    st.stop()

st.set_page_config(page_title="MTFSA Live", layout="wide")
st.title("MTFSA: Hybrid Meta-Few-Shot Autoscaler (LIVE)")
st.markdown("**TL + FSL + 1-Step Meta + RF Refinement** | RMSE = 0.117 | R² = 0.876")

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
    for r in range(1, 6):
        path = DATA_DIR / f"r{r}_d0_to_d30_hourly.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                datasets[f"R{r}"] = df
            except Exception as e:
                st.warning(f"Failed to load {path.name}: {e}")
    return datasets

data = load_all_data()
if not data:
    st.error("No data in `data/processed/`")
    st.stop()

# ---------- UI ----------
st.sidebar.header("Controls")
region = st.sidebar.selectbox("Region", options=list(data.keys()))
func = st.sidebar.selectbox("Function", options=data[region]['funcName'].unique())
lookback = st.sidebar.slider("Lookback", 1, 6, 3)
refresh = st.sidebar.button("Refresh")

# ---------- PREDICTION ----------
def predict(region_df, func_name, lookback):
    df = region_df[region_df['funcName'] == func_name].sort_values('hour')
    if len(df) < lookback + 1:
        return None, None, None, None
    recent = df.tail(lookback + 1)['cold_count'].values.astype(float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(recent.reshape(-1, 1)).flatten()
    X = scaled[:-1].reshape(1, lookback, 1)
    lstm_pred = float(scaler.inverse_transform([[lstm_model.predict(X, verbose=0)[0][0]]])[0][0])
    rf_pred = float(scaler.inverse_transform([[rf_model.predict([[lstm_pred]])[0]]])[0][0])
    actual = float(recent[-1])
    mae = abs(rf_pred - actual)
    return lstm_pred, rf_pred, actual, mae

# ---------- DISPLAY ----------
col1, col2 = st.columns(2)
df = data[region]

with col1:
    st.subheader(f"`{func}` | {region}")
    if len(df[df['funcName'] == func]) >= lookback + 1:
        lstm_p, rf_p, actual, mae = predict(df, func, lookback)
        st.metric("LSTM", f"{lstm_p:.3f}")
        st.metric("**Final (RF)**", f"**{rf_p:.3f}**")
        st.metric("Actual", f"{actual:.1f}")
        st.metric("MAE", f"{mae:.3f}")
        st.write("**SCALE UP!**" if rf_p > 1.5 else "No action")
    else:
        st.warning("Not enough data")

with col2:
    st.subheader("History")
    hist = df[df['funcName'] == func].tail(10)
    fig, ax = plt.subplots()
    ax.plot(hist['hour'], hist['cold_count'], 'o-', label='Actual')
    if len(hist) >= lookback + 1:
        _, rf_p, _, _ = predict(df, func, lookback)
        ax.plot(hist['hour'].iloc[-1], rf_p, 'r*', markersize=15, label=f"Pred: {rf_p:.2f}")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Cold Starts")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

if refresh:
    st.rerun()

st.markdown("---")
st.caption("**MTFSA v5** — 3-shot | No fine-tuning | Live on Render")
