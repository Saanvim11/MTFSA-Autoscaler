import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# ---------------------- APP CONFIG ----------------------
st.set_page_config(page_title="MTFSA Cloud Autoscaler", layout="wide")
st.title("☁️ MTFSA: Meta-LSTM Cloud Autoscaler")

# ---------------------- PATH SETUP ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "final_model2.keras")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_refiner.pkl")

# ---------------------- LOAD MODELS ----------------------
@st.cache_resource(show_spinner="🔄 Loading models...")
def load_models():
    if not os.path.exists(LSTM_MODEL_PATH):
        st.error(f"LSTM model not found at {LSTM_MODEL_PATH}")
        st.stop()
    if not os.path.exists(RF_MODEL_PATH):
        st.error(f"RF model not found at {RF_MODEL_PATH}")
        st.stop()

    lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
    rf_model = joblib.load(RF_MODEL_PATH)
    return lstm_model, rf_model


lstm_model, rf_model = load_models()
st.sidebar.success("✅ Models loaded successfully!")

# ---------------------- INPUT SECTION ----------------------
st.header("📈 Input Metrics")

cpu = st.number_input("CPU Utilization (%)", 0.0, 100.0, 60.0)
memory = st.number_input("Memory Utilization (%)", 0.0, 100.0, 55.0)
disk = st.number_input("Disk I/O (MB/s)", 0.0, 500.0, 120.0)
network = st.number_input("Network I/O (MB/s)", 0.0, 500.0, 80.0)

if st.button("🔮 Predict Scaling Decision"):
    input_data = np.array([[cpu, memory, disk, network]])
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    try:
        prediction = lstm_model.predict(input_scaled)
        st.subheader("🚀 Predicted Scaling Output:")
        st.write(prediction)
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# ---------------------- VISUALIZATION ----------------------
st.header("📊 Metric Distribution (Demo)")

demo_data = pd.DataFrame({
    "CPU": np.random.normal(cpu, 5, 50),
    "Memory": np.random.normal(memory, 5, 50),
    "Disk": np.random.normal(disk, 10, 50),
    "Network": np.random.normal(network, 10, 50),
})

fig, ax = plt.subplots()
sns.boxplot(data=demo_data, ax=ax)
ax.set_title("System Metrics Distribution")
st.pyplot(fig)

st.caption("🧠 Built with TensorFlow 2.15.0 • Python 3.10 • Streamlit 1.30.0")
