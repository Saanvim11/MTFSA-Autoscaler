import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown
import joblib

# ---------------------- APP CONFIG ----------------------
st.set_page_config(page_title="MTFSA Cloud Autoscaler", layout="wide")
st.title("☁️ MTFSA: Meta-LSTM Cloud Autoscaler")

# ---------------------- MODEL DOWNLOAD ----------------------
@st.cache_resource(show_spinner="Downloading models...", persist=False)
def download_models():
    os.makedirs("/tmp/models", exist_ok=True)

    # ✅ Replace with your actual Google Drive file IDs
    drive_links = {
        "final_model2.keras": "https://drive.google.com/file/d/191BZrVEkET0lMD9RQiQZzbRItXHYalHU/view?usp=drive_link",
        "rf_refiner.pkl": "https://drive.google.com/file/d/1bWIwYk4bn-nWnfKInCXGGgl1aXY-HQLH/view?usp=drive_link",
    }

    local_paths = {}
    for name, file_id in drive_links.items():
        output_path = f"/tmp/models/{name}"
        if not os.path.exists(output_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
        local_paths[name] = output_path

    return local_paths


# ---------------------- LOAD MODELS ----------------------
def load_lstm_model(path):
    return load_model(path, compile=False)


def load_rf_model(path):
    return joblib.load(path)


# ---------------------- EXECUTION ----------------------
paths = download_models()
LSTM_MODEL_PATH = paths["final_model2.keras"]
RF_MODEL_PATH = paths["rf_refiner.pkl"]

st.write("✅ LSTM model path:", LSTM_MODEL_PATH)
st.write("✅ Exists:", os.path.exists(LSTM_MODEL_PATH))

try:
    lstm_model = load_lstm_model(LSTM_MODEL_PATH)
    st.success("✅ LSTM Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading LSTM model: {e}")

# ---------------------- INPUT SECTION ----------------------
st.header("📈 Input Metrics")

cpu = st.number_input("CPU Utilization (%)", 0.0, 100.0, 60.0)
memory = st.number_input("Memory Utilization (%)", 0.0, 100.0, 55.0)
disk = st.number_input("Disk I/O (MB/s)", 0.0, 500.0, 120.0)
network = st.number_input("Network I/O (MB/s)", 0.0, 500.0, 80.0)

if st.button("Predict Scaling Decision"):
    input_data = np.array([[cpu, memory, disk, network]])
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    try:
        prediction = lstm_model.predict(input_scaled)
        st.subheader("🔮 Predicted Scaling Output:")
        st.write(prediction)
    except Exception as e:
        st.error(f"Prediction error: {e}")

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
st.pyplot(fig)

st.info("Built with TensorFlow 2.15.0, Python 3.10, and Streamlit 1.30.0")
