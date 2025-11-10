import streamlit as st
import pickle
import numpy as np
import os # <-- Tambahkan import os

# --- Perbaikan Path Model ---
# Dapatkan path absolut dari folder tempat script ini (iris.py) berada
script_dir = os.path.dirname(os.path.abspath(__file__))

# Tentukan nama file model Anda
model_filename = "save_data_iris.pkcls"

# Gabungkan path folder script dengan nama file model
# Ini mengasumsikan 'save_data_iris.pkcls' ada di folder YANG SAMA dengan 'iris.py'
model_path = os.path.join(script_dir, model_filename)
# -----------------------------


# Load Model menggunakan path yang sudah diperbaiki
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: File model tidak ditemukan di path: {model_path}")
    st.stop()
except ModuleNotFoundError:
    st.error("Error: ModuleNotFoundError. Pastikan 'Orange3' ada di requirements.txt Anda.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi error saat me-load model: {e}")
    st.stop()


st.title("😭 Iris Flower Classifier 😭")
st.write("Aplikasi sederhana untuk memprediksi jenis bunga iris menggunakan model Neural Network dari Orange.")

# Input features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# Prepare data
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict
if st.button("Prediksi"):
    try:
        # Orange model (terutama NN) mungkin mengembalikan probabilitas atau
        # array nama. Kita asumsikan model(features) mengembalikan array prediksi.
        prediction = model(features)[0] 
        
        st.subheader("Hasil Prediksi")
        st.success(f"👌 Jenis Bunga: *{prediction}*")
        st.balloons()
        
    except Exception as e:
        st.error(f"Error saat melakukan prediksi: {e}")
        st.write("Model Anda mungkin mengharapkan format input yang berbeda.")
