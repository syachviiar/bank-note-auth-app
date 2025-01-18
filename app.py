import streamlit as st
import numpy as np
import joblib
import os

# Load model dengan penanganan error
model = None
try:
    if not os.path.exists('model.pkl'):
        st.error("File model.pkl tidak ditemukan! Pastikan file tersebut ada di direktori aplikasi.")
    else:
        model = joblib.load('model.pkl')
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")

# Judul aplikasi
st.title("Bank Note Authentication App")
st.write("Aplikasi ini memprediksi apakah bank note asli atau palsu berdasarkan data input. dibuat oleh A11.2022.14784")

# Input data
st.header("Masukkan Nilai")
variance = st.number_input("Variance", value=0.0, help="Nilai variance dari fitur bank note.")
skewness = st.number_input("Skewness", value=0.0, help="Nilai skewness dari fitur bank note.")
curtosis = st.number_input("Curtosis", value=0.0, help="Nilai curtosis dari fitur bank note.")
entropy = st.number_input("Entropy", value=0.0, help="Nilai entropy dari fitur bank note.")

# Prediksi
if st.button("Predict"):
    if model is None:
        st.error("Model belum dimuat. Periksa file model.pkl Anda.")
    else:
        # Masukkan data ke model
        input_data = np.array([[variance, skewness, curtosis, entropy]])
        st.write(f"Input data: {input_data}")  # Log untuk debugging

        try:
            prediction = model.predict(input_data)

            # Tampilkan hasil
            if prediction[0] == 0:
                st.success("Hasil: Bank Note Asli")
            else:
                st.error("Hasil: Bank Note Palsu")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
