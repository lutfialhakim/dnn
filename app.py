import streamlit as st
import numpy as np
import pickle
import os

# Coba muat model jika tersedia
model = None
model_path = 'model_diabetes.h5'

if os.path.exists(model_path):
    from tensorflow.keras.models import load_model
    model = load_model(model_path, compile=False)
else:
    st.warning(f"❗ File model '{model_path}' tidak ditemukan. Harap pastikan file tersebut berada di folder yang sama dengan app.py.")

# Coba muat scaler jika tersedia
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.warning("❗ File scaler.pkl tidak ditemukan.")

# ------------------------------------------------------------------------------

st.title('🩺 Aplikasi Deteksi Diabetes')
st.markdown("Masukkan nilai untuk fitur-fitur berikut:")

glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.47)
age = st.number_input('Age', min_value=0, max_value=100, value=33)
bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, value=23.5)

if st.button('🔍 Prediksi'):
    if model is None:
        st.error("❌ Model belum dimuat. Harap periksa file 'model_diabetes.h5'.")
    elif 'scaler' not in locals():
        st.error("❌ Scaler belum dimuat. Harap periksa file 'scaler.pkl'.")
    else:
        try:
            # Perhatikan urutan input harus sesuai saat training
            input_features = np.array([[glucose, diabetes_pedigree, age, bmi]])

            # Transformasi dengan scaler
            scaled_input = scaler.transform(input_features)

            # Prediksi probabilitas
            prob = model.predict(scaled_input)[0][0]

            # Tampilkan hasil
            st.subheader("📊 Hasil Prediksi")
            st.write(f"**Probabilitas diabetes:** `{prob:.4f}`")
            st.write("**Data yang sudah diskalakan:**", scaled_input)

            if prob > 0.5:
                st.error('⚠️ Pasien diprediksi **DIABETES**.')
            else:
                st.success('✅ Pasien diprediksi **TIDAK DIABETES**.')

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {str(e)}")
