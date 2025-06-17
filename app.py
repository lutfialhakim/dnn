import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------------------
# 1️⃣ Muat Model dan scaler, imputer yang digunakan saat training
model = load_model('model_diabetes.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ------------------------------------------------------------------------------
# 2️⃣ Judul Aplikasi Streamlit
st.title('Aplikasi Deteksi Diabetes')

# ------------------------------------------------------------------------------
# 3️⃣ Input Fitur dari Pengguna
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=80)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=79)
bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, value=23.5)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.47)
age = st.number_input('Age', min_value=0, max_value=100, value=33)

# ------------------------------------------------------------------------------
# 4️⃣ Jika Tombol "Prediksi" ditekan
if st.button('Prediksi'):
    # Gabungkan input
    input_features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

       # Scaling
    input_features = scaler.transform(input_features)

    # Prediksi
    prediction = (model.predict(input_features) > 0.5).astype(int)[0][0]

    if prediction == 1:
        st.error('Pasien diprediksi DIABETES.')
    else:
        st.success('Pasien diprediksi TIDAK DIABETES.')
