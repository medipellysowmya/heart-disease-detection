import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open("Artifacts/Model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Heart Disease Prediction")

age = st.number_input("Age")
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain", [0,1,2,3])
trestbps = st.number_input("Resting BP")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("FBS", [0,1])
restecg = st.selectbox("ECG", [0,1,2])
thalach = st.number_input("Max HR")
exang = st.selectbox("Exercise Angina", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("CA", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

if st.button("Predict"):
    data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    # Load preprocessor
    with open("Artifacts/Preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
        
    scaled_data = preprocessor.transform(data)
    result = model.predict(scaled_data)[0]

    if result == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")
