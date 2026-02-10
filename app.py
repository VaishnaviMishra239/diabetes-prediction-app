import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

st.title("Diabetes Prediction System")

preg = st.number_input("Pregnancies")
glu = st.number_input("Glucose Level")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
ins = st.number_input("Insulin Level")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    data = scaler.transform(data)

    result = model.predict(data)

    if result[0] == 1:
        st.error("Person is Diabetic")
    else:
        st.success("Person is Not Diabetic")
