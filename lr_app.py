import streamlit as st
import numpy as np
import joblib

model = joblib.load("linear_regression_model.joblib")

st.title("Scikit-learn Linear Regression Model")

tv = st.text_input("Enter TV sales")
radio = st.text_input("Enter Radio sales")
newspaper = st.text_input("Enter Newspaper sales")

if st.button("Predict"):
    try:
        tv = float(tv)
        radio = float(radio)
        newspaper = float(newspaper)

        features = np.array([[tv, radio, newspaper]])
        prediction = model.predict(features)

        st.success(f"Predicted Sales: {prediction[0]:.2f}")
    except:
        st.error("Please enter valid numbers")
