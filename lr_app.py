import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("advertising.csv")

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Train model inside app
model = LinearRegression()
model.fit(X, y)

st.title("Scikit-learn Linear Regression Model")

tv = st.text_input("Enter TV sales")
radio = st.text_input("Enter Radio sales")
newspaper = st.text_input("Enter Newspaper sales")

if st.button("Predict"):
    try:
        features = np.array([[float(tv), float(radio), float(newspaper)]])
        prediction = model.predict(features)
        st.success(f"Predicted Sales: {prediction[0]:.2f}")
    except:
        st.error("Please enter valid numeric values")
