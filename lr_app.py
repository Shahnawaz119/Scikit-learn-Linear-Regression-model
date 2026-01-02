import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Scikit-learn Linear Regression model')
tv=st.text_input('Enter TV sales...')
radio=st.text_input('Enter radio sales...')
newspaper=st.text_input('Enter newspaper sales...')

if st.button("predict"):
    features=np.array([[tv,radio,newspaper]],dtype=np.float64)
    result=model.predict(features).reshape(1,-1)
    st.write("Predict Sales:::",result[0])