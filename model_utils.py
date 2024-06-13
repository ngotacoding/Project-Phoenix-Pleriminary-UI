import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
import dill
import streamlit as st

# Load the saved model
with open("artifacts/xgb_mm.pkl", 'rb') as f:
    model = dill.load(f)

# Load the saved preprocessor
with open('./artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = dill.load(f)

def predict_claim():
    severity = st.session_state['severity']
    private_attorney = st.session_state['private_attorney']
    marital_status = st.session_state['marital_status']
    specialty = st.session_state['specialty']
    insurance = st.session_state['insurance']
    gender = st.session_state['gender']
    age = st.session_state['age']
       
    # Create a DataFrame for the single entry
    data = pd.DataFrame({
        'severity': [severity],
        'private_attorney': [private_attorney],
        'marital_status': [marital_status],
        'specialty': [specialty],
        'insurance': [insurance],
        'gender': [gender],
        'age': [age]
    })

    # Preprocess the data
    data_transformed = preprocessor.transform(data).toarray()

    # Make the prediction
    prediction = model.predict(data_transformed)

    return prediction[0]
