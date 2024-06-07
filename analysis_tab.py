import streamlit as st
import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("./data/Kaggle_medical_practice_20.csv")
    return df

def display_analysis():
    df = load_data()
    st.header("Analysis")
    st.write("This is the analysis tab.")
    # Add your analysis content here
    st.write(df.describe())
    st.write("Correlation matrix:")
    st.write(df.select_dtypes(include=[np.number]).corr())
