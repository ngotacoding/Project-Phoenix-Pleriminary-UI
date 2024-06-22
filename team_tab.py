import streamlit as st
import pandas as pd

# Data
data = {
    "Collaborators": [
        "Rasha Salim", "Josh Clements", "Bhushan Chougule (Project Manager)", "Melissa Wykle (Phoenix)",
        "Behnaz Hosseini (Lead)", "Sahar Nikoo", "Chua Shin Ying", "Sri Hari Sivashanmugam",
        "Yani Stancheva (Lead)", "Mariam Mukami Njeru", "Tanvi Thanekar", "Indrajith C (Lead)",
        "Varun Yadav", "Ekoue LOGOSU-TEKO (Lead)", "Krishna Karthik", "David Hartsman",
        "Nandini K", "Farjana", "Mohit Mishra", "Joseph Ngota Chilo (Lead)", "Fahira Hameed",
        "John Paul Curada", "Khushal Gogia", "Ashwitha Kassetty", "Mohamad Motaz",
        "Ahmed Mostafa Attia"
    ]
}


# Convert data to DataFrame
df = pd.DataFrame(data)

# Function to display team members


def display_team():
    st.header(":blue[Meet our team members👋]")
    col1, col2 = st.columns(2)
    with col1:
        st.text("\n".join(df['Collaborators'][:len(df)//2]))
    with col2:
        st.text("\n".join(df['Collaborators'][len(df)//2:]))
