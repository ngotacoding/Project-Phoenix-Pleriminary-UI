import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import AdaBoostRegressor
import dill
import joblib
import streamlit as st

# Load the saved model
with open("artifacts/adaboost_model_insurance_csv.pkl", 'rb') as f:
    model = joblib.load(f)
    
# Load the saved preprocessor
with open('./artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = dill.load(f)

def basic_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and engineer input features in preparation for advanced preprocessing

    Args:
        data (pd.DataFrame): Dataframe with columns to be preprocessed.

    Returns:
        data (pd.DataFrame): The preprocessed dataframe.
    """

    # create age category
    data['age_category'] = np.where((data['age']>=0) & (data['age']<2), 'infant(0-2)',
                                            np.where((data['age']>=2) & (data['age']<12),'kid(2-12)',
                                            np.where((data['age']>=12) & (data['age']<18),'teen(12-18)',
                                            np.where((data['age']>=18) & (data['age']<35),'young adult(18-35)',
                                            np.where((data['age']>=35) & (data['age']<60),'adult(35-60)',
                                            np.where(data['age']>=60,'senior(60+)','error'))))))


    # remove if umbrella limit is negative and convert it into categorical feature
    data['umbrella_limit'].replace({0:'0M', 2000000:'2M', 3000000:'3M', 4000000:'4M', 5000000:'5M', 6000000:'6M',
                                        7000000:'7M',8000000:'8M',9000000:'9M',10000000:'10M'},inplace=True)

    # convert bodily injuries into categorical feature
    data['bodily_injuries'].replace({0:'no_injury', 1:'minor_injury', 2:'major_injury'},inplace=True)

    # drop age column as it can cause multi-collinearity
    data.drop('age',axis=1,inplace=True)


    return data

def predict_claim() -> float:
    """Function to predict the claim amount with data collected using streamlit form.

    Returns:
        estimated_claim (float): The estimated claim amount 
    """
    # Define form data
    form_data = {
        'age': st.session_state.age,
        'policy_state': st.session_state.policy_state,
        'policy_csl': st.session_state.policy_csl,
        'umbrella_limit': st.session_state.umbrella_limit,
        'insured_sex': st.session_state.insured_sex,
        'accident_type': st.session_state.accident_type,
        'collision_type': st.session_state.collision_type,
        'incident_severity': st.session_state.incident_severity,
        'authorities_contacted': st.session_state.authorities_contacted,
        'state': st.session_state.state,
        'property_damage': st.session_state.property_damage,
        'bodily_injuries': st.session_state.bodily_injuries,
        'police_report_available': st.session_state.police_report_available,
        'auto_make': st.session_state.auto_make,
        'total_claim_amount': 1,
        'injury_claim': 1
    }
    
    # Create a DataFrame for the single entry
    data = pd.DataFrame([form_data])
    data = basic_preprocessing(data)
    
    # Preprocess the data
    data_transformed = preprocessor.transform(data)

    # Make the prediction
    prediction = model.predict(data_transformed)
    estimated_claim = prediction[0]
    
    return estimated_claim