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

# Preprocessing the data
    # get the data and do the basic pre-processing before splitting
    def basic_preprocessing(data: pd.DataFrame):

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


    # final preprocessing
    def final_preprocessing(data):
        numerical_pipeline = Pipeline([
                                        ('scale', StandardScaler())
                                    ])

        categorical_pipeline = Pipeline(steps=[
                                                ('impute',SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
                                                ('onehot',OneHotEncoder(handle_unknown='ignore'))
                                        ])

        num_features = data.select_dtypes('number').columns.tolist()
        obj_features = data.select_dtypes('object').columns.tolist()

        full_pipeline = ColumnTransformer([
                                            ('numerical',numerical_pipeline,num_features),
                                            ('categorical',categorical_pipeline,obj_features)
                                        ],remainder='passthrough'
                                        )

        return full_pipeline

# Load the saved preprocessor
with open('./artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = dill.load(f)

def predict_claim(form_data):
    
    # Create a DataFrame for the single entry
    data = pd.DataFrame([form_data])
    data = basic_preprocessing(data)
    
    # Preprocess the data
    data_transformed = preprocessor.transform(data)

    # Make the prediction
    prediction = model.predict(data_transformed)

    return prediction[0]