import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from utils import load_data, clean_data, split_data

df_path = 'data/math_subject_grade_prediction.csv'

df = load_data(df_path)

df = clean_data(df)

print(f"DataFrame Shape: {df.shape}")


onehot_cols = ['Father_Job', 'Mother_Job']
lab_enc_cols = ['School_Support', 'Higher_Edu', 'Extra_Paid_Classes']
scaling_cols = ['Past_Grade_Record', 'School_Absences', 'Past_Class_Failure_Count', 
                'Family_Relationship', 'Goes_Out', 'Weekly_Study_Time', 'Age',
                'Alcohol_Consumption', 'Freetime_After_School', 'Parents_Education']

one_hot_enc = OneHotEncoder()
lab_enc = LabelEncoder()
rob_scaler = RobustScaler()

one_hot_encoded_features = one_hot_enc.get_feature_names_out(onehot_cols)

transformers = [('onehot_enc', one_hot_enc, onehot_cols),
                ('lab_enc', lab_enc, lab_enc_cols),
                ('scaler', rob_scaler, scaling_cols)]
col_trnsfrmr = ColumnTransformer(transformers=transformers, verbose=True, n_jobs=6)
col_trnsfrmr.fit

# steps = [()]
# pipeline = Pipeline(steps=steps, verbose=True)

x_train, x_test, y_train, y_test = split_data(df)