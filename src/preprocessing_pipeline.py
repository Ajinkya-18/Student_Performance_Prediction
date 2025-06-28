import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from utils import load_data, clean_data

df_path = 'data/math_subject_grade_prediction.csv'

df = load_data(df_path)

df = clean_data(df)


onehot_cols = ['Father_Job', 'Mother_Job']
lab_enc_cols = ['School_Support', 'Higher_Edu', 'Extra_Paid_Classes']
scaling_cols = ['Past_Grade_Record', 'School_Absences', 'Past_Class_Failure_Count', 
                'Family_Relationship', 'Goes_Out', 'Mother_Job', 'Weekly_Study_Time', 
                'Alcohol_Consumption', 'Father_Job', 'Freetime_After_School', 'Age', 
                'Parents_Education']


transformers = [('onehot_enc', OneHotEncoder(), onehot_cols),
                ('lab_enc', LabelEncoder(), lab_enc_cols),
                ('scaler', RobustScaler(), scaling_cols)]
col_trnsfrmr = ColumnTransformer(transformers=transformers, verbose=True, n_jobs=6)


steps = [(), ('col_trnsfrmr', col_trnsfrmr), ()]
pipeline = Pipeline(steps=steps, verbose=True)

