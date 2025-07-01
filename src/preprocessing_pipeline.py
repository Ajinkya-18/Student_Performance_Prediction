import pandas as pd
import numpy as np
import os

from utils import load_data, clean_data, preprocess_data


df_path = 'data/math_subject_grade_prediction.csv'

df = load_data(df_path)
df = clean_data(df)
print(f"DataFrame Shape: {df.shape}")

x_train, x_test, y_train, y_test = preprocess_data(df)

