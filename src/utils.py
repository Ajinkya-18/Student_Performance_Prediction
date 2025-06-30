def load_data(csv_path:str, ends_with='.csv'):
    import os
    import pandas as pd
    from pathlib import Path
    
    cwd = os.getcwd()
    full_data_path = os.path.join(cwd, Path(csv_path))

    print(full_data_path)

    if os.path.exists(full_data_path) and csv_path.endswith(ends_with):
        return pd.read_csv(full_data_path)
    
    else:
        raise ValueError('Invalid dataset path!')


def clean_data(df, clean_col_names:bool=True):

    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    from sklearn.compose import ColumnTransformer

    if clean_col_names:
        df['Past_Performance_Grade'] = (df.G1 + df.G2) / 2
        df['Parents_Education'] = (df.Medu + df.Fedu) / 2
        df['Alcohol_Consumption'] = (df.Walc + df.Dalc)

        df.drop(['school', 'sex','G1', 'G2', 'Medu', 'Fedu', 'Dalc', 'Walc', 'guardian', 'reason'], 
        axis=1, inplace=True)

        df.columns=['Age', 'Locality', 'Family_Size', 'Parents_Cohab_Status', 'Mother_Job',
            'Father_Job', 'Home_to_School_Travel_Time', 'Weekly_Study_Time', 'Past_Class_Failure_Count', 
            'School_Support', 'Family_Support', 'Extra_Paid_Classes', 'Extra_Curr_Activities', 
            'Attended_Kindergarten', 'Higher_Edu', 'Internet', 'Dating', 'Family_Relationship', 
            'Freetime_After_School', 'Goes_Out', 'Current_Health_Status', 'School_Absences', 
            'Final_Grade', 'Past_Grade_Record', 'Parents_Education', 'Alcohol_Consumption']


    print("Returning cleaned data!")
    return df


def split_data(df, target:str='Final_Grade', test_ratio:float=0.25):
    X = df.drop([target], axis=1)
    Y = df[target]

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio, random_state=42)

    return x_train, x_test, y_train, y_test


