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

#----------------------------------------------------------------------------------------------------------

def clean_data(df):

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

#----------------------------------------------------------------------------------------------------------

def split_data(df, target:str='Final_Grade', test_ratio:float=0.25):
    X = df.drop([target], axis=1)
    Y = df[target]

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio, random_state=42)

    return x_train, x_test, y_train, y_test

#----------------------------------------------------------------------------------------------------------

def preprocess_data(df, col_transformer_path:str='models/col_transformer_fitted.joblib'):
    # from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    # from sklearn.compose import ColumnTransformer
    # from sklearn.pipeline import Pipeline

    # onehot_cols = ['Father_Job', 'Mother_Job']
    # ord_enc_cols = ['School_Support', 'Family_Support', 'Higher_Edu', 'Extra_Paid_Classes', 
    #                 'Locality', 'Family_Size', 'Parents_Cohab_Status', 'Dating', 
    #                 'Extra_Curr_Activities', 'Attended_Kindergarten', 'Internet']
    # df_cols = list(df.columns)
    # enc_cols = onehot_cols+ord_enc_cols
    # scaler_cols = [col for col in df_cols if col not in enc_cols]
    # scaler_cols.remove('Final_Grade')

    # transformers = [('onehot', OneHotEncoder(sparse_output=False), onehot_cols), 
    #                 ('ord_enc', OrdinalEncoder(), ord_enc_cols), 
    #                 ('scaler', StandardScaler(), scaler_cols)]

    # col_transformer = ColumnTransformer(transformers=transformers, remainder='passthrough', 
    #                                sparse_threshold=0, n_jobs=6, verbose=True)
    # col_transformer.set_output(transform='pandas')
    # col_transformer.fit(x_train)
    # col_transformer.save_model('../models/col_transformer_fitted.joblib')

    import os
    from pathlib import Path
    cwd = os.getcwd()
    full_col_transformer_path = os.path.join(cwd, Path(col_transformer_path))
    col_transformer = load_model(full_col_transformer_path)

    x_train, x_test, y_train, y_test = split_data(df)
    x_train_new = col_transformer.transform(x_train)  # scales the data and encodes the categorical
    x_test_new = col_transformer.transform(x_test)    # features in train and test splits of X.

    # now get the best features
    best_features = select_best_features('models/grid_search_best_estimator_features.joblib')

    return x_train_new[best_features], x_test_new[best_features], y_train, y_test

#----------------------------------------------------------------------------------------------------------

def select_best_features(best_features_path:str='models/grid_search_best_estimator_features.joblib'):
    import os
    from pathlib import Path

    cwd = os.getcwd()
    full_best_features_path = os.path.join(cwd, Path(best_features_path))

    if os.path.exists(full_best_features_path) and best_features_path.endswith('.joblib'):
        from joblib import load

        with open(full_best_features_path, 'rb') as f:
            return load(f)

    else:
        raise ValueError('Invalid file path or file extension!')
    
#--------------------------------------------------------------------------------------

def load_model(model_path:str):
    import os
    from pathlib import Path
    cwd = os.getcwd()
    full_model_path = os.path.join(cwd, Path(model_path))

    if os.path.exists(full_model_path) and full_model_path.endswith('.joblib'):
        from joblib import load

        with open(full_model_path, 'rb') as f:
            return load(f)
        
    else:
        raise ValueError('Invalid path or path extension!')

#-------------------------------------------------------------------------------------

def save_model(save_obj, save_path:str):
    import os
    from pathlib import Path

    cwd = os.getcwd()
    full_save_path = os.path.join(cwd, Path(save_path))

    if os.path.exists(full_save_path) and full_save_path.endswith('.joblib'):
        from joblib import dump
        with open(full_save_path, 'wb') as f:
            dump(save_obj, f)
            
            return True
        
    else:
        raise ValueError('Invalid path or path extension!')

#----------------------------------------------------------------------------------------------------------

def train_model(x_train, y_train, model_instance=None):
    try:
        if model_instance is not None:
            model_instance.fit(x_train, y_train)
            print('Trained the given model instance. Returning it now.')
            return model_instance
        
        else:
            raise ValueError('Please provide a valid model instance for training.')
    
    except Exception as e:
        raise e

#-----------------------------------------------------------------------------------------------------------

def test_model(x_test, model_instance=None, fitted_model_path:str='models/rfr_fitted_with_best_params.joblib'):

    try:
        if model_instance is not None:
            y_preds = model_instance.predict(x_test)
            return y_preds
    
        else:
            from joblib import load
            import os
            from pathlib import Path

            cwd = os.getcwd()
            full_fitted_model_path = os.path.join(cwd, Path(fitted_model_path))

            if os.path.exists(full_fitted_model_path) and full_fitted_model_path.endswith('.joblib'):
                with open(full_fitted_model_path, 'rb') as f:
                    model = load(f)
                    y_preds = model.predict(x_test)
                    return y_preds
            else:
                raise ValueError('Invalid model path or file extension')
            
    except Exception as e:
        raise e
    



