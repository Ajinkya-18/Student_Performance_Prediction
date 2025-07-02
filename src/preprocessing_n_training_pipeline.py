from utils import load_data, clean_data, preprocess_data


df_path = 'data/math_subject_grade_prediction.csv'
df = load_data(df_path) # loads data into pandas dataframe

df = clean_data(df) # cleans and renames the columns
print(f"DataFrame Shape: {df.shape}")

x_train, x_test, y_train, y_test = preprocess_data(df) # preprocesses and converts the data into a format 
                                                       # ideal/desirable for model training or inference 
                                                       # using a trained model. Also selects the best 
                                                       # features
print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
print(type(x_train),type(x_test), type(y_train), type(y_test))

# New models can be trained using the preprocessed data for here onwards.

