from utils import load_data, clean_data, preprocess_data, load_model
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output



df_path = 'data/math_subject_grade_prediction.csv'
df = load_data(df_path) # loads data into pandas dataframe

df = clean_data(df) # cleans and renames the columns
# print(f"DataFrame Shape: {df.shape}")

print('Processing data...')

x_train, x_test, y_train, y_test = preprocess_data(df)

# print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
# print(type(x_train),type(x_test), type(y_train), type(y_test))


model = load_model('models/rfr_fitted_with_best_params.joblib') # loads the trained model from disk
# print(type(model))

y_preds = model.predict(x_test) # makes predictions on the test set

from sklearn.metrics import r2_score
print(f"Model score: {round(r2_score(y_test, y_preds)*100, 2)} %")

