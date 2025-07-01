from joblib import load

model_default_path = '../models/trained_rfr_model.joblib'
# model_path = input('Enter trained model file path: ')

x = 

with open(model_default_path, 'rb') as f:
    model = load(f)
    y_hat = model.predict(x)


