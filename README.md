# 📊 Student Performance Predictor

This project predicts students' **Mathematics grades** using a variety of regression models. It analyzes the influence of **alcohol consumption** and other socio-academic factors on student performance using a dataset sourced from Kaggle. The goal is to explore how external lifestyle factors may impact academic results and to compare multiple machine learning models for predictive accuracy.

---

## 📁 Project Structure

student-performance-prediction/
│
├── data/
│ └── math_subject_grade_prediction.csv # Dataset used in training (originally from Kaggle)
│
├── models/
│ ├── trained_abr_model.pkl # Trained AdaBoost Regressor
│ ├── trained_hgbr_model.pkl # Trained Histogram-based Gradient Boosting Regressor
│ └── trained_rfr_model.pkl # Trained Random Forest Regressor
│
├── notebooks/
│ └── Performance_Predictor.ipynb # Jupyter notebook with data processing and training code
│
├── reports/
│ └── Models_Performance.png # Comparison chart of model performance
│
├── requirements.txt # Required Python packages
├── environment.yml # Conda environment configuration
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🧠 Features Used

- Student demographics (age, gender)
- Parental education level
- Study time and failures
- Alcohol consumption (`Dalc`, `Walc`)
- Health and absences
- School support and family relationship

---

## 🛠️ Models Implemented

- 📌 **Random Forest Regressor**
- 📌 **AdaBoost Regressor**
- 📌 **Histogram-based Gradient Boosting Regressor**

Each model was evaluated using metrics like **Mean Absolute Error (MAE)** and **R² Score**.

The RandomForestRegressor algorithm performs the best.
Feature Engineering has been implemented followed by feature selection technique - SelectKBest to determine the optimal combination of parameter values.
k= 15 resulted in a high accuracy level of about 85% for RandomForestRegressor, HistGradientBoostingRegressor and AdaBoostRegressor algorithms.

---

## 📈 Output

- Predicts students’ final math grades.
- Compares the effectiveness of different ensemble learning methods.
- Visual performance chart included in `reports/Models_Performance.png`.

---

## 🧪 Requirements

Install dependencies using:
pip install -r requirements.txt

# or if using conda
conda env create -f environment.yml
conda activate stats_ml

📦 Dataset Source
This project uses the maths.csv file from the Kaggle dataset:

https://www.kaggle.com/datasets/whenamancodes/alcohol-effects-on-study

The dataset is included in this project directory under data/ for convenience.