import streamlit as st
import os
import sys
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_model


st.set_page_config(
    page_title='Student Performance Predictor',
    page_icon='🎓',
    layout='wide'
)

@st.cache_resource
def load_artifacts():
    transformer = load_model('models/col_transformer_fitted.joblib')
    model = load_model('models/rfr_fitted_with_best_params.joblib')
    best_features = load_model('models/grid_search_best_estimator_features.joblib')

    return transformer, model, best_features


col_transformer, model, best_features = load_artifacts()


st.title('Student Final Grade Predictor')
st.markdown("""
            This application predicts a student's final grade (G3) based on their demographic, social, and school-related information. 
            Please provide the student's details using the input fields on the left.
            """)

st.sidebar.header('Input Student Details')

def get_user_input():
    """
    Creates sidebar widgets to get user input for all necessary features.
    The keys in the returned dictionary must match the column names your model
    was trained on before preprocessing.
    """
    job_options = ['teacher', 'other', 'services', 'health', 'at_home']
    support_options = ['yes', 'no']
    binary_options = ['yes', 'no']
    locality_options = ['U', 'R']
    family_size_options = ['LE3', 'GT3']
    cohab_status_options = ['T', 'A'] 

    age = st.sidebar. slider('Age', 13, 22, 17)
    Medu = st.sidebar.slider("Mother's Education (0-4)", 0, 4, 3)
    Fedu = st.sidebar.slider("Father's Education (0-4)", 0, 4, 3)
    Mother_Job = st.sidebar.selectbox("Mother's Job", job_options)
    Father_Job = st.sidebar.selectbox("Father's Job", job_options)
    Home_to_School_Travel_Time = st.sidebar.slider("Travel Time (1-4)", 1, 4, 2)
    Weekly_Study_Time = st.sidebar.slider('Weekly Study Time (1-4)', 1, 4, 2)
    Past_Class_Failure_Count = st.sidebar.slider('Number of Past Failures', 0, 4, 0)
    School_Support = st.sidebar.selectbox('Extra Educational Support', support_options)
    Family_Support = st.sidebar.selectbox('Family Educational Support', support_options)
    Extra_Paid_Classes = st.sidebar.selectbox('Paid Classes', binary_options)
    Extra_Curr_Activities = st.sidebar.selectbox('Extra-curricular Activities', binary_options)
    Attended_Kindergarten = st.sidebar.selectbox('Attended Nursery/Kindergarten', binary_options)
    Higher_Edu = st.sidebar.selectbox('Wants to take higher education', binary_options)
    Internet = st.sidebar.selectbox('Internet access at home', binary_options)
    Dating = st.sidebar.selectbox('In a romantic relationship', binary_options)
    Family_Relationship = st.sidebar.slider('Quality of Family Relationships (1-5)', 1, 5, 4)
    Freetime_After_School = st.sidebar.slider('Free Time After School (1-5)', 1, 5, 3)
    Goes_Out = st.sidebar.slider('Going out with friends (1-5)', 1, 5, 3)
    Dalc = st.sidebar.slider('Workday Alcohol Consumption (1-5)', 1, 5, 1)
    Walc = st.sidebar.slider('Weekend Alcohol Consumption (1-5)', 1, 5, 1)
    Current_Health_Status = st.sidebar.slider('Current Health Status (1-5)', 1, 5, 4)
    School_Absences = st.sidebar.slider('NUmber of School Absences', 0, 93, 4)
    G1 = st.sidebar.slider('First Period Grade (0-20)', 0, 20, 10)
    G2 = st.sidebar.slider('Second Period Grade (0-20)', 0, 20, 10)
    Locality = st.sidebar.selectbox('Home Address Type', locality_options)
    Family_Size = st.sidebar.selectbox('Family Size', family_size_options)
    Parents_Cohab_Status = st.sidebar.selectbox('Parents Cohabitation Status', cohab_status_options)

    user_data = {
        'age': age,
        'Medu': Medu,
        'Fedu': Fedu,
        'Mjob': Mother_Job,
        'Fjob': Father_Job,
        'traveltime': Home_to_School_Travel_Time,
        'studytime': Weekly_Study_Time,
        'failures': Past_Class_Failure_Count,
        'schoolsup': School_Support,
        'famsup': Family_Support,
        'paid': Extra_Paid_Classes,
        'activities': Extra_Curr_Activities,
        'nursery': Attended_Kindergarten,
        'higher': Higher_Edu,
        'internet': Internet,
        'romantic': Dating,
        'famrel': Family_Relationship,
        'freetime': Freetime_After_School,
        'goout': Goes_Out,
        'Dalc': Dalc,
        'Walc': Walc,
        'health': Current_Health_Status,
        'absences': School_Absences,
        'G1': G1,
        'G2': G2,
        'address': Locality,
        'famsize': Family_Size,
        'Pstatus': Parents_Cohab_Status
    }

    features = pd.DataFrame(user_data, index=[0])

    return features


input_df = get_user_input()


st.header('Prediction')

if st.sidebar.button('Predict Final Grade'):
    try:
        processed_df = input_df.copy()
        processed_df['Past_Performance_Grade'] = (processed_df['G1'] + processed_df['G2']) / 2
        processed_df['Parents_education'] = (processed_df['Medu'] + processed_df['Fedu']) / 2
        processed_df['Alcohol_Consumption'] = (processed_df['Dalc'] + processed_df['Walc'])

        processed_df.drop(['G1', 'G2', 'Medu', 'Fedu', 'Dalc', 'Walc'], axis=1, inplace=True)

        column_mapping = {
            'age': 'Age', 'address': 'Locality', 'famsize': 'Family_Size', 
            'Pstatus': 'Parents_Cohab_Status', 'Mjob': 'Mother_Job', 'Fjob': 'Father_Job',
            'traveltime': 'Home_to_School_Travel_Time', 'studytime': 'Weekly_Study_Time',
            'failures': 'Past_Class_Failure_Count', 'schoolsup': 'School_Support',
            'famsup': 'Family_Support', 'paid': 'Extra_Paid_Classes', 'activities': 'Extra_Curr_Activities',
            'nursery': 'Attended_Kindergarten', 'higher': 'Higher_Edu', 'internet': 'Internet',
            'romantic': 'Dating', 'famrel': 'Family_Relationship', 'freetime': 'Freetime_After_School',
            'goout': 'Goes_Out', 'health': 'Current_Health_Status', 'absences': 'School_Absences',
            'Past_Performance_Grade': 'Past_Grade_Record', 'Parents_education': 'Parents_Education',
            'Alcohol_Consumption': 'Alcohol_Consumption'
        }

        processed_df.rename(columns=column_mapping, inplace=True)
        transformed_df = col_transformer.transform(processed_df)
        final_features = transformed_df[best_features]

        prediction = model.predict(final_features)
        predicted_grade = prediction[0]

        st.success(f"**Predicted Final Grade (G3): {predicted_grade:.2f} / 20**")

        st.progress(predicted_grade / 20.0)
        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure all inputs are correct. The column names or data types might not match the model's expectations.")

else:
    st.info('Click the "Predict Final Grade" button in the sidebar to see the result.')




