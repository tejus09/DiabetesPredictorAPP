import streamlit as st
from pickle import load
import numpy as np

# Load the scaler and model
scaler = load(open('models/standard_scaler.pkl', 'rb'))

# Set page config
st.set_page_config(page_title="Diabetes Prediction", page_icon=":clipboard:", layout="wide")

# Change font and color
st.write(
    f"""
    <style>
    .* {{
        font-family: "Ubuntu", sans-serif;
        color: blue;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Define the input fields
st.markdown("<h1 style='text-align: center; color: red; font-weight: 900;'>Welcome to the Diabetes Prediction App!</h1>", unsafe_allow_html=True)
st.sidebar.title("Welcome to Diabetes Info")
st.sidebar.write("This app provides information about diabetes and helps you predict your risk of developing diabetes.")
st.sidebar.image('diabetes.jpg')
# What is Diabetes section
st.sidebar.header("What is Diabetes?")
st.sidebar.write("Diabetes is a chronic condition that affects how your body processes blood sugar (glucose). It occurs when your body either doesn't make enough insulin or can't use the insulin it produces effectively. There are three main types of diabetes: type 1 diabetes, type 2 diabetes, and gestational diabetes.")

# Symptoms of Diabetes section
st.sidebar.header("Symptoms of Diabetes")
st.sidebar.write("Symptoms of diabetes include frequent urination, increased thirst, unexplained weight loss, increased hunger, blurry vision, and fatigue. However, some people with diabetes may not experience any symptoms.")

# Diabetes Risk Factors section
st.sidebar.header("Diabetes Risk Factors")
st.sidebar.write("Risk factors for developing diabetes include being overweight or obese, having a family history of diabetes, being over 45 years old, being physically inactive, having high blood pressure or high cholesterol, and having a history of gestational diabetes.")

# Diabetes Prediction section
st.sidebar.header("Diabetes Prediction")
st.sidebar.write("Use the form on the right to input your information and predict your risk of developing diabetes.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Tejus Kavishwar")

with st.container():
    st.markdown("<p style='font-size: 20px; color: blue; font-weight: 900;'>Patient Information</p>", unsafe_allow_html=True)
    st.markdown("<hr style='background-color: black; height: 2px; border: 0;'>", unsafe_allow_html=True)
    left_column, right_column = st.columns(2)

    with left_column:
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'], key='gender', help='Select your gender', 
                             )
        age = st.number_input('Age', min_value=0, max_value=120, step=1, key='age', help='Enter your age in years')
        hypertension = st.selectbox('Hypertension', ['No', 'Yes'], key='hypertension', help='Do you have hypertension?')
        heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'], key='heart_disease', help='Do you have any history of heart disease?')

    with right_column:
        smoking_history = st.selectbox('Smoking History', ['No Info', 'Never', 'Former', 'Current', 'Ever'], key='smoking_history', help='Select your smoking history')
        bmi = st.number_input('BMI', min_value=10.0, max_value=80.0, step=0.1, key='bmi', help='Enter your Body Mass Index (BMI)')
        HbA1c_level = st.number_input('HbA1c Level', min_value=0.0, max_value=20.0, step=0.1, key='hba1c_level', help='Enter your HbA1c level')
        blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0, max_value=500, step=1, key='blood_glucose_level', help='Enter your blood glucose level')

with st.container():
    st.markdown("<hr style='background-color: black; height: 2px; border: 0;'>", unsafe_allow_html=True)
select_model = st.selectbox("Choose Model", ['Logistic Regression', 'Decision Tree', 'K Nearest Neighbor', 'Naive Bayes', "Support Vector Classifier"])

if select_model == 'Logistic Regression':
    model = load(open('models/lr_model.pkl', 'rb'))
elif select_model == 'Decision Tree':
    model = load(open('models/dt_model.pkl', 'rb'))
elif select_model == 'K Nearest Neighbor':
    model = load(open('models/knn_model.pkl', 'rb'))
elif select_model == 'Naive Bayes':
    model = load(open('models/nb_model.pkl', 'rb'))
elif select_model == 'Support Vector Classifier':
    model = load(open('models/sv_model.pkl', 'rb'))
# Define the predict function
def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level):
    # Map categorical inputs to numeric values
    if gender == 'Male':
        gender_numeric = 1
    elif gender == 'Female':
        gender_numeric = 0
    else:
        gender_numeric = np.nan

    if hypertension == 'Yes':
        hypertension_numeric = 1
    else:
        hypertension_numeric = 0

    if heart_disease == 'Yes':
        heart_disease_numeric = 1
    else:
        heart_disease_numeric = 0

    if smoking_history == 'Never':
        smoking_history_numeric = 0
    elif smoking_history == 'Former':
        smoking_history_numeric = 1
    elif smoking_history == 'Current' or smoking_history == 'Ever':
        smoking_history_numeric = 2
    else:
        smoking_history_numeric = -1

    # Check if all inputs are valid
    if np.isnan([gender_numeric, age, hypertension_numeric, heart_disease_numeric, smoking_history_numeric, bmi, HbA1c_level, blood_glucose_level]).any():
        st.error('Please fill in all fields with valid input')
        return

    # Transform input with scaler
    query_point_transformed = scaler.transform([[gender_numeric, age, hypertension_numeric, heart_disease_numeric, smoking_history_numeric, bmi, HbA1c_level, blood_glucose_level]])

    # Make prediction with model
    pred = model.predict(query_point_transformed)

    # Return prediction
    st.success(f'{pred[0]}')

# Add predict button
if st.button('Predict'):
    predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level)
