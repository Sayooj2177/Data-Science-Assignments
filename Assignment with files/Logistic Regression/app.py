import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page title
st.title("Diabetes Prediction App")
st.write("Enter the patient's details to predict the likelihood of diabetes.")

# User inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction button
if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Scale the input using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get probability
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error("Prediction: Diabetes")
    else:
        st.success("Prediction: No Diabetes")

    st.write(f"Diabetes Probability: {probability:.2%}")

