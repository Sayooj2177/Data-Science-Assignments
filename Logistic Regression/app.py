import streamlit as st
import pandas as pd
import joblib


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------

model = joblib.load("logistic_model.pkl")


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title("🩺 Diabetes Prediction using Logistic Regression")

st.write(
    "Enter the patient's medical information below "
    "to predict the likelihood of diabetes."
)


# ------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------

st.subheader("Patient Information")

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)

glucose = st.number_input(
    "Glucose",
    min_value=0.0,
    max_value=300.0,
    value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

insulin = st.number_input(
    "Insulin",
    min_value=0.0,
    max_value=900.0,
    value=80.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30,
    step=1
)


# ------------------------------------------------------------
# CREATE INPUT DATAFRAME
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

if st.button("Predict Diabetes"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Likely Diabetic")
    else:
        st.success("Prediction: Not Diabetic")

    st.write(
        f"Diabetes Probability: **{probability * 100:.2f}%**"
    )

    st.progress(float(probability))

    st.info(
        "This application is for educational purposes only "
        "and should not be used as a medical diagnosis."
    )