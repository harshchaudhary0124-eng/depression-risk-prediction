import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")   # or whatever you named it

st.title("Depression Risk Prediction System")
st.write("This app predicts whether a person is at risk of depression based on lifestyle and mental health features.")

# Example input fields – adjust to your real features
age = st.number_input("Age", min_value=10, max_value=80, value=20)
gender = st.selectbox("Gender", ["Male", "Female"])
city = st.text_input("City")
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0)
sleep_hours = st.slider("Average sleep (hours)", 0, 12, 7)
academic_pressure = st.slider("Academic pressure (1–5)", 1, 5, 3)
suicidal_thoughts = st.selectbox("Suicidal thoughts", ["No", "Yes"])

if st.button("Predict"):
    # Build a single-row DataFrame – must match your training preprocessing
    raw_input = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "City": city,
        "CGPA": cgpa,
        "Sleep_hrs": sleep_hours,
        "Academic_pressure": academic_pressure,
        "Suicidal_thoughts": suicidal_thoughts
        # ... add rest of your features here
    }])

    # Apply same preprocessing as in train/predict script
    # e.g. encoding, scaling etc.
    # For now I'm assuming your scaler can transform directly:
    X_scaled = scaler.transform(raw_input)

    proba = model.predict_proba(X_scaled)[0][1]
    pred = model.predict(X_scaled)[0]

    label = "At Risk of Depression" if pred == 1 else "No Significant Risk"
    st.subheader(f"Prediction: {label}")
    st.write(f"Risk probability: **{proba:.2%}**")
