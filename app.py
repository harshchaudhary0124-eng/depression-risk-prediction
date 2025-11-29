import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1. Load trained model & scaler
# -----------------------------

# Change these filenames if you used different names
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"   # this can also be a full preprocessing pipeline

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# 2. Page config
# -----------------------------

st.set_page_config(
    page_title="Depression Risk Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Depression Risk Prediction System")
st.write(
    """
This app predicts whether a person is at **risk of depression** based on lifestyle 
and mental-health factors.  

> ⚠️ This is a **learning project** and **not a medical tool**.  
> Decisions should **not** be made based solely on this prediction.
"""
)

st.markdown("---")

# ---------------------------------------------
# 3. Define UI inputs (match your dataset here)
# ---------------------------------------------

st.subheader("Enter Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=80, value=20, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city = st.text_input("City (e.g., Delhi, Mumbai)")

    academic_pressure = st.slider("Academic pressure (1 = low, 5 = very high)", 1, 5, 3)
    sleep_quality = st.slider("Sleep quality (1 = poor, 5 = excellent)", 1, 5, 3)
    sleep_hours = st.slider("Average sleep per day (hours)", 0, 14, 7)

with col2:
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    job_satisfaction = st.slider("Job/Study satisfaction (1–5)", 1, 5, 3)
    diet_quality = st.slider("Diet quality (1–5)", 1, 5, 3)

    suicidal_thoughts = st.selectbox("Suicidal thoughts", ["No", "Yes"])
    family_history = st.selectbox("Family history of mental illness", ["No", "Yes"])
    profession = st.selectbox("Profession", ["Student", "Working", "Other"])


# --------------------------------------------------------
# 4. Build a raw input DataFrame (MUST match training data)
# --------------------------------------------------------

def build_raw_input_df() -> pd.DataFrame:
    """
    IMPORTANT:
    The column names and order here MUST match what you used during training
    before feeding to your preprocessing pipeline / scaler.
    Adjust them to your real feature names from preprocessing.py.
    """

    data = {
        "Age": [age],
        "Gender": [gender],
        "City": [city],
        "Academic_pressure": [academic_pressure],
        "Sleep_quality": [sleep_quality],
        "Sleep_hours": [sleep_hours],
        "CGPA": [cgpa],
        "Job_satisfaction": [job_satisfaction],
        "Diet_quality": [diet_quality],
        "Suicidal_thoughts": [suicidal_thoughts],
        "Family_history": [family_history],
        "Profession": [profession],
    }

    df = pd.DataFrame(data)

    return df


# --------------------------------------------------------
# 5. Preprocess input (apply same logic as training)
# --------------------------------------------------------

def preprocess_input(raw_df: pd.DataFrame) -> np.ndarray:
    """
    This should apply EXACTLY the same transformations you used in training:
    - encoding categorical variables
    - frequency encoding, one-hot, etc.
    - scaling numeric columns

    If your `scaler.pkl` already wraps all preprocessing (e.g., a ColumnTransformer
    or Pipeline), you can simply call: scaler.transform(raw_df).

    Otherwise, you may want to import and reuse your preprocessing functions
    from preprocessing.py.
    """

    # Example assuming `scaler` is a full pipeline:
    X_processed = scaler.transform(raw_df)
    return X_processed


# --------------------------------------------------------
# 6. Run prediction when user clicks the button
# --------------------------------------------------------

if st.button("Predict Depression Risk"):
    raw_df = build_raw_input_df()

    with st.expander("🔍 See raw input data"):
        st.write(raw_df)

    try:
        X = preprocess_input(raw_df)
        proba = model.predict_proba(X)[0][1]   # probability of class 1 (at risk)
        pred = model.predict(X)[0]

        label = "🚨 At Risk of Depression" if pred == 1 else "✅ No Significant Risk Detected"

        st.markdown("---")
        st.subheader("Prediction Result")
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Estimated risk probability:** `{proba:.2%}`")

        st.info(
            "This prediction is based on a machine learning model trained on historical survey data. "
            "It is **not** a clinical diagnosis. If you feel distressed, please consult a professional."
        )

    except Exception as e:
        st.error("⚠️ Error while making prediction. Check the preprocessing pipeline.")
        st.exception(e)

