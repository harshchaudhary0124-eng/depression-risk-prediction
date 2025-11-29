# streamlit_app.py

import joblib
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_artifacts():
    """
    Load trained XGBoost model and scaler + metadata artifacts.
    These are created by train.py (model.pkl and scaler.pkl).
    """
    model = joblib.load("model.pkl")
    scaler_artifacts = joblib.load("scaler.pkl")

    scaler = scaler_artifacts["scaler"]
    feature_cols = scaler_artifacts["feature_cols"]
    numeric_cols = scaler_artifacts["numeric_cols"]
    city_freq_map = scaler_artifacts["city_freq_map"]

    degree_cols = [c for c in feature_cols if c.startswith("Degree_")]
    degree_options = [c.replace("Degree_", "") for c in degree_cols]

    return model, scaler, feature_cols, numeric_cols, city_freq_map, degree_options


def build_feature_row_streamlit(
    feature_cols,
    numeric_cols,
    city_freq_map,
    scaler,
    inputs,
):

    # Creating 1xN row filled with zeros for all features
    row = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Fill numeric features
    row.loc[0, "Age"] = inputs["age"]
    row.loc[0, "Academic Pressure"] = inputs["acad_press"]
    row.loc[0, "Work Pressure"] = inputs["work_press"]
    row.loc[0, "CGPA"] = inputs["cgpa"]
    row.loc[0, "Study Satisfaction"] = inputs["study_sat"]
    row.loc[0, "Job Satisfaction"] = inputs["job_sat"]
    row.loc[0, "Sleep Quality"] = inputs["sleep_q"]
    row.loc[0, "Diet Quality"] = inputs["diet_q"]
    row.loc[0, "Work/Study Hours"] = inputs["hours"]
    row.loc[0, "Financial Stress"] = inputs["fin_stress"]

    # City frequency
    if "City_freq" in row.columns:
        if city_freq_map is not None and len(city_freq_map) > 0:
            if inputs["city_choice"] == "Not in list / Prefer not to say":
                city_freq_value = city_freq_map.median()
            else:
                # Selected from list, must exist in map
                city_freq_value = city_freq_map[inputs["city_choice"]]
        else:
            city_freq_value = 0.0

        row.loc[0, "City_freq"] = city_freq_value

    # Fill binary / categorical encodings

    # Gender
    if "Gender" in row.columns:
        row.loc[0, "Gender"] = 1 if inputs["gender"] == "Male" else 0

    # Suicidal thoughts
    suic_col = "Have you ever had suicidal thoughts ?"
    if suic_col in row.columns:
        row.loc[0, suic_col] = 1 if inputs["suicidal"] == "Yes" else 0

    # Family history of mental illness
    fam_col = "Family History of Mental Illness"
    if fam_col in row.columns:
        row.loc[0, fam_col] = 1 if inputs["family_history"] == "Yes" else 0

    # Profession: Student vs Working
    if "Profession_Student" in row.columns:
        is_student = 1 if inputs["profession"] == "Student" else 0
        row.loc[0, "Profession_Student"] = is_student
    if "Profession_Working" in row.columns:
        # Complement of Student
        is_student = 1 if inputs["profession"] == "Student" else 0
        row.loc[0, "Profession_Working"] = 1 - is_student

    # Degree dummy
    degree_dummy = f"Degree_{inputs['degree']}"
    if degree_dummy in row.columns:
        row.loc[0, degree_dummy] = 1
    else:
        # If user-selected degree isn't in training features,
        # we simply keep all degree dummies at 0.
        st.info(
            f"The selected degree '{inputs['degree']}' was not seen during training. "
            "The model will still make a prediction, but it may be slightly less accurate."
        )

    # Scaling numeric columns using trained scaler
    row[numeric_cols] = scaler.transform(row[numeric_cols])

    return row


def main():
    st.set_page_config(page_title="Depression & Stress Tracker", page_icon="🧠", layout="centered")

    st.title("Depression & Stress Tracker")
    st.write(
        """
        This app uses an XGBoost model trained on a dataset of 27,902 observations recorded all over India   
        to estimate whether a person is likely to be **depressed** or **not depressed**  
        based on lifestyle, academic/work, city(pollution index) and mental health factors.
        """
    )

    # Load artifacts
    try:
        model, scaler, feature_cols, numeric_cols, city_freq_map, degree_options = load_artifacts()
    except Exception as e:
        st.error(
            "Error loading model/scaler artifacts. "
            "Make sure `model.pkl` and `scaler.pkl` are in the same folder as this file."
        )
        st.exception(e)
        return

    # Sidebar info
    st.sidebar.header("About the Model")
    st.sidebar.markdown(
        """
        - Trained using **XGBoostClassifier**
        - Numeric features scaled with **StandardScaler**
        - City encoded using **frequency encoding (City_freq)**  
        - Binary and categorical features encoded as in `preprocessing.py`
        """
    )

    if city_freq_map is not None and len(city_freq_map) > 0:
        city_options = sorted(list(city_freq_map.index))
        city_options.append("Not in list / Prefer not to say")
    else:
        city_options = ["Not in list / Prefer not to say"]

    if not degree_options:
        # Fallback if for any reason no degree_* columns found
        degree_options = ["B.Tech", "B.Com", "B.Sc", "B.A", "Class 12", "Other"]

    st.subheader("Enter your details")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=10, max_value=80, value=20, step=1)
            acad_press = st.slider("Academic Pressure (1–5)", 1.0, 5.0, 3.0, step=1.0)
            work_press = st.slider("Work Pressure (1–5)", 1.0, 5.0, 2.0, step=1.0)
            cgpa = st.number_input("CGPA (0–10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            study_sat = st.slider("Study Satisfaction (1–5)", 1.0, 5.0, 3.0, step=1.0)
            job_sat = st.slider("Job Satisfaction (1–5)", 1.0, 5.0, 3.0, step=1.0)

        with col2:
            sleep_q = st.slider("Sleep Quality (1–5)", 1.0, 5.0, 3.0, step=1.0)
            diet_q = st.slider("Diet Quality (1–5)", 1.0, 5.0, 2.0, step=1.0)
            hours = st.number_input(
                "Work/Study Hours per day",
                min_value=0.0,
                max_value=24.0,
                value=6.0,
                step=0.5,
            )
            fin_stress = st.slider("Financial Stress (1–5)", 1.0, 5.0, 2.0, step=1.0)

            gender = st.selectbox("Gender", ["Male", "Female"])
            suicidal = st.selectbox("Ever had suicidal thoughts?", ["No", "Yes"])
            family_history = st.selectbox("Family history of mental illness?", ["No", "Yes"])

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            profession = st.selectbox("Profession", ["Student", "Working"])

        with col4:
            degree = st.selectbox("Degree (as in training data)", degree_options)

        city_choice = st.selectbox("City", city_options)

        submitted = st.form_submit_button("Predict Stress / Depression")

    if submitted:
        # Collect all inputs in a dict
        inputs = {
            "age": age,
            "acad_press": acad_press,
            "work_press": work_press,
            "cgpa": cgpa,
            "study_sat": study_sat,
            "job_sat": job_sat,
            "sleep_q": sleep_q,
            "diet_q": diet_q,
            "hours": hours,
            "fin_stress": fin_stress,
            "gender": gender,
            "suicidal": suicidal,
            "family_history": family_history,
            "profession": profession,
            "degree": degree,
            "city_choice": city_choice,
        }

        # Build feature row and predict
        try:
            user_row = build_feature_row_streamlit(
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                city_freq_map=city_freq_map,
                scaler=scaler,
                inputs=inputs,
            )

            pred_label = model.predict(user_row)[0]
            pred_prob = model.predict_proba(user_row)[0, 1]  # probability of class 1

            st.markdown("## Prediction Result")

            if pred_label == 1:
                st.error(
                    f"Based on the data you entered, the model predicts that you are **likely DEPRESSED**.\n\n"
                    f"**Stress / depression probability:** `{pred_prob:.2f}`"
                )
            else:
                st.success(
                    f"Based on the data you entered, the model predicts that you are **NOT DEPRESSED**.\n\n"
                    f"**Stress / depression probability:** `{pred_prob:.2f}`"
                )

            with st.expander("See the processed feature row (debug / explainability)"):
                st.dataframe(user_row)

        except Exception as e:
            st.error("There was an error while building features or making prediction.")
            st.exception(e)


if __name__ == "__main__":
    main()
