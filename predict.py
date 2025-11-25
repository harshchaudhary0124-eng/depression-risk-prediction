# predict.py

import joblib
import numpy as np
import pandas as pd


def ask_float(prompt: str) -> float:
    """Ask repeatedly until user provides a valid float."""
    while True:
        s = input(prompt).strip()
        try:
            return float(s)
        except ValueError:
            print("Please enter a numeric value (e.g. 20 or 3.5).")


def build_user_feature_row(feature_cols, numeric_cols, city_freq_map, scaler):
    """
    Interactively ask the user for inputs and build a 1-row DataFrame
    with the same columns and scaling as used during training.
    """

    # ----- 1. NUMERIC INPUTS -----
    age = ask_float("Age: ")
    acad_press = ask_float("Academic Pressure (1–5): ")
    work_press = ask_float("Work Pressure (1–5): ")
    cgpa = ask_float("CGPA (e.g. 7.5): ")
    study_sat = ask_float("Study Satisfaction (1–5): ")
    job_sat = ask_float("Job Satisfaction (1–5): ")
    sleep_q = ask_float("Sleep Quality (1–5): ")
    diet_q = ask_float("Diet Quality (1–5): ")
    hours = ask_float("Work/Study Hours per day: ")
    fin_stress = ask_float("Financial Stress (1–5): ")

    # ----- 2. BINARY / CATEGORICAL INPUTS -----
    gender_in = input("Gender (Male/Female): ").strip().title()
    suicidal_in = input("Ever had suicidal thoughts? (Yes/No): ").strip().title()
    family_in = input("Family history of mental illness? (Yes/No): ").strip().title()

    profession_in = input("Profession (Student / other): ").strip().title()
    is_student = 1 if profession_in == "Student" else 0

    print(
        "\nEnter degree exactly like in your data (e.g. B.Tech, B.Com, Class 12, MSc, etc.)"
    )
    degree_in = input("Degree: ").strip()
    degree_dummy = f"Degree_{degree_in}"  # e.g. Degree_B.Tech

    city_input = input("City (e.g. Lucknow, Kalyan, etc.): ").strip()

    # Map city to its training frequency; if unseen, use median frequency
    if city_freq_map is not None and len(city_freq_map) > 0:
        if city_input in city_freq_map.index:
            city_freq_value = city_freq_map[city_input]
        else:
            city_freq_value = city_freq_map.median()
    else:
        city_freq_value = 0.0

    # ----- 3. CREATE 1×N ROW WITH ALL FEATURE COLUMNS -----
    row = pd.DataFrame(0, index=[0], columns=feature_cols)

    # ----- 4. FILL RAW NUMERIC VALUES (BEFORE SCALING) -----
    row.loc[0, "Age"] = age
    row.loc[0, "Academic Pressure"] = acad_press
    row.loc[0, "Work Pressure"] = work_press
    row.loc[0, "CGPA"] = cgpa
    row.loc[0, "Study Satisfaction"] = study_sat
    row.loc[0, "Job Satisfaction"] = job_sat
    row.loc[0, "Sleep Quality"] = sleep_q
    row.loc[0, "Diet Quality"] = diet_q
    row.loc[0, "Work/Study Hours"] = hours
    row.loc[0, "Financial Stress"] = fin_stress
    if "City_freq" in row.columns:
        row.loc[0, "City_freq"] = city_freq_value

    # ----- 5. FILL BINARY / CATEGORICAL ENCODINGS -----
    if "Gender" in row.columns:
        row.loc[0, "Gender"] = 1 if gender_in == "Male" else 0

    suic_col = "Have you ever had suicidal thoughts ?"
    if suic_col in row.columns:
        row.loc[0, suic_col] = 1 if suicidal_in == "Yes" else 0

    fam_col = "Family History of Mental Illness"
    if fam_col in row.columns:
        row.loc[0, fam_col] = 1 if family_in == "Yes" else 0

    if "Profession_Student" in row.columns:
        row.loc[0, "Profession_Student"] = is_student
    if "Profession_Working" in row.columns:
        row.loc[0, "Profession_Working"] = 1 - is_student

    # Degree dummy
    if degree_dummy in row.columns:
        row.loc[0, degree_dummy] = 1
    else:
        print(
            f"⚠️ Warning: degree dummy '{degree_dummy}' not found in training features."
        )

    # ----- 6. SCALE NUMERIC COLUMNS USING TRAINED SCALER -----
    row[numeric_cols] = scaler.transform(row[numeric_cols])

    return row


def main():
    # ---------- 1. LOAD MODEL & SCALER ARTIFACTS ----------
    model = joblib.load("model.pkl")
    scaler_artifacts = joblib.load("scaler.pkl")

    scaler = scaler_artifacts["scaler"]
    feature_cols = scaler_artifacts["feature_cols"]
    numeric_cols = scaler_artifacts["numeric_cols"]
    city_freq_map = scaler_artifacts["city_freq_map"]

    # ---------- 2. BUILD USER FEATURE ROW ----------
    user_row = build_user_feature_row(feature_cols, numeric_cols, city_freq_map, scaler)

    # ---------- 3. PREDICT ----------
    pred_label = model.predict(user_row)[0]
    pred_prob = model.predict_proba(user_row)[0, 1]

    # ---------- 4. DISPLAY RESULT ----------
    if pred_label == 1:
        print(f"\n Prediction: DEPRESSION (probability = {pred_prob:.2f})")
    else:
        print(f"\n Prediction: NO DEPRESSION (probability = {pred_prob:.2f})")


if __name__ == "__main__":
    main()
