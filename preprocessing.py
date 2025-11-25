# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def load_and_preprocess(csv_path: str = "Stress.csv"):
    """
    Load the Stress.csv dataset, clean it, encode categorical variables,
    create City_freq, and return:
        - features_df: processed feature DataFrame (unscaled)
        - labels: target Series (Depression)
        - metadata: dictionary with helper objects (e.g., city_freq_map)
    """

    # ---------- 1. LOAD RAW DATA ----------
    df = pd.read_csv(csv_path)

    # ---------- 2. HANDLE MISSING VALUES ----------
    # Financial Stress: impute median
    imputer = SimpleImputer(strategy="median")
    df[["Financial Stress"]] = imputer.fit_transform(df[["Financial Stress"]])

    # ---------- 3. TRANSFORM SLEEP DURATION & DIETARY HABITS ----------
    # Sleep Duration
    df["Sleep Duration"] = df["Sleep Duration"].replace(
        {
            "Less than 5 hours": 1.0,
            "5-6 hours": 2.0,
            "7-8 hours": 3.0,
            "More than 8 hours": 4.0,
        }
    )
    df["Sleep Duration"] = df["Sleep Duration"].replace("Others", np.nan)

    # Dietary Habits
    df["Dietary Habits"] = df["Dietary Habits"].replace(
        {"Unhealthy": 1.0, "Moderate": 2.0, "Healthy": 3.0}
    )
    df["Dietary Habits"] = df["Dietary Habits"].replace("Others", np.nan)

    # Replace remaining NaNs in these columns with approximate means
    df["Sleep Duration"] = df["Sleep Duration"].replace(np.nan, 2.3)
    df["Dietary Habits"] = df["Dietary Habits"].replace(np.nan, 1.9)

    # Rename to Sleep Quality / Diet Quality
    df.rename(columns={"Sleep Duration": "Sleep Quality"}, inplace=True)
    df.rename(columns={"Dietary Habits": "Diet Quality"}, inplace=True)

    # ---------- 4. COPY & CLEAN COLUMN NAMES ----------
    df_final = df.copy()
    df_final.columns = df_final.columns.str.strip()

    # ---------- 5. ENCODE BINARY CATEGORICAL FEATURES ----------
    # Gender
    if "Gender" in df_final.columns:
        df_final["Gender"] = df_final["Gender"].map({"Male": 1, "Female": 0})

    # Suicidal thoughts
    suic_col = "Have you ever had suicidal thoughts ?"
    if suic_col in df_final.columns:
        df_final[suic_col] = df_final[suic_col].map({"Yes": 1, "No": 0})

    # Family history of mental illness
    fam_col = "Family History of Mental Illness"
    if fam_col in df_final.columns:
        df_final[fam_col] = df_final[fam_col].map({"Yes": 1, "No": 0})

    # ---------- 6. PROFESSION → STUDENT / WORKING ----------
    if "Profession" in df_final.columns:
        df_final["Profession_clean"] = df_final["Profession"].apply(
            lambda x: "Student" if x == "Student" else "Working"
        )
        df_final["Profession_Student"] = (
            df_final["Profession_clean"] == "Student"
        ).astype(int)
        df_final["Profession_Working"] = (
            df_final["Profession_clean"] == "Working"
        ).astype(int)
        df_final.drop(["Profession", "Profession_clean"], axis=1, inplace=True)

    # ---------- 7. DEGREE → ONE-HOT ENCODING ----------
    if "Degree" in df_final.columns:
        df_final = pd.get_dummies(df_final, columns=["Degree"], drop_first=True)

    # ---------- 8. CITY CLEANING + FREQUENCY ENCODING ----------
    if "City" in df_final.columns:
        # Fix obvious typos
        df_final["City"] = df_final["City"].replace(
            {
                "Nalyan": "Kalyan",
                "Khaziabad": "Ghaziabad",
                "Less Delhi": "Delhi",
                "Less than 5 Kalyan": "Kalyan",
            }
        )

        invalid_cities = [
            "Saanvi",
            "Bhavna",
            "City",
            "Harsha",
            "Vaanya",
            "Gaurav",
            "Harsh",
            "Reyansh",
            "Kibara",
            "Rashi",
            "Mira",
            "Nalini",
            "Nandini",
            "M.Tech",
            "ME",
            "M.Com",
            "3.0",
        ]
        df_final.loc[df_final["City"].isin(invalid_cities), "City"] = np.nan

        # Build frequency map from CLEANED City
        city_freq_map = df_final["City"].value_counts()

        # Map to frequency
        df_final["City_freq"] = df_final["City"].map(city_freq_map)

        # Drop original City
        df_final.drop("City", axis=1, inplace=True)
    else:
        city_freq_map = pd.Series(dtype="int64")
        df_final["City_freq"] = 0.0

    # ---------- 9. CONVERT BOOL TO INT ----------
    bool_cols = df_final.select_dtypes(include=["bool"]).columns
    df_final[bool_cols] = df_final[bool_cols].astype(int)

    # ---------- 10. DROP ID COLUMN IF PRESENT ----------
    if "id" in df_final.columns:
        df_final = df_final.drop("id", axis=1)

    # ---------- 11. IMPUTE City_freq MISSING WITH MEDIAN ----------
    imputer_city = SimpleImputer(strategy="median")
    df_final[["City_freq"]] = imputer_city.fit_transform(df_final[["City_freq"]])

    # ---------- 12. SPLIT FEATURES & LABELS ----------
    labels = df_final["Depression"].copy()
    features = df_final.drop("Depression", axis=1).copy()

    metadata = {
        "city_freq_map": city_freq_map,
    }

    return features, labels, metadata
