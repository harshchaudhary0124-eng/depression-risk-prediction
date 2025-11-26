# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

"""
    Here we are loading the Stress.csv dataset , cleaning it , encoding categorical variables , creating City_freq and then returning:
          features_df: which is the processed feature DataFrame
          labels: target Series (Depression)
          metadata: dictionary with helper objects
    """
    
def load_and_preprocess(csv_path: str = "Stress.csv"):   

    # LOADING RAW DATA
    
    df = pd.read_csv(csv_path)

    # HANDLING MISSING VALUES 
    
    # We see that Financial Stress has 3 missing or Null values , therefore using impute median
    imputer = SimpleImputer(strategy="median")
    df[["Financial Stress"]] = imputer.fit_transform(df[["Financial Stress"]])

    # TRANSFORMING SLEEP DURATION & DIETARY HABITS
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

    # Replacing remaining NaNs in these columns with approximate means
    df["Sleep Duration"] = df["Sleep Duration"].replace(np.nan, 2.3)
    df["Dietary Habits"] = df["Dietary Habits"].replace(np.nan, 1.9)

    # Renaming to Sleep Quality and Diet Quality respectively
    df.rename(columns={"Sleep Duration": "Sleep Quality"}, inplace=True)
    df.rename(columns={"Dietary Habits": "Diet Quality"}, inplace=True)

    # COPYING AND CLEANING COLUMN NAMES
    df_final = df.copy()
    df_final.columns = df_final.columns.str.strip()

    # ENCODING THE BINARY CATEGORICAL FEATURES
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

    # Since Profession column contains mostly StudentS therfore dividing PROFESSION with either STUDENT OR WORKING
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

    # PROCESSING DEGREE WITH ONE-HOT ENCODING
    if "Degree" in df_final.columns:
        df_final = pd.get_dummies(df_final, columns=["Degree"], drop_first=True)

    # CLEANING CITY COLUMN WITH FREQUENCY ENCODING
    if "City" in df_final.columns:
        # FixING typos
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

        # Building frequency map from CLEANED City
        city_freq_map = df_final["City"].value_counts()

        # Mapping to frequency
        df_final["City_freq"] = df_final["City"].map(city_freq_map)

        # Drop original City
        df_final.drop("City", axis=1, inplace=True)
    else:
        city_freq_map = pd.Series(dtype="int64")
        df_final["City_freq"] = 0.0

    # CONVERTING BOOLEAN VALUES TO INT TYPE
    bool_cols = df_final.select_dtypes(include=["bool"]).columns
    df_final[bool_cols] = df_final[bool_cols].astype(int)

    # DROPING ID COLUMN
    if "id" in df_final.columns:
        df_final = df_final.drop("id", axis=1)

    # IMPUTING City_freq MISSING WITH MEDIAN
    imputer_city = SimpleImputer(strategy="median")
    df_final[["City_freq"]] = imputer_city.fit_transform(df_final[["City_freq"]])

    # SPLITING FEATURES & LABELS
    labels = df_final["Depression"].copy()
    features = df_final.drop("Depression", axis=1).copy()

    metadata = {
        "city_freq_map": city_freq_map,
    }

    return features, labels, metadata
