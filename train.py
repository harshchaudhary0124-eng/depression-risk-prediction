# train.py

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

from preprocessing import load_and_preprocess


def main():
    # ---------- 1. LOAD & PREPROCESS ----------
    features, labels, metadata = load_and_preprocess("Stress.csv")

    # All feature names
    feature_cols = features.columns.tolist()

    # Numeric columns to scale (exactly as in your notebook)
    numeric_cols = [
        "Age",
        "Academic Pressure",
        "Work Pressure",
        "CGPA",
        "Study Satisfaction",
        "Job Satisfaction",
        "Sleep Quality",
        "Diet Quality",
        "Work/Study Hours",
        "Financial Stress",
        "City_freq",
    ]

    # ---------- 2. TRAIN / TEST SPLIT ----------
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # ---------- 3. SCALE NUMERIC COLUMNS ----------
    scaler = StandardScaler()
    X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

    # ---------- 4. TRAIN XGBOOST ----------
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
    )

    model.fit(X_train, y_train)

    # ---------- 5. EVALUATE ----------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100

    print(f"Test Accuracy: {acc:.2f}%\n")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Feature importance (top 20)
    fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(
        ascending=False
    )
    print("\nTop 20 Features:\n", fi.head(20))

    # ---------- 6. SAVE ARTIFACTS ----------
    # model.pkl: just the XGBoost model
    joblib.dump(model, "model.pkl")

    # scaler.pkl: scaler + metadata needed at prediction time
    scaler_artifacts = {
        "scaler": scaler,
        "feature_cols": X_train.columns.tolist(),
        "numeric_cols": numeric_cols,
        "city_freq_map": metadata["city_freq_map"],
    }
    joblib.dump(scaler_artifacts, "scaler.pkl")

    print("\nSaved model.pkl and scaler.pkl")


if __name__ == "__main__":
    main()
