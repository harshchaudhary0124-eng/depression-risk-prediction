<<<<<<< HEAD
# Depression / Stress Prediction using XGBoost

This project builds an end-to-end machine learning pipeline to predict **depression / stress risk** using a real-world mental health dataset (`Stress.csv`).

It includes:

- Full **data preprocessing & cleaning**
- Advanced **feature engineering**
- Training an **XGBoost classifier**
- An **interactive CLI predictor** where a user can input their details and get a prediction: `Depression` or `No Depression`.

---

## 🚀 Features

### 1. Data Preprocessing (`preprocessing.py`)

- Handles missing values using `SimpleImputer` (median strategy for `Financial Stress` and `City_freq`).
- Converts `Sleep Duration` and `Dietary Habits` to numeric scores and renames them to:
  - `Sleep Quality`
  - `Diet Quality`
- Encodes multiple categorical variables:
  - `Gender` → 0/1
  - `Have you ever had suicidal thoughts ?` → 0/1
  - `Family History of Mental Illness` → 0/1
  - `Profession` → `Profession_Student` and `Profession_Working`
  - `Degree` → one-hot encoded (e.g., `Degree_B.Tech`, `Degree_B.Com`, etc.)
- Cleans `City`, removes noisy values, and introduces:
  - `City_freq`: a **frequency encoding** of cities
- Splits final data into:
  - `features` (X)
  - `labels` (y = `Depression`)

### 2. Model Training (`train.py`)

- Uses `train_test_split` with `stratify=labels`.
- Scales numeric columns using `StandardScaler`:
  - `Age`, `Academic Pressure`, `Work Pressure`, `CGPA`, `Study Satisfaction`,
    `Job Satisfaction`, `Sleep Quality`, `Diet Quality`, `Work/Study Hours`,
    `Financial Stress`, `City_freq`
- Trains an `XGBClassifier` with:
  - `scale_pos_weight` to handle class imbalance
  - 300 estimators, depth=4, learning rate=0.05
- Evaluates performance with:
  - Accuracy
  - Confusion matrix
  - Classification report
- Saves:
  - `model.pkl` – trained XGBoost model
  - `scaler.pkl` – `StandardScaler` + feature metadata

### 3. Interactive Prediction (`predict.py`)

- Loads `model.pkl` and `scaler.pkl`.
- Asks the user for inputs:
  - Numeric: age, CGPA, stress levels, sleep quality, diet quality, etc.
  - Categorical: gender, suicidal thoughts, family history, profession, degree, city.
- Rebuilds a **1-row feature vector** matching the training columns.
- Scales numeric features with the same scaler as training.
- Calls `model.predict()` and `model.predict_proba()` to output:
  - `Depression` or `No Depression`
  - Probability of depression.

---

## 📂 Project Structure

```text
.
├── Stress.csv
├── preprocessing.py
├── train.py
├── predict.py
├── model.pkl          # created after running train.py
├── scaler.pkl         # created after running train.py
└── README.md
=======
# depression-risk-prediction
Machine learning model using XGBoost to predict depression risk with preprocessing, feature engineering, and interactive CLI predictor.
>>>>>>> cdaa3a715258ddddcaba7f986ee3938d48d0ded3
