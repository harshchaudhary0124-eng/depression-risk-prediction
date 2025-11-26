# Depression Risk Prediction System (Machine Learning)

  Hello everyone . 
  Welcome to this project that predicts whether a person is at risk of depression using a full machine learning pipeline.  
It includes **data preprocessing**, **feature engineering**, **XGBoost training**, and a **real-time user prediction system** that stores each user's input in an Excel file.



## Features
- Complete data cleaning & preprocessing
- Missing value imputation (median)
- Categorical encoding (Gender, Suicidal Thoughts, Family History)
- Profession grouping (Student / Working)
- Degree one-hot encoding (40+ categories)
- City **frequency encoding** for 60+ cities
- Numerical feature scaling (StandardScaler)
- XGBoost classifier with 82% accuracy
- Real-time CLI user prediction
- Automatic Excel logging of all user inputs + predictions (WE ARE CURRENTLY WORKING ON THIS)



## Project Structure

So , This is how I have structured the project:
```

├── preprocessing.py  # It cleans & preprocesses the dataset (Stress.csv)
├── train.py  # Trains the XGBoost model & saves model + scaler
├── predict.py  # Loads model, takes user input, predicts depression, logs result
├── model.pkl  # Saved trained model
├── scaler.pkl  # Saved scaler + feature metadata
├── Stress.csv  # Dataset used for training
├── user_inputs.xlsx  #  stores user inputs & prediction (OUR ONGOING WORK)
└── README.md


To begin , you need to install all the necessary dependencies:

Install required dependencies using:

     pip install -r requirements.txt

Or manually using:

     pip install pandas numpy scikit-learn xgboost joblib openpyxl
```
## Here's How you can Run
Follow the below steps:

### 1) Preprocess the dataset
```bash
python preprocessing.py
```

### 2) Train the model
```bash
python train.py
```
This will generate:
- `model.pkl`
- `scaler.pkl`

### 3) Run prediction for a new user
```bash
python predict.py
```

You will see:
- Model prediction → *Depression / No Depression*
- Probability score
- User input automatically saved to `user_inputs.xlsx` (WE ARE CURRENTLY WORKING ON THIS)

---

## Model Summary
- **Algorithm:** XGBoost Classifier  
- **Accuracy:** ~82%  
- **Preprocessing steps:**
  - Missing value filling  
  - City frequency encoding  
  - Degree one-hot encoding  
  - Binary feature conversion  
  - StandardScaler on numeric features  

---

## Dataset
Dataset used: **Stress.csv**, containing attributes like:
- Academic pressure
- Sleep quality
- Job satisfaction
- CGPA
- Diet quality
- City
- Profession
- Mental health indicators

---

## Contact
For suggestions or improvements, feel free to open an issue on GitHub.

