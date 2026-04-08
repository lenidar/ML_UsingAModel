import joblib
import numpy as np
import pandas as pd

def predict_salary(experience, education, job):
    # Load artifacts
    model = joblib.load("salary_model.pkl")
    scaler = joblib.load("salary_scaler.pkl")
    columns = joblib.load("salary_columns.pkl")

    # Build raw input
    raw = pd.DataFrame([{
        "experience_years": experience,
        "education_level": education,
        "job_title": job
    }])

    # One-hot encode
    raw = pd.get_dummies(raw, drop_first=True)

    # Add missing columns
    for col in columns:
        if col not in raw:
            raw[col] = 0

    # Ensure correct order
    raw = raw[columns]

    print("CHECKPOINT — final raw input to model:")
    print(raw)

    # Scale
    raw_scaled = scaler.transform(raw)

    # Predict (log scale)
    log_pred = model.predict(raw_scaled)

    # Convert back to real salary
    salary = np.exp(log_pred)[0]

    return salary 

if __name__ == "__main__":
    pred = predict_salary(1, "Bachelor", "Data Analyst")
    print("Predicted salary:", f"{pred:,.2f}")
    pred = predict_salary(20, "Master", "Cybersecurity Analyst")
    print("Predicted salary:", f"{pred:,.2f}")