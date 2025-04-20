import streamlit as st
import numpy as np
import joblib
import sqlite3
import pandas as pd

# Load trained models and scaler
lr_model = joblib.load(r"D:\loan_approval_pred\ml_model\lr_model.pkl")
dt_model = joblib.load(r"D:\loan_approval_pred\ml_model\dtree_model.pkl")
rf_model = joblib.load(r"D:\loan_approval_pred\ml_model\rf_model.pkl")
scaler = joblib.load(r"D:\loan_approval_pred\ml_model\scaler.pkl")

# SQLite3 setup
conn = sqlite3.connect('loan_predictions.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS LoanApplications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cibil_score REAL,
    loan_term INTEGER,
    income_annum REAL,
    no_of_dependents INTEGER,
    loan_amount REAL,
    luxury_assets_value REAL,
    predicted_status TEXT
)
""")
conn.commit()

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("Loan Approval Prediction App")
st.write("Enter your loan application details below:")

# User input
cibil_score = st.number_input("CIBIL Score", min_value=0.0, max_value=900.0, value=750.0)
loan_term = st.number_input("Loan Term (Months)", min_value=1, max_value=360, value=60)
income_annum = st.number_input("Annual Income (₹)", min_value=0.0, value=500000.0)
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0.0, value=300000.0)
luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0.0, value=1000000.0)
gender = st.selectbox("Gender (1-Male,0-Female)", [1,0])
married = st.selectbox("Married (1-Yes, 0-No)", [1,0])
education = st.selectbox("Education(College Graduate)(1-Yes, 0-No)", [1,0])
self_employed = st.selectbox("Self Employed (1-Yes, 0-No)", [1,0])


if st.button("Predict Loan Status"):
    # Prepare the full 10-feature input
    input_data = np.array([[no_of_dependents, income_annum, loan_amount, loan_term, cibil_score, luxury_assets_value,
                            gender, married, education, self_employed]])

    # Scale all 10 inputs (as expected by your scaler)
    scaled_input = scaler.transform(input_data)

    # Slice only the first 6 features for models that were trained on 6 inputs
    scaled_input_for_model = scaled_input[:, :6]

    # Predictions
    lr_pred = lr_model.predict(scaled_input[:, :6])[0]
    dt_pred = dt_model.predict(scaled_input[:, :6])[0]
    rf_pred = rf_model.predict(scaled_input[:, :6])[0]


    # Majority vote logic
    votes = [lr_pred, dt_pred, rf_pred].count(1)
    final_status = "Approved" if votes >= 2 else "Rejected"

    # Show result to user
    st.subheader("Prediction Results:")
    st.write(f"Logistic Regression: {' Approved' if lr_pred == 1 else ' Rejected'}")
    st.write(f"Decision Tree: {' Approved' if dt_pred == 1 else ' Rejected'}")
    st.write(f"Random Forest: {' Approved' if rf_pred == 1 else ' Rejected'}")

    if final_status == "Approved":
        st.success(f" Majority Decision: Loan {final_status}!")
    else:
        st.error(f" Majority Decision: Loan {final_status}.")

    # Insert prediction into SQLite
    cursor.execute("""
    INSERT INTO LoanApplications (cibil_score, loan_term, income_annum, no_of_dependents, loan_amount, luxury_assets_value, predicted_status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (cibil_score, loan_term, income_annum, no_of_dependents, loan_amount, luxury_assets_value, final_status))
    conn.commit()

    st.info(" Application data saved to database.")

# View previous predictions
if st.sidebar.button(" Show Saved Applications"):
    cursor.execute("SELECT * FROM LoanApplications ORDER BY id DESC")
    records = cursor.fetchall()
    st.sidebar.write("### Saved Applications")
    for record in records:
        st.sidebar.write(record)
# Sidebar: Export to Excel
if st.sidebar.button(" Export to Excel"):
    df = pd.read_sql_query("SELECT * FROM LoanApplications", conn)
    df.to_excel('LoanApplications.xlsx', index=False)
    st.sidebar.success(" Exported to LoanApplications.xlsx")

# Close DB connection on exit
conn.close()
#cd /d D:\loan_approval_pred
#streamlit run loan_ui.py
