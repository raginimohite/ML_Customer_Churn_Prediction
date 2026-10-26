import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction")

st.title("📊 Customer Churn Prediction")
st.write("Machine Learning Dashboard")

# Load model and dataset
model = joblib.load("Marvellous_ML_Customer_Churn.pkl")
df = pd.read_csv("customer_churn.csv")

# Dashboard metrics
col1, col2 = st.columns(2)
col1.metric("Total Customers", len(df))

if "Churn" in df.columns:
    col2.metric("Churned Customers", (df["Churn"] == 1).sum())

st.divider()
st.subheader("Enter Customer Details")

age = st.number_input("Customer Age", min_value=1)
tenure = st.number_input("Tenure (Months)", min_value=0)
charges = st.number_input("Monthly Charges", min_value=0)
calls = st.number_input("Support Calls", min_value=0)

if st.button("Predict Churn"):
    customer = pd.DataFrame({
        "Age": [age],
        "Tenure": [tenure],
        "MonthlyCharges": [charges],
        "SupportCalls": [calls]
    })

    prediction = model.predict(customer)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn!")
        st.info("Suggestion: Consider offering a retention plan.")
    else:
        st.success("✅ Customer is unlikely to churn!")

st.divider()
st.caption("Customer Churn Prediction | ML Project")