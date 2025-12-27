import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("ann_churn_model.h5")

# -----------------------------
# Load scaler
# -----------------------------
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦")
st.title("🏦 Bank Customer Churn Prediction")
st.write("Fill in the customer details to predict churn.")

# -----------------------------
# User Inputs
# -----------------------------
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years with bank)", min_value=0, max_value=10, value=5)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_credit_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# -----------------------------
# Encoding (MATCH TRAINING)
# -----------------------------
gender_val = 1 if gender == "Male" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# -----------------------------
# Create DataFrame with SAME feature names
# -----------------------------
input_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Gender": gender_val,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_credit_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
    "Geography_Germany": geo_germany,
    "Geography_Spain": geo_spain
}])

# -----------------------------
# Scale input
# -----------------------------
input_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_scaled)
    probability = prediction[0][0]

    if probability > 0.5:
        st.error(f"❌ Customer is likely to LEAVE the bank\n\nProbability: **{probability:.2f}**")
    else:
        st.success(f"✅ Customer is likely to STAY with the bank\n\nProbability: **{probability:.2f}**")

