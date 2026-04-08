import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Hazardous Waste AI System", layout="wide")

st.title("🚨 AI-Based Hazardous Waste Prediction System")
st.markdown("Automobile Workshop Environmental Monitoring")

# ---------------- INPUT ----------------
col1, col2, col3 = st.columns(3)

with col1:
    vehicles = st.number_input("Vehicles per day", 1, 100, 20)

with col2:
    oil_changes = st.number_input("Oil changes/day", 0, 50, 10)

with col3:
    batteries = st.number_input("Battery replacements/day", 0, 20, 3)

# ---------------- CALCULATION ----------------
oil_waste = oil_changes * 4
acid_waste = batteries * 1.5
coolant_waste = vehicles * 0.7

total_waste = oil_waste + acid_waste + coolant_waste

# ---------------- AI MODEL ----------------
days = np.array([1,2,3,4,5]).reshape(-1,1)
waste_data = np.array([20,25,30,35,40])

model = LinearRegression()
model.fit(days, waste_data)

future_days = np.array([6,7,8,9,10]).reshape(-1,1)
prediction = model.predict(future_days)

# ---------------- RISK SCORE ----------------
risk_score = min(100, int((total_waste / 100) * 100))

if risk_score < 30:
    risk_level = "LOW"
    color = "green"
elif risk_score < 70:
    risk_level = "MEDIUM"
    color = "orange"
else:
    risk_level = "HIGH"
    color = "red"

# ---------------- OUTPUT ----------------
st.subheader("📊 Waste Generated")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Oil Waste", f"{oil_waste} L")
c2.metric("Coolant Waste", f"{coolant_waste} L")
c3.metric("Acid Waste", f"{acid_waste} L")
c4.metric("Total Waste", f"{total_waste} L")

st.subheader("📈 Waste Prediction")
st.line_chart(prediction)

st.subheader("⚠️ Risk Assessment")
st.markdown(f"<h2 style='color:{color}'>Risk Level: {risk_level}</h2>", unsafe_allow_html=True)
st.write(f"Risk Score: {risk_score}/100")

# ---------------- RECOMMENDATION ----------------
st.subheader("🧠 AI Recommendations")

if risk_level == "LOW":
    st.success("Safe operation. Maintain current waste handling practices.")
elif risk_level == "MEDIUM":
    st.warning("Increase waste monitoring and improve disposal frequency.")
else:
    st.error("Immediate action required. Contact hazardous waste management authority.")
