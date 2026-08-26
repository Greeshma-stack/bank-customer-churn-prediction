import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Churn Risk Intelligence", page_icon="🏦", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("bank_churn_model.joblib")

bundle = load_model()
model = bundle["pipeline"]
threshold = bundle["threshold"]

st.title("🏦 Bank Customer Churn Risk Intelligence")
st.caption("Predictive modeling, risk scoring and what-if analysis")

with st.sidebar:
    st.header("Customer Profile")
    credit = st.slider("Credit Score", 300, 850, 650)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 300000.0, 75000.0, step=1000.0)
    products = st.slider("Number of Products", 1, 4, 2)
    has_card = st.selectbox("Has Credit Card", [0,1], format_func=lambda x: "Yes" if x else "No")
    active = st.selectbox("Is Active Member", [0,1], format_func=lambda x: "Yes" if x else "No")
    salary = st.number_input("Estimated Salary", 0.0, 300000.0, 100000.0, step=1000.0)

def make_row(credit, geography, gender, age, tenure, balance, products, has_card, active, salary):
    return pd.DataFrame([{
        "CreditScore": credit,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": active,
        "EstimatedSalary": salary,
        "BalanceSalaryRatio": balance/(salary+1),
        "ProductDensity": products/(tenure+1),
        "EngagementProduct": active*products,
        "AgeTenureInteraction": age*(tenure+1)
    }])

row = make_row(credit, geography, gender, age, tenure, balance, products, has_card, active, salary)
prob = float(model.predict_proba(row)[:,1][0])
pred = int(prob >= threshold)

if prob < 0.30:
    band = "Low"
elif prob < 0.60:
    band = "Medium"
elif prob < 0.80:
    band = "High"
else:
    band = "Critical"

c1,c2,c3 = st.columns(3)
c1.metric("Churn Probability", f"{prob:.1%}")
c2.metric("Risk Band", band)
c3.metric("Decision", "Likely to Churn" if pred else "Likely to Stay")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Risk Distribution")
    fig, ax = plt.subplots(figsize=(7,3.5))
    ax.bar(["Churn probability"], [prob])
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

with right:
    st.subheader("Retention Interpretation")
    if band in ["High","Critical"]:
        st.warning("Prioritize this customer for proactive retention. Review engagement, product fit and service experience.")
    elif band == "Medium":
        st.info("Monitor the customer and consider a targeted engagement or product-fit campaign.")
    else:
        st.success("No immediate high-risk intervention indicated. Maintain engagement.")

st.subheader("What-if Scenario Simulator")
st.write("Adjust the fields in the sidebar and rerun the prediction. Compare how engagement and product changes affect the score.")

scenario = row.copy()
col1,col2 = st.columns(2)
with col1:
    new_active = st.selectbox("Scenario: Active Member", [0,1], index=int(active))
    new_products = st.slider("Scenario: Number of Products", 1, 4, products)
with col2:
    new_tenure = st.slider("Scenario: Tenure", 0, 10, tenure)
    new_balance = st.number_input("Scenario: Balance", 0.0, 300000.0, balance, step=1000.0)

scenario = make_row(credit, geography, gender, age, new_tenure, new_balance,
                    new_products, has_card, new_active, salary)
scenario_prob = float(model.predict_proba(scenario)[:,1][0])

s1,s2 = st.columns(2)
s1.metric("Current probability", f"{prob:.1%}")
s2.metric("Scenario probability", f"{scenario_prob:.1%}", delta=f"{scenario_prob-prob:+.1%}")

st.caption("Risk bands are operational thresholds and should be validated with historical outcomes and retention economics before production use.")
