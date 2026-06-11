# STAGE 5: FRONTEND INTERFACE WITH STREAMLIT
import streamlit as st
import requests

# Page layout configuration
st.set_page_config(page_title="MoCredit - Automated Credit Underwriting System", layout="centered")

st.title("🏦 MoCredit AI - Automated Credit Underwriting")
st.write("Enter the customer's behavioral and financial metrics to get an instant AI-driven credit underwriting decision.")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 App Behavioral Metrics")
    financial_apps = st.number_input("Financial apps installed", min_value=0, max_value=50, value=2)
    screen_time = st.number_input("Avg daily screen time (hours/day)", min_value=0.0, max_value=24.0, value=4.5)
    unique_apps = st.number_input("Unique apps opened per day", min_value=0, max_value=200, value=25)
    bank_sms = st.number_input("Bank SMS alerts received", min_value=0, value=10)

with col2:
    st.subheader("💰 Financial & Personal History")
    online_txns = st.number_input("Online transactions (last 30 days)", min_value=0, value=15)
    avg_txn_amt = st.number_input("Avg transaction amount (VND)", min_value=0.0, value=150000.0, step=10000.0)
    existing_debt = st.number_input("Outstanding debt across financial institutions (VND)", min_value=0.0, value=50000000.0, step=1000000.0)
    
    income_bracket = st.selectbox("Income Bracket", options=[1, 2, 3], format_func=lambda x: f"Bracket {x} (Low -> High)")
    employment_type = st.selectbox("Employment Type", options=[1, 2, 3], format_func=lambda x: f"Type {x}")
    education_level = st.selectbox("Education Level", options=[1, 2, 3], format_func=lambda x: f"Level {x}")
    cluster_id = st.selectbox("Customer Segment (Cluster ID)", options=[0, 1, 2])

st.markdown("---")

if st.button("🚀 INITIATE AUTOMATED UNDERWRITING", type="primary", use_container_width=True):
    # Package data into a standard JSON payload expected by the FastAPI backend
    payload = {
        "financial_apps_installed": int(financial_apps),
        "avg_daily_screen_time_hrs": float(screen_time),
        "unique_apps_per_day": int(unique_apps),
        "bank_sms_count": int(bank_sms),
        "online_txn_count_last_30d": int(online_txns),
        "avg_txn_amount": float(avg_txn_amt),
        "existing_debt_amount_vnd": float(existing_debt),
        "income_bracket": int(income_bracket),
        "employment_type": int(employment_type),
        "education_level": int(education_level),
        "cluster_id": int(cluster_id)
    }
    
    try:
        # Send POST request to FastAPI Backend 
        with st.spinner("AI model is processing consumer profiles..."):
            response = requests.post("http://127.0.0.1:8000/api/v1/credit-approval", json=payload)
            
        if response.status_code == 200:
            result = response.json()
            
            # Extract response attributes for UI rendering
            risk_score = result["risk_score_probability"]
            decision = result["decision"]
            allocated_limit = result["allocated_limit_vnd"]
            
            # Render UI elements based on the underwriting decision outcome
            st.subheader("🎯 Underwriting Assessment Summary:")
            
            if decision == "Approved":
                st.success(f"🎉 APPLICATION APPROVED (DECISION: {decision})")
                st.metric(label="Assigned Credit Limit", value=f"{allocated_limit:,} VND")
            else:
                st.error(f"❌ APPLICATION REJECTED (DECISION: {decision})")
                st.metric(label="Assigned Credit Limit", value="0 VND")
                
            st.info(f"🤖 AI-evaluated Probability of Default (PD): {risk_score * 100:.2f}%")
            
        else:
            st.error(f"System API Error: Status Code {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Failed to connect to the API Backend Server! Make sure your FastAPI backend (`app.py`) is running.")